"""
Stage 3, calibrated version — the realistic deployment pattern instead of
pure zero-shot: extract features from the pretrained encoders (bearing
DANN, audio DANN, VGGish, MFCC), then fit a lightweight downstream
classifier on a REAL train split of the held-out data, evaluate on a
REAL held-out test split. This is standard transfer-learning practice
(feature extraction + few-shot calibration), not zero-shot generalization
— that distinction is reported explicitly, not blurred.

Given AI Mechanic is tiny (36 usable files), a single train/test split is
noisy — uses stratified 5-fold cross-validation instead, reporting the
mean +/- std across folds, which is the honest way to report a small-N
result rather than presenting one lucky split.

Car Diagnostics (1386 files) gets a real held-out 70/30 split — large
enough for a stable single-split estimate.

Feature fusion: concatenates ALL available embedding sources (bearing
vibration-DANN, audio-DANN, VGGish, MFCC) into one feature vector per
clip, since combining diverse feature sources is standard practice and
each one captures something different.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cpu_guard import CPUGuard
from data.preprocess import SpectrogramConfig, signal_to_spectrogram_batch
from data.readers import (
    ai_mechanic_label_fn, ai_mechanic_reader,
    car_diagnostics_label_fn, car_diagnostics_reader,
)
from models.baseline_mfcc import extract_features
from models.encoder import SpectrogramEncoder, load_encoder  # single source of truth — see models/encoder.py

warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "checkpoints"


# Each encoder must see spectrograms built at the SAME sample rate it was
# trained on (vib: 25600Hz per build_all_vibration_domains, aud: 16000Hz per
# build_all_audio_domains) — feeding both encoders one shared sr_for_cfg
# silently remaps the mel-frequency axis the vibration encoder learned to
# expect. Fixed: two separate configs, one per encoder.
VIB_ENCODER_CFG = SpectrogramConfig(sample_rate=25600)
AUD_ENCODER_CFG = SpectrogramConfig(sample_rate=16000)


def build_feature_matrix(files, reader, vib_encoder, aud_encoder, vggish):
    rows = []
    for f in files:
        sig, sr = reader(f)
        vib_specs = signal_to_spectrogram_batch(sig, orig_sr=sr, cfg=VIB_ENCODER_CFG)
        aud_specs = signal_to_spectrogram_batch(sig, orig_sr=sr, cfg=AUD_ENCODER_CFG)

        with torch.no_grad():
            # Mean-pool the embedding across EVERY window in the file, not
            # just window 0 — using only the first 1.5s of a longer
            # recording silently discarded the rest of it.
            vib_windows = torch.from_numpy(vib_specs).unsqueeze(1)
            aud_windows = torch.from_numpy(aud_specs).unsqueeze(1)
            vib_emb = vib_encoder(vib_windows).mean(dim=0).numpy().flatten()
            aud_emb = aud_encoder(aud_windows).mean(dim=0).numpy().flatten()
            vgg_raw = vggish.forward(sig, fs=sr)
            # FIX: VGGish returns (n_frames, 128) normally, but a single
            # internal frame comes back as a bare (128,) with no batch dim
            # for very short clips — .mean(dim=0) on THAT collapses all 128
            # features into one scalar instead of preserving the embedding,
            # which silently produced inconsistent feature-vector lengths
            # across files and crashed np.stack. Force 2D first.
            if vgg_raw.dim() == 1:
                vgg_raw = vgg_raw.unsqueeze(0)
            vgg_emb = vgg_raw.mean(dim=0).numpy().flatten()

        mfcc_feat = extract_features(sig, sr)
        rows.append(np.concatenate([vib_emb, aud_emb, vgg_emb, mfcc_feat]))
    return np.stack(rows).astype(np.float32)


def report(name: str, y_true, y_pred, y_score) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    print(f"  [{name}] accuracy={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    return {"accuracy": acc, "f1": f1, "auc": auc}


def main():
    guard = CPUGuard()
    if not guard.check("stage3-calibrated setup"):
        return

    print("Loading pretrained feature extractors (frozen, not fine-tuned)...")
    vib_encoder = load_encoder(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
    aud_encoder = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish.eval()

    print("\n" + "=" * 70)
    print("AI MECHANIC — calibrated (5-fold stratified CV, real BMW, real induced faults)")
    print("=" * 70)
    ai_files, ai_labels = [], []
    for f in sorted((RAW_DIR / "ai_mechanic").rglob("*.wav")):
        lbl = ai_mechanic_label_fn(f)
        if lbl != -1:
            ai_files.append(f)
            ai_labels.append(lbl)
    ai_labels = np.array(ai_labels)
    print(f"{len(ai_labels)} usable files ({(ai_labels==0).sum()} normal, {(ai_labels==1).sum()} faulty)")

    ai_X = build_feature_matrix(ai_files, ai_mechanic_reader, vib_encoder, aud_encoder, vggish)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(ai_X, ai_labels)):
        pipeline = Pipeline([("scale", StandardScaler()),
                             ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42))])
        pipeline.fit(ai_X[train_idx], ai_labels[train_idx])
        preds = pipeline.predict(ai_X[test_idx])
        scores = pipeline.predict_proba(ai_X[test_idx])[:, 1]
        m = report(f"fold {fold+1}/5", ai_labels[test_idx], preds, scores)
        fold_metrics.append(m)

    mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
    std_acc = np.std([m["accuracy"] for m in fold_metrics])
    mean_auc = np.nanmean([m["auc"] for m in fold_metrics])
    std_auc = np.nanstd([m["auc"] for m in fold_metrics])
    mean_f1 = np.mean([m["f1"] for m in fold_metrics])
    print(f"\n  AI MECHANIC calibrated 5-fold CV: accuracy={mean_acc:.3f}+/-{std_acc:.3f}  "
          f"F1={mean_f1:.3f}  AUC={mean_auc:.3f}+/-{std_auc:.3f}")

    print("\n" + "=" * 70)
    print("CAR DIAGNOSTICS — calibrated (70/30 split, real, 1386 files)")
    print("=" * 70)
    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    cd_files = sorted(cd_root.rglob("*.wav"))
    cd_labels = np.array([car_diagnostics_label_fn(f) for f in cd_files])
    print(f"{len(cd_labels)} files ({(cd_labels==0).sum()} normal, {(cd_labels==1).sum()} faulty)")

    cd_X = build_feature_matrix(cd_files, car_diagnostics_reader, vib_encoder, aud_encoder, vggish)
    Xtr, Xte, ytr, yte = train_test_split(cd_X, cd_labels, test_size=0.3, random_state=42, stratify=cd_labels)

    pipeline = Pipeline([("scale", StandardScaler()),
                         ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42))])
    pipeline.fit(Xtr, ytr)
    preds = pipeline.predict(Xte)
    scores = pipeline.predict_proba(Xte)[:, 1]
    report("car_diagnostics held-out 30%", yte, preds, scores)

    print("\nDone. These are CALIBRATED (feature-extraction + fine-tuned downstream classifier on real "
          "target data) results — NOT zero-shot transfer. That distinction matters for how this gets presented.")


if __name__ == "__main__":
    main()
