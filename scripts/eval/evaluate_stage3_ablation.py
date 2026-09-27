"""
Ablation: which feature source is actually responsible for the calibrated
AUC/accuracy numbers reported earlier? Extracts each feature block
(vibration-DANN, audio-DANN, VGGish, MFCC) ONCE, then trains/evaluates a
classifier on each block ALONE and on the full combination, under the
exact same CV protocol as evaluate_stage3_calibrated.py — so the
"combined" row here should reproduce those earlier numbers, and the
per-source rows tell us what's actually earning its place.
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
from evaluate_stage3_calibrated import load_encoder
from models.baseline_mfcc import extract_features

warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "checkpoints"


# Each encoder must see spectrograms at the sample rate it was TRAINED on
# (vib: 25600Hz, aud: 16000Hz) — a single shared sr_for_cfg silently remapped
# the vibration encoder's mel-frequency axis. Fixed: two separate configs.
VIB_ENCODER_CFG = SpectrogramConfig(sample_rate=25600)
AUD_ENCODER_CFG = SpectrogramConfig(sample_rate=16000)


def build_feature_blocks(files, reader, vib_encoder, aud_encoder, vggish):
    """Returns a dict of separate feature blocks (not concatenated), so
    ablation can slice per-source without re-running any model."""
    vib_rows, aud_rows, vgg_rows, mfcc_rows = [], [], [], []
    for f in files:
        sig, sr = reader(f)
        vib_specs = signal_to_spectrogram_batch(sig, orig_sr=sr, cfg=VIB_ENCODER_CFG)
        aud_specs = signal_to_spectrogram_batch(sig, orig_sr=sr, cfg=AUD_ENCODER_CFG)

        with torch.no_grad():
            # Mean-pool across every window in the file instead of using
            # only the first 1.5s (window 0) — was silently discarding the
            # rest of every recording longer than 1.5s.
            vib_windows = torch.from_numpy(vib_specs).unsqueeze(1)
            aud_windows = torch.from_numpy(aud_specs).unsqueeze(1)
            vib_rows.append(vib_encoder(vib_windows).mean(dim=0).numpy().flatten())
            aud_rows.append(aud_encoder(aud_windows).mean(dim=0).numpy().flatten())
            vgg_raw = vggish.forward(sig, fs=sr)
            if vgg_raw.dim() == 1:
                vgg_raw = vgg_raw.unsqueeze(0)
            vgg_rows.append(vgg_raw.mean(dim=0).numpy().flatten())
        mfcc_rows.append(extract_features(sig, sr))

    return {
        "vib": np.stack(vib_rows).astype(np.float32),
        "aud": np.stack(aud_rows).astype(np.float32),
        "vgg": np.stack(vgg_rows).astype(np.float32),
        "mfcc": np.stack(mfcc_rows).astype(np.float32),
    }


def eval_cv(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s, aucs = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed))])
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        scores = pipe.predict_proba(X[test_idx])[:, 1]
        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds))
        try:
            aucs.append(roc_auc_score(y[test_idx], scores))
        except ValueError:
            aucs.append(np.nan)
    return np.mean(accs), np.std(accs), np.mean(f1s), np.nanmean(aucs), np.nanstd(aucs)


def eval_split(X, y, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=seed))])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    scores = pipe.predict_proba(Xte)[:, 1]
    return accuracy_score(yte, preds), f1_score(yte, preds), roc_auc_score(yte, scores)


def run_ablation(blocks: dict, y: np.ndarray, name: str, protocol: str):
    combos = {
        "Track A (vibration-DANN) alone": ["vib"],
        "Track A' (audio-DANN) alone": ["aud"],
        "Track B (VGGish) alone": ["vgg"],
        "Track C (MFCC) alone": ["mfcc"],
        "A + A' (both DANN encoders)": ["vib", "aud"],
        "B + C (VGGish + MFCC)": ["vgg", "mfcc"],
        "ALL combined (reproduces earlier result)": ["vib", "aud", "vgg", "mfcc"],
    }
    print(f"\n{'='*70}\n{name} ablation ({protocol})\n{'='*70}")
    for label, keys in combos.items():
        X = np.concatenate([blocks[k] for k in keys], axis=1)
        if protocol == "cv":
            acc, acc_std, f1, auc, auc_std = eval_cv(X, y)
            print(f"  {label:42s} dim={X.shape[1]:4d}  acc={acc:.3f}+/-{acc_std:.3f}  f1={f1:.3f}  auc={auc:.3f}+/-{auc_std:.3f}")
        else:
            acc, f1, auc = eval_split(X, y)
            print(f"  {label:42s} dim={X.shape[1]:4d}  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")


def main():
    guard = CPUGuard()
    if not guard.check("ablation setup"):
        return

    print("Loading pretrained feature extractors...")
    vib_encoder = load_encoder(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
    aud_encoder = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish.eval()

    print("Extracting AI Mechanic features (once)...")
    ai_files, ai_labels = [], []
    for f in sorted((RAW_DIR / "ai_mechanic").rglob("*.wav")):
        lbl = ai_mechanic_label_fn(f)
        if lbl != -1:
            ai_files.append(f)
            ai_labels.append(lbl)
    ai_labels = np.array(ai_labels)
    ai_blocks = build_feature_blocks(ai_files, ai_mechanic_reader, vib_encoder, aud_encoder, vggish)
    run_ablation(ai_blocks, ai_labels, "AI MECHANIC", protocol="cv")

    print("\nExtracting Car Diagnostics features (once, 1386 files — this is the slow part)...")
    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    cd_files = sorted(cd_root.rglob("*.wav"))
    cd_labels = np.array([car_diagnostics_label_fn(f) for f in cd_files])
    cd_blocks = build_feature_blocks(cd_files, car_diagnostics_reader, vib_encoder, aud_encoder, vggish)
    run_ablation(cd_blocks, cd_labels, "CAR DIAGNOSTICS", protocol="split")

    print("\nAblation complete.")


if __name__ == "__main__":
    main()
