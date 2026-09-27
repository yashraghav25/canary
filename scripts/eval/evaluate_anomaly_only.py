"""
The DEPLOYABLE version, done properly: anomaly detection calibrated on
NORMAL-ONLY real data (no faulty examples needed from the customer at
all — exactly the constraint that makes this realistic to ship).

WHY THE EARLIER ZERO-SHOT ATTEMPT SCORED ~CHANCE, DIAGNOSED HONESTLY:
  - AI Mechanic has only 19 real normal clips. In a 128-dim embedding
    space, fitting a real covariance matrix needs at least ~128 samples
    to not be singular — with 19, the earlier code's fallback silently
    kicked in and used plain Euclidean centroid-distance, a weak method
    with no awareness of which directions in the embedding space actually
    vary versus which are noise.
  - This script fixes that properly: PCA-regularized Mahalanobis (reduce
    dimensionality using only as many components as the sample size can
    support, THEN compute distance in the reduced space), plus two
    additional standard small-sample anomaly methods (k-NN distance,
    One-Class SVM) for comparison, plus the reconstruction-error method
    that was planned in the original diagram and never built.
  - Also tests on Car Diagnostics' normal set (402 real clips — a much
    larger, statistically solid calibration set) to check whether the
    method improves with more (still normal-only, still realistic to
    collect) calibration data, per the request to test on more data.

Protocol: split NORMAL samples 70/30 (calibration / test-normal). Fit
each anomaly method on calibration-normal ONLY. Evaluate AUC using
test-normal + ALL faulty samples (faulty never touched during fitting).
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from cpu_guard import CPUGuard
from data.preprocess import SpectrogramConfig, signal_to_spectrogram_batch
from data.readers import (
    ai_mechanic_label_fn, ai_mechanic_reader,
    car_diagnostics_label_fn, car_diagnostics_reader,
    engine_journal_bearings_label_fn, engine_journal_bearings_reader,
)
from models.encoder import load_encoder  # single source of truth — see models/encoder.py
from models.baseline_mfcc import extract_features
from models.vggish_finetune import VGGishLastBlockDANN, _load_vggish_input_module

warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "checkpoints"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# Each encoder must see spectrograms at the sample rate it was TRAINED on
# (vib: 25600Hz, aud: 16000Hz) — a single shared sr_for_cfg silently remapped
# the vibration encoder's mel-frequency axis. Fixed: two separate configs.
VIB_ENCODER_CFG = SpectrogramConfig(sample_rate=25600)
AUD_ENCODER_CFG = SpectrogramConfig(sample_rate=16000)


VGGISH_EMBEDDING_DIM = 128


def build_feature_blocks(files, reader, vib_encoder, aud_encoder, vggish, vggish_ft=None, vggish_input=None):
    vib_rows, aud_rows, vgg_rows, mfcc_rows, vgg_ft_rows = [], [], [], [], []
    vgg_failures = []
    vgg_ft_failures = []
    t0 = time.time()
    for f in files:
        sig, sr = reader(f)
        vib_specs = signal_to_spectrogram_batch(sig, orig_sr=sr, cfg=VIB_ENCODER_CFG)
        aud_specs = signal_to_spectrogram_batch(sig, orig_sr=sr, cfg=AUD_ENCODER_CFG)
        with torch.no_grad():
            # Mean-pool across every window in the file instead of using
            # only the first 1.5s (window 0).
            vib_windows = torch.from_numpy(vib_specs).unsqueeze(1)
            aud_windows = torch.from_numpy(aud_specs).unsqueeze(1)
            vib_rows.append(vib_encoder(vib_windows).mean(dim=0).numpy().flatten())
            aud_rows.append(aud_encoder(aud_windows).mean(dim=0).numpy().flatten())
            
            # Optimized VGGish computation
            ex_t = None
            if vggish_ft is not None and vggish_input is not None:
                try:
                    examples = vggish_input.waveform_to_examples(sig, sr, return_tensor=False)
                    if len(examples) > 0:
                        ex_t = torch.from_numpy(examples).unsqueeze(1).float()
                except Exception:
                    pass
            
            try:
                # NOTE: vggish.forward()'s own _preprocess() only accepts a raw
                # numpy waveform or a wav filename string — passing the
                # precomputed ex_t tensor here unconditionally raised a bare
                # AttributeError (confirmed: 100% failure rate on every file,
                # every dataset). ex_t is still valid for vggish_ft below
                # (the custom fine-tuned model, which does accept precomputed
                # patches through its own frozen_blocks/unfrozen_block) — it
                # was never valid for the plain, stock vggish object.
                vgg_raw = vggish.forward(sig, fs=sr)

                if vgg_raw.dim() == 1:
                    vgg_raw = vgg_raw.unsqueeze(0)
                if vgg_raw.numel() == 0:
                    raise RuntimeError("VGGish returned an empty tensor")
                vgg_rows.append(vgg_raw.mean(dim=0).numpy().flatten())
            except Exception as e:
                vgg_failures.append((f.name, str(e)))
                vgg_rows.append(np.zeros(VGGISH_EMBEDDING_DIM, dtype=np.float32))
        mfcc_rows.append(extract_features(sig, sr))
        
        # VGGish-finetuned extraction
        if vggish_ft is not None and ex_t is not None:
            try:
                with torch.no_grad():
                    mid = vggish_ft.frozen_blocks(ex_t)
                    feats = vggish_ft.unfrozen_block(mid)
                    emb = vggish_ft.head(feats)
                    vgg_ft_rows.append(emb.mean(dim=0).numpy().flatten())
            except Exception as e:
                vgg_ft_failures.append((f.name, str(e)))
                vgg_ft_rows.append(np.zeros(128, dtype=np.float32))
        else:
            vgg_ft_rows.append(np.zeros(128, dtype=np.float32))
            
        if len(vib_rows) % 10 == 0 or len(vib_rows) == len(files):
            elapsed = time.time() - t0
            rate = len(vib_rows) / elapsed if elapsed > 0 else 0
            remaining = (len(files) - len(vib_rows)) / rate if rate > 0 else float("nan")
            print(f"    processed {len(vib_rows)}/{len(files)} files "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", flush=True)
            
    if vgg_failures:
        print(f"    [warn] VGGish failed on {len(vgg_failures)}/{len(files)} files (too short for its "
              f"internal framing) — zero-filled, so the 'vgg' feature block is DEGRADED for this "
              f"dataset, not fully missing. First failure: {vgg_failures[0]}")
    return {
        "vib": np.stack(vib_rows).astype(np.float32),
        "aud": np.stack(aud_rows).astype(np.float32),
        "vgg": np.stack(vgg_rows).astype(np.float32),
        "vgg_ft": np.stack(vgg_ft_rows).astype(np.float32),
        "mfcc": np.stack(mfcc_rows).astype(np.float32),
    }


PCA_MAHALANOBIS_DEFAULT_COMPONENTS = 32
"""Fixed cap (not "however many the sample count allows") so the method means
the same thing on both datasets. Previously the default fell back to
cal_X.shape[0]-1, which on Car Diagnostics (281 calibration samples, 128-dim
features) evaluated to min(280,280,128)=128 — i.e. NO actual dimensionality
reduction at all (full-rank PCA is just a rotation), while on AI Mechanic
(13 samples) it genuinely compressed 128->12 dims. Same function name, two
different behaviors — fixed by always capping at a fixed budget instead."""


def pca_mahalanobis_score(cal_X, test_X, max_components=PCA_MAHALANOBIS_DEFAULT_COMPONENTS):
    n_components = min(max_components, cal_X.shape[0] - 1, cal_X.shape[1])
    n_components = max(n_components, 1)
    scaler = StandardScaler().fit(cal_X)
    cal_scaled = scaler.transform(cal_X)
    test_scaled = scaler.transform(test_X)
    pca = PCA(n_components=n_components, random_state=42).fit(cal_scaled)
    cal_reduced = pca.transform(cal_scaled)
    test_reduced = pca.transform(test_scaled)
    cov = EmpiricalCovariance().fit(cal_reduced)
    return cov.mahalanobis(test_reduced)


def knn_distance_score(cal_X, test_X, k=5):
    k = min(k, len(cal_X))
    scaler = StandardScaler().fit(cal_X)
    nn = NearestNeighbors(n_neighbors=k).fit(scaler.transform(cal_X))
    dists, _ = nn.kneighbors(scaler.transform(test_X))
    return dists.mean(axis=1)


def one_class_svm_score(cal_X, test_X, nu=0.1):
    scaler = StandardScaler().fit(cal_X)
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(scaler.transform(cal_X))
    return -ocsvm.decision_function(scaler.transform(test_X))  # higher = more anomalous


def isolation_forest_score(cal_X, test_X):
    scaler = StandardScaler().fit(cal_X)
    iso = IsolationForest(n_estimators=200, random_state=42).fit(scaler.transform(cal_X))
    return -iso.decision_function(scaler.transform(test_X))


def pca_reconstruction_error_score(cal_X, test_X, max_components=PCA_MAHALANOBIS_DEFAULT_COMPONENTS):
    """The 'reconstruction-error' method from the original diagram, never
    implemented until now. PCA reconstruction error is the linear special
    case of an autoencoder — appropriate given how few normal samples
    (19 for AI Mechanic) are available; a full neural autoencoder would
    almost certainly overfit at this sample size.

    Same fixed-cap fix as pca_mahalanobis_score: if n_components equals the
    full feature dimensionality (e.g. 128 on the 281-sample Car Diagnostics
    calibration set, where the old cal_X.shape[0]-1 fallback evaluated to
    128), PCA becomes a lossless rotation and reconstruction error collapses
    to ~0 for every point regardless of label — a degenerate, uninformative
    score. Capping at a fixed budget keeps this method genuinely lossy
    (and therefore genuinely informative) on both datasets."""
    n_components = min(max_components, cal_X.shape[0] - 1, cal_X.shape[1])
    n_components = max(n_components, 1)
    scaler = StandardScaler().fit(cal_X)
    cal_scaled = scaler.transform(cal_X)
    test_scaled = scaler.transform(test_X)
    pca = PCA(n_components=n_components, random_state=42).fit(cal_scaled)
    test_reduced = pca.transform(test_scaled)
    test_reconstructed = pca.inverse_transform(test_reduced)
    return np.mean((test_scaled - test_reconstructed) ** 2, axis=1)


def run_dataset(name, files, labels, blocks, feature_key, test_size=0.3, seed=42):
    y = labels
    idx_normal = np.where(y == 0)[0]
    idx_faulty = np.where(y == 1)[0]
    cal_idx, test_normal_idx = train_test_split(idx_normal, test_size=test_size, random_state=seed)

    X = blocks[feature_key]
    cal_X = X[cal_idx]
    test_idx = np.concatenate([test_normal_idx, idx_faulty])
    test_X = X[test_idx]
    test_y = np.concatenate([np.zeros(len(test_normal_idx)), np.ones(len(idx_faulty))])

    print(f"\n[{name} / {feature_key}] calibration_normal={len(cal_idx)}  "
          f"test_normal={len(test_normal_idx)}  test_faulty={len(idx_faulty)}")

    methods = {
        "centroid/Mahalanobis (old method)": lambda: pca_mahalanobis_score(cal_X, test_X, max_components=1),
        "PCA(regularized)+Mahalanobis": lambda: pca_mahalanobis_score(cal_X, test_X),
        "PCA reconstruction-error": lambda: pca_reconstruction_error_score(cal_X, test_X),
        "k-NN distance": lambda: knn_distance_score(cal_X, test_X),
        "One-Class SVM": lambda: one_class_svm_score(cal_X, test_X),
        "Isolation Forest": lambda: isolation_forest_score(cal_X, test_X),
    }
    results = {}
    for method_name, fn in methods.items():
        try:
            scores = fn()
            auc = roc_auc_score(test_y, scores)
            print(f"    {method_name:35s} AUC={auc:.3f}")
            results[method_name] = auc
        except Exception as e:
            print(f"    {method_name:35s} FAILED: {e}")
    return results


def main():
    guard = CPUGuard()
    if not guard.check("anomaly-only setup"):
        return

    print("Loading feature extractors...", flush=True)
    vib_encoder = load_encoder(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
    aud_encoder = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish.eval()

    print("Loading VGGish-finetuned...", flush=True)
    vggish_ft = VGGishLastBlockDANN(vggish, n_domains=2)
    ckpt = torch.load(CHECKPOINT_DIR / "audio_vggish_finetune_lambda0.15.pt")
    vggish_ft.unfrozen_block.load_state_dict(ckpt["unfrozen_block"])
    vggish_ft.head.load_state_dict(ckpt["head"])
    vggish_ft.eval()
    vggish_input = _load_vggish_input_module()

    print("Extracting AI Mechanic features...", flush=True)
    ai_files, ai_labels = [], []
    for f in sorted((RAW_DIR / "ai_mechanic").rglob("*.wav")):
        lbl = ai_mechanic_label_fn(f)
        if lbl != -1:
            ai_files.append(f); ai_labels.append(lbl)
    ai_labels = np.array(ai_labels)
    ai_blocks = build_feature_blocks(ai_files, ai_mechanic_reader, vib_encoder, aud_encoder, vggish, vggish_ft, vggish_input)

    print("Extracting Car Diagnostics features (1386 files)...", flush=True)
    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    cd_files = sorted(cd_root.rglob("*.wav"))
    cd_labels = np.array([car_diagnostics_label_fn(f) for f in cd_files])
    cd_blocks = build_feature_blocks(cd_files, car_diagnostics_reader, vib_encoder, aud_encoder, vggish, vggish_ft, vggish_input)

    print("Extracting Engine Journal Bearings features (134 files, REAL vehicle vibration, first "
          "time this project has real-vehicle vibration held-out data instead of audio-only)...")
    ejb_root = RAW_DIR / "engine_journal_bearings"
    ejb_files = sorted(ejb_root.rglob("*.csv"))
    ejb_labels = np.array([engine_journal_bearings_label_fn(f) for f in ejb_files])
    ejb_blocks = build_feature_blocks(ejb_files, engine_journal_bearings_reader, vib_encoder, aud_encoder, vggish, vggish_ft, vggish_input)

    print(f"\n{'='*70}\nAI MECHANIC — anomaly-only (19 normal cal, small-sample regime)\n{'='*70}")
    for key in ["aud", "mfcc", "vgg", "vgg_ft", "vib"]:
        run_dataset("ai_mechanic", ai_files, ai_labels, ai_blocks, key)

    print(f"\n{'='*70}\nCAR DIAGNOSTICS — anomaly-only (402 normal available, larger-sample regime)\n{'='*70}")
    for key in ["aud", "mfcc", "vgg", "vgg_ft", "vib"]:
        run_dataset("car_diagnostics", cd_files, cd_labels, cd_blocks, key)

    print(f"\n{'='*70}\nENGINE JOURNAL BEARINGS — anomaly-only (44 normal available, REAL vehicle, "
          f"REAL vibration — not audio)\n{'='*70}")
    print("CAVEAT: this dataset's native sample rate (~296Hz) is far below what the vibration "
          "encoder (trained at 25600Hz) and audio encoder (trained at 16000Hz) expect — resampling "
          "upsamples by ~54-86x, which cannot manufacture real high-frequency content that was never "
          "captured. Results here should be read as an honest stress-test, not a clean apples-to-apples "
          "comparison with AI Mechanic/Car Diagnostics.")
    for key in ["aud", "mfcc", "vgg", "vgg_ft", "vib"]:
        run_dataset("engine_journal_bearings", ejb_files, ejb_labels, ejb_blocks, key)

    print("\nDone.")


if __name__ == "__main__":
    main()
