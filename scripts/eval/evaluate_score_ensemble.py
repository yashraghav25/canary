"""
Part 2 of the new-architecture plan: combine feature sources by their
anomaly SCORES, not their raw embeddings.

Every fusion attempt so far in this project (evaluate_stage3_calibrated.py's
"everything combined", the ablation study) concatenated raw feature vectors
[vib(128) + aud(128) + vgg(128) + mfcc(39)] = 423 dims BEFORE fitting one
anomaly detector. With only 13-30 real calibration-normal samples, fitting
anything in 423 dimensions is fighting the curse of dimensionality — this
is a likely reason "everything combined" never clearly beat the best single
source anywhere in this project.

This script does the opposite: fit ONE anomaly detector PER feature source,
in that source's own (much smaller, better-conditioned) native dimension,
get a single anomaly SCORE per source per test sample, z-normalize each
source's scores using ONLY the calibration-normal distribution (no leakage),
then average the normalized scores across sources. Combining 3-4 scalars is
trivial regardless of calibration-set size — a fundamentally different, and
much cheaper, way to combine information than raw-feature concatenation.

Uses PCA reconstruction-error as the per-source method throughout (chosen
BEFORE looking at which method wins where, to avoid quietly cherry-picking
the test-set-best method per source, which would make any "ensemble beats
everything" claim meaningless).
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cpu_guard import CPUGuard
from data.readers import (
    ai_mechanic_label_fn, ai_mechanic_reader,
    car_diagnostics_label_fn, car_diagnostics_reader,
    engine_journal_bearings_label_fn, engine_journal_bearings_reader,
)
from evaluate_anomaly_only import build_feature_blocks, load_encoder, pca_reconstruction_error_score

warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "checkpoints"


def zscore_using_calibration(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Normalize using ONLY calibration-set statistics — the test-normal and
    test-faulty scores are never used to set the scale, so this can't leak."""
    mean, std = cal_scores.mean(), cal_scores.std() + 1e-8
    return (test_scores - mean) / std


def run_ensemble(name: str, files, labels, blocks: dict, seed: int = 42, test_size: float = 0.3) -> None:
    idx_normal = np.where(labels == 0)[0]
    idx_faulty = np.where(labels == 1)[0]
    cal_idx, test_normal_idx = train_test_split(idx_normal, test_size=test_size, random_state=seed)
    test_idx = np.concatenate([test_normal_idx, idx_faulty])
    test_y = np.concatenate([np.zeros(len(test_normal_idx)), np.ones(len(idx_faulty))])

    # Nested split of calibration-normal ONLY (no faulty data touched anywhere
    # in this function) — fit_idx to fit the anomaly model, weight_idx (truly
    # held out from fitting) to estimate how reliable each source's "model of
    # normal" actually is, before ever looking at the real test set. This is
    # what lets weighting be principled rather than tuned on the answer key.
    fit_idx, weight_idx = train_test_split(cal_idx, test_size=0.3, random_state=seed)

    print(f"\n[{name}] calibration_normal={len(cal_idx)} (fit={len(fit_idx)}, weight-holdout={len(weight_idx)})  "
          f"test_normal={len(test_normal_idx)}  test_faulty={len(idx_faulty)}")

    per_source_test_scores = {}
    per_source_auc = {}
    generalization_gaps = {}
    for source in ["vib", "aud", "vgg", "vgg_ft", "mfcc"]:
        X = blocks[source]
        fit_X, weight_X, cal_X, test_X = X[fit_idx], X[weight_idx], X[cal_idx], X[test_idx]

        # REDESIGNED weight estimation. The first version weighted by raw
        # 1/mean(reconstruction_error) — not scale-invariant, so it compared
        # apples (128-dim learned embeddings) to oranges (39-dim MFCC) and
        # picked whichever source happened to have small numbers for reasons
        # unrelated to quality (it picked AI Mechanic's WORST source, vib at
        # AUC=0.314, as 73% of the ensemble weight).
        #
        # Fixed: measure the GENERALIZATION GAP instead — fit on fit_X only,
        # then ask "does truly-held-out normal data (weight_X, never seen
        # during fitting) score the way the model expects normal data to
        # score, based on the fit set's OWN distribution?" This is measured
        # in units of the fit set's own standard deviation, so it's
        # automatically scale-invariant regardless of each source's native
        # embedding dimension or magnitude — a source with a large gap here
        # is one whose "model of normal" doesn't generalize even to more
        # normal data, which is exactly what should be downweighted.
        fit_scores = pca_reconstruction_error_score(fit_X, fit_X)
        weight_holdout_scores = pca_reconstruction_error_score(fit_X, weight_X)
        gap = (weight_holdout_scores.mean() - fit_scores.mean()) / (fit_scores.std() + 1e-8)
        generalization_gaps[source] = abs(gap)

        # Final scoring: refit on ALL of cal_idx (standard practice — the
        # weight-holdout split above was only for estimating reliability, not
        # for the final model, which should use every calibration sample).
        cal_scores = pca_reconstruction_error_score(cal_X, cal_X)
        test_scores_raw = pca_reconstruction_error_score(cal_X, test_X)
        normalized = zscore_using_calibration(cal_scores, test_scores_raw)
        per_source_test_scores[source] = normalized
        auc = roc_auc_score(test_y, test_scores_raw)
        per_source_auc[source] = auc
        print(f"    {source:6s} alone (PCA reconstruction-error): AUC={auc:.3f}  "
              f"[generalization gap={generalization_gaps[source]:.3f} sigma]")

    # Reliability = inverse gap (smaller gap = more trustworthy), then
    # SHRUNK 50% toward equal weighting — a deliberate safeguard, not an
    # afterthought: the weight-holdout split can be as small as 4 samples on
    # AI Mechanic, so a single noisy estimate should never be allowed to
    # near-zero-out a genuinely good source the way the first version did.
    raw_reliability = {k: 1.0 / (1.0 + v) for k, v in generalization_gaps.items()}
    reliability_sum = sum(raw_reliability.values())
    reliability_weights = {k: v / reliability_sum for k, v in raw_reliability.items()}
    n_sources = len(reliability_weights)
    normalized_weights = {k: 0.5 * v + 0.5 * (1.0 / n_sources) for k, v in reliability_weights.items()}
    print(f"    normalized weights (50% reliability-based, 50% shrunk toward equal): " +
          ", ".join(f"{k}={v:.3f}" for k, v in normalized_weights.items()))

    equal_score = np.mean(list(per_source_test_scores.values()), axis=0)
    equal_auc = roc_auc_score(test_y, equal_score)

    weighted_score = sum(normalized_weights[k] * per_source_test_scores[k] for k in per_source_test_scores)
    weighted_auc = roc_auc_score(test_y, weighted_score)

    best_single = max(per_source_auc, key=per_source_auc.get)
    print(f"    EQUAL-WEIGHT ensemble:    AUC={equal_auc:.3f}")
    print(f"    RELIABILITY-WEIGHTED ensemble: AUC={weighted_auc:.3f}")
    best_overall = max(
        [("best single source", per_source_auc[best_single], best_single),
         ("equal-weight ensemble", equal_auc, None),
         ("weighted ensemble", weighted_auc, None)],
        key=lambda t: t[1],
    )
    print(f"    -> best single source = {best_single} ({per_source_auc[best_single]:.3f})  |  "
          f"WINNER: {best_overall[0]} ({best_overall[1]:.3f})")


def main():
    guard = CPUGuard()
    if not guard.check("score-ensemble setup"):
        return

    print("Loading feature extractors...")
    vib_encoder = load_encoder(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
    aud_encoder = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish.eval()

    from models.vggish_finetune import VGGishLastBlockDANN, _load_vggish_input_module
    print("Loading VGGish-finetuned...")
    vggish_ft = VGGishLastBlockDANN(vggish, n_domains=2)
    ckpt = torch.load(CHECKPOINT_DIR / "audio_vggish_finetune_lambda0.15.pt")
    vggish_ft.unfrozen_block.load_state_dict(ckpt["unfrozen_block"])
    vggish_ft.head.load_state_dict(ckpt["head"])
    vggish_ft.eval()
    vggish_input = _load_vggish_input_module()

    print("Extracting AI Mechanic features...")
    ai_files, ai_labels = [], []
    for f in sorted((RAW_DIR / "ai_mechanic").rglob("*.wav")):
        lbl = ai_mechanic_label_fn(f)
        if lbl != -1:
            ai_files.append(f); ai_labels.append(lbl)
    ai_labels = np.array(ai_labels)
    ai_blocks = build_feature_blocks(ai_files, ai_mechanic_reader, vib_encoder, aud_encoder, vggish, vggish_ft, vggish_input)

    print("Extracting Car Diagnostics features (1386 files)...")
    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    cd_files = sorted(cd_root.rglob("*.wav"))[::5] # REPRESENTATIVE SUBSET: 20%
    cd_labels = np.array([car_diagnostics_label_fn(f) for f in cd_files])
    cd_blocks = build_feature_blocks(cd_files, car_diagnostics_reader, vib_encoder, aud_encoder, vggish, vggish_ft, vggish_input)

    print("Extracting Engine Journal Bearings features (134 files)...")
    ejb_root = RAW_DIR / "engine_journal_bearings"
    ejb_files = sorted(ejb_root.rglob("*.csv"))[::2] # REPRESENTATIVE SUBSET: 50%
    ejb_labels = np.array([engine_journal_bearings_label_fn(f) for f in ejb_files])
    ejb_blocks = build_feature_blocks(ejb_files, engine_journal_bearings_reader, vib_encoder, aud_encoder, vggish, vggish_ft, vggish_input)

    print(f"\n{'='*70}\nAI MECHANIC — score-level ensemble\n{'='*70}")
    run_ensemble("ai_mechanic", ai_files, ai_labels, ai_blocks)

    print(f"\n{'='*70}\nCAR DIAGNOSTICS — score-level ensemble\n{'='*70}")
    run_ensemble("car_diagnostics", cd_files, cd_labels, cd_blocks)

    print(f"\n{'='*70}\nENGINE JOURNAL BEARINGS — score-level ensemble\n{'='*70}")
    run_ensemble("engine_journal_bearings", ejb_files, ejb_labels, ejb_blocks)

    print("\nDone.")


if __name__ == "__main__":
    main()
