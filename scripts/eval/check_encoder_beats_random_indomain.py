"""
Sanity gate (Part 3 of the new-architecture plan): does our TRAINED
vibration/audio encoder actually beat a randomly-initialized, untrained
encoder of the SAME architecture, on the encoder's OWN pretraining domain?

This has never actually been checked. The random-encoder test in
check_random_encoder_baseline.py showed our trained encoders barely beat
(or lost to) a random one on OUT-of-domain held-out vehicle data — but
that could mean either (a) the training pipeline itself doesn't produce a
meaningfully better representation even in-domain, or (b) it does work
in-domain but just doesn't transfer out-of-domain. Those are very
different problems with very different fixes, and this script tells them
apart.

Protocol: for each vibration domain (CWRU/IMS/FEMTO/Paderborn) and each
audio domain (SUBF/MaFaulDa), using the SAME cached windows the original
DANN training used — extract embeddings with (a) the trained encoder,
frozen, and (b) a randomly-initialized encoder of the same architecture,
also frozen. Fit an IDENTICAL downstream classifier (GradientBoosting) on
top of each, train-embeddings-to-train-labels, evaluate on the held-out
val split. Whichever embedding lets the SAME simple classifier do better is
the one actually carrying more real information.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from cpu_guard import CPUGuard
from evaluate_stage3_calibrated import load_encoder
from models.encoder import SpectrogramEncoder

warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"


def embed(encoder, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(X).unsqueeze(1)
        return encoder(t).numpy()


def compare_domain(name: str, X_train, y_train, X_val, y_val, trained_encoder, random_encoder) -> None:
    Xtr_trained, Xval_trained = embed(trained_encoder, X_train), embed(trained_encoder, X_val)
    Xtr_random, Xval_random = embed(random_encoder, X_train), embed(random_encoder, X_val)

    results = {}
    for label, Xtr, Xval in [("trained", Xtr_trained, Xval_trained), ("random (untrained)", Xtr_random, Xval_random)]:
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xval)
        scores = clf.predict_proba(Xval)[:, 1]
        acc = accuracy_score(y_val, preds)
        try:
            auc = roc_auc_score(y_val, scores)
        except ValueError:
            auc = float("nan")
        results[label] = (acc, auc)
        print(f"    [{label:20s}] accuracy={acc:.3f}  AUC={auc:.3f}")

    trained_auc, random_auc = results["trained"][1], results["random (untrained)"][1]
    verdict = "TRAINED WINS" if trained_auc > random_auc else ("TIE" if trained_auc == random_auc else "RANDOM WINS — real problem")
    print(f"    -> {verdict} (trained {trained_auc:.3f} vs random {random_auc:.3f})")


def main():
    guard = CPUGuard()
    if not guard.check("encoder-vs-random in-domain sanity gate"):
        return

    torch.manual_seed(0)

    print("=== VIBRATION domains (encoder's own pretraining data) ===")
    vib_trained = load_encoder(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
    vib_random = SpectrogramEncoder(128)
    vib_random.eval()

    vib_data = np.load(CACHE_DIR / "vib_domains_v2.npz")
    for key in ["cwru", "ims", "femto", "paderborn"]:
        if f"{key}_X_train" not in vib_data:
            continue
        print(f"\n  --- {key} ---")
        compare_domain(key, vib_data[f"{key}_X_train"], vib_data[f"{key}_y_train"],
                        vib_data[f"{key}_X_val"], vib_data[f"{key}_y_val"], vib_trained, vib_random)

    print("\n\n=== AUDIO domains (encoder's own pretraining data) ===")
    aud_trained = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    aud_random = SpectrogramEncoder(128)
    aud_random.eval()

    aud_data = np.load(CACHE_DIR / "aud_domains_v2.npz")
    for key, label in [("subf", "subf"), ("maf", "mafaulda")]:
        print(f"\n  --- {label} ---")
        compare_domain(label, aud_data[f"{key}_X_train"], aud_data[f"{key}_y_train"],
                        aud_data[f"{key}_X_val"], aud_data[f"{key}_y_val"], aud_trained, aud_random)

    print("\nDone. 'RANDOM WINS' on any domain here means the DANN training pipeline is not "
          "producing a better-than-random representation even on the data it was directly trained "
          "on — a training-pipeline problem, not (only) a transfer problem.")


if __name__ == "__main__":
    main()
