"""
Independent verification of audio_vggish_finetune_lambda0.15.pt after the
training run's own printed RESULT SUMMARY was lost (session interruption
severed the log pipe after the checkpoint had already been saved — the
process itself completed normally, this just recovers what its own final
validation would have reported).

The checkpoint only stores the embedding-producing backbone (unfrozen conv
block + small head), not the throwaway fault/domain heads used only during
training — consistent with every other checkpoint in this project. So
this re-fits a fresh downstream classifier on top of the frozen embeddings,
train-domain-data to val-domain-data, exactly the same "frozen encoder +
simple classifier" protocol used in check_encoder_beats_random_indomain.py,
giving a real, comparable, honestly-obtained number.

Processes in small batches (32 at a time, with progress printed per batch)
rather than one giant single-batch forward pass — a first attempt at this
used one monolithic batch of 2,700-3,150 images and was still running after
6+ minutes with no way to tell how much was left. Batches of 32 match what
was actually benchmarked earlier (~257ms/batch on this machine), so this
version is both faster and gives real incremental progress.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
BATCH_SIZE = 32


def build_backbone():
    vggish_full = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    blocks = list(vggish_full.features.children())
    frozen_blocks = nn.Sequential(*blocks[:11])
    unfrozen_block = nn.Sequential(*blocks[11:])
    head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 128))

    ckpt = torch.load(CHECKPOINT_DIR / "audio_vggish_finetune_lambda0.15.pt", map_location="cpu")
    unfrozen_block.load_state_dict(ckpt["unfrozen_block"])
    head.load_state_dict(ckpt["head"])
    frozen_blocks.eval()
    unfrozen_block.eval()
    head.eval()
    print(f"Loaded checkpoint: domains={ckpt['domain_names']}, grl_lambda={ckpt['grl_lambda']}, seed={ckpt['seed']}",
          flush=True)
    return frozen_blocks, unfrozen_block, head


def embed(frozen_blocks, unfrozen_block, head, X: np.ndarray, label: str) -> np.ndarray:
    n = len(X)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = torch.from_numpy(X[i:i + BATCH_SIZE]).unsqueeze(1)
            mid = frozen_blocks(batch)
            feats = unfrozen_block(mid)
            out.append(head(feats).numpy())
            done = i // BATCH_SIZE + 1
            if done % 10 == 0 or done == n_batches:
                elapsed = time.time() - t0
                print(f"  [{label}] batch {done}/{n_batches} ({elapsed:.1f}s elapsed)", flush=True)
    return np.concatenate(out, axis=0)


def evaluate_domain(name: str, frozen_blocks, unfrozen_block, head) -> None:
    d = np.load(CACHE_DIR / f"vggish_input_{name}.npz")
    X_train, y_train, X_val, y_val = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    print(f"[{name}] {len(X_train)} train, {len(X_val)} val examples", flush=True)

    Xtr_emb = embed(frozen_blocks, unfrozen_block, head, X_train, f"{name}/train")
    Xval_emb = embed(frozen_blocks, unfrozen_block, head, X_val, f"{name}/val")

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)
    clf.fit(Xtr_emb, y_train)
    preds = clf.predict(Xval_emb)
    scores = clf.predict_proba(Xval_emb)[:, 1]
    acc = accuracy_score(y_val, preds)
    try:
        auc = roc_auc_score(y_val, scores)
    except ValueError:
        auc = float("nan")
    print(f"  [{name}] fine-tuned-VGGish-backbone + fresh classifier: accuracy={acc:.3f}  AUC={auc:.3f}",
          flush=True)


def main():
    frozen_blocks, unfrozen_block, head = build_backbone()
    print("\n=== Verifying audio_vggish_finetune_lambda0.15.pt on its own training domains ===", flush=True)
    evaluate_domain("subf", frozen_blocks, unfrozen_block, head)
    evaluate_domain("mafaulda", frozen_blocks, unfrozen_block, head)
    print("\nCompare mafaulda AUC above against check_encoder_beats_random_indomain_log.txt: "
          "from-scratch trained encoder scored 0.615 AUC there, a random untrained encoder scored 0.863 AUC "
          "(the from-scratch encoder LOST to random on mafaulda specifically).", flush=True)


if __name__ == "__main__":
    main()
