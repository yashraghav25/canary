"""
Part 1 of the new-architecture plan: stop training a tiny custom encoder
from scratch, and instead lightly fine-tune VGGish — the model that has
outperformed our from-scratch encoders on 2 of 3 held-out real-vehicle
datasets so far (see results-study.md / teammate-update.md).

Design (the "middle ground" option, benchmarked before building this):
  - VGGish's first 3 conv blocks (generic low/mid-level filters) stay
    FROZEN — they already transfer fine and don't need retraining.
  - VGGish's last conv block (the two 512-channel layers — its most
    task-specific filters) is UNFROZEN and fine-tuned.
  - VGGish's own giant 67.6M-param FC head is DISCARDED entirely — a new,
    small (128-dim) projection head is attached directly on the conv
    features instead, mirroring our own SpectrogramEncoder's own head.
  - The usual domain-adversarial machinery (fault head + GRL domain head,
    balanced per-domain batches, warm-start epochs, per-step lambda ramp,
    class-weighted fault loss) is attached on top, trained on the same
    SUBF + MaFaulDa audio pretraining pool as before.
Measured cost on this machine: ~257ms/batch, ~21 minutes for a full
25-epoch run — the deliberately-chosen middle ground between a ~13-minute
option (freeze all of VGGish's conv layers too) and a ~45-minute option
(fine-tune VGGish's own FC layers instead of discarding them).

IMPORTANT: this uses VGGish's OWN input preprocessing (vggish_input.
waveform_to_examples -> (n, 1, 96, 64) log-mel patches), NOT this
project's own SpectrogramConfig — feeding VGGish's pretrained conv filters
data in a shape/normalization they were never trained on would degrade
whatever transfer benefit they have. This means a SEPARATE data cache from
vib_domains_v2.npz/aud_domains_v2.npz, built once here.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from cpu_guard import CPUGuard
from data.readers import mafaulda_label_fn_binary, mafaulda_reader_audio, subf_label_fn, subf_reader
from models.vggish_finetune import VGGishLastBlockDANN, _load_vggish_input_module

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"


def build_vggish_input_domain(files, reader, label_fn, cache_key: str, vggish_input, seed: int = 42) -> dict:
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    labels = [label_fn(f) for f in files]
    train_f, val_f = train_test_split(files, test_size=0.2, random_state=seed, stratify=labels)

    def build(flist):
        X, y = [], []
        for f in flist:
            sig, sr = reader(f)
            lbl = label_fn(f)
            examples = vggish_input.waveform_to_examples(sig, sr, return_tensor=False)
            for ex in examples:
                X.append(ex)
                y.append(lbl)
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)

    X_train, y_train = build(train_f)
    X_val, y_val = build(val_f)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}


def run_training(domain_data: list[dict], grl_lambda: float, epochs: int, seed: int = 42,
                  encoder_lr: float = 5e-4, domain_head_lr_mult: float = 0.5,
                  max_grad_norm: float = 1.0, weight_decay: float = 1e-4, warm_start_epochs: int = 3):
    n_domains = len(domain_data)
    names = [d["name"] for d in domain_data]
    print(f"\n{'='*70}\nVGGish-finetune audio | {n_domains} domains: {names} | grl_lambda={grl_lambda} | seed={seed}\n{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    all_y_train = np.concatenate([d["y_train"] for d in domain_data])
    class_counts = np.bincount(all_y_train, minlength=2).astype(np.float64)
    class_weights = torch.tensor(class_counts.sum() / (2.0 * np.maximum(class_counts, 1)), dtype=torch.float32)

    vggish_full = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    model = VGGishLastBlockDANN(vggish_full, n_domains=n_domains)

    optimizer = torch.optim.Adam(
        [
            {"params": model.unfrozen_block.parameters(), "lr": encoder_lr},
            {"params": model.head.parameters(), "lr": encoder_lr},
            {"params": model.fault_head.parameters(), "lr": encoder_lr},
            {"params": model.domain_head.parameters(), "lr": encoder_lr * domain_head_lr_mult},
        ],
        weight_decay=weight_decay,
    )
    fault_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    domain_loss_fn = nn.CrossEntropyLoss()

    per_domain_pool_size = min(len(d["y_train"]) for d in domain_data)
    half_batch = max(4, min(16, per_domain_pool_size // 4))
    n_batches_per_epoch = max(1, max(len(d["y_train"]) for d in domain_data) // half_batch)
    total_steps = epochs * n_batches_per_epoch

    # vggish_input.waveform_to_examples(..., return_tensor=False) returns
    # (n, 96, 64) — no channel dim (unlike return_tensor=True's (n,1,96,64),
    # confirmed by direct inspection) — add it back here at batch-construction
    # time, consistent with how every other dataset in this project does
    # .unsqueeze(1) rather than baking the channel dim into the cache itself.
    X_trains = [torch.from_numpy(d["X_train"]).unsqueeze(1) for d in domain_data]
    y_trains = [torch.from_numpy(d["y_train"]) for d in domain_data]

    guard = CPUGuard()
    global_step = 0
    t0 = time.time()
    for epoch in range(epochs):
        if not guard.check(context=f"vggish-finetune epoch {epoch+1}/{epochs}"):
            print(f"[cpu_guard] Aborting VGGish fine-tune early at epoch {epoch+1}/{epochs}.")
            break
        model.train()
        model.frozen_blocks.eval()
        perms = [np.random.permutation(len(y)) for y in y_trains]
        total_fault_loss, total_domain_loss = 0.0, 0.0
        last_lambda = 0.0
        for step in range(n_batches_per_epoch):
            batch_specs, batch_faults, batch_domains = [], [], []
            for dom_id, (X, y, perm) in enumerate(zip(X_trains, y_trains, perms)):
                start = (step * half_batch) % len(perm)
                idx = perm[start:start + half_batch]
                if len(idx) < half_batch:
                    idx = np.concatenate([idx, perm[: half_batch - len(idx)]])
                batch_specs.append(X[idx])
                batch_faults.append(y[idx])
                batch_domains.append(torch.full((half_batch,), dom_id, dtype=torch.long))
            specs = torch.cat(batch_specs)
            faults = torch.cat(batch_faults)
            doms = torch.cat(batch_domains)

            if epoch < warm_start_epochs:
                current_lambda = 0.0
            else:
                progress = (global_step - warm_start_epochs * n_batches_per_epoch) / max(
                    total_steps - warm_start_epochs * n_batches_per_epoch - 1, 1
                )
                current_lambda = grl_lambda * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)
            last_lambda = current_lambda

            fault_logits, domain_logits = model(specs, current_lambda)
            l_fault = fault_loss_fn(fault_logits, faults)
            l_domain = domain_loss_fn(domain_logits, doms)
            loss = l_fault + l_domain
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_grad_norm)
            optimizer.step()
            total_fault_loss += l_fault.item() * len(doms)
            total_domain_loss += l_domain.item() * len(doms)
            global_step += 1

        n_seen = n_batches_per_epoch * half_batch * n_domains
        elapsed = time.time() - t0
        tag = "warm-start" if epoch < warm_start_epochs else "adversarial"
        print(f"  epoch {epoch+1}/{epochs} [{tag}] lambda={last_lambda:.3f} "
              f"fault_loss={total_fault_loss/n_seen:.4f} domain_loss={total_domain_loss/n_seen:.4f} "
              f"elapsed={elapsed:.0f}s", flush=True)

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"audio_vggish_finetune_lambda{grl_lambda}_epoch{epoch+1}.pt"
        torch.save({
            "unfrozen_block": model.unfrozen_block.state_dict(),
            "head": model.head.state_dict(),
            "domain_names": names, "grl_lambda": grl_lambda, "seed": seed,
        }, CHECKPOINT_DIR / ckpt_name)

    model.eval()
    results = {}
    with torch.no_grad():
        for dom_id, d in enumerate(domain_data):
            X_val_t = torch.from_numpy(d["X_val"]).unsqueeze(1)
            y_val_t = torch.from_numpy(d["y_val"])
            fault_logits, domain_logits = model(X_val_t, grl_lambda=0.0)
            fault_acc = (fault_logits.argmax(1) == y_val_t).float().mean().item()
            domain_acc = (domain_logits.argmax(1) == dom_id).float().mean().item()
            results[f"{d['name']}_fault_acc"] = fault_acc
            results[f"{d['name']}_domain_acc"] = domain_acc
            print(f"  [{d['name']}] fault_acc={fault_acc:.3f}  domain_acc={domain_acc:.3f} "
                  f"(chance=1/{n_domains}={1/n_domains:.3f})")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"audio_vggish_finetune_lambda{grl_lambda}.pt"
    torch.save({
        "unfrozen_block": model.unfrozen_block.state_dict(),
        "head": model.head.state_dict(),
        "domain_names": names, "grl_lambda": grl_lambda, "seed": seed,
    }, CHECKPOINT_DIR / ckpt_name)
    print(f"  saved checkpoint -> {CHECKPOINT_DIR / ckpt_name}")
    return results


def main():
    guard = CPUGuard()
    if not guard.check("vggish-finetune setup"):
        return

    print("Loading VGGish (to locate its bundled preprocessing module)...")
    torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish_input = _load_vggish_input_module()

    print("Building VGGish-format (96x64 log-mel patch) SUBF domain...")
    subf_files = []
    for c in ["Normal", "Inner Race Fault", "Outer Race Fault"]:
        files = sorted((RAW_DIR / "subf" / "Dataset" / c).glob("*.csv"),
                       key=lambda p: int(p.stem.split("(")[1].rstrip(")")))[:50]
        subf_files += files
    subf_domain = build_vggish_input_domain(subf_files, subf_reader, subf_label_fn,
                                             "vggish_input_subf.npz", vggish_input)
    subf_domain["name"] = "subf"
    print(f"  train={len(subf_domain['y_train'])} val={len(subf_domain['y_val'])}")

    print("Building VGGish-format (96x64 log-mel patch) MaFaulDa domain...")
    maf_files = sorted((RAW_DIR / "mafaulda").rglob("*.csv"))
    maf_domain = build_vggish_input_domain(maf_files, mafaulda_reader_audio, mafaulda_label_fn_binary,
                                            "vggish_input_mafaulda.npz", vggish_input)
    maf_domain["name"] = "mafaulda"
    print(f"  train={len(maf_domain['y_train'])} val={len(maf_domain['y_val'])}")

    results = run_training([subf_domain, maf_domain], grl_lambda=0.15, epochs=25, seed=42)

    print(f"\n{'='*70}\nRESULT SUMMARY\n{'='*70}")
    print(results)
    print("\nCompare mafaulda_fault_acc above against check_encoder_beats_random_indomain_log.txt's "
          "mafaulda numbers (trained-from-scratch encoder scored 0.615 AUC there, a random untrained "
          "encoder scored 0.863 AUC — the from-scratch encoder lost). Metrics differ (accuracy here vs "
          "AUC there) so this is a directional check, not a direct number-for-number comparison.")


if __name__ == "__main__":
    main()
