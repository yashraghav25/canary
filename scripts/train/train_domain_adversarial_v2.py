"""
Generalized N-domain DANN training — the "complete" version of the
training pipeline, covering all four vibration datasets (CWRU, IMS,
FEMTO, Paderborn) and the full 6-class MaFaulDa audio channel alongside
SUBF, instead of the earlier 2-domain-only version.

Carries forward every fix validated so far:
  - file-level splits before windowing (no window-overlap leakage)
  - balanced per-domain batch sampling (was the likely root cause of the
    "always predict majority domain" collapse in the 2-domain version)
  - fault-only warm-start epochs before the adversarial term switches on
  - per-step lambda ramp (Ganin & Lempitsky, 2016)
  - fixed seeds, single-variable ablation (grl_lambda only differs)
  - CPUGuard checked every epoch

All datasets here are REPRESENTATIVE SUBSETS, not full downloads (per
explicit instruction) — exact counts are printed at build time so this is
never silently pretending to use more data than it has.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from cpu_guard import CPUGuard
from data.preprocess import SpectrogramConfig, signal_to_spectrogram_batch
from data.readers import (
    cwru_label_fn_binary, cwru_reader,
    femto_label_fn_binary, femto_reader,
    ims_label_fn_binary, ims_reader,
    mafaulda_label_fn_binary, mafaulda_reader_audio, mafaulda_reader_vibration,
    paderborn_label_fn_binary, paderborn_reader,
    subf_label_fn, subf_reader,
)
from models.encoder import SpectrogramEncoder
from models.grl import grad_reverse

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _split_and_window(files, reader, label_fn, cfg, seed=42):
    """Shared helper: stratified file-level 80/20 split, then window each
    split independently, dropping any window whose label_fn returned -1
    (the FEMTO/IMS 'ambiguous middle of the run' convention)."""
    labels_for_split = []
    usable_files = []
    for f in files:
        lbl = label_fn(f)
        if lbl != -1:
            usable_files.append(f)
            labels_for_split.append(lbl)
    train_f, val_f = train_test_split(usable_files, test_size=0.2, random_state=seed, stratify=labels_for_split)

    def build(flist):
        X, y = [], []
        for f in flist:
            lbl = label_fn(f)
            sig, sr = reader(f)
            specs = signal_to_spectrogram_batch(sig, orig_sr=sr, cfg=cfg)
            for w in specs:
                X.append(w)
                y.append(lbl)
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)

    return build(train_f), build(val_f)


def build_all_vibration_domains(cache_key: str = "vib_domains_v2.npz") -> dict:
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        print(f"Loading cached full vibration domain set from {cache_path}")
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    cfg = SpectrogramConfig(sample_rate=25600)
    domains = {}

    print("CWRU...")
    cwru_files = sorted((RAW_DIR / "cwru" / "Data").rglob("*.npz"))
    (domains["cwru_X_train"], domains["cwru_y_train"]), (domains["cwru_X_val"], domains["cwru_y_val"]) = \
        _split_and_window(cwru_files, cwru_reader, cwru_label_fn_binary, cfg)
    print(f"  train={len(domains['cwru_y_train'])} val={len(domains['cwru_y_val'])}")

    print("IMS...")
    ims_files = sorted((RAW_DIR / "ims").rglob("*")) if (RAW_DIR / "ims").exists() else []
    # BUG FIX: IMS filenames are timestamps like "2004.02.12.10.32.39" — many
    # dots, so Path.suffix (text after the LAST dot) reads ".39" as a fake
    # extension, and the old `suffix == ""` filter silently excluded every
    # single IMS file (confirmed: previous run logged "SKIPPED - no IMS
    # files found" even though 984 real files exist on disk). Check
    # is_file() and exclude the actual 7z/rar leftovers by name instead.
    ims_files = [f for f in ims_files if f.is_file() and f.name not in ("IMS.7z", "2nd_test.rar")]
    if ims_files:
        (domains["ims_X_train"], domains["ims_y_train"]), (domains["ims_X_val"], domains["ims_y_val"]) = \
            _split_and_window(ims_files, ims_reader, ims_label_fn_binary, cfg)
        print(f"  train={len(domains['ims_y_train'])} val={len(domains['ims_y_val'])}")
    else:
        print("  SKIPPED — no IMS files found on disk")

    print("FEMTO...")
    femto_files = sorted((RAW_DIR / "femto" / "Learning_set").rglob("acc_*.csv")) if (RAW_DIR / "femto").exists() else []
    if femto_files:
        (domains["femto_X_train"], domains["femto_y_train"]), (domains["femto_X_val"], domains["femto_y_val"]) = \
            _split_and_window(femto_files, femto_reader, femto_label_fn_binary, cfg)
        print(f"  train={len(domains['femto_y_train'])} val={len(domains['femto_y_val'])}")
    else:
        print("  SKIPPED — no FEMTO files found on disk")

    print("Paderborn...")
    pb_files = sorted((RAW_DIR / "paderborn").rglob("*.mat")) if (RAW_DIR / "paderborn").exists() else []
    if pb_files:
        (domains["paderborn_X_train"], domains["paderborn_y_train"]), (domains["paderborn_X_val"], domains["paderborn_y_val"]) = \
            _split_and_window(pb_files, paderborn_reader, paderborn_label_fn_binary, cfg)
        print(f"  train={len(domains['paderborn_y_train'])} val={len(domains['paderborn_y_val'])}")
    else:
        print("  SKIPPED — no Paderborn .mat files found on disk (still packed in .rar?)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **domains)
    return domains


def build_all_audio_domains(subf_files_per_class: int = 50, cache_key: str = "aud_domains_v2.npz") -> dict:
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        print(f"Loading cached full audio domain set from {cache_path}")
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    cfg = SpectrogramConfig(sample_rate=16000)
    domains = {}

    print(f"SUBF, {subf_files_per_class}/class...")
    subf_files = []
    for c in ["Normal", "Inner Race Fault", "Outer Race Fault"]:
        files = sorted((RAW_DIR / "subf" / "Dataset" / c).glob("*.csv"),
                       key=lambda p: int(p.stem.split("(")[1].rstrip(")")))[:subf_files_per_class]
        subf_files += files
    (domains["subf_X_train"], domains["subf_y_train"]), (domains["subf_X_val"], domains["subf_y_val"]) = \
        _split_and_window(subf_files, subf_reader, subf_label_fn, cfg)
    print(f"  train={len(domains['subf_y_train'])} val={len(domains['subf_y_val'])}")

    print("MaFaulDa (all 6 classes now present)...")
    maf_files = sorted((RAW_DIR / "mafaulda").rglob("*.csv"))
    (domains["maf_X_train"], domains["maf_y_train"]), (domains["maf_X_val"], domains["maf_y_val"]) = \
        _split_and_window(maf_files, mafaulda_reader_audio, mafaulda_label_fn_binary, cfg)
    print(f"  train={len(domains['maf_y_train'])} val={len(domains['maf_y_val'])}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **domains)
    return domains


class DANNModelN(nn.Module):
    """Same as before but with an N-way (not fixed binary) domain head."""

    def __init__(self, n_domains: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = SpectrogramEncoder(embedding_dim)
        self.fault_head = nn.Linear(embedding_dim, 2)
        self.domain_head = nn.Sequential(nn.Linear(embedding_dim, 32), nn.ReLU(), nn.Linear(32, n_domains))

    def forward(self, spec: torch.Tensor, grl_lambda: float):
        emb = self.encoder(spec)
        return self.fault_head(emb), self.domain_head(grad_reverse(emb, grl_lambda))


def run_dann_n(
    domain_data: list[dict],  # each dict: {"X_train","y_train","X_val","y_val","name"}
    modality_name: str,
    grl_lambda: float,
    epochs: int,
    encoder_lr: float = 5e-4,
    domain_head_lr_mult: float = 0.5,
    max_grad_norm: float = 1.0,
    weight_decay: float = 1e-4,
    seed: int = 42,
    warm_start_epochs: int = 3,
) -> dict:
    n_domains = len(domain_data)
    names = [d["name"] for d in domain_data]
    print(f"\n{'='*70}\n{modality_name} | {n_domains} domains: {names} | grl_lambda={grl_lambda}\n{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    per_domain_pool_size = min(len(d["y_train"]) for d in domain_data)  # balance batches to the SMALLEST domain
    half_batch = max(4, min(16, per_domain_pool_size // 4))
    batch_size = half_batch * n_domains
    n_batches_per_epoch = max(1, max(len(d["y_train"]) for d in domain_data) // half_batch)
    total_steps = epochs * n_batches_per_epoch

    # Class-weighted fault loss: balancing per-DOMAIN batch composition
    # (half_batch draws equally from each domain) says nothing about the
    # fault/normal imbalance WITHIN a domain — e.g. CWRU has ~4 Normal files
    # vs. ~157 fault files. Left unweighted, the fault head (and, more
    # importantly, whatever the shared encoder trunk learns to represent)
    # was being shaped by an unaddressed within-domain imbalance. Inverse-
    # frequency class weights fix this without touching the domain-balancing
    # logic above, which addresses a different imbalance entirely.
    all_y_train = np.concatenate([d["y_train"] for d in domain_data])
    class_counts = np.bincount(all_y_train, minlength=2).astype(np.float64)
    class_weights = torch.tensor(class_counts.sum() / (2.0 * np.maximum(class_counts, 1)), dtype=torch.float32)

    model = DANNModelN(n_domains=n_domains)
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.fault_head.parameters(), "lr": encoder_lr},
            {"params": model.domain_head.parameters(), "lr": encoder_lr * domain_head_lr_mult},
        ],
        weight_decay=weight_decay,
    )
    fault_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    domain_loss_fn = nn.CrossEntropyLoss()

    X_trains = [torch.from_numpy(d["X_train"]).unsqueeze(1) for d in domain_data]
    y_trains = [torch.from_numpy(d["y_train"]) for d in domain_data]

    guard = CPUGuard()
    global_step = 0
    for epoch in range(epochs):
        if not guard.check(context=f"{modality_name}-N epoch {epoch+1}"):
            print(f"[cpu_guard] Aborting {modality_name} run early at epoch {epoch+1}/{epochs}.")
            break
        model.train()
        perms = [np.random.permutation(len(y)) for y in y_trains]
        total_fault_loss, total_domain_loss = 0.0, 0.0
        last_lambda = 0.0
        for step in range(n_batches_per_epoch):
            batch_specs, batch_faults, batch_domains = [], [], []
            for dom_id, (X, y, perm) in enumerate(zip(X_trains, y_trains, perms)):
                start = (step * half_batch) % len(perm)
                idx = perm[start : start + half_batch]
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_fault_loss += l_fault.item() * len(doms)
            total_domain_loss += l_domain.item() * len(doms)
            global_step += 1
        n_seen = n_batches_per_epoch * batch_size
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            tag = "warm-start" if epoch < warm_start_epochs else "adversarial"
            print(f"  epoch {epoch+1}/{epochs} [{tag}] lambda={last_lambda:.3f} "
                  f"fault_loss={total_fault_loss/n_seen:.4f} domain_loss={total_domain_loss/n_seen:.4f}")

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
    # seed suffix only for non-default seeds, so the original seed=42
    # checkpoint names (still the ones every downstream eval script loads
    # by default) don't change.
    seed_suffix = "" if seed == 42 else f"_seed{seed}"
    ckpt_name = f"{modality_name}_dann_lambda{grl_lambda}{seed_suffix}.pt"
    torch.save({"model": model.state_dict(), "domain_names": names, "grl_lambda": grl_lambda, "seed": seed},
               CHECKPOINT_DIR / ckpt_name)
    print(f"  saved checkpoint -> {CHECKPOINT_DIR / ckpt_name}")
    return results


STABILITY_SEEDS = [42, 43, 44]
"""Seed=42 is the original/default run (unchanged checkpoint names, still
what every downstream eval script loads). 43/44 are ADDITIONAL runs whose
only purpose is answering a criticism this pipeline never checked before:
was the reported adversarial-training result a stable outcome, or one lucky
run of a training method already proven to be capable of collapsing
entirely (see results-study.md Section 4)? Only the ADVERSARIAL
(grl_lambda>0) runs are repeated across all 3 seeds — the grl_lambda=0.0
baselines exist only for comparison and were never the object of the
stability question, so they run once (seed=42) to save real compute time."""


def _summarize_stability(per_seed_results: list[dict], label: str) -> None:
    print(f"\n  --- stability across {len(per_seed_results)} seeds: {label} ---")
    keys = sorted(per_seed_results[0].keys())
    for k in keys:
        vals = [r[k] for r in per_seed_results]
        print(f"    {k:28s} mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
              f"values={[round(v,3) for v in vals]}")


def build_paderborn_channel_domain(reader, cache_key: str) -> dict:
    """A single-channel Paderborn domain (current/force/torque), file-level
    split + windowed exactly like the vibration domains. Only ONE dataset
    (Paderborn) has these channels, so there is no second domain to be
    adversarial against here — this trains a plain encoder+fault-classifier,
    same situation as the original single-dataset SUBF smoke test (the
    domain head is a structural no-op with n_domains=1, not a bug)."""
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        d = np.load(cache_path)
        return {"name": cache_key.replace(".npz", ""),
                "X_train": d["X_train"], "y_train": d["y_train"],
                "X_val": d["X_val"], "y_val": d["y_val"]}

    from data.readers import PADERBORN_AUX_SAMPLE_RATE, PADERBORN_SAMPLE_RATE, paderborn_label_fn_binary
    sample_rate = PADERBORN_SAMPLE_RATE if "current" in cache_key else PADERBORN_AUX_SAMPLE_RATE
    cfg = SpectrogramConfig(sample_rate=sample_rate)
    files = sorted((RAW_DIR / "paderborn").rglob("*.mat"))
    (X_train, y_train), (X_val, y_val) = _split_and_window(files, reader, paderborn_label_fn_binary, cfg)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    return {"name": cache_key.replace(".npz", ""), "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}


def run_new_modalities() -> None:
    """Exhaustive follow-through on 'only 1 channel used per multi-sensor
    dataset': trains real encoders on Paderborn's current, force, and torque
    channels — genuine electrical/mechanical data, downloaded specifically
    for this and never used until now. HONEST SCOPE LIMIT, stated up front:
    neither held-out real-vehicle dataset (AI Mechanic, Car Diagnostics) has
    a current/force/torque channel — they're audio-only. So these new
    encoders can only be evaluated on Paderborn's OWN held-out split, the
    same structural limitation already documented for the fusion layer
    (results-study.md 5.5/5.6). This is a real, working capability
    demonstration, not a new number for the deployable vehicle-fault result."""
    from data.readers import paderborn_reader_current, paderborn_reader_force, paderborn_reader_torque

    print(f"\n\n{'='*70}\n=== NEW MODALITIES from Paderborn (current, force, torque) ===\n{'='*70}")
    channel_specs = [
        ("current", paderborn_reader_current, "paderborn_current_domain.npz"),
        ("force", paderborn_reader_force, "paderborn_force_domain.npz"),
        ("torque", paderborn_reader_torque, "paderborn_torque_domain.npz"),
    ]
    for name, reader, cache_key in channel_specs:
        print(f"\nBuilding Paderborn '{name}' domain...")
        domain = build_paderborn_channel_domain(reader, cache_key)
        print(f"  train={len(domain['y_train'])} val={len(domain['y_val'])}")
        result = run_dann_n([domain], f"paderborn_{name}", grl_lambda=0.0, epochs=20)
        fault_acc = result.get(f"{domain['name']}_fault_acc", float("nan"))
        print(f"  [{name}] Paderborn-internal held-out fault_acc: {fault_acc:.3f} "
              f"— NOT evaluable on AI Mechanic/Car Diagnostics (audio-only, no current/force/torque channel there)")


def main():
    print("=== Building vibration domains (CWRU + whatever else has landed) ===")
    vib_raw = build_all_vibration_domains()
    vib_domain_data = []
    for key in ["cwru", "ims", "femto", "paderborn"]:
        if f"{key}_X_train" in vib_raw:
            vib_domain_data.append({
                "name": key, "X_train": vib_raw[f"{key}_X_train"], "y_train": vib_raw[f"{key}_y_train"],
                "X_val": vib_raw[f"{key}_X_val"], "y_val": vib_raw[f"{key}_y_val"],
            })
    print(f"Vibration domains available: {[d['name'] for d in vib_domain_data]}")

    vib_adv_runs, aud_adv_runs = [], []
    if len(vib_domain_data) >= 2:
        vib_no_adv = run_dann_n(vib_domain_data, "vibration", grl_lambda=0.0, epochs=20, seed=42)
        for seed in STABILITY_SEEDS:
            vib_adv_runs.append(run_dann_n(vib_domain_data, "vibration", grl_lambda=0.3, epochs=20, seed=seed))
        _summarize_stability(vib_adv_runs, "vibration, adversarial (grl_lambda=0.3)")
    else:
        print("Fewer than 2 vibration domains available — skipping DANN (nothing to be adversarial against).")

    print("\n\n=== Building audio domains (SUBF + full 6-class MaFaulDa) ===")
    aud_raw = build_all_audio_domains()
    aud_domain_data = [
        {"name": "subf", "X_train": aud_raw["subf_X_train"], "y_train": aud_raw["subf_y_train"],
         "X_val": aud_raw["subf_X_val"], "y_val": aud_raw["subf_y_val"]},
        {"name": "mafaulda", "X_train": aud_raw["maf_X_train"], "y_train": aud_raw["maf_y_train"],
         "X_val": aud_raw["maf_X_val"], "y_val": aud_raw["maf_y_val"]},
    ]
    aud_no_adv = run_dann_n(aud_domain_data, "audio", grl_lambda=0.0, epochs=25, seed=42)
    for seed in STABILITY_SEEDS:
        aud_adv_runs.append(run_dann_n(aud_domain_data, "audio", grl_lambda=0.15, epochs=25, seed=seed))
    _summarize_stability(aud_adv_runs, "audio, adversarial (grl_lambda=0.15)")

    print(f"\n\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    print("vibration (no adversarial, seed=42):", vib_no_adv if len(vib_domain_data) >= 2 else "skipped")
    print("vibration (adversarial, 3 seeds):    ", "see stability summary above" if vib_adv_runs else "skipped")
    print("audio (no adversarial, seed=42):", aud_no_adv)
    print("audio (adversarial, 3 seeds):    see stability summary above")

    run_new_modalities()


if __name__ == "__main__":
    main()
