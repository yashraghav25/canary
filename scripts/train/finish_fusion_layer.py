"""
Actually finishing the cross-modal fusion layer (Stage 2 of the plan),
properly this time — not a token re-run, a real fix to what was
incomplete:

WHAT WAS INCOMPLETE: the original fusion test (train_mafaulda_fusion.py)
trained two encoders FROM SCRATCH on a small, mostly single-class MaFaulDa
slice, and was never revisited after (a) the full 6-class MaFaulDa data
arrived and (b) the domain-adversarial encoders were trained to
convergence. It also never fed back into the actual pipeline anywhere.

THE HONEST DESIGN PROBLEM WITH "FINISHING" IT: the fusion layer needs a
PAIRED (vibration, audio) sample to do anything — but our held-out real
vehicle datasets (AI Mechanic, Car Diagnostics) are AUDIO-ONLY. There is
no real vehicle vibration channel to pair with. So the fusion layer
itself literally cannot run at inference time on the held-out data.

THE CORRECT FIX (standard practice for this exact situation — this is
how CLIP-style joint encoders are actually used downstream): don't try
to run the fusion layer at inference time. Instead, use the cross-modal
CONTRASTIVE LOSS as an additional fine-tuning objective on the ALREADY-
TRAINED audio and vibration encoders, using MaFaulDa's real paired
channels (now all 6 classes, not just "normal"). The audio encoder that
comes out the other end has been shaped by "does this sound correspond
to this real vibration event", even though at deployment it only ever
sees audio alone — exactly how a jointly-pretrained encoder is meant to
be used. This is then compared, head-to-head, against the pre-fusion
audio encoder on the exact same held-out evaluation, so we know whether
finishing this step actually helped or not.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from cpu_guard import CPUGuard
from data.preprocess import SpectrogramConfig, signal_to_spectrogram_batch
from data.readers import mafaulda_reader_paired
from models.encoder import SpectrogramEncoder
from models.fusion import FusionLayer

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "checkpoints"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_encoder_weights(ckpt_path: Path, embedding_dim: int = 128) -> SpectrogramEncoder:
    encoder = SpectrogramEncoder(embedding_dim)
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw["model"] if "model" in raw else raw
    encoder_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    return encoder


def build_all_mafaulda_pairs_by_file():
    """Returns PER-FILE lists of windows (not one flat concatenated array) so
    the caller can split train/val by FILE before concatenating — splitting
    the flat concatenation instead (the old behavior, despite a log message
    that incorrectly claimed "file-level split preserved") put overlapping
    windows from the SAME file on both sides of the split, since windows
    share 50% overlap (hop_seconds=0.75 vs window_seconds=1.5). This is the
    identical leakage class already found and fixed for CWRU and SUBF
    earlier in the project, reintroduced here and now fixed the same way:
    split whole files first, window each side independently."""
    cache_path = CACHE_DIR / "mafaulda_all6_pairs_by_file.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return list(d["vib_per_file"]), list(d["aud_per_file"])

    cfg = SpectrogramConfig(sample_rate=25600, window_seconds=1.5, hop_seconds=0.75)
    files = sorted(RAW_DIR.glob("mafaulda/**/*.csv"))
    print(f"Building fusion pairs from {len(files)} real MaFaulDa files (all 6 classes)")
    vib_per_file, aud_per_file = [], []
    t0 = time.time()
    for f in files:
        vib_sig, aud_sig, sr = mafaulda_reader_paired(f)
        vw = signal_to_spectrogram_batch(vib_sig, orig_sr=sr, cfg=cfg)
        aw = signal_to_spectrogram_batch(aud_sig, orig_sr=sr, cfg=cfg)
        n = min(len(vw), len(aw))
        vib_per_file.append(vw[:n].astype(np.float32))
        aud_per_file.append(aw[:n].astype(np.float32))
    total_windows = sum(len(v) for v in vib_per_file)
    print(f"Built {total_windows} real paired windows across {len(files)} files in {time.time()-t0:.0f}s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path,
             vib_per_file=np.array(vib_per_file, dtype=object),
             aud_per_file=np.array(aud_per_file, dtype=object))
    return vib_per_file, aud_per_file


def main():
    guard = CPUGuard()
    if not guard.check("fusion fine-tune setup"):
        return

    print("Loading the ALREADY domain-adversarially-trained encoders (warm start, not from scratch)...")
    vib_encoder = load_encoder_weights(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
    aud_encoder = load_encoder_weights(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")

    vib_per_file, aud_per_file = build_all_mafaulda_pairs_by_file()
    file_idx = np.arange(len(vib_per_file))
    train_files, val_files = train_test_split(file_idx, test_size=0.2, random_state=42)

    vib_all_train = np.concatenate([vib_per_file[i] for i in train_files])
    aud_all_train = np.concatenate([aud_per_file[i] for i in train_files])
    vib_all_val = np.concatenate([vib_per_file[i] for i in val_files])
    aud_all_val = np.concatenate([aud_per_file[i] for i in val_files])
    print(f"Fusion fine-tune: {len(train_files)} train files ({len(vib_all_train)} windows), "
          f"{len(val_files)} val files ({len(vib_all_val)} windows) — split by FILE before windowing, "
          f"so no overlapping window from a train file can appear in val.")

    vib_train = torch.from_numpy(vib_all_train).unsqueeze(1)
    aud_train = torch.from_numpy(aud_all_train).unsqueeze(1)
    vib_val = torch.from_numpy(vib_all_val).unsqueeze(1)
    aud_val = torch.from_numpy(aud_all_val).unsqueeze(1)

    # Small learning rate: this is FINE-TUNING an already-good encoder, not
    # training from scratch — a large LR here would just destroy the
    # domain-adversarial progress already made.
    optimizer = torch.optim.Adam(
        list(vib_encoder.parameters()) + list(aud_encoder.parameters()), lr=1e-4
    )

    epochs = 15
    batch_size = 32
    best_val_retrieval = 0.0
    for epoch in range(epochs):
        if not guard.check(f"fusion fine-tune epoch {epoch+1}"):
            print("Aborting fusion fine-tune early.")
            break
        vib_encoder.train(); aud_encoder.train()
        perm = np.random.permutation(len(vib_train))
        total_loss = 0.0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            vb = vib_encoder(vib_train[idx])
            ab = aud_encoder(aud_train[idx])
            loss = FusionLayer.paired_contrastive_loss(vb, ab)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(vib_encoder.parameters()) + list(aud_encoder.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)

        vib_encoder.eval(); aud_encoder.eval()
        with torch.no_grad():
            vb_val = vib_encoder(vib_val)
            ab_val = aud_encoder(aud_val)
            vib_n = torch.nn.functional.normalize(vb_val, dim=-1)
            aud_n = torch.nn.functional.normalize(ab_val, dim=-1)
            sims = vib_n @ aud_n.T
            preds = sims.argmax(dim=1)
            targets = torch.arange(len(vb_val))
            val_retrieval = (preds == targets).float().mean().item()

        print(f"epoch {epoch+1}/{epochs}  train_loss={total_loss/len(vib_train):.4f}  "
              f"val_top1_retrieval={val_retrieval:.3f} (chance={1/len(vib_val):.4f})")

        if val_retrieval > best_val_retrieval:
            best_val_retrieval = val_retrieval
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"vib_encoder": vib_encoder.state_dict()}, CHECKPOINT_DIR / "vibration_encoder_postfusion.pt")
            torch.save({"aud_encoder": aud_encoder.state_dict()}, CHECKPOINT_DIR / "audio_encoder_postfusion.pt")

    print(f"\nBest val top-1 retrieval after fine-tuning on ALL 6 MaFaulDa classes: {best_val_retrieval:.3f} "
          f"(chance={1/len(vib_val):.4f}) — compare against the original 0.100 (chance=0.020) from the "
          f"single-class MaFaulDa test.")
    print("Saved post-fusion encoders (state_dict wrapped under 'vib_encoder'/'aud_encoder' keys, "
          "NOT the DANN-model format — load via SpectrogramEncoder directly).")


if __name__ == "__main__":
    main()
