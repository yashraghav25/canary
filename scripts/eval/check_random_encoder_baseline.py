"""
Sanity check for the open question in results-study.md / teammate-update.md:
why do our trained encoders (and even generic VGGish) do reasonably well on
datasets/modalities they were never trained on? One candidate explanation:
a randomly-initialized CNN's convolutional filters are decent generic
edge/texture detectors even with ZERO training (a known effect in anomaly
detection literature) — meaning the "advantage" we've been crediting to
domain-adversarial pretraining might not require any real learning at all.

This script builds a SpectrogramEncoder with RANDOM, UNTRAINED weights (same
architecture as vib_encoder/aud_encoder, just never fit on any data) and
runs it through the exact same anomaly-only protocol as
evaluate_anomaly_only.py, on the same two datasets already tested there.
If the random encoder scores close to the trained ones, that's real
evidence the trained encoders aren't earning their keep on out-of-domain
data. If it scores much worse, that's real evidence the trained encoders
ARE contributing real, learned signal.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import warnings
from pathlib import Path

import numpy as np
import torch

from cpu_guard import CPUGuard
from data.preprocess import SpectrogramConfig, signal_to_spectrogram_batch
from data.readers import (
    car_diagnostics_label_fn, car_diagnostics_reader,
    engine_journal_bearings_label_fn, engine_journal_bearings_reader,
)
from evaluate_anomaly_only import run_dataset
from models.encoder import SpectrogramEncoder

warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RANDOM_CFG = SpectrogramConfig(sample_rate=16000)  # matches the audio encoder's training rate


def build_random_encoder_blocks(files, reader, encoder):
    rows = []
    for f in files:
        sig, sr = reader(f)
        specs = signal_to_spectrogram_batch(sig, orig_sr=sr, cfg=RANDOM_CFG)
        with torch.no_grad():
            windows = torch.from_numpy(specs).unsqueeze(1)
            rows.append(encoder(windows).mean(dim=0).numpy().flatten())
    return np.stack(rows).astype(np.float32)


def main():
    guard = CPUGuard()
    if not guard.check("random-encoder sanity check setup"):
        return

    torch.manual_seed(0)
    random_encoder = SpectrogramEncoder(128)
    random_encoder.eval()
    print("Built a randomly-initialized, completely UNTRAINED SpectrogramEncoder "
          "(same architecture as vib_encoder/aud_encoder, zero training).")

    print("\nExtracting Car Diagnostics features with the random encoder...")
    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    cd_files = sorted(cd_root.rglob("*.wav"))
    cd_labels = np.array([car_diagnostics_label_fn(f) for f in cd_files])
    cd_random_X = build_random_encoder_blocks(cd_files, car_diagnostics_reader, random_encoder)

    print("Extracting Engine Journal Bearings features with the random encoder...")
    ejb_root = RAW_DIR / "engine_journal_bearings"
    ejb_files = sorted(ejb_root.rglob("*.csv"))
    ejb_labels = np.array([engine_journal_bearings_label_fn(f) for f in ejb_files])
    ejb_random_X = build_random_encoder_blocks(ejb_files, engine_journal_bearings_reader, random_encoder)

    print(f"\n{'='*70}\nCAR DIAGNOSTICS — RANDOM (untrained) encoder anomaly-only\n"
          f"(compare against trained 'aud'=0.869/0.890 best, 'vib'=0.907/0.933 best)\n{'='*70}")
    run_dataset("car_diagnostics", cd_files, cd_labels, {"random": cd_random_X}, "random")

    print(f"\n{'='*70}\nENGINE JOURNAL BEARINGS — RANDOM (untrained) encoder anomaly-only\n"
          f"(compare against trained 'aud'=0.944, 'vib'=0.953, 'vgg'=0.996 best)\n{'='*70}")
    run_dataset("engine_journal_bearings", ejb_files, ejb_labels, {"random": ejb_random_X}, "random")

    print("\nDone. If these numbers are close to the trained encoders' numbers above, the "
          "trained encoders are not clearly earning their keep on this out-of-domain data — the "
          "'advantage' seen elsewhere may be generic CNN structure, not learned bearing-fault knowledge.")


if __name__ == "__main__":
    main()
