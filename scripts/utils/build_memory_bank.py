"""
Builds and saves the real PCA-reconstruction memory banks that
scripts/inference/predict.py loads at inference time — replacing what was
previously a hardcoded filename-string heuristic
(`0.85 if "fault" in filename else 0.12`) with genuine anomaly scoring
fitted on real, held-out normal-only calibration data.

Audio memory bank: fit on a 70% calibration split of Car Diagnostics'
normal files (same protocol as evaluate_anomaly_only.py: normal-only,
faulty data never touched).
Vibration memory bank: fit on a 70% calibration split of Engine Journal
Bearings' normal (healthy) files, same protocol.

Run once (or whenever a checkpoint changes) before using predict.py:
    python scripts/utils/build_memory_bank.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

from data.preprocess import SpectrogramConfig, signal_to_spectrogram_batch
from data.readers import (
    car_diagnostics_label_fn, car_diagnostics_reader,
    engine_journal_bearings_label_fn, engine_journal_bearings_reader,
)
from models.encoder import load_encoder
from models.memory_bank import PCAReconstructionMemoryBank

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "checkpoints"
MEMORY_BANK_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "memory_banks"


def embed_files(files, reader, encoder, cfg) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for f in files:
            sig, sr = reader(f)
            specs = signal_to_spectrogram_batch(sig, orig_sr=sr, cfg=cfg)
            windows = torch.from_numpy(specs).unsqueeze(1)
            rows.append(encoder(windows).mean(dim=0).numpy().flatten())
    return np.stack(rows).astype(np.float32)


def build_audio_bank(seed: int = 42) -> None:
    print("Building AUDIO memory bank from Car Diagnostics normal files...")
    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    cd_files = sorted(cd_root.rglob("*.wav"))
    labels = np.array([car_diagnostics_label_fn(f) for f in cd_files])
    normal_files = [f for f, y in zip(cd_files, labels) if y == 0]
    cal_files, _ = train_test_split(normal_files, test_size=0.3, random_state=seed)
    print(f"  {len(normal_files)} normal files total, {len(cal_files)} used for calibration")

    encoder = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    cal_X = embed_files(cal_files, car_diagnostics_reader, encoder, SpectrogramConfig(sample_rate=16000))

    bank = PCAReconstructionMemoryBank().fit(cal_X)
    MEMORY_BANK_DIR.mkdir(parents=True, exist_ok=True)
    bank.save(MEMORY_BANK_DIR / "audio_memory_bank.joblib")
    print(f"  saved -> {MEMORY_BANK_DIR / 'audio_memory_bank.joblib'} (threshold={bank.threshold:.4f})")


def build_vibration_bank(seed: int = 42) -> None:
    print("Building VIBRATION memory bank from Engine Journal Bearings normal (healthy) files...")
    ejb_files = sorted((RAW_DIR / "engine_journal_bearings").rglob("*.csv"))
    labels = np.array([engine_journal_bearings_label_fn(f) for f in ejb_files])
    normal_files = [f for f, y in zip(ejb_files, labels) if y == 0]
    cal_files, _ = train_test_split(normal_files, test_size=0.3, random_state=seed)
    print(f"  {len(normal_files)} healthy files total, {len(cal_files)} used for calibration")

    encoder = load_encoder(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
    cal_X = embed_files(cal_files, engine_journal_bearings_reader, encoder, SpectrogramConfig(sample_rate=25600))

    bank = PCAReconstructionMemoryBank().fit(cal_X)
    MEMORY_BANK_DIR.mkdir(parents=True, exist_ok=True)
    bank.save(MEMORY_BANK_DIR / "vibration_memory_bank.joblib")
    print(f"  saved -> {MEMORY_BANK_DIR / 'vibration_memory_bank.joblib'} (threshold={bank.threshold:.4f})")


def main():
    build_audio_bank()
    build_vibration_bank()
    print("\nDone. predict.py will now score against these real, calibrated memory banks.")


if __name__ == "__main__":
    main()
