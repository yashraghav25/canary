"""
Universal signal -> log-mel spectrogram representation.

This is the single function every dataset (audio, vibration, or current)
passes through before touching a model, per Section 4.1 of
vehicle-health-ai-plan.md: "everything becomes a spectrogram."
"""

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class SpectrogramConfig:
    sample_rate: int
    n_mels: int = 128
    n_fft: int = 1024
    fmin: int = 20
    fmax: int | None = None
    window_seconds: float = 1.5
    hop_seconds: float = 0.75


def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal.astype(np.float32)
    return librosa.resample(signal.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def window_signal(signal: np.ndarray, sr: int, window_seconds: float, hop_seconds: float) -> np.ndarray:
    """Returns shape (n_windows, window_samples). Drops a trailing partial window."""
    win_len = int(window_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(signal) < win_len:
        pad = np.zeros(win_len - len(signal), dtype=signal.dtype)
        signal = np.concatenate([signal, pad])
    n_windows = 1 + (len(signal) - win_len) // hop_len
    windows = np.stack([signal[i * hop_len : i * hop_len + win_len] for i in range(max(n_windows, 1))])
    return windows


def log_mel_spectrogram(window: np.ndarray, cfg: SpectrogramConfig) -> np.ndarray:
    """One window -> (n_mels, time) log-mel spectrogram, unit-normalized."""
    fmax = cfg.fmax or cfg.sample_rate // 2
    mel = librosa.feature.melspectrogram(
        y=window,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # per-frame normalization so different sources/recording gains are comparable
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)


def signal_to_spectrogram_batch(
    signal: np.ndarray, orig_sr: int, cfg: SpectrogramConfig
) -> np.ndarray:
    """Full pipeline: raw signal (any orig_sr) -> (n_windows, n_mels, time) log-mel batch."""
    resampled = resample_signal(signal, orig_sr, cfg.sample_rate)
    windows = window_signal(resampled, cfg.sample_rate, cfg.window_seconds, cfg.hop_seconds)
    specs = np.stack([log_mel_spectrogram(w, cfg) for w in windows])
    return specs
