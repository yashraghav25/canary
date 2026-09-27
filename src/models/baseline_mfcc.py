"""
Track C — the cheap sanity-check baseline (plan Section 4.4, point 5),
following the MFCC + feature-selection + gradient-boosting approach that
reported near-perfect accuracy on real engine knock/normal classification
(see References [9] in vehicle-health-ai-plan.md).

Deliberately scikit-learn, not deep learning: trains in seconds on a CPU,
uses negligible RAM, and gives an honest floor the fancier pipeline must beat.
"""

import librosa
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_features(signal: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """39 features: MFCCs + delta + delta-delta, mean-pooled over time —
    matches the feature count used in the reference paper.

    librosa.feature.delta's default width=9 needs at least 9 time-frames;
    every dataset used until now was long/fast-sampled enough for that to
    always hold. A short, low-native-sample-rate clip (e.g. ~2s at ~296Hz)
    can produce as few as 2 MFCC frames, which crashed here the first time
    such a file was tried. Fixed by shrinking delta's width to fit however
    many frames are actually available, down to skipping delta entirely
    (zeros) if there aren't enough frames for it to mean anything."""
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    n_frames = mfcc.shape[1]
    if n_frames >= 3:
        width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
        delta = librosa.feature.delta(mfcc, width=width)
        delta2 = librosa.feature.delta(mfcc, order=2, width=width)
    else:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
    feats = np.concatenate([mfcc.mean(axis=1), delta.mean(axis=1), delta2.mean(axis=1)])
    return feats.astype(np.float32)


def build_pipeline(k_best: int = 8) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k_best)),
        ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ])
