"""
A persistable version of the PCA-reconstruction-error anomaly detector
already validated in scripts/eval/evaluate_anomaly_only.py (that script's
`pca_reconstruction_error_score` always re-fits from a calibration array
passed in fresh each call — fine for an evaluation sweep, but predict.py
needs to fit ONCE on real calibration data and then score one query
embedding at inference time, without needing the whole calibration set
present). This wraps the same method (same fixed-32-component cap, same
StandardScaler+PCA approach) in a fit/score/save/load class.

Threshold convention matches the rest of this project: the 90th percentile
of the calibration-normal scores themselves — not an arbitrary constant.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_MAX_COMPONENTS = 32


class PCAReconstructionMemoryBank:
    def __init__(self, max_components: int = DEFAULT_MAX_COMPONENTS):
        self.max_components = max_components
        self.scaler = None
        self.pca = None
        self.threshold = None

    def fit(self, cal_X: np.ndarray, threshold_percentile: float = 90.0) -> "PCAReconstructionMemoryBank":
        n_components = min(self.max_components, cal_X.shape[0] - 1, cal_X.shape[1])
        n_components = max(n_components, 1)
        self.scaler = StandardScaler().fit(cal_X)
        cal_scaled = self.scaler.transform(cal_X)
        self.pca = PCA(n_components=n_components, random_state=42).fit(cal_scaled)
        cal_scores = self._reconstruction_error(cal_scaled)
        self.threshold = float(np.percentile(cal_scores, threshold_percentile))
        return self

    def _reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        reduced = self.pca.transform(X_scaled)
        reconstructed = self.pca.inverse_transform(reduced)
        return np.mean((X_scaled - reconstructed) ** 2, axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.pca is None:
            raise RuntimeError("Memory bank not fitted/loaded yet.")
        return self._reconstruction_error(self.scaler.transform(X))

    def save(self, path) -> None:
        import joblib
        joblib.dump({"scaler": self.scaler, "pca": self.pca, "threshold": self.threshold,
                     "max_components": self.max_components}, path)

    @classmethod
    def load(cls, path) -> "PCAReconstructionMemoryBank":
        import joblib
        state = joblib.load(path)
        bank = cls(max_components=state["max_components"])
        bank.scaler = state["scaler"]
        bank.pca = state["pca"]
        bank.threshold = state["threshold"]
        return bank
