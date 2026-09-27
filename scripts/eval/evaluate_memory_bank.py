import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cpu_guard import CPUGuard
from data.readers import (
    ai_mechanic_label_fn, ai_mechanic_reader,
    car_diagnostics_label_fn, car_diagnostics_reader,
    engine_journal_bearings_label_fn, engine_journal_bearings_reader,
)

warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def extract_window_features(files, reader, vggish):
    # Returns a list of 2D numpy arrays, one per file
    features = []
    failures = []
    for f in files:
        sig, sr = reader(f)
        try:
            with torch.no_grad():
                vgg_raw = vggish.forward(sig, fs=sr)
            if vgg_raw.dim() == 1:
                vgg_raw = vgg_raw.unsqueeze(0)
            if vgg_raw.numel() == 0:
                raise RuntimeError("VGGish returned an empty tensor (clip too short for its framing)")
            features.append(vgg_raw.cpu().numpy().astype(np.float32))
        except Exception as e:
            failures.append((f.name, str(e)))
            features.append(np.zeros((1, 128), dtype=np.float32))
    if failures:
        print(f"    [warn] feature extraction failed on {len(failures)}/{len(files)} files — "
              f"zero-filled, so results for this dataset are DEGRADED, not fully missing. "
              f"First failure: {failures[0]}")
    return features

def run_memory_bank(name, files, labels, window_features, test_size=0.3, seed=42):
    y = np.array(labels)
    idx_normal = np.where(y == 0)[0]
    idx_faulty = np.where(y == 1)[0]
    cal_idx, test_normal_idx = train_test_split(idx_normal, test_size=test_size, random_state=seed)
    
    test_idx = np.concatenate([test_normal_idx, idx_faulty])
    test_y = np.concatenate([np.zeros(len(test_normal_idx)), np.ones(len(idx_faulty))])
    
    print(f"\n[{name}] cal_normal={len(cal_idx)} test_normal={len(test_normal_idx)} test_faulty={len(idx_faulty)}")
    
    # 1. Build Memory Bank (concatenate all windows from calibration normal files)
    cal_windows = []
    for i in cal_idx:
        cal_windows.append(window_features[i])
    memory_bank = np.vstack(cal_windows)
    
    # 2. Fit k-NN Memory Bank
    scaler = StandardScaler().fit(memory_bank)
    memory_bank_scaled = scaler.transform(memory_bank)
    
    k = min(5, len(memory_bank_scaled))
    nn = NearestNeighbors(n_neighbors=k).fit(memory_bank_scaled)
    
    # 3. Score Test Files
    scores_mean = []
    scores_max = []
    scores_p90 = []
    
    for i in test_idx:
        file_windows = window_features[i]
        file_windows_scaled = scaler.transform(file_windows)
        dists, _ = nn.kneighbors(file_windows_scaled)
        # Average distance to k nearest neighbors for each window
        window_anomaly_scores = dists.mean(axis=1)
        
        # Aggregate to file level
        scores_mean.append(np.mean(window_anomaly_scores))
        scores_max.append(np.max(window_anomaly_scores))
        scores_p90.append(np.percentile(window_anomaly_scores, 90))
        
    auc_mean = roc_auc_score(test_y, scores_mean)
    auc_max = roc_auc_score(test_y, scores_max)
    auc_p90 = roc_auc_score(test_y, scores_p90)
    
    print(f"    AUC (Mean pooling - old way) : {auc_mean:.3f}")
    print(f"    AUC (90th percentile)        : {auc_p90:.3f}")
    print(f"    AUC (Max pooling - PatchCore): {auc_max:.3f}")

def main():
    guard = CPUGuard()
    if not guard.check("memory-bank setup"):
        return

    print("Loading VGGish...")
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish.eval()

    print("Extracting AI Mechanic window features...")
    ai_files, ai_labels = [], []
    for f in sorted((RAW_DIR / "ai_mechanic").rglob("*.wav")):
        lbl = ai_mechanic_label_fn(f)
        if lbl != -1:
            ai_files.append(f); ai_labels.append(lbl)
    ai_features = extract_window_features(ai_files, ai_mechanic_reader, vggish)
    run_memory_bank("AI Mechanic", ai_files, ai_labels, ai_features)

    print("\nExtracting Car Diagnostics window features (20% subset to be fast)...")
    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    cd_files = sorted(cd_root.rglob("*.wav"))[::5]
    cd_labels = [car_diagnostics_label_fn(f) for f in cd_files]
    cd_features = extract_window_features(cd_files, car_diagnostics_reader, vggish)
    run_memory_bank("Car Diagnostics (20%)", cd_files, cd_labels, cd_features)

    print("\nExtracting Engine Journal Bearings window features...")
    ejb_root = RAW_DIR / "engine_journal_bearings"
    ejb_files = sorted(ejb_root.rglob("*.csv"))
    ejb_labels = [engine_journal_bearings_label_fn(f) for f in ejb_files]
    ejb_features = extract_window_features(ejb_files, engine_journal_bearings_reader, vggish)
    run_memory_bank("Engine Journal Bearings", ejb_files, ejb_labels, ejb_features)

if __name__ == "__main__":
    main()
