import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import warnings
from pathlib import Path
import numpy as np
import torch
import librosa
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from data.readers import ai_mechanic_label_fn, ai_mechanic_reader

warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def bandpass_filter(data, fs, lowcut=100.0, highcut=8000.0, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    if low >= 1.0 or high <= 0.0 or low >= high:
        return data
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_hybrid_features(f, reader, vggish):
    sig, sr = reader(f)
    
    # 1. Bandpass Filtering (Remove wind/rumble below 100Hz and hiss above 7500Hz)
    sig = bandpass_filter(sig, sr, lowcut=100.0, highcut=7500.0)
    
    # Process through VGGish
    with torch.no_grad():
        try:
            vgg_raw = vggish.forward(sig, fs=sr)
        except Exception:
            return None
            
    if vgg_raw.dim() == 1:
        vgg_raw = vgg_raw.unsqueeze(0)
    if vgg_raw.numel() == 0:
        return None
        
    vgg_feats = vgg_raw.cpu().numpy().astype(np.float32) # (num_windows, 128)
    num_windows = len(vgg_feats)
    
    # Calculate MFCC and RMS for each 0.96s window (which matches VGGish framing)
    window_length = int(0.96 * sr)
    hybrid_feats = []
    
    for i in range(num_windows):
        start = int(i * window_length)
        end = min(start + window_length, len(sig))
        window_sig = sig[start:end]
        
        # 2. Energy Gating preparation (calculate RMS)
        rms = np.sqrt(np.mean(window_sig**2)) if len(window_sig) > 0 else 0.0
        
        # Hybrid MFCC Feature
        if len(window_sig) >= 2048:
            mfccs = librosa.feature.mfcc(y=window_sig, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512)
            mfcc_feat = np.mean(mfccs, axis=1) # (40,)
        else:
            mfcc_feat = np.zeros(40, dtype=np.float32)
            
        # Combine VGGish (128) and MFCC (40) -> (168)
        combined = np.concatenate([vgg_feats[i], mfcc_feat])
        hybrid_feats.append((combined, rms))
        
    return hybrid_feats

def main():
    print("Loading VGGish...")
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish.eval()

    print("Processing AI Mechanic files with Bandpass Filtering, MFCC Hybrid, and Energy Gating...")
    files = []
    labels = []
    for f in sorted((RAW_DIR / "ai_mechanic").rglob("*.wav")):
        lbl = ai_mechanic_label_fn(f)
        if lbl != -1:
            files.append(f)
            labels.append(lbl)
            
    all_hybrid_feats = []
    for f in files:
        feats = extract_hybrid_features(f, ai_mechanic_reader, vggish)
        all_hybrid_feats.append(feats)
        
    # Energy Gating Threshold
    # Filter out silent pauses/background noise windows across the dataset
    all_rms = []
    for feats in all_hybrid_feats:
        if feats is not None:
            all_rms.extend([r for _, r in feats])
            
    if not all_rms:
        print("No features extracted.")
        return
        
    max_rms = np.max(all_rms)
    rms_threshold = max_rms * 0.05 # Discard anything less than 5% of peak volume
    
    window_features = []
    valid_labels = []
    valid_files = []
    
    for i, feats in enumerate(all_hybrid_feats):
        if feats is None:
            continue
        # Drop quiet windows!
        valid_windows = [f for f, r in feats if r >= rms_threshold]
        
        if len(valid_windows) == 0:
            # if all windows are quiet, just take the loudest one to avoid crashing
            loudest = max(feats, key=lambda x: x[1])
            valid_windows = [loudest[0]]
            
        window_features.append(np.vstack(valid_windows))
        valid_labels.append(labels[i])
        valid_files.append(files[i])
        
    # --- SPLIT AND EVALUATE ---
    y = np.array(valid_labels)
    idx_normal = np.where(y == 0)[0]
    idx_faulty = np.where(y == 1)[0]
    cal_idx, test_normal_idx = train_test_split(idx_normal, test_size=0.3, random_state=42)
    
    test_idx = np.concatenate([test_normal_idx, idx_faulty])
    test_y = np.concatenate([np.zeros(len(test_normal_idx)), np.ones(len(idx_faulty))])
    
    print(f"\n[AI Mechanic] cal_normal={len(cal_idx)} test_normal={len(test_normal_idx)} test_faulty={len(idx_faulty)}")
    
    cal_windows = []
    for i in cal_idx:
        cal_windows.append(window_features[i])
    memory_bank = np.vstack(cal_windows)
    
    scaler = StandardScaler().fit(memory_bank)
    memory_bank_scaled = scaler.transform(memory_bank)
    
    k = min(5, len(memory_bank_scaled))
    nn = NearestNeighbors(n_neighbors=k).fit(memory_bank_scaled)
    
    scores_mean = []
    scores_p90 = []
    scores_max = []
    
    for i in test_idx:
        fw = window_features[i]
        fw_scaled = scaler.transform(fw)
        dists, _ = nn.kneighbors(fw_scaled)
        window_anomaly_scores = dists.mean(axis=1)
        
        scores_mean.append(np.mean(window_anomaly_scores))
        scores_p90.append(np.percentile(window_anomaly_scores, 90))
        scores_max.append(np.max(window_anomaly_scores))
        
    print(f"    AUC (Mean pooling - old way) : {roc_auc_score(test_y, scores_mean):.3f}")
    print(f"    AUC (90th percentile)        : {roc_auc_score(test_y, scores_p90):.3f}")
    print(f"    AUC (Max pooling - PatchCore): {roc_auc_score(test_y, scores_max):.3f}")

if __name__ == "__main__":
    main()
