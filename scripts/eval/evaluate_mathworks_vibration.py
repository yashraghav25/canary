import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import os
import glob
import torch
import numpy as np
import scipy.io as sio
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from evaluate_anomaly_only import load_encoder, signal_to_spectrogram_batch, SpectrogramConfig
from tqdm import tqdm

# Was hardcoded "../data/..." (only correct if invoked from inside
# scripts/eval/ itself). Fixed to resolve relative to this file.
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "checkpoints"


def main():
    print("Evaluating on MathWorks Bearing Fault Dataset...")
    data_dir = RAW_DIR / "mathworks_data" / "RollingElementBearingFaultDiagnosis-Data-master"

    normal_files = list(data_dir.rglob("baseline_*.mat"))
    anom_files = list(data_dir.rglob("InnerRaceFault_*.mat")) + list(data_dir.rglob("OuterRaceFault_*.mat"))

    print(f"Found {len(normal_files)} normal and {len(anom_files)} anomalous files.")

    cfg = SpectrogramConfig(sample_rate=25600)
    try:
        # Was "vibration_dann_lambda0.15.pt" — that checkpoint has never existed
        # (0.15 was only ever the AUDIO domain's lambda; vibration was trained
        # at 0.3), so this fallback was silently triggering on every single run.
        enc = load_encoder(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
        print("Loaded vibration_dann_lambda0.3.pt")
    except Exception as e:
        print(f"Could not load vibration encoder ({e}) — falling back to audio encoder")
        enc = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    enc.eval()
    
    def extract_features_from_mats(mat_files):
        feats = []
        labels = []
        for f in tqdm(mat_files, desc="Extracting Features"):
            data = sio.loadmat(str(f))
            b = data['bearing'][0,0]
            signal = b['gs'].flatten()
            sr = int(b['sr'][0,0])
            
            chunk_size = sr * 2 # 2 seconds
            num_chunks = len(signal) // chunk_size
            
            for i in range(num_chunks):
                chunk = signal[i*chunk_size:(i+1)*chunk_size]
                specs = signal_to_spectrogram_batch(chunk, sr, cfg)
                if len(specs) == 0: continue
                
                batch = torch.from_numpy(specs).float().unsqueeze(1)
                with torch.no_grad():
                    emb = enc(batch).numpy() # Shape: (windows, features)
                feats.append(emb) # Append the whole array of patches
        return feats
        
    print("Extracting Normal Features (Memory Bank)...")
    train_normal_chunks = extract_features_from_mats(normal_files[:1]) # Use 1st baseline file for training
    test_normal_chunks = extract_features_from_mats(normal_files[1:])  # Use remaining for test
    
    print("Extracting Anomalous Features...")
    test_anom_chunks = extract_features_from_mats(anom_files)
    
    # Flatten train patches for PCA
    train_patches = np.vstack(train_normal_chunks)
    
    pca = PCA(n_components=min(16, train_patches.shape[1], train_patches.shape[0]))
    pca.fit(train_patches)
    
    def get_chunk_scores(pca_model, chunk_list):
        scores = []
        for chunk in chunk_list:
            if len(chunk) == 0: continue
            reconstructed = pca_model.inverse_transform(pca_model.transform(chunk))
            patch_errors = np.mean((chunk - reconstructed)**2, axis=1)
            # PatchCore: take max error across the patches in this chunk
            scores.append(np.max(patch_errors))
        return scores
        
    norm_scores = get_chunk_scores(pca, test_normal_chunks)
    anom_scores = get_chunk_scores(pca, test_anom_chunks)
    
    print("\n--- Anomaly Score Distributions ---")
    print(f"Normal Test Scores: Min={np.min(norm_scores):.4f}, Max={np.max(norm_scores):.4f}, Mean={np.mean(norm_scores):.4f}")
    print(f"Anomalous Scores  : Min={np.min(anom_scores):.4f}, Max={np.max(anom_scores):.4f}, Mean={np.mean(anom_scores):.4f}")
    print("-----------------------------------")
    
    if len(norm_scores) == 0 or len(anom_scores) == 0:
        print("Not enough test data for AUC calculation!")
        return

    y_true = np.concatenate([np.zeros(len(norm_scores)), np.ones(len(anom_scores))])
    y_scores = np.concatenate([norm_scores, anom_scores])
    
    auc = roc_auc_score(y_true, y_scores)
    print(f"\n[NEW DATASET RESULT: MathWorks Rolling Element Bearing]")
    print(f"Zero-Shot AUC Score (Vibration Encoder): {auc:.3f}")

if __name__ == '__main__':
    main()
