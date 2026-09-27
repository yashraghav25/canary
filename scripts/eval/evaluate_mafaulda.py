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

# Was a hardcoded "../data/raw/mafaulda" — only correct if invoked from inside
# scripts/eval/ itself, which contradicts this repo's own documented
# `cd vehicle-health-ai && python scripts/eval/...` convention. Fixed to
# resolve relative to this file, regardless of the caller's cwd.
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "checkpoints"


def main():
    from data.readers import mafaulda_reader_vibration, mafaulda_label_fn_binary
    print("Evaluating on MaFaulDa Dataset (calibrated anomaly detection, vibration encoder)...")
    data_dir = RAW_DIR / "mafaulda"
    
    # MaFaulDa structure: normal/*.csv, imbalance/*.csv, etc.
    normal_files = list(data_dir.rglob("normal/**/*.csv"))
    anom_files = []
    for cls in ["imbalance", "horizontal-misalignment", "vertical-misalignment", "overhang", "underhang"]:
        anom_files.extend(list(data_dir.rglob(f"{cls}/**/*.csv")))
    
    # It's a huge dataset, let's take a subset for time (but still large)
    # We will use the first 5 normal files for Memory Bank training
    # And 44 normal files for testing, 200 anomalous files for testing.
    np.random.seed(42)
    np.random.shuffle(normal_files)
    np.random.shuffle(anom_files)
    
    train_normal = normal_files[:5]
    test_normal = normal_files[5:5+44]
    test_anom = anom_files[:200]
    
    print(f"Using {len(train_normal)} normal for train, {len(test_normal)} normal for test, {len(test_anom)} anom for test.")
    
    cfg = SpectrogramConfig(sample_rate=25600)
    try:
        enc = load_encoder(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
        print("Loaded vibration_dann_lambda0.3.pt")
    except Exception as e:
        print(f"Could not load vibration encoder ({e}) — falling back to audio encoder")
        enc = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    enc.eval()
    
    def extract_features(file_list):
        feats = []
        for f in tqdm(file_list, desc="Extracting"):
            try:
                sig, sr = mafaulda_reader_vibration(f)
                
                chunk_size = sr * 2 # 2 seconds
                num_chunks = len(sig) // chunk_size
                if num_chunks == 0: continue
                
                for i in range(num_chunks):
                    chunk = sig[i*chunk_size:(i+1)*chunk_size]
                    specs = signal_to_spectrogram_batch(chunk, sr, cfg)
                    if len(specs) == 0: continue
                    
                    batch = torch.from_numpy(specs).float().unsqueeze(1)
                    with torch.no_grad():
                        emb = enc(batch).numpy()
                    feats.append(emb)
            except Exception as e:
                print(f"Error on {f}: {e}")
        return feats
        
    print("Extracting Normal Features (Memory Bank)...")
    train_normal_chunks = extract_features(train_normal)
    test_normal_chunks = extract_features(test_normal)
    
    print("Extracting Anomalous Features...")
    test_anom_chunks = extract_features(test_anom)
    
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
    print(f"\n[RESULT: MaFaulDa (calibrated on {len(train_normal)} real normal files)]")
    print(f"AUC Score (Vibration Encoder): {auc:.3f}")

if __name__ == '__main__':
    main()
