"""
NOT A FAULT-DETECTION EVALUATION — read this before citing any number this
script prints anywhere. It compares ESC-50's category 44 ("engine idling",
an ambient environmental recording, not a fault of any kind) against ESC-50
categories 0-4, which in ESC-50's actual taxonomy are dog bark, rooster,
pig, cow, and frog. This tests whether the audio encoder's embedding space
can tell an idling engine apart from farm animals — a sanity check on
whether the encoder responds to *some* structured difference between sound
classes at all, not a measurement of fault-detection capability. Any AUC
here says nothing about vehicle health screening and should never be
presented alongside the real held-out results (AI Mechanic, Car
Diagnostics, Engine Journal Bearings) as if it were comparable.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from evaluate_anomaly_only import load_encoder, signal_to_spectrogram_batch, SpectrogramConfig
from tqdm import tqdm

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "checkpoints"


def main():
    print("Loading ESC-50 dataset from HuggingFace (NOT a fault-detection test — see module docstring)...")
    ds = load_dataset('ashraq/esc50', split='train')

    # Category 41 is 'chainsaw', Category 42 is 'siren', Category 43 is 'car_horn', Category 44 is 'engine'.
    # Categories 0-4 are dog bark, rooster, pig, cow, frog (farm/animal sounds) — used here only as an
    # arbitrary "clearly different sound" class, NOT as a stand-in for "normal vehicle operation".
    
    normal_data = []
    anomalous_data = []
    
    print("Partitioning dataset into Normal and Anomalous...")
    for item in tqdm(ds, desc="Processing"):
        audio = item['audio']['array']
        sr = item['audio']['sampling_rate']
        label = item['target']
        
        # Taking a subset of normal (e.g. ambient sounds, class < 10) for speed
        if label < 5 and len(normal_data) < 50:
            normal_data.append((audio, sr))
        elif label == 44 and len(anomalous_data) < 20: # Engine
            anomalous_data.append((audio, sr))
            
        if len(normal_data) >= 50 and len(anomalous_data) >= 20:
            break

    cfg = SpectrogramConfig(sample_rate=16000)
    enc = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    enc.eval()
    
    def extract_features(data_list):
        feats = []
        for signal, sr in tqdm(data_list, desc="Extracting Features"):
            specs = signal_to_spectrogram_batch(signal, sr, cfg)
            if len(specs) == 0: continue
            batch = torch.from_numpy(specs).float().unsqueeze(1)
            with torch.no_grad():
                emb = enc(batch).mean(dim=0).numpy()
            feats.append(emb)
        return np.array(feats)
        
    print("Extracting Normal Features (Memory Bank)...")
    train_feats = extract_features(normal_data[:30])
    test_normal_feats = extract_features(normal_data[30:])
    test_anom_feats = extract_features(anomalous_data)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(10, len(train_feats)))
    pca.fit(train_feats)
    
    def get_pca_anomaly_scores(pca_model, feats):
        reconstructed = pca_model.inverse_transform(pca_model.transform(feats))
        return np.mean((feats - reconstructed)**2, axis=1)
        
    norm_scores = get_pca_anomaly_scores(pca, test_normal_feats)
    anom_scores = get_pca_anomaly_scores(pca, test_anom_feats)
    
    y_true = np.concatenate([np.zeros(len(norm_scores)), np.ones(len(anom_scores))])
    y_scores = np.concatenate([norm_scores, anom_scores])
    
    auc = roc_auc_score(y_true, y_scores)
    print(f"\n[NON-FAULT-DETECTION SANITY CHECK: ESC-50 'engine idling' vs. farm-animal sounds]")
    print(f"This is NOT a measurement of fault-detection capability — see module docstring.")
    print(f"AUC Score: {auc:.3f}")

if __name__ == '__main__':
    main()
