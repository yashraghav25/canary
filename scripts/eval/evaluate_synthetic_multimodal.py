"""
IMPORTANT CAVEAT, added after review: despite the name, this does NOT test
genuine multimodal fusion. Car Diagnostics (audio) and Engine Journal
Bearings (vibration) are two unrelated datasets with no physical
correspondence between individual files — they are paired here purely by
list index (the Nth audio file with the Nth vibration file), not because
they describe the same real event. Any "fused" AUC below reflects whatever
statistical artifact comes from concatenating two arbitrarily-matched
feature vectors, not a real cross-modal signal. Treat this as an
illustrative feature-concatenation experiment, not evidence about
multimodal fusion capability — see the real, physically-paired fusion
result (MaFaulDa's genuine synchronized audio+vibration) in
results-study.md Section 5.5 instead.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import os
import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

from models.encoder import SpectrogramEncoder as AudioEncoder, SpectrogramEncoder as VibrationEncoder
from data.preprocess import SpectrogramConfig
from data.readers import car_diagnostics_reader, engine_journal_bearings_reader
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from evaluate_anomaly_only import load_encoder

def main():
    print("Building ARBITRARILY-PAIRED (not physically co-occurring) multimodal dataset...")
    print("Pairing Car Diagnostics (Unseen Audio) with Engine Journal Bearings (Unseen Vibration) by list index only")
    
    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    
    from evaluate_anomaly_only import car_diagnostics_label_fn, engine_journal_bearings_label_fn, build_feature_blocks, car_diagnostics_reader, engine_journal_bearings_reader
    
    # 1. Load Audio
    print("Loading Audio...")
    aud_cfg = SpectrogramConfig(sample_rate=16000)
    aud_enc = load_encoder(Path(__file__).resolve().parents[2] / "data/processed/checkpoints/audio_dann_lambda0.15.pt")
    
    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    if not cd_root.exists():
        cd_root = RAW_DIR / "car_diagnostics"
    aud_files = sorted(list(cd_root.rglob("*.wav")))
    aud_labels = [car_diagnostics_label_fn(f) for f in aud_files]
    aud_train = aud_files[:50]
    aud_test = aud_files[50:150]
    aud_labels = aud_labels[50:150]
    
    # 2. Load Vibration
    print("Loading Vibration...")
    vib_cfg = SpectrogramConfig(sample_rate=25600)
    # Was "vibration_dann_lambda0.15.pt" — never existed (0.15 was only ever
    # the AUDIO domain's lambda; vibration was trained at 0.3), so this
    # silently fell back to the audio encoder on every run.
    ckpt_path = Path(__file__).resolve().parents[2] / "data/processed/checkpoints/vibration_dann_lambda0.3.pt"
    if ckpt_path.exists():
        vib_enc = load_encoder(ckpt_path)
    else:
        print(f"[warn] {ckpt_path.name} not found — falling back to audio encoder for the 'vibration' role")
        vib_enc = load_encoder(Path(__file__).resolve().parents[2] / "data/processed/checkpoints/audio_dann_lambda0.15.pt")
    
    ejb_root = RAW_DIR / "engine_journal_bearings"
    vib_files = sorted(list(ejb_root.rglob("*.csv")))
    vib_labels_full = [engine_journal_bearings_label_fn(f) for f in vib_files]
    vib_train = vib_files[:20]
    vib_test = vib_files[20:]
    vib_labels = vib_labels_full[20:]
    
    print(f"Audio: {len(aud_train)} normal train, {len(aud_test)} test")
    print(f"Vib:   {len(vib_train)} normal train, {len(vib_test)} test")
    
    # 3. Create synthetic pairs
    # Pair normal with normal for training
    min_train = min(len(aud_train), len(vib_train))
    syn_train_aud = aud_train[:min_train]
    syn_train_vib = vib_train[:min_train]
    
    # Separate testing by labels
    aud_test_norm = [x for x, y in zip(aud_test, aud_labels) if y == 0]
    aud_test_anom = [x for x, y in zip(aud_test, aud_labels) if y == 1]
    
    vib_test_norm = [x for x, y in zip(vib_test, vib_labels) if y == 0]
    vib_test_anom = [x for x, y in zip(vib_test, vib_labels) if y == 1]
    
    min_test_norm = min(len(aud_test_norm), len(vib_test_norm))
    min_test_anom = min(len(aud_test_anom), len(vib_test_anom))
    
    syn_test_norm_aud = aud_test_norm[:min_test_norm]
    syn_test_norm_vib = vib_test_norm[:min_test_norm]
    
    syn_test_anom_aud = aud_test_anom[:min_test_anom]
    syn_test_anom_vib = vib_test_anom[:min_test_anom]
    
    syn_test_aud = syn_test_norm_aud + syn_test_anom_aud
    syn_test_vib = syn_test_norm_vib + syn_test_anom_vib
    syn_test_labels = [0]*min_test_norm + [1]*min_test_anom
    
    print(f"Synthetic Pairs: {min_train} train, {len(syn_test_labels)} test")
    
    from evaluate_anomaly_only import signal_to_spectrogram_batch
    
    def get_feats(enc, cfg, files, reader):
        feats = []
        with torch.no_grad():
            for f in tqdm(files, desc="Extracting"):
                try:
                    signal, sr = reader(f)
                    specs = signal_to_spectrogram_batch(signal, sr, cfg)
                    if len(specs) == 0: continue
                    batch = torch.from_numpy(specs).float().unsqueeze(1)
                    emb = enc(batch).mean(dim=0).numpy()
                    feats.append(emb)
                except Exception as e:
                    print(f"Skipping {f.name}: {e}")
                    # append zero array to keep shapes aligned
                    feats.append(np.zeros(128))
        return np.stack(feats)
        
    print("Extracting features...")
    tr_aud_feat = get_feats(aud_enc, aud_cfg, syn_train_aud, car_diagnostics_reader)
    te_aud_feat = get_feats(aud_enc, aud_cfg, syn_test_aud, car_diagnostics_reader)
    
    tr_vib_feat = get_feats(vib_enc, vib_cfg, syn_train_vib, engine_journal_bearings_reader)
    te_vib_feat = get_feats(vib_enc, vib_cfg, syn_test_vib, engine_journal_bearings_reader)
    
    # Fuse them
    tr_fused = np.concatenate([tr_aud_feat, tr_vib_feat], axis=1)
    te_fused = np.concatenate([te_aud_feat, te_vib_feat], axis=1)
    
    # 5. Evaluate
    print("Training PCA Reconstruction Anomaly Detector...")
    
    def score_pca(train_feat, test_feat):
        pca = PCA(n_components=min(32, train_feat.shape[1], train_feat.shape[0]))
        pca.fit(train_feat)
        reconstructed = pca.inverse_transform(pca.transform(test_feat))
        errors = np.mean((test_feat - reconstructed) ** 2, axis=1)
        return errors
        
    scores_fused = score_pca(tr_fused, te_fused)
    auc_fused = roc_auc_score(syn_test_labels, scores_fused)
    
    print(f"\n[Synthetic Multimodal Result]")
    print(f"AUC (Audio+Vibration Fused): {auc_fused:.3f}")
    
    # Compare against Audio-only and Vib-only
    scores_aud = score_pca(tr_aud_feat, te_aud_feat)
    auc_aud = roc_auc_score(syn_test_labels, scores_aud)
    
    scores_vib = score_pca(tr_vib_feat, te_vib_feat)
    auc_vib = roc_auc_score(syn_test_labels, scores_vib)
    
    print(f"AUC (Audio Only)           : {auc_aud:.3f}")
    print(f"AUC (Vibration Only)       : {auc_vib:.3f}")

if __name__ == '__main__':
    main()
