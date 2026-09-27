import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import argparse
import torch
import numpy as np
from pathlib import Path
import os

from data.preprocess import SpectrogramConfig, signal_to_spectrogram_batch
from models.encoder import load_encoder
from models.memory_bank import PCAReconstructionMemoryBank
from data.readers import car_diagnostics_reader, engine_journal_bearings_reader

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "data" / "processed" / "checkpoints"
MEMORY_BANK_DIR = REPO_ROOT / "data" / "processed" / "memory_banks"


def main():
    parser = argparse.ArgumentParser(description="Inference Script for Hackathon Demo")
    parser.add_argument("--file", type=str, required=True, help="Path to .wav or .csv file")
    parser.add_argument("--modality", type=str, choices=["audio", "vibration"], required=True)
    parser.add_argument("--notes", type=str, default="", help="Optional mechanic notes to give Jev more context")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {args.file}")
        return

    print(f"Running Inference on {args.modality.upper()} file: {file_path.name}")

    if args.modality == "audio":
        cfg = SpectrogramConfig(sample_rate=16000)
        ckpt = CHECKPOINT_DIR / "audio_dann_lambda0.15.pt"
        reader = car_diagnostics_reader
        bank_path = MEMORY_BANK_DIR / "audio_memory_bank.joblib"
    else:
        cfg = SpectrogramConfig(sample_rate=25600)
        # NOTE: vibration was always trained at grl_lambda=0.3, never 0.15
        # (0.15 was only ever the audio domain's lambda) — this file name was
        # previously wrong and pointed at a checkpoint that never existed,
        # which silently fell back to the audio encoder on every run.
        ckpt = CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt"
        reader = engine_journal_bearings_reader
        bank_path = MEMORY_BANK_DIR / "vibration_memory_bank.joblib"

    if not ckpt.exists():
        print(f"Encoder checkpoint not found: {ckpt}")
        print("Run the training pipeline first (see scripts/train/) to produce it.")
        return
    if not bank_path.exists():
        print(f"Memory bank not found: {bank_path}")
        print("Run `python scripts/utils/build_memory_bank.py` first to build it from real calibration data.")
        return

    enc = load_encoder(ckpt)
    enc.eval()
    bank = PCAReconstructionMemoryBank.load(bank_path)

    print("Extracting Features...")
    try:
        signal, sr = reader(file_path)
        specs = signal_to_spectrogram_batch(signal, sr, cfg)

        if len(specs) == 0:
            print("Signal too short for analysis.")
            return

        batch = torch.from_numpy(specs).float().unsqueeze(1)

        with torch.no_grad():
            emb = enc(batch).mean(dim=0).numpy()

        print(f"Extracted Embedding Vector of shape {emb.shape}")
    except Exception as e:
        # Was previously caught and replaced with a fabricated filename-based
        # score ("[Demo Mode] Bypassing... simulated Edge Anomaly Trigger").
        # That masked real failures (e.g. feeding this reader a file format
        # it doesn't parse) behind a plausible-looking fake result. Failing
        # honestly here instead — no fabricated number stands in for a real
        # one this project spent a lot of effort establishing honestly.
        print(f"Feature extraction failed: {e}")
        print(f"(modality={args.modality} expects the file format read by "
              f"{reader.__name__} — check the file matches that format)")
        return

    # -------------------------------------------------------------
    # STAGE 1: Edge Anomaly Detection (real PCA-reconstruction memory bank,
    # fitted on real normal-only calibration data — see
    # scripts/utils/build_memory_bank.py. Not a placeholder.)
    # -------------------------------------------------------------
    anomaly_score = float(bank.score(emb.reshape(1, -1))[0])
    threshold = bank.threshold

    print("\n" + "="*40)
    print(f"STAGE 1 (EDGE): Anomaly Score = {anomaly_score:.6f}  (threshold = {threshold:.6f})")

    if anomaly_score <= threshold:
        print("Status: NORMAL. No cloud API required. Halting.")
        print("="*40)
        return

    print("Status: ANOMALY DETECTED! Triggering Stage 2 (Cloud).")
    print("="*40 + "\n")
    
    # -------------------------------------------------------------
    # STAGE 2: Cloud Decision Engine (Jev by TypeSafe AI)
    # -------------------------------------------------------------
    print("STAGE 2 (CLOUD): Routing to TypeSafe AI Jev-Omni...")
    
    # Check for either TypeSafe or OpenRouter API keys
    api_key = os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    is_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    
    if not api_key:
        print("[Demo Mode] No API Key found (checked TYPESAFE_API_KEY and OPENROUTER_API_KEY). Mocking Jev response...")
        print("    -> Fault Type: Bearing_Failure")
        print("    -> Severity: High")
        print("    -> Recommended Action: Replace pulley bearing immediately.")
        return
        
    try:
        from typesafe_sdk import TypeSafeClient, Choice, Score
        
        # Initialize client. If using OpenRouter, point the base_url to OpenRouter's API
        if is_openrouter:
            client = TypeSafeClient(api_key=api_key, base_url="https://openrouter.ai/api/alpha")
        else:
            client = TypeSafeClient(api_key=api_key)
            
        # We pass rich context about the anomaly, vehicle, and mechanic notes
        state = (
            f"Vehicle Diagnostic Event (Automated Edge Trigger).\n"
            f"Sensor Modality: {args.modality.upper()}\n"
            f"File Analyzed: {file_path.name}\n"
            f"Edge Anomaly Score: {anomaly_score:.6g} (Threshold: {threshold:.6g}, "
            f"{anomaly_score / max(threshold, 1e-12):.1f}x over threshold). "
            f"This mathematically indicates a severe physical deviation from the healthy baseline.\n"
        )
        if args.notes:
            state += f"Mechanic Notes: '{args.notes}'\n"
            
        state += "Please analyze this telemetry and context to classify the likely fault."
        
        # Jev is a System One model. We ask structured questions:
        questions = {
            "fault_type": Choice(
                criteria={
                    "bearing_failure": "Mechanical defect in bearing races or rolling elements", 
                    "belt_squeal": "Slipping or worn serpentine belt", 
                    "imbalance": "Rotor mass imbalance", 
                    "misalignment": "Shaft misalignment", 
                    "exhaust_leak": "Hole or gap in the exhaust system", 
                    "unknown": "None of the above"
                },
                instructions="Based on the high anomaly score and mechanical context, what is the most likely fault?"
            ),
            "severity": Score(
                criteria=["low", "medium", "high", "critical"],
                instructions="How urgent is this mechanical fault?"
            ),
            "action": Choice(
                criteria={
                    "schedule_maintenance": "Fix during next scheduled service", 
                    "immediate_stop": "Turn off engine immediately to prevent catastrophic failure", 
                    "replace_part": "Swap out the defective component", 
                    "inspect_only": "Requires manual teardown and inspection"
                },
                instructions="What is the next operational action?"
            )
        }
        
        import json
        import requests
        
        # Use OpenRouter model ID if applicable
        model_name = "typesafe/jev-1.13" if is_openrouter else "jev-omni-latest"
        
        if is_openrouter:
            # Direct HTTP call for OpenRouter's alpha decisions endpoint to avoid SDK path mismatch
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/deltaparticle/Granica"
            }
            payload = {
                "model": model_name,
                "state": state,
                "questions": {
                    "fault_type": {
                        "type": "choice",
                        "criteria": {
                            "bearing_failure": "Mechanical defect in bearing races or rolling elements", 
                            "belt_squeal": "Slipping or worn serpentine belt", 
                            "imbalance": "Rotor mass imbalance", 
                            "misalignment": "Shaft misalignment", 
                            "exhaust_leak": "Hole or gap in the exhaust system", 
                            "unknown": "None of the above"
                        },
                        "instructions": "Based on the high anomaly score and mechanical context, what is the most likely fault?"
                    },
                    "severity": {
                        "type": "score",
                        "criteria": ["low", "medium", "high", "critical"],
                        "instructions": "How urgent is this mechanical fault?"
                    },
                    "action": {
                        "type": "choice",
                        "criteria": {
                            "schedule_maintenance": "Fix during next scheduled service", 
                            "immediate_stop": "Turn off engine immediately to prevent catastrophic failure", 
                            "replace_part": "Swap out the defective component", 
                            "inspect_only": "Requires manual teardown and inspection"
                        },
                        "instructions": "What is the next operational action?"
                    }
                }
            }
            resp = requests.post("https://openrouter.ai/api/alpha/decisions", json=payload, headers=headers)
            
            if resp.status_code != 200:
                print(f"API Error: {resp.status_code} - {resp.text}")
                return
                
            response = resp.json()
            print("\n--- JEV CLASSIFICATION RESULTS ---")
            try:
                answers = response.get('answers', response)
                print(f"Fault Type : {answers['fault_type']['choice']} (Confidence: {answers['fault_type']['confidence']:.2f})")
                print(f"Severity   : {answers['severity']['score']}")
                print(f"Next Action: {answers['action']['choice']}")
            except Exception:
                print(f"Raw Output: {json.dumps(response, indent=2)}")
        else:
            client = TypeSafeClient(api_key=api_key)
            response = client.system_one(
                model=model_name, 
                state=state,
                questions=questions
            )
            print("\n--- JEV CLASSIFICATION RESULTS ---")
            try:
                print(f"Fault Type : {response['fault_type'].choice} (Confidence: {response['fault_type'].confidence:.2f})")
                print(f"Severity   : {response['severity'].score}")
                print(f"Next Action: {response['action'].choice}")
            except Exception:
                print(f"Raw Output: {response}")
        
    except ImportError:
        print("Error: Please install the SDK by running: pip install typesafe-sdk")
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == '__main__':
    main()
