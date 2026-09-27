import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "milling_anomaly"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    api = KaggleApi()
    api.authenticate()
    
    files_to_download = [
        "imi_vm20i/dataset1/20240229_053831.227000_acc.csv",
        "imi_vm20i/dataset1/20240229_053831.227000_s0.wav",
        "imi_vm20i/dataset1/label.csv"
    ]
    
    dataset = 'manufuturetoday/multi-sensor-for-metal-milling-anomaly'
    
    print("Downloading specific files from the 14GB dataset to save time...")
    for f in files_to_download:
        print(f"Downloading {f}...")
        api.dataset_download_file(dataset, f, path=str(RAW_DIR))
        
        # Unzip if it downloaded as zip
        zip_path = RAW_DIR / (Path(f).name + '.zip')
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            os.remove(zip_path)
            
    print("Download complete!")

if __name__ == "__main__":
    download_dataset()
