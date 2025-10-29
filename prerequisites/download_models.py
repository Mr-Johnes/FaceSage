"""
download_models.py
-----------------------------------
Downloads pretrained model files (mask & age classifiers)
from Google Drive and saves them in the 'models/' directory.
-----------------------------------
Usage:
    python prerequisites/download_models.py
"""

import gdown
import os


MASK_MODEL_ID = "16Sx3b2UZ3S-xU7LJNLgEnTZ8X1wIJ52v"
AGE_MODEL_ID = "19Ym1i1_yEIGE33EJE1K-zmpXqdGEo6Iq"

SAVE_DIR = os.path.join("models")
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================
# DOWNLOAD HELPER
# =========================================
def download_from_drive(file_id, save_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[INFO] Downloading {os.path.basename(save_path)} ...")
    gdown.download(url, save_path, quiet=False)
    print(f"[INFO] Saved to {save_path}\n")

# =========================================
# MAIN
# =========================================
def main():
    print("=== Model Downloader ===")
    print("This script will download required deep learning models into the 'models/' folder.\n")

    # Download mask detector
    mask_path = os.path.join(SAVE_DIR, "mask_detector.h5")
    download_from_drive(MASK_MODEL_ID, mask_path)

    # Download age classifier
    age_path = os.path.join(SAVE_DIR, "age_classifier_v2.h5")
    download_from_drive(AGE_MODEL_ID, age_path)

    print("✅ All models downloaded successfully.")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        print("[INFO] Installing 'gdown' package...")
        os.system("pip install gdown")

    main()
