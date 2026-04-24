"""
CouncilMed - Direct Dataset Download from Sartaj Dataset (GitHub)
Downloads the Brain Tumor Classification dataset (4 classes, ~3000+ images)
from the Sartaj dataset which is publicly available.
"""
import os
import sys
import shutil
import zipfile
import random
import glob
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent / "data"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Sartaj Brain Tumor Classification Dataset (public, no auth needed)
# This is the well-known 4-class brain tumor MRI dataset
DATASET_URL = "https://data.mendeley.com/public-files/datasets/v5249m648g/files/7c76af76-db20-40a8-9db4-3a4b48dcfa1a/file_downloaded"

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    sys.stdout.write(f"\r  Downloading: {percent}% ({count * block_size // 1024 // 1024}MB)")
    sys.stdout.flush()

def main():
    print("=" * 60)
    print("CouncilMed - Brain Tumor MRI Dataset Download")
    print("=" * 60)
    
    # Check if already prepared
    if TRAIN_DIR.exists():
        count = sum(1 for _ in TRAIN_DIR.rglob("*") if _.is_file())
        if count > 100:
            print(f"\n[OK] Dataset already prepared with {count} training images.")
            return
    
    os.makedirs(BASE_DIR, exist_ok=True)
    zip_path = BASE_DIR / "brain_tumor_dataset.zip"
    
    if not zip_path.exists():
        print("\n[1/3] Downloading Brain Tumor MRI Dataset...")
        print("  Source: Mendeley Data (v5249m648g)")
        print("  This is a ~150MB download, please wait...\n")
        
        try:
            urllib.request.urlretrieve(DATASET_URL, str(zip_path), download_progress)
            print(f"\n\n  Download complete: {zip_path.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            print(f"\n[ERROR] Download failed: {e}")
            
            # Try alternative: Kaggle via opendatasets
            print("\n[FALLBACK] Trying opendatasets (requires Kaggle credentials)...")
            try:
                import opendatasets as od
                od.download(
                    "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
                    data_dir=str(BASE_DIR)
                )
            except Exception as e2:
                print(f"[ERROR] Fallback also failed: {e2}")
                print("\nPlease manually download from:")
                print("  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
                print(f"  and place the zip at: {zip_path}")
                return
    else:
        print(f"\n[OK] ZIP already exists: {zip_path.stat().st_size / 1e6:.1f} MB")
    
    # Extract
    print("\n[2/3] Extracting dataset...")
    raw_dir = BASE_DIR / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(raw_dir)
        print("  Extraction complete.")
    except zipfile.BadZipFile:
        print("[ERROR] Downloaded file is not a valid ZIP. It may be an HTML error page.")
        print(f"  File size: {zip_path.stat().st_size} bytes")
        if zip_path.stat().st_size < 1_000_000:
            print("  File is too small - likely a download error.")
            # Read first few bytes
            with open(zip_path, 'rb') as f:
                header = f.read(100)
                print(f"  Header: {header[:50]}")
            zip_path.unlink()
            print("  Deleted invalid file. Please try again.")
        return
    
    # Find and organize
    print("\n[3/3] Organizing into train/val/test splits...")
    
    all_images = {}
    alt_names = {
        "glioma": ["glioma", "glioma_tumor", "Glioma"],
        "meningioma": ["meningioma", "meningioma_tumor", "Meningioma"],
        "notumor": ["notumor", "no_tumor", "no tumor", "Normal", "healthy"],
        "pituitary": ["pituitary", "pituitary_tumor", "Pituitary"],
    }
    
    for class_name in CLASSES:
        imgs = []
        for name in alt_names.get(class_name, [class_name]):
            for found_dir in raw_dir.rglob("*"):
                if found_dir.is_dir() and found_dir.name.lower() == name.lower():
                    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                        imgs.extend(glob.glob(str(found_dir / ext)))
        
        imgs = list(set(imgs))
        all_images[class_name] = imgs
        print(f"  {class_name}: {len(imgs)} images")
    
    total = sum(len(v) for v in all_images.values())
    if total < 50:
        print(f"\n[WARN] Only {total} images found. Listing extracted structure:")
        for p in sorted(raw_dir.rglob("*")):
            if p.is_dir():
                fc = sum(1 for f in p.iterdir() if f.is_file())
                if fc > 0:
                    print(f"    {p.relative_to(raw_dir)} ({fc} files)")
        return
    
    # Create directories and split
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for c in CLASSES:
            os.makedirs(split_dir / c, exist_ok=True)
    
    random.seed(42)
    total_copied = 0
    
    for class_name, imgs in all_images.items():
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        for split_name, start, end in [
            ("train", 0, n_train),
            ("val", n_train, n_train + n_val),
            ("test", n_train + n_val, n),
        ]:
            dest_dir = BASE_DIR / split_name / class_name
            for img_path in imgs[start:end]:
                dest = dest_dir / Path(img_path).name
                if not dest.exists():
                    shutil.copy2(img_path, dest)
                total_copied += 1
    
    print(f"\n[OK] Dataset organized: {total_copied} total images")
    for split in ["train", "val", "test"]:
        count = sum(1 for _ in (BASE_DIR / split).rglob("*") if _.is_file())
        print(f"  {split}: {count} images")
    
    print("\n[DONE] Dataset ready for training!")


if __name__ == "__main__":
    main()
