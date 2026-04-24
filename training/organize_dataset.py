"""Extract and organize the HuggingFace Brain Tumor dataset into train/val/test"""
import os
import sys
import shutil
import zipfile
import random
import glob
from pathlib import Path

BASE_DIR = Path(__file__).parent / "data"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"
RAW_DIR = BASE_DIR / "raw"
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
CANONICAL = ["glioma", "meningioma", "notumor", "pituitary"]

def main():
    print("=" * 60)
    print("Extracting & Organizing Brain Tumor MRI Dataset")
    print("=" * 60)
    
    # Check existing
    if TRAIN_DIR.exists():
        count = sum(1 for _ in TRAIN_DIR.rglob("*") if _.is_file())
        if count > 100:
            print(f"Already prepared: {count} training images. Skipping.")
            return
    
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Extract both zips
    for zname in ["training.zip", "testing.zip"]:
        zpath = BASE_DIR / zname
        if zpath.exists():
            print(f"Extracting {zname}...")
            with zipfile.ZipFile(zpath, 'r') as z:
                z.extractall(RAW_DIR)
            print(f"  Done. Size: {zpath.stat().st_size / 1e6:.1f}MB")
    
    # List what we got
    print("\nExtracted structure:")
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_dir():
            fc = sum(1 for f in p.iterdir() if f.is_file())
            if fc > 0:
                print(f"  {p.relative_to(RAW_DIR)}: {fc} files")
    
    # Collect all images per class
    all_images = {}
    for cls, canon in zip(CLASSES, CANONICAL):
        imgs = []
        for found in RAW_DIR.rglob("*"):
            if found.is_dir() and found.name.lower() == cls.lower():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                    imgs.extend(glob.glob(str(found / ext)))
        imgs = list(set(imgs))
        all_images[canon] = imgs
        print(f"  {canon}: {len(imgs)} images total")
    
    total = sum(len(v) for v in all_images.values())
    print(f"\nTotal: {total} images")
    
    if total < 50:
        print("ERROR: Not enough images found!")
        return
    
    # Create split dirs
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for c in CANONICAL:
            os.makedirs(split_dir / c, exist_ok=True)
    
    # Split 70/15/15
    random.seed(42)
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
    
    print("\nFinal splits:")
    for split in ["train", "val", "test"]:
        count = sum(1 for _ in (BASE_DIR / split).rglob("*") if _.is_file())
        print(f"  {split}: {count} images")
    
    print("\n[DONE] Dataset ready for training!")

if __name__ == "__main__":
    main()
