import os
import sys
from pathlib import Path

def main():
    print("Importing MONAI...")
    try:
        from monai.bundle import download
    except ImportError:
        print("MONAI is not installed. Please run: pip install monai")
        sys.exit(1)
        
    out_dir = Path("model_registry/weights")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    bundle_name = "brats_mri_segmentation"
    print(f"Downloading MONAI Bundle: {bundle_name} to {out_dir}")
    
    try:
        download(
            name=bundle_name,
            bundle_dir=str(out_dir)
        )
        print(f"Successfully downloaded {bundle_name}!")
    except Exception as e:
        print(f"Failed to download bundle: {e}")
        
if __name__ == "__main__":
    main()
