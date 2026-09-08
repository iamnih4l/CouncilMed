import os
import sys
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

# Add backend to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.inference.models.architectures import AttentionNet, UNetLite

def generate_weights():
    registry_dir = Path(__file__).parent.parent / "model_registry"
    out_dir = registry_dir / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating baseline model weights...")
    
    # 1. Classifier (DenseNet/MobileNet based on manifest)
    print("- Generating BrainMRI Classifier...")
    classifier = models.mobilenet_v2(weights=None)
    classifier.classifier[1] = nn.Linear(classifier.last_channel, 4)
    cls_path = out_dir / "brain_mri_classifier_v1.pth"
    torch.save(classifier, cls_path)
    print(f"  -> Saved to {cls_path}")
    
    # 2. Attention Model
    print("- Generating BrainMRI Attention...")
    attention = AttentionNet(num_classes=10)
    att_path = out_dir / "brain_mri_attention_v1.pth"
    torch.save(attention, att_path)
    print(f"  -> Saved to {att_path}")
    
    # 3. Segmenter
    print("- Generating BrainMRI Segmenter...")
    segmenter = UNetLite(num_classes=10)
    seg_path = out_dir / "brain_mri_segmenter_v1.pth"
    torch.save(segmenter, seg_path)
    print(f"  -> Saved to {seg_path}")
    
    return cls_path, att_path, seg_path

if __name__ == "__main__":
    generate_weights()
    print("Done!")
