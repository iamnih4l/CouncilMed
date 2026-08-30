import os
import sys
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

# --- Dummy Architectures (Mimicking the real ones for testing) ---

class AttentionNet(nn.Module):
    """Council Member B: MobileNetV2 + Spatial Attention + Region Classification"""
    def __init__(self, num_classes=10):
        super().__init__()
        # Simplified for dummy generation
        self.base = models.mobilenet_v2(weights=None)
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(self.base.last_channel, num_classes),
        )
        # Dummy spatial attention head
        self.attention = nn.Conv2d(1280, 1, kernel_size=1)
        
    def forward(self, x):
        features = self.base.features(x)
        attn = torch.sigmoid(self.attention(features))
        pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        logits = self.base.classifier(pooled)
        return logits, attn

class UNetLite(nn.Module):
    """Council Member C: Lightweight Encoder-Decoder"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.region_head = nn.Linear(256, num_classes)
        self.seg_head = nn.Conv2d(256, 1, 1) # simplified
        
    def forward(self, x):
        features = self.encoder(x)
        cls = self.region_head(features.flatten(1))
        return cls, None

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
    torch.save(classifier.state_dict(), cls_path)
    print(f"  -> Saved to {cls_path}")
    
    # 2. Attention Model
    print("- Generating BrainMRI Attention...")
    attention = AttentionNet(num_classes=10)
    att_path = out_dir / "brain_mri_attention_v1.pth"
    torch.save(attention.state_dict(), att_path)
    print(f"  -> Saved to {att_path}")
    
    # 3. Segmenter
    print("- Generating BrainMRI Segmenter...")
    segmenter = UNetLite(num_classes=10)
    seg_path = out_dir / "brain_mri_segmenter_v1.pth"
    torch.save(segmenter.state_dict(), seg_path)
    print(f"  -> Saved to {seg_path}")
    
    return cls_path, att_path, seg_path

if __name__ == "__main__":
    generate_weights()
    print("Done!")
