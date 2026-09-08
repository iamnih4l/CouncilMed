import torch
import torch.nn as nn
import torchvision.models as models

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
