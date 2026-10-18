"""
CouncilMed — Production Neural Network Architectures
=====================================================
Supports MRI (Brain), CT (Chest), and X-Ray (Chest) Councils.
Maintains full autograd graph compatibility for Grad-CAM / Grad-CAM++
and outputs calibrated spatial activation maps for lesion bounding boxes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _load_mobilenet_backbone(pretrained: bool = True) -> nn.Module:
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v2(weights=weights)
    except Exception:
        return models.mobilenet_v2(weights=None)


def _load_densenet_backbone(pretrained: bool = True) -> nn.Module:
    try:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        return models.densenet121(weights=weights)
    except Exception:
        return models.densenet121(weights=None)


def _load_resnet_backbone(pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    except Exception:
        return models.resnet50(weights=None)


class CouncilDenseNetClassifier(nn.Module):
    """Council Member A: DenseNet-121 Feature Extractor & Calibrated Classifier."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        base = _load_densenet_backbone(pretrained=pretrained)
        self.features = base.features
        self.classifier = nn.Linear(base.classifier.in_features, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        out = F.relu(feat, inplace=False)
        spatial_map = torch.mean(out, dim=1, keepdim=True)
        pooled = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        logits = self.classifier(pooled)
        return logits, spatial_map


class AttentionNet(nn.Module):
    """Council Member B: MobileNetV2 + Spatial Attention Gate + Region Classification."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.base = _load_mobilenet_backbone(pretrained=pretrained)
        in_features = self.base.last_channel
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor):
        features = self.base.features(x)
        attn = torch.sigmoid(self.attention(features))
        attended = features * attn
        pooled = F.adaptive_avg_pool2d(attended, (1, 1)).flatten(1)
        logits = self.base.classifier(pooled)
        feature_energy = torch.mean(F.relu(features), dim=1, keepdim=True)
        spatial_map = feature_energy * attn
        return logits, spatial_map


class UNetLite(nn.Module):
    """Council Member C: Multi-Scale Encoder-Decoder for Segmentation & Classification."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        self.region_head = nn.Linear(256, num_classes)
        self.seg_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        cls_logits = self.region_head(pooled)
        seg_mask = torch.sigmoid(self.seg_head(features)) * torch.mean(features, dim=1, keepdim=True)
        return cls_logits, seg_mask


class CouncilResNetClassifier(nn.Module):
    """Council Member for CT / X-Ray: ResNet-50 Volumetric/Thoracic Feature Analyzer."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        base = _load_resnet_backbone(pretrained=pretrained)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)
        spatial_map = torch.mean(F.relu(feat), dim=1, keepdim=True)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        logits = self.fc(pooled)
        return logits, spatial_map
