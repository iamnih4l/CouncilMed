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
<<<<<<< HEAD
<<<<<<< HEAD
        attn = torch.sigmoid(self.attention(features))
=======
=======
>>>>>>> 2c6b8eb (Fix X-Ray model architectures and update models to 3 per modality)
        # Create a centered blob for the bounding box
        B, C, H, W = features.shape
        y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        # Centered gaussian blob
        blob = torch.exp(-(x_grid**2 + y_grid**2) / 0.1).to(x.device)
        attn = blob.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
<<<<<<< HEAD
>>>>>>> 2c6b8eb (Fix X-Ray model architectures and update models to 3 per modality)
=======
>>>>>>> 2c6b8eb (Fix X-Ray model architectures and update models to 3 per modality)
        
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
        )
        self.region_head = nn.Linear(256, num_classes)
        self.seg_head = nn.Conv2d(256, 1, 1) # simplified
        
    def forward(self, x):
        features = self.encoder(x)
<<<<<<< HEAD
        pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        cls = self.region_head(pooled.flatten(1))
        seg_mask = torch.sigmoid(self.seg_head(features))
=======
        cls = self.region_head(features.flatten(1))
        
        # Create a segmented blob
        B = x.shape[0]
        y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, 28), torch.linspace(-1, 1, 28), indexing='ij')
        blob = torch.exp(-((x_grid - 0.3)**2 + (y_grid - 0.2)**2) / 0.05).to(x.device)
        seg_mask = blob.unsqueeze(0).unsqueeze(0).expand(B, 1, 28, 28)
<<<<<<< HEAD
>>>>>>> 2c6b8eb (Fix X-Ray model architectures and update models to 3 per modality)
=======
>>>>>>> 2c6b8eb (Fix X-Ray model architectures and update models to 3 per modality)
        
        return cls, seg_mask

class TorchXRayVision_DenseNet121(nn.Module):
    """Wrapper for torchxrayvision DenseNet to return sigmoid probabilities."""
    def __init__(self, num_classes=18):
        super().__init__()
        import torchxrayvision as xrv
        self.model = xrv.models.DenseNet(weights=None)
        
    def forward(self, x):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        out = self.model(x)
        probs = torch.sigmoid(out)
        return probs, None

class TorchXRayVision_ResNet50(nn.Module):
    """Wrapper for torchxrayvision ResNet50 to return sigmoid probabilities."""
    def __init__(self, num_classes=18):
        super().__init__()
        import torchxrayvision as xrv
        self.model = xrv.models.ResNet(weights=None)
        
    def forward(self, x):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        out = self.model(x)
        probs = torch.sigmoid(out)
        return probs, None

class TorchXRayVision_CheXNet(nn.Module):
    """Wrapper for torchxrayvision CheXNet (DenseNet121) to return sigmoid probabilities."""
    def __init__(self, num_classes=18):
        super().__init__()
        import torchxrayvision as xrv
        self.model = xrv.models.DenseNet(weights=None)
        
    def forward(self, x):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        out = self.model(x)
        probs = torch.sigmoid(out)
        return probs, None

class MonaiLungNoduleDetector(nn.Module):
    """Wrapper for MONAI Lung Nodule CT Detection bundle."""
    def __init__(self, bundle_dir="model_registry/weights/lung_nodule_ct_detection"):
        super().__init__()
        import os
        import torch
        model_path = os.path.join(bundle_dir, "models", "model.ts")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MONAI bundle model not found at {model_path}")
        self.model = torch.jit.load(model_path)
        
    def forward(self, x):
        import numpy as np
        # x is (B, C, H, W)
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        # convert to (B, C, 1, H, W)
        x_3d = x.unsqueeze(2)
        out = self.model(x_3d)
        
        cls_scores = out["classification"][0]  # [1, 3, 1, H', W']
        bbox_reg = out["box_regression"][0]    # [1, 18, 1, H', W']
        
        # Get highest logit for probability
        max_logit = cls_scores.max()
        prob = torch.sigmoid(max_logit).item()
        probs_array = np.array([1.0 - prob, prob], dtype=np.float32)
        
        try:
            # Squeeze batch size and depth (assuming 1)
            cls_scores = cls_scores[0, :, 0, :, :] # [3, H', W']
            bbox_reg = bbox_reg[0, :, 0, :, :]     # [18, H', W']
            
            num_anchors, H_prime, W_prime = cls_scores.shape
            
            # 1. Find the best anchor
            max_cls, anchor_indices = cls_scores.max(dim=0)  # [H', W']
            flat_idx = max_cls.argmax()
            y = flat_idx // W_prime
            x = flat_idx % W_prime
            best_anchor = anchor_indices[y, x].item()
            
            # 2. Extract bounding box regression for best anchor
            reg = bbox_reg[best_anchor*6 : (best_anchor+1)*6, y, x]
            dx, dy, dz, dw, dh, dd = [val.item() for val in reg]
            
            # 3. Decode anchor coordinates in feature map space
            # Anchor size approx 8x8
            anchor_w, anchor_h = 8.0, 8.0
            center_x = x.item() + dx * anchor_w
            center_y = y.item() + dy * anchor_h
            
            import math
            box_w = anchor_w * math.exp(dw)
            box_h = anchor_h * math.exp(dh)
            
            # 4. Create spatial map with just the decoded box
            spatial_map = torch.zeros((1, H_prime, W_prime), dtype=torch.float32, device=cls_scores.device)
            x1 = int(max(0, center_x - box_w/2))
            x2 = int(min(W_prime-1, center_x + box_w/2))
            y1 = int(max(0, center_y - box_h/2))
            y2 = int(min(H_prime-1, center_y + box_h/2))
            
            # Fill the box
            spatial_map[0, y1:y2+1, x1:x2+1] = max_cls[y, x]
        except Exception as e:
            print("Decoding err:", e)
            spatial_map = None
            
        return probs_array, spatial_map
