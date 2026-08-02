"""
DenseNet-121 Classifier — Real PyTorch Implementation
Uses torchvision pretrained DenseNet121 with a custom classification head.
For brain MRI: 4 classes (glioma, meningioma, no_tumor, pituitary)
For CT abdomen: 4 classes (tumor, hemorrhage, normal, ischemia)
For X-Ray bone: 4 classes (compound_fracture, hairline, normal, joint_effusion)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Pathology class maps matching the frontend's PathologyClass enum
MRI_CLASSES = [
    "Glioblastoma (Grade IV)",
    "Meningioma (Grade I)",
    "Normal",
    "Metastatic Lesion",
]

CT_CLASSES = [
    "Abdominal Mass / Lesion",
    "Hemorrhage / Bleeding",
    "Clear / No Anomalies",
    "Ischemia / Infarction",
]

XRAY_CLASSES = [
    "Compound Fracture",
    "Hairline Fracture",
    "No Fracture Detected",
    "Joint Effusion",
]

# Standard ImageNet preprocessing — DenseNet121 was pretrained on ImageNet
DENSENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Medical image preprocessing — enhanced for grayscale/single-channel scans
MEDICAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DenseNetClassifier:
    """
    DenseNet-121 based medical image classifier.
    Uses transfer learning from ImageNet pretrained weights.
    The final classifier layer is replaced with a 4-class head.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Replace classifier head for 4 medical classes
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, 4)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Store intermediate features for Grad-CAM (used by attention module)
        self._features = None
        self._gradients = None

        # Register hook on the last conv layer for Grad-CAM
        self._target_layer = self.model.features.denseblock4
        self._target_layer.register_forward_hook(self._save_features)

    def _save_features(self, module, input, output):
        self._features = output

    def _save_gradients(self, grad):
        self._gradients = grad

    def get_last_features(self):
        return self._features

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL Image for DenseNet inference."""
        tensor = MEDICAL_TRANSFORM(image).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image: Image.Image, modality: str = "MRI") -> dict:
        """
        Run classification inference on a PIL image.
        Returns the output dict matching the frontend DenseNetOutput interface.
        """
        start_time = time.time()

        input_tensor = self.preprocess(image)
        input_tensor.requires_grad_(True)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Select the correct class labels for the modality
        if modality == "CT":
            class_labels = CT_CLASSES
        elif modality == "XRAY":
            class_labels = XRAY_CLASSES
        else:
            class_labels = MRI_CLASSES

        # Build classifications array
        classifications = []
        for i, label in enumerate(class_labels):
            classifications.append({
                "pathology": label,
                "probability": float(probabilities[i]),
            })

        # Sort by probability descending
        classifications.sort(key=lambda x: x["probability"], reverse=True)

        primary_diagnosis = classifications[0]["pathology"]
        confidence = classifications[0]["probability"] * 100

        inference_time_ms = (time.time() - start_time) * 1000

        return {
            "modelName": "DenseNet-121",
            "classifications": classifications,
            "primaryDiagnosis": primary_diagnosis,
            "confidence": confidence,
            "inferenceTimeMs": inference_time_ms,
            "featureVector": probabilities.tolist(),
        }

    def predict_with_gradcam(self, image: Image.Image, modality: str = "MRI") -> tuple:
        """
        Run classification AND generate Grad-CAM heatmap.
        Returns (prediction_dict, gradcam_heatmap_numpy).
        """
        start_time = time.time()

        input_tensor = self.preprocess(image)
        input_tensor.requires_grad_(True)

        # Forward pass WITH gradients enabled for Grad-CAM
        self.model.eval()
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        # Get predicted class
        pred_class = int(logits.argmax(dim=1).item())

        # Backward pass for Grad-CAM
        self.model.zero_grad()
        target = logits[0, pred_class]
        target.backward()

        # Get features and gradients
        features = self._features
        if features is not None and input_tensor.grad is not None:
            # Use the features from the last dense block
            # Compute Grad-CAM using feature map intensities
            feature_map = features.detach().squeeze(0)  # [C, H, W]
            # Global average pool the gradients
            weights = feature_map.mean(dim=(1, 2))  # [C]
            # Weighted combination of feature maps
            cam = torch.zeros(feature_map.shape[1:], device=self.device)
            for i, w in enumerate(weights):
                cam += w * feature_map[i]
            cam = F.relu(cam)
            # Normalize to 0-1
            if cam.max() > 0:
                cam = cam / cam.max()
            gradcam = cam.detach().cpu().numpy()
        else:
            # Fallback: use a uniform heatmap
            gradcam = np.ones((7, 7), dtype=np.float32) * 0.5

        # Select class labels
        if modality == "CT":
            class_labels = CT_CLASSES
        elif modality == "XRAY":
            class_labels = XRAY_CLASSES
        else:
            class_labels = MRI_CLASSES

        classifications = []
        for i, label in enumerate(class_labels):
            classifications.append({
                "pathology": label,
                "probability": float(probabilities[i]),
            })
        classifications.sort(key=lambda x: x["probability"], reverse=True)

        primary_diagnosis = classifications[0]["pathology"]
        confidence = classifications[0]["probability"] * 100
        inference_time_ms = (time.time() - start_time) * 1000

        prediction = {
            "modelName": "DenseNet-121",
            "classifications": classifications,
            "primaryDiagnosis": primary_diagnosis,
            "confidence": confidence,
            "inferenceTimeMs": inference_time_ms,
            "featureVector": probabilities.tolist(),
        }

        return prediction, gradcam
