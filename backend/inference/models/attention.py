"""
Attention-Net Localization — Grad-CAM based spatial attention.
Uses the DenseNet121 feature maps + Grad-CAM to produce real spatial attention heatmaps.
This replaces the fake heuristic-based localization from the frontend.
"""

import time
import numpy as np
from PIL import Image
from scipy import ndimage

# Region definitions matching the frontend types.ts
BRAIN_REGIONS = {
    "FRONTAL_LEFT":   {"id": 0,  "label": "Left Frontal Lobe",    "center": [1.2, 0.8, 1.5]},
    "FRONTAL_RIGHT":  {"id": 1,  "label": "Right Frontal Lobe",   "center": [-1.2, 0.8, 1.5]},
    "TEMPORAL_LEFT":  {"id": 2,  "label": "Left Temporal Lobe",    "center": [1.8, -0.3, 0.5]},
    "TEMPORAL_RIGHT": {"id": 3,  "label": "Right Temporal Lobe",   "center": [-1.8, -0.3, 0.5]},
    "PARIETAL_LEFT":  {"id": 4,  "label": "Left Parietal Lobe",    "center": [0.8, 1.2, -0.3]},
    "PARIETAL_RIGHT": {"id": 5,  "label": "Right Parietal Lobe",   "center": [-0.8, 1.2, -0.3]},
    "OCCIPITAL":      {"id": 6,  "label": "Occipital Lobe",        "center": [0, 0.2, -1.8]},
    "CEREBELLUM":     {"id": 7,  "label": "Cerebellum",            "center": [0, -1.5, -1.2]},
    "BRAINSTEM":      {"id": 8,  "label": "Brainstem",             "center": [0, -1.8, 0]},
    "THALAMUS":       {"id": 9,  "label": "Thalamus",              "center": [0, 0, 0.2]},
}

TORSO_REGIONS = {
    "LIVER":          {"id": 10, "label": "Liver",                 "center": [0.8, 0.5, 0]},
    "SPLEEN":         {"id": 11, "label": "Spleen",                "center": [-0.8, 0.5, -0.2]},
    "KIDNEY_LEFT":    {"id": 12, "label": "Left Kidney",           "center": [-0.6, 0, -0.5]},
    "KIDNEY_RIGHT":   {"id": 13, "label": "Right Kidney",          "center": [0.6, 0, -0.5]},
    "PANCREAS":       {"id": 14, "label": "Pancreas",              "center": [0, 0.2, -0.2]},
    "LOWER_ABDOMEN":  {"id": 15, "label": "Lower Abdomen",         "center": [0, -1.0, 0]},
}

SKELETON_REGIONS = {
    "TIBIA_PROXIMAL":  {"id": 20, "label": "Proximal Tibia",       "center": [0, 2.0, 0]},
    "TIBIA_DIAPHYSIS": {"id": 21, "label": "Tibial Diaphysis",     "center": [0, 0, 0]},
    "TIBIA_DISTAL":    {"id": 22, "label": "Distal Tibia",         "center": [0, -2.0, 0]},
    "FIBULA":          {"id": 23, "label": "Fibula",               "center": [0.5, 0, 0]},
}


def get_region_dict(modality: str) -> dict:
    if modality == "CT":
        return TORSO_REGIONS
    elif modality == "XRAY":
        return SKELETON_REGIONS
    return BRAIN_REGIONS


def find_closest_region(position: list, region_dict: dict) -> str:
    """Find the anatomical region closest to a 3D position."""
    closest_key = list(region_dict.keys())[0]
    min_dist = float("inf")

    for key, data in region_dict.items():
        center = data["center"]
        dist = sum((a - b) ** 2 for a, b in zip(position, center)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_key = key

    return closest_key


def run_attention_inference(gradcam_heatmap: np.ndarray, modality: str = "MRI", densenet_confidence: float = 50.0) -> dict:
    """
    Generate spatial attention output from a real Grad-CAM heatmap.
    
    Args:
        gradcam_heatmap: 2D numpy array from DenseNet's Grad-CAM (H x W, values 0-1).
        modality: 'MRI', 'CT', or 'XRAY'.
        densenet_confidence: The DenseNet classification confidence to calibrate attention confidence.
    
    Returns:
        Dict matching the frontend AttentionNetOutput interface.
    """
    start_time = time.time()

    # Resize heatmap to 14x14 for the attention map output
    from PIL import Image as PILImage
    heatmap_pil = PILImage.fromarray((gradcam_heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_pil.resize((14, 14), PILImage.BILINEAR)) / 255.0

    attention_map = heatmap_resized.tolist()

    # Find the peak activation (center of mass of the highest activation region)
    # Threshold the heatmap to find the hottest region
    threshold = max(0.5, heatmap_resized.max() * 0.7)
    binary_mask = heatmap_resized > threshold

    if binary_mask.sum() > 0:
        center_of_mass = ndimage.center_of_mass(heatmap_resized * binary_mask)
        peak_y, peak_x = center_of_mass
    else:
        # If nothing passes threshold, find simple argmax
        peak_idx = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
        peak_y, peak_x = float(peak_idx[0]), float(peak_idx[1])

    # Map from [0, 14] grid coordinates to [-3, 3] 3D space
    mapped_x = (peak_x / 14.0) * 6.0 - 3.0
    mapped_y = -((peak_y / 14.0) * 6.0 - 3.0)
    mapped_z = 0.0  # 2D image → 0 depth

    focus_center = [mapped_x, mapped_y, mapped_z]

    # Find closest anatomical region
    region_dict = get_region_dict(modality)
    focus_region = find_closest_region(focus_center, region_dict)

    # Confidence is derived from the peak activation intensity and DenseNet confidence
    peak_activation = float(heatmap_resized.max())
    confidence = min(95.0, densenet_confidence * 0.6 + peak_activation * 40.0)

    # Focus radius: larger activation area → larger radius
    activated_fraction = float(binary_mask.sum()) / binary_mask.size
    focus_radius = 0.3 + activated_fraction * 1.5

    inference_time_ms = (time.time() - start_time) * 1000

    return {
        "modelName": "Attention-Net",
        "attentionMap": attention_map,
        "focusRegion": focus_region,
        "focusCenter": focus_center,
        "focusRadius": focus_radius,
        "confidence": confidence,
        "inferenceTimeMs": inference_time_ms,
    }
