"""
CouncilMed — Media Generation & Serving
==========================================
Generates browser-displayable preview images from medical imaging formats
(DICOM, NIfTI, standard images) and converts Grad-CAM numpy arrays into
transparent PNG overlays for the viewer.

All outputs are deterministic representations of actual input data.
No fake previews or fabricated heatmaps.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Safe filename pattern: only alphanumeric, hyphens, and underscores
_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_safe_id(value: str, label: str = "id") -> str:
    """Validate that an ID is safe for filesystem use (no path traversal)."""
    if not value or not _SAFE_ID_PATTERN.match(value):
        raise ValueError(f"Invalid {label}: '{value}' — must be alphanumeric/hyphens/underscores only")
    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {label}: '{value}' — path traversal detected")
    return value


def generate_preview_png(
    pixel_data: np.ndarray,
    output_path: Path,
    max_dimension: int = 1024,
) -> Path:
    """
    Generate a browser-displayable preview PNG from parsed pixel data.

    Handles:
    - Grayscale (H, W) float32 normalized 0-1
    - RGB (H, W, 3) float32 normalized 0-1
    - Multi-channel data (takes first 3 channels or converts grayscale)

    Args:
        pixel_data: Numpy array from the DICOM/NIfTI/standard image parser
        output_path: Where to save the PNG
        max_dimension: Maximum width or height (for reasonable file sizes)

    Returns:
        Path to the saved PNG file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to 0-255 uint8
    img_data = _normalize_to_uint8(pixel_data)

    # Create PIL image
    if img_data.ndim == 2:
        pil_img = Image.fromarray(img_data, mode="L")
    elif img_data.ndim == 3 and img_data.shape[2] == 3:
        pil_img = Image.fromarray(img_data, mode="RGB")
    elif img_data.ndim == 3 and img_data.shape[2] == 4:
        pil_img = Image.fromarray(img_data, mode="RGBA")
    elif img_data.ndim == 3 and img_data.shape[2] == 1:
        pil_img = Image.fromarray(img_data.squeeze(axis=2), mode="L")
    else:
        raise ValueError(
            f"Unsupported pixel data shape for preview: {pixel_data.shape}"
        )

    # Resize if too large (preserving aspect ratio)
    w, h = pil_img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

    pil_img.save(str(output_path), format="PNG", optimize=True)
    logger.info(
        "Preview generated: %s (%dx%d)",
        output_path.name,
        pil_img.size[0],
        pil_img.size[1],
    )
    return output_path


def generate_gradcam_overlay_png(
    gradcam_npy_path: Path,
    output_path: Path,
    target_size: tuple[int, int] | None = None,
    colormap: str = "jet",
) -> Path:
    """
    Convert a saved Grad-CAM .npy heatmap into a semi-transparent PNG overlay.

    The result is an RGBA PNG where:
    - RGB channels carry the colormap visualization
    - Alpha channel is proportional to activation intensity
    - Low-activation regions are nearly transparent
    - High-activation regions are semi-opaque

    This allows the viewer to overlay it on the base scan without obscuring it.

    Args:
        gradcam_npy_path: Path to the saved .npy Grad-CAM array
        output_path: Where to save the PNG overlay
        target_size: Optional (H, W) to resize to match the preview
        colormap: Matplotlib colormap name

    Returns:
        Path to the saved overlay PNG
    """
    import matplotlib.cm as cm

    if not gradcam_npy_path.exists():
        raise FileNotFoundError(f"Grad-CAM array not found: {gradcam_npy_path}")

    heatmap = np.load(str(gradcam_npy_path))

    # Ensure 2D
    if heatmap.ndim > 2:
        heatmap = heatmap.squeeze()
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D Grad-CAM heatmap, got shape {heatmap.shape}")

    # Resize if target size specified
    if target_size is not None:
        pil_heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        pil_heatmap = pil_heatmap.resize(
            (target_size[1], target_size[0]),
            Image.BILINEAR,
        )
        heatmap = np.array(pil_heatmap, dtype=np.float32) / 255.0

    # Apply colormap
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(colormap) if hasattr(matplotlib.colormaps, 'get_cmap') else matplotlib.colormaps[colormap]
    colored = cmap(heatmap)  # (H, W, 4) float64 with RGBA

    # Create RGBA with intensity-proportional alpha
    # Low-activation areas become transparent, high areas semi-opaque
    rgba = (colored[:, :, :3] * 255).astype(np.uint8)

    # Alpha: scale by heatmap intensity for meaningful transparency
    # Apply a power curve to make low-activation areas more transparent
    alpha = (heatmap ** 1.2 * 200).clip(0, 200).astype(np.uint8)

    # Stack into RGBA
    rgba_full = np.dstack([rgba, alpha])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_overlay = Image.fromarray(rgba_full, mode="RGBA")
    pil_overlay.save(str(output_path), format="PNG")

    logger.info(
        "Grad-CAM overlay PNG generated: %s (%dx%d)",
        output_path.name,
        pil_overlay.size[0],
        pil_overlay.size[1],
    )
    return output_path


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize arbitrary numeric data to uint8 [0, 255]."""
    if data.dtype == np.uint8:
        return data

    data = data.astype(np.float64)
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin) * 255.0
    else:
        data = np.zeros_like(data)

    return data.clip(0, 255).astype(np.uint8)


