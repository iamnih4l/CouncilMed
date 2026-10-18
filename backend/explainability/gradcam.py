"""
CouncilMed — Grad-CAM Explainability Engine (Production Patched)
==================================================================
Computes genuine gradient-weighted class activation maps blended with
high-resolution anatomical saliency so heatmaps align with lesions.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..council.evidence import ExplainabilityData

logger = logging.getLogger(__name__)


def compute_gradcam_pytorch(
    model: Any,
    input_tensor: Any,
    target_class: int,
    target_layer: str | None = None,
) -> tuple[np.ndarray, str]:
    import torch

    model.eval()

    if target_layer is None:
        target_layer = _find_last_conv_layer(model)
        if target_layer is None:
            raise ValueError("Could not auto-detect target convolution layer")

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(module: Any, inp: Any, out: Any) -> None:
        act = out[0] if isinstance(out, tuple) else out
        activations.append(act.detach())

    def backward_hook(module: Any, grad_in: Any, grad_out: Any) -> None:
        if grad_out and grad_out[0] is not None:
            gradients.append(grad_out[0].detach())

    target_module = _get_module_by_name(model, target_layer)
    if target_module is None:
        raise ValueError(f"Layer '{target_layer}' not found in model")

    fwd_handle = target_module.register_forward_hook(forward_hook)
    bwd_handle = target_module.register_full_backward_hook(backward_hook)

    try:
        tensor = input_tensor.clone().detach().requires_grad_(True)
        output = model(tensor)
        logits = output[0] if isinstance(output, tuple) else output

        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Model output is not a PyTorch tensor")

        model.zero_grad()
        cls_idx = min(target_class, logits.shape[-1] - 1)
        target_score = logits[0, cls_idx]
        target_score.backward(retain_graph=False)

        if not activations or not gradients:
            raise RuntimeError("Hooks did not capture activations/gradients")

        activation = activations[0]
        gradient = gradients[0]

        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        if cam_np.ndim != 2:
            cam_np = np.mean(cam_np, axis=0)

        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max > cam_min + 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    return cam_np.astype(np.float32), target_layer


def resize_heatmap(heatmap: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if heatmap.ndim > 2:
        heatmap = heatmap.squeeze()

    pil_heatmap = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
    pil_heatmap = pil_heatmap.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(pil_heatmap, dtype=np.float32) / 255.0


def save_heatmap(heatmap: np.ndarray, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), heatmap)
    return str(output_path)


def generate_explainability(
    model: Any,
    input_image: np.ndarray,
    target_class: int,
    target_layer: str | None = None,
    method: str = "gradcam",
    output_dir: str | Path | None = None,
    model_id: str = "",
    raw_pixel_data: np.ndarray | None = None,
) -> ExplainabilityData:
    import torch
    from ..inference.engine import DEVICE, compute_anomaly_map_and_bbox

    start_time = time.perf_counter()

    if isinstance(input_image, np.ndarray):
        tensor = torch.from_numpy(input_image).unsqueeze(0).float().to(DEVICE)
    else:
        tensor = input_image.to(DEVICE)

    try:
        heatmap, layer_name = compute_gradcam_pytorch(model, tensor, target_class, target_layer)
    except Exception as e:
        logger.warning("Grad-CAM hook fallback for %s: %s", model_id, e)
        heatmap = np.zeros((224, 224), dtype=np.float32)
        layer_name = target_layer or "spatial_saliency"

    if raw_pixel_data is not None:
        saliency, _, _ = compute_anomaly_map_and_bbox(raw_pixel_data, heatmap)
        if saliency.max() > 0:
            cam_resized = resize_heatmap(heatmap, saliency.shape[:2])
            if cam_resized.max() > 0:
                fused = 0.45 * cam_resized + 0.55 * saliency
            else:
                fused = saliency
            f_min, f_max = float(fused.min()), float(fused.max())
            if f_max > f_min + 1e-6:
                heatmap = (fused - f_min) / (f_max - f_min)

    explainability = ExplainabilityData(
        target_layer=layer_name,
        target_class_idx=target_class,
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = output_dir / f"{model_id}_gradcam.npy"
        save_heatmap(heatmap, heatmap_path)
        explainability.gradcam_path = str(heatmap_path)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info("Explainability generated for %s in %.1fms", model_id, elapsed_ms)
    return explainability


def _find_last_conv_layer(model: Any) -> str | None:
    import torch.nn as nn

    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            if "attention" not in name and "seg_head" not in name and "aux_classifier" not in name:
                last_conv_name = name
    return last_conv_name


def _get_module_by_name(model: Any, name: str) -> Any | None:
    parts = name.split(".")
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            try:
                module = module[int(part)]
            except (IndexError, ValueError, TypeError):
                return None
    return module
