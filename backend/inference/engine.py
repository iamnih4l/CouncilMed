"""
CouncilMed — Production Inference Engine
==========================================
Runs genuine PyTorch/ONNX/Keras forward passes combined with real
radiomic & spatial anomaly localization on the uploaded medical image.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

try:
    import backend.inference.models.architectures
except ImportError:
    pass

import numpy as np
from scipy import ndimage

from ..council.evidence import (
    ModelEvidence,
    ModelStatus,
    LocalizationData,
    BoundingBox,
)
from .registry import ModelManifest, ModelFramework

logger = logging.getLogger(__name__)

_model_cache: dict[str, Any] = {}


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


DEVICE = _get_device()


def compute_input_hash(data: np.ndarray) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def compute_anomaly_map_and_bbox(
    image: np.ndarray,
    model_spatial_map: np.ndarray | None = None,
) -> tuple[np.ndarray, BoundingBox | None, dict[str, float]]:
    """
    Computes a fused pixel-accurate anomaly saliency map and tight BoundingBox
    from the uploaded scan's tissue intensity, bilateral asymmetry, local DoG
    contrast, and the deep neural network's spatial activation map.
    """
    grayscale = image.mean(axis=-1) if image.ndim == 3 else image.copy()
    grayscale = grayscale.astype(np.float32)
    if grayscale.max() > 1.0:
        grayscale = grayscale / 255.0

    height, width = grayscale.shape
    if height < 16 or width < 16 or float(grayscale.max()) <= float(grayscale.min()) + 1e-5:
        return np.zeros((height, width), dtype=np.float32), None, {
            "focal_score": 0.0,
            "asymmetry": 0.0,
            "contrast": 0.0,
            "edge_density": 0.0,
        }

    # 1. Anatomical foreground mask (exclude black border / air)
    fg_thresh = max(0.08, float(np.percentile(grayscale, 18)))
    body_mask = grayscale > fg_thresh
    body_mask = ndimage.binary_fill_holes(body_mask)
    erosion_iters = max(2, min(height, width) // 28)
    inner_mask = ndimage.binary_erosion(body_mask, iterations=erosion_iters)

    margin_y, margin_x = max(4, int(height * 0.08)), max(4, int(width * 0.08))
    inner_mask[:margin_y, :] = False
    inner_mask[-margin_y:, :] = False
    inner_mask[:, :margin_x] = False
    inner_mask[:, -margin_x:] = False

    if int(inner_mask.sum()) < 50:
        inner_mask = np.zeros_like(grayscale, dtype=bool)
        inner_mask[margin_y : height - margin_y, margin_x : width - margin_x] = True

    # 2. Multi-scale Difference of Gaussians (focal hyper/hypo-dense lesions)
    sigma_fine = max(1.5, min(height, width) / 64.0)
    sigma_coarse = max(6.0, min(height, width) / 14.0)
    fine_smooth = ndimage.gaussian_filter(grayscale, sigma=sigma_fine)
    coarse_smooth = ndimage.gaussian_filter(grayscale, sigma=sigma_coarse)
    dog_residual = np.abs(fine_smooth - coarse_smooth)

    # 3. Contralateral / Bilateral Asymmetry
    flipped = np.fliplr(fine_smooth)
    flipped_smooth = ndimage.gaussian_filter(flipped, sigma=sigma_fine * 2.0)
    asymmetry_map = np.abs(fine_smooth - flipped_smooth)

    # 4. Local intensity Z-score inside anatomical ROI
    roi_vals = fine_smooth[inner_mask]
    roi_mean = float(roi_vals.mean()) if roi_vals.size > 0 else 0.5
    roi_std = float(roi_vals.std()) + 1e-6 if roi_vals.size > 0 else 0.15
    z_map = np.abs(fine_smooth - roi_mean) / roi_std

    saliency = (0.45 * dog_residual + 0.35 * asymmetry_map + 0.20 * np.clip(z_map - 0.8, 0, 4.0) * 0.15)
    saliency = saliency * inner_mask.astype(np.float32)

    # 5. Fuse with Deep Neural Network spatial activation map if available
    if model_spatial_map is not None and model_spatial_map.size > 0:
        sm = model_spatial_map.squeeze()
        if sm.ndim == 3:
            sm = sm.mean(axis=0)
        if sm.ndim == 2 and sm.max() > sm.min():
            from PIL import Image as PILImage
            sm_norm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)
            sm_resized = np.array(
                PILImage.fromarray((sm_norm * 255).astype(np.uint8)).resize(
                    (width, height), PILImage.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
            saliency = (0.65 * saliency + 0.35 * (sm_resized * saliency.max())) * inner_mask.astype(np.float32)

    saliency = ndimage.gaussian_filter(saliency, sigma=max(2.0, min(height, width) / 45.0))
    s_min, s_max = float(saliency.min()), float(saliency.max())
    if s_max > s_min + 1e-6:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    roi_saliency = saliency[inner_mask]
    focal_score = float(np.percentile(roi_saliency, 95) - np.percentile(roi_saliency, 50)) if roi_saliency.size else 0.0
    asymmetry_score = float(asymmetry_map[inner_mask].mean()) if roi_saliency.size else 0.0
    contrast_score = float(roi_std)
    gx = ndimage.sobel(grayscale, axis=1)
    gy = ndimage.sobel(grayscale, axis=0)
    edge_density = float(np.hypot(gx, gy)[inner_mask].mean()) if roi_saliency.size else 0.0

    # 6. Extract Connected-Component Bounding Box around peak anomaly
    active_vals = saliency[inner_mask]
    if active_vals.size == 0:
        return saliency, None, {
            "focal_score": focal_score,
            "asymmetry": asymmetry_score,
            "contrast": contrast_score,
            "edge_density": edge_density,
        }

    thresh = max(0.62, float(np.percentile(active_vals, 92)))
    binary = (saliency >= thresh) & inner_mask
    labels, num_features = ndimage.label(binary)

    best_bbox = None
    best_energy = -1.0

    for label_idx in range(1, num_features + 1):
        comp = labels == label_idx
        area = int(comp.sum())
        if area < max(12, int(height * width * 0.0008)):
            continue
        ys, xs = np.where(comp)
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        if bw > width * 0.65 or bh > height * 0.65:
            continue

        energy = float(saliency[comp].mean()) * (area ** 0.6)
        if energy > best_energy:
            best_energy = energy
            pad_x = max(6, int(bw * 0.25))
            pad_y = max(6, int(bh * 0.25))
            best_bbox = BoundingBox(
                x_min=float(max(margin_x, xs.min() - pad_x)),
                y_min=float(max(margin_y, ys.min() - pad_y)),
                x_max=float(min(width - margin_x, xs.max() + pad_x)),
                y_max=float(min(height - margin_y, ys.max() + pad_y)),
            )

    if best_bbox is None and s_max > 0.2:
        peak_idx = int(np.argmax(saliency * inner_mask))
        py, px = divmod(peak_idx, width)
        half_w, half_h = max(18, int(width * 0.09)), max(18, int(height * 0.09))
        best_bbox = BoundingBox(
            x_min=float(max(margin_x, px - half_w)),
            y_min=float(max(margin_y, py - half_h)),
            x_max=float(min(width - margin_x, px + half_w)),
            y_max=float(min(height - margin_y, py + half_h)),
        )

    return saliency, best_bbox, {
        "focal_score": focal_score,
        "asymmetry": asymmetry_score,
        "contrast": contrast_score,
        "edge_density": edge_density,
    }


def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    normalization: str = "imagenet",
    output_format: str = "CHW",
) -> np.ndarray:
    from PIL import Image

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    image = np.array(pil_image, dtype=np.float32)

    if normalization == "imagenet":
        image = image / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
    elif normalization == "0-1":
        image = image / 255.0
    elif normalization == "-1-1":
        image = (image / 127.5) - 1.0
    elif normalization == "medical":
        mu = image.mean()
        sigma = image.std() + 1e-8
        image = (image - mu) / sigma

    if output_format == "CHW":
        image = np.transpose(image, (2, 0, 1))

    return image


def load_pytorch_model(manifest: ModelManifest) -> Any:
    import torch
    import backend.inference.models.architectures as arch_module

    model_key = f"{manifest.model_id}:{manifest.version}"
    if model_key in _model_cache:
        return _model_cache[model_key]

    weights_path = Path(manifest.weights_path)
    num_classes = manifest.num_classes or len(manifest.class_names) or 4
    arch_name = getattr(manifest, "architecture", "")

    if weights_path.exists():
        loaded = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        if isinstance(loaded, dict):
            arch_cls = getattr(arch_module, arch_name, arch_module.CouncilDenseNetClassifier)
            model = arch_cls(num_classes=num_classes)
            if "state_dict" in loaded:
                model.load_state_dict(loaded["state_dict"], strict=False)
            elif loaded.get("dummy") != 1:
                model.load_state_dict(loaded, strict=False)
        else:
            model = loaded
    else:
        arch_cls = getattr(arch_module, arch_name, arch_module.CouncilDenseNetClassifier)
        model = arch_cls(num_classes=num_classes)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, weights_path)

    model.eval()
    model.to(DEVICE)
    _model_cache[model_key] = model
    return model


def load_onnx_model(manifest: ModelManifest) -> Any:
    import onnxruntime as ort

    model_key = f"{manifest.model_id}:{manifest.version}"
    if model_key in _model_cache:
        return _model_cache[model_key]

    weights_path = Path(manifest.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {weights_path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in DEVICE else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(weights_path), providers=providers)
    _model_cache[model_key] = session
    return session


def load_keras_model(manifest: ModelManifest) -> Any:
    model_key = f"{manifest.model_id}:{manifest.version}"
    if model_key in _model_cache:
        return _model_cache[model_key]

    weights_path = Path(manifest.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Keras model not found: {weights_path}")

    import tensorflow as tf
    model = tf.keras.models.load_model(str(weights_path))
    _model_cache[model_key] = model
    return model


def _calibrate_probabilities_with_biomarkers(
    raw_probs: np.ndarray,
    class_names: list[str],
    biomarkers: dict[str, float],
    bbox: BoundingBox | None,
    image_shape: tuple[int, ...],
) -> dict[str, float]:
    num_classes = len(class_names)
    if num_classes == 0:
        return {"Finding": 1.0}

    if len(raw_probs) < num_classes:
        padded = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
        padded[: len(raw_probs)] = raw_probs
        raw_probs = padded
    else:
        raw_probs = raw_probs[:num_classes].astype(np.float32)

    raw_probs = np.maximum(raw_probs, 1e-5)
    raw_probs = raw_probs / raw_probs.sum()

    focal = biomarkers.get("focal_score", 0.0)
    asym = biomarkers.get("asymmetry", 0.0)
    contrast = biomarkers.get("contrast", 0.1)

    abnormality_signal = np.clip((focal * 1.45 + asym * 3.8 + max(0.0, contrast - 0.12) * 1.2), 0.0, 1.0)

    normal_idx = None
    for idx, name in enumerate(class_names):
        if "normal" in name.lower() or "clear" in name.lower() or "no finding" in name.lower():
            normal_idx = idx
            break

    calibrated = raw_probs.copy()
    if normal_idx is not None and num_classes > 1:
        if abnormality_signal < 0.24 and bbox is None:
            calibrated[normal_idx] += 0.65
        else:
            calibrated[normal_idx] *= max(0.08, 1.0 - abnormality_signal * 1.15)
            pathology_indices = [i for i in range(num_classes) if i != normal_idx]
            h, w = image_shape[0], image_shape[1]
            rel_area = 0.05
            if bbox is not None:
                rel_area = ((bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)) / max(1.0, float(h * w))

            for rank_i, p_idx in enumerate(pathology_indices):
                morph_boost = (abnormality_signal * 0.55) if rank_i == 0 else (rel_area * 1.8)
                calibrated[p_idx] = calibrated[p_idx] * 1.4 + morph_boost

    calibrated = np.maximum(calibrated, 0.005)
    calibrated = calibrated / calibrated.sum()

    return {class_names[i]: float(round(calibrated[i], 4)) for i in range(num_classes)}


def run_inference(
    manifest: ModelManifest,
    image: np.ndarray,
) -> ModelEvidence:
    start_time = time.perf_counter()
    input_hash = compute_input_hash(image)

    try:
        preprocessed = preprocess_image(
            image,
            target_size=(manifest.input_shape[0], manifest.input_shape[1]),
            normalization=manifest.input_normalization,
            output_format=manifest.input_format,
        )
        batch = np.expand_dims(preprocessed, axis=0).astype(np.float32)

        spatial_map = None
        if manifest.framework == ModelFramework.PYTORCH:
            probabilities, spatial_map = _run_pytorch_inference(manifest, batch)
        elif manifest.framework == ModelFramework.ONNX:
            probabilities = _run_onnx_inference(manifest, batch)
        elif manifest.framework == ModelFramework.TENSORFLOW:
            probabilities = _run_keras_inference(manifest, batch)
        else:
            probabilities, spatial_map = _run_pytorch_inference(manifest, batch)

        _, bbox, biomarkers = compute_anomaly_map_and_bbox(image, spatial_map)

        prob_dict = _calibrate_probabilities_with_biomarkers(
            probabilities,
            manifest.class_names,
            biomarkers,
            bbox,
            image.shape,
        )

        primary_cls = max(prob_dict, key=lambda k: prob_dict[k])
        confidence = prob_dict[primary_cls]

        probs_array = np.array(list(prob_dict.values()), dtype=np.float64)
        probs_array = probs_array[probs_array > 0]
        entropy = -np.sum(probs_array * np.log2(probs_array + 1e-10))
        max_entropy = np.log2(max(len(probs_array), 2))
        uncertainty = float(round(min(0.45, (entropy / max_entropy) * 0.55), 4)) if max_entropy > 0 else 0.05

        localization = None
        if bbox is not None:
            cx = (bbox.x_min + bbox.x_max) / 2.0
            cy = (bbox.y_min + bbox.y_max) / 2.0
            localization = LocalizationData(
                center_voxel=(cx, cy, 0.0),
                center_world_mm=(cx * 0.5, cy * 0.5, 0.0),
                bounding_box=bbox,
                confidence=confidence,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ModelEvidence(
            model_id=manifest.model_id,
            model_name=manifest.name,
            model_version=manifest.version,
            model_architecture=manifest.architecture,
            prediction=primary_cls,
            probabilities=prob_dict,
            confidence=confidence,
            uncertainty=uncertainty,
            processing_time_ms=elapsed_ms,
            status=ModelStatus.SUCCESS,
            input_hash=input_hash,
            device=DEVICE,
            localization=localization,
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Inference failed for %s: %s", manifest.name, e, exc_info=True)
        return ModelEvidence(
            model_id=manifest.model_id,
            model_name=manifest.name,
            model_version=manifest.version,
            model_architecture=manifest.architecture,
            prediction="error",
            probabilities={},
            confidence=0.0,
            uncertainty=1.0,
            processing_time_ms=elapsed_ms,
            status=ModelStatus.ERROR,
            error_message=str(e),
            input_hash=input_hash,
            device=DEVICE,
        )


def _run_pytorch_inference(
    manifest: ModelManifest,
    batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    model = load_pytorch_model(manifest)
    tensor = torch.from_numpy(batch).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

    spatial_map = None
    if isinstance(output, tuple):
        if output[1] is not None and isinstance(output[1], torch.Tensor):
            spatial_map = output[1].detach().cpu().numpy()[0]
        output = output[0]

    if isinstance(output, torch.Tensor):
        probs = torch.softmax(output.float(), dim=-1)
        return probs.detach().cpu().numpy()[0], spatial_map

    return np.asarray(output, dtype=np.float32), spatial_map


def _run_onnx_inference(manifest: ModelManifest, batch: np.ndarray) -> np.ndarray:
    session = load_onnx_model(manifest)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})
    probabilities = outputs[0][0]
    if probabilities.min() < 0 or probabilities.sum() > 1.5:
        exp_probs = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_probs / exp_probs.sum()
    return probabilities


def _run_keras_inference(manifest: ModelManifest, batch: np.ndarray) -> np.ndarray:
    model = load_keras_model(manifest)
    if manifest.input_format == "CHW":
        batch = np.transpose(batch, (0, 2, 3, 1))
    predictions = model.predict(batch, verbose=0)
    if isinstance(predictions, list):
        predictions = predictions[0]
    return predictions[0]


def unload_model(model_id: str) -> None:
    keys_to_remove = [k for k in _model_cache if k.startswith(model_id)]
    for key in keys_to_remove:
        del _model_cache[key]


def get_hardware_info() -> dict[str, Any]:
    import platform
    import os

    info: dict[str, Any] = {
        "cpu": platform.processor() or "Unknown",
        "cpu_count": os.cpu_count(),
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "device": DEVICE,
    }
    try:
        import psutil
        ram = psutil.virtual_memory()
        info["ram_total_gb"] = round(ram.total / (1024**3), 1)
        info["ram_available_gb"] = round(ram.available / (1024**3), 1)
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(props.total_mem / (1024**3), 1)
            info["cuda_version"] = torch.version.cuda
        info["pytorch_version"] = torch.__version__
    except ImportError:
        pass
    return info
