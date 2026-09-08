"""
CouncilMed — Inference Engine
================================
Runs genuine model inference through PyTorch/MONAI/ONNX.

This replaces the previous simulated inference that loaded models but never
called them. Every model forward pass here is REAL — producing actual
predictions from actual neural network weights.

The engine supports three model backends:
  1. PyTorch (native .pt/.pth models)
  2. TensorFlow/Keras (.keras/.h5 models via tf2onnx or keras directly)
  3. ONNX Runtime (.onnx models)

For the initial rebuild, we use the existing trained MobileNetV2 classifier
(training/output/classifier/classifier.keras) converted to ONNX, plus
additional models added incrementally.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

# Import architectures so torch unpickler can find them
try:
    import backend.inference.models.architectures
except ImportError:
    pass

import numpy as np

from ..council.evidence import (
    ModelEvidence,
    ModelStatus,
    LocalizationData,
    QualityMetrics,
    BoundingBox,
)
from .registry import ModelManifest, ModelFramework

logger = logging.getLogger(__name__)


# Global model cache to avoid reloading
_model_cache: dict[str, Any] = {}


def _get_device() -> str:
    """Detect the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info("GPU detected: %s (%.1f GB VRAM)", gpu_name, vram_gb)
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Metal (MPS) detected")
            return "mps"
    except ImportError:
        pass
    
    logger.info("Using CPU for inference")
    return "cpu"


DEVICE = _get_device()


def compute_input_hash(data: np.ndarray) -> str:
    """Compute a SHA-256 hash of input data for reproducibility tracking."""
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    normalization: str = "imagenet",
    output_format: str = "CHW",
) -> np.ndarray:
    """
    Preprocess a medical image for model inference.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        target_size: Target spatial dimensions (H, W)
        normalization: Normalization scheme
        output_format: "CHW" for PyTorch or "HWC" for TensorFlow
    
    Returns:
        Preprocessed image as numpy array
    """
    from PIL import Image
    
    # Ensure 3-channel
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]  # Drop alpha
    
    # Convert to uint8 if float
    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Resize
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    image = np.array(pil_image, dtype=np.float32)
    
    # Normalize
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
        # Z-score normalization per-image
        mu = image.mean()
        sigma = image.std() + 1e-8
        image = (image - mu) / sigma
    
    # Format
    if output_format == "CHW":
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    
    return image


def load_pytorch_model(manifest: ModelManifest) -> Any:
    """Load a PyTorch model from weights file."""
    import torch
    
    model_key = f"{manifest.model_id}:{manifest.version}"
    if model_key in _model_cache:
        return _model_cache[model_key]
    
    weights_path = Path(manifest.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    logger.info("Loading PyTorch model: %s from %s", manifest.name, weights_path)
    
    # Load the model
    model = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    
    if isinstance(model, dict):
        # If it's a state dict, we need the architecture separately
        raise ValueError(
            f"Model file contains a state_dict, not a full model. "
            f"Provide a complete model or define the architecture in the manifest."
        )
    
    model.eval()
    model.to(DEVICE)
    
    _model_cache[model_key] = model
    logger.info("Model loaded successfully: %s", manifest.name)
    
    return model


def load_onnx_model(manifest: ModelManifest) -> Any:
    """Load an ONNX model for inference."""
    import onnxruntime as ort
    
    model_key = f"{manifest.model_id}:{manifest.version}"
    if model_key in _model_cache:
        return _model_cache[model_key]
    
    weights_path = Path(manifest.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {weights_path}")
    
    logger.info("Loading ONNX model: %s from %s", manifest.name, weights_path)
    
    providers = []
    if "cuda" in DEVICE:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    
    session = ort.InferenceSession(str(weights_path), providers=providers)
    
    _model_cache[model_key] = session
    logger.info("ONNX model loaded: %s (providers: %s)", manifest.name, providers)
    
    return session


def load_keras_model(manifest: ModelManifest) -> Any:
    """Load a Keras/TensorFlow model."""
    model_key = f"{manifest.model_id}:{manifest.version}"
    if model_key in _model_cache:
        return _model_cache[model_key]
    
    weights_path = Path(manifest.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Keras model not found: {weights_path}")
    
    logger.info("Loading Keras model: %s from %s", manifest.name, weights_path)
    
    import tensorflow as tf
    model = tf.keras.models.load_model(str(weights_path))
    
    _model_cache[model_key] = model
    logger.info("Keras model loaded: %s", manifest.name)
    
    return model


def run_inference(
    manifest: ModelManifest,
    image: np.ndarray,
) -> ModelEvidence:
    """
    Run genuine model inference and return structured evidence.
    
    This is the core function that replaces the simulated inference.
    It actually calls the model's forward pass and returns real predictions.
    
    Args:
        manifest: Model manifest with architecture and weight information
        image: Preprocessed input image (H, W, C) as numpy array
    
    Returns:
        ModelEvidence with real predictions, probabilities, and uncertainty
    """
    start_time = time.perf_counter()
    input_hash = compute_input_hash(image)
    
    try:
        # Preprocess
        preprocessed = preprocess_image(
            image,
            target_size=(manifest.input_shape[0], manifest.input_shape[1]),
            normalization=manifest.input_normalization,
            output_format=manifest.input_format,
        )
        
        # Add batch dimension
        batch = np.expand_dims(preprocessed, axis=0).astype(np.float32)
        
        # Run inference based on framework
        spatial_map = None
        if manifest.framework == ModelFramework.PYTORCH:
            probabilities, spatial_map = _run_pytorch_inference(manifest, batch)
        elif manifest.framework == ModelFramework.ONNX:
            probabilities = _run_onnx_inference(manifest, batch)
        elif manifest.framework == ModelFramework.TENSORFLOW:
            probabilities = _run_keras_inference(manifest, batch)
        else:
            raise ValueError(f"Unsupported framework: {manifest.framework}")
        
        # Map probabilities to class names
        prob_dict: dict[str, float] = {}
        for i, cls_name in enumerate(manifest.class_names):
            if i < len(probabilities):
                prob_dict[cls_name] = float(probabilities[i])
        
        # Primary prediction
        primary_cls = max(prob_dict, key=lambda k: prob_dict[k])
        confidence = prob_dict[primary_cls]
        
        # Compute uncertainty as entropy of the probability distribution
        probs_array = np.array(list(prob_dict.values()))
        probs_array = probs_array[probs_array > 0]
        entropy = -np.sum(probs_array * np.log2(probs_array + 1e-10))
        max_entropy = np.log2(max(len(probs_array), 2))
        uncertainty = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Compute bounding box from spatial map
        localization = None
        if spatial_map is not None:
            if spatial_map.ndim == 3:
                spatial_map = spatial_map.max(axis=0)
            
            threshold = spatial_map.max() * 0.5
            y_indices, x_indices = np.where(spatial_map > threshold)
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                map_h, map_w = spatial_map.shape
                orig_h, orig_w = 512, 512 # Standardized coordinate space
                
                x_min = (x_indices.min() / map_w) * orig_w
                x_max = ((x_indices.max() + 1) / map_w) * orig_w
                y_min = (y_indices.min() / map_h) * orig_h
                y_max = ((y_indices.max() + 1) / map_h) * orig_h
                
                localization = LocalizationData(
                    bounding_box=BoundingBox(
                        x_min=float(x_min),
                        y_min=float(y_min),
                        x_max=float(x_max),
                        y_max=float(y_max),
                    )
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
        
    except FileNotFoundError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Model weights not found for %s: %s", manifest.name, e)
        return ModelEvidence(
            model_id=manifest.model_id,
            model_name=manifest.name,
            model_version=manifest.version,
            model_architecture=manifest.architecture,
            prediction="unavailable",
            probabilities={},
            confidence=0.0,
            uncertainty=1.0,
            processing_time_ms=elapsed_ms,
            status=ModelStatus.UNAVAILABLE,
            error_message=str(e),
            input_hash=input_hash,
            device=DEVICE,
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
    """Run inference with a PyTorch model."""
    import torch
    
    model = load_pytorch_model(manifest)
    
    tensor = torch.from_numpy(batch).to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor)
        
    spatial_map = None
    if isinstance(output, tuple):
        spatial_map = output[1].cpu().numpy()[0] if output[1] is not None else None
        output = output[0]
    
    # Apply softmax if not already applied
    if isinstance(output, torch.Tensor):
        probs = torch.softmax(output, dim=-1)
        return probs.cpu().numpy()[0], spatial_map
    
    return output, spatial_map


def _run_onnx_inference(
    manifest: ModelManifest,
    batch: np.ndarray,
) -> np.ndarray:
    """Run inference with an ONNX model."""
    session = load_onnx_model(manifest)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})
    
    probabilities = outputs[0][0]
    
    # Apply softmax if raw logits
    if probabilities.min() < 0 or probabilities.sum() > 1.5:
        exp_probs = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_probs / exp_probs.sum()
    
    return probabilities


def _run_keras_inference(
    manifest: ModelManifest,
    batch: np.ndarray,
) -> np.ndarray:
    """Run inference with a Keras/TensorFlow model."""
    model = load_keras_model(manifest)
    
    # Keras models expect HWC format
    if manifest.input_format == "CHW":
        batch = np.transpose(batch, (0, 2, 3, 1))
    
    predictions = model.predict(batch, verbose=0)
    
    if isinstance(predictions, list):
        predictions = predictions[0]
    
    return predictions[0]


def unload_model(model_id: str) -> None:
    """Unload a model from cache to free memory."""
    keys_to_remove = [k for k in _model_cache if k.startswith(model_id)]
    for key in keys_to_remove:
        del _model_cache[key]
        logger.info("Unloaded model: %s", key)
    
    # Try to free GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_hardware_info() -> dict[str, Any]:
    """Get hardware information for the current system."""
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
    
    # RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        info["ram_total_gb"] = round(ram.total / (1024**3), 1)
        info["ram_available_gb"] = round(ram.available / (1024**3), 1)
    except ImportError:
        pass
    
    # GPU
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
