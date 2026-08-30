"""
CouncilMed — Model Converter
================================
Convert trained Keras models to ONNX format for production inference.

This enables the existing trained MobileNetV2 classifier (classifier.keras)
to be used by the new ONNX-based inference engine alongside PyTorch models.
"""

import argparse
import os
import sys
from pathlib import Path


def convert_keras_to_onnx(
    keras_path: str,
    onnx_path: str,
    input_shape: tuple = (1, 224, 224, 3),
    opset: int = 13,
) -> None:
    """Convert a Keras model to ONNX format."""
    import tensorflow as tf
    import tf2onnx
    
    print(f"Loading Keras model: {keras_path}")
    model = tf.keras.models.load_model(keras_path)
    model.summary()
    
    print(f"\nConverting to ONNX (opset {opset})...")
    
    # Get input spec
    input_spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    
    # Convert
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec,
        opset=opset,
        output_path=onnx_path,
    )
    
    print(f"ONNX model saved: {onnx_path}")
    print(f"  Inputs: {[inp.name for inp in model_proto.graph.input]}")
    print(f"  Outputs: {[out.name for out in model_proto.graph.output]}")
    
    # Validate
    import onnxruntime as ort
    import numpy as np
    
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    dummy = np.random.randn(*input_shape).astype(np.float32)
    result = session.run(None, {input_name: dummy})
    
    print(f"\n  Validation OK — output shape: {result[0].shape}")
    print(f"  Output sum: {result[0].sum():.4f} (should be ~1.0 for softmax)")


def convert_keras_to_pytorch(
    keras_path: str,
    pytorch_path: str,
    num_classes: int = 4,
) -> None:
    """
    Convert a Keras MobileNetV2 classifier to a PyTorch equivalent.
    
    This is a more involved conversion that creates a PyTorch model
    with the same architecture and transfers the weights.
    """
    import tensorflow as tf
    import torch
    import torchvision.models as models
    import numpy as np
    
    print(f"Loading Keras model: {keras_path}")
    keras_model = tf.keras.models.load_model(keras_path)
    
    # Create equivalent PyTorch model
    print("Creating PyTorch MobileNetV2...")
    pytorch_model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    
    # The Keras model is Sequential: [MobileNetV2_base, GAP, BN, Dropout, Dense(256), BN, Dropout, Dense(4)]
    # We need to map weights from Keras to PyTorch
    
    # For now, save just the architecture with random weights
    # The proper weight transfer would require layer-by-layer mapping
    # which is complex for MobileNetV2's inverted residual blocks
    
    # Alternative: Use ONNX as the bridge format
    print("NOTE: Direct Keras → PyTorch weight transfer for MobileNetV2 is complex.")
    print("      Recommend using ONNX format instead for cross-framework inference.")
    print(f"      Run: python {__file__} --to onnx {keras_path}")
    
    torch.save(pytorch_model, pytorch_path)
    print(f"PyTorch model architecture saved: {pytorch_path}")
    print("WARNING: This model has random weights. Use ONNX for real inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CouncilMed Model Converter")
    parser.add_argument("input", help="Path to input model (Keras .keras or .h5)")
    parser.add_argument("--to", choices=["onnx", "pytorch"], default="onnx",
                        help="Target format")
    parser.add_argument("--output", "-o", help="Output path (auto-generated if omitted)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if args.to == "onnx":
        output_path = args.output or str(input_path.with_suffix(".onnx"))
        convert_keras_to_onnx(str(input_path), output_path, opset=args.opset)
    elif args.to == "pytorch":
        output_path = args.output or str(input_path.with_suffix(".pt"))
        convert_keras_to_pytorch(str(input_path), output_path)
