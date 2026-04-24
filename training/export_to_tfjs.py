"""
CouncilMed - TF.js Graph Model Export
=====================================
Converts trained models to TF.js Graph Model format by:
1. Loading the Keras model
2. Getting its concrete function
3. Freezing the graph
4. Manually constructing the graph model JSON + weight shards
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import struct
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil

OUTPUT_DIR = Path(__file__).parent / "output"
FRONTEND_MODELS_DIR = Path(__file__).parent.parent / "public" / "models"


def freeze_and_export(keras_path, output_dir, model_name):
    """
    Convert Keras model -> frozen graph -> TF.js graph model.
    This bypasses the buggy tensorflowjs library entirely.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = tf.keras.models.load_model(keras_path)
    print(f"  Loaded: {model.name}")
    
    # Get concrete function
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    
    spec = tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    
    @tf.function(input_signature=[spec])
    def serve(x):
        return model(x, training=False)
    
    concrete = serve.get_concrete_function()
    
    # Get the frozen graph
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen = convert_variables_to_constants_v2(concrete)
    
    graph_def = frozen.graph.as_graph_def()
    
    # Get input/output names
    input_names = [t.name for t in frozen.inputs]
    output_names = [t.name for t in frozen.outputs]
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")
    
    # Extract weights from the frozen graph
    weights_data = b''
    weight_specs = []
    
    for node in graph_def.node:
        if node.op == 'Const':
            tensor = node.attr['value'].tensor
            dtype = tensor.dtype
            
            # Only export float32 weights (skip other consts like shapes)
            if dtype != 1:  # 1 = DT_FLOAT
                continue
            
            shape = [d.size for d in tensor.tensor_shape.dim]
            if not shape or all(s == 0 for s in shape):
                # Scalar or empty - try to get from tensor_content
                if tensor.tensor_content:
                    num_elements = len(tensor.tensor_content) // 4
                    if num_elements <= 1:
                        continue
                    shape = [num_elements]
                elif tensor.float_val:
                    if len(tensor.float_val) <= 1:
                        continue
                    shape = [len(tensor.float_val)]
                else:
                    continue
            
            # Get raw data
            if tensor.tensor_content:
                raw = tensor.tensor_content
            elif tensor.float_val:
                raw = struct.pack(f'{len(tensor.float_val)}f', *tensor.float_val)
                # If shape implies broadcasting, expand
                total_elems = 1
                for s in shape:
                    total_elems *= s
                if len(tensor.float_val) < total_elems:
                    # Broadcasting single value
                    raw = struct.pack(f'{total_elems}f', *([tensor.float_val[0]] * total_elems))
            else:
                continue
            
            expected_size = 1
            for s in shape:
                expected_size *= s
            expected_bytes = expected_size * 4
            
            if len(raw) != expected_bytes:
                continue
            
            weight_specs.append({
                'name': node.name,
                'shape': shape,
                'dtype': 'float32',
            })
            weights_data += raw
    
    print(f"  Extracted {len(weight_specs)} weight tensors ({len(weights_data)/1e6:.1f} MB)")
    
    # Write weight shards
    MAX_SHARD = 4 * 1024 * 1024
    total = len(weights_data)
    num_shards = max(1, (total + MAX_SHARD - 1) // MAX_SHARD)
    
    paths = []
    for i in range(num_shards):
        name = f"group1-shard{i+1}of{num_shards}.bin"
        paths.append(name)
        with open(output_dir / name, 'wb') as f:
            f.write(weights_data[i*MAX_SHARD : min((i+1)*MAX_SHARD, total)])
    
    # Build the graph model JSON
    # Convert the graph_def to a serializable format
    nodes = []
    for node in graph_def.node:
        n = {
            'name': node.name,
            'op': node.op,
            'input': list(node.input),
        }
        # Add relevant attributes
        attr = {}
        for key, val in node.attr.items():
            if key == 'value':
                continue  # Skip weight values (stored in shards)
            if key == 'dtype':
                attr[key] = {'type': _dtype_to_string(val.type)}
            elif key == 'T':
                attr[key] = {'type': _dtype_to_string(val.type)}
            elif key == 'shape':
                dims = [{'size': str(d.size)} for d in val.shape.dim]
                attr[key] = {'shape': {'dim': dims}}
            elif key in ('strides', 'ksize', 'dilations'):
                attr[key] = {'list': {'i': [str(v) for v in val.list.i]}}
            elif key == 'padding':
                attr[key] = {'s': val.s.decode('utf-8') if isinstance(val.s, bytes) else str(val.s)}
            elif key == 'data_format':
                attr[key] = {'s': val.s.decode('utf-8') if isinstance(val.s, bytes) else str(val.s)}
            elif key == 'N':
                attr[key] = {'i': str(val.i)}
            elif key == 'num_split':
                attr[key] = {'i': str(val.i)}
            elif key == 'axis':
                attr[key] = {'i': str(val.i)}
            elif key == 'keep_dims':
                attr[key] = {'b': val.b}
            elif key == 'transpose_a' or key == 'transpose_b':
                attr[key] = {'b': val.b}
            elif key == 'epsilon':
                attr[key] = {'f': val.f}
            elif key == 'alpha':
                attr[key] = {'f': val.f}
        if attr:
            n['attr'] = attr
        nodes.append(n)
    
    model_json = {
        'format': 'graph-model',
        'generatedBy': 'CouncilMed v3',
        'convertedBy': 'CouncilMed Frozen Graph Exporter',
        'modelTopology': {
            'node': nodes,
            'versions': {'producer': 1},
        },
        'signature': {
            'inputs': {clean_name(n): {'name': n, 'dtype': 'DT_FLOAT', 'tensorShape': {'dim': [{'size': str(s) if s else '-1'} for s in input_shape]}} for n in input_names},
            'outputs': {},
        },
        'weightsManifest': [{
            'paths': paths,
            'weights': weight_specs,
        }],
    }
    
    # Add output signature
    for out_name in output_names:
        clean = clean_name(out_name)
        model_json['signature']['outputs'][clean] = {
            'name': out_name,
            'dtype': 'DT_FLOAT',
        }
    
    with open(output_dir / 'model.json', 'w') as f:
        json.dump(model_json, f)
    
    print(f"  [OK] Graph model exported to {output_dir}")
    
    # Verify
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    pred = model.predict(dummy, verbose=0)
    if isinstance(pred, list):
        for j, p in enumerate(pred):
            print(f"  Verify output {j}: {p.flatten()[:4]}")
    else:
        print(f"  Verify output: {pred.flatten()[:4]}")
    
    return True


def _dtype_to_string(dtype_enum):
    """Convert TF dtype enum to string."""
    mapping = {
        1: 'DT_FLOAT',
        2: 'DT_DOUBLE',
        3: 'DT_INT32',
        4: 'DT_UINT8',
        5: 'DT_INT16',
        6: 'DT_INT8',
        7: 'DT_STRING',
        9: 'DT_INT64',
        10: 'DT_BOOL',
    }
    return mapping.get(dtype_enum, f'DT_UNKNOWN_{dtype_enum}')


def clean_name(tensor_name):
    """Clean tensor name for use as key."""
    return tensor_name.split(':')[0].replace('/', '_')


def main():
    print("=" * 60)
    print("CouncilMed - TF.js Graph Model Export")
    print("=" * 60)
    
    models = [
        ("classifier", OUTPUT_DIR / "classifier" / "classifier.keras"),
        ("attention", OUTPUT_DIR / "attention" / "attention.keras"),
        ("segmenter", OUTPUT_DIR / "segmenter" / "segmenter.keras"),
    ]
    
    success = 0
    for name, path in models:
        print(f"\n-> {name}...")
        try:
            if freeze_and_export(path, FRONTEND_MODELS_DIR / name, name):
                success += 1
                meta = OUTPUT_DIR / name / "model_metadata.json"
                if meta.exists():
                    shutil.copy2(meta, FRONTEND_MODELS_DIR / name / "model_metadata.json")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print(f"Exported {success}/{len(models)} graph models")
    print("=" * 60)


if __name__ == "__main__":
    main()
