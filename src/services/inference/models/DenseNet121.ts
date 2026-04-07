// ============================================================================
// DenseNet-121 Architecture — Local TensorFlow.js Implementation
// Based on: "Densely Connected Convolutional Networks" (Huang et al. 2017)
// Dense Block config: [6, 12, 24, 16] — 121 parameterized layers
// Optimized for MRI brain tumor classification (BraTS / TCIA UPENN-GBM)
// ============================================================================

import * as tf from '@tensorflow/tfjs';
import type { DenseNetOutput, ModelConfig } from '../types';
import { PathologyClass, DEFAULT_MODEL_CONFIG } from '../types';

// Growth rate (k) — number of feature maps each layer produces
const GROWTH_RATE = 32;
const COMPRESSION_FACTOR = 0.5;

// Dense block configuration: number of layers in each block
const BLOCK_CONFIG = [6, 12, 24, 16];

/**
 * Composite function: BN → ReLU → Conv(1x1) → BN → ReLU → Conv(3x3)
 * This is the bottleneck variant (DenseNet-B)
 */
function denseLayer(input: tf.SymbolicTensor, growthRate: number, name: string): tf.SymbolicTensor {
  // Bottleneck: 1x1 conv to reduce channel count to 4*k
  let x = tf.layers.batchNormalization({ name: `${name}_bn1` }).apply(input) as tf.SymbolicTensor;
  x = tf.layers.activation({ activation: 'relu', name: `${name}_relu1` }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.conv2d({
    filters: 4 * growthRate,
    kernelSize: 1,
    padding: 'same',
    useBias: false,
    name: `${name}_conv1`,
  }).apply(x) as tf.SymbolicTensor;

  // 3x3 conv
  x = tf.layers.batchNormalization({ name: `${name}_bn2` }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.activation({ activation: 'relu', name: `${name}_relu2` }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.conv2d({
    filters: growthRate,
    kernelSize: 3,
    padding: 'same',
    useBias: false,
    name: `${name}_conv2`,
  }).apply(x) as tf.SymbolicTensor;

  // Dense connectivity: concatenate input with output
  return tf.layers.concatenate({ name: `${name}_concat` }).apply([input, x]) as tf.SymbolicTensor;
}

/**
 * Dense Block: stack of dense layers with dense connectivity
 */
function denseBlock(input: tf.SymbolicTensor, numLayers: number, growthRate: number, blockName: string): tf.SymbolicTensor {
  let x = input;
  for (let i = 0; i < numLayers; i++) {
    x = denseLayer(x, growthRate, `${blockName}_layer${i}`);
  }
  return x;
}

/**
 * Transition Layer: BN → 1x1 Conv (compression) → 2x2 AvgPool
 */
function transitionLayer(input: tf.SymbolicTensor, compressionFactor: number, name: string): tf.SymbolicTensor {
  const numFilters = Math.floor((input.shape[input.shape.length - 1] as number) * compressionFactor);

  let x = tf.layers.batchNormalization({ name: `${name}_bn` }).apply(input) as tf.SymbolicTensor;
  x = tf.layers.activation({ activation: 'relu', name: `${name}_relu` }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.conv2d({
    filters: numFilters,
    kernelSize: 1,
    padding: 'same',
    useBias: false,
    name: `${name}_conv`,
  }).apply(x) as tf.SymbolicTensor;

  x = tf.layers.averagePooling2d({
    poolSize: 2,
    strides: 2,
    name: `${name}_pool`,
  }).apply(x) as tf.SymbolicTensor;

  return x;
}

/**
 * Build the complete DenseNet-121 model
 */
export function buildDenseNet121(config: ModelConfig = DEFAULT_MODEL_CONFIG): tf.LayersModel {
  const [h, w] = config.inputSize;
  const input = tf.input({ shape: [h, w, 3], name: 'densenet_input' });

  // Initial convolution: 7x7 conv, stride 2 → 3x3 max pool, stride 2
  let x = tf.layers.conv2d({
    filters: 64,
    kernelSize: 7,
    strides: 2,
    padding: 'same',
    useBias: false,
    name: 'initial_conv',
  }).apply(input) as tf.SymbolicTensor;

  x = tf.layers.batchNormalization({ name: 'initial_bn' }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.activation({ activation: 'relu', name: 'initial_relu' }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.maxPooling2d({ poolSize: 3, strides: 2, padding: 'same', name: 'initial_pool' }).apply(x) as tf.SymbolicTensor;

  // Dense Blocks + Transition Layers
  for (let i = 0; i < BLOCK_CONFIG.length; i++) {
    x = denseBlock(x, BLOCK_CONFIG[i], GROWTH_RATE, `dense_block_${i}`);

    // No transition after the last dense block
    if (i < BLOCK_CONFIG.length - 1) {
      x = transitionLayer(x, COMPRESSION_FACTOR, `transition_${i}`);
    }
  }

  // Final BN → ReLU → GlobalAvgPool → Dense(softmax)
  x = tf.layers.batchNormalization({ name: 'final_bn' }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.activation({ activation: 'relu', name: 'final_relu' }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.globalAveragePooling2d({ name: 'global_avg_pool' }).apply(x) as tf.SymbolicTensor;

  const output = tf.layers.dense({
    units: config.numClasses,
    activation: 'softmax',
    name: 'classification_head',
  }).apply(x) as tf.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: output, name: 'DenseNet121' });
  return model;
}

// Pathology classes in order (matching model output indices)
const PATHOLOGY_CLASSES: PathologyClass[] = [
  PathologyClass.NORMAL,
  PathologyClass.MENINGIOMA,
  PathologyClass.GLIOMA_LOW,
  PathologyClass.GLIOMA_HIGH,
  PathologyClass.GLIOBLASTOMA,
  PathologyClass.METASTASIS,
  PathologyClass.EDEMA,
  PathologyClass.NECROSIS,
];

/**
 * Run inference on a preprocessed image tensor
 */
export async function runDenseNet121Inference(
  model: tf.LayersModel,
  imageTensor: tf.Tensor
): Promise<DenseNetOutput> {
  const startTime = performance.now();

  // Ensure correct input shape [1, 224, 224, 3]
  let input = imageTensor;
  if (input.shape.length === 3) {
    input = input.expandDims(0);
  }

  // Run inference
  const predictions = model.predict(input) as tf.Tensor;
  const probabilities = await predictions.data();

  // Build classification results
  const classifications = PATHOLOGY_CLASSES.map((pathology, i) => ({
    pathology,
    probability: probabilities[i] || 0,
  }));

  // Sort by probability (descending)
  classifications.sort((a, b) => b.probability - a.probability);

  const primaryDiagnosis = classifications[0].pathology;
  const confidence = classifications[0].probability * 100;

  // Extract feature vector from the global average pooling layer
  const featureModel = tf.model({
    inputs: model.input,
    outputs: model.getLayer('global_avg_pool').output,
  });
  const features = featureModel.predict(input) as tf.Tensor;
  const featureVector = Array.from(await features.data());

  const inferenceTimeMs = performance.now() - startTime;

  // Cleanup intermediate tensors
  predictions.dispose();
  features.dispose();

  return {
    modelName: 'DenseNet-121',
    classifications,
    primaryDiagnosis,
    confidence,
    inferenceTimeMs,
    featureVector,
  };
}
