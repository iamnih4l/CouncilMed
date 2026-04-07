// ============================================================================
// Attention Net Architecture — Local TensorFlow.js Implementation
// Based on: "Attention U-Net" (Oktay et al. 2018) + CBAM spatial attention
// Produces spatial attention heatmaps for Region of Interest localization
// Optimized for MRI brain tumor ROI detection (BraTS 2023)
// ============================================================================

import * as tf from '@tensorflow/tfjs';
import type { AttentionNetOutput, BrainRegionKey } from '../types';
import { BRAIN_REGIONS } from '../types';

/**
 * Channel Attention Module (Squeeze-and-Excitation style)
 * Learns "what" to attend to
 */
function channelAttention(input: tf.SymbolicTensor, reductionRatio: number, name: string): tf.SymbolicTensor {
  const channels = input.shape[input.shape.length - 1] as number;

  // Global Average Pooling
  let avgPool = tf.layers.globalAveragePooling2d({ name: `${name}_gavg` }).apply(input) as tf.SymbolicTensor;

  // MLP: FC → ReLU → FC → Sigmoid
  avgPool = tf.layers.dense({
    units: Math.floor(channels / reductionRatio),
    activation: 'relu',
    name: `${name}_fc1`,
  }).apply(avgPool) as tf.SymbolicTensor;

  avgPool = tf.layers.dense({
    units: channels,
    activation: 'sigmoid',
    name: `${name}_fc2`,
  }).apply(avgPool) as tf.SymbolicTensor;

  // Reshape to [1, 1, channels] for broadcasting
  const reshaped = tf.layers.reshape({
    targetShape: [1, 1, channels],
    name: `${name}_reshape`,
  }).apply(avgPool) as tf.SymbolicTensor;

  // Scale input features
  return tf.layers.multiply({ name: `${name}_scale` }).apply([input, reshaped]) as tf.SymbolicTensor;
}

/**
 * Spatial Attention Module
 * Learns "where" to attend to — produces the heatmap
 */
function spatialAttention(input: tf.SymbolicTensor, name: string): tf.SymbolicTensor {
  // 7x7 convolution to produce a spatial attention map
  const attnMap = tf.layers.conv2d({
    filters: 1,
    kernelSize: 7,
    padding: 'same',
    activation: 'sigmoid',
    useBias: false,
    name: `${name}_conv`,
  }).apply(input) as tf.SymbolicTensor;

  return tf.layers.multiply({ name: `${name}_apply` }).apply([input, attnMap]) as tf.SymbolicTensor;
}

/**
 * CBAM Block (Convolutional Block Attention Module)
 * Channel Attention → Spatial Attention (sequential)
 */
function cbamBlock(input: tf.SymbolicTensor, reductionRatio: number, name: string): tf.SymbolicTensor {
  let x = channelAttention(input, reductionRatio, `${name}_channel`);
  x = spatialAttention(x, `${name}_spatial`);
  return x;
}

/**
 * Encoder block: Conv → BN → ReLU → Conv → BN → ReLU → CBAM → MaxPool
 */
function encoderBlock(
  input: tf.SymbolicTensor,
  filters: number,
  name: string
): { encoded: tf.SymbolicTensor; skip: tf.SymbolicTensor } {
  let x = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', useBias: false, name: `${name}_conv1` }).apply(input) as tf.SymbolicTensor;
  x = tf.layers.batchNormalization({ name: `${name}_bn1` }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.activation({ activation: 'relu', name: `${name}_relu1` }).apply(x) as tf.SymbolicTensor;

  x = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', useBias: false, name: `${name}_conv2` }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.batchNormalization({ name: `${name}_bn2` }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.activation({ activation: 'relu', name: `${name}_relu2` }).apply(x) as tf.SymbolicTensor;

  // Apply CBAM attention
  x = cbamBlock(x, 8, `${name}_cbam`);

  const skip = x; // Before pooling for skip connection
  const encoded = tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: `${name}_pool` }).apply(x) as tf.SymbolicTensor;

  return { encoded, skip };
}



/**
 * Build the complete Attention Net model
 * Returns both classification output and the spatial attention map
 */
export function buildAttentionNet(inputSize: [number, number] = [224, 224]): tf.LayersModel {
  const input = tf.input({ shape: [inputSize[0], inputSize[1], 3], name: 'attention_input' });

  // Encoder path with CBAM attention at each level
  const enc1 = encoderBlock(input, 64, 'enc1');
  const enc2 = encoderBlock(enc1.encoded, 128, 'enc2');
  const enc3 = encoderBlock(enc2.encoded, 256, 'enc3');
  const enc4 = encoderBlock(enc3.encoded, 512, 'enc4');

  // Bottleneck
  let bottleneck = tf.layers.conv2d({ filters: 1024, kernelSize: 3, padding: 'same', useBias: false, name: 'bottleneck_conv1' }).apply(enc4.encoded) as tf.SymbolicTensor;
  bottleneck = tf.layers.batchNormalization({ name: 'bottleneck_bn' }).apply(bottleneck) as tf.SymbolicTensor;
  bottleneck = tf.layers.activation({ activation: 'relu', name: 'bottleneck_relu' }).apply(bottleneck) as tf.SymbolicTensor;

  // Classification head (from bottleneck features)
  let classHead = tf.layers.globalAveragePooling2d({ name: 'class_gap' }).apply(bottleneck) as tf.SymbolicTensor;
  classHead = tf.layers.dense({ units: 256, activation: 'relu', name: 'class_fc1' }).apply(classHead) as tf.SymbolicTensor;
  classHead = tf.layers.dropout({ rate: 0.3, name: 'class_dropout' }).apply(classHead) as tf.SymbolicTensor;

  // Output: spatial attention summary (14x14 map flattened) + region classification
  const regionOutput = tf.layers.dense({
    units: Object.keys(BRAIN_REGIONS).length,
    activation: 'softmax',
    name: 'region_output',
  }).apply(classHead) as tf.SymbolicTensor;

  const model = tf.model({
    inputs: input,
    outputs: [regionOutput, enc4.skip], // enc4.skip is 14x14 after CBAM
    name: 'AttentionNet',
  });

  return model;
}

/**
 * Run Attention Net inference — produces real spatial attention data
 */
export async function runAttentionNetInference(
  model: tf.LayersModel,
  imageTensor: tf.Tensor
): Promise<AttentionNetOutput> {
  const startTime = performance.now();

  let input = imageTensor;
  if (input.shape.length === 3) {
    input = input.expandDims(0);
  }

  // Run model inference (multi-output)
  const [predictions, activations] = model.predict(input) as tf.Tensor[];
  const regionProbs = await predictions.data();
  
  // Extract attention from activations (average across channels for heatmap)
  const actData = activations.squeeze() as tf.Tensor;
  const heatmapTensor = actData.mean(-1); // [14, 14]
  
  // Normalize heatmap to 0-1
  const min = heatmapTensor.min();
  const max = heatmapTensor.max();
  const normalizedHeatmap = heatmapTensor.sub(min).div(max.sub(min).add(1e-5));
  const heatmapData = await normalizedHeatmap.array() as number[][];

  // Determine focus region (highest probability)
  const regionKeys = Object.keys(BRAIN_REGIONS) as BrainRegionKey[];
  let maxIdx = 0;
  let maxProb = 0;
  for (let i = 0; i < regionKeys.length; i++) {
    if (regionProbs[i] > maxProb) {
      maxProb = regionProbs[i];
      maxIdx = i;
    }
  }

  const focusRegion = regionKeys[maxIdx];
  const regionData = BRAIN_REGIONS[focusRegion];

  const inferenceTimeMs = performance.now() - startTime;
  
  // Cleanup
  predictions.dispose();
  activations.dispose();
  actData.dispose();
  heatmapTensor.dispose();
  normalizedHeatmap.dispose();

  return {
    modelName: 'Attention-Net',
    attentionMap: heatmapData,
    focusRegion,
    focusCenter: [...regionData.center] as [number, number, number],
    focusRadius: 0.4 + maxProb * 0.6,
    confidence: maxProb * 100,
    inferenceTimeMs,
  };
}
