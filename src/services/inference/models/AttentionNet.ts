// ============================================================================
// Attention Net — Pretrained TF.js Graph Model
// Loads pretrained weights from /models/attention/model.json
// Produces spatial attention heatmap + brain region classification
// ============================================================================

import * as tf from '@tensorflow/tfjs';
import type { AttentionNetOutput, BrainRegionKey } from '../types';
import { BRAIN_REGIONS } from '../types';

/**
 * Load the pretrained AttentionNet model.
 */
export async function buildAttentionNet(inputSize: [number, number] = [224, 224]): Promise<tf.GraphModel | tf.LayersModel> {
  try {
    console.log('[AttentionNet] Loading pretrained attention model from /models/attention/model.json...');
    const model = await tf.loadGraphModel('/models/attention/model.json');
    console.log('[AttentionNet] Pretrained attention model loaded (graph model).');
    return model;
  } catch (err) {
    console.warn('[AttentionNet] Failed to load pretrained model:', err);
    // Build minimal fallback
    const input = tf.input({ shape: [inputSize[0], inputSize[1], 3], name: 'attention_input' });
    let x = tf.layers.conv2d({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', name: 'fb_conv1' }).apply(input) as tf.SymbolicTensor;
    x = tf.layers.globalAveragePooling2d({ name: 'class_gap' }).apply(x) as tf.SymbolicTensor;
    const regionOutput = tf.layers.dense({
      units: Object.keys(BRAIN_REGIONS).length,
      activation: 'softmax',
      name: 'region_output',
    }).apply(x) as tf.SymbolicTensor;
    return tf.model({ inputs: input, outputs: regionOutput, name: 'AttentionNet_Fallback' });
  }
}

/**
 * Run Attention Net inference — produces spatial attention data
 */
export async function runAttentionNetInference(
  model: tf.GraphModel | tf.LayersModel,
  imageTensor: tf.Tensor
): Promise<AttentionNetOutput> {
  const startTime = performance.now();

  let input = imageTensor;
  if (input.shape.length === 3) {
    input = input.expandDims(0);
  }

  // Run inference
  let regionProbs: Float32Array | Int32Array | Uint8Array;
  let heatmapData: number[][] = [];

  if ('execute' in model && typeof (model as tf.GraphModel).execute === 'function') {
    // Graph model — returns tensor or array of tensors
    const result = (model as tf.GraphModel).execute(input);
    
    if (Array.isArray(result)) {
      // Multi-output: [region_probs, attention_flat]
      regionProbs = await result[0].data();
      const flatData = await result[1].data();
      
      // Build 7x7 heatmap from flat data
      if (flatData.length >= 49) {
        for (let r = 0; r < 7; r++) {
          const row: number[] = [];
          for (let c = 0; c < 7; c++) {
            row.push(flatData[r * 7 + c]);
          }
          heatmapData.push(row);
        }
      }
      result.forEach(t => t.dispose());
    } else {
      regionProbs = await result.data();
      // Generate default heatmap
      for (let r = 0; r < 7; r++) {
        heatmapData.push(new Array(7).fill(0.5));
      }
      result.dispose();
    }
  } else {
    // Layers model fallback
    const result = (model as tf.LayersModel).predict(input) as tf.Tensor;
    regionProbs = await result.data();
    for (let r = 0; r < 7; r++) {
      heatmapData.push(new Array(7).fill(0.5));
    }
    result.dispose();
  }

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
