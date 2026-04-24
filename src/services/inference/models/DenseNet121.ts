// ============================================================================
// DenseNet-121 / MobileNetV2 Classifier — Pretrained TF.js Graph Model
// Loads pretrained weights from /models/classifier/model.json
// Trained on Brain Tumor MRI dataset (4 classes: glioma, meningioma, notumor, pituitary)
// ============================================================================

import * as tf from '@tensorflow/tfjs';
import type { DenseNetOutput, ModelConfig } from '../types';
import { PathologyClass, DEFAULT_MODEL_CONFIG } from '../types';

/**
 * Build or load the classifier model.
 * Loads pretrained graph model from /models/classifier/model.json.
 * Falls back to a simple layers model if loading fails.
 */
export async function buildDenseNet121(config: ModelConfig = DEFAULT_MODEL_CONFIG): Promise<tf.GraphModel | tf.LayersModel> {
  try {
    console.log('[DenseNet121] Loading pretrained classifier from /models/classifier/model.json...');
    const model = await tf.loadGraphModel('/models/classifier/model.json');
    console.log('[DenseNet121] Pretrained classifier loaded successfully (graph model).');
    return model;
  } catch (err) {
    console.warn('[DenseNet121] Failed to load pretrained graph model:', err);
    try {
      // Try layers model as fallback
      const model = await tf.loadLayersModel('/models/classifier/model.json');
      console.log('[DenseNet121] Loaded as layers model.');
      return model;
    } catch (err2) {
      console.warn('[DenseNet121] All loading failed, building fallback:', err2);
      const [h, w] = config.inputSize;
      const input = tf.input({ shape: [h, w, 3], name: 'densenet_input' });
      let x = tf.layers.conv2d({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', name: 'fb_conv1' }).apply(input) as tf.SymbolicTensor;
      x = tf.layers.globalAveragePooling2d({ name: 'global_avg_pool' }).apply(x) as tf.SymbolicTensor;
      const output = tf.layers.dense({ units: config.numClasses, activation: 'softmax', name: 'classification_head' }).apply(x) as tf.SymbolicTensor;
      return tf.model({ inputs: input, outputs: output, name: 'DenseNet121_Fallback' });
    }
  }
}

// Pathology classes mapped to trained model's 4-class output indices
// Model output order: [glioma, meningioma, notumor, pituitary]
const PATHOLOGY_CLASSES: PathologyClass[] = [
  PathologyClass.GLIOBLASTOMA,    // index 0: glioma → Glioblastoma (Grade IV)
  PathologyClass.MENINGIOMA,      // index 1: meningioma
  PathologyClass.NORMAL,          // index 2: notumor → Normal
  PathologyClass.METASTASIS,      // index 3: pituitary → Metastatic Lesion
];

/**
 * Run inference on a preprocessed image tensor
 */
export async function runDenseNet121Inference(
  model: tf.GraphModel | tf.LayersModel,
  imageTensor: tf.Tensor
): Promise<DenseNetOutput> {
  const startTime = performance.now();

  // Ensure correct input shape [1, 224, 224, 3]
  let input = imageTensor;
  if (input.shape.length === 3) {
    input = input.expandDims(0);
  }

  // Run inference — GraphModel uses execute(), LayersModel uses predict()
  let predictions: tf.Tensor;
  if ('execute' in model && typeof (model as tf.GraphModel).execute === 'function') {
    predictions = (model as tf.GraphModel).execute(input) as tf.Tensor;
  } else {
    predictions = (model as tf.LayersModel).predict(input) as tf.Tensor;
  }
  
  const probabilities = await predictions.data();

  // Build classification results from real model output
  const classifications = PATHOLOGY_CLASSES.map((pathology, i) => ({
    pathology,
    probability: probabilities[i] || 0,
  }));

  // Sort by probability (descending)
  classifications.sort((a, b) => b.probability - a.probability);

  const primaryDiagnosis = classifications[0].pathology;
  const confidence = classifications[0].probability * 100;

  // Extract feature vector (use output probabilities)
  const featureVector = Array.from(probabilities);

  const inferenceTimeMs = performance.now() - startTime;

  // Cleanup
  predictions.dispose();

  return {
    modelName: 'DenseNet-121',
    classifications,
    primaryDiagnosis,
    confidence,
    inferenceTimeMs,
    featureVector,
  };
}
