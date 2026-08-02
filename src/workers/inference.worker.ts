import * as tf from '@tensorflow/tfjs';
import { buildDenseNet121, runDenseNet121Inference } from '../services/inference/models/DenseNet121';
import { buildAttentionNet, runAttentionNetInference } from '../services/inference/models/AttentionNet';
import { buildSwinUNETR, runSwinUNETRInference } from '../services/inference/models/SwinUNETR';
import { runAgenticCouncilConsensus } from '../services/inference/CouncilConsensus';
import { DEFAULT_MODEL_CONFIG } from '../services/inference/types';
import type { Modality, CouncilConsensusResult } from '../services/inference/types';

let densenetModel: tf.GraphModel | tf.LayersModel | null = null;
let attentionModel: tf.GraphModel | tf.LayersModel | null = null;
let swinModel: tf.GraphModel | tf.LayersModel | null = null;
let initialized = false;

// Custom message types
export type WorkerMessage = 
  | { type: 'INIT' }
  | { type: 'PROCESS'; file: File; modality: Modality };

export type WorkerResponse = 
  | { type: 'READY' }
  | { type: 'PROGRESS'; stage: string; progress: number; statusUpdate?: Partial<Record<string, string>> }
  | { type: 'COMPLETE'; result: CouncilConsensusResult; dataUrl: string }
  | { type: 'ERROR'; error: string; stage?: string };

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  const msg = e.data;

  try {
    if (msg.type === 'INIT') {
      if (!initialized) {
        await tf.ready();
        densenetModel = await buildDenseNet121(DEFAULT_MODEL_CONFIG);
        attentionModel = await buildAttentionNet(DEFAULT_MODEL_CONFIG.inputSize);
        swinModel = await buildSwinUNETR(DEFAULT_MODEL_CONFIG.inputSize);
        initialized = true;
      }
      self.postMessage({ type: 'READY' });
    } 
    else if (msg.type === 'PROCESS') {
      if (!initialized) {
        throw new Error('Worker not initialized');
      }

      const { file, modality } = msg;

      self.postMessage({ type: 'PROGRESS', stage: 'preprocessing', progress: 5 });

      // Preprocessing in Web Worker
      const imageBitmap = await createImageBitmap(file);
      const [h, w] = DEFAULT_MODEL_CONFIG.inputSize;
      
      let tensor = tf.browser.fromPixels(imageBitmap);
      tensor = tf.image.resizeBilinear(tensor, [h, w]);
      tensor = tensor.div(255.0);
      
      // Convert to dataUrl for UI to display (since UI doesn't have the original data if we don't pass it back)
      // Actually we could just create an objectURL in the main thread.
      // But we'll do standard canvas data extraction in worker to be thorough if needed.
      // For now, let main thread handle objectURL to avoid passing massive strings back.
      self.postMessage({ type: 'PROGRESS', stage: 'preprocessing', progress: 15 });

      // DenseNet
      self.postMessage({ type: 'PROGRESS', stage: 'densenet-running', progress: 25, statusUpdate: { densenetStatus: 'running' } });
      const densenetResult = await runDenseNet121Inference(densenetModel!, tensor, modality);
      self.postMessage({ type: 'PROGRESS', stage: 'densenet-running', progress: 45, statusUpdate: { densenetStatus: 'complete' } });

      // AttentionNet
      self.postMessage({ type: 'PROGRESS', stage: 'attention-running', progress: 50, statusUpdate: { attentionStatus: 'running' } });
      const attentionResult = await runAttentionNetInference(attentionModel!, tensor, modality);
      self.postMessage({ type: 'PROGRESS', stage: 'attention-running', progress: 70, statusUpdate: { attentionStatus: 'complete' } });

      // SwinUNETR
      self.postMessage({ type: 'PROGRESS', stage: 'swin-running', progress: 75, statusUpdate: { swinStatus: 'running' } });
      const swinResult = await runSwinUNETRInference(swinModel!, tensor, modality);
      self.postMessage({ type: 'PROGRESS', stage: 'swin-running', progress: 90, statusUpdate: { swinStatus: 'complete' } });

      // Consensus
      self.postMessage({ type: 'PROGRESS', stage: 'consensus', progress: 95 });
      const consensusResult = await runAgenticCouncilConsensus(densenetResult, attentionResult, swinResult, modality);

      tensor.dispose();
      imageBitmap.close();

      self.postMessage({ 
        type: 'COMPLETE', 
        result: consensusResult 
      });
    }
  } catch (err) {
    self.postMessage({ 
      type: 'ERROR', 
      error: err instanceof Error ? err.message : String(err) 
    });
  }
};
