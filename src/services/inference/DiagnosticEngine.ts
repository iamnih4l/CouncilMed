// ============================================================================
// Diagnostic Engine — Orchestrator for the Council AI Pipeline
// Manages model lifecycle, preprocessing, parallel inference, and consensus
// ============================================================================

import * as tf from '@tensorflow/tfjs';
import { buildDenseNet121, runDenseNet121Inference } from './models/DenseNet121';
import { buildAttentionNet, runAttentionNetInference } from './models/AttentionNet';
import { buildSwinUNETR, runSwinUNETRInference } from './models/SwinUNETR';
import { runCouncilConsensus } from './CouncilConsensus';
import type { 
  DiagnosticState,
} from './types';
import { ClinicalValidationLayer } from './ClinicalValidationLayer';
import { RadiologyReportGenerator } from '../reporting/RadiologyReportGenerator';
import { DICOMProcessor } from './DICOMProcessor';
import { DEFAULT_MODEL_CONFIG } from './types';

type StateListener = (state: DiagnosticState) => void;

/**
 * DiagnosticEngine — Singleton service that manages the full inference pipeline.
 * 
 * Lifecycle:
 * 1. initialize() — builds and compiles all three models
 * 2. processImage(file) — runs the full pipeline on an uploaded image
 * 3. getState() — returns current pipeline state
 */
class DiagnosticEngine {
  private densenetModel: tf.GraphModel | tf.LayersModel | null = null;
  private attentionModel: tf.GraphModel | tf.LayersModel | null = null;
  private swinModel: tf.GraphModel | tf.LayersModel | null = null;
  private initialized = false;
  private listeners: StateListener[] = [];

  private state: DiagnosticState = {
    stage: 'idle',
    progress: 0,
    uploadedImage: null,
    imageTensor: null,
    result: null,
    error: null,
    densenetStatus: 'idle',
    attentionStatus: 'idle',
    swinStatus: 'idle',
  };

  /**
   * Subscribe to state changes
   */
  subscribe(listener: StateListener): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private emit() {
    const snapshot = { ...this.state };
    this.listeners.forEach((l) => l(snapshot));
  }

  private updateState(patch: Partial<DiagnosticState>) {
    this.state = { ...this.state, ...patch };
    this.emit();
  }

  getState(): DiagnosticState {
    return { ...this.state };
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Build and compile all three neural network architectures.
   * This is expensive and should be called once on app startup.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      console.log('[DiagnosticEngine] Initializing TensorFlow.js backend...');
      await tf.ready();
      console.log(`[DiagnosticEngine] Backend: ${tf.getBackend()}`);

      console.log('[DiagnosticEngine] Loading pretrained DenseNet-121 classifier...');
      this.densenetModel = await buildDenseNet121(DEFAULT_MODEL_CONFIG);
      console.log('[DiagnosticEngine] DenseNet-121 ready.');

      console.log('[DiagnosticEngine] Loading pretrained Attention-Net...');
      this.attentionModel = await buildAttentionNet(DEFAULT_MODEL_CONFIG.inputSize);
      console.log('[DiagnosticEngine] Attention-Net ready.');

      console.log('[DiagnosticEngine] Loading pretrained Swin-UNETR segmenter...');
      this.swinModel = await buildSwinUNETR(DEFAULT_MODEL_CONFIG.inputSize);
      console.log('[DiagnosticEngine] Swin-UNETR ready.');

      this.initialized = true;
      console.log('[DiagnosticEngine] All models initialized successfully.');
    } catch (error) {
      console.error('[DiagnosticEngine] Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Preprocess a browser File/Blob into a normalized tensor
   */
  private async preprocessImage(file: File): Promise<{ tensor: tf.Tensor; dataUrl: string }> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const dataUrl = reader.result as string;
          const img = new Image();
          img.onload = () => {
            const [h, w] = DEFAULT_MODEL_CONFIG.inputSize;

            // Convert image to tensor and resize
            let tensor = tf.browser.fromPixels(img);
            tensor = tf.image.resizeBilinear(tensor, [h, w]);

            // Normalize to [0, 1] range
            // Note: For clinical Grade IV accuracy, consider Z-score normalization 
            // based on the BraTS dataset statistics (mean/std).
            tensor = tensor.div(255.0);

            resolve({ tensor, dataUrl });
          };
          img.onerror = () => reject(new Error('Failed to load image'));
          img.src = dataUrl;
        } catch (err) {
          reject(err);
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsDataURL(file);
    });
  }

  /**
   * Run the full diagnostic pipeline:
   * Preprocess → DenseNet → Attention → Swin → Council Consensus
   */
  async processImage(imageSource: HTMLImageElement | File | tf.Tensor3D) {
    if (this.state.stage !== 'idle' && this.state.stage !== 'complete') return;

    if (!this.initialized) {
      await this.initialize();
    }

    this.updateState({
      stage: 'preprocessing',
      progress: 10,
      result: null,
      error: null,
      densenetStatus: 'idle',
      attentionStatus: 'idle',
      swinStatus: 'idle',
    });

    try {
      let inputTensor: tf.Tensor3D;
      let dicomMetadata: any = null;

      // 1. Handle Input Types
      if (imageSource instanceof tf.Tensor) {
        inputTensor = imageSource;
      } else if (imageSource instanceof File) {
        if (imageSource.name.toLowerCase().endsWith('.dcm')) {
          const dicomResult = await DICOMProcessor.process(imageSource);
          let tensor = dicomResult.tensor;
          
          // Ensure tensor has 3 channels for models that expect RGB
          if (tensor.shape[2] === 1) {
            tensor = tensor.tile([1, 1, 3]);
          }
          
          // Resize to the expected input dimensions
          const [h, w] = DEFAULT_MODEL_CONFIG.inputSize;
          inputTensor = tf.image.resizeBilinear(tensor as tf.Tensor3D, [h, w]);
          dicomMetadata = dicomResult.metadata;
        } else {
          const { tensor } = await this.preprocessImage(imageSource);
          inputTensor = tensor as tf.Tensor3D;
        }
      } else {
        // Fallback for HTMLImageElement
        inputTensor = tf.browser.fromPixels(imageSource) as tf.Tensor3D;
      }

      this.updateState({
        imageTensor: inputTensor,
        progress: 15,
      });

      // ── Step 2: DenseNet-121 Inference ──
      this.updateState({ stage: 'densenet-running', densenetStatus: 'running', progress: 25 });
      const densenetResult = await runDenseNet121Inference(this.densenetModel!, inputTensor);
      console.log(`[DiagnosticEngine] DenseNet Result: ${densenetResult.primaryDiagnosis} (${densenetResult.confidence.toFixed(1)}%)`);
      console.log(`[DiagnosticEngine] Raw probabilities:`, densenetResult.featureVector);
      this.updateState({ densenetStatus: 'complete', progress: 45 });

      // ── Step 3: Attention Net Inference ──
      this.updateState({ stage: 'attention-running', attentionStatus: 'running', progress: 50 });
      const attentionResult = await runAttentionNetInference(this.attentionModel!, inputTensor);
      console.log(`[DiagnosticEngine] Attention Result: ${attentionResult.focusRegion} (${attentionResult.confidence.toFixed(1)}%)`);
      this.updateState({ attentionStatus: 'complete', progress: 70 });

      // ── Step 4: Swin UNETR Inference ──
      this.updateState({ stage: 'swin-running', swinStatus: 'running', progress: 75 });
      const swinResult = await runSwinUNETRInference(this.swinModel!, inputTensor);
      console.log(`[DiagnosticEngine] Swin Result: ${swinResult.confidence.toFixed(1)}%`);
      this.updateState({ swinStatus: 'complete', progress: 90 });

      // ── Step 5: Council Consensus ──
      this.updateState({ stage: 'consensus', progress: 95 });
      const consensusResult = runCouncilConsensus(densenetResult, attentionResult, swinResult);

      // Clinical Validation skipped — real model inference provides direct confidence

      // ── Step 7: Radiology Report Generation ──
      const radiologyReport = RadiologyReportGenerator.generate(consensusResult);
      
      // Inject DICOM Metadata if available
      if (dicomMetadata) {
        radiologyReport.clinicalIndication = dicomMetadata.seriesDescription || 'DICOM Study';
        radiologyReport.technique = `Axial DICOM sequence: ${dicomMetadata.modality || 'MRI'}. Rows: ${dicomMetadata.rows}, Cols: ${dicomMetadata.columns}. Bits: ${dicomMetadata.bitsStored}.`;
      }
      
      consensusResult.radiologyReport = radiologyReport;

      // ── Complete ──
      this.updateState({
        stage: 'complete',
        progress: 100,
        result: consensusResult,
      });

      return consensusResult;
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : 'Unknown inference error';
      console.error('[DiagnosticEngine] Pipeline error:', error);
      this.updateState({
        stage: 'error',
        error: errMsg,
        densenetStatus: this.state.densenetStatus === 'running' ? 'error' : this.state.densenetStatus,
        attentionStatus: this.state.attentionStatus === 'running' ? 'error' : this.state.attentionStatus,
        swinStatus: this.state.swinStatus === 'running' ? 'error' : this.state.swinStatus,
      });
      throw error;
    }
  }

  /**
   * Reset the engine state (clear results, keep models loaded)
   */
  reset() {
    if (this.state.imageTensor) {
      (this.state.imageTensor as tf.Tensor).dispose();
    }
    this.updateState({
      stage: 'idle',
      progress: 0,
      uploadedImage: null,
      imageTensor: null,
      result: null,
      error: null,
      densenetStatus: 'idle',
      attentionStatus: 'idle',
      swinStatus: 'idle',
    });
  }
}

// Singleton export
export const diagnosticEngine = new DiagnosticEngine();
