// ============================================================================
// Diagnostic Engine — Orchestrator for the Council AI Pipeline
// Now delegates all inference to the Python backend via /api/inference.
// The frontend is purely a presentation layer — no TF.js, no heuristics.
// ============================================================================

import type {
  CouncilConsensusResult,
  DiagnosticState,
  Modality
} from './types';

type StateListener = (state: DiagnosticState) => void;

/**
 * DiagnosticEngine — Singleton service that manages the inference pipeline.
 * Sends images to the backend for real PyTorch inference.
 */
class DiagnosticEngine {
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
    // Backend is always "initialized" — no model loading needed on frontend
    return true;
  }

  async initialize(): Promise<void> {
    // No-op: backend handles model loading
    return;
  }

  /**
   * Run the full diagnostic pipeline via backend API
   */
  async processImage(file: File, modality: Modality = 'MRI'): Promise<CouncilConsensusResult> {
    // Reset state
    this.updateState({
      stage: 'preprocessing',
      progress: 5,
      result: null,
      error: null,
      densenetStatus: 'idle',
      attentionStatus: 'idle',
      swinStatus: 'idle',
    });

    // Generate data URL for the UI to display the uploaded image
    const dataUrl = await this.readAsDataURL(file);
    this.updateState({ uploadedImage: dataUrl });

    // Simulate pipeline progress while the backend processes
    this.updateState({ stage: 'preprocessing', progress: 10 });

    // Build multipart form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('modality', modality);

    // Show progress stages to keep the UI alive
    this.updateState({ stage: 'densenet-running', progress: 25, densenetStatus: 'running' });

    try {
      const response = await fetch('http://localhost:8000/api/inference', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Backend inference failed: ${errorText}`);
      }

      // Simulate progress through the pipeline stages
      this.updateState({ stage: 'densenet-running', progress: 45, densenetStatus: 'complete' });
      this.updateState({ stage: 'attention-running', progress: 60, attentionStatus: 'running' });

      const result: CouncilConsensusResult = await response.json();

      this.updateState({ attentionStatus: 'complete' });
      this.updateState({ stage: 'swin-running', progress: 80, swinStatus: 'running' });
      this.updateState({ swinStatus: 'complete' });
      this.updateState({ stage: 'consensus', progress: 95 });

      this.updateState({
        stage: 'complete',
        progress: 100,
        result,
      });

      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      console.error('[DiagnosticEngine] Backend inference error:', errorMessage);
      this.updateState({
        stage: 'error',
        error: errorMessage,
        densenetStatus: this.state.densenetStatus === 'running' ? 'error' : this.state.densenetStatus,
        attentionStatus: this.state.attentionStatus === 'running' ? 'error' : this.state.attentionStatus,
        swinStatus: this.state.swinStatus === 'running' ? 'error' : this.state.swinStatus,
      });
      throw err;
    }
  }

  private readAsDataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  reset() {
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
