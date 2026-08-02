// ============================================================================
// Diagnostic Engine — Orchestrator for the Council AI Pipeline
// Manages model lifecycle, preprocessing, parallel inference, and consensus
// ============================================================================

import type {
  CouncilConsensusResult,
  DiagnosticState,
  Modality
} from './types';

type StateListener = (state: DiagnosticState) => void;

/**
 * DiagnosticEngine — Singleton service that manages the full inference pipeline.
 * Now acts as a proxy to the background Web Worker to keep the UI thread unblocked.
 */
class DiagnosticEngine {
  private worker: Worker | null = null;
  private initialized = false;
  private listeners: StateListener[] = [];
  
  // Keep track of the current Promise resolve/reject to finish processImage
  private currentResolver: ((result: CouncilConsensusResult) => void) | null = null;
  private currentRejecter: ((error: Error) => void) | null = null;

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
    return this.initialized;
  }

  /**
   * Initialize the Web Worker.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    return new Promise((resolve, reject) => {
      try {
        console.log('[DiagnosticEngine] Initializing Web Worker...');
        this.worker = new Worker(new URL('../../workers/inference.worker.ts', import.meta.url), { type: 'module' });
        
        this.worker.onmessage = (e) => {
          this.handleWorkerMessage(e.data);
        };
        
        this.worker.onerror = (e) => {
          console.error('[DiagnosticEngine] Worker error:', e);
          reject(e);
        };

        // We listen for READY to resolve
        const readyListener = (e: MessageEvent) => {
          if (e.data.type === 'READY') {
            this.initialized = true;
            console.log('[DiagnosticEngine] Worker initialized successfully.');
            this.worker?.removeEventListener('message', readyListener);
            resolve();
          }
        };
        this.worker.addEventListener('message', readyListener);

        this.worker.postMessage({ type: 'INIT' });
      } catch (error) {
        console.error('[DiagnosticEngine] Initialization failed:', error);
        reject(error);
      }
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private handleWorkerMessage(msg: any) {
    switch (msg.type) {
      case 'PROGRESS':
        this.updateState({
          stage: msg.stage,
          progress: msg.progress,
          ...(msg.statusUpdate || {})
        });
        break;

      case 'COMPLETE':
        this.updateState({
          stage: 'complete',
          progress: 100,
          result: msg.result,
        });
        if (this.currentResolver) {
          this.currentResolver(msg.result);
          this.currentResolver = null;
          this.currentRejecter = null;
        }
        break;

      case 'ERROR':
        console.error('[DiagnosticEngine] Pipeline error:', msg.error);
        this.updateState({
          stage: 'error',
          error: msg.error,
          densenetStatus: this.state.densenetStatus === 'running' ? 'error' : this.state.densenetStatus,
          attentionStatus: this.state.attentionStatus === 'running' ? 'error' : this.state.attentionStatus,
          swinStatus: this.state.swinStatus === 'running' ? 'error' : this.state.swinStatus,
        });
        if (this.currentRejecter) {
          this.currentRejecter(new Error(msg.error));
          this.currentResolver = null;
          this.currentRejecter = null;
        }
        break;
    }
  }

  /**
   * Run the full diagnostic pipeline via Web Worker
   */
  async processImage(file: File, modality: Modality = 'MRI'): Promise<CouncilConsensusResult> {
    if (!this.initialized || !this.worker) {
      await this.initialize();
    }

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

    // We can generate a quick data URL on the main thread for the UI to display the image
    const dataUrl = await this.readAsDataURL(file);
    this.updateState({ uploadedImage: dataUrl });

    return new Promise((resolve, reject) => {
      this.currentResolver = resolve;
      this.currentRejecter = reject;

      this.worker!.postMessage({
        type: 'PROCESS',
        file,
        modality
      });
    });
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
