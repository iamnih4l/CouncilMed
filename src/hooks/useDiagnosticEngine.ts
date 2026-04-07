// ============================================================================
// useDiagnosticEngine — React hook for the Council AI diagnostic pipeline
// ============================================================================

import { useState, useEffect, useCallback } from 'react';
import { diagnosticEngine } from '../services/inference/DiagnosticEngine';
import type { DiagnosticState, CouncilConsensusResult } from '../services/inference/types';

interface UseDiagnosticEngineReturn {
  state: DiagnosticState;
  isInitialized: boolean;
  isProcessing: boolean;
  result: CouncilConsensusResult | null;
  processImage: (file: File) => Promise<CouncilConsensusResult | void>;
  reset: () => void;
  initialize: () => Promise<void>;
}

export function useDiagnosticEngine(): UseDiagnosticEngineReturn {
  const [state, setState] = useState<DiagnosticState>(diagnosticEngine.getState());
  const [isInitialized, setIsInitialized] = useState(diagnosticEngine.isInitialized());

  useEffect(() => {
    const unsubscribe = diagnosticEngine.subscribe((newState) => {
      setState(newState);
    });
    return unsubscribe;
  }, []);

  const initialize = useCallback(async () => {
    try {
      await diagnosticEngine.initialize();
      setIsInitialized(true);
    } catch (err) {
      console.error('Failed to initialize diagnostic engine:', err);
    }
  }, []);

  const processImage = useCallback(async (file: File) => {
    try {
      return await diagnosticEngine.processImage(file);
    } catch (err) {
      console.error('Diagnostic pipeline failed:', err);
    }
  }, []);

  const reset = useCallback(() => {
    diagnosticEngine.reset();
  }, []);

  const isProcessing = state.stage !== 'idle' && state.stage !== 'complete' && state.stage !== 'error';

  return {
    state,
    isInitialized,
    isProcessing,
    result: state.result,
    processImage,
    reset,
    initialize,
  };
}
