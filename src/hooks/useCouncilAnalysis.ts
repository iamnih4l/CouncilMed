import { useState, useCallback, useRef } from 'react';
import { api } from '../services/api';
import type { AnalyzeResponse } from '../services/api';

export type AnalysisStatus =
  | 'idle'
  | 'uploading'
  | 'analyzing'
  | 'complete'
  | 'error';

export interface CouncilAnalysisState {
  status: AnalysisStatus;
  progress: number;
  result: AnalyzeResponse | null;
  error: string | null;
  elapsedMs: number;
}

export function useCouncilAnalysis() {
  const [state, setState] = useState<CouncilAnalysisState>({
    status: 'idle',
    progress: 0,
    result: null,
    error: null,
    elapsedMs: 0,
  });

  const startTimeRef = useRef<number>(0);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startProgressAnimation = useCallback(() => {
    startTimeRef.current = performance.now();
    let progress = 0;
    progressIntervalRef.current = setInterval(() => {
      const elapsed = performance.now() - startTimeRef.current;
      progress = 90 * (1 - Math.exp(-elapsed / 3000));
      setState(prev => ({
        ...prev,
        progress,
        elapsedMs: elapsed,
      }));
    }, 100);
  }, []);

  const stopProgressAnimation = useCallback(() => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  }, []);

  const analyzeFile = useCallback(async (
    file: File,
    options?: {
      modality?: string;
      anatomy?: string;
      enableExplainability?: boolean;
      enableAgent?: boolean;
    }
  ): Promise<AnalyzeResponse | null> => {
    setState({
      status: 'uploading',
      progress: 0,
      result: null,
      error: null,
      elapsedMs: 0,
    });

    startProgressAnimation();

    try {
      setState(prev => ({ ...prev, status: 'analyzing' }));

      const result = await api.analyzeUpload({
        file,
        modality: options?.modality,
        anatomy: options?.anatomy,
        enable_explainability: options?.enableExplainability ?? true,
        enable_agent: options?.enableAgent ?? true,
      });

      stopProgressAnimation();

      setState({
        status: 'complete',
        progress: 100,
        result,
        error: null,
        elapsedMs: result.total_time_ms,
      });

      return result;
    } catch (err) {
      stopProgressAnimation();
      const message = err instanceof Error ? err.message : 'Analysis failed';
      setState(prev => ({
        ...prev,
        status: 'error',
        error: message,
        progress: 0,
      }));
      return null;
    }
  }, [startProgressAnimation, stopProgressAnimation]);

  const analyzeByPath = useCallback(async (
    filePath: string,
    options?: {
      studyId?: string;
      modality?: string;
      anatomy?: string;
      enableExplainability?: boolean;
      enableAgent?: boolean;
    }
  ): Promise<AnalyzeResponse | null> => {
    setState({
      status: 'analyzing',
      progress: 0,
      result: null,
      error: null,
      elapsedMs: 0,
    });

    startProgressAnimation();

    try {
      const result = await api.analyzeByPath({
        file_path: filePath,
        study_id: options?.studyId,
        modality_override: options?.modality,
        anatomy_override: options?.anatomy,
        enable_explainability: options?.enableExplainability ?? true,
        enable_agent: options?.enableAgent ?? true,
      });

      stopProgressAnimation();

      setState({
        status: 'complete',
        progress: 100,
        result,
        error: null,
        elapsedMs: result.total_time_ms,
      });

      return result;
    } catch (err) {
      stopProgressAnimation();
      const message = err instanceof Error ? err.message : 'Analysis failed';
      setState(prev => ({
        ...prev,
        status: 'error',
        error: message,
        progress: 0,
      }));
      return null;
    }
  }, [startProgressAnimation, stopProgressAnimation]);

  const reset = useCallback(() => {
    stopProgressAnimation();
    setState({
      status: 'idle',
      progress: 0,
      result: null,
      error: null,
      elapsedMs: 0,
    });
  }, [stopProgressAnimation]);

  return {
    ...state,
    analyzeFile,
    analyzeByPath,
    reset,
    isProcessing: state.status === 'uploading' || state.status === 'analyzing',
    isComplete: state.status === 'complete',
    isError: state.status === 'error',
  };
}
