import { useState, useEffect, useCallback } from 'react';
import type { CouncilConsensusResult } from '../services/inference/types';

export interface PersistentReport {
  id: string;
  patientName?: string;
  modality: 'MRI' | 'CT' | 'X-Ray';
  date: string;
  result: CouncilConsensusResult;
}

const STORAGE_KEY = 'council_med_reports_vault';

export function usePersistentReports() {
  const [reports, setReports] = useState<PersistentReport[]>([]);

  // Load reports on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setReports(JSON.parse(stored));
      } catch (err) {
        console.error('Failed to parse diagnostic vault:', err);
      }
    }
  }, []);

  const addReport = useCallback((result: CouncilConsensusResult) => {
    const newReport: PersistentReport = {
      id: `PT-${Math.floor(Math.random() * 9000) + 1000}`,
      modality: 'MRI', // Default for now, can be parameterized
      date: new Date().toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      }),
      result,
    };

    setReports((prev) => {
      const updated = [newReport, ...prev].slice(0, 50); // Cap at 50 for local storage efficiency
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
    
    return newReport;
  }, []);

  const deleteReport = useCallback((id: string) => {
    setReports((prev) => {
      const updated = prev.filter(r => r.id !== id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
  }, []);

  const clearAll = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setReports([]);
  }, []);

  return { reports, addReport, deleteReport, clearAll };
}
