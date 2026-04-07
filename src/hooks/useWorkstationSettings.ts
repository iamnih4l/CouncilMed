import { useState, useEffect, useCallback } from 'react';

export interface WorkstationSettings {
  doctorName: string;
  specialization: string;
  clinicalId: string;
  darkMode: boolean;
  aiSensitivity: number; // 0.1 to 0.9 — used for classification thresholds
  pacsConnected: boolean;
}

const SETTINGS_KEY = 'council_med_workstation_settings';

const DEFAULT_SETTINGS: WorkstationSettings = {
  doctorName: 'Dr. Sarah Jenkins',
  specialization: 'Chief Radiologist',
  clinicalId: 'SR-8842-MD',
  darkMode: true,
  aiSensitivity: 0.85,
  pacsConnected: true,
};

export function useWorkstationSettings() {
  const [settings, setSettings] = useState<WorkstationSettings>(DEFAULT_SETTINGS);

  useEffect(() => {
    const stored = localStorage.getItem(SETTINGS_KEY);
    if (stored) {
      try {
        setSettings(JSON.parse(stored));
      } catch (err) {
        console.error('Failed to load clinical workstation settings:', err);
      }
    }
  }, []);

  const updateSettings = useCallback((patch: Partial<WorkstationSettings>) => {
    setSettings((prev) => {
      const updated = { ...prev, ...patch };
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(updated));
      return updated;
    });
  }, []);

  const resetSettings = useCallback(() => {
    localStorage.removeItem(SETTINGS_KEY);
    setSettings(DEFAULT_SETTINGS);
  }, []);

  return { settings, updateSettings, resetSettings };
}
