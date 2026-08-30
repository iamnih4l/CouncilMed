/**
 * CouncilMed — Settings Page
 * ============================
 * System configuration: backend URL, agent settings,
 * model paths, and knowledge base management.
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Settings, Server, Brain, Database, Cpu,
  Shield,  CheckCircle,
  XCircle, Loader2, Globe,
} from 'lucide-react';
import { GlassCard } from '../components/ui/GlassCard';
import { api } from '../services/api';

interface SettingSection {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

function SettingsSection({ title, icon, children }: SettingSection) {
  return (
    <GlassCard>
      <div className="flex items-center gap-2 mb-4">
        {icon}
        <h3 className="text-sm font-semibold text-white">{title}</h3>
      </div>
      {children}
    </GlassCard>
  );
}

function SettingsRow({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-white/5 last:border-b-0">
      <div>
        <p className="text-sm text-zinc-300">{label}</p>
        {hint && <p className="text-[10px] text-zinc-600 mt-0.5">{hint}</p>}
      </div>
      <div className="text-sm text-zinc-400 font-mono">{value}</div>
    </div>
  );
}

export default function SettingsPage() {
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [healthInfo, setHealthInfo] = useState<{
    version: string;
    device: string;
    models: number;
  } | null>(null);

  useEffect(() => {
    async function checkBackend() {
      try {
        const health = await api.health();
        setBackendStatus('online');
        setHealthInfo({
          version: health.version,
          device: health.device,
          models: health.models_loaded,
        });
      } catch {
        setBackendStatus('offline');
      }
    }
    checkBackend();
  }, []);

  return (
    <motion.div
      className="max-w-3xl mx-auto space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div>
        <h1 className="text-2xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-400">
          System Settings
        </h1>
        <p className="text-zinc-500 text-sm mt-0.5">Configure CouncilMed workstation</p>
      </div>

      {/* Backend Connection */}
      <SettingsSection
        title="Backend Connection"
        icon={<Server className="w-4 h-4 text-cyan-400" />}
      >
        <SettingsRow
          label="API URL"
          value={<span className="text-cyan-400">http://localhost:8000</span>}
          hint="Python FastAPI backend"
        />
        <SettingsRow
          label="Status"
          value={
            backendStatus === 'checking' ? (
              <span className="flex items-center gap-1 text-zinc-400">
                <Loader2 className="w-3 h-3 animate-spin" /> Checking...
              </span>
            ) : backendStatus === 'online' ? (
              <span className="flex items-center gap-1 text-emerald-400">
                <CheckCircle className="w-3 h-3" /> Online
              </span>
            ) : (
              <span className="flex items-center gap-1 text-red-400">
                <XCircle className="w-3 h-3" /> Offline
              </span>
            )
          }
        />
        {healthInfo && (
          <>
            <SettingsRow label="API Version" value={`v${healthInfo.version}`} />
            <SettingsRow label="Inference Device" value={healthInfo.device} />
            <SettingsRow label="Models Loaded" value={healthInfo.models.toString()} />
          </>
        )}
      </SettingsSection>

      {/* Council Configuration */}
      <SettingsSection
        title="Council Configuration"
        icon={<Brain className="w-4 h-4 text-purple-400" />}
      >
        <SettingsRow
          label="Fusion Method"
          value="Uncertainty-Weighted Ensemble"
          hint="Calibrated entropy-based model weighting"
        />
        <SettingsRow
          label="Minimum Council Size"
          value="1 model"
          hint="Recommended: 3+ for robust consensus"
        />
        <SettingsRow
          label="Human Review"
          value={<span className="text-emerald-400">Always Required</span>}
          hint="All analyses require clinician review"
        />
        <SettingsRow
          label="Consensus Threshold"
          value="75% agreement"
          hint="Below this, consensus is marked as weak"
        />
      </SettingsSection>

      {/* Agent Configuration */}
      <SettingsSection
        title="AI Agent"
        icon={<Globe className="w-4 h-4 text-amber-400" />}
      >
        <SettingsRow
          label="LLM Provider"
          value="Ollama (Local)"
          hint="No data leaves your machine"
        />
        <SettingsRow
          label="Model"
          value="llama3:8b-instruct-fp16"
          hint="Structured JSON output via system prompt"
        />
        <SettingsRow
          label="Ollama URL"
          value={<span className="text-cyan-400">http://localhost:11434</span>}
        />
        <SettingsRow
          label="Output Schema"
          value="INSIGHT_SCHEMA (structured)"
          hint="Forced conservative, evidence-traceable reasoning"
        />
      </SettingsSection>

      {/* Storage */}
      <SettingsSection
        title="Data & Storage"
        icon={<Database className="w-4 h-4 text-emerald-400" />}
      >
        <SettingsRow
          label="Database"
          value="SQLite (local)"
          hint="data/councilmed.db"
        />
        <SettingsRow
          label="Model Registry"
          value="YAML manifests"
          hint="model_registry/ directory"
        />
        <SettingsRow
          label="Knowledge Base"
          value="ChromaDB (local)"
          hint="data/knowledge_db/"
        />
        <SettingsRow
          label="Explainability Output"
          value="data/gradcam/"
          hint="Grad-CAM heatmaps stored here"
        />
      </SettingsSection>

      {/* Privacy */}
      <SettingsSection
        title="Privacy & Security"
        icon={<Shield className="w-4 h-4 text-red-400" />}
      >
        <SettingsRow
          label="Data Residency"
          value={<span className="text-emerald-400">100% Local</span>}
          hint="No data leaves your machine"
        />
        <SettingsRow
          label="CORS Policy"
          value="localhost only"
          hint="Restricted to local development origins"
        />
        <SettingsRow
          label="Audit Trail"
          value={<span className="text-emerald-400">Enabled</span>}
          hint="Every analysis, review, and action is logged"
        />
        <SettingsRow
          label="Patient Data"
          value="De-identified only"
          hint="DICOM metadata stripped of PHI before processing"
        />
      </SettingsSection>

      {/* Version */}
      <div className="text-center py-4 text-xs text-zinc-700">
        <p>CouncilMed v0.1.0 • Local-First Medical AI Workstation</p>
        <p className="mt-1">Evidence-based consensus through multi-model inference</p>
      </div>
    </motion.div>
  );
}
