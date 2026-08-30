/**
 * CouncilMed — Model Registry Page
 * ====================================
 * Shows all registered models, their architecture, validation metrics,
 * and installation status. Also displays hardware info.
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Cpu, Brain, ShieldCheck, ShieldX, Download,
  CheckCircle, XCircle, HardDrive, Gauge, Info,
} from 'lucide-react';
import { GlassCard } from '../components/ui/GlassCard';
import { api } from '../services/api';
import type { RegistryResponse, HealthResponse, ModelInfo } from '../services/api';

function ModelCard({ model, index }: { model: ModelInfo; index: number }) {
  const frameworkColors: Record<string, string> = {
    pytorch: 'text-orange-400 bg-orange-500/10 border-orange-500/20',
    tensorflow: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    onnx: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
    monai: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
  };

  const fwStyle = frameworkColors[model.framework] || 'text-zinc-400 bg-zinc-500/10 border-zinc-500/20';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08 }}
    >
      <GlassCard className="hover:border-white/20 transition-all duration-300">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-teal-500/20
              flex items-center justify-center border border-cyan-500/20">
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">{model.name}</h3>
              <p className="text-[11px] text-zinc-500">v{model.version} • {model.architecture_family}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {model.installed ? (
              <span className="flex items-center gap-1 text-[11px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                <CheckCircle className="w-3 h-3" /> Installed
              </span>
            ) : (
              <span className="flex items-center gap-1 text-[11px] text-zinc-500 bg-zinc-500/10 px-2 py-0.5 rounded-full border border-zinc-500/20">
                <Download className="w-3 h-3" /> Not Installed
              </span>
            )}
          </div>
        </div>

        {/* Tags */}
        <div className="flex items-center gap-2 mb-3 flex-wrap">
          <span className={`text-[10px] px-2 py-0.5 rounded-full border ${fwStyle}`}>
            {model.framework}
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-zinc-400">
            {model.modality}
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-zinc-400">
            {model.anatomy}
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-zinc-400">
            {model.num_classes} classes
          </span>
        </div>

        {/* Classes */}
        <div className="mb-3">
          <p className="text-[11px] text-zinc-500 mb-1.5">Output Classes</p>
          <div className="flex flex-wrap gap-1">
            {model.class_names.map((cls) => (
              <span key={cls} className="text-[10px] px-1.5 py-0.5 rounded bg-white/[0.03] text-zinc-400 border border-white/5">
                {cls}
              </span>
            ))}
          </div>
        </div>

        {/* Validation metrics */}
        <div className="flex items-center gap-4 pt-3 border-t border-white/5">
          {model.validated ? (
            <>
              <div className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-[11px] text-emerald-400 font-medium">Validated</span>
              </div>
              {model.accuracy !== null && (
                <div className="text-[11px] text-zinc-400">
                  Accuracy: <span className="text-white font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                </div>
              )}
              {model.auroc !== null && (
                <div className="text-[11px] text-zinc-400">
                  AUROC: <span className="text-white font-medium">{(model.auroc * 100).toFixed(1)}%</span>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center gap-1">
              <ShieldX className="w-3.5 h-3.5 text-zinc-600" />
              <span className="text-[11px] text-zinc-600">Not validated</span>
            </div>
          )}
        </div>
      </GlassCard>
    </motion.div>
  );
}

export default function ModelRegistryPage() {
  const [registry, setRegistry] = useState<RegistryResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [regData, healthData] = await Promise.all([
          api.getModels(),
          api.health(),
        ]);
        setRegistry(regData);
        setHealth(healthData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to connect to backend');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center">
          <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-zinc-500 mt-4">Connecting to backend...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-lg mx-auto mt-16">
        <GlassCard className="border-red-500/20 bg-red-500/5">
          <div className="flex items-start gap-3">
            <XCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <div>
              <p className="text-sm font-semibold text-red-400">Backend Connection Failed</p>
              <p className="text-xs text-zinc-400 mt-1">{error}</p>
              <p className="text-xs text-zinc-500 mt-3">
                Make sure the Python backend is running:<br />
                <code className="text-cyan-400 bg-white/5 px-1 py-0.5 rounded text-[11px]">
                  uvicorn backend.api.app:app --reload
                </code>
              </p>
            </div>
          </div>
        </GlassCard>
      </div>
    );
  }

  return (
    <motion.div
      className="max-w-6xl mx-auto space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div>
        <h1 className="text-2xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-400">
          Model Registry
        </h1>
        <p className="text-zinc-500 text-sm mt-0.5">Council member models and hardware configuration</p>
      </div>

      {/* Hardware & System Info */}
      {health && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <GlassCard>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                <Cpu className="w-4 h-4 text-cyan-400" />
              </div>
              <div>
                <p className="text-[11px] text-zinc-500">Device</p>
                <p className="text-sm font-medium text-white">{health.device}</p>
              </div>
            </div>
          </GlassCard>

          <GlassCard>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                <Brain className="w-4 h-4 text-emerald-400" />
              </div>
              <div>
                <p className="text-[11px] text-zinc-500">Models Loaded</p>
                <p className="text-sm font-medium text-white">{health.models_loaded}</p>
              </div>
            </div>
          </GlassCard>

          <GlassCard>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10 border border-purple-500/20">
                <HardDrive className="w-4 h-4 text-purple-400" />
              </div>
              <div>
                <p className="text-[11px] text-zinc-500">Status</p>
                <p className="text-sm font-medium text-emerald-400">{health.status}</p>
              </div>
            </div>
          </GlassCard>

          <GlassCard>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                <Gauge className="w-4 h-4 text-amber-400" />
              </div>
              <div>
                <p className="text-[11px] text-zinc-500">API Version</p>
                <p className="text-sm font-medium text-white">v{health.version}</p>
              </div>
            </div>
          </GlassCard>
        </div>
      )}

      {/* Registry Summary */}
      {registry && (
        <div className="flex items-center gap-4 text-sm text-zinc-400">
          <span>{registry.total_models} total models</span>
          <span className="text-zinc-700">•</span>
          <span>{registry.installed} installed</span>
          <span className="text-zinc-700">•</span>
          <span>{registry.validated} validated</span>
          <span className="text-zinc-700">•</span>
          <span>Modalities: {registry.modalities.join(', ') || 'none'}</span>
        </div>
      )}

      {/* Model Cards */}
      {registry && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {registry.models.map((model, index) => (
            <ModelCard key={model.model_id} model={model} index={index} />
          ))}
        </div>
      )}

      {registry && registry.models.length === 0 && (
        <GlassCard className="flex flex-col items-center justify-center py-16">
          <Info className="w-10 h-10 text-zinc-700 mb-4" />
          <p className="text-sm text-zinc-500 font-medium">No models registered</p>
          <p className="text-xs text-zinc-600 mt-1">Add YAML manifests to model_registry/ directory</p>
        </GlassCard>
      )}
    </motion.div>
  );
}
