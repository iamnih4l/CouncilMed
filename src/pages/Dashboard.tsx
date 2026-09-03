/**
 * CouncilMed — Dashboard (Rebuilt)
 * ===================================
 * System overview connected to the real backend.
 * 
 * Replaces hardcoded stats (1284 scans, 347 anomalies, 42 radiologists)
 * with live data from the backend API.
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { GlassCard } from '../components/ui/GlassCard';
import {
  Activity, Brain, Cpu, Shield, Scan, Database,
  CheckCircle, XCircle,
  ChevronRight, Loader2, WifiOff,
} from 'lucide-react';
import { motion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { api } from '../services/api';
import type { HealthResponse, RegistryResponse, AnalysisHistoryItem } from '../services/api';

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 }
  }
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

// ==============================
// System Status Card
// ==============================

function SystemStatusCard({
  health,
  loading,
  error,
}: {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading) {
    return (
      <GlassCard className="col-span-full relative overflow-hidden">
        <div className="flex items-center gap-3">
          <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
          <span className="text-sm text-zinc-400">Connecting to backend...</span>
        </div>
      </GlassCard>
    );
  }

  if (error) {
    return (
      <GlassCard className="col-span-full border-red-500/20 bg-red-500/5">
        <div className="flex items-start gap-3">
          <WifiOff className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-400">Backend Offline</p>
            <p className="text-xs text-zinc-500 mt-1">{error}</p>
            <p className="text-xs text-zinc-600 mt-2">
              Start the backend with:{' '}
              <code className="text-cyan-400 bg-white/5 px-1.5 py-0.5 rounded text-[11px]">
                python -m uvicorn backend.api.app:app --reload --port 8000
              </code>
            </p>
          </div>
        </div>
      </GlassCard>
    );
  }

  return (
    <GlassCard className="col-span-full relative overflow-hidden border-emerald-500/10">
      <div className="absolute top-0 right-0 w-48 h-48 bg-emerald-500/5 rounded-full blur-3xl -mr-20 -mt-20" />
      <div className="relative z-10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-emerald-500 relative">
            <div className="absolute inset-0 rounded-full bg-emerald-500 animate-ping opacity-75" />
          </div>
          <div>
            <p className="text-sm font-semibold text-emerald-400">System Online</p>
            <p className="text-xs text-zinc-500">
              v{health?.version} • {health?.device} • {health?.models_loaded} models loaded
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-[11px] text-zinc-500">
          <span className="flex items-center gap-1"><Cpu className="w-3 h-3" /> {health?.device}</span>
          <span className="flex items-center gap-1"><Database className="w-3 h-3" /> {health?.models_loaded} models</span>
        </div>
      </div>
    </GlassCard>
  );
}

// ==============================
// Recent Analysis Row
// ==============================

function RecentAnalysisRow({ item, index }: { item: AnalysisHistoryItem; index: number }) {
  const statusIcon = {
    completed: <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />,
    running: <Loader2 className="w-3.5 h-3.5 text-cyan-400 animate-spin" />,
    failed: <XCircle className="w-3.5 h-3.5 text-red-400" />,
  };

  const date = item.created_at ? new Date(item.created_at) : null;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="flex items-center justify-between px-3 py-2.5 rounded-xl bg-white/[0.02] border border-white/5
        hover:bg-white/[0.05] hover:border-white/10 transition-all cursor-pointer group"
    >
      <div className="flex items-center gap-3">
        {statusIcon[item.status as keyof typeof statusIcon] || statusIcon.completed}
        <div>
          <p className="text-sm font-medium text-white group-hover:text-cyan-400 transition-colors">
            {item.modality || 'Unknown'} / {item.anatomy || 'Unknown'}
          </p>
          <p className="text-[10px] text-zinc-600">{item.request_id?.slice(0, 12)}...</p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        {item.total_time_ms && (
          <span className="text-[11px] text-zinc-500">{item.total_time_ms.toFixed(0)}ms</span>
        )}
        {date && (
          <span className="text-[10px] text-zinc-600">{date.toLocaleTimeString()}</span>
        )}
        <ChevronRight className="w-3.5 h-3.5 text-zinc-700 group-hover:text-zinc-400 transition-colors" />
      </div>
    </motion.div>
  );
}

// ==============================
// Dashboard
// ==============================

export default function Dashboard() {
  const navigate = useNavigate();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [registry, setRegistry] = useState<RegistryResponse | null>(null);
  const [analyses, setAnalyses] = useState<AnalysisHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchDashboardData() {
      try {
        const [healthData, registryData, analysisData] = await Promise.allSettled([
          api.health(),
          api.getModels(),
          api.getAnalyses(10, 0),
        ]);

        if (healthData.status === 'fulfilled') setHealth(healthData.value);
        else throw new Error('Backend unreachable');

        if (registryData.status === 'fulfilled') setRegistry(registryData.value);
        if (analysisData.status === 'fulfilled') setAnalyses(analysisData.value);

        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to connect');
      } finally {
        setLoading(false);
      }
    }

    fetchDashboardData();
    // Poll every 30s
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const completedAnalyses = analyses.filter(a => a.status === 'completed').length;

  return (
    <motion.div
      className="max-w-7xl mx-auto space-y-6"
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-400">
            CouncilMed Workstation
          </h1>
          <p className="text-zinc-500 mt-1">Multi-model medical imaging analysis system</p>
        </div>
      </div>

      {/* System Status */}
      <motion.div variants={itemVariants}>
        <SystemStatusCard health={health} loading={loading} error={error} />
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div variants={itemVariants}>
          <GlassCard className="h-full relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-cyan-500/20 transition-all duration-500" />
            <div className="flex justify-between items-start">
              <div>
                <p className="text-zinc-400 text-sm font-medium">Council Members</p>
                <h3 className="text-3xl font-bold text-white mt-1">
                  {registry?.total_models ?? '—'}
                </h3>
                <p className="text-xs text-zinc-500 mt-2">
                  {registry?.installed ?? 0} installed • {registry?.validated ?? 0} validated
                </p>
              </div>
              <div className="p-3 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                <Brain className="w-5 h-5" />
              </div>
            </div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="h-full relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-emerald-500/20 transition-all duration-500" />
            <div className="flex justify-between items-start">
              <div>
                <p className="text-zinc-400 text-sm font-medium">Analyses Run</p>
                <h3 className="text-3xl font-bold text-white mt-1">
                  {analyses.length}
                </h3>
                <p className="text-xs text-emerald-400 mt-2">
                  {completedAnalyses} completed
                </p>
              </div>
              <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <Scan className="w-5 h-5" />
              </div>
            </div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="h-full relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-purple-500/20 transition-all duration-500" />
            <div className="flex justify-between items-start">
              <div>
                <p className="text-zinc-400 text-sm font-medium">Modalities</p>
                <h3 className="text-3xl font-bold text-white mt-1">
                  {registry?.modalities?.length ?? 0}
                </h3>
                <p className="text-xs text-zinc-500 mt-2">
                  {registry?.modalities?.join(', ') || 'None registered'}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
                <Activity className="w-5 h-5" />
              </div>
            </div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="h-full relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-amber-500/20 transition-all duration-500" />
            <div className="flex justify-between items-start">
              <div>
                <p className="text-zinc-400 text-sm font-medium">Inference Device</p>
                <h3 className="text-3xl font-bold text-white mt-1 text-lg">
                  {health?.device || '—'}
                </h3>
                <p className="text-xs text-zinc-500 mt-2">
                  {health?.hardware?.gpu_name as string || 'CPU inference'}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <Cpu className="w-5 h-5" />
              </div>
            </div>
          </GlassCard>
        </motion.div>
      </div>

      {/* Bottom Grid: Recent Analyses + Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Analyses */}
        <motion.div variants={itemVariants} className="col-span-1 lg:col-span-2">
          <GlassCard>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-bold text-white">Recent Analyses</h2>
                <p className="text-xs text-zinc-500 mt-0.5">Latest council analyses from real backend</p>
              </div>
              <button
                onClick={() => navigate('/history')}
                className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors font-medium"
              >
                View All →
              </button>
            </div>

            {analyses.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Scan className="w-10 h-10 text-zinc-700 mb-3" />
                <p className="text-sm text-zinc-500">No analyses yet</p>
                <button
                  onClick={() => navigate('/analyze')}
                  className="mt-3 flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium
                    bg-cyan-500/10 text-cyan-400 border border-cyan-500/30
                    hover:bg-cyan-500/20 transition-all"
                >
                  <Scan className="w-3.5 h-3.5" />
                  Start First Analysis
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                {analyses.slice(0, 8).map((item, index) => (
                  <RecentAnalysisRow key={item.id} item={item} index={index} />
                ))}
              </div>
            )}
          </GlassCard>
        </motion.div>

        {/* Quick Actions */}
        <motion.div variants={itemVariants} className="col-span-1">
          <GlassCard className="h-full">
            <h2 className="text-lg font-bold text-white mb-4">Quick Actions</h2>
            <div className="space-y-3">
              <button
                onClick={() => navigate('/analyze')}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-gradient-to-r from-cyan-500/10 to-teal-500/10
                  border border-cyan-500/20 hover:border-cyan-500/40 transition-all group"
              >
                <div className="p-2 rounded-lg bg-cyan-500/20">
                  <Scan className="w-4 h-4 text-cyan-400" />
                </div>
                <div className="text-left">
                  <p className="text-sm font-medium text-white group-hover:text-cyan-400 transition-colors">
                    New Analysis
                  </p>
                  <p className="text-[10px] text-zinc-500">Upload medical image for council review</p>
                </div>
                <ChevronRight className="w-4 h-4 text-zinc-600 ml-auto" />
              </button>

              <button
                onClick={() => navigate('/models')}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-xl
                  bg-white/[0.02] border border-white/5 hover:border-white/15 transition-all group"
              >
                <div className="p-2 rounded-lg bg-purple-500/10">
                  <Database className="w-4 h-4 text-purple-400" />
                </div>
                <div className="text-left">
                  <p className="text-sm font-medium text-white group-hover:text-purple-400 transition-colors">
                    Model Registry
                  </p>
                  <p className="text-[10px] text-zinc-500">View council member models</p>
                </div>
                <ChevronRight className="w-4 h-4 text-zinc-600 ml-auto" />
              </button>

              <button
                onClick={() => navigate('/reports')}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-xl
                  bg-white/[0.02] border border-white/5 hover:border-white/15 transition-all group"
              >
                <div className="p-2 rounded-lg bg-amber-500/10">
                  <Shield className="w-4 h-4 text-amber-400" />
                </div>
                <div className="text-left">
                  <p className="text-sm font-medium text-white group-hover:text-amber-400 transition-colors">
                    Reports
                  </p>
                  <p className="text-[10px] text-zinc-500">View generated clinical reports</p>
                </div>
                <ChevronRight className="w-4 h-4 text-zinc-600 ml-auto" />
              </button>
            </div>

            {/* Council Architecture */}
            <div className="mt-6 pt-4 border-t border-white/5">
              <p className="text-[10px] text-zinc-600 uppercase tracking-wider mb-2">Architecture</p>
              <p className="text-xs text-zinc-500 leading-relaxed">
                Multiple independent AI models analyze the same image.
                Their outputs are fused through an uncertainty-weighted
                ensemble to reach evidence-based consensus.
              </p>
            </div>
          </GlassCard>
        </motion.div>
      </div>
    </motion.div>
  );
}
