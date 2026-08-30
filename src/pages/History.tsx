/**
 * CouncilMed — Analysis History Page
 * ======================================
 * Lists past council analyses with results.
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Clock, Brain,  FileText, ChevronRight, RefreshCw } from 'lucide-react';
import { GlassCard } from '../components/ui/GlassCard';
import { api } from '../services/api';
import type { AnalysisHistoryItem } from '../services/api';

function HistoryRow({ item, index }: { item: AnalysisHistoryItem; index: number }) {
  const statusColors: Record<string, string> = {
    completed: 'text-emerald-400 bg-emerald-500/10',
    running: 'text-cyan-400 bg-cyan-500/10',
    pending: 'text-amber-400 bg-amber-500/10',
    failed: 'text-red-400 bg-red-500/10',
  };

  const sc = statusColors[item.status] || 'text-zinc-400 bg-zinc-500/10';
  const date = item.created_at ? new Date(item.created_at) : null;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="flex items-center justify-between px-4 py-3 rounded-xl bg-white/[0.02] border border-white/5
        hover:bg-white/[0.05] hover:border-white/10 transition-all cursor-pointer group"
    >
      <div className="flex items-center gap-4">
        <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
          <Brain className="w-4 h-4 text-cyan-400" />
        </div>
        <div>
          <p className="text-sm font-medium text-white group-hover:text-cyan-400 transition-colors">
            {item.modality} / {item.anatomy}
          </p>
          <p className="text-[11px] text-zinc-600">
            {item.request_id.slice(0, 8)}...
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <span className={`text-[10px] px-2 py-0.5 rounded-full ${sc}`}>
          {item.status}
        </span>

        {item.total_time_ms && (
          <span className="text-[11px] text-zinc-500 flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {item.total_time_ms.toFixed(0)}ms
          </span>
        )}

        {date && (
          <span className="text-[11px] text-zinc-600">
            {date.toLocaleDateString()} {date.toLocaleTimeString()}
          </span>
        )}

        <ChevronRight className="w-4 h-4 text-zinc-700 group-hover:text-zinc-400 transition-colors" />
      </div>
    </motion.div>
  );
}

export default function HistoryPage() {
  const [analyses, setAnalyses] = useState<AnalysisHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getAnalyses(50, 0);
      setAnalyses(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchHistory(); }, []);

  return (
    <motion.div
      className="max-w-4xl mx-auto space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-400">
            Analysis History
          </h1>
          <p className="text-zinc-500 text-sm mt-0.5">Past council analyses and their results</p>
        </div>
        <button
          onClick={fetchHistory}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
            bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-white transition-all"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {error && (
        <GlassCard className="border-red-500/20 bg-red-500/5">
          <p className="text-sm text-red-400">{error}</p>
          <p className="text-xs text-zinc-500 mt-1">Ensure the backend is running.</p>
        </GlassCard>
      )}

      {!loading && analyses.length === 0 && !error && (
        <GlassCard className="flex flex-col items-center justify-center py-16">
          <FileText className="w-10 h-10 text-zinc-700 mb-4" />
          <p className="text-sm text-zinc-500 font-medium">No analyses yet</p>
          <p className="text-xs text-zinc-600 mt-1">Analyze a medical image from the Workstation to see results here</p>
        </GlassCard>
      )}

      <div className="space-y-2">
        {analyses.map((item, index) => (
          <HistoryRow key={item.id} item={item} index={index} />
        ))}
      </div>
    </motion.div>
  );
}
