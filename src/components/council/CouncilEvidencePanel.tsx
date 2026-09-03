/**
 * CouncilMed — Council Evidence Panel
 * ======================================
 * Displays the structured output from the AI Council:
 * - Consensus level with visual indicator
 * - Per-model evidence breakdown  
 * - Agreement/disagreement analysis
 * - Uncertainty visualization
 * - Human review status
 *
 * This is the PRIMARY output display of CouncilMed — the equivalent of
 * a radiology workstation's findings panel.
 */

import { useState } from 'react';
import { HudPanel } from '../ui/HudPanel';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield, ShieldAlert, ShieldCheck, ShieldX,
  ChevronDown, ChevronUp, AlertTriangle,
  CheckCircle, XCircle, Clock, Cpu, Brain,
  Eye, Activity, TrendingUp,
} from 'lucide-react';
import type { CouncilResultResponse, ModelEvidenceResponse } from '../../services/api';

interface CouncilEvidencePanelProps {
  result: CouncilResultResponse;
  onReview?: (decision: 'accepted' | 'rejected' | 'modified') => void;
}

// ==============================
// Consensus Badge
// ==============================

function ConsensusBadge({ level }: { level: string }) {
  const config: Record<string, { icon: typeof Shield; color: string; bg: string; label: string }> = {
    strong: {
      icon: ShieldCheck,
      color: 'text-emerald-400',
      bg: 'bg-emerald-500/10 border-emerald-500/30',
      label: 'Strong Consensus',
    },
    moderate: {
      icon: Shield,
      color: 'text-cyan-400',
      bg: 'bg-cyan-500/10 border-cyan-500/30',
      label: 'Moderate Consensus',
    },
    weak: {
      icon: ShieldAlert,
      color: 'text-amber-400',
      bg: 'bg-amber-500/10 border-amber-500/30',
      label: 'Weak Consensus',
    },
    none: {
      icon: ShieldX,
      color: 'text-red-400',
      bg: 'bg-red-500/10 border-red-500/30',
      label: 'No Consensus',
    },
    insufficient: {
      icon: ShieldX,
      color: 'text-zinc-400',
      bg: 'bg-zinc-500/10 border-zinc-500/30',
      label: 'Insufficient Data',
    },
  };

  const c = config[level] || config.insufficient;
  const Icon = c.icon;

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border ${c.bg}`}>
      <Icon className={`w-4 h-4 ${c.color}`} />
      <span className={`text-sm font-semibold ${c.color}`}>{c.label}</span>
    </div>
  );
}

// ==============================
// Severity Indicator
// ==============================

function SeverityIndicator({ severity }: { severity: string }) {
  const config: Record<string, { color: string; pulse: boolean; label: string }> = {
    clear: { color: 'bg-emerald-500', pulse: false, label: 'Clear' },
    info: { color: 'bg-blue-500', pulse: false, label: 'Info' },
    warning: { color: 'bg-amber-500', pulse: true, label: 'Warning' },
    critical: { color: 'bg-red-500', pulse: true, label: 'Critical' },
  };

  const c = config[severity] || config.info;

  return (
    <div className="flex items-center gap-2">
      <div className={`relative w-2.5 h-2.5 rounded-full ${c.color}`}>
        {c.pulse && (
          <div className={`absolute inset-0 rounded-full ${c.color} animate-ping opacity-75`} />
        )}
      </div>
      <span className="text-xs font-medium text-zinc-300 uppercase tracking-wider">{c.label}</span>
    </div>
  );
}

// ==============================
// Confidence Arc
// ==============================

function ConfidenceArc({ value, size = 80 }: { value: number; size?: number }) {
  const radius = (size - 8) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - value);
  const percentage = Math.round(value * 100);

  const color = value >= 0.8 ? '#10b981' : value >= 0.6 ? '#06b6d4' : value >= 0.4 ? '#f59e0b' : '#ef4444';

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={4}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={4}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-sm font-bold" style={{ color }}>{percentage}%</span>
      </div>
    </div>
  );
}

// ==============================
// Model Evidence Card
// ==============================

function ModelEvidenceCard({ evidence, index }: { evidence: ModelEvidenceResponse; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const isSuccess = evidence.status === 'success';

  const architectureLabel: Record<string, string> = {
    CNN: '🧠',
    'CNN-Attention': '👁️',
    'UNet-Lite': '🎯',
    ViT: '🔬',
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
    >
      <div
        className={`rounded-xl border transition-all duration-300 ${
          isSuccess
            ? 'bg-white/[0.03] border-white/10 hover:border-white/20'
            : 'bg-red-500/5 border-red-500/20'
        }`}
      >
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full px-4 py-3 flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center text-sm">
              {architectureLabel[evidence.model_architecture] || '🤖'}
            </div>
            <div className="text-left">
              <p className="text-sm font-medium text-white">{evidence.model_name}</p>
              <p className="text-[11px] text-zinc-500">v{evidence.model_version} • {evidence.model_architecture}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {isSuccess ? (
              <>
                <div className="text-right mr-2">
                  <p className="text-sm font-semibold text-white">{evidence.prediction}</p>
                  <p className="text-[11px] text-zinc-500">
                    {(evidence.confidence * 100).toFixed(1)}% conf
                  </p>
                </div>
                <ConfidenceArc value={evidence.confidence} size={36} />
              </>
            ) : (
              <span className="text-xs text-red-400 flex items-center gap-1">
                <XCircle className="w-3 h-3" />
                {evidence.status}
              </span>
            )}
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-zinc-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-zinc-500" />
            )}
          </div>
        </button>

        <AnimatePresence>
          {expanded && isSuccess && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="px-4 pb-4 space-y-3">
                {/* Probability distribution */}
                <div>
                  <p className="text-[11px] text-zinc-500 uppercase tracking-wider mb-2">Class Probabilities</p>
                  <div className="space-y-1.5">
                    {Object.entries(evidence.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([cls, prob]) => (
                        <div key={cls} className="flex items-center gap-2">
                          <span className="text-[11px] text-zinc-400 w-36 truncate">{cls}</span>
                          <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                            <motion.div
                              className="h-full rounded-full"
                              style={{
                                backgroundColor: prob === evidence.confidence ? '#06b6d4' : 'rgba(255,255,255,0.2)',
                              }}
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              transition={{ duration: 0.5, delay: 0.1 }}
                            />
                          </div>
                          <span className="text-[11px] font-mono text-zinc-400 w-12 text-right">
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Metadata row */}
                <div className="flex items-center gap-4 pt-2 border-t border-white/5">
                  <div className="flex items-center gap-1 text-[11px] text-zinc-500">
                    <Activity className="w-3 h-3" />
                    <span>Uncertainty: {(evidence.uncertainty * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex items-center gap-1 text-[11px] text-zinc-500">
                    <Clock className="w-3 h-3" />
                    <span>{evidence.processing_time_ms.toFixed(0)}ms</span>
                  </div>
                  <div className="flex items-center gap-1 text-[11px] text-zinc-500">
                    <Cpu className="w-3 h-3" />
                    <span>{evidence.device}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

// ==============================
// Main Evidence Panel
// ==============================

export function CouncilEvidencePanel({ result, onReview }: CouncilEvidencePanelProps) {
  

  return (
    <div className="space-y-4">
      {/* Header: Primary Finding */}
      <HudPanel className="relative overflow-hidden border-cyan-500/20">
        <div className="absolute top-0 right-0 w-48 h-48 rounded-full blur-3xl -mr-20 -mt-20 opacity-20"
          style={{
            backgroundColor: result.severity === 'critical' ? '#ef4444' :
              result.severity === 'warning' ? '#f59e0b' :
              result.severity === 'clear' ? '#10b981' : '#06b6d4'
          }}
        />

        <div className="relative z-10">
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Council Finding</p>
              <h2 className="text-2xl font-bold text-white">{result.primary_finding}</h2>
            </div>
            <SeverityIndicator severity={result.severity} />
          </div>

          <div className="flex items-center gap-4 flex-wrap">
            <ConsensusBadge level={result.consensus_level} />

            <div className="flex items-center gap-2 text-zinc-400 text-sm">
              <TrendingUp className="w-4 h-4" />
              <span>Evidence: <strong className="text-white">{result.evidence_strength}</strong></span>
            </div>

            <div className="flex items-center gap-2 text-zinc-400 text-sm">
              <Brain className="w-4 h-4" />
              <span>{result.models_agreeing}/{result.models_total} models agree</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-white/5">
            <div className="text-center">
              <ConfidenceArc value={result.primary_finding_confidence} size={56} />
              <p className="text-[11px] text-zinc-500 mt-1">Confidence</p>
            </div>
            <div className="text-center">
              <ConfidenceArc value={result.agreement_score} size={56} />
              <p className="text-[11px] text-zinc-500 mt-1">Agreement</p>
            </div>
            <div className="text-center">
              <ConfidenceArc value={1 - result.ensemble_uncertainty} size={56} />
              <p className="text-[11px] text-zinc-500 mt-1">Certainty</p>
            </div>
          </div>
        </div>
      </HudPanel>

      {/* Human Review Alert */}
      {result.requires_human_review && (
        <HudPanel className="border-amber-500/30 bg-amber-500/5">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-semibold text-amber-400">Human Review Required</p>
              <ul className="mt-2 space-y-1">
                {result.human_review_reasons.map((reason, i) => (
                  <li key={i} className="text-xs text-zinc-400 flex items-start gap-1.5">
                    <span className="text-amber-500 mt-0.5">•</span>
                    {reason}
                  </li>
                ))}
              </ul>

              {onReview && (
                <div className="flex gap-2 mt-3">
                  <button
                    onClick={() => onReview('accepted')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                      bg-emerald-500/10 text-emerald-400 border border-emerald-500/30
                      hover:bg-emerald-500/20 transition-all"
                  >
                    <CheckCircle className="w-3.5 h-3.5" /> Accept
                  </button>
                  <button
                    onClick={() => onReview('rejected')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                      bg-red-500/10 text-red-400 border border-red-500/30
                      hover:bg-red-500/20 transition-all"
                  >
                    <XCircle className="w-3.5 h-3.5" /> Reject
                  </button>
                  <button
                    onClick={() => onReview('modified')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                      bg-cyan-500/10 text-cyan-400 border border-cyan-500/30
                      hover:bg-cyan-500/20 transition-all"
                  >
                    <Eye className="w-3.5 h-3.5" /> Modify
                  </button>
                </div>
              )}
            </div>
          </div>
        </HudPanel>
      )}

      {/* Disagreements */}
      {result.disagreements.length > 0 && (
        <HudPanel className="border-amber-500/20 bg-amber-500/5">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <p className="text-sm font-mono tracking-widest text-amber-400 uppercase">Model Disagreements</p>
          </div>
          <div className="space-y-2">
            {result.disagreements.map((d, i) => (
              <div key={i} className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.02] border border-white/5">
                <div>
                  <span className="text-xs text-zinc-400">{d.model_id}</span>
                  <span className="text-xs text-zinc-600 mx-2">predicts</span>
                  <span className="text-xs font-medium text-amber-400">{d.model_prediction}</span>
                </div>
                <span className="text-xs text-zinc-500">
                  {(d.model_confidence * 100).toFixed(1)}% conf
                </span>
              </div>
            ))}
          </div>
        </HudPanel>
      )}

      {/* Individual Model Evidence */}
      <div>
        <p className="text-[11px] text-zinc-500 uppercase tracking-wider mb-3 px-1">
          Individual Model Evidence ({result.member_evidence.length} members)
        </p>
        <div className="space-y-2">
          {result.member_evidence.map((evidence, index) => (
            <ModelEvidenceCard key={evidence.model_id} evidence={evidence} index={index} />
          ))}
        </div>
      </div>

      {/* Timing footer */}
      <div className="flex items-center justify-between text-[11px] text-zinc-600 px-1">
        <span>Council Run: {result.council_run_id.slice(0, 8)}...</span>
        <span>Total: {result.total_inference_time_ms.toFixed(0)}ms</span>
      </div>
    </div>
  );
}
