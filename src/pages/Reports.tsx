/**
 * CouncilMed — Reports Page
 * ============================
 * View and generate structured clinical reports from council analyses.
 * Reports are generated from real evidence — no fabricated content.
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FileText, Download, Loader2, AlertCircle, ChevronDown,
  ChevronRight, Shield, Brain, Clock, Copy, Check,
  AlertTriangle, Info,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { GlassCard } from '../components/ui/GlassCard';
import { api } from '../services/api';
import type { AnalysisHistoryItem, ReportResponse, ReportSectionResponse } from '../services/api';

function ReportSection({ section, depth = 0 }: { section: ReportSectionResponse; depth?: number }) {
  const [expanded, setExpanded] = useState(true);

  const severityStyles: Record<string, string> = {
    critical: 'border-l-red-500/50 bg-red-500/5',
    warning: 'border-l-amber-500/50 bg-amber-500/5',
    '': '',
  };

  const style = severityStyles[section.severity] || '';

  return (
    <div className={`border-l-2 ${style || 'border-l-white/10'} pl-3 ${depth > 0 ? 'ml-3 mt-2' : 'mt-3'}`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 w-full text-left group"
      >
        {expanded ? (
          <ChevronDown className="w-3 h-3 text-zinc-600 flex-shrink-0" />
        ) : (
          <ChevronRight className="w-3 h-3 text-zinc-600 flex-shrink-0" />
        )}
        <span className={`text-xs font-semibold uppercase tracking-wider ${
          depth === 0 ? 'text-zinc-300' : 'text-zinc-500'
        } group-hover:text-white transition-colors`}>
          {section.title}
        </span>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="space-y-1 mt-1.5 pl-4">
              {section.content.map((line, i) => (
                <p key={i} className="text-[12px] text-zinc-400 leading-relaxed font-mono">
                  {line}
                </p>
              ))}
            </div>

            {section.subsections.map((sub, i) => (
              <ReportSection key={i} section={sub} depth={depth + 1} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ReportViewer({ report }: { report: ReportResponse }) {
  const [copied, setCopied] = useState(false);
  const [showText, setShowText] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(report.report_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      const textArea = document.createElement('textarea');
      textArea.value = report.report_text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownload = () => {
    const blob = new Blob([report.report_text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `councilmed_report_${report.report_id.slice(0, 8)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Report Header */}
      <GlassCard>
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/20">
                <FileText className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">Analysis Report</h2>
                <p className="text-xs text-zinc-500">{report.study_type} • v{report.version}</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className={`text-[10px] px-2 py-0.5 rounded-full ${
              report.report_status === 'final'
                ? 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20'
                : 'text-amber-400 bg-amber-500/10 border border-amber-500/20'
            }`}>
              {report.report_status.toUpperCase()}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4 mt-3 text-[11px] text-zinc-500">
          <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{new Date(report.generated_at).toLocaleString()}</span>
          <span>{report.modality} / {report.anatomy}</span>
          <span>ID: {report.report_id.slice(0, 8)}</span>
        </div>
      </GlassCard>

      {/* Impression */}
      <GlassCard className="border-cyan-500/10 bg-cyan-500/[0.03]">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-cyan-400 mb-2 flex items-center gap-1.5">
          <Brain className="w-3.5 h-3.5" /> Impression
        </h3>
        <p className="text-sm text-zinc-300 leading-relaxed">{report.impression}</p>
      </GlassCard>

      {/* Report Sections */}
      <GlassCard>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-1">Detailed Findings</h3>
        {report.sections.map((section, i) => (
          <ReportSection key={i} section={section} />
        ))}
      </GlassCard>

      {/* Recommendations */}
      {report.recommendations.length > 0 && (
        <GlassCard>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-400 mb-2 flex items-center gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5" /> Recommendations
          </h3>
          <ul className="space-y-1.5">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="text-[12px] text-zinc-400 flex items-start gap-2">
                <span className="text-amber-400 mt-0.5">•</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </GlassCard>
      )}

      {/* Limitations */}
      {report.limitations.length > 0 && (
        <GlassCard className="border-zinc-700/50">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-2 flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5" /> Limitations
          </h3>
          <ul className="space-y-1.5">
            {report.limitations.map((lim, i) => (
              <li key={i} className="text-[12px] text-zinc-500 flex items-start gap-2">
                <span className="text-zinc-600 mt-0.5">•</span>
                <span>{lim}</span>
              </li>
            ))}
          </ul>
        </GlassCard>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium
            bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-white transition-all"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          {copied ? 'Copied!' : 'Copy Report'}
        </button>
        <button
          onClick={handleDownload}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium
            bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-white transition-all"
        >
          <Download className="w-3.5 h-3.5" /> Download .txt
        </button>
        <button
          onClick={() => setShowText(!showText)}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium
            bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-white transition-all"
        >
          <FileText className="w-3.5 h-3.5" /> {showText ? 'Hide' : 'Show'} Plain Text
        </button>
      </div>

      {/* Plain Text View */}
      <AnimatePresence>
        {showText && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            <GlassCard className="p-0">
              <pre className="text-[11px] text-zinc-400 font-mono p-4 overflow-x-auto whitespace-pre-wrap leading-relaxed max-h-[500px] overflow-y-auto">
                {report.report_text}
              </pre>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Disclaimer */}
      <div className="px-4 py-3 rounded-xl bg-white/[0.02] border border-white/5">
        <div className="flex items-start gap-2">
          <Shield className="w-4 h-4 text-zinc-600 flex-shrink-0 mt-0.5" />
          <p className="text-[11px] text-zinc-600 leading-relaxed">
            This report is AI-generated and intended to assist clinical decision-making.
            It does NOT constitute a diagnosis and MUST be reviewed by a qualified clinician
            before any clinical action is taken.
          </p>
        </div>
      </div>
    </motion.div>
  );
}

export default function ReportsPage() {
  const navigate = useNavigate();
  const [analyses, setAnalyses] = useState<AnalysisHistoryItem[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchAnalyses() {
      setLoading(true);
      try {
        const data = await api.getAnalyses(50, 0);
        setAnalyses(data.filter(a => a.status === 'completed'));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load analyses');
      } finally {
        setLoading(false);
      }
    }
    fetchAnalyses();
  }, []);

  const handleGenerateReport = async (analysisId: string) => {
    setGenerating(true);
    setError(null);
    setReport(null);
    setSelectedAnalysis(analysisId);

    try {
      const reportData = await api.generateReport({ analysis_id: analysisId });
      setReport(reportData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Report generation failed');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <motion.div
      className="max-w-5xl mx-auto space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="flex items-start justify-between">
        <div>
        <h1 className="text-2xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-400">
          Clinical Reports
        </h1>
        <p className="text-zinc-500 text-sm mt-0.5">Generate structured reports from council evidence</p>
      </div>
        <button onClick={() => navigate('/')} className="flex items-center gap-2 px-4 py-2 border border-cyan-500/30 rounded-lg text-xs font-mono tracking-widest bg-black/40 text-cyan-400 hover:bg-cyan-500/20 hover:text-white transition-all shadow-[0_0_15px_rgba(0,240,255,0.1)] hover:shadow-[0_0_25px_rgba(0,240,255,0.25)]">
          &larr; BACK TO WORKSTATION
        </button>
      </div>

      {error && (
        <GlassCard className="border-red-500/20 bg-red-500/5">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <div>
              <p className="text-sm text-red-400 font-medium">Error</p>
              <p className="text-xs text-zinc-400 mt-1">{error}</p>
            </div>
          </div>
        </GlassCard>
      )}

      {/* Analysis Selector */}
      {!report && (
        <GlassCard>
          <h2 className="text-sm font-semibold text-white mb-3">Select Analysis</h2>
          <p className="text-xs text-zinc-500 mb-4">Choose a completed analysis to generate a report from.</p>

          {loading ? (
            <div className="flex items-center gap-2 py-8 justify-center">
              <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
              <span className="text-sm text-zinc-500">Loading analyses...</span>
            </div>
          ) : analyses.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
              <p className="text-sm text-zinc-500">No completed analyses</p>
              <p className="text-xs text-zinc-600 mt-1">Run an analysis from the Workstation first</p>
            </div>
          ) : (
            <div className="space-y-2">
              {analyses.map((item) => {
                const date = item.created_at ? new Date(item.created_at) : null;
                const isSelected = selectedAnalysis === item.id;

                return (
                  <button
                    key={item.id}
                    onClick={() => handleGenerateReport(item.id)}
                    disabled={generating}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all
                      ${isSelected && generating
                        ? 'bg-cyan-500/10 border border-cyan-500/30'
                        : 'bg-white/[0.02] border border-white/5 hover:bg-white/[0.05] hover:border-white/10'
                      }`}
                  >
                    <div className="flex items-center gap-3">
                      <Brain className="w-4 h-4 text-cyan-400" />
                      <div className="text-left">
                        <p className="text-sm font-medium text-white">
                          {item.modality || 'Unknown'} / {item.anatomy || 'Unknown'}
                        </p>
                        <p className="text-[10px] text-zinc-600">
                          {item.request_id?.slice(0, 12)}...
                          {date && ` • ${date.toLocaleString()}`}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {isSelected && generating ? (
                        <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
                      ) : (
                        <span className="text-[11px] text-cyan-400">Generate Report →</span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </GlassCard>
      )}

      {/* Report Display */}
      {report && (
        <>
          <button
            onClick={() => { setReport(null); setSelectedAnalysis(null); }}
            className="text-xs text-zinc-500 hover:text-white transition-colors"
          >
            ← Back to Analysis List
          </button>
          <ReportViewer report={report} />
        </>
      )}
    </motion.div>
  );
}
