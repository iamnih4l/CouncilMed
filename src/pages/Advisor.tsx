import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Brain, ShieldAlert, TrendingUp, ChevronRight, MessageSquare, Loader2, Send } from 'lucide-react';
import { GlassCard } from '../components/ui/GlassCard';
import { advisorApi, type AdvisorReport, type NaturalLanguageQueryResponse } from '../services/api/advisor';
import { toast } from 'sonner';

export default function Advisor() {
  const [report, setReport] = useState<AdvisorReport | null>(null);
  const [loadingReport, setLoadingReport] = useState(false);
  const [query, setQuery] = useState('');
  const [querying, setQuerying] = useState(false);
  const [chatHistory, setChatHistory] = useState<{ role: 'user' | 'advisor'; content: string | NaturalLanguageQueryResponse }[]>([]);
  
  // Dummy scan data for context if API is not fully linked
  const contextData = {
    region: "District A",
    risk_score: 0.83,
    population: 150000,
    resource_availability: "Low",
    historical_cases: 42,
    recent_events: ["Sudden spike in high-grade anomalies", "Reduced staff availability"],
    anomaly_flags: ["Volume increase detected", "High confidence from Swin-UNETR"],
    trend_summary: "Increasing risk due to declining healthcare capacity.",
    prediction_confidence: 91.5
  };

  useEffect(() => {
    generateReport();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generateReport = async () => {
    setLoadingReport(true);
    try {
      const res = await advisorApi.generateReport(contextData);
      setReport(res);
      toast.success('Executive Report Generated');
    } catch (err) {
      toast.error('Failed to generate report. Make sure the local LLM server is running.');
      console.error(err);
    } finally {
      setLoadingReport(false);
    }
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userQuery = query.trim();
    setQuery('');
    setChatHistory(prev => [...prev, { role: 'user', content: userQuery }]);
    setQuerying(true);

    try {
      const res = await advisorApi.queryAdvisor(userQuery, contextData);
      setChatHistory(prev => [...prev, { role: 'advisor', content: res }]);
    } catch (err) {
      console.error(err);
      toast.error('Query failed.');
    } finally {
      setQuerying(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto h-[calc(100vh-8rem)] flex flex-col space-y-4">
      <div className="flex items-center justify-between shrink-0">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">AI Policy Advisor</h1>
            <span className="px-2.5 py-1 rounded-full bg-[var(--color-accent-cyan)]/20 text-[var(--color-accent-cyan)] text-xs font-bold border border-[var(--color-accent-cyan)]/30 flex items-center space-x-1">
              <Brain className="w-3 h-3" />
              <span>OFFLINE LLM REASONING</span>
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1">
            Analyzing predictive outputs to generate strategic recommendations.
          </p>
        </div>
        <button
          onClick={generateReport}
          disabled={loadingReport}
          className="px-4 py-2 bg-[var(--color-accent-cyan)] hover:bg-[var(--color-accent-teal)] text-[#0d1324] font-bold rounded-lg transition-colors flex items-center space-x-2 disabled:opacity-50"
        >
          {loadingReport ? <Loader2 className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
          <span>Regenerate Report</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1 min-h-0">
        <div className="col-span-1 lg:col-span-2 overflow-y-auto pr-1 pb-4 flex flex-col space-y-4">
          {loadingReport ? (
            <GlassCard className="flex flex-col items-center justify-center py-20">
              <Loader2 className="w-12 h-12 text-[var(--color-accent-cyan)] animate-spin mb-4" />
              <h3 className="text-lg font-medium text-white">Analyzing Data...</h3>
              <p className="text-sm text-zinc-400">The offline reasoning engine is generating insights.</p>
            </GlassCard>
          ) : report ? (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
              <GlassCard className="p-6">
                <h2 className="text-lg font-bold text-white mb-3">Executive Summary</h2>
                <p className="text-zinc-300 leading-relaxed">{report.summary}</p>
                <div className="mt-4 flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-zinc-500">Confidence:</span>
                    <span className={`text-xs font-bold px-2 py-1 rounded border ${
                      report.confidence === 'High' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' :
                      report.confidence === 'Medium' ? 'bg-amber-500/20 text-amber-400 border-amber-500/30' :
                      'bg-red-500/20 text-red-400 border-red-500/30'
                    }`}>{report.confidence}</span>
                  </div>
                </div>
              </GlassCard>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <GlassCard className="p-5">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4 border-b border-white/10 pb-2 flex items-center">
                    <ShieldAlert className="w-4 h-4 mr-2 text-amber-400" />
                    Risk Analysis
                  </h3>
                  <p className="text-sm text-zinc-300">{report.risk_analysis}</p>
                </GlassCard>

                <GlassCard className="p-5">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4 border-b border-white/10 pb-2 flex items-center">
                    <TrendingUp className="w-4 h-4 mr-2 text-cyan-400" />
                    Key Findings
                  </h3>
                  <ul className="space-y-2">
                    {report.key_findings.map((finding, idx) => (
                      <li key={idx} className="text-sm text-zinc-300 flex items-start">
                        <ChevronRight className="w-4 h-4 text-zinc-500 mr-1 shrink-0 mt-0.5" />
                        <span>{finding}</span>
                      </li>
                    ))}
                  </ul>
                </GlassCard>
              </div>

              <GlassCard className="p-6">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4 border-b border-white/10 pb-2">
                  Recommended Actions
                </h3>
                <div className="space-y-4">
                  {report.recommended_actions.map((action, idx) => (
                    <div key={idx} className="bg-white/5 border border-white/10 rounded-xl p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-base font-medium text-white">{action.action}</h4>
                        <span className={`text-xs font-bold px-2 py-1 rounded ${
                          action.priority === 'High' ? 'bg-red-500/20 text-red-400' :
                          action.priority === 'Medium' ? 'bg-amber-500/20 text-amber-400' :
                          'bg-emerald-500/20 text-emerald-400'
                        }`}>
                          {action.priority} Priority
                        </span>
                      </div>
                      <p className="text-sm text-zinc-400 mb-2">{action.reason}</p>
                      <div className="flex items-center text-xs text-[var(--color-accent-cyan)] bg-[var(--color-accent-cyan)]/10 px-3 py-1.5 rounded-lg inline-flex">
                        <span className="font-semibold mr-1">Expected Impact:</span> {action.expected_impact}
                      </div>
                    </div>
                  ))}
                </div>
              </GlassCard>
            </motion.div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-zinc-500">
              No report generated yet.
            </div>
          )}
        </div>

        <div className="col-span-1 flex flex-col">
          <GlassCard className="flex-1 flex flex-col overflow-hidden p-0">
            <div className="p-4 border-b border-white/10 bg-black/20">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <MessageSquare className="w-4 h-4" />
                <span>Advisor Query Engine</span>
              </h3>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {chatHistory.length === 0 && (
                <div className="text-center text-sm text-zinc-500 mt-10">
                  Ask me about the current situation, historical trends, or simulate scenarios.
                </div>
              )}
              {chatHistory.map((msg, idx) => (
                <motion.div 
                  key={idx}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[85%] rounded-xl p-3 text-sm ${
                    msg.role === 'user' 
                      ? 'bg-[var(--color-accent-cyan)]/20 border border-[var(--color-accent-cyan)]/30 text-white' 
                      : 'bg-white/5 border border-white/10 text-zinc-300'
                  }`}>
                    {msg.role === 'user' ? (
                      msg.content as string
                    ) : (
                      <div className="space-y-2">
                        <p>{(msg.content as NaturalLanguageQueryResponse).answer}</p>
                        {((msg.content as NaturalLanguageQueryResponse).supporting_evidence?.length ?? 0) > 0 && (
                          <div className="mt-2 pt-2 border-t border-white/10">
                            <p className="text-xs text-zinc-500 mb-1">Evidence:</p>
                            <ul className="list-disc pl-4 text-xs text-zinc-400 space-y-1">
                              {(msg.content as NaturalLanguageQueryResponse).supporting_evidence.map((ev, i) => (
                                <li key={i}>{ev}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
              {querying && (
                <div className="flex justify-start">
                  <div className="max-w-[85%] rounded-xl p-3 bg-white/5 border border-white/10 text-zinc-300">
                    <Loader2 className="w-4 h-4 animate-spin" />
                  </div>
                </div>
              )}
            </div>

            <div className="p-4 border-t border-white/10 bg-black/20">
              <form onSubmit={handleQuery} className="relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask a question..."
                  className="w-full bg-black/40 border border-white/10 rounded-xl pl-4 pr-12 py-3 text-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-[var(--color-accent-cyan)]/50 transition-colors"
                  disabled={querying}
                />
                <button 
                  type="submit"
                  disabled={querying || !query.trim()}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-[var(--color-accent-cyan)] hover:bg-[var(--color-accent-cyan)]/10 rounded-lg transition-colors disabled:opacity-50"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
