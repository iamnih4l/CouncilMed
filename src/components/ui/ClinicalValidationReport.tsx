import { AlertTriangle, ShieldCheck, Info, ClipboardList, ShieldAlert } from 'lucide-react';
import { GlassCard } from './GlassCard';
import type { ClinicalValidationResult } from '../../services/inference/types';
import { ClinicalValidationLayer } from '../../services/inference/ClinicalValidationLayer';

interface Props {
  validationResult: ClinicalValidationResult;
  primaryDiagnosis: string;
}

export const ClinicalValidationReport: React.FC<Props> = ({ validationResult, primaryDiagnosis }) => {
  const { detectedIssues, correctionsApplied, status, finalFindings, patientSafetyOverride } = validationResult;

  return (
    <div className="space-y-4">
      {/* ⚠️ Detected Issues */}
      <GlassCard className="p-4 border-l-4 border-l-[var(--color-accent-ruby)] bg-[var(--color-accent-ruby)]/5">
        <div className="flex items-center space-x-2 mb-3">
          <AlertTriangle className="w-5 h-5 text-[var(--color-accent-ruby)]" />
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">⚠️ Detected Issues</h3>
        </div>
        <ul className="space-y-3">
          {detectedIssues.length > 0 ? (
            detectedIssues.map((issue, i) => (
              <li key={i} className="text-xs">
                <div className="font-bold text-zinc-200 mb-0.5">• {issue.type}</div>
                <div className="text-zinc-400 pl-3">{issue.description}</div>
                <div className="text-[var(--color-accent-ruby)]/80 pl-3 mt-1 italic">
                  <span className="font-bold">Risk:</span> {issue.risk}
                </div>
              </li>
            ))
          ) : (
            <li className="text-xs text-zinc-400">No critical issues detected.</li>
          )}
        </ul>
      </GlassCard>

      {/* 🔧 Corrections Applied */}
      <GlassCard className="p-4 border-l-4 border-l-[var(--color-accent-amber)] bg-[var(--color-accent-amber)]/5">
        <div className="flex items-center space-x-2 mb-3">
          <ClipboardList className="w-5 h-5 text-[var(--color-accent-amber)]" />
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">🔧 Corrections Applied</h3>
        </div>
        <ul className="space-y-2">
          {correctionsApplied.length > 0 ? (
            correctionsApplied.map((fix, i) => (
              <li key={i} className="text-xs text-zinc-300 flex items-start space-x-2">
                <span className="text-[var(--color-accent-amber)]">•</span>
                <span>{fix}</span>
              </li>
            ))
          ) : (
            <li className="text-xs text-zinc-400">Standard clinical normalization applied.</li>
          )}
        </ul>
      </GlassCard>

      {/* 🧾 Final CouncilAI Output */}
      <GlassCard className="p-5 border border-[var(--color-accent-cyan)]/30 bg-[var(--color-accent-cyan)]/5" focused>
        <div className="flex items-center space-x-2 mb-4">
          <ShieldCheck className="w-5 h-5 text-[var(--color-accent-cyan)]" />
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">🧾 Final CouncilAI Output</h3>
        </div>
        
        <div className="space-y-4">
          <div className="flex justify-between items-center border-b border-white/10 pb-2">
            <span className="text-xs text-zinc-500">Status</span>
            <span className={`text-sm font-bold ${
              status === 'High Confidence' ? 'text-[var(--color-accent-emerald)]'
                : status === 'Suspicious' ? 'text-[var(--color-accent-amber)]'
                : 'text-[var(--color-accent-ruby)]'
            }`}>
              {status.toUpperCase()}
            </span>
          </div>

          <div className="space-y-2">
            <h4 className="text-[10px] font-bold text-zinc-500 uppercase">Findings</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white/5 p-2 rounded border border-white/5">
                <p className="text-[10px] text-zinc-500 mb-1">Pattern Detected</p>
                <p className={`text-xs font-bold ${
                  finalFindings.patternDetected === 'Yes' ? 'text-[var(--color-accent-ruby)]'
                    : finalFindings.patternDetected === 'No' ? 'text-[var(--color-accent-emerald)]'
                    : 'text-[var(--color-accent-amber)]'
                }`}>
                  {finalFindings.patternDetected}
                </p>
              </div>
              <div className="bg-white/5 p-2 rounded border border-white/5">
                <p className="text-[10px] text-zinc-500 mb-1">Region</p>
                <p className="text-xs font-bold text-white">
                  {finalFindings.region || 'N/A (Inconclusive)'}
                </p>
              </div>
            </div>
          </div>

          <div className="pt-2">
            <div className={`p-3 rounded-lg border flex items-start space-x-3 ${
              patientSafetyOverride ? 'bg-[var(--color-accent-ruby)]/10 border-[var(--color-accent-ruby)]/20' : 'bg-[var(--color-accent-cyan)]/10 border-[var(--color-accent-cyan)]/20'
            }`}>
              {patientSafetyOverride ? <ShieldAlert className="w-4 h-4 text-[var(--color-accent-ruby)] mt-0.5" /> : <Info className="w-4 h-4 text-[var(--color-accent-cyan)] mt-0.5" />}
              <div>
                <p className="text-[10px] font-bold text-white mb-0.5">Clinical Observation</p>
                <p className="text-xs text-zinc-300 leading-relaxed italic">
                  "{ClinicalValidationLayer.getSafeLabel(primaryDiagnosis as any)}"
                </p>
              </div>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* 📌 Conclusion & Recommendation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <GlassCard className="p-4 bg-white/5">
          <h3 className="text-[10px] font-bold text-white uppercase mb-2">📌 Conclusion</h3>
          <p className="text-[11px] text-zinc-400 leading-relaxed">
            AI-assisted analysis indicates {status === 'Inconclusive' ? 'ambiguous patterns requiring clinical arbitration' : 'notable patterns for radiological review'}. Observations are provided for decision support only.
          </p>
        </GlassCard>
        <GlassCard className="p-4 bg-white/5">
          <h3 className="text-[10px] font-bold text-white uppercase mb-2">🏥 Recommendation</h3>
          <ul className="text-[11px] text-zinc-400 space-y-1">
            <li className="flex items-start space-x-1">
              <span className="text-[var(--color-accent-cyan)]">•</span>
              <span>Mandatory radiologist review</span>
            </li>
            <li className="flex items-start space-x-1">
              <span className="text-[var(--color-accent-cyan)]">•</span>
              <span>Correlate with clinical history</span>
            </li>
          </ul>
        </GlassCard>
      </div>

      {/* ⚠️ Disclaimer */}
      <div className="p-3 rounded-lg border border-white/5 bg-black/20 text-center">
        <p className="text-[9px] text-zinc-500 uppercase tracking-widest font-bold">
          ⚠️ Disclaimer: This is an AI-assisted analysis and NOT a medical diagnosis.
        </p>
      </div>
    </div>
  );
};
