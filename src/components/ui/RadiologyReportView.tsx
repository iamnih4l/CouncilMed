import React from 'react';
import { GlassCard } from './GlassCard';
import type { RadiologyReportData, CouncilConsensusResult } from '../../services/inference/types';
import { FileText, Shield, AlertCircle, Info, Stethoscope } from 'lucide-react';

interface Props {
  report: RadiologyReportData;
  result: CouncilConsensusResult;
}

export const RadiologyReportView: React.FC<Props> = ({ report, result }) => {
  return (
    <GlassCard className="p-8 bg-white text-zinc-900 border-2 border-zinc-200 shadow-2xl relative overflow-hidden">
      {/* Official Watermark */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-[0.03] rotate-[-25deg] pointer-events-none scale-[2]">
        <h1 className="text-9xl font-black tracking-tighter">COUNCILAI</h1>
      </div>

      {/* Header */}
      <div className="flex justify-between items-start border-b-2 border-zinc-900 pb-6 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Shield className="w-8 h-8 text-zinc-900" />
            <h1 className="text-3xl font-black tracking-tighter uppercase">CouncilAI</h1>
          </div>
          <p className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Advanced Clinical Diagnostics Platform</p>
        </div>
        <div className="text-right">
          <h2 className="text-lg font-black uppercase tracking-tight">Official Diagnostic Report</h2>
          <p className="text-sm font-medium text-zinc-600">{new Date().toLocaleDateString()}</p>
          <div className="mt-2 flex items-center justify-end gap-2 text-[10px] bg-zinc-100 px-2 py-1 rounded border border-zinc-200">
            <span className="font-bold text-zinc-500 uppercase">Fingerprint:</span>
            <span className="font-mono text-zinc-800">{result.totalInferenceTimeMs.toFixed(0)}_MS_ENGINE</span>
          </div>
        </div>
      </div>

      {/* Patient & Study Info */}
      <div className="grid grid-cols-2 gap-x-12 gap-y-4 mb-8">
        <div>
          <h3 className="text-[10px] font-black text-zinc-400 uppercase tracking-widest mb-2">Study Information</h3>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between border-b border-zinc-100 pb-1">
              <span className="text-zinc-500">Modality:</span>
              <span className="font-bold">{report.modality}</span>
            </div>
            <div className="flex justify-between border-b border-zinc-100 pb-1">
              <span className="text-zinc-500">Study Type:</span>
              <span className="font-bold">{report.studyType} weighted</span>
            </div>
            <div className="flex justify-between border-b border-zinc-100 pb-1">
              <span className="text-zinc-500">Data Quality:</span>
              <span className="font-bold text-emerald-600">{report.dataQuality}</span>
            </div>
          </div>
        </div>
        <div>
          <h3 className="text-[10px] font-black text-zinc-400 uppercase tracking-widest mb-2">Clinical Context</h3>
          <div className="space-y-1 text-sm text-zinc-700">
            <p><strong>Clinical Indication:</strong> {report.clinicalIndication}</p>
          </div>
        </div>
      </div>

      {/* Technique */}
      <div className="mb-8">
        <h3 className="text-sm font-black text-zinc-900 uppercase tracking-tight mb-2 border-b border-zinc-900 flex items-center gap-2">
          <Stethoscope className="w-4 h-4" />
          Technique
        </h3>
        <p className="text-sm text-zinc-700 leading-relaxed">{report.technique}</p>
      </div>

      {/* Findings */}
      <div className="mb-8">
        <h3 className="text-sm font-black text-zinc-900 uppercase tracking-tight mb-3 border-b border-zinc-900 flex items-center gap-2">
          <FileText className="w-4 h-4" />
          Findings
        </h3>
        
        <div className="space-y-4">
          <div>
            <h4 className="text-xs font-black text-zinc-900 uppercase mb-1 underline">Brain Parenchyma</h4>
            <p className="text-sm text-zinc-700 leading-relaxed">{report.findings.brainParenchyma}</p>
          </div>

          {report.findings.lesionAssessment.included && (
            <div>
              <h4 className="text-xs font-black text-zinc-900 uppercase mb-1 underline">Lesion Assessment</h4>
              <ul className="text-sm text-zinc-700 space-y-1 ml-4 list-disc">
                <li><strong>Location:</strong> {report.findings.lesionAssessment.location}</li>
                <li><strong>Size/Volume:</strong> {report.findings.lesionAssessment.sizeVolume}</li>
                <li><strong>Characteristics:</strong> {report.findings.lesionAssessment.characteristics}</li>
              </ul>
            </div>
          )}

          <div className="grid grid-cols-3 gap-4 border-t border-zinc-100 pt-3 mt-4">
            <div>
              <h4 className="text-[10px] font-black text-zinc-400 uppercase mb-1">Edema / Mass Effect</h4>
              <p className="text-xs font-bold text-zinc-900">{report.findings.edemaMassEffect}</p>
            </div>
            <div>
              <h4 className="text-[10px] font-black text-zinc-400 uppercase mb-1">Ventricular System</h4>
              <p className="text-xs font-bold text-zinc-900">{report.findings.ventricularSystem}</p>
            </div>
            <div>
              <h4 className="text-[10px] font-black text-zinc-400 uppercase mb-1">Midline structures</h4>
              <p className="text-xs font-bold text-zinc-900">Shift: {report.findings.midlineStructures.shift}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Impression */}
      <div className="mb-8 bg-zinc-50 p-4 border border-zinc-200 rounded-lg">
        <h3 className="text-sm font-black text-zinc-900 uppercase tracking-tight mb-2 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-zinc-900" />
          Impression
        </h3>
        <p className="text-md font-bold text-zinc-900 leading-relaxed uppercase tracking-tight">
          {report.impression}
        </p>
      </div>

      {/* AI Confidence Summary */}
      <div className="mb-8 grid grid-cols-2 gap-8 text-[11px] border-t border-zinc-200 pt-6">
        <div>
          <h4 className="font-black text-zinc-400 uppercase tracking-widest mb-3">AI Confidence Scores</h4>
          <div className="space-y-1.5">
            <div className="flex justify-between items-center bg-zinc-50 px-2 py-1 rounded">
              <span className="text-zinc-500">Classification (DenseNet-121):</span>
              <span className="font-bold">{result.densenetResult.confidence.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center bg-zinc-50 px-2 py-1 rounded">
              <span className="text-zinc-500">Localization (Attention-Net):</span>
              <span className="font-bold">{result.attentionResult.confidence.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center bg-zinc-50 px-2 py-1 rounded">
              <span className="text-zinc-500">Segmentation (Swin-UNETR):</span>
              <span className="font-bold">{result.swinResult.confidence.toFixed(1)}%</span>
            </div>
          </div>
        </div>
        <div>
          <h4 className="font-black text-zinc-400 uppercase tracking-widest mb-3">Interpretation Metrics</h4>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-zinc-900"></div>
              <span className="text-zinc-600">Model Agreement:</span>
              <span className="font-bold uppercase">{(result.modelAgreement * 100).toFixed(0)}% ({result.modelAgreement > 0.8 ? 'High' : (result.modelAgreement > 0.5 ? 'Moderate' : 'Low')})</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-zinc-400"></div>
              <span className="text-zinc-600 italic">This analysis is based on limited AI interpretation.</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations & Limitations */}
      <div className="mb-10 grid grid-cols-2 gap-8">
        <div className="bg-emerald-50 border border-emerald-100 p-3 rounded-lg">
          <h4 className="text-[10px] font-black text-emerald-800 uppercase tracking-widest mb-2 flex items-center gap-1">
             <Info className="w-3 h-3" />
             Clinical Recommendation
          </h4>
          <ul className="text-[11px] text-emerald-900 space-y-1 list-disc ml-4 font-medium italic">
            {report.recommendations.map((rec, i) => <li key={i}>{rec}</li>)}
          </ul>
        </div>
        <div className="bg-zinc-50 border border-zinc-100 p-3 rounded-lg">
          <h4 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-2">Technical Limitations</h4>
          <ul className="text-[11px] text-zinc-600 space-y-1 list-disc ml-4 italic">
            {report.limitations.map((lim, i) => <li key={i}>{lim}</li>)}
          </ul>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="text-center pt-6 border-t-2 border-zinc-900">
        <p className="text-[10px] font-black text-zinc-900 uppercase tracking-[0.2em] mb-1">
          MANDATORY CLINICAL DISCLAIMER
        </p>
        <p className="text-[10px] leading-relaxed text-zinc-600 italic">
          This report is AI-assisted and does NOT constitute a medical diagnosis. Final interpretation must be performed by a licensed medical professional. 
          Use of this automated interpretation system is subject to the Hospital Clinical Governance Agreement.
        </p>
      </div>
    </GlassCard>
  );
};
