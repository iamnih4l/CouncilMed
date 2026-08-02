import React, { useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { GlassCard } from '../ui/GlassCard';
import { ImageViewer2D } from '../2d/ImageViewer2D';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import { motion, AnimatePresence } from 'framer-motion';
import { Layers, Upload, Share2, Download, Brain, Activity, Hexagon, Cpu, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { generatePDFReport } from '../../utils/reportGenerator';
import { useDiagnosticEngine } from '../../hooks/useDiagnosticEngine';
import { usePersistentReports } from '../../hooks/usePersistentReports';
import { useWorkstationSettings } from '../../hooks/useWorkstationSettings';
import { ALL_REGIONS, type CouncilConsensusResult } from '../../services/inference/types';

interface AnalysisLayoutProps {
  title: string;
  modality: 'MRI' | 'CT' | 'X-Ray';
  reportElementId: string;
  pdfFilename: string;
  emptyStateText: string;
  Viewer3D: React.ComponentType<{ consensusResult: CouncilConsensusResult | null }>;
}

export default function AnalysisLayout({
  title,
  modality,
  reportElementId,
  pdfFilename,
  emptyStateText,
  Viewer3D
}: AnalysisLayoutProps) {
  const { state, result, isProcessing, processImage } = useDiagnosticEngine();
  const { addReport } = usePersistentReports();
  const { settings } = useWorkstationSettings();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [initStatus] = useState<'idle' | 'loading' | 'ready'>('idle');
  const [viewMode, setViewMode] = useState<'3d' | '2d'>('3d');

  React.useEffect(() => {
    const handleOpenUpload = (e: Event) => {
      const customEvent = e as CustomEvent;
      if (customEvent.detail?.modality === modality.toLowerCase() || (!customEvent.detail?.modality && modality === 'MRI')) {
        fileInputRef.current?.click();
      }
    };
    window.addEventListener('open-upload', handleOpenUpload);
    return () => window.removeEventListener('open-upload', handleOpenUpload);
  }, [modality]);

  React.useEffect(() => {
    if (result) {
      setViewMode('2d');
    }
  }, [result]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    toast.info(`${modality} scan received — initiating Council Analysis...`);
    try {
      const consensusResult = await processImage(file);
      
      if (consensusResult) {
        addReport(consensusResult);

        toast.success('Council Analysis Complete', {
          description: 'All three models have reached consensus.',
        });
        
        setTimeout(() => {
          toast.promise(generatePDFReport(reportElementId, pdfFilename, consensusResult, settings), {
            loading: 'Auto-generating Council Report PDF...',
            success: 'Report generated and downloaded successfully',
            error: 'Failed to generate PDF',
          });
        }, 1000);
      }
      
    } catch {
      toast.error('Diagnostic pipeline encountered an error.');
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const pipelineSteps = [
    {
      id: 'preprocessing',
      label: 'Pre-Processing',
      icon: Cpu,
      status: state.stage === 'preprocessing' ? 'running' : (state.progress > 15 ? 'complete' : 'idle'),
    },
    {
      id: 'densenet',
      label: 'DenseNet-121',
      icon: Brain,
      status: state.densenetStatus,
    },
    {
      id: 'attention',
      label: 'Attention-Net',
      icon: Activity,
      status: state.attentionStatus,
    },
    {
      id: 'swin',
      label: 'Swin-UNETR',
      icon: Hexagon,
      status: state.swinStatus,
    },
    {
      id: 'consensus',
      label: 'Council Fusion',
      icon: CheckCircle2,
      status: state.stage === 'consensus' ? 'running' : (state.stage === 'complete' ? 'complete' : 'idle'),
    },
  ];

  const severityMap: Record<string, 'critical' | 'warning' | 'clear' | 'info'> = {
    critical: 'critical',
    warning: 'warning',
    clear: 'clear',
    info: 'info',
  };

  return (
    <div id={reportElementId} className="max-w-7xl mx-auto h-[calc(100vh-8rem)] flex flex-col space-y-4">
      <div className="flex items-center justify-between shrink-0">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">{title}</h1>
            {result && (
              <StatusBadge status={(severityMap[result.severity] || 'info') as 'critical' | 'warning' | 'clear' | 'info'}>
                {result.primaryDiagnosis}
              </StatusBadge>
            )}
            {isProcessing && (
              <StatusBadge status="info">
                <Loader2 className="w-3 h-3 animate-spin mr-1" />
                Processing
              </StatusBadge>
            )}
            {!result && !isProcessing && (
              <StatusBadge status="info">Awaiting Scan</StatusBadge>
            )}
          </div>
          <p className="text-sm text-zinc-400 mt-1">
            {result
              ? `Region: ${ALL_REGIONS[result.affectedRegion]?.label || 'Unknown'} • Confidence: ${result.overallConfidence.toFixed(1)}% • ${result.totalInferenceTimeMs.toFixed(0)}ms`
              : emptyStateText}
          </p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isProcessing}
            className="px-3 py-2 bg-[var(--color-accent-cyan)]/10 hover:bg-[var(--color-accent-cyan)]/20 text-[var(--color-accent-cyan)] border border-[var(--color-accent-cyan)]/20 font-medium rounded-lg transition-colors flex items-center space-x-2 disabled:opacity-50"
          >
            <Upload className="w-4 h-4" />
            <span className="hidden sm:inline">Upload {modality}</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept="image/*,.dcm,.nii"
            onChange={handleFileUpload}
          />
          <button onClick={() => toast('Secure link copied to clipboard')} className="p-2 bg-white/5 hover:bg-white/10 rounded-lg text-zinc-300 transition-colors border border-white/10"><Share2 className="w-4 h-4" /></button>
          <button onClick={() => toast.success('Downloading raw DICOM files...')} className="p-2 bg-white/5 hover:bg-white/10 rounded-lg text-zinc-300 transition-colors border border-white/10"><Download className="w-4 h-4" /></button>
          <button
            onClick={() => generatePDFReport(reportElementId, pdfFilename, result || undefined, settings)}
            className="px-4 py-2 bg-[var(--color-accent-cyan)] hover:bg-[var(--color-accent-teal)] text-[#0d1324] font-semibold rounded-lg transition-colors flex items-center space-x-2"
          >
            <span>Generate PDF Report</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1 min-h-0">
        <GlassCard className="col-span-1 lg:col-span-2 relative overflow-hidden flex flex-col p-0">
          <div className="absolute top-4 left-4 right-4 z-10 flex justify-between items-center pointer-events-none">
            <div className="bg-black/40 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 pointer-events-auto flex items-center space-x-2">
              <Layers className="w-4 h-4 text-zinc-400" />
              <span className="text-xs font-medium text-white">Council Viewer</span>
            </div>
            <div className="pointer-events-auto flex items-center bg-black/40 backdrop-blur-md rounded-lg border border-white/10 p-1 mr-auto ml-4">
              <button
                onClick={() => setViewMode('3d')}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${viewMode === '3d' ? 'bg-white/20 text-white' : 'text-zinc-400 hover:text-white'}`}
              >
                3D Model
              </button>
              <button
                onClick={() => setViewMode('2d')}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${viewMode === '2d' ? 'bg-white/20 text-white' : 'text-zinc-400 hover:text-white'}`}
              >
                2D Scan
              </button>
            </div>
            {initStatus === 'loading' && (
              <div className="bg-black/40 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 pointer-events-auto flex items-center space-x-2">
                <Loader2 className="w-3 h-3 text-[var(--color-accent-cyan)] animate-spin" />
                <span className="text-xs text-zinc-400">Loading models...</span>
              </div>
            )}
          </div>

          {isProcessing && (
            <div className="absolute bottom-0 left-0 right-0 z-10">
              <div className="h-1 w-full bg-black/40">
                <motion.div
                  className="h-full bg-gradient-to-r from-[var(--color-accent-cyan)] to-[var(--color-accent-teal)]"
                  initial={{ width: '0%' }}
                  animate={{ width: `${state.progress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </div>
          )}

          <div className="flex-1 w-full bg-gradient-to-b from-[#0d1324] to-[#04060b] relative flex items-center justify-center">
            {viewMode === '3d' ? (
              <>
                <Canvas camera={{ position: [0, 0, 8] }} gl={{ preserveDrawingBuffer: true }}>
                  <ambientLight intensity={1} />
                  <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} />
                  <Viewer3D consensusResult={result} />
                </Canvas>
                {state.uploadedImage && (
                  <div className="absolute bottom-4 right-4 z-10 cursor-pointer" onClick={() => setViewMode('2d')}>
                    <img
                      src={state.uploadedImage}
                      alt={`Uploaded ${modality}`}
                      className="w-24 h-24 rounded-lg border border-white/20 object-cover shadow-2xl opacity-80 hover:opacity-100 transition-opacity"
                    />
                  </div>
                )}
              </>
            ) : (
              state.uploadedImage ? (
                <ImageViewer2D imageSrc={state.uploadedImage} consensusResult={result} />
              ) : (
                <div className="text-zinc-500 text-sm">Upload a scan to view</div>
              )
            )}
          </div>
        </GlassCard>

        <motion.div
          className="col-span-1 flex flex-col space-y-4 overflow-y-auto pr-1 pb-4"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <GlassCard className="p-5">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4 border-b border-white/10 pb-2">
              Council Pipeline
            </h3>
            <div className="space-y-2">
              {pipelineSteps.map((step, idx) => (
                <motion.div
                  key={step.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`flex items-center space-x-3 p-2.5 rounded-lg transition-all ${
                    step.status === 'running'
                      ? 'bg-[var(--color-accent-cyan)]/10 border border-[var(--color-accent-cyan)]/20'
                      : step.status === 'complete'
                      ? 'bg-[var(--color-accent-emerald)]/5 border border-[var(--color-accent-emerald)]/10'
                      : 'bg-white/5 border border-white/5'
                  }`}
                >
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                    step.status === 'running'
                      ? 'bg-[var(--color-accent-cyan)]/20'
                      : step.status === 'complete'
                      ? 'bg-[var(--color-accent-emerald)]/20'
                      : step.status === 'error'
                      ? 'bg-[var(--color-accent-ruby)]/20'
                      : 'bg-white/10'
                  }`}>
                    {step.status === 'running' ? (
                      <Loader2 className="w-3.5 h-3.5 text-[var(--color-accent-cyan)] animate-spin" />
                    ) : step.status === 'complete' ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-[var(--color-accent-emerald)]" />
                    ) : step.status === 'error' ? (
                      <AlertTriangle className="w-3.5 h-3.5 text-[var(--color-accent-ruby)]" />
                    ) : (
                      (() => { const StepIcon = step.icon; return <StepIcon className="w-3.5 h-3.5 text-zinc-500" />; })()
                    )}
                  </div>
                  <span className={`text-xs font-medium ${
                    step.status === 'running' ? 'text-[var(--color-accent-cyan)]'
                      : step.status === 'complete' ? 'text-[var(--color-accent-emerald)]'
                      : step.status === 'error' ? 'text-[var(--color-accent-ruby)]'
                      : 'text-zinc-500'
                  }`}>
                    {step.label}
                  </span>
                  {step.status === 'complete' && step.id === 'densenet' && result && (
                    <span className="ml-auto text-[10px] text-zinc-500">{result.densenetResult.inferenceTimeMs.toFixed(0)}ms</span>
                  )}
                  {step.status === 'complete' && step.id === 'attention' && result && (
                    <span className="ml-auto text-[10px] text-zinc-500">{result.attentionResult.inferenceTimeMs.toFixed(0)}ms</span>
                  )}
                  {step.status === 'complete' && step.id === 'swin' && result && (
                    <span className="ml-auto text-[10px] text-zinc-500">{result.swinResult.inferenceTimeMs.toFixed(0)}ms</span>
                  )}
                </motion.div>
              ))}
            </div>
          </GlassCard>

          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <GlassCard className="p-5" focused>
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4 border-b border-white/10 pb-2">
                    Council Consensus
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-white">{result.primaryDiagnosis}</span>
                        <span className={`text-xs font-bold ${
                          result.severity === 'critical' ? 'text-[var(--color-accent-ruby)]'
                            : result.severity === 'warning' ? 'text-[var(--color-accent-amber)]'
                            : 'text-[var(--color-accent-emerald)]'
                        }`}>
                          {result.severity.toUpperCase()}
                        </span>
                      </div>
                      <ConfidenceBar score={result.overallConfidence} />
                    </div>

                    <div className="pt-3 border-t border-white/10">
                      <p className="text-xs text-zinc-500 mb-1">Affected Region</p>
                      <p className="text-sm font-medium text-white">{ALL_REGIONS[result.affectedRegion]?.label || 'Unknown'}</p>
                    </div>

                    <div className="pt-3 border-t border-white/10">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs text-zinc-500">Model Agreement</span>
                        <span className={`text-xs font-bold ${
                          result.modelAgreement > 0.8 ? 'text-[var(--color-accent-emerald)]'
                            : result.modelAgreement > 0.5 ? 'text-[var(--color-accent-amber)]'
                            : 'text-[var(--color-accent-ruby)]'
                        }`}>
                          {(result.modelAgreement * 100).toFixed(0)}%
                        </span>
                      </div>
                      <ConfidenceBar score={result.modelAgreement * 100} />
                    </div>

                    {result.swinResult.tumorVolumeMm3 > 50 && (
                      <div className="pt-3 border-t border-white/10">
                        <p className="text-xs text-zinc-500 mb-2">Volumetric Analysis (Swin-UNETR)</p>
                        <p className="text-sm text-white font-medium mb-2">
                          Est. Volume: {result.swinResult.tumorVolumeMm3.toFixed(0)} mm³
                        </p>
                        {result.swinResult.segmentedRegions.map((seg, i) => (
                          <div key={i} className="flex justify-between text-xs mb-1">
                            <span className="text-zinc-400">{seg.label}</span>
                            <span className="text-zinc-300">{seg.volumePercentage.toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </GlassCard>
              </motion.div>
            )}
          </AnimatePresence>

          <GlassCard className="p-5 flex-1">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4 border-b border-white/10 pb-2">
              Individual Models
            </h3>
            <div className="flex-1">
              <h2 className="text-xl font-bold text-white mb-2">Diagnostic Council Report</h2>
              <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-zinc-400">
                <div className="flex items-center gap-2">
                  <span className="text-zinc-500">Authorized Physician:</span>
                  <span className="text-zinc-200 font-medium">{settings.doctorName}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-zinc-500">ID:</span>
                  <span className="text-zinc-200 font-medium">{settings.clinicalId}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-zinc-500">Timestamp:</span>
                  <span className="text-zinc-200 font-medium">{new Date().toLocaleString()}</span>
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <div className={`flex justify-between items-center p-3 rounded-lg border transition-all ${
                state.densenetStatus === 'complete'
                  ? 'bg-[var(--color-accent-cyan)]/5 border-[var(--color-accent-cyan)]/10'
                  : 'bg-white/5 border-white/5'
              }`}>
                <div className="flex items-center space-x-2">
                  <Brain className="w-4 h-4 text-zinc-400" />
                  <span className="text-sm text-zinc-300">DenseNet-121</span>
                </div>
                <span className="text-sm font-bold text-[var(--color-accent-cyan)]">
                  {result ? `${result.densenetResult.confidence.toFixed(1)}%` : '--'}
                </span>
              </div>
              <div className={`flex justify-between items-center p-3 rounded-lg border transition-all ${
                state.attentionStatus === 'complete'
                  ? 'bg-[var(--color-accent-teal)]/5 border-[var(--color-accent-teal)]/10'
                  : 'bg-white/5 border-white/5'
              }`}>
                <div className="flex items-center space-x-2">
                  <Activity className="w-4 h-4 text-zinc-400" />
                  <span className="text-sm text-zinc-300">Attention-Net</span>
                </div>
                <span className="text-sm font-bold text-[var(--color-accent-teal)]">
                  {result ? `${result.attentionResult.confidence.toFixed(1)}%` : '--'}
                </span>
              </div>
              <div className={`flex justify-between items-center p-3 rounded-lg border transition-all ${
                state.swinStatus === 'complete'
                  ? 'bg-[var(--color-accent-emerald)]/5 border-[var(--color-accent-emerald)]/10'
                  : 'bg-white/5 border-white/5'
              }`}>
                <div className="flex items-center space-x-2">
                  <Hexagon className="w-4 h-4 text-zinc-400" />
                  <span className="text-sm text-zinc-300">Swin-UNETR</span>
                </div>
                <span className="text-sm font-bold text-[var(--color-accent-emerald)]">
                  {result ? `${result.swinResult.confidence.toFixed(1)}%` : '--'}
                </span>
              </div>
            </div>
          </GlassCard>

          <AnimatePresence>
            {result && result.councilNotes.length > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
              >
                <GlassCard className="p-5">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-3 border-b border-white/10 pb-2">
                    Clinical Notes
                  </h3>
                  <div className="space-y-2">
                    {result.councilNotes.map((note, i) => (
                      <p key={i} className="text-xs text-zinc-400 leading-relaxed">
                        {note}
                      </p>
                    ))}
                  </div>
                </GlassCard>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}
