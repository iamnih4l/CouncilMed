/**
 * CouncilMed — Workstation Analysis Page
 * ==========================================
 * The primary workstation view: image on the left, council evidence on the right.
 * This replaces the separate MRI/CT/XRay pages with a single unified analysis view.
 *
 * Layout: [Image Viewer | Council Evidence Panel]
 * The image/study is the CENTER of the experience.
 */

import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, FileImage, Loader2, AlertCircle, ChevronRight,
  RotateCcw, Info,
  Bot, Sparkles, Brain
} from 'lucide-react';
import { GlassCard } from '../components/ui/GlassCard';
import { CouncilEvidencePanel } from '../components/council/CouncilEvidencePanel';
import { MedicalImageViewer } from '../components/viewer/MedicalImageViewer';
import { useCouncilAnalysis } from '../hooks/useCouncilAnalysis';
import { api } from '../services/api';


export default function WorkstationAnalysis() {
  const analysis = useCouncilAnalysis();
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  
  const [showAgent, setShowAgent] = useState(false);
  const [selectedModality, setSelectedModality] = useState<string>('Brain MRI');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);

    // Create image preview for standard images
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (['jpg', 'jpeg', 'png', 'bmp', 'tiff'].includes(ext || '')) {
      const reader = new FileReader();
      reader.onload = (ev) => setImagePreview(ev.target?.result as string);
      reader.readAsDataURL(file);
    } else {
      // DICOM/NIfTI — no browser preview, backend will handle
      setImagePreview(null);
    }

    // Reset file input so same file can be selected again
    if (fileInputRef.current) fileInputRef.current.value = '';

    // Run analysis with mapped modality/anatomy
    const anatomy = selectedModality.split(' ')[0].toLowerCase();
    const modality = selectedModality.split(' ')[1].replace('-', '');
    await analysis.analyzeFile(file, { modality, anatomy });
  }, [analysis, selectedModality]);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files[0];
    if (!file) return;

    setFileName(file.name);

    const ext = file.name.split('.').pop()?.toLowerCase();
    if (['jpg', 'jpeg', 'png', 'bmp', 'tiff'].includes(ext || '')) {
      const reader = new FileReader();
      reader.onload = (ev) => setImagePreview(ev.target?.result as string);
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }

    const anatomy = selectedModality.split(' ')[0].toLowerCase();
    const modality = selectedModality.split(' ')[1].replace('-', '');
    await analysis.analyzeFile(file, { modality, anatomy });
  }, [analysis, selectedModality]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleReview = useCallback(async (decision: 'accepted' | 'rejected' | 'modified') => {
    if (!analysis.result) return;
    try {
      await api.submitReview({
        analysis_id: analysis.result.request_id,
        decision,
        reviewer_name: 'Clinician',
      });
    } catch (err) {
      console.error('Review submission failed:', err);
    }
  }, [analysis.result]);

  return (
    <div className="h-full flex flex-col">
      {/* Top bar */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-400">
            Analysis Workstation
          </h1>
          <p className="text-zinc-500 text-sm mt-0.5">
            {fileName
              ? `Analyzing: ${fileName}`
              : 'Drop a medical image to begin council analysis'}
          </p>
        </div>

        {analysis.isComplete && analysis.result && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">
              {analysis.result.total_time_ms.toFixed(0)}ms total
            </span>
            <span className="text-xs text-zinc-600">•</span>
            <span className="text-xs text-zinc-500">
              {analysis.result.modality} / {analysis.result.anatomy}
            </span>
          </div>
        )}
      </div>

      {/* Main workspace: Image | Evidence */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-5 gap-4 min-h-0">

        {/* Left: Image Viewer (3/5 width) */}
        <div className="lg:col-span-3 flex flex-col min-h-0">
          {!analysis.result && !analysis.isProcessing ? (
            /* Upload Drop Zone */
            <GlassCard
              className="flex-1 flex flex-col items-center justify-center cursor-pointer min-h-[400px]
                border-dashed border-2 border-white/10 hover:border-cyan-500/40 transition-all duration-300"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".dcm,.dicom,.nii,.gz,.jpg,.jpeg,.png,.bmp,.tiff"
                className="hidden"
                onChange={handleFileSelect}
              />
              
              {/* Modality Selector (Click stops propagation so it doesn't trigger file dialog) */}
              <div 
                className="absolute top-6 flex items-center justify-center w-full z-10"
                onClick={e => e.stopPropagation()}
              >
                <div className="bg-black/50 backdrop-blur-md border border-white/10 p-1 rounded-xl flex gap-1 shadow-2xl">
                  {['Brain MRI', 'Chest CT', 'Chest X-Ray'].map(mod => (
                    <button
                      key={mod}
                      onClick={() => setSelectedModality(mod)}
                      className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        selectedModality === mod
                          ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                          : 'text-zinc-400 hover:text-white hover:bg-white/5 border border-transparent'
                      }`}
                    >
                      {mod}
                    </button>
                  ))}
                </div>
              </div>

              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="flex flex-col items-center"
              >
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-teal-500/20 
                  flex items-center justify-center mb-6 border border-cyan-500/20">
                  <Upload className="w-8 h-8 text-cyan-400" />
                </div>
                <p className="text-lg font-semibold text-white mb-2">Drop Medical Image</p>
                <p className="text-sm text-zinc-500 text-center max-w-sm">
                  DICOM (.dcm), NIfTI (.nii), JPEG, PNG supported.<br />
                  The AI Council will analyze the image with {' '}
                  <span className="text-cyan-400">multiple independent models</span>.
                </p>
                <div className="flex items-center gap-4 mt-6 text-[11px] text-zinc-600">
                  <span className="flex items-center gap-1"><FileImage className="w-3 h-3" /> DICOM</span>
                  <span className="flex items-center gap-1"><FileImage className="w-3 h-3" /> NIfTI</span>
                  <span className="flex items-center gap-1"><FileImage className="w-3 h-3" /> JPEG/PNG</span>
                </div>
              </motion.div>
            </GlassCard>
          ) : (
            /* Medical Image Viewer */
            <div className="flex-1 flex flex-col min-h-[400px] relative">
              {/* Processing overlay */}
              {analysis.isProcessing && (
                <div className="absolute inset-0 flex flex-col items-center justify-center z-30 bg-black/60 backdrop-blur-sm rounded-2xl">
                  <Loader2 className="w-10 h-10 text-cyan-400 animate-spin mb-4" />
                  <p className="text-sm text-white font-medium mb-1">Council Analysis in Progress</p>
                  <p className="text-xs text-zinc-500">
                    {analysis.status === 'uploading' ? 'Uploading image...' : 'Running model inference...'}
                  </p>
                  <div className="w-48 h-1.5 rounded-full bg-white/10 mt-4 overflow-hidden">
                    <motion.div
                      className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-teal-500"
                      style={{ width: `${analysis.progress}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  <p className="text-[11px] text-zinc-600 mt-2">{analysis.progress.toFixed(0)}%</p>
                </div>
              )}

              {/* New Image button */}
              <div className="absolute top-2 right-2 z-20">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                    bg-black/60 text-zinc-400 hover:bg-black/80 hover:text-white transition-all backdrop-blur-sm"
                >
                  <Upload className="w-3.5 h-3.5" /> New Image
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".dcm,.dicom,.nii,.gz,.jpg,.jpeg,.png,.bmp,.tiff"
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </div>

              <GlassCard className="flex-1 flex flex-col p-0 overflow-hidden">
                <MedicalImageViewer
                  imageUrl={imagePreview}
                  fileName={fileName}
                  imageShape={analysis.result?.image_shape}
                  metadata={analysis.result ? {
                    'Modality': analysis.result.modality,
                    'Anatomy': analysis.result.anatomy,
                    'Image Shape': analysis.result.image_shape.join(' × '),
                    'Device': (analysis.result.hardware?.device as string) || 'cpu',
                    'Total Time': `${analysis.result.total_time_ms.toFixed(0)}ms`,
                    'Models Used': `${analysis.result.council_result.models_total}`,
                  } : undefined}
                  className="flex-1"
                />
              </GlassCard>
            </div>
          )}
        </div>

        {/* Right: Council Evidence Panel (2/5 width) */}
        <div className="lg:col-span-2 flex flex-col min-h-0 overflow-y-auto">
          {analysis.isError && (
            <GlassCard className="border-red-500/20 bg-red-500/5">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-red-400">Analysis Failed</p>
                  <p className="text-xs text-zinc-400 mt-1">{analysis.error}</p>
                  <button
                    onClick={analysis.reset}
                    className="mt-3 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                      bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-white transition-all"
                  >
                    <RotateCcw className="w-3.5 h-3.5" /> Try Again
                  </button>
                </div>
              </div>
            </GlassCard>
          )}

          {analysis.isComplete && analysis.result && (
            <>
              {/* Warnings */}
              {analysis.result.warnings.length > 0 && (
                <GlassCard className="border-amber-500/10 mb-4">
                  {analysis.result.warnings.map((w, i) => (
                    <p key={i} className="text-xs text-amber-400 flex items-start gap-1.5">
                      <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
                      {w}
                    </p>
                  ))}
                </GlassCard>
              )}

              <CouncilEvidencePanel
                result={analysis.result.council_result}
                onReview={handleReview}
              />

              {/* Agent Insight (if available) */}
              {analysis.result.agent_insight && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4"
                >
                  <button
                    onClick={() => setShowAgent(!showAgent)}
                    className="flex items-center gap-2 w-full text-left mb-2"
                  >
                    <Bot className="w-4 h-4 text-purple-400" />
                    <span className="text-[11px] text-zinc-500 uppercase tracking-wider">
                      AI Agent Insights
                    </span>
                    <ChevronRight className={`w-3 h-3 text-zinc-600 transition-transform ${showAgent ? 'rotate-90' : ''}`} />
                  </button>

                  <AnimatePresence>
                    {showAgent && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                      >
                        <GlassCard className="border-purple-500/20">
                          <div className="space-y-3">
                            {analysis.result.agent_insight.key_insights.length > 0 && (
                              <div>
                                <p className="text-[11px] text-purple-400 uppercase tracking-wider mb-1.5">Key Insights</p>
                                {analysis.result.agent_insight.key_insights.map((insight, i) => (
                                  <p key={i} className="text-xs text-zinc-300 mb-1 flex items-start gap-1.5">
                                    <Sparkles className="w-3 h-3 text-purple-400 mt-0.5 flex-shrink-0" />
                                    {insight}
                                  </p>
                                ))}
                              </div>
                            )}

                            {analysis.result.agent_insight.safety_flags.length > 0 && (
                              <div className="pt-2 border-t border-white/5">
                                <p className="text-[11px] text-red-400 uppercase tracking-wider mb-1.5">Safety Flags</p>
                                {analysis.result.agent_insight.safety_flags.map((flag, i) => (
                                  <p key={i} className="text-xs text-red-300 mb-1 flex items-start gap-1.5">
                                    <AlertCircle className="w-3 h-3 text-red-400 mt-0.5 flex-shrink-0" />
                                    {flag}
                                  </p>
                                ))}
                              </div>
                            )}

                            {analysis.result.agent_insight.recommended_actions.length > 0 && (
                              <div className="pt-2 border-t border-white/5">
                                <p className="text-[11px] text-cyan-400 uppercase tracking-wider mb-1.5">Recommended Actions</p>
                                {analysis.result.agent_insight.recommended_actions.map((action, i) => (
                                  <p key={i} className="text-xs text-zinc-300 mb-1 flex items-start gap-1.5">
                                    <ChevronRight className="w-3 h-3 text-cyan-400 mt-0.5 flex-shrink-0" />
                                    {action}
                                  </p>
                                ))}
                              </div>
                            )}

                            <div className="pt-2 border-t border-white/5 flex items-center justify-between text-[11px] text-zinc-600">
                              <span>{analysis.result.agent_insight.agent_model}</span>
                              <span>{analysis.result.agent_insight.processing_time_ms.toFixed(0)}ms</span>
                            </div>
                          </div>
                        </GlassCard>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}
            </>
          )}

          {/* Idle state: instructions */}
          {analysis.status === 'idle' && (
            <GlassCard className="flex-1 flex flex-col items-center justify-center text-center">
              <Brain className="w-10 h-10 text-zinc-700 mb-4" />
              <p className="text-sm text-zinc-500 font-medium mb-2">Council Evidence Panel</p>
              <p className="text-xs text-zinc-600 max-w-xs">
                Upload a medical image to see real-time evidence from multiple AI models,
                their agreement analysis, and council consensus.
              </p>
            </GlassCard>
          )}
        </div>
      </div>
    </div>
  );
}
