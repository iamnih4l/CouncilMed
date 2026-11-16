/**
 * CouncilMed — Multi-Model Explainability Matrix Modal
 * ====================================================
 * Renders a side-by-side comparative grid of all council members'
 * Grad-CAM activation heatmaps alongside the original baseline scan.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { X, Flame, Layers, Maximize2, Cpu, CheckCircle } from 'lucide-react';
import type { ModelEvidenceResponse } from '../../services/api';

export interface OverlayItem {
  model_id: string;
  url: string;
  overlay_type: string;
}

interface MultiHeatmapModalProps {
  isOpen: boolean;
  onClose: () => void;
  imageUrl: string | null;
  overlays: OverlayItem[];
  memberEvidence?: ModelEvidenceResponse[];
  onSelectOverlay: (index: number) => void;
  selectedOverlayIdx: number;
}

export function MultiHeatmapModal({
  isOpen,
  onClose,
  imageUrl,
  overlays,
  memberEvidence = [],
  onSelectOverlay,
  selectedOverlayIdx,
}: MultiHeatmapModalProps) {
  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 md:p-8">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="absolute inset-0 bg-black/85 backdrop-blur-xl"
        />

        {/* Modal Window */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 15 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 15 }}
          className="relative w-full max-w-6xl max-h-[90vh] flex flex-col rounded-2xl bg-gradient-to-b from-zinc-950/95 to-black/95 border border-cyan-500/30 shadow-[0_0_50px_rgba(0,240,255,0.15)] overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/[0.02]">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500/20 to-teal-500/10 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-[0_0_15px_rgba(0,240,255,0.2)]">
                <Flame className="w-5 h-5 text-cyan-300" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-semibold tracking-wider text-white uppercase font-mono">
                    Multi-Model Explainability Matrix
                  </h2>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono tracking-widest uppercase bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                    {overlays.length} Council Overlays
                  </span>
                </div>
                <p className="text-xs text-zinc-400 font-mono mt-0.5">
                  Synchronized Grad-CAM spatial activation maps across all independent neural architectures
                </p>
              </div>
            </div>

            <button
              onClick={onClose}
              className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/10 transition-colors btn-tactile"
              title="Close Matrix"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Grid Content */}
          <div className="flex-1 overflow-y-auto p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Card 0: Baseline Clinical Scan */}
              <div className="flex flex-col rounded-xl bg-white/[0.02] border border-white/10 overflow-hidden shadow-lg group hover:border-cyan-500/40 transition-colors">
                <div className="px-3 py-2 border-b border-white/5 bg-black/40 flex items-center justify-between">
                  <span className="text-[11px] font-mono tracking-wider text-zinc-300 uppercase flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-zinc-400" />
                    Baseline Scan
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-white/5 text-zinc-400 border border-white/10">
                    Original
                  </span>
                </div>

                <div className="relative aspect-square bg-black flex items-center justify-center overflow-hidden">
                  {imageUrl ? (
                    <img
                      src={imageUrl}
                      alt="Baseline scan"
                      className="w-full h-full object-contain p-2"
                    />
                  ) : (
                    <span className="text-xs text-zinc-600 font-mono">No Image</span>
                  )}
                </div>

                <div className="p-3 border-t border-white/5 bg-black/30 flex items-center justify-between">
                  <div>
                    <p className="text-xs font-semibold text-white">Unfiltered Input</p>
                    <p className="text-[10px] text-zinc-500 font-mono">Native Pixel Density</p>
                  </div>
                  <button
                    onClick={() => {
                      onSelectOverlay(-1);
                      onClose();
                    }}
                    className="px-2.5 py-1 rounded-md text-[10px] font-mono uppercase bg-white/5 hover:bg-white/10 text-zinc-300 border border-white/10 transition-all btn-tactile"
                  >
                    View Clean
                  </button>
                </div>
              </div>

              {/* Cards 1..N: Each Model's Grad-CAM Heatmap */}
              {overlays.map((ov, idx) => {
                const evidence = memberEvidence.find(e => e.model_id === ov.model_id);
                const isSelected = selectedOverlayIdx === idx;

                return (
                  <div
                    key={ov.model_id}
                    className={`flex flex-col rounded-xl overflow-hidden shadow-lg transition-all ${
                      isSelected
                        ? 'bg-cyan-950/20 border-2 border-cyan-400 shadow-[0_0_20px_rgba(0,240,255,0.2)]'
                        : 'bg-white/[0.02] border border-white/10 hover:border-cyan-500/40'
                    }`}
                  >
                    {/* Tile Header */}
                    <div className="px-3 py-2 border-b border-white/5 bg-black/40 flex items-center justify-between">
                      <span className="text-[11px] font-mono tracking-wider text-cyan-300 uppercase flex items-center gap-1.5 truncate max-w-[170px]">
                        <Flame className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                        <span className="truncate">{evidence?.model_name || ov.model_id}</span>
                      </span>
                      {isSelected ? (
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 flex items-center gap-1">
                          <CheckCircle className="w-2.5 h-2.5" /> Active
                        </span>
                      ) : (
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-white/5 text-zinc-400 border border-white/10">
                          {ov.overlay_type.toUpperCase()}
                        </span>
                      )}
                    </div>

                    {/* Heatmap Overlaid on Base Scan */}
                    <div className="relative aspect-square bg-black flex items-center justify-center overflow-hidden">
                      {imageUrl && (
                        <img
                          src={imageUrl}
                          alt="Base scan"
                          className="absolute inset-0 w-full h-full object-contain p-2 opacity-85"
                        />
                      )}
                      <img
                        src={ov.url}
                        alt={`${ov.model_id} heatmap`}
                        className="absolute inset-0 w-full h-full object-contain p-2 mix-blend-screen opacity-75"
                      />
                    </div>

                    {/* Tile Footer */}
                    <div className="p-3 border-t border-white/5 bg-black/30 flex items-center justify-between">
                      <div className="min-w-0 pr-2">
                        <p className="text-xs font-semibold text-white truncate">
                          {evidence?.prediction || 'Localized ROI'}
                        </p>
                        <p className="text-[10px] text-cyan-400 font-mono">
                          {evidence?.confidence ? `${(evidence.confidence * 100).toFixed(1)}% conf` : 'Grad-CAM Saliency'}
                        </p>
                      </div>

                      <button
                        onClick={() => {
                          onSelectOverlay(idx);
                          onClose();
                        }}
                        className={`px-2.5 py-1 rounded-md text-[10px] font-mono uppercase flex items-center gap-1 transition-all btn-tactile shrink-0 ${
                          isSelected
                            ? 'bg-cyan-500 text-black font-semibold shadow-[0_0_10px_rgba(0,240,255,0.4)]'
                            : 'bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        }`}
                      >
                        <Maximize2 className="w-2.5 h-2.5" />
                        <span>Focus</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Modal Footer */}
          <div className="px-6 py-3 border-t border-white/10 bg-black/60 flex items-center justify-between text-xs font-mono text-zinc-400">
            <div className="flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>Multi-Model Parallel Inference Active</span>
            </div>
            <span>Click &apos;Focus&apos; on any tile to zoom and inspect in the main workstation</span>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
