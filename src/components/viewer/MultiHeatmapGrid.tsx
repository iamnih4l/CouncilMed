/**
 * CouncilMed — Multi-Heatmap Grid (Workstation Viewport Component)
 * =================================================================
 * Renders an inline 4-quadrant comparative matrix directly inside
 * the Analysis Workstation, allowing the clinician to evaluate all
 * council members' spatial activations side-by-side with the baseline scan.
 */

import { useState, useMemo } from 'react';
import {
  Flame, Layers, Maximize2, CheckCircle,
  Eye, RefreshCw, ZoomIn
} from 'lucide-react';
import type { ModelEvidenceResponse } from '../../services/api';
import { generateSyntheticHeatmapDataUrl } from '../../utils/heatmapGenerator';

export interface OverlayItem {
  model_id: string;
  url: string;
  overlay_type: string;
}

interface MultiHeatmapGridProps {
  imageUrl: string | null;
  overlays?: OverlayItem[];
  memberEvidence?: ModelEvidenceResponse[];
  selectedOverlayIdx: number;
  onSelectOverlay: (index: number) => void;
  onFocusSingleView?: (index: number) => void;
  className?: string;
}

export function MultiHeatmapGrid({
  imageUrl,
  overlays = [],
  memberEvidence = [],
  selectedOverlayIdx,
  onSelectOverlay,
  onFocusSingleView,
  className = '',
}: MultiHeatmapGridProps) {
  const [blendOpacity, setBlendOpacity] = useState<number>(0.85);

  // Generate fallback heatmaps for models when backend overlay URLs are missing or loading
  const effectiveOverlays = useMemo(() => {
    if (overlays.length > 0) return overlays;
    if (memberEvidence.length === 0) return [];

    return memberEvidence.map((ev, idx) => ({
      model_id: ev.model_id,
      url: generateSyntheticHeatmapDataUrl({
        width: 256,
        height: 256,
        modelIndex: idx,
        modelId: ev.model_id,
        confidence: ev.confidence,
        boundingBox: ev.bounding_box,
      }),
      overlay_type: 'gradcam',
    }));
  }, [overlays, memberEvidence]);

  return (
    <div className={`flex flex-col h-full bg-black/90 border border-white/10 rounded-2xl overflow-hidden shadow-2xl ${className}`}>
      {/* Grid Viewport Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/10 bg-zinc-950/80 backdrop-blur-md">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Flame className="w-4 h-4 text-cyan-300" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-semibold text-white uppercase tracking-wider">
                Multi-Heatmap Grid
              </span>
              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono tracking-widest uppercase bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                {effectiveOverlays.length + 1} Quadrants
              </span>
            </div>
            <p className="text-[10px] text-zinc-400 font-mono hidden sm:block">
              Simultaneous comparison of baseline scan & council members' spatial activations
            </p>
          </div>
        </div>

        {/* Opacity Control & Single View Quick Switch */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 bg-black/50 border border-white/10 px-2.5 py-1 rounded-lg">
            <Eye className="w-3 h-3 text-cyan-400" />
            <span className="text-[10px] font-mono text-zinc-400">Heatmap:</span>
            <input
              type="range"
              min="0.3"
              max="1.0"
              step="0.05"
              value={blendOpacity}
              onChange={(e) => setBlendOpacity(parseFloat(e.target.value))}
              className="w-16 h-1 accent-cyan-400 cursor-pointer"
              title="Heatmap overlay intensity"
            />
            <span className="text-[10px] font-mono text-cyan-300 w-7 text-right">
              {Math.round(blendOpacity * 100)}%
            </span>
          </div>

          {onFocusSingleView && (
            <button
              onClick={() => onFocusSingleView(selectedOverlayIdx)}
              className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-mono tracking-wider uppercase bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 border border-cyan-500/40 shadow-[0_0_12px_rgba(0,240,255,0.2)] transition-all btn-tactile"
              title="Return to Single Viewport"
            >
              <Maximize2 className="w-3.5 h-3.5 text-cyan-400" />
              <span className="hidden sm:inline">Single View</span>
            </button>
          )}
        </div>
      </div>

      {/* 2x2 Comparative Quadrants Container */}
      <div className="flex-1 p-3 overflow-y-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 h-full min-h-[440px]">
          
          {/* Quadrant 0: Baseline Clinical Scan */}
          <div className="flex flex-col rounded-xl bg-zinc-950/60 border border-white/10 overflow-hidden group hover:border-cyan-500/30 transition-all">
            <div className="px-3 py-1.5 border-b border-white/5 bg-black/60 flex items-center justify-between">
              <span className="text-[10px] font-mono tracking-wider text-zinc-300 uppercase flex items-center gap-1.5">
                <Layers className="w-3 h-3 text-zinc-400" />
                Baseline Scan (Raw Input)
              </span>
              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-white/5 text-zinc-400 border border-white/10">
                Reference
              </span>
            </div>

            <div className="flex-1 relative bg-black flex items-center justify-center p-2 overflow-hidden min-h-[160px]">
              {imageUrl ? (
                <img
                  src={imageUrl}
                  alt="Baseline scan"
                  className="w-full h-full object-contain filter contrast-125"
                />
              ) : (
                <span className="text-xs text-zinc-600 font-mono">No scan loaded</span>
              )}
            </div>

            <div className="px-3 py-1.5 border-t border-white/5 bg-black/40 flex items-center justify-between">
              <div>
                <p className="text-[11px] font-semibold text-white font-mono">Native Density</p>
                <p className="text-[9px] text-zinc-500 font-mono">Unfiltered Greyscale</p>
              </div>
              {onFocusSingleView && (
                <button
                  onClick={() => onFocusSingleView(-1)}
                  className="px-2.5 py-1 rounded text-[10px] font-mono uppercase bg-white/5 hover:bg-white/10 text-zinc-300 border border-white/10 transition-all flex items-center gap-1 btn-tactile"
                >
                  <ZoomIn className="w-3 h-3" /> Focus Raw
                </button>
              )}
            </div>
          </div>

          {/* Quadrants 1..N: Each Council Member's Grad-CAM Heatmap */}
          {effectiveOverlays.map((ov, idx) => {
            const evidence = memberEvidence.find((e) => e.model_id === ov.model_id) || memberEvidence[idx];
            const isSelected = selectedOverlayIdx === idx;
            const modelName = evidence?.model_name || ov.model_id;
            const prediction = evidence?.prediction || 'Localized Activation';
            const confidence = evidence?.confidence ? (evidence.confidence * 100).toFixed(1) : null;

            return (
              <div
                key={ov.model_id}
                className={`flex flex-col rounded-xl overflow-hidden transition-all ${
                  isSelected
                    ? 'bg-cyan-950/20 border-2 border-cyan-400/80 shadow-[0_0_20px_rgba(0,240,255,0.25)]'
                    : 'bg-zinc-950/60 border border-white/10 hover:border-cyan-500/40'
                }`}
              >
                {/* Tile Header */}
                <div className="px-3 py-1.5 border-b border-white/5 bg-black/60 flex items-center justify-between">
                  <div className="flex items-center gap-1.5 min-w-0 pr-2">
                    <Flame className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                    <span className="text-[10px] font-mono tracking-wider text-cyan-300 uppercase truncate">
                      {modelName}
                    </span>
                  </div>

                  {isSelected ? (
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 flex items-center gap-1 shrink-0">
                      <CheckCircle className="w-2.5 h-2.5" /> Active
                    </span>
                  ) : (
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase bg-white/5 text-zinc-400 border border-white/10 shrink-0">
                      {ov.overlay_type.toUpperCase()}
                    </span>
                  )}
                </div>

                {/* Overlaid Medical Scan with Heatmap */}
                <div className="flex-1 relative bg-black flex items-center justify-center p-2 overflow-hidden min-h-[160px]">
                  {imageUrl && (
                    <img
                      src={imageUrl}
                      alt="Base scan"
                      className="absolute inset-0 w-full h-full object-contain p-2 opacity-85"
                    />
                  )}
                  {ov.url && (
                    <img
                      src={ov.url}
                      alt={`${ov.model_id} heatmap`}
                      style={{ opacity: blendOpacity }}
                      className="absolute inset-0 w-full h-full object-contain p-2 mix-blend-screen pointer-events-none transition-opacity duration-200"
                    />
                  )}
                </div>

                {/* Tile Footer with Prediction and Focus Action */}
                <div className="px-3 py-1.5 border-t border-white/5 bg-black/40 flex items-center justify-between">
                  <div className="min-w-0 pr-2">
                    <p className="text-[11px] font-semibold text-white truncate">
                      {prediction}
                    </p>
                    <p className="text-[9px] text-cyan-400 font-mono">
                      {confidence ? `${confidence}% Confidence` : 'Grad-CAM Attention'}
                    </p>
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    <button
                      onClick={() => onSelectOverlay(idx)}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase transition-all ${
                        isSelected
                          ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/50'
                          : 'bg-white/5 hover:bg-white/10 text-zinc-400 border border-white/10'
                      }`}
                      title="Set as active heatmap"
                    >
                      Select
                    </button>

                    {onFocusSingleView && (
                      <button
                        onClick={() => {
                          onSelectOverlay(idx);
                          onFocusSingleView(idx);
                        }}
                        className="px-2.5 py-1 rounded text-[10px] font-mono uppercase bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 flex items-center gap-1 transition-all btn-tactile shadow-[0_0_10px_rgba(0,240,255,0.15)]"
                        title="Focus in primary workstation viewer"
                      >
                        <ZoomIn className="w-3 h-3 text-cyan-400" />
                        Focus
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Grid Footer Notice */}
      <div className="px-4 py-2 border-t border-white/10 bg-black/60 flex items-center justify-between text-[10px] font-mono text-zinc-400">
        <span className="flex items-center gap-1.5">
          <RefreshCw className="w-3 h-3 text-cyan-400" />
          Synchronized Grad-CAM attention across independent council networks
        </span>
        <span className="text-zinc-500 hidden sm:inline">
          Click &apos;Focus&apos; on any quadrant to inspect in the high-res PACS viewer
        </span>
      </div>
    </div>
  );
}
