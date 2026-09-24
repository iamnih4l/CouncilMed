/**
 * CouncilMed — Medical Image Viewer
 * ======================================
 * Production-grade medical image viewer with:
 * - Window/Level (brightness/contrast) controls
 * - Zoom, pan, and measurement tools
 * - Grad-CAM heatmap overlay toggle
 * - DICOM metadata panel
 * - Proper grayscale rendering for medical images
 *
 * This is a critical workstation component — the image is the
 * CENTER of the CouncilMed experience.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ZoomIn, ZoomOut, RotateCcw, Move,
  SunMedium, Contrast, Ruler, Eye, EyeOff,
  Layers, Info, Grid3x3,
} from 'lucide-react';

interface ViewerState {
  zoom: number;
  panX: number;
  panY: number;
  windowWidth: number;
  windowCenter: number;
  rotation: number;
  invert: boolean;
  showOverlay: boolean;
  activeTool: 'pan' | 'window' | 'zoom' | 'measure';
}

interface OverlayItem {
  model_id: string;
  url: string;
  overlay_type: string;
}

interface MedicalImageViewerProps {
  imageUrl: string | null;
  overlayUrl?: string | null;
  overlays?: OverlayItem[];
  fileName?: string;
  imageShape?: number[];
  metadata?: Record<string, string>;
  modality?: string;
  anatomy?: string;
  boxes?: { x: number, y: number, width: number, height: number }[];
  className?: string;
}

export function MedicalImageViewer({
  imageUrl,
  overlayUrl,
  overlays = [],
  fileName = '',
  imageShape,
  metadata,
  modality = 'Unknown',
  anatomy = 'Unknown',
  boxes = [],
  className = '',
}: MedicalImageViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);
  const lastPosRef = useRef({ x: 0, y: 0 });
  const [showMetadata, setShowMetadata] = useState(false);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const [pixelValue, setPixelValue] = useState<number | null>(null);
  const [measurements, setMeasurements] = useState<{start: {x: number, y: number}, end: {x: number, y: number}}[]>([]);
  const [currentMeasurement, setCurrentMeasurement] = useState<{start: {x: number, y: number}, end: {x: number, y: number}} | null>(null);

  const [state, setState] = useState<ViewerState>({
    zoom: 1,
    panX: 0,
    panY: 0,
    windowWidth: 400,
    windowCenter: 200,
    rotation: 0,
    invert: false,
    showOverlay: true,
    activeTool: 'pan',
  });

  const imageRef = useRef<HTMLImageElement | null>(null);
  const overlayRef = useRef<HTMLImageElement | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [overlayLoaded, setOverlayLoaded] = useState(false);
  const [selectedOverlayIdx, setSelectedOverlayIdx] = useState(0);
  const glowPhaseRef = useRef(0);
  const animationRef = useRef<number | null>(null);

  // Determine the active overlay URL — prefer the overlays array, fall back to overlayUrl
  const activeOverlayUrl = overlays.length > 0
    ? overlays[selectedOverlayIdx]?.url ?? null
    : overlayUrl ?? null;
  const hasOverlays = activeOverlayUrl !== null || overlays.length > 0;

  // Load images when URLs change
  useEffect(() => {
    if (!imageUrl) {
      imageRef.current = null;
      setImageLoaded(false);
      return;
    }

    setImageLoaded(false);
    const img = new Image();
    
    img.onload = () => { console.log('Image successfully loaded!', { width: img.width, height: img.height, src: img.src.substring(0, 50) + '...' });
      imageRef.current = img;
      setImageLoaded(true);
    };
    img.onerror = (e) => { console.error('Image failed to load!', e, { src: imageUrl.substring(0, 50) + '...' });
      imageRef.current = null;
      setImageLoaded(false);
    };
        const urlObj = new URL(imageUrl, window.location.origin);
    urlObj.searchParams.set('cb', Date.now().toString());
    const finalUrl = imageUrl.startsWith('data:') || imageUrl.startsWith('blob:') ? imageUrl : urlObj.toString();
    img.crossOrigin = 'anonymous';
    img.src = finalUrl;

    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageUrl]);

  // Load overlay image
  useEffect(() => {
    if (!activeOverlayUrl) {
      overlayRef.current = null;
      setOverlayLoaded(false);
      return;
    }

    setOverlayLoaded(false);
    const img = new Image();
    
    img.onload = () => { overlayRef.current = img;
      setOverlayLoaded(true);
    };
    img.onerror = () => {
      console.error('Overlay failed to load:', activeOverlayUrl.substring(0, 50) + '...');
      overlayRef.current = null;
      setOverlayLoaded(false);
    };
        const overlayUrlObj = new URL(activeOverlayUrl, window.location.origin);
    overlayUrlObj.searchParams.set('cb', Date.now().toString());
    const finalOverlayUrl = activeOverlayUrl.startsWith('data:') || activeOverlayUrl.startsWith('blob:') ? activeOverlayUrl : overlayUrlObj.toString();
    img.crossOrigin = 'anonymous';
    img.src = finalOverlayUrl;

    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [activeOverlayUrl]);

  // Render the canvas
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI displays for crisp rendering
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();

    const viewW = rect.width; const viewH = rect.height; canvas.width = viewW * dpr;
    canvas.height = viewH * dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, viewW, viewH);

    // If no image, render empty state
    if (!imageUrl) {
      ctx.fillStyle = '#18181b'; // zinc-900
      ctx.fillRect(0, 0, viewW, viewH);
      return;
    }

    // Define annotation drawing function here to fix ESLint "used before defined"
    const _drawAnnotations = (context: CanvasRenderingContext2D, w: number, h: number) => {
      context.fillStyle = 'rgba(255,255,255,0.5)';
      context.font = '11px monospace';
      
      // Top Left: Modality/Anatomy
      context.textAlign = 'left';
      context.fillText(`${modality} / ${anatomy}`, 10, 20);
      
      // Top Right: Window Level
      context.textAlign = 'right';
      context.fillText(`W: ${state.windowWidth.toFixed(0)} L: ${state.windowCenter.toFixed(0)}`, viewW - 10, 20);
      
      // Bottom Left: Zoom
      context.textAlign = 'left';
      context.fillText(`Zoom: ${(state.zoom * 100).toFixed(0)}%`, 10, viewH - 10);
      
      // Bottom Right: Image size
      context.textAlign = 'right';
      const sizeLabel = imageShape?.length
        ? `${imageShape[0]} x ${imageShape[1]}`
        : imageRef.current
          ? `${imageRef.current.width} x ${imageRef.current.height}`
          : '—';
      context.fillText(sizeLabel, viewW - 10, viewH - 10);
      
      if (state.invert) {
        context.fillText('INVERTED', viewW - 10, 35);
      }
      
      // Draw crosshairs at center
      context.beginPath();
      context.strokeStyle = 'rgba(0, 240, 255, 0.3)';
      context.lineWidth = 1;
      context.moveTo(viewW / 2, 0);
      context.lineTo(viewW / 2, viewH);
      context.moveTo(0, viewH / 2);
      context.lineTo(viewW, viewH / 2);
      context.stroke();
      
      // Draw targeting corners
      const cornerSize = 20;
      context.strokeStyle = 'rgba(0, 240, 255, 0.8)';
      context.lineWidth = 2;
      
      // Top Left
      context.beginPath(); context.moveTo(10, 10 + cornerSize); context.lineTo(10, 10); context.lineTo(10 + cornerSize, 10); context.stroke();
      // Top Right
      context.beginPath(); context.moveTo(viewW - 10 - cornerSize, 10); context.lineTo(viewW - 10, 10); context.lineTo(viewW - 10, 10 + cornerSize); context.stroke();
      // Bottom Left
      context.beginPath(); context.moveTo(10, viewH - 10 - cornerSize); context.lineTo(10, viewH - 10); context.lineTo(10 + cornerSize, viewH - 10); context.stroke();
      // Bottom Right
      context.beginPath(); context.moveTo(viewW - 10 - cornerSize, viewH - 10); context.lineTo(viewW - 10, viewH - 10); context.lineTo(viewW - 10, viewH - 10 - cornerSize); context.stroke();
    };

    ctx.save();

    // Center origin
    ctx.translate(viewW / 2 + state.panX, viewH / 2 + state.panY);

    // Apply rotation
    ctx.rotate((state.rotation * Math.PI) / 180);

    // Apply zoom
    ctx.scale(state.zoom, state.zoom);

    // Simulate Window/Level with contrast/brightness filters
    // Default width=400, center=200
    ctx.filter = `contrast(${Math.max(0, 400 / Math.max(1, state.windowWidth))}) brightness(${Math.max(0, 1 + (state.windowCenter - 200) / 200)})`;
    
    const img = imageRef.current;
    if (img && img.width > 0 && imageLoaded) {
      // Calculate scaling to fit within canvas initially
      const size = Math.min(viewW, viewH) * 0.8;
      const scale = Math.min(size / img.width, size / img.height);
      const imgW = img.width * scale;
      const imgH = img.height * scale;
      
      // Draw actual medical image
      ctx.drawImage(img, -imgW/2, -imgH/2, imgW, imgH);

      // Draw Grad-CAM overlay when available and loaded
      const overlay = overlayRef.current;
      if (overlay && overlay.width > 0 && overlayLoaded && state.showOverlay) {
        ctx.filter = 'none'; // Don't apply W/L to overlay — it has its own RGBA
        ctx.globalAlpha = 0.5;
        ctx.drawImage(overlay, -imgW/2, -imgH/2, imgW, imgH);
        ctx.globalAlpha = 1;
      }
      
      // Draw glowing bounding boxes from council evidence
      // Boxes come in image-pixel coordinates (relative to the original image dims).
      // Map them to canvas space using the actual natural image dimensions.
      if (boxes && boxes.length > 0) {
        ctx.filter = 'none';
        const glowIntensity = 0.55 + 0.45 * Math.sin(glowPhaseRef.current);
        // Use actual image natural dimensions for coordinate mapping
        const imgNatW = img.naturalWidth || img.width;
        const imgNatH = img.naturalHeight || img.height;

        boxes.forEach(box => {
          const mappedX = -imgW/2 + (box.x / imgNatW) * imgW;
          const mappedY = -imgH/2 + (box.y / imgNatH) * imgH;
          const mappedW = (box.width / imgNatW) * imgW;
          const mappedH = (box.height / imgNatH) * imgH;

          ctx.save();
          ctx.shadowColor = `rgba(0, 240, 255, ${0.35 + glowIntensity * 0.45})`;
          ctx.shadowBlur = 12 + glowIntensity * 10;

          ctx.beginPath();
          ctx.rect(mappedX, mappedY, mappedW, mappedH);
          ctx.strokeStyle = `rgba(0, 240, 255, ${0.75 + glowIntensity * 0.25})`;
          ctx.lineWidth = 2 / state.zoom;
          ctx.stroke();

          ctx.shadowBlur = 0;
          ctx.fillStyle = `rgba(0, 240, 255, ${0.12 + glowIntensity * 0.12})`;
          ctx.fill();
          ctx.restore();
        });
      }
    } else if (imageUrl) {
      // Fallback placeholder
      const size = Math.min(viewW, viewH) * 0.8;
      ctx.fillStyle = '#27272a'; // zinc-800
      ctx.fillRect(-size/2, -size/2, size, size);
      ctx.fillStyle = '#52525b';
      ctx.beginPath();
      ctx.arc(0, 0, size/4, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Draw manual measurements
    ctx.strokeStyle = '#00f0ff';
    ctx.fillStyle = '#00f0ff';
    ctx.lineWidth = 2 / state.zoom;
    
    const drawMeasurement = (m: {start: {x: number, y: number}, end: {x: number, y: number}}) => {
      ctx.beginPath();
      ctx.moveTo(m.start.x, m.start.y);
      ctx.lineTo(m.end.x, m.end.y);
      ctx.stroke();
      
      // Draw endpoints
      ctx.beginPath();
      ctx.arc(m.start.x, m.start.y, 3 / state.zoom, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(m.end.x, m.end.y, 3 / state.zoom, 0, Math.PI * 2);
      ctx.fill();
      
      // Calculate physical distance (placeholder conversion, e.g., 1 pixel = 0.5mm)
      const distPixels = Math.hypot(m.end.x - m.start.x, m.end.y - m.start.y);
      const distMm = (distPixels * 0.5).toFixed(1); // placeholder physical distance
      
      // Draw text
      ctx.save();
      ctx.translate((m.start.x + m.end.x) / 2, (m.start.y + m.end.y) / 2);
      // Keep text upright relative to the screen (undo rotation)
      ctx.rotate((-state.rotation * Math.PI) / 180);
      ctx.scale(1 / state.zoom, 1 / state.zoom);
      
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(-20, -15, 40, 20);
      ctx.fillStyle = '#00f0ff';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${distMm}mm`, 0, -2);
      ctx.restore();
    };

    measurements.forEach(drawMeasurement);
    if (currentMeasurement) {
      drawMeasurement(currentMeasurement);
    }

    ctx.filter = 'none';
    ctx.restore();

    // Draw viewer annotations
    _drawAnnotations(ctx, viewW, viewH);
  }, [state, imageUrl, imageLoaded, overlayLoaded, fileName, imageShape, pixelValue, cursorPos, boxes, modality, anatomy, measurements, currentMeasurement]);

  // Re-render when state changes
  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Pulse bounding-box glow while detections are visible
  useEffect(() => {
    if (!boxes.length) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      return;
    }

    let running = true;
    const animate = () => {
      if (!running) return;
      glowPhaseRef.current += 0.08;
      renderCanvas();
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);
    return () => {
      running = false;
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [boxes, renderCanvas]);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver(() => renderCanvas());
    observer.observe(container);
    return () => observer.disconnect();
  }, [renderCanvas]);

  // Mouse handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isDraggingRef.current = true;
    lastPosRef.current = { x: e.clientX, y: e.clientY };

    if (state.activeTool === 'measure') {
      const rect = containerRef.current!.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const viewW = rect.width;
      const viewH = rect.height;
      const canvasX = (x - viewW/2 - state.panX) / state.zoom;
      const canvasY = (y - viewH/2 - state.panY) / state.zoom;
      setCurrentMeasurement({ start: { x: canvasX, y: canvasY }, end: { x: canvasX, y: canvasY } });
    }
  }, [state.activeTool, state.panX, state.panY, state.zoom]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    setCursorPos({ x, y });

    if (state.activeTool === 'measure' && isDraggingRef.current && currentMeasurement) {
      const viewW = rect.width;
      const viewH = rect.height;
      const canvasX = (x - viewW/2 - state.panX) / state.zoom;
      const canvasY = (y - viewH/2 - state.panY) / state.zoom;
      setCurrentMeasurement(prev => prev ? { ...prev, end: { x: canvasX, y: canvasY } } : null);
    }

    if (!isDraggingRef.current) return;

    const dx = e.clientX - lastPosRef.current.x;
    const dy = e.clientY - lastPosRef.current.y;
    lastPosRef.current = { x: e.clientX, y: e.clientY };

    if (state.activeTool === 'pan') {
      setState(prev => ({
        ...prev,
        panX: prev.panX + dx,
        panY: prev.panY + dy,
      }));
    } else if (state.activeTool === 'window') {
      setState(prev => ({
        ...prev,
        windowWidth: Math.max(1, prev.windowWidth + dx),
        windowCenter: Math.max(0, prev.windowCenter - dy),
      }));
    } else if (state.activeTool === 'zoom') {
      setState(prev => ({
        ...prev,
        zoom: Math.max(0.1, Math.min(10, prev.zoom + dy * -0.005)),
      }));
    }
  }, [state.activeTool]);

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
    if (state.activeTool === 'measure' && currentMeasurement) {
      if (Math.hypot(currentMeasurement.end.x - currentMeasurement.start.x, currentMeasurement.end.y - currentMeasurement.start.y) > 5) {
        setMeasurements(prev => [...prev, currentMeasurement]);
      }
      setCurrentMeasurement(null);
    }
  }, [state.activeTool, currentMeasurement]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setState(prev => ({
      ...prev,
      zoom: Math.max(0.1, Math.min(10, prev.zoom - e.deltaY * 0.001)),
    }));
  }, []);

  const resetView = useCallback(() => {
    setState(prev => ({
      ...prev,
      zoom: 1,
      panX: 0,
      panY: 0,
      rotation: 0,
      invert: false,
    }));
    setMeasurements([]);
  }, []);

  const toolButtons = [
    { id: 'pan' as const, icon: Move, label: 'Pan' },
    { id: 'zoom' as const, icon: ZoomIn, label: 'Zoom' },
    { id: 'window' as const, icon: SunMedium, label: 'W/L' },
    { id: 'measure' as const, icon: Ruler, label: 'Measure' },
  ];

  return (
    <div className={`flex flex-col ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-2 py-1.5 bg-black/40 border-b border-white/5 rounded-t-xl">
        <div className="flex items-center gap-1">
          {toolButtons.map(tool => (
            <button
              key={tool.id}
              onClick={() => setState(prev => ({ ...prev, activeTool: tool.id }))}
              className={`p-1.5 rounded-lg transition-all text-xs flex items-center gap-1 ${
                state.activeTool === tool.id
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
              }`}
              title={tool.label}
            >
              <tool.icon className="w-3.5 h-3.5" />
              <span className="hidden md:inline">{tool.label}</span>
            </button>
          ))}

          <div className="w-px h-5 bg-white/10 mx-1" />

          <button
            onClick={() => setState(prev => ({ ...prev, zoom: Math.min(10, prev.zoom + 0.25) }))}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-300 hover:bg-white/5"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setState(prev => ({ ...prev, zoom: Math.max(0.1, prev.zoom - 0.25) }))}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-300 hover:bg-white/5"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={resetView}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-300 hover:bg-white/5"
            title="Reset View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          <div className="w-px h-5 bg-white/10 mx-1" />

          <button
            onClick={() => setState(prev => ({ ...prev, invert: !prev.invert }))}
            className={`p-1.5 rounded-lg transition-all ${
              state.invert ? 'text-amber-400 bg-amber-500/10' : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
            }`}
            title="Invert"
          >
            <Contrast className="w-3.5 h-3.5" />
          </button>

          {hasOverlays && (
            <>
              <button
                onClick={() => setState(prev => ({ ...prev, showOverlay: !prev.showOverlay }))}
                className={`p-1.5 rounded-lg transition-all flex items-center gap-1 ${
                  state.showOverlay ? 'text-emerald-400 bg-emerald-500/10' : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
                }`}
                title="Toggle Heatmap"
              >
                {state.showOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                <span className="text-[10px] hidden md:inline">Heatmap</span>
              </button>
              {overlays.length > 1 && (
                <select
                  value={selectedOverlayIdx}
                  onChange={e => setSelectedOverlayIdx(Number(e.target.value))}
                  className="bg-black/60 border border-white/10 rounded-lg text-[10px] text-zinc-400 px-1 py-1 cursor-pointer"
                  title="Select overlay model"
                >
                  {overlays.map((ov, i) => (
                    <option key={ov.model_id} value={i}>
                      {ov.model_id} ({ov.overlay_type})
                    </option>
                  ))}
                </select>
              )}
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] text-zinc-600 font-mono">
            W:{state.windowWidth} L:{state.windowCenter}
          </span>

          <button
            onClick={() => setShowMetadata(!showMetadata)}
            className={`p-1.5 rounded-lg transition-all ${
              showMetadata ? 'text-cyan-400 bg-cyan-500/10' : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
            }`}
            title="Metadata"
          >
            <Info className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Canvas area */}
      <div className="flex-1 relative flex min-h-[420px]">
        <div
          ref={containerRef}
          className="flex-1 relative bg-black cursor-crosshair overflow-hidden"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        >
                      <button 
              onClick={() => {
                fetch(imageUrl)
                  .then(r => r.blob())
                  .then(b => alert('Fetched blob size: ' + b.size + ' type: ' + b.type))
                  .catch(e => alert('Fetch err: ' + e));
              }}
              className="absolute top-2 left-2 z-50 bg-red-500 text-white px-2 py-1 rounded"
            >
              Debug Fetch
            </button>
            <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
          />

          {!imageUrl || !imageLoaded ? (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center text-cyan-500/30">
                <Grid3x3 className="w-12 h-12 mx-auto mb-2 opacity-50 animate-pulse" />
                <p className="text-sm font-mono tracking-widest uppercase">
                  {!imageUrl ? 'No data source' : 'Loading image...'}
                </p>
              </div>
            </div>
          ) : null}

          {/* Cinematic Scanline Overlay */}
          <div className="absolute inset-0 pointer-events-none bg-[url('/scanline.png')] opacity-20 mix-blend-overlay" />
          <div className="absolute inset-0 pointer-events-none shadow-[inset_0_0_100px_rgba(0,0,0,0.8)]" />
        </div>

        {/* Metadata panel */}
        {showMetadata && metadata && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 240, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="bg-black/60 backdrop-blur-sm border-l border-white/5 overflow-y-auto"
          >
            <div className="p-3">
              <p className="text-[11px] text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                <Layers className="w-3 h-3" /> DICOM Metadata
              </p>
              <div className="space-y-1.5">
                {Object.entries(metadata).map(([key, value]) => (
                  <div key={key} className="flex flex-col">
                    <span className="text-[10px] text-zinc-600">{key}</span>
                    <span className="text-[11px] text-zinc-300 font-mono truncate">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Status bar */}
      <div className="flex items-center justify-between px-2 py-1 bg-black/40 border-t border-white/5 rounded-b-xl">
        <span className="text-[10px] text-zinc-600 font-mono">
          Tool: {state.activeTool.toUpperCase()}
        </span>
        <span className="text-[10px] text-zinc-600 font-mono">
          Zoom: {Math.round(state.zoom * 100)}% | Pos: ({cursorPos.x}, {cursorPos.y})
          {pixelValue !== null ? ` | Val: ${pixelValue}` : ''}
        </span>
      </div>
    </div>
  );
}




















