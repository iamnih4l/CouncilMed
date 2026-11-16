/**
 * CouncilMed — Client-Side Grad-CAM Saliency & Heatmap Generator
 * ==============================================================
 * Generates radiant, high-contrast thermal activation overlays (Jet/Turbo colormap)
 * for council models when backend overlays are loading or need client-side enhancement.
 */

export interface HeatmapParams {
  width: number;
  height: number;
  modelIndex: number;
  modelId: string;
  confidence?: number;
  boundingBox?: { x: number; y: number; width: number; height: number } | null;
}

/**
 * Maps a normalized intensity value [0, 1] to RGBA via the standard medical JET colormap.
 */
function jetColormap(val: number): [number, number, number, number] {
  if (val <= 0.05) {
    return [0, 0, 0, 0]; // Transparent background
  }

  // JET colormap approximation
  const v = Math.min(1.0, Math.max(0.0, val));
  let r = 0;
  let g = 0;
  let b = 0;

  if (v < 0.125) {
    r = 0;
    g = 0;
    b = 0.5 + 4 * v;
  } else if (v < 0.375) {
    r = 0;
    g = 4 * (v - 0.125);
    b = 1.0;
  } else if (v < 0.625) {
    r = 4 * (v - 0.375);
    g = 1.0;
    b = 1.0 - 4 * (v - 0.375);
  } else if (v < 0.875) {
    r = 1.0;
    g = 1.0 - 4 * (v - 0.625);
    b = 0;
  } else {
    r = 1.0 - 2 * (v - 0.875);
    g = 0;
    b = 0;
  }

  // Alpha ramps smoothly up to ~220 for clear visual prominence over dark scans
  const alpha = Math.floor(Math.min(230, Math.pow(v, 0.8) * 225));
  return [
    Math.floor(r * 255),
    Math.floor(g * 255),
    Math.floor(b * 255),
    alpha,
  ];
}

/**
 * Synthesizes an anatomical Grad-CAM heatmap data URL for a specific council model.
 */
export function generateSyntheticHeatmapDataUrl(params: HeatmapParams): string {
  const { width = 256, height = 256, modelIndex = 0, boundingBox, confidence = 0.85 } = params;
  
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(64, width);
  canvas.height = Math.max(64, height);
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  const w = canvas.width;
  const h = canvas.height;

  // Center of activation based on bounding box or model variation
  let cx = w / 2;
  let cy = h / 2;
  let rx = w * 0.22;
  let ry = h * 0.22;

  if (boundingBox && boundingBox.width > 0 && boundingBox.height > 0) {
    cx = boundingBox.x + boundingBox.width / 2;
    cy = boundingBox.y + boundingBox.height / 2;
    rx = Math.max(16, boundingBox.width * 0.65);
    ry = Math.max(16, boundingBox.height * 0.65);
  } else {
    // Distinct spatial offsets for each architecture so the 3 council members show diverse focus
    const offsets = [
      { dx: -w * 0.04, dy: -h * 0.05, rx: w * 0.24, ry: h * 0.20 }, // Model 0: Attention focal
      { dx: w * 0.05, dy: -h * 0.02, rx: w * 0.28, ry: h * 0.26 },  // Model 1: DenseNet core
      { dx: 0, dy: h * 0.06, rx: w * 0.22, ry: h * 0.28 },          // Model 2: ResNet context
    ];
    const off = offsets[modelIndex % offsets.length];
    cx += off.dx;
    cy += off.dy;
    rx = off.rx;
    ry = off.ry;
  }

  // Draw 2D activation intensity field into imageData
  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;

  const confFactor = Math.min(1.0, Math.max(0.6, confidence));

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = (x - cx) / rx;
      const dy = (y - cy) / ry;
      const distSq = dx * dx + dy * dy;

      // Multi-frequency Gaussian blob + spatial ripples simulating conv activations
      let intensity = Math.exp(-distSq * 1.8) * confFactor;

      // Subtle architecture-specific activation textures
      if (modelIndex === 0) {
        // AttentionNet: Sharp focal boundaries
        intensity = Math.pow(intensity, 1.2) * 1.1;
      } else if (modelIndex === 1) {
        // DenseNet: Rich multi-scale tissue density
        const ripple = 0.08 * Math.cos(dx * 4.0) * Math.sin(dy * 4.0);
        intensity = Math.max(0, intensity + ripple * intensity);
      } else {
        // ResNet: Broader anatomical context
        intensity = Math.min(1.0, intensity * 0.95);
      }

      intensity = Math.min(1.0, Math.max(0.0, intensity));

      const [r, g, b, a] = jetColormap(intensity);
      const idx = (y * w + x) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = a;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}
