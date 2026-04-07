// ============================================================================
// Council Consensus Algorithm — Decision Level Fusion
// Implements a weighted multi-model consensus for diagnostic arbitration.
// Inspired by ensemble learning and clinical committee decision-making.
// ============================================================================

import type {
  DenseNetOutput,
  AttentionNetOutput,
  SwinUNETROutput,
  CouncilConsensusResult,
  BrainRegionKey,
} from './types';
import {
  PathologyClass,
  SeverityLevel,
  BRAIN_REGIONS,
} from './types';

// Model weights for the consensus (tunable based on validation performance)
const MODEL_WEIGHTS = {
  densenet: 0.40, // Classification specialist
  attention: 0.30, // Localization specialist
  swin: 0.30, // Segmentation specialist
};

/**
 * Map a pathology class to its clinical severity
 */
function getSeverity(pathology: PathologyClass): SeverityLevel {
  switch (pathology) {
    case PathologyClass.NORMAL:
      return SeverityLevel.CLEAR;
    case PathologyClass.MENINGIOMA:
    case PathologyClass.EDEMA:
      return SeverityLevel.INFO;
    case PathologyClass.GLIOMA_LOW:
    case PathologyClass.METASTASIS:
      return SeverityLevel.WARNING;
    case PathologyClass.GLIOMA_HIGH:
    case PathologyClass.GLIOBLASTOMA:
    case PathologyClass.NECROSIS:
      return SeverityLevel.CRITICAL;
    default:
      return SeverityLevel.INFO;
  }
}

/**
 * Compute the inter-model agreement score.
 * Higher means all models point to the same region/diagnosis.
 */
function computeAgreement(
  densenet: DenseNetOutput,
  attention: AttentionNetOutput,
  swin: SwinUNETROutput
): number {
  // 1. Check if attention focus and swin tumor center are in similar regions
  const attnCenter = attention.focusCenter;
  const swinCenter = swin.tumorCenter;

  // Euclidean distance between the two spatial predictions
  const spatialDist = Math.sqrt(
    (attnCenter[0] - swinCenter[0]) ** 2 +
    (attnCenter[1] - swinCenter[1]) ** 2 +
    (attnCenter[2] - swinCenter[2]) ** 2
  );

  // Normalize: 0 distance = 1.0 agreement, 5+ distance = 0.0
  const spatialAgreement = Math.max(0, 1 - spatialDist / 5);

  // 2. Check if densenet classification is consistent with detection
  const isAbnormal = densenet.primaryDiagnosis !== PathologyClass.NORMAL;
  const swinDetected = swin.confidence > 30;
  const attnDetected = attention.confidence > 30;

  let classAgreement = 0;
  if (isAbnormal && swinDetected && attnDetected) classAgreement = 1.0;
  else if (!isAbnormal && !swinDetected && !attnDetected) classAgreement = 1.0;
  else if (isAbnormal && (swinDetected || attnDetected)) classAgreement = 0.6;
  else classAgreement = 0.2;

  // Weighted combination
  return spatialAgreement * 0.5 + classAgreement * 0.5;
}

/**
 * Fuse spatial coordinates from Attention Net and Swin UNETR
 * Uses weighted centroid fusion
 */
function fuseAnomalyPosition(
  attention: AttentionNetOutput,
  swin: SwinUNETROutput
): [number, number, number] {
  const attnW = attention.confidence / (attention.confidence + swin.confidence || 1);
  const swinW = 1 - attnW;

  return [
    attention.focusCenter[0] * attnW + swin.tumorCenter[0] * swinW,
    attention.focusCenter[1] * attnW + swin.tumorCenter[1] * swinW,
    attention.focusCenter[2] * attnW + swin.tumorCenter[2] * swinW,
  ];
}

/**
 * Determine the closest brain region to a given 3D position
 */
function findClosestRegion(position: [number, number, number]): BrainRegionKey {
  let closestRegion: BrainRegionKey = 'FRONTAL_LEFT';
  let minDist = Infinity;

  for (const [key, data] of Object.entries(BRAIN_REGIONS)) {
    const dist = Math.sqrt(
      (position[0] - data.center[0]) ** 2 +
      (position[1] - data.center[1]) ** 2 +
      (position[2] - data.center[2]) ** 2
    );
    if (dist < minDist) {
      minDist = dist;
      closestRegion = key as BrainRegionKey;
    }
  }

  return closestRegion;
}

/**
 * Generate clinical notes based on model outputs
 */
function generateCouncilNotes(
  densenet: DenseNetOutput,
  attention: AttentionNetOutput,
  swin: SwinUNETROutput,
  agreement: number
): string[] {
  const notes: string[] = [];

  // Primary finding
  if (densenet.primaryDiagnosis !== PathologyClass.NORMAL) {
    notes.push(
      `DenseNet-121 classifies primary finding as ${densenet.primaryDiagnosis} with ${densenet.confidence.toFixed(1)}% confidence.`
    );
  } else {
    notes.push('DenseNet-121 classification indicates no significant pathology detected.');
  }

  // Spatial findings
  const regionLabel = BRAIN_REGIONS[attention.focusRegion].label;
  notes.push(
    `Attention-Net localizes peak activation in the ${regionLabel} (confidence: ${attention.confidence.toFixed(1)}%).`
  );

  // Segmentation findings
  if (swin.tumorVolumeMm3 > 100) {
    notes.push(
      `Swin-UNETR delineates lesion volume at approximately ${swin.tumorVolumeMm3.toFixed(0)} mm³.`
    );
    for (const seg of swin.segmentedRegions) {
      if (seg.volumePercentage > 5) {
        notes.push(`  → ${seg.label}: ${seg.volumePercentage.toFixed(1)}% of total volume.`);
      }
    }
  } else {
    notes.push('Swin-UNETR segmentation indicates minimal lesion volume.');
  }

  // Agreement assessment
  if (agreement > 0.8) {
    notes.push('Council Assessment: HIGH AGREEMENT — All models converge on consistent findings.');
  } else if (agreement > 0.5) {
    notes.push('Council Assessment: MODERATE AGREEMENT — Partial model convergence. Clinical review recommended.');
  } else {
    notes.push('Council Assessment: LOW AGREEMENT — Model outputs are divergent. Manual review strongly recommended.');
  }

  return notes;
}

// ============================================================================
// Main Consensus Function
// ============================================================================

/**
 * Run the Council Consensus Algorithm
 * Fuses outputs from DenseNet-121, Attention Net, and Swin UNETR into
 * a single optimal diagnostic result.
 */
export function runCouncilConsensus(
  densenet: DenseNetOutput,
  attention: AttentionNetOutput,
  swin: SwinUNETROutput
): CouncilConsensusResult {
  // 1. Compute overall confidence (weighted)
  const overallConfidence =
    densenet.confidence * MODEL_WEIGHTS.densenet +
    attention.confidence * MODEL_WEIGHTS.attention +
    swin.confidence * MODEL_WEIGHTS.swin;

  // 2. Primary diagnosis from DenseNet (classification specialist)
  const primaryDiagnosis = densenet.primaryDiagnosis;
  const severity = getSeverity(primaryDiagnosis);

  // 3. Fuse spatial data from Attention + Swin
  const anomalyPosition = fuseAnomalyPosition(attention, swin);
  const anomalyRadius = (attention.focusRadius + swin.tumorRadius) / 2;
  const affectedRegion = findClosestRegion(anomalyPosition);

  // 4. Compute model agreement
  const modelAgreement = computeAgreement(densenet, attention, swin);

  // 5. Generate clinical notes
  const councilNotes = generateCouncilNotes(densenet, attention, swin, modelAgreement);

  // 6. Total inference time
  const totalInferenceTimeMs =
    densenet.inferenceTimeMs + attention.inferenceTimeMs + swin.inferenceTimeMs;

  return {
    primaryDiagnosis,
    severity,
    overallConfidence,
    anomalyPosition,
    anomalyRadius,
    affectedRegion,
    densenetResult: densenet,
    attentionResult: attention,
    swinResult: swin,
    modelAgreement,
    councilNotes,
    totalInferenceTimeMs,
  };
}
