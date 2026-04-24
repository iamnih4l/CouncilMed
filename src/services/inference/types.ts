// ============================================================================
// CouncilMed Clinical Diagnostic Types
// Shared type definitions for the multi-model MRI inference pipeline
// ============================================================================

// --- Brain Region Mapping (BraTS-compliant anatomical labels) ---

export const BRAIN_REGIONS = {
  FRONTAL_LEFT: { id: 0, label: 'Left Frontal Lobe', center: [1.2, 0.8, 1.5] as [number, number, number] },
  FRONTAL_RIGHT: { id: 1, label: 'Right Frontal Lobe', center: [-1.2, 0.8, 1.5] as [number, number, number] },
  TEMPORAL_LEFT: { id: 2, label: 'Left Temporal Lobe', center: [1.8, -0.3, 0.5] as [number, number, number] },
  TEMPORAL_RIGHT: { id: 3, label: 'Right Temporal Lobe', center: [-1.8, -0.3, 0.5] as [number, number, number] },
  PARIETAL_LEFT: { id: 4, label: 'Left Parietal Lobe', center: [0.8, 1.2, -0.3] as [number, number, number] },
  PARIETAL_RIGHT: { id: 5, label: 'Right Parietal Lobe', center: [-0.8, 1.2, -0.3] as [number, number, number] },
  OCCIPITAL: { id: 6, label: 'Occipital Lobe', center: [0, 0.2, -1.8] as [number, number, number] },
  CEREBELLUM: { id: 7, label: 'Cerebellum', center: [0, -1.5, -1.2] as [number, number, number] },
  BRAINSTEM: { id: 8, label: 'Brainstem', center: [0, -1.8, 0] as [number, number, number] },
  THALAMUS: { id: 9, label: 'Thalamus', center: [0, 0, 0.2] as [number, number, number] },
} as const;

export type BrainRegionKey = keyof typeof BRAIN_REGIONS;

// --- Pathology Classification (WHO Grade I-IV compliant) ---

export const PathologyClass = {
  NORMAL: 'Normal',
  MENINGIOMA: 'Meningioma (Grade I)',
  GLIOMA_LOW: 'Low-Grade Glioma (Grade II)',
  GLIOMA_HIGH: 'High-Grade Glioma (Grade III)',
  GLIOBLASTOMA: 'Glioblastoma (Grade IV)',
  METASTASIS: 'Metastatic Lesion',
  EDEMA: 'Peritumoral Edema',
  NECROSIS: 'Necrotic Core',
} as const;

export type PathologyClass = (typeof PathologyClass)[keyof typeof PathologyClass];

export const SeverityLevel = {
  CLEAR: 'clear',
  INFO: 'info',
  WARNING: 'warning',
  CRITICAL: 'critical',
} as const;

export type SeverityLevel = (typeof SeverityLevel)[keyof typeof SeverityLevel];

// --- Individual Model Output Types ---

/** DenseNet-121 produces global classification probabilities */
export interface DenseNetOutput {
  modelName: 'DenseNet-121';
  classifications: {
    pathology: PathologyClass;
    probability: number; // 0-1
  }[];
  primaryDiagnosis: PathologyClass;
  confidence: number; // 0-100
  inferenceTimeMs: number;
  featureVector: number[]; // 1024-dim feature for fusion
}

/** Attention Net produces spatial attention heatmaps */
export interface AttentionNetOutput {
  modelName: 'Attention-Net';
  attentionMap: number[][]; // 2D spatial attention weights (normalized 0-1)
  focusRegion: BrainRegionKey;
  focusCenter: [number, number, number]; // 3D coordinates of peak attention
  focusRadius: number; // Estimated radius of focus area
  confidence: number; // 0-100
  inferenceTimeMs: number;
}

/** Swin UNETR produces volumetric segmentation masks */
export interface SwinUNETROutput {
  modelName: 'Swin-UNETR';
  segmentationMask: number[][][]; // 3D voxel mask
  tumorVolumeMm3: number;
  tumorCenter: [number, number, number];
  tumorRadius: number;
  segmentedRegions: {
    label: string;
    volumePercentage: number;
  }[];
  confidence: number; // 0-100
  inferenceTimeMs: number;
}

// --- Council Consensus Output ---

export interface CouncilConsensusResult {
  // Final fused diagnosis
  primaryDiagnosis: PathologyClass;
  severity: SeverityLevel;
  overallConfidence: number; // 0-100 (weighted average)

  // Spatial findings (from Attention + Swin fusion)
  anomalyPosition: [number, number, number]; // 3D coords for BrainModel
  anomalyRadius: number;
  affectedRegion: BrainRegionKey;

  // Individual model contributions
  densenetResult: DenseNetOutput;
  attentionResult: AttentionNetOutput;
  swinResult: SwinUNETROutput;

  // Consensus metadata
  modelAgreement: number; // 0-1 (1 = full agreement)
  councilNotes: string[];
  totalInferenceTimeMs: number;

  // Added Clinical Validation

  radiologyReport?: RadiologyReportData;
}

// --- Radiology Report Types (RSNA-style) ---

export interface RadiologyReportData {
  modality: string;
  studyType: 'T1' | 'T2' | 'FLAIR' | 'Contrast' | 'Unknown';
  dataQuality: 'Adequate' | 'Limited' | 'Non-diagnostic';
  clinicalIndication: string;
  technique: string;
  findings: {
    brainParenchyma: string;
    lesionAssessment: {
      included: boolean;
      location: string | null;
      sizeVolume: string | null;
      characteristics: string;
    };
    edemaMassEffect: 'Present' | 'Absent' | 'Uncertain';
    ventricularSystem: 'Normal' | 'Dilated' | 'Compressed';
    midlineStructures: {
      shift: 'Yes' | 'No' | 'Uncertain';
    };
  };
  impression: string;
  limitations: string[];
  recommendations: string[];
}



// --- Diagnostic Pipeline State ---

export type PipelineStage =
  | 'idle'
  | 'preprocessing'
  | 'densenet-running'
  | 'attention-running'
  | 'swin-running'
  | 'consensus'
  | 'complete'
  | 'error';

export interface DiagnosticState {
  stage: PipelineStage;
  progress: number; // 0-100
  uploadedImage: string | null; // base64 data URL
  imageTensor: unknown; // tf.Tensor — stored as unknown to avoid top-level tf import
  result: CouncilConsensusResult | null;
  error: string | null;

  // Per-model status
  densenetStatus: 'idle' | 'running' | 'complete' | 'error';
  attentionStatus: 'idle' | 'running' | 'complete' | 'error';
  swinStatus: 'idle' | 'running' | 'complete' | 'error';
}

// --- Model Configuration ---

export interface ModelConfig {
  inputSize: [number, number]; // [height, width]
  numClasses: number;
  weights: 'random' | 'pretrained';
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  inputSize: [224, 224],
  numClasses: 4, // Matches the 4-class trained brain tumor classifier
  weights: 'pretrained',
};
