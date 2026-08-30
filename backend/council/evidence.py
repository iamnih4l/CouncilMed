"""
CouncilMed — Structured Evidence Schema
========================================
Every council member must return structured evidence in this format.
The Council Engine reasons over these evidence objects, NOT raw model outputs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ModelStatus(str, Enum):
    """Status of a model inference run."""
    SUCCESS = "success"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"


class ConsensusLevel(str, Enum):
    """How strongly the council agrees."""
    STRONG = "strong"          # 3/3 models agree
    MODERATE = "moderate"      # 2/3 agree
    WEAK = "weak"              # Models differ substantially
    NO_CONSENSUS = "none"      # Cannot support a conclusion
    INSUFFICIENT = "insufficient"  # Not enough models ran


class SeverityLevel(str, Enum):
    """Clinical severity classification."""
    CLEAR = "clear"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EvidenceStrength(str, Enum):
    """Strength of the evidence supporting the finding."""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass
class BoundingBox:
    """Bounding box in image coordinates (pixels)."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    slice_idx: int | None = None  # For volumetric data


@dataclass
class LocalizationData:
    """Spatial localization of a finding."""
    center_voxel: tuple[float, float, float] | None = None
    center_world_mm: tuple[float, float, float] | None = None
    bounding_box: BoundingBox | None = None
    anatomical_region: str | None = None
    confidence: float = 0.0  # 0-1


@dataclass
class SegmentationData:
    """Segmentation mask and statistics."""
    mask_path: str | None = None  # Path to saved mask file (NIfTI/numpy)
    mask_shape: tuple[int, ...] | None = None
    volume_mm3: float = 0.0
    volume_voxels: int = 0
    dice_score: float | None = None  # If ground truth available
    regions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Image quality assessment for the input."""
    is_acceptable: bool = True
    resolution_ok: bool = True
    contrast_ok: bool = True
    artifacts_detected: bool = False
    signal_to_noise: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ExplainabilityData:
    """Explainability artifacts from inference."""
    gradcam_path: str | None = None     # Path to saved Grad-CAM heatmap
    attention_map_path: str | None = None
    target_layer: str | None = None
    target_class: str | None = None
    target_class_idx: int | None = None


@dataclass
class ModelEvidence:
    """
    Structured evidence returned by every Council member.
    This is the fundamental unit of the CouncilMed system.
    
    Every model MUST return this structure. The Council Engine
    uses these evidence objects to compute consensus.
    """
    # Identity
    model_id: str
    model_name: str
    model_version: str
    model_architecture: str
    
    # Core prediction
    prediction: str  # Human-readable finding name
    probabilities: dict[str, float]  # Class → probability (0-1)
    confidence: float  # 0-1, the max probability or calibrated score
    uncertainty: float  # 0-1, ensemble disagreement / entropy / MC dropout
    
    # Spatial evidence
    localization: LocalizationData | None = None
    segmentation: SegmentationData | None = None
    
    # Quality
    quality: QualityMetrics | None = None
    explainability: ExplainabilityData | None = None
    
    # Metadata
    processing_time_ms: float = 0.0
    status: ModelStatus = ModelStatus.SUCCESS
    error_message: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    
    # Reproducibility
    input_hash: str | None = None  # Hash of the input data
    device: str = "cpu"  # "cpu", "cuda:0", "mps"


@dataclass
class CouncilResult:
    """
    The unified consensus result from the AI Council.
    Produced by the Council Engine after evaluating all member evidence.
    """
    # Identity
    council_run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Consensus
    consensus_level: ConsensusLevel = ConsensusLevel.INSUFFICIENT
    evidence_strength: EvidenceStrength = EvidenceStrength.INSUFFICIENT
    
    # Primary finding
    primary_finding: str = "No analysis performed"
    primary_finding_confidence: float = 0.0  # 0-1
    severity: SeverityLevel = SeverityLevel.INFO
    
    # Model agreement
    models_agreeing: int = 0
    models_total: int = 0
    agreement_score: float = 0.0  # 0-1
    
    # Uncertainty
    ensemble_uncertainty: float = 1.0  # 0-1
    requires_human_review: bool = True
    human_review_reasons: list[str] = field(default_factory=list)
    
    # Individual evidence
    member_evidence: list[ModelEvidence] = field(default_factory=list)
    
    # Disagreement analysis
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    
    # Spatial consensus
    spatial_agreement: float = 0.0  # 0-1
    consensus_localization: LocalizationData | None = None
    
    # Timing
    total_inference_time_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    
    # Study reference
    study_id: str | None = None
    series_id: str | None = None
    modality: str | None = None
    anatomy: str | None = None


@dataclass
class AgentInsight:
    """
    Structured output from the local agentic AI layer.
    The agent reasons over CouncilResult, NOT raw images.
    """
    key_insights: list[str] = field(default_factory=list)
    clinical_significance: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    evidence_references: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1
    
    # Metadata
    agent_model: str = ""
    agent_version: str = ""
    processing_time_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
