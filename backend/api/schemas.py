"""
CouncilMed — Pydantic API Schemas
====================================
Request/response models for the REST API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


# ==============================
# Analysis
# ==============================

class AnalyzeRequest(BaseModel):
    """Request to analyze a medical image."""
    file_path: str = Field(..., description="Path to the image file")
    study_id: str | None = Field(None, description="Optional study ID")
    modality_override: str | None = Field(None, description="Override auto-detected modality")
    anatomy_override: str | None = Field(None, description="Override auto-detected anatomy")
    enable_explainability: bool = Field(True, description="Generate Grad-CAM heatmaps")
    enable_agent: bool = Field(False, description="Run agentic AI reasoning (requires Ollama)")


class ModelEvidenceResponse(BaseModel):
    """Evidence from a single council member."""
    model_id: str
    model_name: str
    model_version: str
    model_architecture: str
    prediction: str
    probabilities: dict[str, float]
    confidence: float
    uncertainty: float
    processing_time_ms: float
    status: str
    error_message: str | None = None
    device: str = "cpu"


class CouncilResultResponse(BaseModel):
    """Council fusion result."""
    council_run_id: str
    consensus_level: str
    evidence_strength: str
    primary_finding: str
    primary_finding_confidence: float
    severity: str
    models_agreeing: int
    models_total: int
    agreement_score: float
    ensemble_uncertainty: float
    requires_human_review: bool
    human_review_reasons: list[str]
    member_evidence: list[ModelEvidenceResponse]
    disagreements: list[dict[str, Any]]
    spatial_agreement: float
    total_inference_time_ms: float


class AgentInsightResponse(BaseModel):
    """Agentic AI reasoning output."""
    key_insights: list[str] = []
    clinical_significance: list[str] = []
    recommended_actions: list[str] = []
    missing_information: list[str] = []
    safety_flags: list[str] = []
    evidence_references: list[str] = []
    confidence: float = 0.0
    agent_model: str = ""
    processing_time_ms: float = 0.0


class AnalyzeResponse(BaseModel):
    """Complete analysis response."""
    request_id: str
    council_result: CouncilResultResponse
    agent_insight: AgentInsightResponse | None = None
    modality: str
    anatomy: str
    image_shape: list[int]
    total_time_ms: float
    hardware: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []


# ==============================
# Registry
# ==============================

class ModelInfo(BaseModel):
    """Model information from registry."""
    model_id: str
    name: str
    version: str
    architecture: str
    architecture_family: str
    framework: str
    modality: str
    anatomy: str
    num_classes: int
    class_names: list[str]
    installed: bool
    validated: bool
    accuracy: float | None = None
    auroc: float | None = None


class RegistryResponse(BaseModel):
    """Registry summary."""
    total_models: int
    installed: int
    validated: int
    modalities: list[str]
    anatomies: list[str]
    models: list[ModelInfo]


# ==============================
# Health
# ==============================

class HealthResponse(BaseModel):
    """System health check."""
    status: str
    version: str
    models_loaded: int
    device: str
    hardware: dict[str, Any] = {}


# ==============================
# Human Review
# ==============================

class ReviewDecision(BaseModel):
    """Human review decision for a council result."""
    analysis_id: str
    decision: str = Field(..., description="accepted, rejected, or modified")
    reviewer_name: str | None = None
    notes: str | None = None
    modified_finding: str | None = None


# ==============================
# Reports
# ==============================

class ReportSectionResponse(BaseModel):
    """A section within a structured report."""
    title: str
    content: list[str] = []
    subsections: list['ReportSectionResponse'] = []
    severity: str = ""


class ReportResponse(BaseModel):
    """Structured radiology report from council evidence."""
    report_id: str
    version: int
    study_type: str
    modality: str
    anatomy: str
    study_date: str = ""
    sections: list[ReportSectionResponse]
    impression: str
    recommendations: list[str]
    limitations: list[str]
    generated_at: str
    report_status: str
    report_text: str = ""  # Full plain-text rendering


class ReportRequest(BaseModel):
    """Request to generate a report from a completed analysis."""
    analysis_id: str
    study_date: str | None = None

