"""
CouncilMed — Database Models
================================
Production database schema for medical imaging AI workstation.

Tables:
  - studies: DICOM study records
  - series: DICOM series within studies
  - analyses: Council analysis runs
  - model_evidence: Individual model predictions
  - council_results: Consensus fusion results
  - reports: Generated clinical reports
  - agent_insights: Agentic AI reasoning outputs
  - audit_log: Full decision audit trail
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Study(Base):
    """DICOM study record."""
    __tablename__ = "studies"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    study_instance_uid = Column(String, unique=True, nullable=True)
    study_date = Column(String, nullable=True)
    study_description = Column(String, nullable=True)
    patient_name = Column(String, nullable=True)  # De-identified
    patient_id = Column(String, nullable=True)     # Hashed
    modality = Column(String, nullable=False, default="UNKNOWN")
    anatomy = Column(String, nullable=False, default="unknown")
    file_path = Column(String, nullable=True)
    file_hash = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    
    # Relationships
    series = relationship("Series", back_populates="study", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="study", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_studies_modality", "modality"),
        Index("ix_studies_anatomy", "anatomy"),
        Index("ix_studies_created", "created_at"),
    )


class Series(Base):
    """DICOM series within a study."""
    __tablename__ = "series"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    study_id = Column(String, ForeignKey("studies.id"), nullable=False)
    series_instance_uid = Column(String, nullable=True)
    series_number = Column(Integer, nullable=True)
    series_description = Column(String, nullable=True)
    modality = Column(String, nullable=True)
    num_instances = Column(Integer, default=1)
    is_volume = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=utcnow)
    
    # Relationships
    study = relationship("Study", back_populates="series")


class Analysis(Base):
    """A single analysis run (one invocation of the pipeline)."""
    __tablename__ = "analyses"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    study_id = Column(String, ForeignKey("studies.id"), nullable=True)
    request_id = Column(String, nullable=False, unique=True)
    
    # Input
    file_path = Column(String, nullable=True)
    modality = Column(String, nullable=True)
    anatomy = Column(String, nullable=True)
    image_shape = Column(JSON, nullable=True)
    
    # Status
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Timing
    total_time_ms = Column(Float, nullable=True)
    
    # Hardware
    device = Column(String, nullable=True)
    hardware_info = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    study = relationship("Study", back_populates="analyses")
    evidence = relationship("ModelEvidenceRecord", back_populates="analysis", cascade="all, delete-orphan")
    council_result = relationship("CouncilResultRecord", back_populates="analysis", uselist=False, cascade="all, delete-orphan")
    agent_insight = relationship("AgentInsightRecord", back_populates="analysis", uselist=False, cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="analysis", cascade="all, delete-orphan")
    audit_events = relationship("AuditEvent", back_populates="analysis", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_analyses_status", "status"),
        Index("ix_analyses_created", "created_at"),
    )


class ModelEvidenceRecord(Base):
    """Individual model prediction evidence."""
    __tablename__ = "model_evidence"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    
    # Model identity
    model_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    model_architecture = Column(String, nullable=True)
    
    # Prediction
    prediction = Column(String, nullable=False)
    probabilities = Column(JSON, nullable=True)  # {class_name: probability}
    confidence = Column(Float, nullable=False, default=0.0)
    uncertainty = Column(Float, nullable=False, default=1.0)
    
    # Status
    status = Column(String, nullable=False, default="success")
    error_message = Column(Text, nullable=True)
    
    # Performance
    processing_time_ms = Column(Float, nullable=True)
    device = Column(String, nullable=True)
    input_hash = Column(String, nullable=True)
    
    # Spatial
    localization = Column(JSON, nullable=True)
    segmentation_path = Column(String, nullable=True)
    
    # Explainability
    gradcam_path = Column(String, nullable=True)
    target_layer = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=utcnow)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="evidence")
    
    __table_args__ = (
        Index("ix_evidence_model", "model_id"),
        Index("ix_evidence_prediction", "prediction"),
    )


class CouncilResultRecord(Base):
    """Council consensus fusion result."""
    __tablename__ = "council_results"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False, unique=True)
    council_run_id = Column(String, nullable=False)
    
    # Consensus
    consensus_level = Column(String, nullable=False)  # strong, moderate, weak, none
    evidence_strength = Column(String, nullable=False)
    
    # Primary finding
    primary_finding = Column(String, nullable=False)
    primary_finding_confidence = Column(Float, nullable=False, default=0.0)
    severity = Column(String, nullable=False, default="info")
    
    # Agreement
    models_agreeing = Column(Integer, nullable=False, default=0)
    models_total = Column(Integer, nullable=False, default=0)
    agreement_score = Column(Float, nullable=False, default=0.0)
    
    # Uncertainty
    ensemble_uncertainty = Column(Float, nullable=False, default=1.0)
    
    # Human review
    requires_human_review = Column(Boolean, nullable=False, default=True)
    human_review_reasons = Column(JSON, nullable=True)
    human_review_status = Column(String, nullable=True)  # pending, accepted, rejected, modified
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # Disagreements
    disagreements = Column(JSON, nullable=True)
    
    # Spatial
    spatial_agreement = Column(Float, nullable=True)
    consensus_localization = Column(JSON, nullable=True)
    
    # Timing
    total_inference_time_ms = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=utcnow)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="council_result")
    
    __table_args__ = (
        Index("ix_council_consensus", "consensus_level"),
        Index("ix_council_severity", "severity"),
        Index("ix_council_review_status", "human_review_status"),
    )


class AgentInsightRecord(Base):
    """Agentic AI reasoning output."""
    __tablename__ = "agent_insights"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False, unique=True)
    
    # Insight content
    key_insights = Column(JSON, nullable=True)
    clinical_significance = Column(JSON, nullable=True)
    recommended_actions = Column(JSON, nullable=True)
    missing_information = Column(JSON, nullable=True)
    safety_flags = Column(JSON, nullable=True)
    evidence_references = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Agent metadata
    agent_model = Column(String, nullable=True)
    agent_version = Column(String, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    raw_response = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=utcnow)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="agent_insight")


class Report(Base):
    """Generated clinical report."""
    __tablename__ = "reports"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    
    # Report content
    report_type = Column(String, nullable=False, default="radiology")
    version = Column(Integer, nullable=False, default=1)
    content = Column(JSON, nullable=True)  # Structured report data
    
    # Report metadata
    modality = Column(String, nullable=True)
    study_type = Column(String, nullable=True)
    impression = Column(Text, nullable=True)
    
    # PDF
    pdf_path = Column(String, nullable=True)
    
    # Status
    status = Column(String, nullable=False, default="draft")  # draft, final, amended
    
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="reports")


class AuditEvent(Base):
    """
    Full decision audit trail.
    Every significant action in the system is recorded here.
    """
    __tablename__ = "audit_log"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=True)
    
    # Event
    event_type = Column(String, nullable=False)  # analysis_started, model_inference, council_fusion, report_generated, human_review, etc.
    event_data = Column(JSON, nullable=True)
    
    # Actor
    actor = Column(String, nullable=False, default="system")  # system, model, agent, user
    
    # Timestamp
    timestamp = Column(DateTime, default=utcnow, nullable=False)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="audit_events")
    
    __table_args__ = (
        Index("ix_audit_type", "event_type"),
        Index("ix_audit_timestamp", "timestamp"),
        Index("ix_audit_actor", "actor"),
    )
