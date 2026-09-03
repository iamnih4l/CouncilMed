"""
CouncilMed — FastAPI Application
====================================
Production-grade REST API for the medical imaging AI council.

Replaces the previous main.py which had:
  - Hardcoded overall_agreement_score=1.0
  - Hardcoded requires_manual_review=False
  - Hardcoded severity="Calculated"
  - Open CORS (allow_origins=["*"])
  
All of those are now computed from real inference results.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CouncilResultResponse,
    ModelEvidenceResponse,
    AgentInsightResponse,
    RegistryResponse,
    ModelInfo,
    HealthResponse,
    ReviewDecision,
    ReportRequest,
    ReportResponse,
    ReportSectionResponse,
)

logger = logging.getLogger(__name__)

# Pipeline instance (initialized on startup)
_pipeline = None

# Accepted origins — restricted to localhost only
ALLOWED_ORIGINS = [
    "http://localhost:1420",   # Tauri dev
    "http://localhost:5173",   # Vite dev
    "http://localhost:3000",   # Alternate dev
    "http://127.0.0.1:1420",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "tauri://localhost",       # Tauri production
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown."""
    global _pipeline
    
    # Startup
    logger.info("CouncilMed Backend starting...")
    
    # Initialize database
    from ..database import init_database
    await init_database()
    
    # Initialize pipeline
    from ..paths import get_base_dir
    
    registry_dir = get_base_dir() / "model_registry"
    output_dir = get_base_dir() / "output"
    
    _pipeline = AnalysisPipeline(
        registry_dir=registry_dir,
        output_dir=output_dir,
    )
    
    summary = _pipeline.get_registry_summary()
    logger.info("Pipeline ready: %s", summary)
    
    yield
    
    # Shutdown
    from ..database import close_database
    await close_database()
    logger.info("CouncilMed Backend shut down")


# Create application
app = FastAPI(
    title="CouncilMed API",
    description="Medical Imaging AI Council — Real Inference, Real Evidence, Real Consensus",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — restricted to localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)


# ==============================
# Health
# ==============================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    summary = _pipeline.get_registry_summary() if _pipeline else {}
    
    return HealthResponse(
        status="healthy" if _pipeline else "initializing",
        version="0.1.0",
        models_loaded=summary.get("registry", {}).get("installed", 0),
        device=summary.get("hardware", {}).get("device", "cpu"),
        hardware=summary.get("hardware", {}),
    )


# ==============================
# Analysis
# ==============================

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: AnalyzeRequest):
    """
    Analyze a medical image through the AI Council.
    
    The pipeline runs:
    1. Parse input (DICOM, NIfTI, or standard image)
    2. Detect modality and anatomy
    3. Select council members from registry
    4. Run real model inference on each member
    5. Aggregate evidence
    6. Run council fusion
    7. Generate explainability (Grad-CAM)
    8. Optionally run agentic AI reasoning
    
    All predictions are GENUINE — computed from real neural network forward passes.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file exists
    if not Path(request.file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    
    from ..pipeline import AnalysisRequest
    
    analysis_request = AnalysisRequest(
        file_path=request.file_path,
        study_id=request.study_id,
        modality_override=request.modality_override,
        anatomy_override=request.anatomy_override,
        enable_explainability=request.enable_explainability,
        enable_agent=request.enable_agent,
    )
    
    # Run the pipeline
    result = _pipeline.analyze(analysis_request)
    
    # Store in database
    await _store_analysis(result)
    
    # Map to API response
    return _map_to_response(result)


@app.post("/api/analyze/upload", response_model=AnalyzeResponse)
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    modality: str | None = Query(None),
    anatomy: str | None = Query(None),
    enable_explainability: bool = Query(True),
    enable_agent: bool = Query(False),
):
    """
    Upload and analyze a medical image.
    Accepts DICOM, NIfTI, JPEG, and PNG files.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file size (max 100MB)
    MAX_SIZE = 100 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)} bytes (max {MAX_SIZE})"
        )
    
    # Validate file extension
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    allowed_extensions = {".dcm", ".dicom", ".nii", ".gz", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if suffix and suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {suffix}"
        )
    
    # Save to temp file
    from ..paths import get_base_dir
    upload_dir = get_base_dir() / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    temp_path = upload_dir / f"{uuid.uuid4()}{suffix}"
    with open(temp_path, "wb") as f:
        f.write(content)
    
    try:
        from ..pipeline import AnalysisRequest
        
        analysis_request = AnalysisRequest(
            file_path=str(temp_path),
            modality_override=modality,
            anatomy_override=anatomy,
            enable_explainability=enable_explainability,
            enable_agent=enable_agent,
        )
        
        result = _pipeline.analyze(analysis_request)
        await _store_analysis(result)
        
        return _map_to_response(result)
    
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# Registry
# ==============================

@app.get("/api/models", response_model=RegistryResponse)
async def list_models():
    """List all registered models."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    summary = _pipeline.registry.summary()
    models = []
    
    for manifest in _pipeline.registry.list_all():
        models.append(ModelInfo(
            model_id=manifest.model_id,
            name=manifest.name,
            version=manifest.version,
            architecture=manifest.architecture,
            architecture_family=manifest.architecture_family,
            framework=manifest.framework.value,
            modality=manifest.modality,
            anatomy=manifest.anatomy,
            num_classes=manifest.num_classes,
            class_names=manifest.class_names,
            installed=manifest.installed,
            validated=manifest.metrics.validated,
            accuracy=manifest.metrics.accuracy,
            auroc=manifest.metrics.auroc,
        ))
    
    return RegistryResponse(
        total_models=summary["total_models"],
        installed=summary["installed"],
        validated=summary["validated"],
        modalities=summary["modalities"],
        anatomies=summary["anatomies"],
        models=models,
    )


# ==============================
# History
# ==============================

@app.get("/api/analyses")
async def list_analyses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List recent analyses."""
    from ..database import get_session, Analysis
    from sqlalchemy import select
    
    async with await get_session() as session:
        stmt = (
            select(Analysis)
            .order_by(Analysis.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(stmt)
        analyses = result.scalars().all()
        
        return [
            {
                "id": a.id,
                "request_id": a.request_id,
                "modality": a.modality,
                "anatomy": a.anatomy,
                "status": a.status,
                "total_time_ms": a.total_time_ms,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in analyses
        ]


# ==============================
# Human Review
# ==============================

@app.post("/api/review")
async def submit_review(review: ReviewDecision):
    """Submit a human review decision for a council result."""
    from ..database import get_session, CouncilResultRecord, AuditEvent
    from sqlalchemy import select
    
    async with await get_session() as session:
        stmt = select(CouncilResultRecord).where(
            CouncilResultRecord.analysis_id == review.analysis_id
        )
        result = await session.execute(stmt)
        council_record = result.scalar_one_or_none()
        
        if not council_record:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        council_record.human_review_status = review.decision
        council_record.reviewed_by = review.reviewer_name
        council_record.reviewed_at = datetime.now(timezone.utc)
        
        # Audit trail
        audit = AuditEvent(
            analysis_id=review.analysis_id,
            event_type="human_review",
            event_data={
                "decision": review.decision,
                "reviewer": review.reviewer_name,
                "notes": review.notes,
                "modified_finding": review.modified_finding,
            },
            actor=review.reviewer_name or "user",
        )
        session.add(audit)
        
        await session.commit()
    
    return {"status": "ok", "decision": review.decision}


# ==============================
# Reports
# ==============================

# In-memory report cache (production would use database)
_reports_cache: dict[str, dict] = {}


@app.post("/api/reports/generate", response_model=ReportResponse)
async def generate_report_endpoint(request: ReportRequest):
    """Generate a structured report from a completed analysis."""
    from ..reports.generator import generate_report
    from ..council.evidence import (
        CouncilResult, ModelEvidence, ModelStatus,
        ConsensusLevel, EvidenceStrength, SeverityLevel,
    )

    # Reconstruct council result from the analysis ID
    # In production, this would query the database for the stored result
    # For now, we use the in-memory pipeline result cache
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Try to reconstruct from pipeline's last result or database
    try:
        from ..database import get_session, CouncilResultRecord, ModelEvidenceRecord
        from sqlalchemy import select

        async with await get_session() as session:
            # Get council result
            stmt = select(CouncilResultRecord).where(
                CouncilResultRecord.analysis_id == request.analysis_id
            )
            result = await session.execute(stmt)
            council_record = result.scalar_one_or_none()

            if not council_record:
                raise HTTPException(status_code=404, detail="Analysis not found")

            # Get model evidence
            stmt = select(ModelEvidenceRecord).where(
                ModelEvidenceRecord.analysis_id == request.analysis_id
            )
            result = await session.execute(stmt)
            evidence_records = result.scalars().all()

            # Reconstruct ModelEvidence objects
            member_evidence = []
            for er in evidence_records:
                member_evidence.append(ModelEvidence(
                    model_id=er.model_id,
                    model_name=er.model_name,
                    model_version=er.model_version,
                    model_architecture=er.model_architecture,
                    prediction=er.prediction or "",
                    probabilities=er.probabilities or {},
                    confidence=er.confidence or 0.0,
                    uncertainty=er.uncertainty or 0.0,
                    status=ModelStatus(er.status),
                    processing_time_ms=er.processing_time_ms or 0.0,
                    error_message=er.error_message,
                    device=er.device or "cpu",
                ))

            # Reconstruct CouncilResult
            council_result = CouncilResult(
                council_run_id=council_record.council_run_id,
                consensus_level=ConsensusLevel(council_record.consensus_level),
                evidence_strength=EvidenceStrength(council_record.evidence_strength),
                primary_finding=council_record.primary_finding,
                primary_finding_confidence=council_record.primary_finding_confidence,
                severity=SeverityLevel(council_record.severity),
                models_agreeing=council_record.models_agreeing,
                models_total=council_record.models_total,
                agreement_score=council_record.agreement_score,
                ensemble_uncertainty=council_record.ensemble_uncertainty,
                requires_human_review=council_record.requires_human_review,
                human_review_reasons=council_record.human_review_reasons or [],
                member_evidence=member_evidence,
                disagreements=council_record.disagreements or [],
                spatial_agreement=council_record.spatial_agreement or 0.0,
                total_inference_time_ms=council_record.total_inference_time_ms,
            )

            # Generate report
            report = generate_report(
                council_result,
                study_date=request.study_date or "",
            )

            # Cache it
            _reports_cache[report.report_id] = report.to_dict()

            # Map to response
            def section_to_response(s):
                return ReportSectionResponse(
                    title=s.title,
                    content=s.content,
                    subsections=[section_to_response(sub) for sub in s.subsections],
                    severity=s.severity,
                )

            return ReportResponse(
                report_id=report.report_id,
                version=report.version,
                study_type=report.study_type,
                modality=report.modality,
                anatomy=report.anatomy,
                study_date=report.study_date,
                sections=[section_to_response(s) for s in report.sections],
                impression=report.impression,
                recommendations=report.recommendations,
                limitations=report.limitations,
                generated_at=report.generated_at,
                report_status=report.report_status,
                report_text=report.to_text(),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Report generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    """Retrieve a previously generated report."""
    if report_id not in _reports_cache:
        raise HTTPException(status_code=404, detail="Report not found")
    return _reports_cache[report_id]


# ==============================
# Internal helpers
# ==============================

def _map_to_response(result) -> AnalyzeResponse:
    """Map pipeline AnalysisResponse to API AnalyzeResponse."""
    cr = result.council_result
    
    evidence_responses = []
    for ev in cr.member_evidence:
        evidence_responses.append(ModelEvidenceResponse(
            model_id=ev.model_id,
            model_name=ev.model_name,
            model_version=ev.model_version,
            model_architecture=ev.model_architecture,
            prediction=ev.prediction,
            probabilities=ev.probabilities,
            confidence=ev.confidence,
            uncertainty=ev.uncertainty,
            processing_time_ms=ev.processing_time_ms,
            status=ev.status.value,
            error_message=ev.error_message,
            device=ev.device,
        ))
    
    council_response = CouncilResultResponse(
        council_run_id=cr.council_run_id,
        consensus_level=cr.consensus_level.value,
        evidence_strength=cr.evidence_strength.value,
        primary_finding=cr.primary_finding,
        primary_finding_confidence=cr.primary_finding_confidence,
        severity=cr.severity.value,
        models_agreeing=cr.models_agreeing,
        models_total=cr.models_total,
        agreement_score=cr.agreement_score,
        ensemble_uncertainty=cr.ensemble_uncertainty,
        requires_human_review=cr.requires_human_review,
        human_review_reasons=cr.human_review_reasons,
        member_evidence=evidence_responses,
        disagreements=cr.disagreements,
        spatial_agreement=cr.spatial_agreement,
        total_inference_time_ms=cr.total_inference_time_ms,
    )
    
    agent_response = None
    if result.agent_insight:
        ai = result.agent_insight
        agent_response = AgentInsightResponse(
            key_insights=ai.key_insights,
            clinical_significance=ai.clinical_significance,
            recommended_actions=ai.recommended_actions,
            missing_information=ai.missing_information,
            safety_flags=ai.safety_flags,
            evidence_references=ai.evidence_references,
            confidence=ai.confidence,
            agent_model=ai.agent_model,
            processing_time_ms=ai.processing_time_ms,
        )
    
    return AnalyzeResponse(
        request_id=result.request_id,
        council_result=council_response,
        agent_insight=agent_response,
        modality=result.modality,
        anatomy=result.anatomy,
        image_shape=list(result.image_shape),
        total_time_ms=result.total_time_ms,
        hardware=result.hardware,
        errors=result.errors,
        warnings=result.warnings,
    )


async def _store_analysis(result) -> None:
    """Store analysis results in the database."""
    try:
        from ..database import get_session, Analysis, ModelEvidenceRecord, CouncilResultRecord, AuditEvent
        
        async with await get_session() as session:
            # Create analysis record
            analysis = Analysis(
                request_id=result.request_id,
                file_path=result.metadata.file_path if result.metadata else None,
                modality=result.modality,
                anatomy=result.anatomy,
                image_shape=list(result.image_shape),
                status="completed",
                total_time_ms=result.total_time_ms,
                device=result.hardware.get("device", "cpu"),
                hardware_info=result.hardware,
                completed_at=datetime.now(timezone.utc),
            )
            session.add(analysis)
            await session.flush()  # Get the analysis ID
            
            # Store individual evidence
            cr = result.council_result
            for ev in cr.member_evidence:
                evidence_record = ModelEvidenceRecord(
                    analysis_id=analysis.id,
                    model_id=ev.model_id,
                    model_name=ev.model_name,
                    model_version=ev.model_version,
                    model_architecture=ev.model_architecture,
                    prediction=ev.prediction,
                    probabilities=ev.probabilities,
                    confidence=ev.confidence,
                    uncertainty=ev.uncertainty,
                    status=ev.status.value,
                    error_message=ev.error_message,
                    processing_time_ms=ev.processing_time_ms,
                    device=ev.device,
                    input_hash=ev.input_hash,
                    gradcam_path=ev.explainability.gradcam_path if ev.explainability else None,
                    target_layer=ev.explainability.target_layer if ev.explainability else None,
                )
                session.add(evidence_record)
            
            # Store council result
            council_record = CouncilResultRecord(
                analysis_id=analysis.id,
                council_run_id=cr.council_run_id,
                consensus_level=cr.consensus_level.value,
                evidence_strength=cr.evidence_strength.value,
                primary_finding=cr.primary_finding,
                primary_finding_confidence=cr.primary_finding_confidence,
                severity=cr.severity.value,
                models_agreeing=cr.models_agreeing,
                models_total=cr.models_total,
                agreement_score=cr.agreement_score,
                ensemble_uncertainty=cr.ensemble_uncertainty,
                requires_human_review=cr.requires_human_review,
                human_review_reasons=cr.human_review_reasons,
                human_review_status="pending",
                disagreements=cr.disagreements,
                spatial_agreement=cr.spatial_agreement,
                total_inference_time_ms=cr.total_inference_time_ms,
            )
            session.add(council_record)
            
            # Audit trail
            audit = AuditEvent(
                analysis_id=analysis.id,
                event_type="analysis_completed",
                event_data={
                    "primary_finding": cr.primary_finding,
                    "confidence": cr.primary_finding_confidence,
                    "consensus": cr.consensus_level.value,
                    "models_used": [ev.model_id for ev in cr.member_evidence],
                },
                actor="system",
            )
            session.add(audit)
            
            await session.commit()
            logger.info("Analysis stored: %s", analysis.id)
            
    except Exception as e:
        logger.error("Failed to store analysis: %s", e, exc_info=True)


# ==============================
# Knowledge RAG Endpoints
# ==============================

@app.get("/api/knowledge/status")
async def knowledge_status():
    """Get knowledge base status."""
    try:
        from ..knowledge.rag import KnowledgeStore
        store = KnowledgeStore()
        return {
            "available": True,
            "document_count": store.document_count,
            "backend": "chromadb",
        }
    except ImportError:
        # Check baseline knowledge
        from ..knowledge.rag import BASELINE_KNOWLEDGE
        return {
            "available": True,
            "document_count": len(BASELINE_KNOWLEDGE),
            "backend": "baseline",
        }
    except Exception as e:
        return {
            "available": False,
            "document_count": 0,
            "backend": "none",
            "error": str(e),
        }


@app.post("/api/knowledge/ingest")
async def ingest_knowledge():
    """Ingest knowledge files from data/knowledge/ directory."""
    try:
        from ..knowledge.rag import KnowledgeStore
        from ..paths import get_base_dir
        
        knowledge_dir = get_base_dir() / "data" / "knowledge"
        if not knowledge_dir.exists():
            raise HTTPException(404, "Knowledge directory not found")
        
        store = KnowledgeStore()
        count = store.ingest_directory(knowledge_dir)
        
        return {
            "status": "success",
            "chunks_ingested": count,
            "total_documents": store.document_count,
        }
    except ImportError:
        raise HTTPException(
            501,
            "ChromaDB not installed. Install with: pip install chromadb"
        )


@app.post("/api/knowledge/query")
async def query_knowledge(query: str, n_results: int = 5):
    """Query the knowledge base."""
    try:
        from ..knowledge.rag import KnowledgeStore
        
        store = KnowledgeStore()
        if store.document_count == 0:
            # Fall back to baseline
            from ..knowledge.rag import get_baseline_context
            baseline = get_baseline_context(query)
            return {
                "source": "baseline",
                "results": [{"content": baseline, "source": "baseline"}] if baseline else [],
            }
        
        result = store.query(query, n_results=n_results)
        return {
            "source": "chromadb",
            "results": [
                {
                    "content": chunk.content,
                    "source": chunk.source,
                    "section": chunk.section,
                    "distance": dist,
                }
                for chunk, dist in zip(result.chunks, result.distances)
            ],
        }
    except ImportError:
        from ..knowledge.rag import get_baseline_context
        baseline = get_baseline_context(query)
        return {
            "source": "baseline",
            "results": [{"content": baseline, "source": "baseline"}] if baseline else [],
        }

