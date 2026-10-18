"""
CouncilMed — FastAPI Application (Production Patched)
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..council.evidence import ModelStatus
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
    OverlayResponse,
    MediaResponse,
)
from ..media import validate_safe_id

logger = logging.getLogger(__name__)

_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("CouncilMed Backend starting...")

    from ..database import init_database
    await init_database()

    from ..paths import get_base_dir
    from ..pipeline import AnalysisPipeline

    registry_dir = get_base_dir() / "model_registry"
    output_dir = get_base_dir() / "output"

    _pipeline = AnalysisPipeline(
        registry_dir=registry_dir,
        output_dir=output_dir,
    )

    summary = _pipeline.get_registry_summary()
    logger.info("Pipeline ready: %s", summary)
    yield

    from ..database import close_database
    await close_database()


app = FastAPI(
    title="CouncilMed API",
    description="Medical Imaging AI Council — Real Inference, Real Evidence, Real Consensus",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    summary = _pipeline.get_registry_summary() if _pipeline else {}
    return HealthResponse(
        status="healthy" if _pipeline else "initializing",
        version="0.1.0",
        models_loaded=summary.get("registry", {}).get("installed", 0),
        device=summary.get("hardware", {}).get("device", "cpu"),
        hardware=summary.get("hardware", {}),
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: AnalyzeRequest):
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not Path(request.file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    from ..pipeline import AnalysisRequest as PipelineRequest

    analysis_request = PipelineRequest(
        file_path=request.file_path,
        study_id=request.study_id,
        modality_override=request.modality_override,
        anatomy_override=request.anatomy_override,
        enable_explainability=request.enable_explainability,
        enable_agent=True,
    )

    result = _pipeline.analyze(analysis_request)
    await _store_analysis(result)
    return _map_to_response(result)


@app.post("/api/analyze/upload", response_model=AnalyzeResponse)
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    modality: str | None = Query(None),
    anatomy: str | None = Query(None),
    enable_explainability: bool = Query(True),
    enable_agent: bool = Query(True),
):
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    MAX_SIZE = 100 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large: {len(content)} bytes")

    filename = file.filename or "upload.png"
    suffix = Path(filename).suffix.lower() or ".png"
    allowed_extensions = {".dcm", ".dicom", ".nii", ".gz", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix}")

    from ..paths import get_base_dir
    upload_dir = get_base_dir() / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(filename).stem[:32]
    temp_path = upload_dir / f"{safe_stem}_{uuid.uuid4().hex[:8]}{suffix}"
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        from ..pipeline import AnalysisRequest as PipelineRequest

        analysis_request = PipelineRequest(
            file_path=str(temp_path),
            modality_override=modality,
            anatomy_override=anatomy,
            enable_explainability=enable_explainability,
            enable_agent=True,
        )

        result = _pipeline.analyze(analysis_request)
        await _store_analysis(result)
        return _map_to_response(result)
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug")
async def debug_log(request: Request):
    data = await request.body()
    with open("frontend_debug.log", "a", encoding="utf-8") as f:
        f.write(data.decode("utf-8", errors="ignore") + "\n")
    return {"status": "ok"}


@app.get("/api/analyses/{request_id}/preview")
async def get_analysis_preview(request_id: str):
    try:
        validate_safe_id(request_id, "request_id")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    from ..paths import get_base_dir
    preview_path = get_base_dir() / "output" / "previews" / request_id / "preview.png"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")

    return FileResponse(
        str(preview_path),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600", "Access-Control-Allow-Origin": "*"},
    )


@app.get("/api/analyses/{request_id}/overlays/{model_id}")
async def get_analysis_overlay(request_id: str, model_id: str):
    try:
        validate_safe_id(request_id, "request_id")
        validate_safe_id(model_id, "model_id")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    from ..paths import get_base_dir
    overlay_path = get_base_dir() / "output" / "explainability" / request_id / f"{model_id}_gradcam.png"
    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="Overlay not found")

    return FileResponse(
        str(overlay_path),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600", "Access-Control-Allow-Origin": "*"},
    )


@app.get("/api/models", response_model=RegistryResponse)
async def list_models():
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


@app.get("/api/analyses")
async def list_analyses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    from ..database import get_session, Analysis
    from sqlalchemy import select

    async with await get_session() as session:
        stmt = select(Analysis).order_by(Analysis.created_at.desc()).limit(limit).offset(offset)
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


@app.post("/api/review")
async def submit_review(review: ReviewDecision):
    from ..database import get_session, CouncilResultRecord, AuditEvent
    from sqlalchemy import select

    async with await get_session() as session:
        stmt = select(CouncilResultRecord).where(CouncilResultRecord.analysis_id == review.analysis_id)
        result = await session.execute(stmt)
        council_record = result.scalar_one_or_none()
        if not council_record:
            raise HTTPException(status_code=404, detail="Analysis not found")

        council_record.human_review_status = review.decision
        council_record.reviewed_by = review.reviewer_name
        council_record.reviewed_at = datetime.now(timezone.utc)

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


_reports_cache: dict[str, dict] = {}


@app.post("/api/reports/generate", response_model=ReportResponse)
async def generate_report_endpoint(request: ReportRequest):
    from ..reports.generator import generate_report
    from ..council.evidence import (
        CouncilResult, ModelEvidence, ConsensusLevel, EvidenceStrength, SeverityLevel,
        ExplainabilityData, LocalizationData, BoundingBox, ModelStatus, AgentInsight,
    )
    from ..database import get_session, CouncilResultRecord, ModelEvidenceRecord, Analysis, AgentInsightRecord
    from sqlalchemy import select, or_

    async with await get_session() as session:
        stmt = (
            select(CouncilResultRecord, Analysis)
            .join(Analysis, Analysis.id == CouncilResultRecord.analysis_id)
            .where(
                or_(
                    CouncilResultRecord.analysis_id == request.analysis_id,
                    Analysis.request_id == request.analysis_id,
                )
            )
        )
        result = await session.execute(stmt)
        row = result.first()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        council_record, analysis_record = row

        stmt = select(ModelEvidenceRecord).where(ModelEvidenceRecord.analysis_id == council_record.analysis_id)
        result = await session.execute(stmt)
        evidence_records = result.scalars().all()

        member_evidence = []
        for er in evidence_records:
            exp_data = ExplainabilityData(gradcam_path=er.gradcam_path, target_layer=er.target_layer, target_class_idx=0) if (er.gradcam_path or er.target_layer) else None
            loc_data = None
            if er.localization and "bounding_box" in er.localization:
                bb = er.localization["bounding_box"]
                loc_data = LocalizationData(bounding_box=BoundingBox(x_min=bb.get("x_min", 0.0), y_min=bb.get("y_min", 0.0), x_max=bb.get("x_max", 0.0), y_max=bb.get("y_max", 0.0)))

            try:
                ev_status = ModelStatus(er.status)
            except Exception:
                ev_status = ModelStatus.SUCCESS

            member_evidence.append(ModelEvidence(
                model_id=er.model_id,
                model_name=er.model_name,
                model_version=er.model_version,
                model_architecture=er.model_architecture,
                prediction=er.prediction or "",
                probabilities=er.probabilities or {},
                confidence=er.confidence or 0.0,
                uncertainty=er.uncertainty or 0.0,
                status=ev_status,
                processing_time_ms=er.processing_time_ms or 0.0,
                error_message=er.error_message,
                device=er.device or "cpu",
                explainability=exp_data,
                localization=loc_data,
            ))

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
            modality=analysis_record.modality,
            anatomy=analysis_record.anatomy,
            study_id=analysis_record.request_id,
        )

        # Check for agent insight record
        agent_insight = None
        stmt_ai = select(AgentInsightRecord).where(AgentInsightRecord.analysis_id == council_record.analysis_id)
        ai_res = await session.execute(stmt_ai)
        ai_record = ai_res.scalar_one_or_none()
        if ai_record:
            agent_insight = AgentInsight(
                key_insights=ai_record.key_insights or [],
                clinical_significance=ai_record.clinical_significance or [],
                recommended_actions=ai_record.recommended_actions or [],
                missing_information=ai_record.missing_information or [],
                safety_flags=ai_record.safety_flags or [],
                evidence_references=ai_record.evidence_references or [],
                confidence=ai_record.confidence or 0.0,
                agent_model=ai_record.agent_model or "",
            )

        study_date = request.study_date or (
            analysis_record.completed_at.strftime("%Y-%m-%d") if analysis_record.completed_at else ""
        )
        report = generate_report(council_result, agent_insight=agent_insight, study_date=study_date)
        _reports_cache[report.report_id] = report.to_dict()

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


@app.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    if report_id not in _reports_cache:
        raise HTTPException(status_code=404, detail="Report not found")
    return _reports_cache[report_id]


def _map_to_response(result) -> AnalyzeResponse:
    cr = result.council_result
    evidence_responses = []
    for ev in cr.member_evidence:
        bbox = None
        if ev.localization and ev.localization.bounding_box:
            bb = ev.localization.bounding_box
            bbox = {
                "x": int(bb.x_min),
                "y": int(bb.y_min),
                "width": int(max(12, bb.x_max - bb.x_min)),
                "height": int(max(12, bb.y_max - bb.y_min)),
            }

        explainability_status = None
        if ev.explainability:
            explainability_status = "available" if ev.explainability.gradcam_path else "generated_no_artifact"
        elif ev.status == ModelStatus.SUCCESS:
            explainability_status = "unsupported"

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
            bounding_box=bbox,
            explainability_status=explainability_status,
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

    preview_url = f"/api/analyses/{result.request_id}/preview" if result.preview_path else None
    overlay_responses = [
        OverlayResponse(
            model_id=ov.model_id,
            url=f"/api/analyses/{result.request_id}/overlays/{ov.model_id}",
            overlay_type=ov.overlay_type,
        )
        for ov in getattr(result, "overlays", [])
        if ov.png_path
    ]

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
        media=MediaResponse(preview_url=preview_url, overlays=overlay_responses),
    )


async def _store_analysis(result) -> None:
    try:
        from ..database import get_session, Analysis, ModelEvidenceRecord, CouncilResultRecord, AuditEvent

        async with await get_session() as session:
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
            await session.flush()

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
                    localization={
                        "bounding_box": {
                            "x_min": ev.localization.bounding_box.x_min,
                            "y_min": ev.localization.bounding_box.y_min,
                            "x_max": ev.localization.bounding_box.x_max,
                            "y_max": ev.localization.bounding_box.y_max,
                        }
                    } if ev.localization and ev.localization.bounding_box else None,
                )
                session.add(evidence_record)

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
    except Exception as e:
        logger.error("Failed to store analysis: %s", e, exc_info=True)
