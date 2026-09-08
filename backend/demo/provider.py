"""
CouncilMed — Demo Simulation Provider
====================================
Isolated fallback provider for demonstration purposes.
Provides deterministic, mathematically sound Council results
without requiring actual PyTorch inference.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

from ..council.evidence import (
    ModelEvidence,
    ModelStatus,
    LocalizationData,
    BoundingBox,
    ExplainabilityData,
)
from ..council.engine import CouncilEngine, ModelCalibration
from ..pipeline import AnalysisResponse, DICOMMetadata

logger = logging.getLogger(__name__)


async def run_demo_analysis(
    file_path: str,
    modality: str | None,
    anatomy: str | None,
    enable_explainability: bool,
    enable_agent: bool,
) -> AnalysisResponse:
    """Run a controlled demonstration analysis."""
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    
    logger.info("=== DEMO MODE: Analysis Pipeline Start: %s ===", request_id)
    
    # 1. Simulate Preprocessing
    logger.info("DEMO: Preprocessing image...")
    await asyncio.sleep(0.5)
    
    mod = modality or "MRI"
    anat = anatomy or "brain"
    
    # 2. Simulate Council Member Evidence
    # We create deterministic outputs that support a "Glioblastoma" finding.
    logger.info("DEMO: Running simulated council inference...")
    await asyncio.sleep(0.8) # Classifier
    
    evidence_classifier = ModelEvidence(
        model_id="brain_mri_classifier_v1",
        model_name="BrainMRI-Classifier-v1",
        model_version="1.0.0",
        model_architecture="MobileNetV2 + Dense",
        prediction="Glioblastoma (Grade IV)",
        probabilities={
            "Glioblastoma (Grade IV)": 0.91,
            "Meningioma (Grade I)": 0.05,
            "Normal": 0.02,
            "Metastatic Lesion": 0.02
        },
        confidence=0.91,
        uncertainty=0.12,
        processing_time_ms=850.5,
        status=ModelStatus.SUCCESS,
        device="cuda:0",
        explainability=ExplainabilityData(
            target_class="Glioblastoma (Grade IV)",
            target_layer="features.18"
        ) if enable_explainability else None
    )
    
    await asyncio.sleep(0.9) # Attention
    
    evidence_attention = ModelEvidence(
        model_id="brain_mri_attention_v1",
        model_name="BrainMRI-Attention-v1",
        model_version="1.0.0",
        model_architecture="MobileNetV2 + Spatial Attention",
        prediction="Glioblastoma (Grade IV)",
        probabilities={
            "Glioblastoma (Grade IV)": 0.88,
            "Meningioma (Grade I)": 0.08,
            "Normal": 0.01,
            "Metastatic Lesion": 0.03
        },
        confidence=0.88,
        uncertainty=0.15,
        processing_time_ms=920.1,
        status=ModelStatus.SUCCESS,
        device="cuda:0",
        localization=LocalizationData(
            bounding_box=BoundingBox(120, 85, 180, 150),
            anatomical_region="Right Temporal Lobe",
            confidence=0.85
        ),
        explainability=ExplainabilityData(
            target_class="Glioblastoma (Grade IV)",
            target_layer="attention"
        ) if enable_explainability else None
    )
    
    await asyncio.sleep(0.6) # Segmenter
    
    evidence_segmenter = ModelEvidence(
        model_id="brain_mri_segmenter_v1",
        model_name="BrainMRI-Segmenter-v1",
        model_version="1.0.0",
        model_architecture="UNetLite",
        prediction="Glioblastoma (Grade IV)",
        probabilities={
            "Glioblastoma (Grade IV)": 0.95,
            "Meningioma (Grade I)": 0.01,
            "Normal": 0.01,
            "Metastatic Lesion": 0.03
        },
        confidence=0.95,
        uncertainty=0.08,
        processing_time_ms=630.2,
        status=ModelStatus.SUCCESS,
        device="cuda:0"
    )
    
    evidence_list = [evidence_classifier, evidence_attention, evidence_segmenter]
    
    # 3. Council Aggregation
    logger.info("DEMO: Running council fusion...")
    await asyncio.sleep(0.4)
    
    # We use the REAL CouncilEngine so the math matches actual logic!
    calibrations = {
        "brain_mri_classifier_v1": ModelCalibration("brain_mri_classifier_v1", auroc=0.93, calibration_ece=0.05, reliability_weight=1.0),
        "brain_mri_attention_v1": ModelCalibration("brain_mri_attention_v1", auroc=0.95, calibration_ece=0.04, reliability_weight=1.1),
        "brain_mri_segmenter_v1": ModelCalibration("brain_mri_segmenter_v1", auroc=0.91, calibration_ece=0.06, reliability_weight=0.9)
    }
    
    engine = CouncilEngine(calibrations=calibrations)
    council_result = engine.run(
        evidence_list=evidence_list,
        study_id="demo_study",
        series_id="demo_series",
        modality=mod,
        anatomy=anat
    )
    
    # 4. Agent Reasoning (Ollama)
    agent_insight = None
    if enable_agent:
        logger.info("DEMO: Running real agent reasoning via Ollama...")
        try:
            from ..agents.orchestrator import run_council_agent
            # Execute in thread pool to not block asyncio
            agent_insight = await asyncio.to_thread(run_council_agent, council_result)
        except Exception as e:
            logger.error("DEMO Agent reasoning failed: %s", e)
    
    total_time_ms = (time.perf_counter() - start_time) * 1000
    logger.info("=== DEMO MODE: Complete ===")
    
    return AnalysisResponse(
        request_id=request_id,
        council_result=council_result,
        agent_insight=agent_insight,
        metadata=DICOMMetadata(
            rows=256,
            columns=256,
            modality=mod,
            file_path=file_path
        ),
        modality=mod,
        anatomy=anat,
        image_shape=(256, 256, 3),
        total_time_ms=total_time_ms,
        hardware={"device": "demo_simulation_cpu", "source": "demo_simulation"},
        errors=[],
        warnings=["DEMO MODE ACTIVE: This analysis is a simulation and not a real diagnosis."]
    )
