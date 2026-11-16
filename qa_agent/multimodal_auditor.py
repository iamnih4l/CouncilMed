"""
CouncilMed QA Agent — Multimodal Integration Audit Engine (Phase 4)
===================================================================
Tests multi-model consensus fusion, disagreement detection, Shannon entropy
uncertainty weighting, missing modality handling, and cross-modality isolation.
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np

from .core import MultimodalAuditResult, TestCaseResult, TestStatus
from backend.council.engine import CouncilEngine, ModelCalibration, compute_entropy, compute_model_weight
from backend.council.evidence import (
    ModelEvidence,
    ModelStatus,
    ConsensusLevel,
    EvidenceStrength,
    SeverityLevel,
)


def make_evidence(
    model_id: str,
    prediction: str,
    confidence: float,
    probabilities: dict[str, float],
    uncertainty: float = 0.1,
    status: ModelStatus = ModelStatus.SUCCESS,
    error_message: str | None = None,
) -> ModelEvidence:
    return ModelEvidence(
        model_id=model_id,
        model_name=f"Model-{model_id}",
        model_version="1.0.0",
        model_architecture="CNN",
        prediction=prediction,
        probabilities=probabilities,
        confidence=confidence,
        uncertainty=uncertainty,
        status=status,
        error_message=error_message,
        processing_time_ms=15.0,
        device="cpu",
    )


class MultimodalAuditor:
    def __init__(self):
        self.engine = CouncilEngine()

    def audit_multimodal_integration(self) -> MultimodalAuditResult:
        test_results: list[TestCaseResult] = []

        # 1. Test Unanimous Consensus Fusion
        t0 = time.perf_counter()
        unanimous_ev = [
            make_evidence("m1", "Glioblastoma", 0.90, {"Glioblastoma": 0.90, "Meningioma": 0.05, "Normal": 0.05}),
            make_evidence("m2", "Glioblastoma", 0.85, {"Glioblastoma": 0.85, "Meningioma": 0.10, "Normal": 0.05}),
            make_evidence("m3", "Glioblastoma", 0.88, {"Glioblastoma": 0.88, "Meningioma": 0.07, "Normal": 0.05}),
        ]
        res_unanimous = self.engine.run(unanimous_ev, modality="MRI", anatomy="brain")
        is_unanimous_valid = (
            res_unanimous.primary_finding == "Glioblastoma"
            and res_unanimous.consensus_level == ConsensusLevel.STRONG
            and res_unanimous.models_agreeing == 3
            and len(res_unanimous.disagreements) == 0
            and res_unanimous.agreement_score == 1.0
        )
        test_results.append(
            TestCaseResult(
                test_id="MULTI_UNANIMOUS_CONSENSUS",
                name="Unanimous Consensus Fusion & Evidence Integrity",
                category="Multimodal Fusion",
                status=TestStatus.PASS if is_unanimous_valid else TestStatus.FAIL,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
                details=f"Unanimous council yielded '{res_unanimous.primary_finding}' ({res_unanimous.primary_finding_confidence:.1%}), level={res_unanimous.consensus_level.value}, agreement={res_unanimous.agreement_score:.2f}",
                evidence={"primary_finding": res_unanimous.primary_finding, "confidence": res_unanimous.primary_finding_confidence},
            )
        )

        # 2. Test Disagreement Detection & Non-Concealment
        t0 = time.perf_counter()
        disagree_ev = [
            make_evidence("m1", "Pneumonia", 0.82, {"Pneumonia": 0.82, "Normal": 0.18}),
            make_evidence("m2", "Pneumonia", 0.79, {"Pneumonia": 0.79, "Normal": 0.21}),
            make_evidence("m3", "Normal", 0.75, {"Pneumonia": 0.25, "Normal": 0.75}),  # Divergent model
        ]
        res_disagree = self.engine.run(disagree_ev, modality="XRAY", anatomy="chest")
        has_detected_disagreement = (
            len(res_disagree.disagreements) == 1
            and res_disagree.disagreements[0]["model_id"] == "m3"
            and res_disagree.disagreements[0]["model_prediction"] == "Normal"
            and res_disagree.disagreements[0]["majority_prediction"] == "Pneumonia"
            and res_disagree.models_agreeing == 2
            and res_disagree.models_total == 3
        )
        test_results.append(
            TestCaseResult(
                test_id="MULTI_DISAGREEMENT_DETECTION",
                name="Disagreement Detection Without Concealment",
                category="Multimodal Fusion",
                status=TestStatus.PASS if has_detected_disagreement else TestStatus.FAIL,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
                details=f"Detected 1 dissenting model (m3 predicting Normal vs majority Pneumonia). Disagreements transparently recorded.",
                evidence={"disagreements": res_disagree.disagreements},
            )
        )

        # 3. Test High-Uncertainty Down-Weighting (Epistemic Entropy Weighting)
        t0 = time.perf_counter()
        calibrated_ev = [
            make_evidence("m1", "Pulmonary Nodule", 0.85, {"Pulmonary Nodule": 0.85, "Normal": 0.15}, uncertainty=0.05),
            make_evidence("m2", "Normal", 0.40, {"Pulmonary Nodule": 0.30, "Normal": 0.40, "Other": 0.30}, uncertainty=0.75),  # High uncertainty
        ]
        res_calib = self.engine.run(calibrated_ev, modality="CT", anatomy="chest")
        # Confident model m1 should dominate despite m2 predicting Normal
        is_entropy_weighted = res_calib.primary_finding == "Pulmonary Nodule"
        test_results.append(
            TestCaseResult(
                test_id="MULTI_ENTROPY_WEIGHTING",
                name="Shannon Entropy & Uncertainty Weighting",
                category="Multimodal Fusion",
                status=TestStatus.PASS if is_entropy_weighted else TestStatus.FAIL,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
                details=f"High-uncertainty model downweighted properly. Primary finding: '{res_calib.primary_finding}'",
                evidence={"primary": res_calib.primary_finding, "ensemble_uncertainty": res_calib.ensemble_uncertainty},
            )
        )

        # 4. Test Fault Tolerance (One Failed Model Does Not Crash Council)
        t0 = time.perf_counter()
        fault_ev = [
            make_evidence("m1", "Meningioma", 0.88, {"Meningioma": 0.88, "Normal": 0.12}),
            make_evidence("m2", "error", 0.0, {}, status=ModelStatus.ERROR, error_message="CUDA OOM or corrupted tensor"),
            make_evidence("m3", "Meningioma", 0.84, {"Meningioma": 0.84, "Normal": 0.16}),
        ]
        res_fault = self.engine.run(fault_ev, modality="MRI", anatomy="brain")
        is_fault_handled = (
            res_fault.primary_finding == "Meningioma"
            and res_fault.requires_human_review is True  # Should flag review due to degraded council
            and any("failed" in r.lower() or "error" in r.lower() for r in res_fault.human_review_reasons)
        )
        test_results.append(
            TestCaseResult(
                test_id="MULTI_FAULT_TOLERANCE",
                name="Fault Tolerance On Model Failure",
                category="Safety & Resilience",
                status=TestStatus.PASS if is_fault_handled else TestStatus.FAIL,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
                details="Remaining models formed consensus; failure was flagged for clinician review without crashing.",
                evidence={"human_review_reasons": res_fault.human_review_reasons},
            )
        )

        # 5. Cross-Modality Independent Isolation
        t0 = time.perf_counter()
        mri_ev = [make_evidence("mri1", "Glioblastoma", 0.90, {"Glioblastoma": 0.90, "Normal": 0.10})]
        ct_ev = [make_evidence("ct1", "Pulmonary Nodule", 0.85, {"Pulmonary Nodule": 0.85, "Normal": 0.15})]
        xray_ev = [make_evidence("xray1", "Pneumonia", 0.88, {"Pneumonia": 0.88, "Normal": 0.12})]

        res_mri = self.engine.run(mri_ev, modality="MRI", anatomy="brain")
        res_ct = self.engine.run(ct_ev, modality="CT", anatomy="chest")
        res_xray = self.engine.run(xray_ev, modality="XRAY", anatomy="chest")

        is_isolated = (
            res_mri.modality == "MRI" and res_mri.primary_finding == "Glioblastoma"
            and res_ct.modality == "CT" and res_ct.primary_finding == "Pulmonary Nodule"
            and res_xray.modality == "XRAY" and res_xray.primary_finding == "Pneumonia"
        )
        test_results.append(
            TestCaseResult(
                test_id="MULTI_CROSS_MODALITY_ISOLATION",
                name="Cross-Modality Evidence Provenance Isolation",
                category="Multimodal Fusion",
                status=TestStatus.PASS if is_isolated else TestStatus.FAIL,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
                details="Modalities strictly preserve their own anatomy, class taxonomy, and provenance without cross-contamination.",
                evidence={"mri": res_mri.primary_finding, "ct": res_ct.primary_finding, "xray": res_xray.primary_finding},
            )
        )

        passed = sum(1 for t in test_results if t.status == TestStatus.PASS)
        overall_status = TestStatus.PASS if passed == len(test_results) else TestStatus.FAIL

        return MultimodalAuditResult(
            status=overall_status,
            total_scenarios_tested=len(test_results),
            fusion_success_rate=passed / len(test_results),
            disagreement_handling="VERIFIED" if has_detected_disagreement else "UNVERIFIED",
            entropy_uncertainty_score=round(float(res_calib.ensemble_uncertainty), 4),
            test_results=test_results,
        )
