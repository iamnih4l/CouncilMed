"""
CouncilMed — Council Engine Unit Tests
==========================================
Tests for the evidence-aware multi-model fusion engine.

These tests verify that the council produces correct consensus results
from genuine (or realistic mock) model evidence — NOT from random numbers.
"""

import pytest
import numpy as np

from backend.council.evidence import (
    ModelEvidence,
    ModelStatus,
    CouncilResult,
    ConsensusLevel,
    EvidenceStrength,
    SeverityLevel,
    LocalizationData,
)
from backend.council.engine import (
    CouncilEngine,
    ModelCalibration,
    compute_entropy,
    compute_model_weight,
    fuse_predictions,
    compute_agreement,
    compute_ensemble_uncertainty,
    compute_spatial_agreement,
    classify_severity,
    determine_human_review,
)


# ==============================
# Test Fixtures
# ==============================

def make_evidence(
    model_id: str = "test_model",
    prediction: str = "Glioblastoma",
    probabilities: dict | None = None,
    confidence: float = 0.9,
    uncertainty: float = 0.1,
    status: ModelStatus = ModelStatus.SUCCESS,
    localization: LocalizationData | None = None,
) -> ModelEvidence:
    """Helper to create test evidence."""
    return ModelEvidence(
        model_id=model_id,
        model_name=f"Test {model_id}",
        model_version="1.0.0",
        model_architecture="TestNet",
        prediction=prediction,
        probabilities=probabilities or {
            "Glioblastoma": 0.85,
            "Meningioma": 0.08,
            "Normal": 0.05,
            "Pituitary": 0.02,
        },
        confidence=confidence,
        uncertainty=uncertainty,
        status=status,
        processing_time_ms=100.0,
        localization=localization,
    )


# ==============================
# Entropy Tests
# ==============================

class TestEntropy:
    def test_uniform_distribution_has_max_entropy(self):
        """Uniform distribution → entropy = 1.0."""
        probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        entropy = compute_entropy(probs)
        assert abs(entropy - 1.0) < 0.01
    
    def test_certain_distribution_has_zero_entropy(self):
        """One-hot distribution → entropy = 0.0."""
        probs = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
        entropy = compute_entropy(probs)
        assert entropy == 0.0
    
    def test_confident_prediction_has_low_entropy(self):
        """90% confident → low entropy."""
        probs = {"A": 0.90, "B": 0.05, "C": 0.03, "D": 0.02}
        entropy = compute_entropy(probs)
        assert entropy < 0.5
    
    def test_empty_distribution(self):
        """Empty distribution → entropy = 0.0."""
        assert compute_entropy({}) == 0.0
    
    def test_single_class(self):
        """Single class → entropy = 0.0."""
        assert compute_entropy({"A": 1.0}) == 0.0


# ==============================
# Model Weight Tests
# ==============================

class TestModelWeight:
    def test_high_confidence_gets_high_weight(self):
        """A confident model should get a higher weight."""
        confident = make_evidence(confidence=0.95, uncertainty=0.05)
        uncertain = make_evidence(confidence=0.5, uncertainty=0.5)
        
        w_confident = compute_model_weight(confident)
        w_uncertain = compute_model_weight(uncertain)
        
        assert w_confident > w_uncertain
    
    def test_well_calibrated_model_gets_higher_weight(self):
        """A well-calibrated model should be trusted more."""
        evidence = make_evidence()
        
        good_cal = ModelCalibration("test", auroc=0.95, calibration_ece=0.05, reliability_weight=1.0)
        bad_cal = ModelCalibration("test", auroc=0.55, calibration_ece=0.40, reliability_weight=1.0)
        
        w_good = compute_model_weight(evidence, good_cal)
        w_bad = compute_model_weight(evidence, bad_cal)
        
        assert w_good > w_bad
    
    def test_weight_never_zero(self):
        """Weight should never be exactly zero (floor at 0.01)."""
        worst = make_evidence(
            confidence=0.0,
            uncertainty=1.0,
            probabilities={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        )
        
        weight = compute_model_weight(worst)
        assert weight >= 0.01


# ==============================
# Prediction Fusion Tests
# ==============================

class TestFusePredictions:
    def test_unanimous_agreement(self):
        """All models agree → fused prediction matches."""
        evidence = [
            make_evidence("A", "Glioblastoma", {"Glioblastoma": 0.9, "Normal": 0.1}),
            make_evidence("B", "Glioblastoma", {"Glioblastoma": 0.85, "Normal": 0.15}),
            make_evidence("C", "Glioblastoma", {"Glioblastoma": 0.88, "Normal": 0.12}),
        ]
        
        finding, probs, confidence = fuse_predictions(evidence)
        
        assert finding == "Glioblastoma"
        assert confidence > 0.8
        assert probs["Glioblastoma"] > probs["Normal"]
    
    def test_disagreement_reduces_confidence(self):
        """Model disagreement → lower fused confidence."""
        agreeing = [
            make_evidence("A", "Glioblastoma", {"Glioblastoma": 0.9, "Normal": 0.1}),
            make_evidence("B", "Glioblastoma", {"Glioblastoma": 0.85, "Normal": 0.15}),
            make_evidence("C", "Glioblastoma", {"Glioblastoma": 0.88, "Normal": 0.12}),
        ]
        
        disagreeing = [
            make_evidence("A", "Glioblastoma", {"Glioblastoma": 0.9, "Normal": 0.1}),
            make_evidence("B", "Normal", {"Glioblastoma": 0.2, "Normal": 0.8}),
            make_evidence("C", "Meningioma", {"Meningioma": 0.7, "Normal": 0.3}),
        ]
        
        _, _, conf_agree = fuse_predictions(agreeing)
        _, _, conf_disagree = fuse_predictions(disagreeing)
        
        assert conf_agree > conf_disagree
    
    def test_single_model_fusion(self):
        """Single model → fused result equals model output."""
        evidence = [
            make_evidence("A", "Normal", {"Normal": 0.95, "Glioblastoma": 0.05}),
        ]
        
        finding, probs, confidence = fuse_predictions(evidence)
        
        assert finding == "Normal"
        assert abs(confidence - 0.95) < 0.01


# ==============================
# Agreement Tests
# ==============================

class TestAgreement:
    def test_full_agreement(self):
        """3/3 agree → agreement = 1.0."""
        evidence = [
            make_evidence("A", "Glioblastoma"),
            make_evidence("B", "Glioblastoma"),
            make_evidence("C", "Glioblastoma"),
        ]
        
        score, count, disagreements = compute_agreement(evidence)
        
        assert score == 1.0
        assert count == 3
        assert disagreements == []
    
    def test_partial_agreement(self):
        """2/3 agree → agreement ≈ 0.67."""
        evidence = [
            make_evidence("A", "Glioblastoma"),
            make_evidence("B", "Glioblastoma"),
            make_evidence("C", "Normal"),
        ]
        
        score, count, disagreements = compute_agreement(evidence)
        
        assert abs(score - 2/3) < 0.01
        assert count == 2
        assert len(disagreements) == 1
        assert disagreements[0]["model_id"] == "C"
    
    def test_no_agreement(self):
        """3 different predictions → agreement ≈ 0.33."""
        evidence = [
            make_evidence("A", "Glioblastoma"),
            make_evidence("B", "Meningioma"),
            make_evidence("C", "Normal"),
        ]
        
        score, count, disagreements = compute_agreement(evidence)
        
        assert abs(score - 1/3) < 0.01
        assert count == 1
        assert len(disagreements) == 2


# ==============================
# Ensemble Uncertainty Tests
# ==============================

class TestEnsembleUncertainty:
    def test_agreeing_models_low_uncertainty(self):
        """Agreeing models → low ensemble uncertainty."""
        evidence = [
            make_evidence("A", "Glioblastoma", {"Glioblastoma": 0.9, "Normal": 0.1}, uncertainty=0.1),
            make_evidence("B", "Glioblastoma", {"Glioblastoma": 0.88, "Normal": 0.12}, uncertainty=0.1),
            make_evidence("C", "Glioblastoma", {"Glioblastoma": 0.85, "Normal": 0.15}, uncertainty=0.15),
        ]
        
        uncertainty = compute_ensemble_uncertainty(evidence)
        assert uncertainty < 0.4
    
    def test_disagreeing_models_high_uncertainty(self):
        """Disagreeing models → high ensemble uncertainty."""
        evidence = [
            make_evidence("A", "Glioblastoma", {"Glioblastoma": 0.9, "Normal": 0.1}, uncertainty=0.1),
            make_evidence("B", "Normal", {"Glioblastoma": 0.1, "Normal": 0.9}, uncertainty=0.1),
            make_evidence("C", "Meningioma", {"Meningioma": 0.8, "Normal": 0.2}, uncertainty=0.2),
        ]
        
        uncertainty = compute_ensemble_uncertainty(evidence)
        assert uncertainty > 0.3


# ==============================
# Spatial Agreement Tests
# ==============================

class TestSpatialAgreement:
    def test_same_location(self):
        """Models pointing to the same location → spatial agreement ≈ 1.0."""
        loc = LocalizationData(center_world_mm=(50.0, 60.0, 30.0), confidence=0.9)
        evidence = [
            make_evidence("A", localization=loc),
            make_evidence("B", localization=loc),
        ]
        
        score, consensus_loc = compute_spatial_agreement(evidence)
        
        assert score > 0.95
        assert consensus_loc is not None
    
    def test_distant_locations(self):
        """Models pointing far apart → low spatial agreement."""
        evidence = [
            make_evidence("A", localization=LocalizationData(
                center_world_mm=(10.0, 10.0, 10.0), confidence=0.9
            )),
            make_evidence("B", localization=LocalizationData(
                center_world_mm=(80.0, 80.0, 80.0), confidence=0.9
            )),
        ]
        
        score, _ = compute_spatial_agreement(evidence)
        
        assert score < 0.5


# ==============================
# Severity Classification Tests
# ==============================

class TestSeverity:
    def test_normal_finding_is_clear(self):
        assert classify_severity("Normal", 0.95, ConsensusLevel.STRONG) == SeverityLevel.CLEAR
    
    def test_high_confidence_abnormal_is_critical(self):
        assert classify_severity("Glioblastoma", 0.9, ConsensusLevel.STRONG) == SeverityLevel.CRITICAL
    
    def test_low_confidence_is_info(self):
        assert classify_severity("Glioblastoma", 0.3, ConsensusLevel.WEAK) == SeverityLevel.INFO
    
    def test_medium_confidence_is_warning(self):
        assert classify_severity("Glioblastoma", 0.6, ConsensusLevel.MODERATE) == SeverityLevel.WARNING


# ==============================
# Human Review Tests
# ==============================

class TestHumanReview:
    def test_always_requires_review(self):
        """CouncilMed ALWAYS requires human review."""
        requires, reasons = determine_human_review(
            ConsensusLevel.STRONG, 0.1, 0.95, []
        )
        assert requires is True
    
    def test_weak_consensus_adds_reason(self):
        requires, reasons = determine_human_review(
            ConsensusLevel.WEAK, 0.1, 0.95, []
        )
        assert any("weak" in r.lower() for r in reasons)
    
    def test_high_uncertainty_adds_reason(self):
        requires, reasons = determine_human_review(
            ConsensusLevel.STRONG, 0.8, 0.95, []
        )
        assert any("uncertainty" in r.lower() for r in reasons)


# ==============================
# Full Council Engine Tests
# ==============================

class TestCouncilEngine:
    def setup_method(self):
        self.engine = CouncilEngine()
    
    def test_strong_consensus_3_3(self):
        """All 3 models agree with high confidence → STRONG consensus."""
        evidence = [
            make_evidence("classifier", "Glioblastoma", 
                         {"Glioblastoma": 0.92, "Meningioma": 0.04, "Normal": 0.03, "Pituitary": 0.01},
                         confidence=0.92, uncertainty=0.08),
            make_evidence("attention", "Glioblastoma",
                         {"Glioblastoma": 0.87, "Meningioma": 0.06, "Normal": 0.05, "Pituitary": 0.02},
                         confidence=0.87, uncertainty=0.12),
            make_evidence("segmenter", "Glioblastoma",
                         {"Glioblastoma": 0.89, "Meningioma": 0.05, "Normal": 0.04, "Pituitary": 0.02},
                         confidence=0.89, uncertainty=0.10),
        ]
        
        result = self.engine.run(evidence)
        
        assert result.consensus_level == ConsensusLevel.STRONG
        assert result.primary_finding == "Glioblastoma"
        assert result.primary_finding_confidence > 0.8
        assert result.models_agreeing == 3
        assert result.agreement_score == 1.0
        assert result.ensemble_uncertainty < 0.3
        assert result.severity == SeverityLevel.CRITICAL
    
    def test_moderate_consensus_2_3(self):
        """2/3 models agree → MODERATE or WEAK consensus."""
        evidence = [
            make_evidence("classifier", "Glioblastoma",
                         {"Glioblastoma": 0.85, "Meningioma": 0.1, "Normal": 0.05},
                         confidence=0.85, uncertainty=0.1),
            make_evidence("attention", "Glioblastoma",
                         {"Glioblastoma": 0.80, "Meningioma": 0.15, "Normal": 0.05},
                         confidence=0.80, uncertainty=0.15),
            make_evidence("segmenter", "Normal",
                         {"Glioblastoma": 0.1, "Meningioma": 0.05, "Normal": 0.85},
                         confidence=0.85, uncertainty=0.1),
        ]
        
        result = self.engine.run(evidence)
        
        assert result.consensus_level in (ConsensusLevel.MODERATE, ConsensusLevel.WEAK)
        assert result.models_agreeing == 2
        assert len(result.disagreements) == 1
        assert result.requires_human_review is True
    
    def test_normal_consensus(self):
        """All models say Normal → CLEAR severity."""
        evidence = [
            make_evidence("classifier", "Normal",
                         {"Glioblastoma": 0.02, "Meningioma": 0.03, "Normal": 0.93, "Pituitary": 0.02},
                         confidence=0.93, uncertainty=0.07),
            make_evidence("attention", "Normal",
                         {"Glioblastoma": 0.05, "Meningioma": 0.02, "Normal": 0.90, "Pituitary": 0.03},
                         confidence=0.90, uncertainty=0.10),
            make_evidence("segmenter", "Normal",
                         {"Glioblastoma": 0.03, "Meningioma": 0.04, "Normal": 0.91, "Pituitary": 0.02},
                         confidence=0.91, uncertainty=0.09),
        ]
        
        result = self.engine.run(evidence)
        
        assert result.consensus_level == ConsensusLevel.STRONG
        assert result.primary_finding == "Normal"
        assert result.severity == SeverityLevel.CLEAR
    
    def test_all_models_failed(self):
        """All models error → INSUFFICIENT consensus."""
        evidence = [
            make_evidence("A", status=ModelStatus.ERROR),
            make_evidence("B", status=ModelStatus.ERROR),
        ]
        
        result = self.engine.run(evidence)
        
        assert result.consensus_level == ConsensusLevel.INSUFFICIENT
        assert result.requires_human_review is True
    
    def test_single_model(self):
        """Single model → INSUFFICIENT consensus (but still produces a finding)."""
        evidence = [
            make_evidence("classifier", "Meningioma",
                         {"Glioblastoma": 0.1, "Meningioma": 0.85, "Normal": 0.05},
                         confidence=0.85, uncertainty=0.1),
        ]
        
        result = self.engine.run(evidence)
        
        assert result.consensus_level == ConsensusLevel.INSUFFICIENT
        assert result.primary_finding == "Meningioma"
        assert result.requires_human_review is True
    
    def test_result_preserves_all_evidence(self):
        """Council result should contain all original evidence."""
        evidence = [
            make_evidence("A"),
            make_evidence("B"),
            make_evidence("C"),
        ]
        
        result = self.engine.run(evidence)
        
        assert len(result.member_evidence) == 3
        assert result.models_total == 3
    
    def test_timing_is_summed(self):
        """Total inference time should sum individual model times."""
        evidence = [
            make_evidence("A"),  # 100ms each (default)
            make_evidence("B"),
            make_evidence("C"),
        ]
        
        result = self.engine.run(evidence)
        
        assert result.total_inference_time_ms == 300.0
    
    def test_study_metadata_preserved(self):
        """Study metadata should pass through to the result."""
        evidence = [make_evidence("A")]
        
        result = self.engine.run(
            evidence,
            study_id="STUDY_001",
            series_id="SERIES_001",
            modality="MRI",
            anatomy="brain",
        )
        
        assert result.study_id == "STUDY_001"
        assert result.modality == "MRI"
        assert result.anatomy == "brain"
