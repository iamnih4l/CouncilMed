"""
CouncilMed — Report Generator Unit Tests
============================================
Tests for the structured report generation from real council evidence.
"""

import pytest

from backend.council.evidence import (
    ModelEvidence,
    ModelStatus,
    CouncilResult,
    ConsensusLevel,
    EvidenceStrength,
    SeverityLevel,
    AgentInsight,
)
from backend.reports.generator import generate_report


# ==============================
# Helpers
# ==============================

def make_evidence(
    model_id: str = "test_model",
    prediction: str = "Glioblastoma",
    probabilities: dict | None = None,
    confidence: float = 0.9,
    uncertainty: float = 0.1,
    status: ModelStatus = ModelStatus.SUCCESS,
) -> ModelEvidence:
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
    )


def make_council_result(
    finding: str = "Glioblastoma",
    confidence: float = 0.9,
    consensus: ConsensusLevel = ConsensusLevel.STRONG,
    severity: SeverityLevel = SeverityLevel.CRITICAL,
    evidence: list[ModelEvidence] | None = None,
) -> CouncilResult:
    if evidence is None:
        evidence = [
            make_evidence("A", finding, confidence=confidence),
            make_evidence("B", finding, confidence=confidence * 0.95),
            make_evidence("C", finding, confidence=confidence * 0.92),
        ]
    
    return CouncilResult(
        consensus_level=consensus,
        evidence_strength=EvidenceStrength.HIGH,
        primary_finding=finding,
        primary_finding_confidence=confidence,
        severity=severity,
        models_agreeing=len(evidence),
        models_total=len(evidence),
        agreement_score=1.0,
        ensemble_uncertainty=0.1,
        requires_human_review=True,
        human_review_reasons=["All analyses require human review"],
        member_evidence=evidence,
        disagreements=[],
        total_inference_time_ms=300.0,
        modality="MRI",
        anatomy="brain",
    )


# ==============================
# Tests
# ==============================

class TestReportGeneration:
    def test_generates_report_id(self):
        result = make_council_result()
        report = generate_report(result)
        assert report.report_id is not None
        assert len(report.report_id) > 0
    
    def test_has_required_sections(self):
        result = make_council_result()
        report = generate_report(result)
        section_titles = [s.title for s in report.sections]
        
        assert "Study Information" in section_titles
        assert "AI Council Consensus" in section_titles
        assert "Individual Model Evidence" in section_titles
        assert "Review Requirements" in section_titles
    
    def test_impression_reflects_finding(self):
        result = make_council_result(finding="Glioblastoma", confidence=0.9)
        report = generate_report(result)
        
        assert "Glioblastoma" in report.impression
        assert "90" in report.impression  # 90.0%
    
    def test_impression_mentions_consensus(self):
        result = make_council_result(consensus=ConsensusLevel.STRONG)
        report = generate_report(result)
        assert "strong" in report.impression.lower()
    
    def test_weak_consensus_shows_caution(self):
        result = make_council_result(consensus=ConsensusLevel.WEAK)
        report = generate_report(result)
        assert "caution" in report.impression.lower()
    
    def test_critical_finding_recommends_evaluation(self):
        result = make_council_result(severity=SeverityLevel.CRITICAL)
        report = generate_report(result)
        
        has_eval_rec = any(
            "evaluation" in rec.lower() or "clinical" in rec.lower()
            for rec in report.recommendations
        )
        assert has_eval_rec
    
    def test_always_recommends_clinician_review(self):
        result = make_council_result()
        report = generate_report(result)
        
        has_clinician = any(
            "clinical" in rec.lower() or "judgment" in rec.lower()
            for rec in report.recommendations
        )
        assert has_clinician
    
    def test_normal_finding_impression(self):
        result = make_council_result(
            finding="Normal",
            severity=SeverityLevel.CLEAR,
        )
        report = generate_report(result)
        assert "Normal" in report.impression
    
    def test_limitations_always_present(self):
        result = make_council_result()
        report = generate_report(result)
        assert len(report.limitations) >= 1
    
    def test_failed_model_mentioned_in_limitations(self):
        evidence = [
            make_evidence("A"),
            make_evidence("B", status=ModelStatus.ERROR),
        ]
        result = make_council_result(evidence=evidence)
        report = generate_report(result)
        
        has_failed_mention = any("failed" in lim.lower() for lim in report.limitations)
        assert has_failed_mention
    
    def test_single_model_limitation(self):
        evidence = [make_evidence("A")]
        result = make_council_result(evidence=evidence)
        report = generate_report(result)
        
        has_single_model_warning = any("1 model" in lim for lim in report.limitations)
        assert has_single_model_warning
    
    def test_text_rendering(self):
        result = make_council_result()
        report = generate_report(result)
        text = report.to_text()
        
        assert "COUNCILMED AI-ASSISTED ANALYSIS REPORT" in text
        assert "IMPRESSION" in text
        assert "DISCLAIMER" in text
        assert "Glioblastoma" in text
    
    def test_dict_serialization(self):
        result = make_council_result()
        report = generate_report(result)
        d = report.to_dict()
        
        assert d["report_id"] == report.report_id
        assert isinstance(d["sections"], list)
        assert len(d["sections"]) >= 3
    
    def test_agent_insight_included(self):
        result = make_council_result()
        agent = AgentInsight(
            key_insights=["Consistent with high-grade glioma"],
            safety_flags=["High uncertainty for differential diagnosis"],
            recommended_actions=["Compare with prior imaging"],
            confidence=0.7,
            agent_model="llama3",
        )
        report = generate_report(result, agent_insight=agent)
        
        section_titles = [s.title for s in report.sections]
        assert "AI Agent Analysis" in section_titles
    
    def test_disagreements_section(self):
        evidence = [
            make_evidence("A", "Glioblastoma"),
            make_evidence("B", "Meningioma"),
        ]
        result = CouncilResult(
            consensus_level=ConsensusLevel.WEAK,
            evidence_strength=EvidenceStrength.LOW,
            primary_finding="Glioblastoma",
            primary_finding_confidence=0.6,
            severity=SeverityLevel.WARNING,
            models_agreeing=1,
            models_total=2,
            agreement_score=0.5,
            ensemble_uncertainty=0.5,
            requires_human_review=True,
            human_review_reasons=["Weak consensus"],
            member_evidence=evidence,
            disagreements=[{
                "model_id": "B",
                "model_prediction": "Meningioma",
                "model_confidence": 0.7,
                "majority_prediction": "Glioblastoma",
                "confidence_gap": 0.1,
            }],
            total_inference_time_ms=200.0,
        )
        report = generate_report(result)
        
        section_titles = [s.title for s in report.sections]
        assert "Model Disagreements" in section_titles
