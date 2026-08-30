"""
CouncilMed — Integration Tests
===================================
End-to-end tests for the pipeline components working together.
Tests the flow: Evidence → Council → Agent → Report
without requiring actual model weights or Ollama.
"""

import pytest
import json
from pathlib import Path

from backend.council.evidence import (
    ModelEvidence,
    ModelStatus,
    CouncilResult,
    ConsensusLevel,
    EvidenceStrength,
    SeverityLevel,
)
from backend.council.engine import CouncilEngine
from backend.reports.generator import generate_report
from backend.agents.orchestrator import (
    format_council_evidence,
    build_system_prompt,
    _parse_agent_response,
    _retrieve_knowledge_context,
)
from backend.knowledge.rag import (
    chunk_markdown,
    get_baseline_context,
    BASELINE_KNOWLEDGE,
)


# ==============================
# Helpers
# ==============================

def make_realistic_evidence(
    model_id: str,
    prediction: str,
    confidence: float,
    probabilities: dict | None = None,
    status: ModelStatus = ModelStatus.SUCCESS,
) -> ModelEvidence:
    """Create realistic model evidence matching real inference output."""
    if probabilities is None:
        probabilities = {prediction: confidence}
        remaining = 1.0 - confidence
        # Add secondary classes
        secondary = ["Normal", "Glioblastoma", "Meningioma", "Pituitary"]
        secondary = [s for s in secondary if s != prediction]
        for i, cls in enumerate(secondary[:3]):
            share = remaining * (0.5 ** (i + 1))
            probabilities[cls] = round(share, 4)

    return ModelEvidence(
        model_id=model_id,
        model_name=f"TestModel-{model_id}",
        model_version="1.0.0",
        model_architecture="ResNet50",
        prediction=prediction,
        probabilities=probabilities,
        confidence=confidence,
        uncertainty=1.0 - confidence,
        status=status,
        processing_time_ms=150.0,
        device="cpu",
    )


# ==============================
# Pipeline Integration Tests
# ==============================

class TestCouncilToReport:
    """Tests: Evidence → Council Engine → Report Generator"""

    def test_full_pipeline_strong_consensus(self):
        """3 models agree → strong consensus → clean report."""
        evidence = [
            make_realistic_evidence("densenet", "Glioblastoma", 0.92),
            make_realistic_evidence("attention", "Glioblastoma", 0.88),
            make_realistic_evidence("swin", "Glioblastoma", 0.85),
        ]

        engine = CouncilEngine()
        result = engine.run(evidence, modality="MRI", anatomy="brain")

        # Council should reach strong consensus
        assert result.consensus_level == ConsensusLevel.STRONG
        assert result.primary_finding == "Glioblastoma"
        assert result.models_agreeing == 3
        assert result.severity == SeverityLevel.CRITICAL

        # Report should reflect the consensus
        report = generate_report(result)
        assert "Glioblastoma" in report.impression
        assert "strong" in report.impression.lower()
        assert len(report.sections) >= 3
        assert len(report.recommendations) >= 1

        # Report text should be renderable
        text = report.to_text()
        assert "COUNCILMED" in text
        assert "IMPRESSION" in text

    def test_full_pipeline_weak_consensus(self):
        """Models disagree → weak consensus → cautious report."""
        evidence = [
            make_realistic_evidence("densenet", "Glioblastoma", 0.55),
            make_realistic_evidence("attention", "Meningioma", 0.60),
            make_realistic_evidence("swin", "Glioblastoma", 0.52),
        ]

        engine = CouncilEngine()
        result = engine.run(evidence, modality="MRI", anatomy="brain")

        # Consensus should be weak or moderate due to disagreement
        assert result.consensus_level in (ConsensusLevel.WEAK, ConsensusLevel.MODERATE)
        assert result.agreement_score < 1.0
        assert len(result.disagreements) >= 1

        # Report should show caution
        report = generate_report(result)
        text = report.to_text()
        assert len(report.limitations) >= 1
        assert "DISCLAIMER" in text

    def test_full_pipeline_normal_finding(self):
        """All models predict Normal → clear severity."""
        evidence = [
            make_realistic_evidence("densenet", "Normal", 0.95),
            make_realistic_evidence("attention", "Normal", 0.92),
            make_realistic_evidence("swin", "Normal", 0.89),
        ]

        engine = CouncilEngine()
        result = engine.run(evidence, modality="MRI", anatomy="brain")

        assert result.primary_finding == "Normal"
        assert result.severity == SeverityLevel.CLEAR

        report = generate_report(result)
        assert "Normal" in report.impression

    def test_pipeline_with_failed_model(self):
        """One model fails → pipeline still works with reduced council."""
        evidence = [
            make_realistic_evidence("densenet", "Meningioma", 0.90),
            make_realistic_evidence("attention", "Meningioma", 0.85),
            make_realistic_evidence("swin", "", 0.0, status=ModelStatus.ERROR),
        ]

        engine = CouncilEngine()
        result = engine.run(evidence, modality="MRI", anatomy="brain")

        # Should still produce a valid result with 2 working models
        assert result.primary_finding == "Meningioma"
        assert result.models_agreeing == 2

        report = generate_report(result)
        # Should mention the failed model in limitations
        has_failure_note = any("failed" in lim.lower() for lim in report.limitations)
        assert has_failure_note


class TestCouncilToAgent:
    """Tests: Council Result → Agent evidence formatting → knowledge retrieval"""

    def test_evidence_formatting(self):
        """Council result formats correctly for agent consumption."""
        evidence = [
            make_realistic_evidence("densenet", "Glioblastoma", 0.90),
            make_realistic_evidence("attention", "Glioblastoma", 0.85),
        ]

        engine = CouncilEngine()
        result = engine.run(evidence, modality="MRI", anatomy="brain")

        formatted = format_council_evidence(result)

        assert "COUNCIL ANALYSIS EVIDENCE" in formatted
        assert "Glioblastoma" in formatted
        assert "CONSENSUS" in formatted
        assert "MODEL EVIDENCE" in formatted
        assert "TestModel-densenet" in formatted
        assert "TestModel-attention" in formatted

    def test_system_prompt_has_safety_rules(self):
        """System prompt enforces safety constraints."""
        prompt = build_system_prompt()

        assert "CRITICAL RULES" in prompt
        assert "NOT making a diagnosis" in prompt
        assert "clinician review" in prompt
        assert "SAFETY" in prompt

    def test_agent_response_parsing_valid_json(self):
        """Parser handles well-formed JSON."""
        valid = json.dumps({
            "key_insights": ["High-grade glioma suspected"],
            "clinical_significance": ["Urgent evaluation needed"],
            "recommended_actions": ["Contrast-enhanced MRI"],
            "missing_information": ["Clinical history"],
            "safety_flags": [],
            "confidence": 0.8,
        })

        parsed = _parse_agent_response(valid)
        assert parsed["confidence"] == 0.8
        assert "High-grade glioma" in parsed["key_insights"][0]

    def test_agent_response_parsing_markdown_wrapped(self):
        """Parser extracts JSON from markdown code blocks."""
        wrapped = '```json\n{"key_insights": ["test"], "confidence": 0.5}\n```'

        parsed = _parse_agent_response(wrapped)
        assert parsed["confidence"] == 0.5

    def test_agent_response_parsing_garbage(self):
        """Parser handles non-JSON gracefully."""
        garbage = "I think the patient might have a tumor..."

        parsed = _parse_agent_response(garbage)
        assert "safety_flags" in parsed
        assert len(parsed["safety_flags"]) > 0  # Should flag non-structured output

    def test_knowledge_retrieval_baseline(self):
        """Knowledge retrieval returns baseline context for known findings."""
        evidence = [make_realistic_evidence("test", "Glioblastoma", 0.9)]
        engine = CouncilEngine()
        result = engine.run(evidence, modality="MRI", anatomy="brain")

        context = _retrieve_knowledge_context(result)

        assert len(context) > 0
        assert "WHO Grade IV" in context or "KNOWLEDGE" in context

    def test_knowledge_retrieval_unknown_finding(self):
        """Knowledge retrieval returns empty for unknown findings."""
        evidence = [make_realistic_evidence("test", "UnknownConditionXYZ", 0.9)]
        engine = CouncilEngine()
        result = engine.run(evidence)

        context = _retrieve_knowledge_context(result)
        assert context == ""


class TestReportSerialization:
    """Tests: Report → serialization → reconstruction"""

    def test_report_to_dict_roundtrip(self):
        """Report can be serialized to dict and contains all fields."""
        evidence = [
            make_realistic_evidence("densenet", "Meningioma", 0.88),
            make_realistic_evidence("attention", "Meningioma", 0.84),
        ]

        engine = CouncilEngine()
        result = engine.run(evidence, modality="MRI", anatomy="brain")
        report = generate_report(result)

        d = report.to_dict()

        assert d["report_id"] == report.report_id
        assert d["impression"] == report.impression
        assert len(d["sections"]) == len(report.sections)
        assert d["recommendations"] == report.recommendations
        assert d["limitations"] == report.limitations

    def test_report_text_contains_all_sections(self):
        """Text rendering includes all section titles."""
        evidence = [
            make_realistic_evidence("densenet", "Pituitary", 0.82),
        ]

        engine = CouncilEngine()
        result = engine.run(evidence, modality="MRI", anatomy="brain")
        report = generate_report(result)

        text = report.to_text()
        for section in report.sections:
            assert section.title.upper() in text.upper()


class TestKnowledgeIntegration:
    """Tests: Knowledge base files → chunking → baseline retrieval"""

    def test_brain_mri_knowledge_file_chunks(self):
        """Brain MRI knowledge file can be chunked."""
        knowledge_path = Path("data/knowledge/brain_mri.md")
        if not knowledge_path.exists():
            pytest.skip("Knowledge base files not found")

        text = knowledge_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source=str(knowledge_path))

        assert len(chunks) > 3  # Should produce multiple chunks
        # At least one chunk should mention glioblastoma
        has_gbm = any("glioblastoma" in c.content.lower() for c in chunks)
        assert has_gbm

    def test_chest_knowledge_file_chunks(self):
        """Chest imaging knowledge file can be chunked."""
        knowledge_path = Path("data/knowledge/chest_imaging.md")
        if not knowledge_path.exists():
            pytest.skip("Knowledge base files not found")

        text = knowledge_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source=str(knowledge_path))

        assert len(chunks) > 3
        has_pneumonia = any("pneumonia" in c.content.lower() for c in chunks)
        assert has_pneumonia

    def test_baseline_knowledge_covers_common_findings(self):
        """Baseline knowledge has entries for all common brain tumor types."""
        expected = ["glioblastoma", "meningioma", "pituitary_adenoma", "normal_brain"]
        for finding in expected:
            assert finding in BASELINE_KNOWLEDGE, f"Missing baseline: {finding}"
            assert len(BASELINE_KNOWLEDGE[finding]) > 100

    def test_baseline_context_retrieval_chain(self):
        """Full chain: finding name → baseline lookup → context string."""
        for finding in ["Glioblastoma", "Meningioma", "Normal"]:
            context = get_baseline_context(finding)
            assert len(context) > 0, f"No baseline context for {finding}"
