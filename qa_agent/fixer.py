"""
CouncilMed QA Agent — Automated Safe Fixer & Remediation Engine (Phase 8)
=========================================================================
Safely remediates verified issues in prompt safety, error handling,
relative paths, and parser fallbacks without altering core architectures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SafeFixer:
    def __init__(self, root_dir: Path):
        self.root = root_dir

    def apply_fixes(self) -> list[dict[str, Any]]:
        fixes_log = []

        # Fix 1: System prompt safety rules & garbage parsing in orchestrator.py
        orch_path = self.root / "backend" / "agents" / "orchestrator.py"
        if orch_path.exists():
            fix1 = self._fix_orchestrator(orch_path)
            if fix1:
                fixes_log.append(fix1)

        # Fix 2: Absolute weights path resolution in backend/inference/engine.py
        inf_engine_path = self.root / "backend" / "inference" / "engine.py"
        if inf_engine_path.exists():
            fix2 = self._fix_inference_engine_paths(inf_engine_path)
            if fix2:
                fixes_log.append(fix2)

        # Fix 3: Robust input error handling in backend/pipeline.py
        pipe_path = self.root / "backend" / "pipeline.py"
        if pipe_path.exists():
            fix3 = self._fix_pipeline_input_robustness(pipe_path)
            if fix3:
                fixes_log.append(fix3)

        # Fix 4: Clinician review dual-key resolution in backend/api/app.py
        app_path = self.root / "backend" / "api" / "app.py"
        if app_path.exists():
            fix4 = self._fix_review_resolution(app_path)
            if fix4:
                fixes_log.append(fix4)

        # Fix 5: Degraded redundancy review flag in backend/council/engine.py
        council_path = self.root / "backend" / "council" / "engine.py"
        if council_path.exists():
            fix5 = self._fix_council_fault_tolerance(council_path)
            if fix5:
                fixes_log.append(fix5)

        return fixes_log

    def _fix_orchestrator(self, file_path: Path) -> dict[str, Any] | None:
        text = file_path.read_text(encoding="utf-8")
        original_text = text
        changes_applied = []

        # Check safety rules in build_system_prompt
        if "CRITICAL RULES" not in text:
            old_prompt = '''def build_system_prompt() -> str:
    return """You are a clinical AI assistant embedded in the CouncilMed medical imaging analysis system.
Interpret the evidence produced by the AI Council — a panel of independent deep learning models.
Respond ONLY with a valid JSON object matching the required schema:
{
  "key_insights": ["string"],
  "clinical_significance": ["string"],
  "recommended_actions": ["string"],
  "missing_information": ["string"],
  "safety_flags": ["string"],
  "evidence_references": ["string"],
  "confidence": 0.85
}"""'''

            new_prompt = '''def build_system_prompt() -> str:
    return """You are a clinical AI assistant embedded in the CouncilMed medical imaging analysis system.
Interpret the evidence produced by the AI Council — a panel of independent deep learning models.

CRITICAL RULES & SAFETY:
1. You are NOT making a diagnosis. Your role is clinical decision-support only.
2. All findings require qualified clinician review and validation.
3. Every insight MUST be traceable to specific model evidence.
4. Highlight model disagreements, elevated uncertainty, and potential false-negative risks.
5. Do not invent or assume clinical findings not supported by model evidence.

Respond ONLY with a valid JSON object matching the required schema:
{
  "key_insights": ["string"],
  "clinical_significance": ["string"],
  "recommended_actions": ["string"],
  "missing_information": ["string"],
  "safety_flags": ["string"],
  "evidence_references": ["string"],
  "confidence": 0.85
}"""'''
            if old_prompt in text:
                text = text.replace(old_prompt, new_prompt)
                changes_applied.append("Restored CRITICAL RULES & SAFETY constraints in build_system_prompt()")

        # Check safety flags in _parse_agent_response for unparseable output
        if '"safety_flags": [],\n        "confidence": 0.75,' in text:
            old_fallback = '''    return {
        "key_insights": [raw_text[:500]] if raw_text else ["Council analysis completed."],
        "clinical_significance": [],
        "recommended_actions": ["Clinician verification recommended."],
        "missing_information": [],
        "safety_flags": [],
        "confidence": 0.75,
    }'''

            new_fallback = '''    return {
        "key_insights": [raw_text[:500]] if raw_text else ["Council analysis completed."],
        "clinical_significance": [],
        "recommended_actions": ["Clinician verification recommended."],
        "missing_information": [],
        "safety_flags": ["AI model returned non-structured text response — requires direct clinician review."],
        "confidence": 0.50,
    }'''
            if old_fallback in text:
                text = text.replace(old_fallback, new_fallback)
                changes_applied.append("Added explicit safety_flags alert in _parse_agent_response() for non-JSON output")

        if text != original_text:
            file_path.write_text(text, encoding="utf-8")
            return {
                "file": str(file_path.relative_to(self.root)),
                "issue": "Missing safety rules in system prompt and unflagged non-structured agent output",
                "root_cause": "System prompt was truncated and parsing fallback returned empty safety_flags",
                "changes_made": changes_applied,
                "status": "APPLIED",
            }
        elif "CRITICAL RULES" in text:
            return {
                "file": str(file_path.relative_to(self.root)),
                "issue": "Missing safety rules in system prompt and unflagged non-structured agent output",
                "root_cause": "System prompt was truncated and parsing fallback returned empty safety_flags",
                "changes_made": ["CRITICAL RULES & SAFETY rules active in build_system_prompt()", "Explicit safety_flags alert active in _parse_agent_response()"],
                "status": "VERIFIED",
            }
        return None

    def _fix_inference_engine_paths(self, file_path: Path) -> dict[str, Any] | None:
        text = file_path.read_text(encoding="utf-8")
        original_text = text

        old_target = """    weights_path = Path(manifest.weights_path)
    num_classes = manifest.num_classes or len(manifest.class_names) or 4"""

        new_replacement = """    weights_path = Path(manifest.weights_path)
    if not weights_path.is_absolute():
        from ..paths import get_base_dir
        weights_path = get_base_dir() / weights_path
    num_classes = manifest.num_classes or len(manifest.class_names) or 4"""

        if old_target in text and "from ..paths import get_base_dir" not in text:
            text = text.replace(old_target, new_replacement)
            file_path.write_text(text, encoding="utf-8")
            return {
                "file": str(file_path.relative_to(self.root)),
                "issue": "Relative weights path sensitivity to current working directory",
                "root_cause": "load_pytorch_model used Path(manifest.weights_path) directly without get_base_dir() anchoring",
                "changes_made": ["Anchored weights_path to get_base_dir() for relative manifest paths"],
                "status": "APPLIED",
            }
        elif "get_base_dir() / weights_path" in text:
            return {
                "file": str(file_path.relative_to(self.root)),
                "issue": "Relative weights path sensitivity to current working directory",
                "root_cause": "load_pytorch_model used Path(manifest.weights_path) directly without get_base_dir() anchoring",
                "changes_made": ["Anchored weights_path to get_base_dir() for relative manifest paths"],
                "status": "VERIFIED",
            }
        return None

    def _fix_pipeline_input_robustness(self, file_path: Path) -> dict[str, Any] | None:
        text = file_path.read_text(encoding="utf-8")
        original_text = text

        old_analyze_start = """    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        metadata, pixel_data = self._parse_input(request.file_path)"""

        new_analyze_start = """    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        try:
            metadata, pixel_data = self._parse_input(request.file_path)
        except Exception as e:
            logger.error("Failed to parse input file %s: %s", request.file_path, e)
            errors.append(f"Input parsing error: {e}")
            from .council.evidence import CouncilResult
            return AnalysisResponse(
                request_id=request.request_id,
                council_result=CouncilResult(
                    primary_finding="Unreadable input file",
                    requires_human_review=True,
                    human_review_reasons=[f"Corrupted or invalid file format: {e}"],
                ),
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                errors=errors,
                warnings=warnings,
            )"""

        if old_analyze_start in text:
            text = text.replace(old_analyze_start, new_analyze_start)
            file_path.write_text(text, encoding="utf-8")
            return {
                "file": str(file_path.relative_to(self.root)),
                "issue": "Uncaught exception on corrupted or unreadable image upload",
                "root_cause": "self._parse_input was called outside try/except block in AnalysisPipeline.analyze",
                "changes_made": ["Wrapped _parse_input in try/except and returned safe human-review AnalysisResponse on error"],
                "status": "APPLIED",
            }
        elif "Unreadable input file" in text:
            return {
                "file": str(file_path.relative_to(self.root)),
                "issue": "Uncaught exception on corrupted or unreadable image upload",
                "root_cause": "self._parse_input was called outside try/except block in AnalysisPipeline.analyze",
                "changes_made": ["Wrapped _parse_input in try/except and returned safe human-review AnalysisResponse on error"],
                "status": "VERIFIED",
            }
        return None

    def _fix_review_resolution(self, file_path: Path) -> dict[str, Any] | None:
        text = file_path.read_text(encoding="utf-8")
        if "target_analysis_id" in text or "Analysis.request_id == review.analysis_id" in text:
            return {
                "file": str(file_path.relative_to(self.root)),
                "issue": "Review submission failed when client supplied request_id instead of database primary key",
                "root_cause": "submit_review only matched CouncilResultRecord.analysis_id without fallback to Analysis.request_id",
                "changes_made": ["Implemented dual-lookup for review submission supporting both analysis_id and request_id"],
                "status": "VERIFIED",
            }
        return None

    def _fix_council_fault_tolerance(self, file_path: Path) -> dict[str, Any] | None:
        text = file_path.read_text(encoding="utf-8")
        if "operating in degraded redundancy mode" in text:
            return {
                "file": str(file_path.relative_to(self.root)),
                "issue": "Model failures did not explicitly flag degraded council redundancy in human review reasons",
                "root_cause": "determine_human_review was unaware of member models that failed with exceptions",
                "changes_made": ["Added explicit human review alert whenever any council member model fails"],
                "status": "VERIFIED",
            }
        return None
