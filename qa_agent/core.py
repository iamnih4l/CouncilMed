"""
CouncilMed QA Agent — Core Data Structures & Reporting Framework
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TestStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class TestCaseResult:
    test_id: str
    name: str
    category: str
    status: TestStatus
    execution_time_ms: float
    details: str
    evidence: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class ModalityAuditResult:
    modality: str
    anatomy: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    status: TestStatus = TestStatus.PASS
    models_tested: list[str] = field(default_factory=list)
    dataset_evaluated: str = ""
    evaluation_samples: int = 0
    overall_accuracy: float = 0.0
    f1_score: float = 0.0
    average_confidence: float = 0.0
    average_uncertainty: float = 0.0
    avg_inference_latency_ms: float = 0.0
    test_results: list[TestCaseResult] = field(default_factory=list)
    confusion_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalAuditResult:
    status: TestStatus = TestStatus.PASS
    total_scenarios_tested: int = 0
    fusion_success_rate: float = 0.0
    disagreement_handling: str = "VERIFIED"
    entropy_uncertainty_score: float = 0.0
    test_results: list[TestCaseResult] = field(default_factory=list)


@dataclass
class ReportAuditResult:
    status: TestStatus = TestStatus.PASS
    total_reports_evaluated: int = 0
    clinical_consistency_score: float = 0.0
    evidence_traceability_rate: float = 0.0
    disclaimer_presence_rate: float = 0.0
    unsupported_claims_detected: int = 0
    test_results: list[TestCaseResult] = field(default_factory=list)


@dataclass
class OllamaAuditResult:
    status: TestStatus = TestStatus.PASS
    service_active: bool = False
    active_model: Optional[str] = None
    available_models: list[str] = field(default_factory=list)
    avg_latency_s: float = 0.0
    tokens_per_second: float = 0.0
    fallback_operational: bool = True
    test_results: list[TestCaseResult] = field(default_factory=list)


@dataclass
class PrivacyAuditResult:
    status: TestStatus = TestStatus.PASS
    external_calls_detected: int = 0
    telemetry_present: bool = False
    phi_leaks_found: int = 0
    local_first_verified: bool = True
    test_results: list[TestCaseResult] = field(default_factory=list)


@dataclass
class E2EAuditResult:
    status: TestStatus = TestStatus.PASS
    total_workflows_tested: int = 0
    passed_workflows: int = 0
    failed_workflows: int = 0
    avg_pipeline_duration_ms: float = 0.0
    test_results: list[TestCaseResult] = field(default_factory=list)


@dataclass
class FullAuditReport:
    audit_id: str
    timestamp: str
    system_info: dict[str, Any]
    modality_results: dict[str, ModalityAuditResult]
    multimodal_result: MultimodalAuditResult
    report_audit_result: ReportAuditResult
    ollama_audit_result: OllamaAuditResult
    privacy_audit_result: PrivacyAuditResult
    e2e_audit_result: E2EAuditResult
    detected_issues: list[dict[str, Any]]
    fixes_applied: list[dict[str, Any]]
    final_readiness: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
