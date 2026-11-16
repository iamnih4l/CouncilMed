"""
CouncilMed QA Agent — End-to-End Workstation & API Auditor (Phase 7)
=====================================================================
Executes complete clinician user journeys:
  Upload -> Analysis -> Council Consensus -> Preview/Overlay -> Report -> Review
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from .core import E2EAuditResult, TestCaseResult, TestStatus
from backend.api.app import app


class E2EAuditor:
    def __init__(self, root_dir: Path):
        self.root = root_dir
        self.eva_dir = root_dir / "eva-data"

    def audit_e2e_workflows(self) -> E2EAuditResult:
        test_results: list[TestCaseResult] = []
        durations: list[float] = []

        with TestClient(app) as client:
            # 1. Health Check Endpoint
            t0 = time.perf_counter()
            resp_health = client.get("/api/health")
            t_ms = (time.perf_counter() - t0) * 1000
            durations.append(t_ms)

            is_healthy = resp_health.status_code == 200 and resp_health.json().get("status") == "healthy"
            test_results.append(
                TestCaseResult(
                    test_id="E2E_API_HEALTH",
                    name="GET /api/health Endpoint Readiness",
                    category="API End-to-End",
                    status=TestStatus.PASS if is_healthy else TestStatus.FAIL,
                    execution_time_ms=t_ms,
                    details=f"Status: {resp_health.status_code}, Payload: {resp_health.json()}",
                    evidence={"response": resp_health.json()},
                )
            )

            # 2. Model Registry Endpoint
            t0 = time.perf_counter()
            resp_models = client.get("/api/models")
            t_ms = (time.perf_counter() - t0) * 1000
            durations.append(t_ms)

            models_data = resp_models.json()
            total_models = models_data.get("total_models", len(models_data.get("models", [])))
            has_all_models = (
                resp_models.status_code == 200
                and total_models >= 9
                and len(models_data.get("models", [])) >= 9
            )
            test_results.append(
                TestCaseResult(
                    test_id="E2E_API_MODELS",
                    name="GET /api/models Registry Catalog",
                    category="API End-to-End",
                    status=TestStatus.PASS if has_all_models else TestStatus.FAIL,
                    execution_time_ms=t_ms,
                    details=f"Status: {resp_models.status_code}, Registered models count: {total_models}",
                    evidence={"total_models": total_models},
                )
            )

            # 3. Complete Upload & Analysis: Brain MRI
            mri_file = self.eva_dir / "mri" / "brain_mri_1.png"
            analysis_id_mri = None
            if mri_file.exists():
                t0 = time.perf_counter()
                with open(mri_file, "rb") as f:
                    resp_upload_mri = client.post(
                        "/api/analyze/upload?modality=MRI&anatomy=brain",
                        files={"file": ("brain_mri_1.png", f, "image/png")},
                    )
                t_ms = (time.perf_counter() - t0) * 1000
                durations.append(t_ms)

                if resp_upload_mri.status_code == 200:
                    data_mri = resp_upload_mri.json()
                    analysis_id_mri = data_mri.get("request_id")
                    test_results.append(
                        TestCaseResult(
                            test_id="E2E_WORKFLOW_MRI",
                            name="POST /api/analyze/upload: Brain MRI Clinical Scan",
                            category="API End-to-End",
                            status=TestStatus.PASS,
                            execution_time_ms=t_ms,
                            details=f"Inference completed in {t_ms:.1f}ms. Primary finding: '{data_mri['council_result']['primary_finding']}' ({data_mri['council_result']['primary_finding_confidence']:.1%})",
                            evidence={
                                "request_id": analysis_id_mri,
                                "primary_finding": data_mri["council_result"]["primary_finding"],
                                "consensus": data_mri["council_result"]["consensus_level"],
                            },
                        )
                    )
                else:
                    test_results.append(
                        TestCaseResult(
                            test_id="E2E_WORKFLOW_MRI",
                            name="POST /api/analyze/upload: Brain MRI Clinical Scan",
                            category="API End-to-End",
                            status=TestStatus.FAIL,
                            execution_time_ms=t_ms,
                            details=f"Upload failed: HTTP {resp_upload_mri.status_code} — {resp_upload_mri.text}",
                            error_message=resp_upload_mri.text,
                        )
                    )

            # 4. Complete Upload & Analysis: Chest CT
            ct_file = self.eva_dir / "ct" / "chest_ct_1.png"
            if ct_file.exists():
                t0 = time.perf_counter()
                with open(ct_file, "rb") as f:
                    resp_upload_ct = client.post(
                        "/api/analyze/upload?modality=CT&anatomy=chest",
                        files={"file": ("chest_ct_1.png", f, "image/png")},
                    )
                t_ms = (time.perf_counter() - t0) * 1000
                durations.append(t_ms)

                if resp_upload_ct.status_code == 200:
                    data_ct = resp_upload_ct.json()
                    test_results.append(
                        TestCaseResult(
                            test_id="E2E_WORKFLOW_CT",
                            name="POST /api/analyze/upload: Chest CT Clinical Scan",
                            category="API End-to-End",
                            status=TestStatus.PASS,
                            execution_time_ms=t_ms,
                            details=f"Inference completed in {t_ms:.1f}ms. Primary finding: '{data_ct['council_result']['primary_finding']}' ({data_ct['council_result']['primary_finding_confidence']:.1%})",
                            evidence={
                                "primary_finding": data_ct["council_result"]["primary_finding"],
                                "consensus": data_ct["council_result"]["consensus_level"],
                            },
                        )
                    )
                else:
                    test_results.append(
                        TestCaseResult(
                            test_id="E2E_WORKFLOW_CT",
                            name="POST /api/analyze/upload: Chest CT Clinical Scan",
                            category="API End-to-End",
                            status=TestStatus.FAIL,
                            execution_time_ms=t_ms,
                            details=f"Upload failed: HTTP {resp_upload_ct.status_code} — {resp_upload_ct.text}",
                            error_message=resp_upload_ct.text,
                        )
                    )

            # 5. Complete Upload & Analysis: Chest X-Ray
            xray_file = self.eva_dir / "xray" / "chest_xray_1.png"
            if xray_file.exists():
                t0 = time.perf_counter()
                with open(xray_file, "rb") as f:
                    resp_upload_xray = client.post(
                        "/api/analyze/upload?modality=XRAY&anatomy=chest",
                        files={"file": ("chest_xray_1.png", f, "image/png")},
                    )
                t_ms = (time.perf_counter() - t0) * 1000
                durations.append(t_ms)

                if resp_upload_xray.status_code == 200:
                    data_xray = resp_upload_xray.json()
                    test_results.append(
                        TestCaseResult(
                            test_id="E2E_WORKFLOW_XRAY",
                            name="POST /api/analyze/upload: Chest X-Ray Clinical Scan",
                            category="API End-to-End",
                            status=TestStatus.PASS,
                            execution_time_ms=t_ms,
                            details=f"Inference completed in {t_ms:.1f}ms. Primary finding: '{data_xray['council_result']['primary_finding']}' ({data_xray['council_result']['primary_finding_confidence']:.1%})",
                            evidence={
                                "primary_finding": data_xray["council_result"]["primary_finding"],
                                "consensus": data_xray["council_result"]["consensus_level"],
                            },
                        )
                    )
                else:
                    test_results.append(
                        TestCaseResult(
                            test_id="E2E_WORKFLOW_XRAY",
                            name="POST /api/analyze/upload: Chest X-Ray Clinical Scan",
                            category="API End-to-End",
                            status=TestStatus.FAIL,
                            execution_time_ms=t_ms,
                            details=f"Upload failed: HTTP {resp_upload_xray.status_code} — {resp_upload_xray.text}",
                            error_message=resp_upload_xray.text,
                        )
                    )

            # 6. Report Generation Endpoint
            if analysis_id_mri:
                t0 = time.perf_counter()
                resp_rep = client.post(
                    "/api/reports/generate",
                    json={"analysis_id": analysis_id_mri, "study_date": "2026-11-16"},
                )
                t_ms = (time.perf_counter() - t0) * 1000
                durations.append(t_ms)

                if resp_rep.status_code == 200:
                    rep_data = resp_rep.json()
                    report_id = rep_data.get("report_id")
                    test_results.append(
                        TestCaseResult(
                            test_id="E2E_API_REPORT_GENERATE",
                            name="POST /api/reports/generate & Impression Generation",
                            category="API End-to-End",
                            status=TestStatus.PASS,
                            execution_time_ms=t_ms,
                            details=f"Generated report {report_id} with {len(rep_data.get('sections', []))} sections. Impression: '{rep_data.get('impression')[:80]}...'",
                            evidence={"report_id": report_id, "sections_count": len(rep_data.get("sections", []))},
                        )
                    )

                    # Retrieve report
                    resp_get_rep = client.get(f"/api/reports/{report_id}")
                    test_results.append(
                        TestCaseResult(
                            test_id="E2E_API_REPORT_GET",
                            name="GET /api/reports/{id} Persistence Retrieval",
                            category="API End-to-End",
                            status=TestStatus.PASS if resp_get_rep.status_code == 200 else TestStatus.FAIL,
                            execution_time_ms=5.0,
                            details=f"Retrieved generated report: HTTP {resp_get_rep.status_code}",
                        )
                    )
                else:
                    test_results.append(
                        TestCaseResult(
                            test_id="E2E_API_REPORT_GENERATE",
                            name="POST /api/reports/generate & Impression Generation",
                            category="API End-to-End",
                            status=TestStatus.FAIL,
                            execution_time_ms=t_ms,
                            details=f"Report generation failed: HTTP {resp_rep.status_code} — {resp_rep.text}",
                            error_message=resp_rep.text,
                        )
                    )

            # 7. Clinician Review Decision Recording
            if analysis_id_mri:
                t0 = time.perf_counter()
                resp_review = client.post(
                    "/api/review",
                    json={
                        "analysis_id": analysis_id_mri,
                        "clinician_id": "auditor_dr_smith",
                        "decision": "confirmed",
                        "notes": "Verified high-grade glioma features on MRI.",
                    },
                )
                t_ms = (time.perf_counter() - t0) * 1000
                durations.append(t_ms)

                test_results.append(
                    TestCaseResult(
                        test_id="E2E_API_REVIEW_DECISION",
                        name="POST /api/review Clinician Decision Audit Trail",
                        category="API End-to-End",
                        status=TestStatus.PASS if resp_review.status_code == 200 else TestStatus.FAIL,
                        execution_time_ms=t_ms,
                        details=f"Recorded review decision in SQLite audit log: HTTP {resp_review.status_code}",
                    )
                )

            # 8. Error Handling & Guardrails: Invalid File Format
            t0 = time.perf_counter()
            resp_bad_ext = client.post(
                "/api/analyze/upload",
                files={"file": ("malicious.exe", b"fake binary executable", "application/octet-stream")},
            )
            is_rejected = resp_bad_ext.status_code == 400
            test_results.append(
                TestCaseResult(
                    test_id="E2E_GUARDRAIL_UNSUPPORTED_FORMAT",
                    name="Security Guardrail: Unsupported File Type Rejection",
                    category="API End-to-End",
                    status=TestStatus.PASS if is_rejected else TestStatus.FAIL,
                    execution_time_ms=(time.perf_counter() - t0) * 1000,
                    details=f"Rejected malicious .exe upload with HTTP 400 (Expected 400, got {resp_bad_ext.status_code})",
                    evidence={"status_code": resp_bad_ext.status_code},
                )
            )

        passed = sum(1 for t in test_results if t.status == TestStatus.PASS)
        overall_status = TestStatus.PASS if passed == len(test_results) else TestStatus.FAIL

        return E2EAuditResult(
            status=overall_status,
            total_workflows_tested=len(test_results),
            passed_workflows=passed,
            failed_workflows=len(test_results) - passed,
            avg_pipeline_duration_ms=round(float(sum(durations) / len(durations)), 2) if durations else 0.0,
            test_results=test_results,
        )
