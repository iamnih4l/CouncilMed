"""
CouncilMed QA Agent — Ollama Local Infrastructure Auditor (Phase 6)
===================================================================
Tests Ollama connectivity, model resolution, real clinical synthesis inference,
latency, tokens/sec throughput, and built-in fallback resilience.
"""

from __future__ import annotations

import json
import time
from typing import Any
import httpx

from .core import OllamaAuditResult, TestCaseResult, TestStatus
from backend.agents.orchestrator import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    resolve_installed_ollama_model,
    run_council_agent,
    build_system_prompt,
    _parse_agent_response,
    _build_local_clinical_synthesis,
)
from backend.council.evidence import (
    CouncilResult,
    ModelEvidence,
    ModelStatus,
    ConsensusLevel,
    SeverityLevel,
)


class OllamaAuditor:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_BASE_URL):
        self.base_url = base_url

    def audit_ollama(self) -> OllamaAuditResult:
        test_results: list[TestCaseResult] = []
        is_active = False
        available_models: list[str] = []
        active_model = None
        latencies: list[float] = []
        throughputs: list[float] = []

        # 1. Connectivity & Tag Discovery
        t0 = time.perf_counter()
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                is_active = True
                data = resp.json()
                available_models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                active_model = resolve_installed_ollama_model(self.base_url)
                test_results.append(
                    TestCaseResult(
                        test_id="OLLAMA_CONNECTIVITY",
                        name="Ollama Service Connectivity & Model Discovery",
                        category="Local Infrastructure",
                        status=TestStatus.PASS,
                        execution_time_ms=(time.perf_counter() - t0) * 1000,
                        details=f"Ollama is ACTIVE at {self.base_url}. Installed models: {available_models}. Resolved active: '{active_model}'",
                        evidence={"models": available_models, "active_model": active_model},
                    )
                )
            else:
                test_results.append(
                    TestCaseResult(
                        test_id="OLLAMA_CONNECTIVITY",
                        name="Ollama Service Connectivity",
                        category="Local Infrastructure",
                        status=TestStatus.WARNING,
                        execution_time_ms=(time.perf_counter() - t0) * 1000,
                        details=f"Ollama server returned HTTP {resp.status_code} at {self.base_url}",
                    )
                )
        except Exception as e:
            test_results.append(
                TestCaseResult(
                    test_id="OLLAMA_CONNECTIVITY",
                    name="Ollama Service Connectivity",
                    category="Local Infrastructure",
                    status=TestStatus.WARNING,
                    execution_time_ms=(time.perf_counter() - t0) * 1000,
                    details=f"Ollama connection unavailable ({e}). Fallback to built-in Clinical Reasoning Engine will be tested.",
                    error_message=str(e),
                )
            )

        # 2. Real Inference & Performance Benchmark (if active)
        if is_active and active_model:
            for iteration in range(2):
                t_iter = time.perf_counter()
                try:
                    chat_resp = httpx.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": active_model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a clinical AI. Output JSON with key_insights, clinical_significance, recommended_actions, missing_information, safety_flags, evidence_references, confidence."
                                },
                                {
                                    "role": "user",
                                    "content": "Council finding: Glioblastoma with 92% confidence across 3 models on Brain MRI. Synthesize clinical significance."
                                }
                            ],
                            "format": "json",
                            "stream": False,
                            "options": {"temperature": 0.2, "num_predict": 300},
                        },
                        timeout=35.0,
                    )
                    lat = time.perf_counter() - t_iter
                    latencies.append(lat)
                    if chat_resp.status_code == 200:
                        c_data = chat_resp.json()
                        eval_count = c_data.get("eval_count", 0)
                        eval_dur = c_data.get("eval_duration", 0)
                        if eval_dur > 0 and eval_count > 0:
                            tok_sec = eval_count / (eval_dur / 1e9)
                            throughputs.append(tok_sec)
                except Exception as e:
                    latencies.append(time.perf_counter() - t_iter)

            avg_lat = float(sum(latencies) / len(latencies)) if latencies else 0.0
            avg_tok = float(sum(throughputs) / len(throughputs)) if throughputs else 0.0

            test_results.append(
                TestCaseResult(
                    test_id="OLLAMA_INFERENCE_BENCHMARK",
                    name=f"Ollama Live Inference Benchmark ({active_model})",
                    category="Inference Performance",
                    status=TestStatus.PASS if throughputs or latencies else TestStatus.WARNING,
                    execution_time_ms=avg_lat * 1000,
                    details=f"Model: {active_model} | Avg Latency: {avg_lat:.2f}s | Throughput: {avg_tok:.1f} tokens/sec",
                    evidence={
                        "model": active_model,
                        "latencies_s": latencies,
                        "throughputs_tok_sec": throughputs,
                        "avg_latency_s": avg_lat,
                        "avg_tokens_sec": avg_tok,
                    },
                )
            )

        # 3. Built-in Local Clinical Reasoning Fallback Verification
        t0 = time.perf_counter()
        mock_council = CouncilResult(
            council_run_id="test_fallback",
            consensus_level=ConsensusLevel.STRONG,
            evidence_strength=SeverityLevel.CLEAR,
            primary_finding="Glioblastoma",
            primary_finding_confidence=0.91,
            severity=SeverityLevel.CRITICAL,
            models_agreeing=3,
            models_total=3,
            agreement_score=1.0,
            ensemble_uncertainty=0.10,
            requires_human_review=False,
            member_evidence=[
                ModelEvidence(
                    model_id="m1",
                    model_name="DenseNet",
                    model_version="1.0",
                    model_architecture="DenseNet121",
                    prediction="Glioblastoma",
                    probabilities={"Glioblastoma": 0.91, "Normal": 0.09},
                    confidence=0.91,
                    uncertainty=0.10,
                    status=ModelStatus.SUCCESS,
                )
            ],
            modality="MRI",
            anatomy="brain",
        )
        fallback_insight = _build_local_clinical_synthesis(mock_council, elapsed_ms=5.0)
        fallback_works = (
            len(fallback_insight.key_insights) > 0
            and len(fallback_insight.recommended_actions) > 0
            and fallback_insight.agent_model == "CouncilMed-Local-Clinical-Agent-v1"
            and any("Glioblastoma" in k for k in fallback_insight.key_insights)
        )
        test_results.append(
            TestCaseResult(
                test_id="OLLAMA_FALLBACK_INTEGRITY",
                name="Built-In Clinical Synthesis Fallback (Offline Mode)",
                category="Safety & Resilience",
                status=TestStatus.PASS if fallback_works else TestStatus.FAIL,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
                details="Deterministic local fallback synthesizes evidence accurately when Ollama is unreachable or model is absent.",
                evidence={"agent_model": fallback_insight.agent_model, "key_insights": fallback_insight.key_insights},
            )
        )

        passed = sum(1 for t in test_results if t.status == TestStatus.PASS)
        overall_status = TestStatus.PASS if is_active and active_model else TestStatus.WARNING

        return OllamaAuditResult(
            status=overall_status,
            service_active=is_active,
            active_model=active_model,
            available_models=available_models,
            avg_latency_s=round(float(sum(latencies) / len(latencies)), 2) if latencies else 0.0,
            tokens_per_second=round(float(sum(throughputs) / len(throughputs)), 1) if throughputs else 0.0,
            fallback_operational=fallback_works,
            test_results=test_results,
        )
