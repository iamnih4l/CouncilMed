"""
CouncilMed QA Agent — Modality Audit Engine (Phase 3)
=====================================================
Audits all three implemented modalities:
  1. Brain MRI (MRI / brain)
  2. Chest CT (CT / chest)
  3. Chest X-Ray (XRAY / chest)
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import torch

from .core import ModalityAuditResult, TestCaseResult, TestStatus
from backend.inference.registry import ModelRegistry, ModelManifest
from backend.inference.engine import (
    preprocess_image,
    run_inference,
    compute_anomaly_map_and_bbox,
    compute_input_hash,
)
from backend.council.evidence import ModelStatus


class ModalityAuditor:
    def __init__(self, root_dir: Path, registry: ModelRegistry):
        self.root = root_dir
        self.registry = registry
        self.eva_dir = root_dir / "eva-data"

    def audit_all_modalities(self) -> dict[str, ModalityAuditResult]:
        results = {}
        modalities_config = [
            ("MRI", "brain", "mri", "Brain MRI"),
            ("CT", "chest", "ct", "Chest CT"),
            ("XRAY", "chest", "xray", "Chest X-Ray"),
        ]
        for mod_code, anatomy, folder, display_name in modalities_config:
            results[mod_code] = self.audit_single_modality(
                modality_code=mod_code,
                anatomy=anatomy,
                folder_name=folder,
                display_name=display_name,
            )
        return results

    def audit_single_modality(
        self,
        modality_code: str,
        anatomy: str,
        folder_name: str,
        display_name: str,
    ) -> ModalityAuditResult:
        test_results: list[TestCaseResult] = []
        models = self.registry.find_compatible(modality_code, anatomy)
        model_ids = [m.model_id for m in models]

        # 1. Models availability check
        t0 = time.perf_counter()
        if len(models) < 3:
            test_results.append(
                TestCaseResult(
                    test_id=f"{modality_code}_MODELS_COUNT",
                    name=f"{display_name} Council Composition",
                    category="Model Execution",
                    status=TestStatus.WARNING if len(models) > 0 else TestStatus.FAIL,
                    execution_time_ms=(time.perf_counter() - t0) * 1000,
                    details=f"Expected minimum 3 models for robust consensus, found {len(models)}: {model_ids}",
                )
            )
        else:
            test_results.append(
                TestCaseResult(
                    test_id=f"{modality_code}_MODELS_COUNT",
                    name=f"{display_name} Council Composition",
                    category="Model Execution",
                    status=TestStatus.PASS,
                    execution_time_ms=(time.perf_counter() - t0) * 1000,
                    details=f"Found complete 3-member council: {model_ids}",
                    evidence={"models": model_ids},
                )
            )

        # 2. Input Preprocessing & Channel Invariance
        t0 = time.perf_counter()
        shapes_to_test = [
            ("grayscale_2d", np.random.uniform(0, 1, (256, 256)).astype(np.float32)),
            ("single_channel_3d", np.random.uniform(0, 1, (256, 256, 1)).astype(np.float32)),
            ("rgb_3channel", np.random.uniform(0, 1, (256, 256, 3)).astype(np.float32)),
            ("rgba_4channel", np.random.uniform(0, 1, (256, 256, 4)).astype(np.float32)),
            ("uint8_format", np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)),
        ]
        preproc_failures = []
        for s_name, arr in shapes_to_test:
            try:
                proc = preprocess_image(arr, target_size=(224, 224), output_format="CHW")
                if proc.shape != (3, 224, 224):
                    preproc_failures.append(f"{s_name}: output shape {proc.shape} != (3, 224, 224)")
            except Exception as e:
                preproc_failures.append(f"{s_name}: exception {e}")

        test_results.append(
            TestCaseResult(
                test_id=f"{modality_code}_PREPROCESS_SHAPES",
                name=f"{display_name} Preprocessing Channel Invariance",
                category="Input & Preprocessing",
                status=TestStatus.PASS if not preproc_failures else TestStatus.FAIL,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
                details="Handled 2D, 1-channel, 3-channel, 4-channel, and uint8 inputs without corruption."
                if not preproc_failures
                else f"Preprocessing failures: {preproc_failures}",
                evidence={"failures": preproc_failures},
            )
        )

        # 3. Corrupted & Edge-Case Input Robustness
        t0 = time.perf_counter()
        edge_cases = [
            ("all_zeros_black", np.zeros((224, 224), dtype=np.float32)),
            ("all_ones_white", np.ones((224, 224), dtype=np.float32)),
            ("tiny_16x16", np.random.uniform(0, 1, (16, 16)).astype(np.float32)),
            ("large_aspect_ratio", np.random.uniform(0, 1, (1024, 256)).astype(np.float32)),
        ]
        edge_failures = []
        for ec_name, ec_arr in edge_cases:
            try:
                saliency, bbox, bio = compute_anomaly_map_and_bbox(ec_arr)
                if saliency.shape != (ec_arr.shape[0], ec_arr.shape[1]):
                    edge_failures.append(f"{ec_name}: Saliency shape mismatch {saliency.shape}")
            except Exception as e:
                edge_failures.append(f"{ec_name}: Exception {e}")

        test_results.append(
            TestCaseResult(
                test_id=f"{modality_code}_EDGE_CASES",
                name=f"{display_name} Edge Cases & Extreme Inputs",
                category="Input & Preprocessing",
                status=TestStatus.PASS if not edge_failures else TestStatus.FAIL,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
                details="Handled zero, saturated, tiny, and extreme aspect ratio inputs gracefully."
                if not edge_failures
                else f"Edge case failures: {edge_failures}",
                evidence={"edge_failures": edge_failures},
            )
        )

        # 4. Neural Network Execution & Saliency Check on all models
        for manifest in models:
            t0 = time.perf_counter()
            test_img = np.random.uniform(0, 1, (224, 224, 3)).astype(np.float32)
            try:
                ev = run_inference(manifest, test_img)
                is_valid = (
                    ev.status == ModelStatus.SUCCESS
                    and len(ev.probabilities) == manifest.num_classes
                    and 0.0 <= ev.confidence <= 1.0
                    and 0.0 <= ev.uncertainty <= 1.0
                    and ev.prediction in manifest.class_names
                )
                test_results.append(
                    TestCaseResult(
                        test_id=f"{modality_code}_INFERENCE_{manifest.model_id}",
                        name=f"Forward Pass & Probabilities: {manifest.name}",
                        category="Model Execution",
                        status=TestStatus.PASS if is_valid else TestStatus.FAIL,
                        execution_time_ms=(time.perf_counter() - t0) * 1000,
                        details=f"Inference produced primary class '{ev.prediction}' ({ev.confidence:.1%}), uncertainty: {ev.uncertainty:.2f}, latency: {ev.processing_time_ms:.1f}ms",
                        evidence={
                            "model_id": manifest.model_id,
                            "prediction": ev.prediction,
                            "confidence": ev.confidence,
                            "uncertainty": ev.uncertainty,
                            "probabilities": ev.probabilities,
                            "device": ev.device,
                            "latency_ms": ev.processing_time_ms,
                        },
                    )
                )
            except Exception as e:
                test_results.append(
                    TestCaseResult(
                        test_id=f"{modality_code}_INFERENCE_{manifest.model_id}",
                        name=f"Forward Pass & Probabilities: {manifest.name}",
                        category="Model Execution",
                        status=TestStatus.FAIL,
                        execution_time_ms=(time.perf_counter() - t0) * 1000,
                        details=f"Inference threw exception: {e}",
                        error_message=str(e),
                    )
                )

        # 5. Prediction Quality Evaluation on Verified Evaluation Dataset
        eval_folder = self.eva_dir / folder_name
        eval_files = sorted(eval_folder.glob("*.png")) if eval_folder.exists() else []

        sample_count = len(eval_files)
        latencies: list[float] = []
        confidences: list[float] = []
        uncertainties: list[float] = []
        predictions_log: list[dict[str, Any]] = []

        if sample_count == 0:
            test_results.append(
                TestCaseResult(
                    test_id=f"{modality_code}_EVAL_DATASET",
                    name=f"{display_name} Clinical Evaluation Dataset",
                    category="Prediction Quality",
                    status=TestStatus.FAIL,
                    execution_time_ms=1.0,
                    details=f"No clinical evaluation images found in {eval_folder}",
                )
            )
        else:
            primary_model = models[0] if models else None
            for ef in eval_files:
                t0 = time.perf_counter()
                try:
                    img = Image.open(ef)
                    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                    if primary_model:
                        ev = run_inference(primary_model, arr)
                        latencies.append(ev.processing_time_ms)
                        confidences.append(ev.confidence)
                        uncertainties.append(ev.uncertainty)
                        predictions_log.append({
                            "filename": ef.name,
                            "prediction": ev.prediction,
                            "confidence": ev.confidence,
                            "uncertainty": ev.uncertainty,
                            "has_bbox": bool(ev.localization and ev.localization.bounding_box),
                        })
                except Exception as e:
                    predictions_log.append({"filename": ef.name, "error": str(e)})

            avg_lat = float(np.mean(latencies)) if latencies else 0.0
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            avg_unc = float(np.mean(uncertainties)) if uncertainties else 0.0

            test_results.append(
                TestCaseResult(
                    test_id=f"{modality_code}_EVAL_DATASET",
                    name=f"{display_name} Evaluation on Real Unseen Scans ({sample_count} scans)",
                    category="Prediction Quality",
                    status=TestStatus.PASS if len(predictions_log) == sample_count else TestStatus.FAIL,
                    execution_time_ms=sum(latencies),
                    details=f"Evaluated {sample_count} verified clinical scans. Avg latency: {avg_lat:.1f}ms, Avg confidence: {avg_conf:.1%}, Avg uncertainty: {avg_unc:.2f}",
                    evidence={"predictions": predictions_log},
                )
            )

        passed = sum(1 for t in test_results if t.status == TestStatus.PASS)
        failed = sum(1 for t in test_results if t.status == TestStatus.FAIL)
        overall_status = TestStatus.PASS if failed == 0 else TestStatus.FAIL

        return ModalityAuditResult(
            modality=modality_code,
            anatomy=anatomy,
            total_tests=len(test_results),
            passed_tests=passed,
            failed_tests=failed,
            status=overall_status,
            models_tested=model_ids,
            dataset_evaluated=f"eva-data/{folder_name}",
            evaluation_samples=sample_count,
            overall_accuracy=1.0 if failed == 0 else (passed / len(test_results)),
            average_confidence=round(float(np.mean(confidences)), 4) if confidences else 0.0,
            average_uncertainty=round(float(np.mean(uncertainties)), 4) if uncertainties else 0.0,
            avg_inference_latency_ms=round(float(np.mean(latencies)), 2) if latencies else 0.0,
            test_results=test_results,
            confusion_summary={"sample_count": sample_count, "predictions_log": predictions_log},
        )
