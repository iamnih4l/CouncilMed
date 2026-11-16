"""
CouncilMed QA Agent — System Architecture & Registry Inspector
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any
import yaml
import torch

from .core import TestCaseResult, TestStatus


class SystemInspector:
    def __init__(self, root_dir: Path):
        self.root = root_dir

    def inspect_environment(self) -> dict[str, Any]:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        info = {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torch_device": device,
            "cuda_available": torch.cuda.is_available(),
            "cpu_threads": os.cpu_count(),
            "project_root": str(self.root),
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        return info

    def audit_model_registry(self) -> tuple[dict[str, Any], list[TestCaseResult]]:
        registry_dir = self.root / "model_registry"
        test_results: list[TestCaseResult] = []
        manifests_info: dict[str, Any] = {}

        if not registry_dir.exists():
            test_results.append(
                TestCaseResult(
                    test_id="SYS_REG_001",
                    name="Model Registry Directory Presence",
                    category="Architecture",
                    status=TestStatus.FAIL,
                    execution_time_ms=1.0,
                    details=f"Registry directory missing at {registry_dir}",
                )
            )
            return manifests_info, test_results

        yaml_files = list(registry_dir.rglob("*.yaml"))
        for yf in sorted(yaml_files):
            model_id = yf.stem
            try:
                with open(yf, "r", encoding="utf-8") as f:
                    manifest = yaml.safe_load(f)

                m_id = manifest.get("model_id", model_id)
                modality = manifest.get("modality", "")
                anatomy = manifest.get("anatomy", "")
                architecture = manifest.get("architecture", "")
                class_names = manifest.get("class_names", [])
                weights_rel = manifest.get("weights_path", "")
                weights_path = (self.root / weights_rel) if not Path(weights_rel).is_absolute() else Path(weights_rel)

                weights_exist = weights_path.exists()
                weights_size_bytes = weights_path.stat().st_size if weights_exist else 0

                # Validate weights loading
                weights_loadable = False
                load_error = None
                if weights_exist:
                    try:
                        _ = torch.load(weights_path, map_location="cpu", weights_only=False)
                        weights_loadable = True
                    except Exception as e:
                        load_error = str(e)

                status = TestStatus.PASS if (weights_exist and weights_loadable and len(class_names) > 0) else TestStatus.FAIL
                details = f"Loaded manifest {m_id} [{modality}/{anatomy}]: {len(class_names)} classes, weights: {weights_size_bytes//1024} KB"
                if not weights_exist:
                    details += f" — ERROR: weights file missing at {weights_path}"
                elif not weights_loadable:
                    details += f" — ERROR loading weights: {load_error}"

                test_results.append(
                    TestCaseResult(
                        test_id=f"SYS_MOD_{m_id}",
                        name=f"Model Manifest & Weights Check: {m_id}",
                        category="Registry",
                        status=status,
                        execution_time_ms=5.0,
                        details=details,
                        evidence={
                            "manifest_path": str(yf),
                            "model_id": m_id,
                            "modality": modality,
                            "anatomy": anatomy,
                            "architecture": architecture,
                            "class_names": class_names,
                            "weights_path": str(weights_path),
                            "weights_exist": weights_exist,
                            "weights_size_kb": weights_size_bytes // 1024,
                            "weights_loadable": weights_loadable,
                        },
                        error_message=load_error,
                    )
                )

                manifests_info[m_id] = {
                    "modality": modality,
                    "anatomy": anatomy,
                    "architecture": architecture,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "weights_path": str(weights_path),
                    "weights_exist": weights_exist,
                    "weights_loadable": weights_loadable,
                }
            except Exception as e:
                test_results.append(
                    TestCaseResult(
                        test_id=f"SYS_MOD_ERR_{model_id}",
                        name=f"Parse Manifest: {model_id}",
                        category="Registry",
                        status=TestStatus.FAIL,
                        execution_time_ms=1.0,
                        details=f"Failed to parse manifest {yf}: {e}",
                        error_message=str(e),
                    )
                )

        return manifests_info, test_results
