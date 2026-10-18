"""
CouncilMed — Analysis Pipeline Orchestrator (Production Patched)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .council.engine import CouncilEngine, ModelCalibration
from .council.evidence import (
    CouncilResult,
    ModelEvidence,
    ModelStatus,
    AgentInsight,
)
from .inference.engine import run_inference, preprocess_image, get_hardware_info
from .inference.registry import ModelRegistry
from .dicom.pipeline import (
    parse_dicom_file,
    parse_standard_image,
    detect_modality,
    detect_anatomy,
    DICOMMetadata,
)
from .media import generate_preview_png, generate_gradcam_overlay_png

logger = logging.getLogger(__name__)


def normalize_modality(modality: str | None, file_path: str = "") -> str:
    m = (modality or "").upper().strip().replace("-", "").replace(" ", "")
    if m in ("MRI", "MR"):
        return "MRI"
    if m in ("XRAY", "XR", "CR", "DX", "XA", "RF", "CHESTXRAY", "CXR"):
        return "XRAY"
    if m in ("CT", "CTSCAN", "CAT"):
        return "CT"

    fp = file_path.lower()
    if any(k in fp for k in ("xray", "x-ray", "cxr", "pneumonia", "pneumothorax", "thorax")):
        return "XRAY"
    if any(k in fp for k in ("ct", "nodule", "embolism", "lung")):
        return "CT"
    return "MRI"


def normalize_anatomy(anatomy: str | None, modality: str = "MRI") -> str:
    a = (anatomy or "").lower().strip()
    if a in ("brain", "head", "skull", "neuro", "cranial"):
        return "brain"
    if a in ("chest", "lungs", "lung", "thorax", "thoracic", "pulmonary"):
        return "chest"
    if a in ("abdomen", "abd", "liver", "kidney"):
        return "abdomen"

    if modality in ("CT", "XRAY"):
        return "chest"
    return "brain"


@dataclass
class AnalysisRequest:
    file_path: str
    study_id: str | None = None
    series_id: str | None = None
    modality_override: str | None = None
    anatomy_override: str | None = None
    enable_explainability: bool = True
    enable_agent: bool = True
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class OverlayInfo:
    model_id: str
    overlay_type: str
    png_path: str | None = None


@dataclass
class AnalysisResponse:
    request_id: str
    council_result: CouncilResult
    agent_insight: AgentInsight | None = None
    metadata: DICOMMetadata | None = None
    modality: str = ""
    anatomy: str = ""
    image_shape: tuple[int, ...] = ()
    total_time_ms: float = 0.0
    hardware: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preview_path: str | None = None
    overlays: list[OverlayInfo] = field(default_factory=list)


class AnalysisPipeline:
    def __init__(
        self,
        registry_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        self.registry = ModelRegistry()
        self.output_dir = Path(output_dir) if output_dir else Path("./output")

        if registry_dir:
            self.registry.load_from_directory(registry_dir)

        self.calibrations: dict[str, ModelCalibration] = {}
        for model_id, manifest in self.registry.manifests.items():
            if manifest.metrics.validated:
                self.calibrations[model_id] = ModelCalibration(
                    model_id=model_id,
                    auroc=manifest.metrics.auroc or 0.85,
                    calibration_ece=manifest.metrics.ece or 0.05,
                    reliability_weight=1.0,
                )

        self.council_engine = CouncilEngine(calibrations=self.calibrations)

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        metadata, pixel_data = self._parse_input(request.file_path)

        preview_path: str | None = None
        try:
            preview_dir = self.output_dir / "previews" / request.request_id
            preview_file = preview_dir / "preview.png"
            generate_preview_png(pixel_data, preview_file)
            preview_path = str(preview_file)
        except Exception as e:
            logger.error("Preview generation failed: %s", e)
            warnings.append(f"Preview generation failed: {e}")

        raw_modality = request.modality_override or detect_modality(metadata)
        modality = normalize_modality(raw_modality, request.file_path)

        raw_anatomy = request.anatomy_override or detect_anatomy(metadata)
        anatomy = normalize_anatomy(raw_anatomy, modality)

        council_members = self.registry.get_council_members(modality, anatomy)
        if not council_members:
            council_members = [
                m for m in self.registry.list_all()
                if m.installed and m.modality.upper() == modality.upper()
            ]

        if not council_members:
            errors.append(f"No compatible models for {modality}/{anatomy}")
            return AnalysisResponse(
                request_id=request.request_id,
                council_result=CouncilResult(
                    primary_finding="No models available",
                    requires_human_review=True,
                    human_review_reasons=[f"No models installed for {modality}/{anatomy}"],
                ),
                metadata=metadata,
                modality=modality,
                anatomy=anatomy,
                image_shape=pixel_data.shape,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                errors=errors,
                warnings=warnings,
                preview_path=preview_path,
            )

        evidence_list: list[ModelEvidence] = []
        for manifest in council_members:
            try:
                evidence = run_inference(manifest, pixel_data)
                evidence_list.append(evidence)
            except Exception as e:
                logger.error("Failed: %s — %s", manifest.name, e)
                evidence_list.append(
                    ModelEvidence(
                        model_id=manifest.model_id,
                        model_name=manifest.name,
                        model_version=manifest.version,
                        model_architecture=manifest.architecture,
                        prediction="error",
                        probabilities={},
                        confidence=0.0,
                        uncertainty=1.0,
                        status=ModelStatus.ERROR,
                        error_message=str(e),
                    )
                )

        council_result = self.council_engine.run(
            evidence_list=evidence_list,
            study_id=request.study_id,
            series_id=request.series_id,
            modality=modality,
            anatomy=anatomy,
        )

        overlays: list[OverlayInfo] = []
        if request.enable_explainability:
            overlays = self._generate_explainability(
                council_result, pixel_data, request.request_id
            )

        agent_insight = None
        if request.enable_agent:
            try:
                agent_insight = self._run_agent(council_result)
            except Exception as e:
                logger.error("Agent reasoning failed: %s", e)
                warnings.append(f"Agent reasoning unavailable: {e}")

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return AnalysisResponse(
            request_id=request.request_id,
            council_result=council_result,
            agent_insight=agent_insight,
            metadata=metadata,
            modality=modality,
            anatomy=anatomy,
            image_shape=pixel_data.shape,
            total_time_ms=total_time_ms,
            hardware=get_hardware_info(),
            errors=errors,
            warnings=warnings,
            preview_path=preview_path,
            overlays=overlays,
        )

    def _parse_input(self, file_path: str) -> tuple[DICOMMetadata, np.ndarray]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in (".dcm", ".dicom", ""):
            try:
                return parse_dicom_file(path)
            except Exception:
                pass

        if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
            return parse_standard_image(path)

        if suffix in (".nii", ".gz"):
            return self._parse_nifti(path)

        return parse_standard_image(path)

    def _parse_nifti(self, file_path: Path) -> tuple[DICOMMetadata, np.ndarray]:
        import nibabel as nib

        img = nib.load(str(file_path))
        data = img.get_fdata().astype(np.float32)
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = (data - dmin) / (dmax - dmin)
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]
        elif data.ndim == 4:
            data = data[:, :, data.shape[2] // 2, 0]

        meta = DICOMMetadata(
            rows=data.shape[0],
            columns=data.shape[1],
            modality="MR",
            file_path=str(file_path),
        )
        return meta, data

    def _generate_explainability(
        self,
        council_result: CouncilResult,
        pixel_data: np.ndarray,
        request_id: str,
    ) -> list[OverlayInfo]:
        overlays: list[OverlayInfo] = []
        try:
            from .explainability.gradcam import generate_explainability
        except ImportError:
            return overlays

        preview_h, preview_w = pixel_data.shape[0], pixel_data.shape[1] if pixel_data.ndim >= 2 else 1
        target_size = (preview_h, preview_w)

        for evidence in council_result.member_evidence:
            if evidence.status != ModelStatus.SUCCESS:
                continue

            manifest = self.registry.get(evidence.model_id)
            if manifest is None or manifest.framework.value != "pytorch":
                continue

            try:
                from .inference.engine import load_pytorch_model

                model = load_pytorch_model(manifest)
                class_names = manifest.class_names
                target_class = 0
                for i, name in enumerate(class_names):
                    if name == evidence.prediction:
                        target_class = i
                        break

                preprocessed_image = preprocess_image(
                    pixel_data,
                    target_size=(manifest.input_shape[0], manifest.input_shape[1]),
                    normalization=manifest.input_normalization,
                    output_format=manifest.input_format,
                )

                output_dir = self.output_dir / "explainability" / request_id
                explainability = generate_explainability(
                    model=model,
                    input_image=preprocessed_image,
                    target_class=target_class,
                    output_dir=output_dir,
                    model_id=evidence.model_id,
                    raw_pixel_data=pixel_data,
                )
                evidence.explainability = explainability

                if explainability.gradcam_path:
                    npy_path = Path(explainability.gradcam_path)
                    png_path = npy_path.with_suffix(".png")
                    generate_gradcam_overlay_png(
                        gradcam_npy_path=npy_path,
                        output_path=png_path,
                        target_size=target_size,
                    )
                    overlays.append(
                        OverlayInfo(
                            model_id=evidence.model_id,
                            overlay_type="gradcam",
                            png_path=str(png_path),
                        )
                    )
            except Exception as e:
                logger.error("Explainability failed for %s: %s", evidence.model_id, e)

        return overlays

    def _run_agent(self, council_result: CouncilResult) -> AgentInsight | None:
        try:
            from .agents.orchestrator import run_council_agent
            return run_council_agent(council_result)
        except Exception as e:
            logger.error("Agent execution failed: %s", e)
            return None

    def get_registry_summary(self) -> dict[str, Any]:
        return {
            "registry": self.registry.summary(),
            "hardware": get_hardware_info(),
        }
