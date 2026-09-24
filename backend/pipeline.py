"""
CouncilMed — Analysis Pipeline Orchestrator
==============================================
End-to-end pipeline: Input → Parse → Preprocess → Council Inference → 
Evidence → Fusion → Explainability → Report

This is the single entry point for analyzing a medical image.
It coordinates all backend modules and returns a complete CouncilResult.
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
from .inference.registry import ModelRegistry, ModelManifest, ModelTask
from .dicom.pipeline import (
    parse_dicom_file,
    parse_standard_image,
    detect_modality,
    detect_anatomy,
    DICOMMetadata,
    ParsedImage,
)
from .media import generate_preview_png, generate_gradcam_overlay_png

logger = logging.getLogger(__name__)

def normalize_modality(modality: str | None) -> str:
    if not modality:
        return "UNKNOWN"
    m = modality.upper().strip().replace("-", "")
    if m in ("MRI", "MR"):
        return "MRI"
    if m in ("XRAY", "XR", "CR", "DX", "XA", "RF"):
        return "XRAY"
    if m in ("CT", "CT SCAN"):
        return "CT"
    if m == "US":
        return "US"
    if m == "PET":
        return "PET"
    return m

def normalize_anatomy(anatomy: str | None) -> str:
    if not anatomy:
        return "unknown"
    a = anatomy.lower().strip()
    if a in ("brain", "head", "skull"):
        return "brain"
    if a in ("chest", "lungs", "thorax"):
        return "chest"
    if a in ("abdomen", "abd"):
        return "abdomen"
    return a


@dataclass
class AnalysisRequest:
    """Request to analyze a medical image."""
    file_path: str
    study_id: str | None = None
    series_id: str | None = None
    modality_override: str | None = None   # Override auto-detection
    anatomy_override: str | None = None    # Override auto-detection
    enable_explainability: bool = True
    enable_agent: bool = False             # Agent reasoning (requires Ollama)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class OverlayInfo:
    """Information about a single explainability overlay."""
    model_id: str
    overlay_type: str  # 'gradcam', 'attention', 'segmentation'
    png_path: str | None = None  # Filesystem path to the overlay PNG


@dataclass
class AnalysisResponse:
    """Complete response from the analysis pipeline."""
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
    # Media artifacts
    preview_path: str | None = None          # Filesystem path to preview PNG
    overlays: list[OverlayInfo] = field(default_factory=list)


class AnalysisPipeline:
    """
    The master analysis pipeline.
    
    Coordinates:
    1. Input parsing (DICOM or standard image)
    2. Modality/anatomy detection
    3. Council member selection from registry
    4. Parallel model inference
    5. Evidence aggregation
    6. Council fusion
    7. Explainability generation
    8. Optional agent reasoning
    """
    
    def __init__(
        self,
        registry_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        self.registry = ModelRegistry()
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        
        # Load model manifests
        if registry_dir:
            self.registry.load_from_directory(registry_dir)
        
        # Build calibration data from registry metrics
        self.calibrations: dict[str, ModelCalibration] = {}
        for model_id, manifest in self.registry.manifests.items():
            if manifest.metrics.validated:
                self.calibrations[model_id] = ModelCalibration(
                    model_id=model_id,
                    auroc=manifest.metrics.auroc or 0.5,
                    calibration_ece=manifest.metrics.ece or 0.5,
                    reliability_weight=1.0,
                )
        
        self.council_engine = CouncilEngine(
            calibrations=self.calibrations,
        )
        
        logger.info(
            "Pipeline initialized: %d models in registry, %d with calibration",
            len(self.registry.manifests),
            len(self.calibrations),
        )
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Run the full analysis pipeline.
        
        Args:
            request: AnalysisRequest with file path and options
        
        Returns:
            Complete AnalysisResponse with council result and metadata
        """
        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []
        
        logger.info("=== Analysis Pipeline Start: %s ===", request.request_id)
        
        # Step 1: Parse input
        logger.info("Step 1: Parsing input file")
        metadata, pixel_data = self._parse_input(request.file_path)
        
        # Step 1b: Generate preview PNG from the actual parsed pixel data
        preview_path: str | None = None
        try:
            preview_dir = self.output_dir / "previews" / request.request_id
            preview_file = preview_dir / "preview.png"
            generate_preview_png(pixel_data, preview_file)
            preview_path = str(preview_file)
            logger.info("Preview generated: %s", preview_path)
        except Exception as e:
            logger.error("Preview generation failed: %s", e)
            warnings.append(f"Preview generation failed: {e}")
        
        # Step 2: Detect modality and anatomy
        raw_modality = request.modality_override or detect_modality(metadata)
        raw_anatomy = request.anatomy_override or detect_anatomy(metadata)
        
        modality = normalize_modality(raw_modality)
        anatomy = normalize_anatomy(raw_anatomy)
        
        logger.info("Step 2: Detected modality=%s, anatomy=%s", modality, anatomy)
        
        # Step 3: Select council members
        logger.info("Step 3: Selecting council members")
        council_members = self.registry.get_council_members(modality, anatomy)
        
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
            return AnalysisResponse(
                request_id=request.request_id,
                council_result=CouncilResult(
                    primary_finding="No models available",
                    requires_human_review=True,
                    human_review_reasons=["No models installed"],
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
        
        logger.info(
            "Council members selected: %s",
            [m.name for m in council_members],
        )
        
        # Step 4: Run inference on each council member
        logger.info("Step 4: Running council inference")
        evidence_list: list[ModelEvidence] = []
        
        for manifest in council_members:
            logger.info("  Running: %s (%s)", manifest.name, manifest.architecture_family)
            try:
                evidence = run_inference(manifest, pixel_data)
                evidence_list.append(evidence)
                logger.info(
                    "  Result: %s (confidence=%.1f%%, time=%.1fms)",
                    evidence.prediction,
                    evidence.confidence * 100,
                    evidence.processing_time_ms,
                )
            except Exception as e:
                logger.error("  Failed: %s — %s", manifest.name, e)
                evidence_list.append(ModelEvidence(
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
                ))
        
        # Step 5: Run council fusion
        logger.info("Step 5: Running council fusion")
        council_result = self.council_engine.run(
            evidence_list=evidence_list,
            study_id=request.study_id,
            series_id=request.series_id,
            modality=modality,
            anatomy=anatomy,
        )
        
        logger.info(
            "Council result: %s (confidence=%.1f%%, consensus=%s, uncertainty=%.2f)",
            council_result.primary_finding,
            council_result.primary_finding_confidence * 100,
            council_result.consensus_level.value,
            council_result.ensemble_uncertainty,
        )
        
        # Step 6: Explainability (if enabled and PyTorch models available)
        overlays: list[OverlayInfo] = []
        if request.enable_explainability:
            logger.info("Step 6: Generating explainability artifacts")
            overlays = self._generate_explainability(
                council_result, pixel_data, request.request_id
            )
        
        # Step 7: Agent reasoning (if enabled)
        agent_insight = None
        if request.enable_agent:
            logger.info("Step 7: Running agent reasoning")
            try:
                agent_insight = self._run_agent(council_result)
            except Exception as e:
                logger.error("Agent reasoning failed: %s", e)
                warnings.append(f"Agent reasoning unavailable: {e}")
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            "=== Analysis Pipeline Complete: %.1fms total ===",
            total_time_ms,
        )
        
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
    
    def _parse_input(
        self,
        file_path: str,
    ) -> tuple[DICOMMetadata, np.ndarray]:
        """Parse input file — auto-detect DICOM vs standard image."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        # Try DICOM first
        if suffix in (".dcm", ".dicom", ""):
            try:
                return parse_dicom_file(path)
            except Exception as e:
                logger.info("Not a DICOM file, trying standard image: %s", e)
        
        # Standard image formats
        if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
            return parse_standard_image(path)
        
        # NIfTI
        if suffix in (".nii", ".gz"):
            return self._parse_nifti(path)
        
        # Last resort: try as standard image
        try:
            return parse_standard_image(path)
        except Exception:
            pass
        
        raise ValueError(f"Unsupported file format: {suffix}")
    
    def _parse_nifti(
        self,
        file_path: Path,
    ) -> tuple[DICOMMetadata, np.ndarray]:
        """Parse NIfTI file for volumetric data."""
        import nibabel as nib
        
        img = nib.load(str(file_path))
        data = img.get_fdata().astype(np.float32)
        
        # Normalize to 0-1
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = (data - dmin) / (dmax - dmin)
        
        # If 3D volume, take center slice for 2D analysis
        if data.ndim == 3:
            center_slice = data.shape[2] // 2
            data = data[:, :, center_slice]
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
        """Generate explainability for PyTorch models in the council.
        
        Returns a list of OverlayInfo with paths to generated PNG overlays.
        """
        overlays: list[OverlayInfo] = []
        
        # Explainability requires PyTorch models with gradient access
        # For Keras/ONNX models, we skip (would need separate implementation)
        try:
            from .explainability.gradcam import generate_explainability
        except ImportError:
            logger.warning("Explainability module not available")
            return overlays
        
        # Determine the preview target size for overlay alignment
        preview_h, preview_w = pixel_data.shape[0], pixel_data.shape[1] if pixel_data.ndim >= 2 else 1
        target_size = (preview_h, preview_w)
        
        for evidence in council_result.member_evidence:
            if evidence.status != ModelStatus.SUCCESS:
                continue
            
            manifest = self.registry.get(evidence.model_id)
            if manifest is None:
                continue
            
            # Only PyTorch models support gradient-based explainability for now
            if manifest.framework.value != "pytorch":
                logger.info(
                    "Skipping explainability for %s (framework=%s, gradient access unavailable)",
                    manifest.name, manifest.framework.value
                )
                continue
            
            try:
                from .inference.engine import load_pytorch_model, preprocess_image
                model = load_pytorch_model(manifest)
                
                # Get the target class index
                class_names = manifest.class_names
                target_class = 0
                for i, name in enumerate(class_names):
                    if name == evidence.prediction:
                        target_class = i
                        break
                        
                # Ensure we use the exact same preprocessing as inference
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
                )
                evidence.explainability = explainability
                
                # Convert the .npy heatmap to a browser-displayable PNG overlay
                if explainability.gradcam_path:
                    npy_path = Path(explainability.gradcam_path)
                    png_path = npy_path.with_suffix(".png")
                    try:
                        generate_gradcam_overlay_png(
                            gradcam_npy_path=npy_path,
                            output_path=png_path,
                            target_size=target_size,
                        )
                        overlay_info = OverlayInfo(
                            model_id=evidence.model_id,
                            overlay_type="gradcam",
                            png_path=str(png_path),
                        )
                        overlays.append(overlay_info)
                        logger.info(
                            "Overlay PNG created for %s: %s",
                            evidence.model_id, png_path
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to generate overlay PNG for %s: %s",
                            evidence.model_id, e
                        )
                
            except Exception as e:
                logger.error(
                    "Explainability failed for %s: %s",
                    evidence.model_id, e, exc_info=True
                )
        
        return overlays
    
    def _run_agent(self, council_result: CouncilResult) -> AgentInsight | None:
        """Run the agentic AI layer over the council result."""
        # Lazy import to avoid requiring Ollama when not using agents
        try:
            from .agents.orchestrator import run_council_agent
            return run_council_agent(council_result)
        except ImportError as e:
            logger.warning("Agent module not available: %s", e)
            return None
        except Exception as e:
            logger.error("Agent execution failed: %s", e)
            return None
    
    def get_registry_summary(self) -> dict[str, Any]:
        """Get a summary of available models and hardware."""
        return {
            "registry": self.registry.summary(),
            "hardware": get_hardware_info(),
        }
