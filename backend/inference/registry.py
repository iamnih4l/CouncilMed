"""
CouncilMed — Model Registry
==============================
Manages model manifests, loading, validation metadata, and compatibility checking.

Each model is described by a YAML manifest that contains:
  - Architecture and version information
  - Supported modality and anatomy
  - Validated performance metrics (AUROC, sensitivity, specificity, etc.)
  - Input/output format specifications
  - Hardware requirements
  - Training and validation dataset information

The registry NEVER invents metrics. If a model has not been validated,
its manifest will explicitly state that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ModelTask(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    LOCALIZATION = "localization"
    DETECTION = "detection"


class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    MONAI = "monai"


@dataclass
class ValidationMetrics:
    """Validated performance metrics for a model."""
    dataset: str = "unknown"
    dataset_size: int = 0
    split_method: str = "unknown"  # e.g., "patient-level 80/10/10"
    
    # Classification metrics
    auroc: float | None = None
    auprc: float | None = None
    accuracy: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    
    # Segmentation metrics
    dice: float | None = None
    iou: float | None = None
    
    # Calibration
    ece: float | None = None  # Expected Calibration Error
    
    # Performance
    inference_latency_ms: float | None = None
    
    # Status
    validated: bool = False
    validation_notes: str = ""


@dataclass
class ModelManifest:
    """Complete description of a council member model."""
    # Identity
    name: str = "unnamed"
    version: str = "0.0.0"
    model_id: str = ""  # Unique identifier
    description: str = ""
    
    # Architecture
    architecture: str = "unknown"
    framework: ModelFramework = ModelFramework.PYTORCH
    architecture_family: str = ""  # e.g., "CNN", "ViT", "U-Net"
    
    # Task
    tasks: list[ModelTask] = field(default_factory=list)
    modality: str = ""  # e.g., "MRI", "CT", "XRAY"
    anatomy: str = ""   # e.g., "brain", "chest", "abdomen"
    
    # Classes
    class_names: list[str] = field(default_factory=list)
    num_classes: int = 0
    
    # Input specification
    input_shape: list[int] = field(default_factory=lambda: [224, 224, 3])
    input_format: str = "CHW"  # "CHW" or "HWC"
    input_normalization: str = "imagenet"  # "imagenet", "0-1", "-1-1", "medical"
    
    # Weights
    weights_path: str = ""
    weights_checksum: str = ""
    weights_size_mb: float = 0.0
    pretrained_on: str = ""  # e.g., "ImageNet", "BraTS 2023"
    
    # Validation
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    
    # Hardware
    min_ram_gb: float = 2.0
    min_vram_gb: float = 0.0
    supports_cpu: bool = True
    supports_cuda: bool = True
    supports_mps: bool = False
    
    # Provenance
    license: str = "unknown"
    source: str = ""
    paper: str = ""
    training_dataset: str = ""
    
    # Status
    installed: bool = False
    compatible: bool = True
    status_message: str = ""


class ModelRegistry:
    """
    Registry of all available council member models.
    
    Loads model manifests from YAML files and provides:
    - Model discovery and compatibility checking
    - Validation metrics access
    - Model loading coordination
    """
    
    def __init__(self, registry_dir: str | Path | None = None):
        self.registry_dir = Path(registry_dir) if registry_dir else None
        self.manifests: dict[str, ModelManifest] = {}
    
    def load_from_directory(self, directory: str | Path) -> None:
        """Load all YAML model manifests from a directory (recursive)."""
        directory = Path(directory)
        if not directory.exists():
            logger.warning("Registry directory does not exist: %s", directory)
            return
        
        for yaml_file in directory.rglob("*.yaml"):
            try:
                self.load_manifest(yaml_file)
            except Exception as e:
                logger.error("Failed to load manifest %s: %s", yaml_file, e)
    
    def load_manifest(self, path: str | Path) -> ModelManifest:
        """Load a single model manifest from a YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        manifest = self._parse_manifest(data)
        
        # Check if weights file exists
        if manifest.weights_path:
            weights_path = Path(manifest.weights_path)
            if not weights_path.is_absolute():
                weights_path = path.parent / weights_path
            manifest.installed = weights_path.exists()
            manifest.weights_path = str(weights_path)
        
        self.manifests[manifest.model_id] = manifest
        logger.info(
            "Loaded model manifest: %s v%s (%s/%s) — installed=%s",
            manifest.name, manifest.version, manifest.modality, manifest.anatomy,
            manifest.installed
        )
        
        return manifest
    
    def _parse_manifest(self, data: dict[str, Any]) -> ModelManifest:
        """Parse a raw YAML dict into a ModelManifest."""
        metrics_data = data.get("metrics", {})
        metrics = ValidationMetrics(
            dataset=metrics_data.get("dataset", "unknown"),
            dataset_size=metrics_data.get("dataset_size", 0),
            split_method=metrics_data.get("split_method", "unknown"),
            auroc=metrics_data.get("auroc"),
            auprc=metrics_data.get("auprc"),
            accuracy=metrics_data.get("accuracy"),
            sensitivity=metrics_data.get("sensitivity"),
            specificity=metrics_data.get("specificity"),
            precision=metrics_data.get("precision"),
            recall=metrics_data.get("recall"),
            f1=metrics_data.get("f1"),
            dice=metrics_data.get("dice"),
            iou=metrics_data.get("iou"),
            ece=metrics_data.get("ece"),
            inference_latency_ms=metrics_data.get("inference_latency_ms"),
            validated=metrics_data.get("validated", False),
            validation_notes=metrics_data.get("notes", ""),
        )
        
        tasks_raw = data.get("tasks", [])
        tasks = []
        for t in tasks_raw:
            try:
                tasks.append(ModelTask(t))
            except ValueError:
                logger.warning("Unknown task type: %s", t)
        
        framework = ModelFramework.PYTORCH
        fw_str = data.get("framework", "pytorch").lower()
        try:
            framework = ModelFramework(fw_str)
        except ValueError:
            pass
        
        return ModelManifest(
            name=data.get("name", "unnamed"),
            version=data.get("version", "0.0.0"),
            model_id=data.get("model_id", data.get("name", "unnamed")),
            description=data.get("description", ""),
            architecture=data.get("architecture", "unknown"),
            framework=framework,
            architecture_family=data.get("architecture_family", ""),
            tasks=tasks,
            modality=data.get("modality", ""),
            anatomy=data.get("anatomy", ""),
            class_names=data.get("class_names", []),
            num_classes=data.get("num_classes", 0),
            input_shape=data.get("input_shape", [224, 224, 3]),
            input_format=data.get("input_format", "CHW"),
            input_normalization=data.get("input_normalization", "imagenet"),
            weights_path=data.get("weights_path", ""),
            weights_checksum=data.get("weights_checksum", ""),
            weights_size_mb=data.get("weights_size_mb", 0.0),
            pretrained_on=data.get("pretrained_on", ""),
            metrics=metrics,
            min_ram_gb=data.get("min_ram_gb", 2.0),
            min_vram_gb=data.get("min_vram_gb", 0.0),
            supports_cpu=data.get("supports_cpu", True),
            supports_cuda=data.get("supports_cuda", True),
            supports_mps=data.get("supports_mps", False),
            license=data.get("license", "unknown"),
            source=data.get("source", ""),
            paper=data.get("paper", ""),
            training_dataset=data.get("training_dataset", ""),
        )
    
    def register(self, manifest: ModelManifest) -> None:
        """Manually register a model manifest."""
        self.manifests[manifest.model_id] = manifest
    
    def get(self, model_id: str) -> ModelManifest | None:
        """Get a model manifest by ID."""
        return self.manifests.get(model_id)
    
    def find_compatible(
        self,
        modality: str,
        anatomy: str,
        task: ModelTask | None = None,
    ) -> list[ModelManifest]:
        """
        Find all models compatible with a given modality, anatomy, and task.
        Only returns installed models.
        """
        results = []
        for manifest in self.manifests.values():
            if not manifest.installed:
                continue
            if manifest.modality.lower() != modality.lower():
                continue
            if manifest.anatomy.lower() != anatomy.lower():
                continue
            if task and task not in manifest.tasks:
                continue
            results.append(manifest)
        return results
    
    def get_council_members(
        self,
        modality: str,
        anatomy: str,
    ) -> list[ModelManifest]:
        """
        Get the set of council members for a given study type.
        Returns models suitable for forming a diverse council.
        """
        compatible = self.find_compatible(modality, anatomy)
        
        if not compatible:
            logger.warning(
                "No compatible models for %s/%s — council unavailable",
                modality, anatomy
            )
            return []
        
        # Group by architecture family for diversity
        by_family: dict[str, list[ModelManifest]] = {}
        for m in compatible:
            family = m.architecture_family or m.architecture
            by_family.setdefault(family, []).append(m)
        
        # Select the best model from each family (by validation AUROC if available)
        council = []
        for family, models in by_family.items():
            # Sort by AUROC (descending), use -1 for unvalidated
            models.sort(
                key=lambda m: m.metrics.auroc if m.metrics.auroc is not None else -1,
                reverse=True
            )
            council.append(models[0])
        
        logger.info(
            "Council for %s/%s: %d members from %d families",
            modality, anatomy, len(council), len(by_family)
        )
        
        return council
    
    def list_all(self) -> list[ModelManifest]:
        """List all registered models."""
        return list(self.manifests.values())
    
    def summary(self) -> dict[str, Any]:
        """Get a summary of the registry state."""
        total = len(self.manifests)
        installed = sum(1 for m in self.manifests.values() if m.installed)
        validated = sum(
            1 for m in self.manifests.values() if m.metrics.validated
        )
        
        modalities = set(m.modality for m in self.manifests.values() if m.modality)
        anatomies = set(m.anatomy for m in self.manifests.values() if m.anatomy)
        
        return {
            "total_models": total,
            "installed": installed,
            "validated": validated,
            "modalities": sorted(modalities),
            "anatomies": sorted(anatomies),
        }
