"""
Inference Engine — Orchestrates the full medical image analysis pipeline.
Routes images through DenseNet (classification), Grad-CAM (attention/localization),
and U-Net (segmentation), then fuses results into a CouncilConsensusResult.
"""

import time
import math
from PIL import Image
from inference.models.densenet import DenseNetClassifier
from inference.models.attention import run_attention_inference, get_region_dict, find_closest_region
from inference.models.swin_unetr import SwinUNETRSegmenter


# Severity mapping (matches frontend CouncilConsensus.ts)
SEVERITY_MAP = {
    "Normal": "clear",
    "Clear / No Anomalies": "clear",
    "No Fracture Detected": "clear",
    "Meningioma (Grade I)": "info",
    "Peritumoral Edema": "info",
    "Low-Grade Glioma (Grade II)": "warning",
    "Metastatic Lesion": "warning",
    "Hemorrhage / Bleeding": "warning",
    "Hairline Fracture": "warning",
    "Joint Effusion": "warning",
    "High-Grade Glioma (Grade III)": "critical",
    "Glioblastoma (Grade IV)": "critical",
    "Necrotic Core": "critical",
    "Abdominal Mass / Lesion": "critical",
    "Ischemia / Infarction": "critical",
    "Compound Fracture": "critical",
}

# Model weights for weighted confidence fusion
MODEL_WEIGHTS = {
    "densenet": 0.40,
    "attention": 0.30,
    "swin": 0.30,
}


class InferenceEngine:
    """
    Orchestrates the full 3-model Council inference pipeline.
    Singleton instance manages model lifecycle.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        print("[InferenceEngine] Initializing PyTorch models...")
        self.densenet = DenseNetClassifier(device=device)
        self.segmenter = SwinUNETRSegmenter(device=device)
        print("[InferenceEngine] Models loaded successfully.")

    def _compute_agreement(self, densenet_result: dict, attention_result: dict, swin_result: dict) -> float:
        """Compute inter-model agreement score (0-1)."""
        attn_center = attention_result["focusCenter"]
        swin_center = swin_result["tumorCenter"]

        # Spatial agreement
        spatial_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(attn_center, swin_center)))
        spatial_agreement = max(0.0, 1.0 - spatial_dist / 5.0)

        # Classification agreement
        is_abnormal = densenet_result["primaryDiagnosis"] not in ("Normal", "Clear / No Anomalies", "No Fracture Detected")
        swin_detected = swin_result["confidence"] > 30
        attn_detected = attention_result["confidence"] > 30

        if is_abnormal and swin_detected and attn_detected:
            class_agreement = 1.0
        elif not is_abnormal and not swin_detected and not attn_detected:
            class_agreement = 1.0
        elif is_abnormal and (swin_detected or attn_detected):
            class_agreement = 0.6
        else:
            class_agreement = 0.2

        return spatial_agreement * 0.5 + class_agreement * 0.5

    def _fuse_positions(self, attention_result: dict, swin_result: dict) -> list:
        """Fuse spatial coordinates from Attention and Swin using weighted centroid."""
        attn_conf = attention_result["confidence"]
        swin_conf = swin_result["confidence"]
        total = attn_conf + swin_conf if (attn_conf + swin_conf) > 0 else 1.0
        attn_w = attn_conf / total
        swin_w = 1.0 - attn_w

        fused = [
            attention_result["focusCenter"][i] * attn_w + swin_result["tumorCenter"][i] * swin_w
            for i in range(3)
        ]
        return fused

    def _generate_notes(self, densenet_result: dict, attention_result: dict, swin_result: dict, agreement: float, modality: str) -> list:
        """Generate clinical notes from model outputs."""
        notes = []

        region_dict = get_region_dict(modality)

        # DenseNet finding
        normal_classes = {"Normal", "Clear / No Anomalies", "No Fracture Detected"}
        if densenet_result["primaryDiagnosis"] not in normal_classes:
            notes.append(
                f"DenseNet-121 classifies primary finding as {densenet_result['primaryDiagnosis']} "
                f"with {densenet_result['confidence']:.1f}% confidence."
            )
        else:
            notes.append("DenseNet-121 classification indicates no significant pathology detected.")

        # Attention finding
        focus_key = attention_result["focusRegion"]
        focus_label = region_dict.get(focus_key, {}).get("label", "Unknown Region")
        notes.append(
            f"Attention-Net localizes peak activation in the {focus_label} "
            f"(confidence: {attention_result['confidence']:.1f}%)."
        )

        # Segmentation finding
        vol = swin_result["tumorVolumeMm3"]
        if vol > 100:
            notes.append(f"Swin-UNETR delineates lesion volume at approximately {vol:.0f} mm³.")
            for seg in swin_result["segmentedRegions"]:
                if seg["volumePercentage"] > 5:
                    notes.append(f"  → {seg['label']}: {seg['volumePercentage']:.1f}% of total volume.")
        else:
            notes.append("Swin-UNETR segmentation indicates minimal lesion volume.")

        # Agreement
        if agreement > 0.8:
            notes.append("Council Assessment: HIGH AGREEMENT — All models converge on consistent findings.")
        elif agreement > 0.5:
            notes.append("Council Assessment: MODERATE AGREEMENT — Partial model convergence. Clinical review recommended.")
        else:
            notes.append("Council Assessment: LOW AGREEMENT — Model outputs are divergent. Manual review strongly recommended.")

        return notes

    def analyze(self, image: Image.Image, modality: str = "MRI") -> dict:
        """
        Run the full Council AI inference pipeline on a PIL Image.
        Returns a dict matching the frontend CouncilConsensusResult interface.
        """
        total_start = time.time()

        # Step 1: DenseNet classification + Grad-CAM
        densenet_result, gradcam = self.densenet.predict_with_gradcam(image, modality)

        # Step 2: Attention localization from Grad-CAM
        attention_result = run_attention_inference(gradcam, modality, densenet_result["confidence"])

        # Step 3: Segmentation
        swin_result = self.segmenter.predict(image, modality, densenet_result["confidence"])

        # Step 4: Council consensus fusion
        overall_confidence = (
            densenet_result["confidence"] * MODEL_WEIGHTS["densenet"]
            + attention_result["confidence"] * MODEL_WEIGHTS["attention"]
            + swin_result["confidence"] * MODEL_WEIGHTS["swin"]
        )

        primary_diagnosis = densenet_result["primaryDiagnosis"]
        severity = SEVERITY_MAP.get(primary_diagnosis, "info")

        anomaly_position = self._fuse_positions(attention_result, swin_result)
        anomaly_radius = (attention_result["focusRadius"] + swin_result["tumorRadius"]) / 2.0

        region_dict = get_region_dict(modality)
        affected_region = find_closest_region(anomaly_position, region_dict)

        model_agreement = self._compute_agreement(densenet_result, attention_result, swin_result)
        council_notes = self._generate_notes(densenet_result, attention_result, swin_result, model_agreement, modality)

        total_inference_ms = (time.time() - total_start) * 1000

        return {
            "primaryDiagnosis": primary_diagnosis,
            "severity": severity,
            "overallConfidence": overall_confidence,
            "anomalyPosition": anomaly_position,
            "anomalyRadius": anomaly_radius,
            "affectedRegion": affected_region,
            "densenetResult": densenet_result,
            "attentionResult": attention_result,
            "swinResult": swin_result,
            "modelAgreement": model_agreement,
            "councilNotes": council_notes,
            "totalInferenceTimeMs": total_inference_ms,
        }


# Singleton instance
inference_engine = InferenceEngine()
