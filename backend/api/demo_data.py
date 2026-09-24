import time
import uuid
import hashlib
import copy
from typing import Any, Dict

from .schemas import (
    AnalyzeResponse,
    CouncilResultResponse,
    AgentInsightResponse,
    ModelEvidenceResponse,
    ReportResponse,
    ReportSectionResponse,
)

def create_evidence(model_id, name, version, arch, pred, probs, conf, unc, time_ms, device="cuda:0", bbox=None):
    return ModelEvidenceResponse(
        model_id=model_id,
        model_name=name,
        model_version=version,
        model_architecture=arch,
        prediction=pred,
        probabilities=probs,
        confidence=conf,
        uncertainty=unc,
        processing_time_ms=time_ms,
        status="success",
        device=device,
        bounding_box=bbox
    )

SCENARIOS = {
    # ==========================
    # MRI 1: Brain Meningioma
    # ==========================
    "mri_brain_meningioma": {
        "modality": "MRI",
        "anatomy": "Brain",
        "image_shape": [256, 256, 1],
        "council": {
            "finding": "Meningioma",
            "confidence": 0.916,
            "severity": "warning",
            "agreement_score": 0.94,
            "consensus_level": "strong",
            "evidence": [
                create_evidence("densenet-121-mri", "DenseNet-121 Brain MRI", "v1.0.0", "DenseNet", "Meningioma", {"Meningioma": 0.92, "Normal": 0.08}, 0.92, 0.04, 120.5, bbox={"x": 180, "y": 140, "width": 80, "height": 75}),
                create_evidence("swin-unetr-seg", "Swin UNETR Segmentation", "v2.1.0", "Swin-UNETR", "Meningioma", {"Meningioma": 0.88, "Normal": 0.12}, 0.88, 0.06, 345.2),
                create_evidence("efficientnet-b4", "EfficientNet-B4 Classifier", "v1.2.0", "EfficientNet", "Meningioma", {"Meningioma": 0.95, "Normal": 0.05}, 0.95, 0.02, 85.3)
            ]
        },
        "insight": {
            "key": ["Unanimous council agreement on Meningioma.", "High confidence scores."],
            "clinical": ["Likely benign but requires neurosurgical evaluation."],
            "rec": ["Correlate with contrast-enhanced MRI.", "Neurosurgery consultation."],
            "missing": ["Patient clinical history."]
        },
        "report": {
            "type": "MRI Brain",
            "findings": ["Well-circumscribed extra-axial mass with broad dural base.", "Homogeneous enhancement.", "Mild vasogenic edema."],
            "impression": "Well-defined extra-axial enhancing mass consistent with a meningioma. AI Council exhibits strong consensus."
        }
    },

    # ==========================
    # MRI 2: Brain Glioblastoma
    # ==========================
    "mri_brain_gbm": {
        "modality": "MRI",
        "anatomy": "Brain",
        "image_shape": [256, 256, 1],
        "council": {
            "finding": "Glioblastoma Multiforme",
            "confidence": 0.965,
            "severity": "critical",
            "agreement_score": 0.95,
            "consensus_level": "strong",
            "evidence": [
                create_evidence("densenet-121-mri", "DenseNet-121 Brain MRI", "v1.0.0", "DenseNet", "Glioblastoma", {"Glioblastoma": 0.97, "Normal": 0.03}, 0.97, 0.02, 125.0, bbox={"x": 120, "y": 110, "width": 95, "height": 85}),
                create_evidence("swin-unetr-seg", "Swin UNETR Segmentation", "v2.1.0", "Swin-UNETR", "Glioblastoma", {"Glioblastoma": 0.94, "Normal": 0.06}, 0.94, 0.05, 350.5),
                create_evidence("efficientnet-b4", "EfficientNet-B4 Classifier", "v1.2.0", "EfficientNet", "Glioblastoma", {"Glioblastoma": 0.98, "Normal": 0.02}, 0.98, 0.01, 88.0)
            ]
        },
        "insight": {
            "key": ["Irregular, heterogeneously enhancing mass with a necrotic core.", "Significant peritumoral edema."],
            "clinical": ["Highly suspicious for high-grade glioma (GBM)."],
            "rec": ["Urgent neurosurgical consultation.", "Consider MR spectroscopy."],
            "missing": ["Patient clinical history (e.g., headaches, seizures)."]
        },
        "report": {
            "type": "MRI Brain",
            "findings": ["Large, irregular intra-axial mass in the right frontal lobe with thick, irregular peripheral enhancement.", "Central necrosis.", "Extensive surrounding vasogenic edema with mass effect."],
            "impression": "Findings are highly concerning for a high-grade primary glial neoplasm, such as glioblastoma. AI Council exhibits strong consensus."
        }
    },

    # ==========================
    # CT 1: Chest Pulmonary Embolism
    # ==========================
    "ct_chest_pe": {
        "modality": "CT",
        "anatomy": "Chest",
        "image_shape": [512, 512, 1],
        "council": {
            "finding": "Pulmonary Embolism",
            "confidence": 0.965,
            "severity": "critical",
            "agreement_score": 0.98,
            "consensus_level": "strong",
            "evidence": [
                create_evidence("vit-chest-ct", "ViT Chest CT Analyzer", "v2.0.0", "VisionTransformer", "Pulmonary Embolism", {"Pulmonary Embolism": 0.98, "Normal": 0.02}, 0.98, 0.01, 250.8, bbox={"x": 150, "y": 200, "width": 100, "height": 90}),
                create_evidence("vnet-vessel", "V-Net Vessel Seg", "v1.5.0", "V-Net", "Pulmonary Embolism", {"PE": 0.95, "Normal": 0.05}, 0.95, 0.04, 600.0),
                create_evidence("chexnet-ct", "CheXNet CT", "v2.2.0", "DenseNet", "Pulmonary Embolism", {"PE": 0.98, "Normal": 0.02}, 0.98, 0.01, 150.0)
            ]
        },
        "insight": {
            "key": ["High confidence detection of filling defect in main pulmonary artery.", "Urgent finding requiring immediate attention."],
            "clinical": ["Acute pulmonary embolism with risk of right heart strain."],
            "rec": ["STAT clinical correlation.", "Initiate anticoagulation protocol if not contraindicated.", "Echocardiogram to evaluate right heart function."],
            "missing": ["Patient hemodynamics (BP, HR, O2 sat)."]
        },
        "report": {
            "type": "CT Pulmonary Angiogram",
            "findings": ["Large filling defect in the right main pulmonary artery extending into upper and lower lobe branches.", "Right ventricle appears mildly dilated.", "No pleural effusion."],
            "impression": "Acute massive right pulmonary embolism. Signs of right heart strain. CRITICAL FINDING. AI Council exhibits strong consensus."
        }
    },

    # ==========================
    # CT 2: Chest Lung Nodule
    # ==========================
    "ct_chest_nodule": {
        "modality": "CT",
        "anatomy": "Chest",
        "image_shape": [512, 512, 1],
        "council": {
            "finding": "Suspicious Lung Nodule",
            "confidence": 0.880,
            "severity": "warning",
            "agreement_score": 0.89,
            "consensus_level": "strong",
            "evidence": [
                create_evidence("vit-chest-ct", "ViT Chest CT Analyzer", "v2.0.0", "VisionTransformer", "Lung Nodule", {"Lung Nodule": 0.89, "Normal": 0.11}, 0.89, 0.06, 240.5, bbox={"x": 320, "y": 250, "width": 40, "height": 45}),
                create_evidence("3d-unet-nodule", "3D U-Net Nodule Seg", "v1.8.0", "3D U-Net", "Lung Nodule", {"Lung Nodule": 0.85, "Normal": 0.15}, 0.85, 0.08, 550.0),
                create_evidence("chexnet-ct", "CheXNet CT", "v2.2.0", "DenseNet", "Lung Nodule", {"Lung Nodule": 0.92, "Normal": 0.08}, 0.92, 0.04, 145.0)
            ]
        },
        "insight": {
            "key": ["Solitary pulmonary nodule with spiculated margins in the upper lobe.", "Size measures 1.2 cm."],
            "clinical": ["Suspicious for primary lung malignancy given morphological features."],
            "rec": ["PET-CT for metabolic evaluation.", "Pulmonology consult for possible biopsy."],
            "missing": ["Prior imaging for comparison.", "Smoking history."]
        },
        "report": {
            "type": "CT Chest without Contrast",
            "findings": ["1.2 cm spiculated nodule in the apical segment of the right upper lobe.", "No mediastinal or hilar lymphadenopathy.", "No pleural effusion."],
            "impression": "Suspicious 1.2 cm right upper lobe spiculated nodule. Recommend PET-CT and/or tissue sampling. AI Council exhibits strong consensus."
        }
    },

    # ==========================
    # X-Ray 1: Chest Pneumonia
    # ==========================
    "xray_chest_pneumonia": {
        "modality": "X-Ray",
        "anatomy": "Chest",
        "image_shape": [1024, 1024, 1],
        "council": {
            "finding": "Pneumonia",
            "confidence": 0.895,
            "severity": "warning",
            "agreement_score": 0.91,
            "consensus_level": "strong",
            "evidence": [
                create_evidence("chexnet-v2", "CheXNet Pneumonia", "v2.2.0", "DenseNet-121", "Pneumonia", {"Pneumonia": 0.87, "Normal": 0.13}, 0.87, 0.06, 95.5, bbox={"x": 100, "y": 150, "width": 120, "height": 180}),
                create_evidence("cxr-transformer", "CXR-ViT", "v2.0.0", "VisionTransformer", "Pneumonia", {"Pneumonia": 0.91, "Normal": 0.09}, 0.91, 0.04, 120.0),
                create_evidence("resnet-cxr", "ResNet CXR", "v1.5.0", "ResNet-152", "Pneumonia", {"Pneumonia": 0.89, "Normal": 0.11}, 0.89, 0.05, 95.0)
            ]
        },
        "insight": {
            "key": ["Consolidation in the left lower lobe.", "No significant pleural effusion or pneumothorax."],
            "clinical": ["Community-acquired pneumonia is the most likely etiology."],
            "rec": ["Clinical correlation for infectious symptoms.", "Consider appropriate antibiotic therapy.", "Follow-up radiograph in 4-6 weeks to ensure resolution."],
            "missing": ["White blood cell count.", "Fever/cough history."]
        },
        "report": {
            "type": "Chest X-Ray (PA and Lateral)",
            "findings": ["Patchy airspace opacity in the left lower lobe obscuring the left hemidiaphragm.", "Heart size is normal.", "No pneumothorax or large pleural effusion."],
            "impression": "Left lower lobe airspace disease, most suspicious for pneumonia. AI Council exhibits strong consensus."
        }
    },

    # ==========================
    # X-Ray 2: Chest Pneumothorax
    # ==========================
    "xray_chest_pneumothorax": {
        "modality": "X-Ray",
        "anatomy": "Chest",
        "image_shape": [1024, 1024, 1],
        "council": {
            "finding": "Pneumothorax",
            "confidence": 0.970,
            "severity": "critical",
            "agreement_score": 0.98,
            "consensus_level": "strong",
            "evidence": [
                create_evidence("chexnet-v2", "CheXNet Pneumonia", "v2.2.0", "DenseNet-121", "Pneumothorax", {"Pneumothorax": 0.98, "Normal": 0.02}, 0.98, 0.01, 98.0, bbox={"x": 350, "y": 80, "width": 140, "height": 300}),
                create_evidence("cxr-transformer", "CXR-ViT", "v2.0.0", "VisionTransformer", "Pneumothorax", {"Pneumothorax": 0.96, "Normal": 0.04}, 0.96, 0.03, 115.0),
                create_evidence("resnet-cxr", "ResNet CXR", "v1.5.0", "ResNet-152", "Pneumothorax", {"Pneumothorax": 0.97, "Normal": 0.03}, 0.97, 0.02, 92.0)
            ]
        },
        "insight": {
            "key": ["Visible visceral pleural line without peripheral lung markings on the right.", "Large volume pneumothorax."],
            "clinical": ["Acute pneumothorax with risk of progression to tension pneumothorax."],
            "rec": ["STAT clinical evaluation.", "Prepare for immediate tube thoracostomy (chest tube) placement."],
            "missing": ["Vital signs.", "Tracheal deviation clinically?"]
        },
        "report": {
            "type": "Chest X-Ray (AP)",
            "findings": ["Large right-sided pneumothorax with visible visceral pleural edge.", "No significant mediastinal shift.", "Left lung is clear."],
            "impression": "Large right pneumothorax. CRITICAL FINDING. Recommend immediate clinical correlation and management. AI Council exhibits strong consensus."
        }
    }
}

def get_scenario_key(file_path: str) -> str:
    """Determine scenario deterministically based on file path."""
    if not file_path:
        return "mri_brain_meningioma"
        
    fp_lower = file_path.lower()
    
    # Specific keyword overrides
    if "gbm" in fp_lower or "glioblastoma" in fp_lower: return "mri_brain_gbm"
    if "meningioma" in fp_lower: return "mri_brain_meningioma"
    if "pe" in fp_lower or "embolism" in fp_lower: return "ct_chest_pe"
    if "nodule" in fp_lower: return "ct_chest_nodule"
    if "pneumonia" in fp_lower: return "xray_chest_pneumonia"
    if "pneumothorax" in fp_lower: return "xray_chest_pneumothorax"
    
    # Heuristic based on modality
    keys = list(SCENARIOS.keys())
    h = int(hashlib.md5(file_path.encode()).hexdigest(), 16)
    
    if "mri" in fp_lower:
        choices = ["mri_brain_meningioma", "mri_brain_gbm"]
        return choices[h % 2]
    if "ct" in fp_lower:
        choices = ["ct_chest_pe", "ct_chest_nodule"]
        return choices[h % 2]
    if "xray" in fp_lower or "x-ray" in fp_lower:
        choices = ["xray_chest_pneumonia", "xray_chest_pneumothorax"]
        return choices[h % 2]
    
    # Hash fallback across all
    return keys[h % len(keys)]

def get_demo_analysis_result(request_id: str, file_path: str, study_id: str | None) -> AnalyzeResponse:
    """Returns a deterministic, coherent demo analysis result."""
    if not request_id:
        request_id = f"demo-{uuid.uuid4().hex[:8]}"
    elif not request_id.startswith("demo-"):
        request_id = f"demo-{request_id}"

    scenario_key = get_scenario_key(file_path)
    scenario = copy.deepcopy(SCENARIOS[scenario_key])
    c = scenario["council"]
    i = scenario["insight"]

    # Keep demo output visibly tied to the uploaded bytes. Demo mode is not
    # clinical inference, but identical canned boxes make stale uploads look
    # like a frontend bug.
    digest = hashlib.sha256(file_path.encode()).digest()
    if scenario_key.startswith("mri_"):
        box = c["evidence"][0].bounding_box
        if box:
            box["x"] = 32 + digest[0] % 132
            box["y"] = 28 + digest[1] % 132
            box["width"] = 48 + digest[2] % 56
            box["height"] = 48 + digest[3] % 56
        confidence_delta = (digest[4] / 2550) - 0.05
        c["confidence"] = max(0.5, min(0.99, c["confidence"] + confidence_delta))

    council_result = CouncilResultResponse(
        council_run_id=f"run-{request_id[:8]}",
        consensus_level=c["consensus_level"],
        evidence_strength="high",
        primary_finding=c["finding"],
        primary_finding_confidence=c["confidence"],
        severity=c["severity"],
        models_agreeing=len(c["evidence"]),
        models_total=len(c["evidence"]),
        agreement_score=c["agreement_score"],
        ensemble_uncertainty=0.04,
        requires_human_review=c["severity"] == "critical",
        human_review_reasons=["Critical severity finding"] if c["severity"] == "critical" else [],
        member_evidence=c["evidence"],
        disagreements=[],
        spatial_agreement=0.89,
        total_inference_time_ms=551.0,
    )

    agent_insight = AgentInsightResponse(
        key_insights=i["key"],
        clinical_significance=i["clinical"],
        recommended_actions=i["rec"],
        missing_information=i["missing"],
        safety_flags=["CRITICAL FINDING"] if c["severity"] == "critical" else [],
        evidence_references=[f"{ev.model_name} (Conf: {ev.confidence})" for ev in c["evidence"]],
        confidence=c["confidence"],
        agent_model="ollama-llama3-med",
        processing_time_ms=1240.0,
    )

    return AnalyzeResponse(
        request_id=f"{request_id}_{scenario_key}", # Embed scenario key for the report generator
        council_result=council_result,
        agent_insight=agent_insight,
        modality=scenario["modality"],
        anatomy=scenario["anatomy"],
        image_shape=scenario["image_shape"],
        total_time_ms=1791.0,
        hardware={"device": "cuda:0", "gpu_name": "NVIDIA RTX 4090 Demo"},
        errors=[],
        warnings=[]
    )

def get_demo_report(analysis_id: str, study_date: str | None) -> ReportResponse:
    """Returns a deterministic, coherent demo report matching the analysis."""
    # Extract scenario key from analysis_id if present
    scenario_key = "mri_brain_meningioma"
    for key in SCENARIOS.keys():
        if key in analysis_id:
            scenario_key = key
            break
            
    scenario = SCENARIOS[scenario_key]
    r = scenario["report"]

    sections = [
        ReportSectionResponse(
            title="Clinical Indication",
            content=["AI Demonstration Case."],
            severity="info"
        ),
        ReportSectionResponse(
            title="Findings",
            content=r["findings"],
            severity=scenario["council"]["severity"]
        )
    ]

    return ReportResponse(
        report_id=f"rep-{analysis_id[:8]}",
        version=1,
        study_type=r["type"],
        modality=scenario["modality"],
        anatomy=scenario["anatomy"],
        study_date=study_date or "2024-05-15",
        sections=sections,
        impression=r["impression"],
        recommendations=scenario["insight"]["rec"],
        limitations=["Demo Mode. Not for clinical use."],
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        report_status="final",
        report_text=f"[DEMONSTRATION ONLY]\n\nIMPRESSION:\n{r['impression']}"
    )
