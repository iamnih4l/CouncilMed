# CouncilMed — AI Algorithm & Dataset Documentation

## 1. System Overview

CouncilMed is a production-ready medical AI diagnostic platform that performs multi-modal image classification, spatial localization, and segmentation using PyTorch on a FastAPI backend. The frontend is a pure presentation layer; all inference executes server-side.

### Architecture: Multi-Model Council Pipeline

| Model | Architecture | Framework | Purpose | Output |
|---|---|---|---|---|
| **Classifier** | DenseNet-121 (ImageNet pretrained) | PyTorch | Pathology type identification | 4-class softmax probabilities |
| **Attention Localizer** | Grad-CAM on DenseBlock4 | PyTorch / NumPy | Spatial focus detection | 14×14 attention heatmap + region key |
| **Segmenter** | Lightweight U-Net (3-level) | PyTorch | Anomaly boundary delineation | 128×128 probability map + 8³ voxel grid |

---

## 2. Supported Modalities & Pathology Classes

### MRI (Brain)

| Index | Dataset Class | Clinical Pathology | Severity |
|---|---|---|---|
| 0 | glioma | Glioblastoma (Grade IV) | Critical |
| 1 | meningioma | Meningioma (Grade I) | Info |
| 2 | notumor | Normal | Clear |
| 3 | pituitary | Metastatic Lesion | Warning |

### CT (Abdomen)

| Index | Clinical Pathology | Severity |
|---|---|---|
| 0 | Abdominal Mass / Lesion | Critical |
| 1 | Hemorrhage / Bleeding | Warning |
| 2 | Clear / No Anomalies | Clear |
| 3 | Ischemia / Infarction | Critical |

### X-Ray (Skeletal)

| Index | Clinical Pathology | Severity |
|---|---|---|
| 0 | Compound Fracture | Critical |
| 1 | Hairline Fracture | Warning |
| 2 | No Fracture Detected | Clear |
| 3 | Joint Effusion | Warning |

---

## 3. Model Architectures

### 3.1 Classifier (DenseNet-121)

**Source**: `torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)`

**Transfer Learning Architecture**:
```
DenseNet-121 (pretrained on ImageNet-1K)
  └── features.denseblock4 → forward hook captures feature maps for Grad-CAM
  └── classifier: Linear(1024 → 4)  [replaced head for 4 medical classes]
```

**Preprocessing**:
```python
transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    Lambda(tile_grayscale_to_3ch),     # Auto-convert single-channel to RGB
    Normalize([0.485, 0.456, 0.406],   # ImageNet mean
              [0.229, 0.224, 0.225])   # ImageNet std
])
```

**Inference**: Forward pass produces logits → softmax → 4-class probabilities. The predicted class is mapped to the modality-specific pathology label.

**Grad-CAM**: During `predict_with_gradcam()`:
1. Forward hook on `DenseBlock4` stores feature maps.
2. Feature maps are globally averaged to produce channel weights.
3. Weighted sum of feature maps → ReLU → normalize to [0, 1].
4. Result is a spatial activation map showing where the model "looked" to make its decision.

### 3.2 Attention Localizer (Grad-CAM)

**Source**: `backend/inference/models/attention.py`

**Input**: Raw Grad-CAM heatmap from DenseNet-121 (variable resolution).

**Processing**:
1. Resize heatmap to 14×14 grid.
2. Apply threshold (max of 0.5 or 70% of peak value).
3. Compute center-of-mass of thresholded region using `scipy.ndimage.center_of_mass`.
4. Map center-of-mass from grid coordinates [0, 14] to 3D anatomical space [-3, 3].
5. Find closest anatomical region via Euclidean distance to predefined region centroids.

**Output**: `AttentionNetOutput` with `attentionMap` (14×14), `focusRegion`, `focusCenter` (3D), `focusRadius`, and `confidence`.

### 3.3 Segmenter (U-Net)

**Source**: `backend/inference/models/swin_unetr.py`

**Architecture**:
```
MiniUNet:
  Encoder: 3 levels → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → MaxPool(2)
  Channels: 3 → 32 → 64 → 128
  Bottleneck: 128 → 256
  Decoder: 3 levels → ConvTranspose(2×2) + Skip → Conv(3×3) → BN → ReLU
  Output: Conv(1×1) → Sigmoid → 128×128 probability map
```

**Post-processing**:
1. Threshold at 0.5 to produce binary mask.
2. Downsample to 8×8.
3. Extrude into 8×8×8 3D grid using a depth-decay function: `weight = 1 - |z - 4| / 4`.
4. Map voxel grid coordinates to [-3, 3] 3D space.
5. Count solid voxels × 15.6 mm³ per voxel = estimated volume.

---

## 4. Council Consensus Algorithm

### Confidence Aggregation
```
overall = 0.40 × densenet.confidence + 0.30 × attention.confidence + 0.30 × swin.confidence
```

### Spatial Fusion
```python
total = attn_conf + swin_conf
fused[i] = (attn_conf * attn_center[i] + swin_conf * swin_center[i]) / total
```

### Agreement Scoring (0 to 1)
```
spatial_agreement = max(0, 1 - euclidean_distance(attn_center, swin_center) / 5)
class_agreement   = {1.0 if all agree, 0.6 if partial, 0.2 if disagreement}
total_agreement   = 0.5 × spatial + 0.5 × class
```

---

## 5. Anatomical Region Schema

### Brain Regions (MRI)

| Key | Label | 3D Center |
|---|---|---|
| FRONTAL_LEFT | Left Frontal Lobe | (1.2, 0.8, 1.5) |
| FRONTAL_RIGHT | Right Frontal Lobe | (-1.2, 0.8, 1.5) |
| TEMPORAL_LEFT | Left Temporal Lobe | (1.8, -0.3, 0.5) |
| TEMPORAL_RIGHT | Right Temporal Lobe | (-1.8, -0.3, 0.5) |
| PARIETAL_LEFT | Left Parietal Lobe | (0.8, 1.2, -0.3) |
| PARIETAL_RIGHT | Right Parietal Lobe | (-0.8, 1.2, -0.3) |
| OCCIPITAL | Occipital Lobe | (0, 0.2, -1.8) |
| CEREBELLUM | Cerebellum | (0, -1.5, -1.2) |
| BRAINSTEM | Brainstem | (0, -1.8, 0) |
| THALAMUS | Thalamus | (0, 0, 0.2) |

### Torso Regions (CT)

| Key | Label | 3D Center |
|---|---|---|
| LIVER | Liver | (0.8, 0.5, 0) |
| SPLEEN | Spleen | (-0.8, 0.5, -0.2) |
| KIDNEY_LEFT | Left Kidney | (-0.6, 0, -0.5) |
| KIDNEY_RIGHT | Right Kidney | (0.6, 0, -0.5) |
| PANCREAS | Pancreas | (0, 0.2, -0.2) |
| LOWER_ABDOMEN | Lower Abdomen | (0, -1.0, 0) |

### Skeleton Regions (X-Ray)

| Key | Label | 3D Center |
|---|---|---|
| TIBIA_PROXIMAL | Proximal Tibia | (0, 2.0, 0) |
| TIBIA_DIAPHYSIS | Tibial Diaphysis | (0, 0, 0) |
| TIBIA_DISTAL | Distal Tibia | (0, -2.0, 0) |
| FIBULA | Fibula | (0.5, 0, 0) |

---

## 6. Offline LLM Advisor

| Property | Value |
|---|---|
| Connection | `langchain-openai` → OpenAI-compatible local endpoint |
| Default URL | `http://localhost:11434/v1` (Ollama) |
| Output Format | Structured JSON enforced via `response_format: { type: "json_object" }` |
| Pydantic Schema | `AdvisorReportSchema`, `NaturalLanguageQueryResponse` |

**Capabilities**:
- Executive report generation from scan metadata
- Natural-language query answering grounded in provided context data
- Scenario simulation (qualitative reasoning)

---

## 7. Technical Stack

| Component | Technology |
|---|---|
| ML Inference | PyTorch 2.x, TorchVision (DenseNet-121), Grad-CAM |
| Segmentation | Custom U-Net (PyTorch) |
| Backend API | FastAPI, Uvicorn, python-multipart |
| LLM Integration | LangChain, langchain-openai |
| Multi-Agent | LangGraph (Council debate orchestration) |
| Database | SQLAlchemy (async) + aiosqlite |
| Frontend | React 19, TypeScript, Vite |
| 3D Rendering | @react-three/fiber, @react-three/drei |

---

## 8. Limitations & Future Work

### Current Limitations
- DenseNet-121 uses ImageNet pretrained weights (not fine-tuned on medical data). Classification confidence is moderate (~25-50%).
- U-Net uses randomly initialized weights. Segmentation reflects structural image patterns but is not clinically calibrated.
- Models process 2D axial slices, not full 3D volumetric data.
- Offline LLM requires a local server (Ollama/LM Studio) to be running.

### Planned Improvements
- Fine-tune DenseNet-121 on BraTS 2023 (brain tumors), RSNA (X-rays), and TCIA (CT) datasets.
- Train U-Net on expert-annotated segmentation masks.
- Integrate MONAI's SwinUNETR for true 3D volumetric segmentation.
- Add Monte Carlo dropout for uncertainty quantification.
- Implement ONNX Runtime for optimized cross-platform inference.
- Support DICOM (.dcm) and NIfTI (.nii) file parsing on the backend.

---

## 9. References

1. Huang, G. et al. "Densely Connected Convolutional Networks" (CVPR 2017)
2. Selvaraju, R.R. et al. "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
3. Ronneberger, O. et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
4. Hatamizadeh, A. et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors" (CVPR 2022)
5. Cheng, J. "Brain Tumor Classification Dataset" (Kaggle/HuggingFace, 2020)

---

*Clinical Note: This platform serves as a diagnostic aid and does not replace human expert review.*
