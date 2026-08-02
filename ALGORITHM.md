# CouncilMed — Algorithm Documentation

## 1. Overview

CouncilMed employs a **Council Consensus** architecture for multi-modal medical image diagnostics. Three independent PyTorch neural networks analyze every scan in parallel, and their outputs are fused into a single diagnostic result using a weighted decision-level fusion algorithm.

The system supports three imaging modalities (MRI, CT, X-Ray), each with dedicated pathology class sets and anatomical region maps.

## 2. Model Architectures

### A. The Classifier — DenseNet-121

| Property | Value |
|---|---|
| Architecture | DenseNet-121 (torchvision) |
| Pretrained Weights | ImageNet-1K |
| Input | 224×224×3 (RGB, normalized) |
| Output | 4-class softmax probabilities per modality |
| Framework | PyTorch 2.x |

**Modality-Specific Classes:**

| Index | MRI (Brain) | CT (Abdomen) | X-Ray (Skeletal) |
|---|---|---|---|
| 0 | Glioblastoma (Grade IV) | Abdominal Mass / Lesion | Compound Fracture |
| 1 | Meningioma (Grade I) | Hemorrhage / Bleeding | Hairline Fracture |
| 2 | Normal | Clear / No Anomalies | No Fracture Detected |
| 3 | Metastatic Lesion | Ischemia / Infarction | Joint Effusion |

**Preprocessing Pipeline:**
1. Load image via PIL, convert to RGB.
2. Resize to 224×224 using bilinear interpolation.
3. Convert to tensor and normalize with ImageNet mean/std:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
4. Single-channel (grayscale) images are tiled to 3 channels automatically.

### B. The Spatial Observer — Grad-CAM Attention

| Property | Value |
|---|---|
| Mechanism | Gradient-weighted Class Activation Mapping (Grad-CAM) |
| Source Layer | DenseNet-121 DenseBlock4 (last convolutional block) |
| Output | 14×14 spatial attention heatmap + anatomical region classification |

**How it works:**
1. During the DenseNet forward pass, feature maps from the last DenseBlock are captured via a registered forward hook.
2. The feature maps are globally averaged across spatial dimensions to produce channel weights.
3. A weighted combination of feature maps produces a Class Activation Map (CAM).
4. ReLU is applied (only positive activations contribute to the class decision).
5. The CAM is normalized to [0, 1] and resized to a 14×14 attention map.
6. The peak activation coordinates are mapped from grid space to 3D anatomical space ([-3, 3]).
7. The closest anatomical region is determined via Euclidean distance to predefined region centroids.

**Anatomical Regions (per modality):**

| Modality | Regions |
|---|---|
| MRI | Frontal L/R, Temporal L/R, Parietal L/R, Occipital, Cerebellum, Brainstem, Thalamus |
| CT | Liver, Spleen, Kidney L/R, Pancreas, Lower Abdomen |
| X-Ray | Proximal Tibia, Tibial Diaphysis, Distal Tibia, Fibula |

### C. The Volumetric Architect — U-Net Segmentation

| Property | Value |
|---|---|
| Architecture | Lightweight U-Net (3-level encoder-decoder) |
| Input | 128×128×3 (RGB, normalized) |
| Output | 128×128 binary segmentation probability map |
| Framework | PyTorch 2.x |

**Architecture Detail:**
```
Encoder:
  Conv(3→32, 3×3) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool(2)
  Conv(32→64)     → BN → ReLU → Conv(64→64)  → BN → ReLU → MaxPool(2)
  Conv(64→128)    → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool(2)

Bottleneck:
  Conv(128→256)   → BN → ReLU → Conv(256→256) → BN → ReLU

Decoder (with skip connections):
  ConvTranspose(256→128) + Skip → Conv(256→128) → BN → ReLU
  ConvTranspose(128→64)  + Skip → Conv(128→64)  → BN → ReLU
  ConvTranspose(64→32)   + Skip → Conv(64→32)   → BN → ReLU

Output:
  Conv(32→1, 1×1) → Sigmoid
```

**3D Voxel Extrusion:**
The 2D segmentation mask is extruded into an 8×8×8 volumetric grid by applying a depth-decay function along the Z-axis. This produces pseudo-3D data for the Three.js point-cloud visualization.

## 3. The Council Consensus Algorithm

### Weighted Confidence Fusion

```
Overall Confidence = 0.40 × DenseNet Confidence
                   + 0.30 × Attention Confidence
                   + 0.30 × Segmentation Confidence
```

### Spatial Position Fusion

The anomaly position is computed as a weighted centroid of the Attention focus center and the Segmentation tumor center, where weights are proportional to each model's confidence score.

```
fused_position[i] = (attn_conf × attn_center[i] + swin_conf × swin_center[i]) / (attn_conf + swin_conf)
```

### Inter-Model Agreement Score

The agreement score (0 to 1) is computed from two components:

1. **Spatial Agreement (50%)**: Based on the Euclidean distance between the Attention focus center and the Segmentation tumor center. Distance of 0 = 1.0 agreement; distance ≥ 5 = 0.0 agreement.

2. **Classification Agreement (50%)**:
   - All three models agree on abnormal/normal → 1.0
   - Classifier says abnormal, at least one spatial model confirms → 0.6
   - Disagreement → 0.2

### Severity Mapping

| Severity | Pathologies |
|---|---|
| **Clear** | Normal, Clear / No Anomalies, No Fracture Detected |
| **Info** | Meningioma (Grade I), Peritumoral Edema |
| **Warning** | Hemorrhage, Hairline Fracture, Low-Grade Glioma, Metastatic Lesion, Joint Effusion |
| **Critical** | Glioblastoma (Grade IV), Abdominal Mass, Ischemia, Compound Fracture |

## 4. Offline LLM Advisor

The system includes an offline LLM reasoning layer that analyzes the structured JSON output from the Council pipeline and generates:

- Executive summaries for policymakers
- Priority-ranked recommended actions with explanations
- Risk analysis narratives
- Natural-language query responses

**Connection**: Uses `langchain-openai` with an OpenAI-compatible local endpoint (Ollama at port 11434, or LM Studio at port 1234).

**Structured Output Enforcement**: JSON mode is enforced via `response_format: { type: "json_object" }` to guarantee parseable output matching the `AdvisorReportSchema`.

## 5. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/inference` | Upload image + modality → Full Council pipeline → JSON result |
| `POST` | `/api/analyze` | Submit pre-computed model outputs → LangGraph debate |
| `POST` | `/api/advisor/report` | Generate executive report from scan context |
| `POST` | `/api/advisor/query` | Answer natural-language query about scan data |
| `GET` | `/api/scans` | Retrieve historical scan records |

## 6. Limitations & Future Work

### Current Limitations
- DenseNet-121 uses ImageNet pretrained weights (not fine-tuned on medical datasets). Classification confidence is moderate.
- U-Net segmentation uses randomly initialized weights. Segmentation output reflects structural patterns but is not clinically calibrated.
- Models process 2D axial slices, not full 3D volumes.

### Planned Improvements
- Fine-tune DenseNet-121 on BraTS 2023 (brain tumors), RSNA (chest X-rays), and TCIA (abdominal CT) datasets.
- Train U-Net on expert-annotated segmentation masks for clinically accurate lesion delineation.
- Integrate MONAI's SwinUNETR for true 3D volumetric segmentation.
- Add Monte Carlo dropout for uncertainty quantification.
- Implement ONNX Runtime for optimized cross-platform inference.

## 7. References

1. Huang, G. et al. "Densely Connected Convolutional Networks" (CVPR 2017)
2. Selvaraju, R.R. et al. "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
3. Ronneberger, O. et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
4. Hatamizadeh, A. et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors" (CVPR 2022)

---

*This system is designed as a clinical assistant to augment physician workflow and is not a replacement for professional medical judgment.*
