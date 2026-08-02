# CouncilMed: Technical System Documentation

## 1. System Overview

CouncilMed is an AI-powered medical imaging platform that provides automated diagnostic analysis for MRI, CT, and X-Ray scans. The system is built on a client-server architecture where the React frontend handles visualization and user interaction, while the FastAPI backend executes all machine learning inference using PyTorch.

### Architecture Summary

| Component | Technology | Responsibility |
|---|---|---|
| Frontend | React 19 + TypeScript + Vite | Upload UI, 3D visualization, report rendering |
| Backend API | FastAPI + Uvicorn | Image processing, model inference, data persistence |
| ML Inference | PyTorch 2.x + TorchVision | DenseNet-121 classification, Grad-CAM, U-Net segmentation |
| LLM Advisor | LangChain + Ollama/LM Studio | Executive reports, natural-language queries |
| Database | SQLite (aiosqlite + SQLAlchemy) | Scan history, inference logs, council reports |

## 2. Backend Inference Engine

### 2.1 Classification — DenseNet-121

- **Role**: Identifies the specific pathology type from the uploaded scan.
- **Implementation**: `backend/inference/models/densenet.py`
- **Model**: `torchvision.models.densenet121` with `IMAGENET1K_V1` pretrained weights.
- **Input**: 224×224×3 normalized tensor.
- **Output**: 4-class probability distribution per modality.
- **Additional**: Registers a forward hook on `DenseBlock4` to capture feature maps for Grad-CAM.

### 2.2 Spatial Localization — Grad-CAM Attention

- **Role**: Pinpoints the anatomical location of the model's focus.
- **Implementation**: `backend/inference/models/attention.py`
- **Mechanism**: Computes Gradient-weighted Class Activation Maps from DenseNet's last convolutional block.
- **Output**: 14×14 spatial attention heatmap, anatomical region key (e.g., `LIVER`, `FRONTAL_LEFT`), 3D focus center coordinates, and confidence score.
- **Region Mapping**: Center-of-mass of the thresholded heatmap is projected to 3D space ([-3, 3]) and matched to the closest predefined anatomical region centroid via Euclidean distance.

### 2.3 Segmentation — U-Net

- **Role**: Delineates anomaly boundaries and estimates volume.
- **Implementation**: `backend/inference/models/swin_unetr.py`
- **Architecture**: 3-level encoder-decoder U-Net with batch normalization, skip connections, and sigmoid output.
- **Input**: 128×128×3 normalized tensor.
- **Output**: 128×128 probability map → thresholded binary mask → 8×8×8 voxel grid (via Z-axis extrusion) → tumor volume in mm³.

### 2.4 Pipeline Orchestrator

- **Implementation**: `backend/inference/engine.py`
- **Singleton**: `inference_engine` is instantiated at module load time.
- **Flow**: `analyze(image, modality)` →
  1. DenseNet classification + Grad-CAM extraction
  2. Attention localization from Grad-CAM heatmap
  3. U-Net segmentation
  4. Council consensus fusion (weighted confidence, spatial centroid, agreement score)
  5. Clinical note generation
  6. Return `CouncilConsensusResult` JSON

## 3. API Endpoints

### `POST /api/inference` (Primary)

Accepts a raw medical image and returns the full Council consensus result.

**Request**: `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `file` | File | The medical scan image (JPEG, PNG, DICOM) |
| `modality` | String | `MRI`, `CT`, or `XRAY` |

**Response**: JSON matching the `CouncilConsensusResult` interface:
```json
{
  "primaryDiagnosis": "Normal",
  "severity": "clear",
  "overallConfidence": 48.4,
  "anomalyPosition": [0.1, -0.3, 0.0],
  "anomalyRadius": 0.65,
  "affectedRegion": "THALAMUS",
  "densenetResult": { ... },
  "attentionResult": { ... },
  "swinResult": { ... },
  "modelAgreement": 0.25,
  "councilNotes": [ ... ],
  "totalInferenceTimeMs": 312.5
}
```

### `POST /api/advisor/report`

Generates an executive report from structured scan data using a local LLM.

**Request**: `application/json`
```json
{ "scan_data": { "region": "...", "risk_score": 0.83, ... } }
```

**Response**: `AdvisorReportSchema` JSON with `summary`, `key_findings`, `risk_analysis`, `recommended_actions`, `confidence`, and `limitations`.

### `POST /api/advisor/query`

Answers a natural-language question using scan context data.

**Request**: `application/json`
```json
{ "query": "What is the highest risk region?", "context_data": { ... } }
```

### `POST /api/analyze` (Legacy)

Accepts pre-computed model outputs and triggers the LangGraph multi-agent debate.

### `GET /api/scans`

Returns the 50 most recent scan records with their inferences and council reports.

## 4. Frontend Architecture

### DiagnosticEngine (`src/services/inference/DiagnosticEngine.ts`)

The frontend `DiagnosticEngine` is a singleton service that:
1. Reads the uploaded file and generates a data URL for the image viewer.
2. Sends the file to `POST /api/inference` as `multipart/form-data`.
3. Simulates pipeline progress stages (preprocessing → DenseNet → Attention → Swin → Consensus) to keep the UI responsive.
4. Parses the JSON response and updates the reactive state.

### Key UI Pages

| Page | Route | Description |
|---|---|---|
| Dashboard | `/` | Overview metrics and recent scans |
| MRI Analysis | `/mri` | Brain MRI upload + 3D brain model viewer |
| CT Analysis | `/ct` | Abdominal CT upload + 3D torso model viewer |
| X-Ray Analysis | `/xray` | Skeletal X-Ray upload + 3D skeleton model viewer |
| AI Advisor | `/advisor` | LLM-powered executive reports + natural-language queries |
| Reports | `/reports` | Historical report browser |
| Settings | `/settings` | Workstation configuration (physician name, clinical ID) |

## 5. Data Flow

```
User Upload (Frontend)
        │
        │ POST /api/inference (multipart/form-data)
        ▼
FastAPI Backend
        │
        ├─── PIL Image.open() → RGB conversion
        │
        ├─── DenseNet-121 → classification + Grad-CAM features
        │         │
        │         ├── predict_with_gradcam() → probabilities + CAM heatmap
        │         │
        ├─── Attention → run_attention_inference(gradcam, modality)
        │         │
        │         ├── Resize CAM to 14×14 → threshold → center-of-mass
        │         ├── Map to 3D coords → find closest anatomical region
        │         │
        ├─── U-Net → segmenter.predict(image, modality)
        │         │
        │         ├── 128×128 segmentation → 8×8×8 voxel extrusion
        │         ├── Volume calculation (solid voxels × 15.6 mm³)
        │         │
        └─── Council Consensus Fusion
                  │
                  ├── Weighted confidence aggregation
                  ├── Spatial centroid fusion
                  ├── Inter-model agreement scoring
                  ├── Clinical note generation
                  │
                  ▼
        CouncilConsensusResult (JSON) → Frontend
```

## 6. Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_LLM_URL` | `http://localhost:11434/v1` | Local LLM server endpoint for the Advisor |

### Model Configuration

Model parameters can be adjusted in `backend/inference/engine.py`:
- `device`: `"cpu"` or `"cuda"` for GPU acceleration.
- DenseNet model name and temperature are configurable in the `DenseNetClassifier` constructor.

---

**Clinical Note**: This platform serves as a diagnostic aid and does not replace human expert review.
