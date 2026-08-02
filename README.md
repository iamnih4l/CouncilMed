# CouncilMed AI Diagnostic Workstation

CouncilMed is a production-grade medical imaging analysis platform powered by a multi-model AI consensus architecture. The system processes MRI, CT, and X-Ray scans through three independent neural networks — DenseNet-121 (classification), Grad-CAM Attention (spatial localization), and U-Net (segmentation) — then fuses their outputs into a single diagnostic consensus using a weighted decision-level fusion algorithm.

All inference runs server-side using **PyTorch** on the FastAPI backend. The React frontend is a pure presentation layer — no model weights are loaded in the browser.

## Key Features

- **Multi-Modal Support**: Dedicated analysis pipelines for Brain MRI, Abdominal CT, and Skeletal X-Ray with modality-specific pathology classes and anatomical region maps.
- **Council Consensus AI**: Three independent models vote on every scan. Spatial agreement, classification consistency, and confidence are mathematically fused into a single diagnostic score.
- **Real PyTorch Inference**: DenseNet-121 with ImageNet pretrained weights, real Grad-CAM attention heatmaps, and a lightweight U-Net segmentation network. No heuristics, no placeholders.
- **Offline LLM Policy Advisor**: An integrated reasoning engine (powered by a local Ollama/LM Studio instance) that analyzes model outputs and generates executive reports, priority-ranked recommendations, and natural-language explanations.
- **3D Visualization**: Interactive Three.js brain, torso, and skeleton models with real-time anomaly overlays driven by segmentation output.
- **Automated PDF Reports**: RSNA-style structured radiology reports generated and persisted via IndexedDB.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 React Frontend                  │
│  (Upload → Progress UI → 3D Viewer → Reports)  │
└───────────────────┬─────────────────────────────┘
                    │  POST /api/inference
                    │  (multipart: image + modality)
                    ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Backend (Python)            │
│                                                 │
│  ┌─────────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ DenseNet-121│ │ Grad-CAM │ │ U-Net Segm.  │ │
│  │ (PyTorch)   │ │ Attention│ │ (PyTorch)    │ │
│  └──────┬──────┘ └─────┬────┘ └──────┬───────┘ │
│         └───────┬───────┘             │         │
│                 ▼                     │         │
│         Council Consensus Fusion ◄────┘         │
│                 │                               │
│                 ▼                               │
│         CouncilConsensusResult (JSON)            │
│                                                 │
│  ┌──────────────────────────────────────┐       │
│  │  Offline LLM Advisor (Ollama)        │       │
│  │  POST /api/advisor/report            │       │
│  │  POST /api/advisor/query             │       │
│  └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite |
| Styling | Tailwind CSS, CSS Variables |
| 3D Rendering | @react-three/fiber, @react-three/drei |
| Backend | FastAPI, Uvicorn, SQLAlchemy (async) |
| ML Inference | PyTorch 2.x, TorchVision, Grad-CAM |
| LLM Reasoning | LangChain, ChatOpenAI (local Ollama/LM Studio) |
| Database | SQLite (via aiosqlite) |
| CI/CD | GitHub Actions |

## Getting Started

### Prerequisites

- **Node.js** v18+
- **Python** 3.12+
- **PyTorch** 2.x (CPU is sufficient; GPU accelerates inference)
- **Ollama** or **LM Studio** (optional, for the AI Advisor feature)

### Setup

```bash
# 1. Clone
git clone https://github.com/iamnih4l/CouncilAI.git
cd CouncilAI

# 2. Install frontend dependencies
npm install

# 3. Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..

# 4. Start both frontend + backend concurrently
npm run dev
```

> On first launch, the backend will automatically download DenseNet-121 pretrained weights (~31MB) from PyTorch Hub.

### Test Data

Pre-generated medical scan images are available in the `test_data/` folder:

| File | Modality | Description |
|---|---|---|
| `mri_scan.png` | MRI | Brain MRI (axial view) |
| `ct_scan.png` | CT | Abdominal CT scan |
| `xray_scan.png` | X-Ray | Skeletal X-ray |

### Running Tests

```bash
npm run test
```

## Diagnostic Pipeline

```
1. Upload       → User uploads image via the frontend UI
2. Transfer     → Image sent as multipart/form-data to POST /api/inference
3. Preprocess   → PIL converts to RGB, resizes to 224×224, normalizes
4. DenseNet-121 → Classification (4 classes per modality)
5. Grad-CAM     → Spatial attention heatmap from last DenseBlock
6. U-Net        → Binary segmentation mask → 3D voxel extrusion
7. Consensus    → Weighted fusion (40% classifier, 30% attention, 30% segmenter)
8. Response     → JSON result returned to frontend
9. Visualization→ 3D model overlay + 2D heatmap + clinical notes
10. PDF Report  → Automatic RSNA-style report generation
```

## Docker Deployment

```bash
# Build and run the full stack
docker build -t councilmed-app .
docker run -p 80:80 -p 8000:8000 councilmed-app
```

## Project Structure

```
CouncilMed/
├── backend/
│   ├── inference/              # PyTorch inference engine
│   │   ├── engine.py           # Pipeline orchestrator
│   │   └── models/
│   │       ├── densenet.py     # DenseNet-121 classifier + Grad-CAM
│   │       ├── attention.py    # Spatial localization from Grad-CAM
│   │       └── swin_unetr.py  # U-Net segmentation
│   ├── advisor/                # Offline LLM reasoning module
│   │   ├── engine.py           # LangChain + ChatOpenAI (local)
│   │   └── schemas.py          # Pydantic response models
│   ├── agents/                 # LangGraph multi-agent orchestrator
│   │   └── orchestrator.py     # Council debate workflow
│   ├── main.py                 # FastAPI application + endpoints
│   ├── models.py               # SQLAlchemy ORM models
│   ├── schemas.py              # Pydantic request/response schemas
│   └── requirements.txt
├── src/
│   ├── components/             # React UI components
│   ├── pages/                  # Route pages (Dashboard, MRI, CT, X-Ray, Advisor)
│   ├── services/
│   │   ├── inference/          # Frontend inference types + DiagnosticEngine
│   │   └── api/                # API client services
│   ├── hooks/                  # React hooks
│   └── workers/                # (Legacy) Web Worker shell
├── test_data/                  # Sample medical images for testing
├── docs/                       # Extended documentation
└── training/                   # Model training scripts
```

---

*CouncilMed is a diagnostic aid designed to augment clinical workflows. It does not replace professional medical judgment.*
