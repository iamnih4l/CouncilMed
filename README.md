<div align="center">

# CouncilMed

**Multi-Model, Agent-Assisted Medical Imaging Analysis Workstation**

*Evidence-based consensus through independent AI council members*

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)](https://typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-82%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/License-Research-yellow)]()

</div>

---

## The Core Idea

> The same medical image is analyzed independently by multiple AI models. Their outputs are collected as evidence and passed into a Council-based fusion layer. The Council evaluates agreement, disagreement, and uncertainty across models and produces a consensus finding with calibrated confidence — never a single model's opinion.

CouncilMed is not another "AI diagnosis" tool. It is an **evidence assembly system** — a workstation where multiple deep learning models act as independent council members, each analyzing the same study. Their predictions, probabilities, and uncertainties are fused through an entropy-weighted ensemble, and the result is interpreted by an agentic AI layer augmented with curated medical knowledge.

**Every statement is traceable to specific model evidence. Nothing is fabricated.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DICOM / Image Input                          │
│                    ↓                                            │
│    ┌──────────────────────────────────────────────┐             │
│    │         Model Registry (YAML Manifests)       │             │
│    │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │             │
│    │  │DenseNet  │ │Attention │ │  Swin-UNETR  │  │             │
│    │  │  -121    │ │   -Net   │ │              │  │             │
│    │  └────┬─────┘ └────┬─────┘ └──────┬───────┘  │             │
│    └───────┼────────────┼──────────────┼──────────┘             │
│            │            │              │                        │
│            ▼            ▼              ▼                        │
│    ┌──────────────────────────────────────────┐                 │
│    │     Council Engine (Uncertainty-Weighted)  │                │
│    │     → Entropy calibration                  │                │
│    │     → Agreement scoring                    │                │
│    │     → Severity classification              │                │
│    │     → Human review determination           │                │
│    └────────────────┬───────────────────────────┘               │
│                     │                                           │
│            ┌────────┴────────┐                                  │
│            ▼                 ▼                                  │
│    ┌──────────────┐  ┌──────────────┐                          │
│    │ Agent (Ollama)│  │   Report     │                          │
│    │ + Knowledge   │  │  Generator   │                          │
│    │   RAG Context │  │              │                          │
│    └──────────────┘  └──────────────┘                          │
│                                                                 │
│    ┌──────────────────────────────────────────┐                 │
│    │           SQLite + Audit Trail            │                │
│    └──────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Properties

| Property | Implementation |
|---|---|
| **Multi-Model Inference** | Each model runs independently via PyTorch/ONNX/Keras — not simulated |
| **Uncertainty-Weighted Fusion** | Shannon entropy calibrates each model's contribution |
| **Evidence Traceability** | Every finding traces to specific model predictions + probabilities |
| **Knowledge-Augmented Agent** | Ollama LLM with ChromaDB RAG context for clinical interpretation |
| **Local-First / Offline** | All processing on the local machine — no cloud, no PHI leakage |
| **Human Review Always Required** | System never claims diagnostic authority |
| **Structured Reports** | Evidence-based clinical reports, not fabricated text |
| **Full Audit Trail** | Every analysis, review, and action logged in SQLite |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) CUDA-capable GPU for faster inference
- (Optional) [Ollama](https://ollama.ai) for agentic AI reasoning
- (Optional) Rust toolchain for Tauri desktop build

### Setup

```bash
# Clone
git clone https://github.com/iamnih4l/CouncilMed.git
cd CouncilMed

# Backend
pip install -r backend/requirements.txt

# Frontend
npm install

# Validate environment
python scripts/startup.py --check
```

### Run

```bash
# Start backend (port 8000)
python -m uvicorn backend.api.app:app --reload --port 8000

# Start frontend (port 5173)
npm run dev

# Or use the startup script
python scripts/startup.py
```

### API Documentation

Once the backend is running, visit: http://localhost:8000/docs

---

## Project Structure

```
CouncilMed/
├── backend/
│   ├── api/                  # FastAPI endpoints + Pydantic schemas
│   │   ├── app.py            # 11 API endpoints
│   │   └── schemas.py        # Request/response models
│   ├── council/              # Evidence fusion engine
│   │   ├── evidence.py       # Core data contracts
│   │   └── engine.py         # Uncertainty-weighted ensemble
│   ├── inference/            # Multi-framework model inference
│   │   ├── registry.py       # YAML manifest system
│   │   └── engine.py         # PyTorch/ONNX/Keras inference
│   ├── agents/               # Agentic AI reasoning
│   │   └── orchestrator.py   # Ollama + RAG context
│   ├── knowledge/            # Knowledge RAG system
│   │   └── rag.py            # ChromaDB + baseline fallback
│   ├── reports/              # Structured clinical reports
│   │   └── generator.py      # Evidence-based report generation
│   ├── explainability/       # Grad-CAM heatmaps
│   │   └── gradcam.py        # Grad-CAM / Grad-CAM++
│   ├── dicom/                # DICOM processing
│   │   └── pipeline.py       # pydicom parsing + metadata
│   ├── database/             # Async SQLite persistence
│   │   ├── models.py         # 8-table schema
│   │   └── engine.py         # SQLAlchemy async engine
│   └── pipeline.py           # Full pipeline orchestration
├── src/                      # React + TypeScript frontend
│   ├── pages/                # 6 workstation pages
│   ├── components/           # Medical viewer, evidence panel
│   ├── hooks/                # useCouncilAnalysis
│   └── services/             # Type-safe API client
├── src-tauri/                # Tauri desktop shell (Rust)
├── model_registry/           # YAML model manifests
├── data/knowledge/           # Medical knowledge base
├── tests/
│   ├── unit/                 # 65 unit tests
│   └── integration/          # 17 integration tests
└── scripts/
    ├── startup.py            # Environment validation
    └── convert_model.py      # Keras → ONNX converter
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | System health + device info |
| `POST` | `/api/analyze` | Run council analysis (file path) |
| `POST` | `/api/analyze/upload` | Run council analysis (file upload) |
| `GET` | `/api/analyses` | List analysis history |
| `GET` | `/api/models` | List registered models |
| `POST` | `/api/review` | Submit human review decision |
| `POST` | `/api/reports/generate` | Generate structured clinical report |
| `GET` | `/api/reports/{id}` | Retrieve existing report |
| `GET` | `/api/knowledge/status` | Knowledge base status |
| `POST` | `/api/knowledge/ingest` | Ingest knowledge files |
| `POST` | `/api/knowledge/query` | Semantic knowledge retrieval |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v
```

**Current: 82 passed, 4 skipped** (ChromaDB tests skipped when not installed)

---

## Adding Models to the Council

Create a YAML manifest in `model_registry/`:

```yaml
model_id: my-custom-model
name: Custom ResNet
version: 1.0.0
architecture: resnet50
framework: pytorch           # pytorch | onnx | keras
modality: MRI
anatomy: brain
class_names:
  - Normal
  - Glioblastoma
  - Meningioma
  - Pituitary
weights:
  path: model_registry/weights/custom_model.pth
  sha256: <hash>
preprocessing:
  input_size: [224, 224]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

The registry auto-discovers manifests on startup. Models are validated against their SHA-256 checksums before loading.

---

## Disclaimer

> **CouncilMed is a research tool and does NOT provide medical diagnoses.** All AI-generated findings are supplementary and MUST be reviewed by qualified clinicians before any clinical decision is made. The system explicitly flags all outputs as requiring human review and never claims diagnostic authority.

---

## Citation

```bibtex
@software{councilmed2026,
  title={CouncilMed: A Multi-Model, Agent-Assisted Framework for Consensus-Driven Medical Image Diagnosis},
  author={Nihal},
  year={2026},
  url={https://github.com/iamnih4l/CouncilMed}
}
```
