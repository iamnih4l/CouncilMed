# CouncilMed Data Flow

This document outlines how data moves through the CouncilMed system, from external input to the final generated LLM report.

## Complete Data Pipeline

```text
External Image (JPEG/PNG/DICOM)
        │
        ▼
[Input Validation & Parsing]
(File size, extension, MIME type checks)
        │
        ▼
[Preprocessing]
(Resizing, normalization via Torchvision)
        │
        ▼
[Inference Pipeline]
(Image distributed to multiple PyTorch models)
        │
        ├──► Model 1 (e.g., Classifier) ──► Logits / Probabilities
        ├──► Model 2 (e.g., Segmenter) ──► Logits / Probabilities
        └──► Model 3 (e.g., Attention) ──► Logits / Probabilities
        │
        ▼
[Explainability Generation]
(Grad-CAM heatmaps generated per model)
        │
        ▼
[Council Fusion Engine]
(Entropy calculation, calibration weighting, confidence scoring)
        │
        ▼
[Primary Findings]
(Fused consensus, uncertainty metric, disagreements)
        │
        ▼
[Database Persistence]
(SQLite Audit Trail: Evidence records, CouncilResult)
        │
        ▼
[Agentic RAG Preparation]
(ChromaDB queried for clinical context based on primary finding)
        │
        ▼
[Ollama LLM Agent]
(System prompt + Fused Evidence + RAG Context)
        │
        ▼
[Structured LLM Output]
(JSON validation against INSIGHT_SCHEMA)
        │
        ▼
[Workstation UI]
(Evidence Panel, Heatmaps, Agent Insights)
```

## Detailed Data Transformations

### 1. Image Upload
- **Component**: `backend/api/app.py` (`/api/analyze/upload`)
- **Action**: Receives `multipart/form-data`. Validates size (<100MB) and extension.
- **Output**: Temporary file stored in `data/uploads/`.

### 2. Preprocessing
- **Component**: `backend/inference/pipeline.py`
- **Action**: Loads image via PIL. Applies torchvision transforms (Resize, CenterCrop, ToTensor, Normalize) based on the model manifest YAML.
- **Output**: PyTorch Tensor `[B, C, H, W]`.

### 3. Inference
- **Component**: `backend/inference/executor.py`
- **Action**: Passes the tensor through the `.pth` models loaded in memory.
- **Output**: Raw logits, which are passed through softmax to generate class probabilities.
- **Explainability**: Intermediate gradients are captured for Grad-CAM.

### 4. Council Fusion
- **Component**: `backend/council/engine.py`
- **Input**: List of `ModelEvidence` objects containing probabilities and individual uncertainties.
- **Transformation**:
  - `compute_entropy()` measures how sharp each prediction is.
  - `compute_model_weight()` fuses entropy, calibration ECE, and historical AUROC into a single weight.
  - Probabilities are averaged using these weights.
- **Output**: `CouncilResult` object containing the primary finding, consensus level, and ensemble uncertainty.

### 5. Knowledge Retrieval
- **Component**: `backend/knowledge/rag.py`
- **Action**: The primary finding (e.g., "Meningioma") is converted into an embedding and queried against ChromaDB.
- **Output**: Markdown string containing clinical guidelines.

### 6. Agentic Generation
- **Component**: `backend/agents/orchestrator.py`
- **Action**: The `CouncilResult` is formatted into a text summary and appended to the RAG context. This is sent to Ollama via `httpx`.
- **Output**: A strict JSON object containing clinical significance, missing info, and safety flags.
