# Backend Module Architecture

The CouncilMed backend is a Python 3.10+ FastAPI application responsible for orchestrating the ML inference pipeline, database persistence, and agentic reasoning.

## Directory: `backend/`

### 1. `api/app.py`
- **Purpose**: The main entry point for the REST API.
- **Why it exists**: Serves as the communication bridge between the Tauri/React frontend and the PyTorch execution layer.
- **How it works**: Uses FastAPI to define endpoints. Initializes the `AnalysisPipeline` inside a `lifespan` context manager so models are loaded into RAM/VRAM exactly once at startup.
- **Endpoints**:
  - `POST /api/analyze/upload`: Handles multipart form data.
  - `GET /api/health`: Returns system status and loaded models.
  - `GET /api/models`: Returns the model registry manifest.
  - `POST /api/reports/generate`: Generates clinical reports from completed analyses.
  - `POST /api/review`: Accepts human review decisions.

### 2. `pipeline.py`
- **Purpose**: The primary orchestrator of the inference process.
- **Why it exists**: Encapsulates the complexity of routing an image through preprocessing, multiple models, fusion, explainability, and LLM reasoning.
- **Inputs**: `AnalysisRequest` (containing the image file path).
- **Outputs**: `AnalysisResponse` (containing fused results, errors, and metadata).
- **Dependencies**: `inference.executor`, `council.engine`, `agents.orchestrator`.

### 3. `database/`
- **Purpose**: SQLite audit and persistence layer.
- **Why it exists**: Medical systems require strict audit trails. Every inference pass must be recorded for legal and clinical traceability.
- **Implementation**: Uses SQLAlchemy ORM async sessions.
- **Models** (`models.py`):
  - `Analysis`: The top-level record of a user requesting an analysis.
  - `ModelEvidenceRecord`: Records the exact prediction and probability of *every individual model* within the council.
  - `CouncilResultRecord`: Records the mathematical fusion of those predictions.
  - `AuditEvent`: Tracks human actions (e.g., a doctor approving or overriding the AI).

### 4. `reports/generator.py`
- **Purpose**: Converts mathematical council outputs into readable text reports.
- **Why it exists**: Doctors cannot paste JSON into an EMR. They need a standard format (Impression, Findings, Limitations).
- **How it works**: Parses the `CouncilResult`, generates a severity-based header, lists individual model evidence as "Detailed Findings," and appends standard medical disclaimers.

### Modification Risk
- **High**: Modifying `pipeline.py` or `api/app.py` risks breaking the communication between the frontend and the PyTorch execution layer. Ensure the `AnalysisPipeline` state remains globally accessible across requests.
