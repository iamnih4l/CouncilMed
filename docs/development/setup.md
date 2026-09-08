# Setup & Installation Guide

This guide covers setting up CouncilMed from scratch.

## Prerequisites
- **Node.js**: v18 or higher
- **Python**: v3.10 or higher
- **Ollama**: Required for Agentic AI features
- **Rust**: Required only if building the Tauri desktop application

## 1. Clone Repository
```bash
git clone https://github.com/iamnih4l/CouncilMed.git
cd CouncilMed
```

## 2. Install Dependencies

**Frontend:**
```bash
npm install
```

**Backend:**
```bash
python -m pip install -r backend/requirements.txt
```

## 3. Configure Ollama
To enable the Agentic reasoning layer, you must have Ollama installed and running on your system.
Pull the default instruction model:
```bash
ollama pull llama3:8b-instruct-fp16
```
*Note: If you wish to use a smaller model, edit `DEFAULT_OLLAMA_MODEL` in `backend/agents/orchestrator.py`.*

## 4. Install Required Models
CouncilMed relies on PyTorch models defined in the `model_registry/`. For development and testing without real clinical weights, you can generate placeholder structural weights:
```bash
python scripts/generate_weights.py
```
This script reads the YAML manifests and generates random `.pth` weights for the Brain MRI models.

## 5. Validate Environment & Run
The included startup script will check your environment and launch both the backend and frontend.
```bash
python scripts/startup.py
```

If successful, the output will display:
```
Frontend:  http://localhost:5173
Backend:   http://localhost:8000
API Docs:  http://localhost:8000/docs
```
