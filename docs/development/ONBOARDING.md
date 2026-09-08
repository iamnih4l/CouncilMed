# Developer Onboarding Guide

Welcome to CouncilMed! This guide will take you from cloning the repository to running your first AI analysis workflow.

## 1. Understand CouncilMed
CouncilMed is an evidence-based medical imaging workstation. Instead of relying on one AI model to make a diagnosis, it runs images through a "council" of independent models, fusing their opinions using uncertainty weighting. An Ollama agent then translates that mathematical consensus into a clinical report.
**Start by reading [ARCHITECTURE.md](../ARCHITECTURE.md) and [DATA_FLOW.md](../DATA_FLOW.md).**

## 2. Set up the Environment
Follow the instructions in [setup.md](setup.md) to install Python, Node.js, and Ollama.

## 3. Run the Application
From the root directory, run the startup script to verify your environment and launch the servers:
```bash
python scripts/startup.py
```
This will start:
- FastAPI Backend: `http://localhost:8000`
- Vite Frontend: `http://localhost:5173`

## 4. Run a Basic Workflow
1. Open `http://localhost:5173` in your browser.
2. Ensure the "Brain MRI" modality is selected.
3. Upload any test JPEG/PNG image.
4. Watch the inference pipeline run.
5. Review the evidence panel on the right side, noting the "Primary Finding", "Consensus Score", and the AI Agent's insights.

## 5. Important Code Locations
- **Backend Entry**: `backend/api/app.py`
- **Inference Pipeline**: `backend/pipeline.py`
- **Council Fusion Algorithm**: `backend/council/engine.py`
- **Ollama Agent Logic**: `backend/agents/orchestrator.py`
- **Frontend App**: `src/pages/WorkstationAnalysis.tsx`

## 6. How to Contribute
- **Adding Models**: To add a new AI model, place its weights in `model_registry/weights/` and create a YAML manifest in `model_registry/` defining its architecture. No core code changes are needed!
- **Modifying the Agent**: Tweak the system prompts in `backend/agents/orchestrator.py`.

## 7. Common Issues
- **Backend crashes on startup**: Ensure `AnalysisPipeline` is correctly imported in `backend/api/app.py`.
- **Ollama agent errors**: Ensure the Ollama daemon is running locally and you have pulled the correct model (`ollama pull llama3:8b-instruct-fp16`).
