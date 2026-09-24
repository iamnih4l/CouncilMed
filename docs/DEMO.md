# CouncilMed Emergency Demo Mode

This is an isolated demonstration mode designed to simulate a complete analysis pipeline in case the real neural network backend fails or cannot be loaded for a presentation.

It provides a deterministic, realistic council finding for a brain MRI image.

### Start Demo

To run the application in Demo Mode, you must set the environment variables before starting the backend and frontend.

**Backend:**
```bash
# Windows (PowerShell)
$env:COUNCILMED_DEMO_MODE="true"
python -m backend.run_backend

# macOS/Linux/Git Bash
export COUNCILMED_DEMO_MODE=true
python -m backend.run_backend
```

**Frontend:**
```bash
# Windows (PowerShell)
$env:VITE_COUNCILMED_DEMO_MODE="true"
npm run dev:frontend

# macOS/Linux/Git Bash
export VITE_COUNCILMED_DEMO_MODE=true
npm run dev:frontend
```

### Enable Demo Mode

See the commands above. Setting `COUNCILMED_DEMO_MODE=true` intercepts the analysis endpoints. Setting `VITE_COUNCILMED_DEMO_MODE=true` displays the "DEMONSTRATION DATA" indicator on the UI.

### Demo Case

The demo data is deterministic and simulates an **MRI Brain** study showing a **Meningioma**.
Upload any sample Brain MRI image (e.g., from `test_data/` or any local image file) when the UI prompts you. The simulated results will attach to this image.

### Expected Flow

1. **UPLOAD:** Select any image file.
2. **ANALYSIS:** Click "Run Analysis".
3. **PROCESSING:** You will see a realistic simulated delay (~2 seconds).
4. **FINDING:** The workstation will present the council findings (Meningioma, Strong Consensus).
5. **REPORT:** Navigate to the Reports tab to see the structured generated report for the case.
6. **ASSISTANT:** (If Ollama is running) You can interact with the Assistant on the right sidebar; it will ingest the demo report and answer questions based on the simulated findings.

### Return to Real Mode

To disable the demo and return to genuine model inference:

**Backend:**
```bash
# Windows
$env:COUNCILMED_DEMO_MODE="false"
python -m backend.run_backend
# or restart the terminal without the variable

# macOS/Linux
export COUNCILMED_DEMO_MODE=false
```

**Frontend:**
```bash
# Windows
set VITE_COUNCILMED_DEMO_MODE=false

# macOS/Linux
export VITE_COUNCILMED_DEMO_MODE=false
```

The real `AnalysisPipeline` and model registry will resume normal operations.
