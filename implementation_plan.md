# System Architecture & Implementation Document: CouncilAI Local AI Diagnostic System

## 1. Executive Summary
This document outlines the architecture for a completely local, privacy-first AI diagnostic system. The system integrates three state-of-the-art diagnostic models (DenseNet-121, Attention U-Net, Swin UNETR) aimed natively at **MRI structural inference**, orchestrated by an on-device Large Language Model (via Ollama) to synthesize findings and generate transparent reasoning logs.

*Note on Team Collaboration:* The user is strictly responsible for the **MRI Analysis** module. The **CT Scan** and **X-Ray** modules will be architected as clean placeholder stubs in both the frontend and backend, allowing the user's teammates to seamlessly integrate their distinct neural network pipelines later.

---

## 2. Global Architecture Diagram (Conceptual)
```mermaid
graph TD
    A[React/Vite Frontend (Local)] -->|POST /api/analyze| B[FastAPI Gateway (Local)]
    B --> C{Computer Vision Pipeline (PyTorch/MONAI)}
    C -->|Global Classification| D[DenseNet-121]
    C -->|2D Spatial Attention| E[Attention U-Net]
    C -->|3D Volumetric Features| F[Swin UNETR]
    D -.->|Logits/Features| G[Feature Aggregator]
    E -.->|Masks/Features| G
    F -.->|3D Coordinates| G
    G -->|Structured Output| H[Contextual Prompt Builder]
    H -->|Local API Call| I[Ollama Local LLM]
    I -->|Generates Clinical Reasoning| J[JSON Response Compiler]
    J -->|Returns AnalysisResult| A
```

---

## 3. Component Details & Pipeline Flow

### 3.1. Frontend Tier: React + Typescript 
*   **Role:** User interaction, scan uploading, advanced 3D visualization, and real-time inference tracking.
*   **Current State:** The UI has been heavily refined for a "wowed" user experience (glassmorphism, Framer Motion animations) but must be streamlined to drop the CT/X-Ray modules and exclusively support the sophisticated MRI pipeline.
*   **Next Steps:**
    *   Update navigation and dashboards to reflect an "MRI-exclusive" focus.
    *   Implement robust `FormData` handling for arbitrary `.nii` or DICOM file uploads.

### 3.2. ML Tier (Phase 1): Training Pipelines & Weights Generation
*   **Role:** To generate the `.pth` weights necessary for the inference engine. 
*   **Data Strategy (Heterogeneous Sourcing):** To ensure robust generalization and avoid overfitting to a single scanner type or demographic, the models will be trained on a federated collection of open-source datasets:
    1.  **BraTS (Brain Tumor Segmentation):** High-grade and low-grade glioma multi-modal MRI scans.
    2.  **fastMRI (NYU Langone/Meta):** Massive repository of knee and brain MRIs, critical for baseline healthy anatomy and artifact reduction.
    3.  **OASIS (Open Access Series of Imaging Studies):** Longitudinal and cross-sectional brain MRI data (Alzheimer’s focus).
    4.  **IXI Dataset:** Healthy subjects collected across three different hospitals in London to provide baseline scanner variance.
*   **Model Specializations (Micro-Anomaly Detection):**
    1.  **DenseNet-121:** Analyzes high-resolution multi-slice inputs. The dense skip-connections preserve vanishing gradients, allowing the network to detect faint, low-contrast irregularities (like early-stage micro-lesions) that might be lost in deeper networks.
    2.  **Attention U-Net:** Employs spatial attention gates to explicitly suppress background noise and aggressively amplify small, localized gradient shifts (edges of minute tumors or demyelination spots) that normal CNNs miss.
    3.  **Swin UNETR:** Computes self-attention across 3D voxel patches. The hierarchical transformer structure allows it to correlate a tiny anomaly in one slice with a global 3D structural deformity, ensuring no micro-irregularity is analyzed out of context.
*   **Actionable Deliverable:** A `training/` folder containing isolated, documented training scripts utilizing `monai` and PyTorch, tuned for high-resolution input tensors.

### 3.3. Inference Gateway (Phase 2): FastAPI Backend
*   **Role:** Act as the central orchestrator and proxy between the client, the PyTorch models, and the local LLM.
*   **File Structure (Proposed):**
    ```text
    backend/
    ├── main.py                 (FastAPI REST endpoints and routing)
    ├── model_server_mri.py     (Handles loading MRI `.pth` files and executing PyTorch inference)
    ├── model_server_ct.py      (PLACEHOLDER for teammate CT models)
    ├── model_server_xray.py    (PLACEHOLDER for teammate X-Ray models)
    ├── prompt_manager.py       (Constructs highly specific agentic prompts formatting the ML outputs)
    ├── ml_models/              (Directory for .pth weights)
    └── requirements.txt
    ```
*   **Performance Optimization:** Implement model caching strictly. Load PyTorch weights onto the GPU (`device='cuda'`) once at server startup, not on every request.

### 3.4. Cognitive Tier: The Well-Structured "Council" (Ollama)
*   **Role:** An explicitly formatted, multi-agent debate simulation running on a single local LLM.
*   **Council Structure:** The prompt engineering will structure the analysis as a formal medical council pipeline:
    1.  **The Radiologist Agent (Attention U-Net proxy):** Reports strictly on local edge highlights and micro-anomalies.
    2.  **The Neurologist Agent (Swin UNETR proxy):** Interprets the 3D structural and volumetric meaning of the findings.
    3.  **The Chief Medical Officer (LLM Meta-Agent):** Weighs the confidence of the deep learning models, factors in the DenseNet general classification risk, and produces the final, synthesized clinical rationale.
*   **Prompt Strategy:** The FastAPI backend will inject the exact confidence scores, top feature highlights, and inference latencies into the system prompt. The LLM acts as the Chief, producing a strictly formatted JSON response detailing the debate steps and the final consensus.

---

## 4. Execution Sandbox & Scoping
To build this robustly, we will stagger development:

1.  **Phase 1 (The ML Foundation):** Create and validate the `training/` scripts. I will establish the dataset ingest patterns and the PyTorch models.
2.  **Phase 2 (The Gateway & Cognitive Layer):** Build `backend/main.py`. We will initially feed the PyTorch models dummy zero-tensors just to ensure the pipeline flows seamlessly into Ollama and back out to the frontend.
3.  **Phase 3 (Frontend Wiring):** Rip out the simulated `setTimeout` logic from [MedicalAnalysis.tsx](file:///c:/Users/nihal/CouncilAI/CouncilAI/src/pages/MedicalAnalysis.tsx), refine the layout strictly for MRI, and wire up the `fetch` calls.

---

## 5. Deployment / Real-World Constraints
*   **Hardware Requirements:** Since we are proposing a local stack running PyTorch inference alongside a local LLM, the target machine *must* have sufficient VRAM (e.g., an Nvidia RTX 3060 12GB+ or equivalent).
*   **Dataset Sourcing:** The user is responsible for sourcing the BraTS or relative dataset folders to execute the scripts locally.
