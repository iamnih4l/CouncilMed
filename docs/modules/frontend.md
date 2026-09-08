# Frontend Module Architecture

The CouncilMed frontend is a React application built with Vite and Tailwind CSS. It is designed to be bundled into a desktop application via Tauri.

## Directory: `src/`

### 1. `pages/WorkstationAnalysis.tsx`
- **Purpose**: The core workspace for clinical review.
- **Why it exists**: Clinicians need a split-pane view showing the medical image on one side and the AI evidence on the other.
- **How it works**: Manages the local state of the uploaded image. Uses `useCouncilAnalysis` to trigger the backend API. Renders the `MedicalImageViewer` and `CouncilEvidencePanel`.

### 2. `components/council/CouncilEvidencePanel.tsx`
- **Purpose**: Visualizes the fused consensus and individual model predictions.
- **Why it exists**: To solve the "black box" problem of AI. Doctors must see *which* models voted for a finding and *how confident* they were.
- **How it works**: Maps over the `member_evidence` array from the API response. Renders progress bars for confidence. Highlights models that disagreed with the majority consensus using red/amber alerts.

### 3. `services/api.ts`
- **Purpose**: Typed Axios HTTP client.
- **Why it exists**: Centralizes all backend network requests and provides TypeScript interfaces for the API payloads, ensuring UI components don't crash due to missing JSON keys.
- **Failure Behavior**: If the backend is down, Axios interceptors catch the network error and return a standardized error object, which the UI displays as an "Offline" badge.

### 4. `hooks/useCouncilAnalysis.ts`
- **Purpose**: Custom React hook managing the asynchronous analysis lifecycle.
- **Why it exists**: Prevents UI freezing during the 5-10 second PyTorch inference window.
- **State**: Manages `isAnalyzing`, `progress`, `result`, and `error` states.

### 5. `components/viewer/MedicalImageViewer.tsx`
- **Purpose**: Renders the base medical scan and overlays the Grad-CAM heatmaps.
- **Why it exists**: Clinicians must visually verify *where* the AI is looking.
- **Implementation Details**: Uses CSS absolute positioning to perfectly align the base image and the semi-transparent heatmap PNG returned by the backend. Includes zoom and pan controls.

### Design Rationale: Tailwind CSS
Vanilla CSS or SCSS was eschewed in favor of Tailwind CSS to allow for rapid, utility-first styling of complex data dashboards without maintaining large separate stylesheets.
