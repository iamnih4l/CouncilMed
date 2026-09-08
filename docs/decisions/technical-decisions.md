# Technical Decisions (Architecture Decision Records)

## ADR-001: Entropy-Weighted Fusion vs. Simple Averaging

### Decision
We chose to implement Shannon entropy calculation combined with calibration metrics to weight model outputs, rather than simple unweighted averaging.

### Context
In medical imaging, out-of-distribution (OOD) data is common (e.g., feeding an abdominal scan into a brain model). A simple average might treat a highly uncertain model's output equally with a highly confident model's output, skewing the final result.

### Reason
Entropy effectively measures a model's "sharpness" or confidence. By mathematically down-weighting models that exhibit high entropy (guessing), the Council Engine protects the final consensus from being corrupted by models that are unsure of what they are seeing.

### Trade-off
- **Gain**: Significantly higher robustness against OOD data and poorly calibrated models.
- **Sacrifice**: Higher computational complexity and the requirement to track calibration metrics (ECE) for every model in the registry.

## ADR-002: Strict Separation of Vision and Language Models

### Decision
We chose to use PyTorch for image analysis and Ollama (LLM) strictly for text-based clinical reasoning, rather than using a single multimodal LLM (like LLaVA).

### Context
We need accurate clinical bounding boxes, segmentations, and explainability heatmaps, while also generating readable text reports.

### Reason
Multimodal LLMs are notorious for hallucinating medical findings and struggle with precise spatial localization (segmentation). By restricting the LLM from "seeing" the image, and forcing it to only read the mathematical text output of specialized PyTorch vision models, we completely eliminate visual hallucinations. The LLM grounds its reasoning in the hard math provided by the Council Engine.

### Trade-off
- **Gain**: High spatial precision, Grad-CAM explainability, and near-zero hallucination of non-existent tumors.
- **Sacrifice**: The LLM cannot describe visual characteristics that the PyTorch models weren't explicitly trained to detect.

## ADR-003: SQLite vs PostgreSQL

### Decision
We chose local SQLite via SQLAlchemy over a distributed database like PostgreSQL.

### Context
The application needs to persist analysis results, model evidence, and human review decisions.

### Reason
CouncilMed is designed as a local-first, offline-capable workstation application (via Tauri) to ensure zero PHI leakage. Requiring hospitals to provision a separate PostgreSQL server breaks the standalone desktop application model.

### Trade-off
- **Gain**: Zero-configuration setup, offline capability, absolute data privacy.
- **Sacrifice**: Harder to aggregate data across an entire hospital network for global analytics.
