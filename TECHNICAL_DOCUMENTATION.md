# CouncilMed: Technical System Documentation

## 1. System Overview
CouncilMed is an AI-powered neuro-radiology platform that provides automated MRI analysis. The system is built on a distributed inference architecture that runs entirely in the client browser using WebGL acceleration.

## 2. Model Deep-Dive

### A. Classification Specialist (MobileNetV2)
- **Role**: Identifies the specific pathology type.
- **Input**: 224x224x3 Normalized Tensor.
- **Classes**: 
  - `Glioblastoma (Grade IV)`
  - `Meningioma (Grade I)`
  - `Normal`
  - `Metastatic Lesion`
- **Output**: Probability distribution across classes.

### B. Localization Specialist (Attention-Net)
- **Role**: Pinpoints the anatomical location.
- **Mechanism**: Convolutional Block Attention Module (CBAM).
- **Output**: Anatomical region focus (e.g., "Right Frontal Lobe") and a 7x7 spatial heatmap.

### C. Volumetric Specialist (Swin-UNETR)
- **Role**: 3D segmentation and volume calculation.
- **Mechanism**: Shifted Window Transformer (Swin) blocks.
- **Output**: 8x8x8 volumetric density grid used for 3D reconstruction.

## 3. Inference Workflow (The "Consensus" Pipeline)

### Step 1: Pre-Processing
- **DICOM**: Extract pixel data, apply Window/Leveling for clinical contrast preservation.
- **Rescale**: Normalize pixels to `[0, 1]`.
- **Interpolation**: Bi-linear resize to model input dimensions.

### Step 2: Parallel Parallel 
- All three models execute simultaneously using the `DiagnosticEngine`.
- **Execution Engine**: TensorFlow.js (Graph Model format).

### Step 3: Council Fusion
- **Diagnosis**: Weighted average determines the primary impression.
- **Agreement**: Calculates spatial and categorical consistency across the "Council."
- **3D Rendering**: Injects the segmentation mask into a Three.js point-cloud scene.

### Step 4: Structured Reporting
- **PDF Generation**: Automatically synthesizes an RSNA-style radiology report with visual evidence markers.

## 4. Dataset Information
- **Total Images**: 3,264 MRI slices.
- **Splits**: 70% Training | 15% Validation | 15% Testing.
- **Model Test Accuracy**: ~87.98%.

---
**Clinical Note**: This platform serves as a diagnostic aid and does not replace human expert review.
