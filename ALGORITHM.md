# CouncilMed Clinical Intelligence Documentation

## 1. Overview
CouncilMed utilizes a multi-modal "Council Consensus" architecture for high-fidelity brain tumor diagnostics. The system fuses outputs from three distinct deep learning architectures to ensure surgical-grade precision and reduce over-interpretation risks.

## 2. Model Architectures

### A. The Specialist Classifier (DenseNet-Mobile)
- **Architecture**: MobileNetV2 base with a custom GlobalAveragePooling and Dense classification head.
- **Target**: Pathology classification (4 classes: Glioma, Meningioma, Pituitary, Normal).
- **Inference Mode**: TF.js Graph Model.
- **Weight Source**: Pretrained on ImageNet, fine-tuned on 3,264 brain MRI images.

### B. The Spatial Observer (Attention-Net)
- **Architecture**: Convolutional Block Attention Module (CBAM) integration.
- **Target**: Identifies the primary focus center and anatomical region of findings.
- **Output**: 7x7 spatial attention heatmap and brain region classification.

### C. The Volumetric Architect (Swin-UNETR)
- **Architecture**: Shifted Window Transformer (Swin) for 3D volumetric segmentation.
- **Target**: Delineates tumor boundaries and calculates volume (mm³).
- **Output**: Voxel-level segmentation mask for 3D point-cloud rendering.

## 3. The Council Consensus Algorithm
The system aggregates outputs using a weighted decision-level fusion:
1. **Confidence Aggregation**: Weighted average of the three models (40% Classifier, 30% Attention, 30% Segmenter).
2. **Spatial Fusion**: Weighted centroid calculation between Attention-Net peak and Swin-UNETR center.
3. **Clinical Agreement Score**: Evaluates spatial and categorical consistency across models to confirm diagnostic validity.

## 4. Dataset & Preprocessing
- **Source**: Kaggle Brain Tumor MRI Dataset.
- **Data Splitting**: 70% Training, 15% Validation, 15% Testing.
- **Preprocessing**: 
  - RGB conversion (3 channels).
  - Bi-linear interpolation to 224x224.
  - Pixel-level normalization (rescale 1.0/255.0).
  - DICOM windowing (Window Center/Width) for medical-grade dynamic range preservation.

## 5. Technical Stack
- **Engine**: TensorFlow.js (WebGL Accelerated).
- **Formats**: Frozen Graph Models for optimized browser memory footprint.
- **Preprocessing**: custom DICOM parser and HTML5 Canvas processing.
- **Report Engine**: SVG/PDF synthesis with RSNA-standard structured findings.

---
*Note: This system is designed as a clinical assistant to augment physician workflow and is not a replacement for professional medical judgment.*
