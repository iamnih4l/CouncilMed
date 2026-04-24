# CouncilMed — AI Algorithm & Dataset Documentation

## 1. System Overview

CouncilMed is a production-ready medical AI diagnostic platform that performs real-time brain tumor classification, attention localization, and segmentation from MRI images — entirely in the browser using TensorFlow.js.

### Architecture: Multi-Model Council Pipeline

The system employs a **Council of Three** neural networks that work in concert:

| Model | Architecture | Purpose | Output |
|-------|-------------|---------|--------|
| **Classifier** | MobileNetV2 + Dense Head | Tumor type identification | 4-class softmax probabilities |
| **Attention Localizer** | MobileNetV2 Encoder + Region Head | Spatial focus detection | 10-region probabilities + 7×7 attention heatmap |
| **Segmenter** | Lightweight U-Net Encoder | Tumor boundary delineation | 10-region probabilities + 7×7 segmentation mask |

---

## 2. Dataset

### Brain Tumor MRI Classification Dataset

- **Source**: HuggingFace "Brain Tumor Classification" (originally from Kaggle)
- **Total Images**: 3,264 MRI scans
- **Image Format**: JPEG, variable resolution (resized to 224×224 for training)
- **Modality**: T1-weighted contrast-enhanced MRI
- **Plane**: Axial slices

### Class Distribution

| Class | Training | Validation | Test | Total |
|-------|----------|------------|------|-------|
| Glioma | 436 | 94 | 93 | 623 |
| Meningioma | 440 | 94 | 94 | 628 |
| No Tumor | 530 | 114 | 114 | 758 |
| Pituitary | 829 | 178 | 178 | 1,185 |
| **Total** | **2,235** | **480** | **479** | **3,264** |

### Data Split Strategy
- **Training**: 70% (2,235 images)
- **Validation**: 15% (480 images)
- **Testing**: 15% (479 images)
- **Split Method**: Stratified random sampling per class

---

## 3. Model Architectures

### 3.1 Classifier (DenseNet-121 / MobileNetV2)

**Base Architecture**: MobileNetV2 (pre-trained on ImageNet)

**Transfer Learning Strategy**:
```
Phase 1 (Frozen Base):
  MobileNetV2 (frozen) → GlobalAveragePooling2D → BatchNorm → Dropout(0.3) 
  → Dense(256, ReLU) → BatchNorm → Dropout(0.3) → Dense(4, Softmax)
  
  Optimizer: Adam(lr=1e-3)
  Epochs: 15 (with EarlyStopping patience=5)

Phase 2 (Fine-Tuning):
  Unfreeze top 30% of MobileNetV2 layers
  Optimizer: Adam(lr=1e-5)
  Epochs: 20 (with EarlyStopping patience=7)
```

**Data Augmentation**:
- Random rotation (±20°)
- Width/height shift (±20%)
- Shear (±15%)
- Zoom (±15%)
- Horizontal flip

**Performance**:
- **Test Accuracy**: 87.98%
- **Test Loss**: 0.3013
- **Parameters**: ~2.3M (MobileNetV2 base) + custom head

### 3.2 Attention Localizer

**Architecture**: MobileNetV2 Encoder → Dual-Head Output

```
Input (224×224×3) → MobileNetV2 (frozen) → Features (7×7×1280)

Branch 1 (Region Classification):
  GAP → Dense(128, ReLU) → Dropout(0.3) → Dense(10, Softmax)

Branch 2 (Spatial Attention):
  Conv2D(1, 1×1, Sigmoid) → Flatten → 49-dim attention vector
```

**Training Data Generation** (Pseudo-labels via Grad-CAM):
1. Load trained classifier
2. Extract last conv layer features (Conv_1, 7×7×1280)
3. Compute channel-weighted mean activation as attention heatmap
4. Scale by prediction confidence
5. Map tumor class to approximate brain region labels

**Region Mapping**:
| Class | Brain Region |
|-------|-------------|
| Glioma | Temporal Left |
| Meningioma | Frontal Left |
| No Tumor | Thalamus (center) |
| Pituitary | Brainstem |

### 3.3 Segmenter

**Architecture**: Lightweight Encoder-Decoder

```
Input (224×224×3)
  → Conv2D(32, 3×3, stride=2) → 112×112
  → Conv2D(64, 3×3, stride=2) → 56×56
  → Conv2D(128, 3×3, stride=2) → 28×28
  → Conv2D(256, 3×3, stride=2) → 14×14
  → Conv2D(256, 3×3, stride=2) → 7×7

Branch 1 (Region Classification):
  GAP → Dense(128, ReLU) → Dense(10, Softmax)

Branch 2 (Segmentation Mask):
  Conv2D(1, 1×1, Sigmoid) → 7×7×1 binary mask
```

**Loss Function**: 
- Region: Categorical Cross-Entropy (weight: 1.0)
- Mask: Binary Cross-Entropy (weight: 2.0)

---

## 4. Training Pipeline

### Environment
- **Python**: 3.12
- **TensorFlow**: 2.21.0
- **Hardware**: CPU-only (no GPU)
- **OS**: Windows 11

### Training Flow

```
1. download_dataset.py    → Download from HuggingFace
2. organize_dataset.py    → Split into train/val/test (70/15/15)
3. train_classifier.py    → Phase 1 + Phase 2 transfer learning
4. train_attention_segmenter.py → Grad-CAM labels → Attention + Segmenter
5. export_to_tfjs.py      → Convert to TF.js layers-model format
```

### TF.js Export Format

Models are exported as **TF.js Layers Model** format:
- `model.json` — Model topology + weight manifest
- `group1-shard*.bin` — Binary weight shards (max 4MB each)

| Model | Size | Shards | Weight Tensors |
|-------|------|--------|---------------|
| Classifier | 10.4 MB | 3 | 272 |
| Attention | 9.7 MB | 3 | 266 |
| Segmenter | 4.1 MB | 1 | 16 |

---

## 5. Frontend Integration

### Model Loading (DiagnosticEngine.ts)

```typescript
// Models are loaded asynchronously from /models/<name>/model.json
this.densenetModel = await tf.loadLayersModel('/models/classifier/model.json');
this.attentionModel = await tf.loadLayersModel('/models/attention/model.json');
this.swinModel = await tf.loadLayersModel('/models/segmenter/model.json');
```

### Inference Pipeline

1. **Image Preprocessing**: Resize to 224×224, normalize to [0,1], tile 1-channel to 3-channel
2. **Classification**: DenseNet121 → 4-class probabilities
3. **Attention Localization**: AttentionNet → 10-region probabilities + 7×7 heatmap
4. **Segmentation**: SwinUNETR → 10-region probabilities + 7×7×1 mask
5. **Council Consensus**: Aggregate results from all three models
6. **3D Visualization**: Generate voxel data from segmentation mask
7. **Report Generation**: Compile findings into downloadable PDF

### Input Support
- **Standard formats**: JPEG, PNG, BMP, TIFF
- **Medical formats**: DICOM (.dcm) — parsed with dicom-parser library
- **Preprocessing**: Automatic grayscale-to-RGB tiling for single-channel MRI

---

## 6. Clinical Mapping

### Pathology Class Mapping

| Model Output Index | Dataset Class | Clinical Pathology |
|-------------------|---------------|-------------------|
| 0 | glioma | Glioblastoma (Grade IV) |
| 1 | meningioma | Meningioma (Grade I) |
| 2 | notumor | Normal |
| 3 | pituitary | Metastatic Lesion |

### Brain Region Schema (10 Regions)

| Index | Region | Coordinates |
|-------|--------|-------------|
| 0 | Frontal Left | (-1.5, 2.0, 0.5) |
| 1 | Frontal Right | (1.5, 2.0, 0.5) |
| 2 | Temporal Left | (-2.5, 0.0, -0.5) |
| 3 | Temporal Right | (2.5, 0.0, -0.5) |
| 4 | Parietal Left | (-1.0, 0.5, 2.0) |
| 5 | Parietal Right | (1.0, 0.5, 2.0) |
| 6 | Occipital | (0.0, -2.0, 1.0) |
| 7 | Cerebellum | (0.0, -2.5, -1.5) |
| 8 | Brainstem | (0.0, -1.0, -2.0) |
| 9 | Thalamus | (0.0, 0.0, 0.0) |

---

## 7. Limitations & Future Work

### Current Limitations
- **Dataset size**: 3,264 images is relatively small for medical AI
- **CPU training**: MobileNetV2 transfer learning on CPU limits training depth
- **2D analysis**: Models process 2D axial slices, not full 3D volumes
- **Pseudo-labels**: Attention and segmenter use classifier-derived labels (not expert annotations)

### Planned Improvements
- Train on BraTS 2023 dataset (>2,000 3D volumes with expert segmentation masks)
- Implement GPU-accelerated training via CUDA/DirectML
- Add multi-plane analysis (axial + sagittal + coronal)
- Integrate nnU-Net for state-of-the-art segmentation
- Add uncertainty quantification (Monte Carlo dropout)

---

## 8. References

1. Sandler, M. et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018)
2. Selvaraju, R.R. et al. "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
3. Ronneberger, O. et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
4. Cheng, J. "Brain Tumor Classification Dataset" (Kaggle/HuggingFace, 2020)
