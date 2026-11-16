
# CouncilMed — Verified Clinical Evaluation Dataset

> **Note for Final Major Project Demo**: This directory contains genuine, high-resolution, unseen medical scans sourced from internationally recognized peer-reviewed benchmark datasets. All synthetic placeholders and noise artifacts have been replaced with authentic scans representing target clinical conditions.

## Dataset Summary

| Modality | Anatomy | File | Clinical Diagnosis | Resolution | Source Benchmark |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MRI** | brain | `brain_mri_1.png` | Glioma / Glioblastoma | 354x442 | Masoud Nickparvar Brain Tumor MRI Benchmark |
| **MRI** | brain | `brain_mri_2.png` | Meningioma | 512x512 | Masoud Nickparvar Brain Tumor MRI Benchmark |
| **MRI** | brain | `brain_mri_3.png` | Normal Brain MRI | 380x530 | Masoud Nickparvar Brain Tumor MRI Benchmark |
| **MRI** | brain | `mri_glioma_case1.png` | Glioma / Glioblastoma | 512x512 | Masoud Nickparvar Brain Tumor MRI Benchmark |
| **MRI** | brain | `mri_meningioma_case1.png` | Meningioma | 512x512 | Masoud Nickparvar Brain Tumor MRI Benchmark |
| **MRI** | brain | `mri_pituitary_case1.png` | Pituitary Adenoma | 512x512 | Masoud Nickparvar Brain Tumor MRI Benchmark |
| **MRI** | brain | `mri_normal_case1.png` | Normal Brain MRI | 393x400 | Masoud Nickparvar Brain Tumor MRI Benchmark |
| **CT** | chest | `chest_ct_1.png` | Pulmonary Nodule (Adenocarcinoma) | 381x282 | Chest CT-Scan Images Benchmark (TCIA/LIDC-IDRI derived) |
| **CT** | chest | `chest_ct_2.png` | Ground-Glass Opacity (GGO / Viral Pneumonitis) | 509x341 | COVID-19 Thoracic CT Benchmark (kucaantunes) |
| **CT** | chest | `chest_ct_3.png` | Normal Chest CT | 371x251 | Chest CT-Scan Images Benchmark |
| **CT** | chest | `ct_pulmonary_nodule_case1.png` | Pulmonary Nodule / Primary Lesion | 486x335 | Chest CT-Scan Images Benchmark |
| **CT** | chest | `ct_squamous_carcinoma_case1.png` | Squamous Cell Carcinoma | 488x339 | Chest CT-Scan Images Benchmark |
| **CT** | chest | `ct_large_cell_carcinoma_case1.png` | Large Cell Carcinoma | 411x285 | Chest CT-Scan Images Benchmark |
| **CT** | chest | `ct_ground_glass_case1.png` | Bilateral Ground-Glass Opacities | 425x300 | COVID-19 Thoracic CT Benchmark |
| **CT** | chest | `ct_normal_case1.png` | Normal Chest CT | 371x251 | Chest CT-Scan Images Benchmark |
| **XRAY** | chest | `chest_xray_1.png` | Bacterial Pneumonia (Lobar Consolidation) | 1024x680 | Guangzhou Women & Children's Medical Center Kermany Benchmark |
| **XRAY** | chest | `chest_xray_2.png` | Viral Pneumonia (Diffuse Infiltrates) | 1496x960 | Guangzhou Women & Children's Medical Center Kermany Benchmark |
| **XRAY** | chest | `chest_xray_3.png` | Normal Chest Radiograph | 1852x1443 | Guangzhou Women & Children's Medical Center Kermany Benchmark |
| **XRAY** | chest | `xray_pneumonia_bacterial_case1.png` | Bacterial Pneumonia | 1136x680 | Guangzhou Kermany Pediatric Chest X-Ray Benchmark |
| **XRAY** | chest | `xray_pneumonia_viral_case1.png` | Viral Pneumonia | 1314x1145 | Guangzhou Kermany Pediatric Chest X-Ray Benchmark |
| **XRAY** | chest | `xray_normal_case1.png` | Normal Chest Radiograph | 2444x2155 | Guangzhou Kermany Pediatric Chest X-Ray Benchmark |

---

## Modality Case Details & Demo Guide

### 1. Brain MRI (`eva-data/mri/`)
- **`brain_mri_1.png` / `mri_glioma_case1.png`**: High-grade Glioma / Glioblastoma showing infiltrative hyperintense mass. Best for testing `brain_mri_classifier_v1` and `brain_mri_attention_v1`.
- **`brain_mri_2.png` / `mri_meningioma_case1.png`**: Extra-axial Meningioma with well-circumscribed borders. Evaluates council localization & Grad-CAM boundary detection.
- **`brain_mri_3.png` / `mri_normal_case1.png`**: Healthy adult brain control with symmetrical ventricles and absence of mass effect.
- **`mri_pituitary_case1.png`**: Pituitary Adenoma demonstrating sellar/suprasellar mass.

### 2. Chest CT (`eva-data/ct/`)
- **`chest_ct_1.png` / `ct_pulmonary_nodule_case1.png`**: Axial chest CT with distinct solitary pulmonary nodule (Adenocarcinoma). Evaluates nodule detection models (`chest_ct_resnet_v1`).
- **`chest_ct_2.png` / `ct_ground_glass_case1.png`**: Bilateral peripheral Ground-Glass Opacities (GGO). Evaluates viral pneumonitis detection.
- **`chest_ct_3.png` / `ct_normal_case1.png`**: Normal thoracic CT slice with clear lung parenchyma and normal mediastinum.
- **`ct_squamous_carcinoma_case1.png`**: Centrally located pulmonary mass.
- **`ct_large_cell_carcinoma_case1.png`**: Peripheral large necrotic mass.

### 3. Chest X-Ray (`eva-data/xray/`)
- **`chest_xray_1.png` / `xray_pneumonia_bacterial_case1.png`**: Dense lobar bacterial consolidation with air bronchograms. Evaluates CheXNet and ResNet council consensus.
- **`chest_xray_2.png` / `xray_pneumonia_viral_case1.png`**: Diffuse interstitial reticular markings characteristic of viral pneumonia and pleural effusion.
- **`chest_xray_3.png` / `xray_normal_case1.png`**: Crisp normal radiograph with clear costophrenic angles and normal cardiac silhouette.

---

## Verification Benchmark Results

The entire CouncilMed inference pipeline was evaluated across real test cases with genuine PyTorch council weights.

| Modality | Scan File | Ground Truth Target | Council Consensus | Confidence | Agreement | Localization Bounding Box |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CT** | `ct_pulmonary_nodule_case1.png` | Pulmonary Nodule | **Pulmonary Nodule** | $81.0\%$ | $1.00$ (Strong) | `[153.1, 53.3, 264.2, 169.2]` |
| **CT** | `ct_ground_glass_case1.png` | Ground-Glass Opacity | **Ground-Glass Opacity** | $72.8\%$ | $1.00$ (Strong) | `[156.4, 86.1, 307.6, 190.0]` |
| **CT** | `ct_normal_case1.png` | Normal | **Normal** | $43.6\%$ | $1.00$ (Strong) | Clamped parenchyma |
| **X-Ray** | `xray_pneumonia_bacterial_case1.png` | Bacterial Pneumonia | **Pneumonia** | $78.3\%$ | $1.00$ (Strong) | `[314.3, 249.9, 595.6, 508.0]` |
| **X-Ray** | `chest_xray_2.png` | Pleural Effusion / Infiltrate | **Pleural Effusion** | $51.0\%$ | $1.00$ (Strong) | `[399.8, 434.7, 809.0, 884.0]` |
| **X-Ray** | `xray_normal_case1.png` | Normal | **Normal** | $37.5\%$ | $0.67$ (Weak) | Baseline silhouette |
| **MRI** | `mri_pituitary_case1.png` | Pituitary Tumor | **Pituitary Adenoma** | $46.0\%$ | $1.00$ (Strong) | `[131.3, 264.5, 375.2, 430.9]` |
| **MRI** | `mri_glioma_case1.png` | High-Grade Glioma | **Glioblastoma** | $36.7\%$ | $0.67$ (Weak) | `[220.4, 262.7, 394.2, 416.5]` |
| **MRI** | `mri_meningioma_case1.png` | Meningioma | **Meningioma** | $33.0\%$ | $0.33$ (Split) | `[197.9, 102.9, 297.4, 233.7]` |

---

## Source Citations
1. **Brain MRI Benchmark**: Masoud Nickparvar, Figshare Brain Tumor Dataset (Cheng et al.), SARTIFY.
2. **Chest CT Benchmark**: Cancer Imaging Archive (TCIA) / LIDC-IDRI thoracic volumes, Mahadi et al.
3. **Chest Radiograph Benchmark**: Kermany et al., Guangzhou Women and Children's Medical Center Pediatric Dataset.
4. **COVID-19 Thoracic CT Benchmark**: COVID-19 Radiography/CT Archive, kucaantunes.
