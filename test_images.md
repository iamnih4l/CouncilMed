# CouncilMed — Test Data Resources

## Local Test Images

Pre-generated medical scan images are available in the `test_data/` directory for quick testing:

| File | Modality | Description | Expected Diagnosis |
|---|---|---|---|
| `mri_scan.png` | MRI | Brain MRI (axial view, normal anatomy) | Normal |
| `ct_scan.png` | CT | Abdominal CT scan (normal anatomy) | Clear / No Anomalies |
| `xray_scan.png` | X-Ray | Skeletal X-ray (normal anatomy) | No Fracture Detected |

### How to Use

1. Start the application: `npm run dev`
2. Navigate to the appropriate analysis page (MRI, CT, or X-Ray).
3. Click **Upload** and select the corresponding image from `test_data/`.
4. The image is uploaded to the backend (`POST /api/inference`), processed through the PyTorch pipeline, and the Council result is displayed.

## Clinical Datasets (Advanced Testing)

For validation with real medical data, consider these official repositories:

### Brain MRI
1. **BraTS (Brain Tumor Segmentation)**: The gold standard for brain tumor AI.
   - [BraTS on TCIA](https://www.cancerimagingarchive.net/collection/brats/)
2. **OASIS (Open Access Series of Imaging Studies)**: Longitudinal MRI data.
   - [OASIS Brains](https://www.oasis-brains.org/)
3. **IXI Dataset**: ~600 MR images from healthy subjects.
   - [IXI Dataset](https://brain-development.org/ixi-dataset/)

### Chest / Abdominal CT
4. **RSNA Intracranial Hemorrhage Detection**: ~700K images.
   - [RSNA on Kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)
5. **Medical Segmentation Decathlon (MSD)**: Multi-organ CT segmentation.
   - [MSD Challenge](http://medicaldecathlon.com/)

### Skeletal X-Ray
6. **MURA (Musculoskeletal Radiographs)**: 40K+ upper extremity X-rays.
   - [MURA on Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/)
7. **FracAtlas**: Fracture classification dataset.
   - [FracAtlas on GitHub](https://github.com/BoneFractureDetection/FracAtlas)

## Testing Protocol

### 1. Baseline (Normal Scans)
Upload the provided normal `mri_scan.png`, `ct_scan.png`, and `xray_scan.png`. Verify that:
- **DenseNet-121** classifies as the "Normal" class for each modality.
- **Grad-CAM** attention is diffuse (no focal hot spots).
- **U-Net** segmentation shows minimal activated voxels.
- **Council consensus** shows `severity: clear`.

### 2. Pathology Test (Abnormal Scans)
Download a pathological scan from one of the datasets above. Verify that:
- **DenseNet-121** identifies the correct pathology class with elevated confidence.
- **Grad-CAM** attention focuses on the lesion region.
- **U-Net** produces a non-trivial segmentation mask with measurable volume.
- **Council notes** describe the findings accurately.

### 3. Report Validation
After analysis, click **Generate PDF Report** and verify:
- All anatomical findings map correctly to the clinical report layout.
- Confidence scores, severity levels, and region labels are present and coherent.
- The PDF downloads successfully.

### 4. AI Advisor Test
Navigate to the **AI Advisor** page (`/advisor`) and:
- Click **Regenerate Report** to generate an executive summary from the context data.
- Ask questions like "What is the highest risk region?" in the Query Engine.
- Verify the LLM responds with structured JSON containing actionable recommendations.

> **Note**: The AI Advisor requires a local LLM server (Ollama or LM Studio) to be running.
