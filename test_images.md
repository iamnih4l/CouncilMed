# CouncilMed MRI Test Resources

This document provides links to medical-grade brain MRI images that can be used to test the CouncilMed multi-model diagnostic pipeline.

## Direct Image Links (For Quick Testing)

Use these URLs in the "Upload MRI" section of the application.

| Image Type | Description | URL |
| :--- | :--- | :--- |
| **T1-Weighted** | Normal brain anatomy, high structural detail. | [Sample T1](https://mrimaster.com/images/anatomy/brain/mri%20brain%20axial%20anatomy%20(1).jpg) |
| **T2-Weighted** | Good for detecting edema and CSF. | [Sample T2](https://mrimaster.com/images/anatomy/brain/mri%20brain%20axial%20t2%20(1).jpg) |
| **FLAIR** | Suppresses CSF, excellent for lesion detection. | [Sample FLAIR](https://mrimaster.com/images/pathology/brain/flair%20brain%20(1).jpg) |
| **Glioblastoma** | High-grade glioma (Grade IV). | [Pathology Sample](https://radiopaedia.org/uploads/radio/0000/0173/GBM-axial-T1-contrast.jpg) |
| **Meningioma** | Extra-axial tumor (Grade I). | [Pathology Sample 2](https://radiopaedia.org/uploads/radio/0000/0174/Meningioma-axial-T1-contrast.jpg) |

## Clinical Datasets (Bulk Testing)

For extensive validation, consider downloading slices from these official repositories:

1.  **BraTS (Brain Tumor Segmentation)**: The gold standard for AI training.
    *   [BraTS on TCIA](https://www.cancerimagingarchive.net/collection/brats/)
2.  **OASIS (Open Access Series of Imaging Studies)**: Longitudinal MRI data.
    *   [OASIS Brains](https://www.oasis-brains.org/)
3.  **IXI Dataset**: Nearly 600 MR images from normal, healthy subjects.
    *   [IXI Dataset](https://brain-development.org/ixi-dataset/)

## Testing Protocol

1.  **Baseline**: Upload a "Normal" T1/T2 scan. The Council should reach consensus on "Normal" pathology with high agreement.
2.  **Pathology Stress Test**: Upload the Glioblastoma sample. Verify that:
    *   **DenseNet** identifies High-Grade Glioma.
    *   **Attention-Net** localizes the ROI in the correct hemisphere.
    *   **Swin-UNETR** provides a volumetric estimation.
3.  **Reporting**: Generate the PDF report after analysis and verify that all anatomical findings are correctly mapped to the clinical layout.
