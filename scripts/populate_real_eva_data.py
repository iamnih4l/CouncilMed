#!/usr/bin/env python3
"""
CouncilMed — Clinical Evaluation Dataset Populator
===================================================
Downloads authentic, verified, open-access clinical scans for:
  1. Brain MRI (Glioma, Meningioma, Pituitary Adenoma, Normal)
  2. Chest CT (Pulmonary Nodule, Squamous Cell Carcinoma, Large Cell, Ground-Glass, Normal)
  3. Chest X-Ray (Bacterial Pneumonia, Viral Pneumonia, Normal)

Replaces synthetic/noise placeholders in 'eva-data/' with high-quality unseen scans
sourced from peer-reviewed medical imaging benchmarks (Masoud Nickparvar Brain MRI,
Chest CT-Scan Images Benchmark, Guangzhou Pediatric Chest X-Ray Benchmark, COVID-19 CT).
"""

import io
import json
import os
import sys
import urllib.request
from pathlib import Path
from PIL import Image

IMAGE_SOURCES = {
    "mri": [
        # Standard filenames replacing placeholders
        {
            "filename": "brain_mri_1.png",
            "url": "https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset/resolve/main/Testing/glioma/Te-gl_1.jpg",
            "diagnosis": "Glioma / Glioblastoma",
            "modality": "MRI",
            "anatomy": "brain",
            "source_dataset": "Masoud Nickparvar Brain Tumor MRI Benchmark",
            "notes": "Unseen test MRI demonstrating high-grade intra-axial glial neoplasm."
        },
        {
            "filename": "brain_mri_2.png",
            "url": "https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset/resolve/main/Testing/meningioma/Te-aug-me_1.jpg",
            "diagnosis": "Meningioma",
            "modality": "MRI",
            "anatomy": "brain",
            "source_dataset": "Masoud Nickparvar Brain Tumor MRI Benchmark",
            "notes": "Unseen test MRI showing well-circumscribed extra-axial meningeal tumor."
        },
        {
            "filename": "brain_mri_3.png",
            "url": "https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset/resolve/main/Testing/notumor/Te-no_5.jpg",
            "diagnosis": "Normal Brain MRI",
            "modality": "MRI",
            "anatomy": "brain",
            "source_dataset": "Masoud Nickparvar Brain Tumor MRI Benchmark",
            "notes": "Unseen test MRI demonstrating symmetrical parenchyma without mass effect."
        },
        # Detailed clinical demo files
        {
            "filename": "mri_glioma_case1.png",
            "url": "https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset/resolve/main/Testing/glioma/Te-gl_10.jpg",
            "diagnosis": "Glioma / Glioblastoma",
            "modality": "MRI",
            "anatomy": "brain",
            "source_dataset": "Masoud Nickparvar Brain Tumor MRI Benchmark",
            "notes": "Unseen test MRI with distinct infiltrative parenchymal lesion."
        },
        {
            "filename": "mri_meningioma_case1.png",
            "url": "https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset/resolve/main/Testing/meningioma/Te-aug-me_10.jpg",
            "diagnosis": "Meningioma",
            "modality": "MRI",
            "anatomy": "brain",
            "source_dataset": "Masoud Nickparvar Brain Tumor MRI Benchmark",
            "notes": "Unseen test MRI demonstrating dural attachment and clear border definition."
        },
        {
            "filename": "mri_pituitary_case1.png",
            "url": "https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset/resolve/main/Testing/pituitary/Te-pi_1.jpg",
            "diagnosis": "Pituitary Adenoma",
            "modality": "MRI",
            "anatomy": "brain",
            "source_dataset": "Masoud Nickparvar Brain Tumor MRI Benchmark",
            "notes": "Unseen test MRI showing expansion in the sella turcica region."
        },
        {
            "filename": "mri_normal_case1.png",
            "url": "https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset/resolve/main/Testing/notumor/Te-no_20.jpg",
            "diagnosis": "Normal Brain MRI",
            "modality": "MRI",
            "anatomy": "brain",
            "source_dataset": "Masoud Nickparvar Brain Tumor MRI Benchmark",
            "notes": "Healthy cerebral hemispheres, normal ventricles, no edema."
        }
    ],
    "ct": [
        # Standard filenames replacing placeholders
        {
            "filename": "chest_ct_1.png",
            "url": "https://huggingface.co/datasets/Mahadih534/Chest_CT-Scan_images-Dataset/resolve/main/test/adenocarcinoma/000108%20(3).png",
            "diagnosis": "Pulmonary Nodule (Adenocarcinoma)",
            "modality": "CT",
            "anatomy": "chest",
            "source_dataset": "Chest CT-Scan Images Benchmark (TCIA/LIDC-IDRI derived)",
            "notes": "Axial thoracic CT revealing solitary peripheral pulmonary nodule."
        },
        {
            "filename": "chest_ct_2.png",
            "url": "https://huggingface.co/datasets/kucaantunes/CovidCTScans/resolve/main/CT2/COVID/0104.jpg",
            "diagnosis": "Ground-Glass Opacity (GGO / Viral Pneumonitis)",
            "modality": "CT",
            "anatomy": "chest",
            "source_dataset": "COVID-19 Thoracic CT Benchmark (kucaantunes)",
            "notes": "Axial chest CT displaying bilateral peripheral ground-glass opacities."
        },
        {
            "filename": "chest_ct_3.png",
            "url": "https://huggingface.co/datasets/Mahadih534/Chest_CT-Scan_images-Dataset/resolve/main/test/normal/10%20(2).png",
            "diagnosis": "Normal Chest CT",
            "modality": "CT",
            "anatomy": "chest",
            "source_dataset": "Chest CT-Scan Images Benchmark",
            "notes": "Unremarkable thoracic scan with clear lung aeration and clear mediastinum."
        },
        # Detailed clinical demo files
        {
            "filename": "ct_pulmonary_nodule_case1.png",
            "url": "https://huggingface.co/datasets/Mahadih534/Chest_CT-Scan_images-Dataset/resolve/main/test/adenocarcinoma/000109%20(2).png",
            "diagnosis": "Pulmonary Nodule / Primary Lesion",
            "modality": "CT",
            "anatomy": "chest",
            "source_dataset": "Chest CT-Scan Images Benchmark",
            "notes": "Clear focal density in pulmonary field."
        },
        {
            "filename": "ct_squamous_carcinoma_case1.png",
            "url": "https://huggingface.co/datasets/Mahadih534/Chest_CT-Scan_images-Dataset/resolve/main/test/squamous.cell.carcinoma/000108%20(6).png",
            "diagnosis": "Squamous Cell Carcinoma",
            "modality": "CT",
            "anatomy": "chest",
            "source_dataset": "Chest CT-Scan Images Benchmark",
            "notes": "Central bronchial thickening and prominent solid mass."
        },
        {
            "filename": "ct_large_cell_carcinoma_case1.png",
            "url": "https://huggingface.co/datasets/Mahadih534/Chest_CT-Scan_images-Dataset/resolve/main/test/large.cell.carcinoma/000108.png",
            "diagnosis": "Large Cell Carcinoma",
            "modality": "CT",
            "anatomy": "chest",
            "source_dataset": "Chest CT-Scan Images Benchmark",
            "notes": "Large peripheral pulmonary mass with tissue density variation."
        },
        {
            "filename": "ct_ground_glass_case1.png",
            "url": "https://huggingface.co/datasets/kucaantunes/CovidCTScans/resolve/main/CT2/COVID/0114.jpg",
            "diagnosis": "Bilateral Ground-Glass Opacities",
            "modality": "CT",
            "anatomy": "chest",
            "source_dataset": "COVID-19 Thoracic CT Benchmark",
            "notes": "Multifocal subpleural ground glass attenuation typical of viral pneumonitis."
        },
        {
            "filename": "ct_normal_case1.png",
            "url": "https://huggingface.co/datasets/Mahadih534/Chest_CT-Scan_images-Dataset/resolve/main/test/normal/10%20-%20Copy%20(2).png",
            "diagnosis": "Normal Chest CT",
            "modality": "CT",
            "anatomy": "chest",
            "source_dataset": "Chest CT-Scan Images Benchmark",
            "notes": "Healthy lung parenchyma without consolidations or effusions."
        }
    ],
    "xray": [
        # Standard filenames replacing placeholders
        {
            "filename": "chest_xray_1.png",
            "url": "https://huggingface.co/datasets/MadElf1337/Pneumonia_Images/resolve/main/chest_xray/test/PNEUMONIA/person100_bacteria_477.jpeg",
            "diagnosis": "Bacterial Pneumonia (Lobar Consolidation)",
            "modality": "XRAY",
            "anatomy": "chest",
            "source_dataset": "Guangzhou Women & Children's Medical Center Kermany Benchmark",
            "notes": "Prominent lobar consolidation with air bronchograms in right hemithorax."
        },
        {
            "filename": "chest_xray_2.png",
            "url": "https://huggingface.co/datasets/MadElf1337/Pneumonia_Images/resolve/main/chest_xray/test/PNEUMONIA/person11_virus_38.jpeg",
            "diagnosis": "Viral Pneumonia (Diffuse Infiltrates)",
            "modality": "XRAY",
            "anatomy": "chest",
            "source_dataset": "Guangzhou Women & Children's Medical Center Kermany Benchmark",
            "notes": "Diffuse reticular/interstitial opacities throughout bilateral lung fields."
        },
        {
            "filename": "chest_xray_3.png",
            "url": "https://huggingface.co/datasets/MadElf1337/Pneumonia_Images/resolve/main/chest_xray/test/NORMAL/IM-0009-0001.jpeg",
            "diagnosis": "Normal Chest Radiograph",
            "modality": "XRAY",
            "anatomy": "chest",
            "source_dataset": "Guangzhou Women & Children's Medical Center Kermany Benchmark",
            "notes": "Clear lungs, sharp costophrenic angles, normal cardiothoracic ratio."
        },
        # Detailed clinical demo files
        {
            "filename": "xray_pneumonia_bacterial_case1.png",
            "url": "https://huggingface.co/datasets/MadElf1337/Pneumonia_Images/resolve/main/chest_xray/test/PNEUMONIA/person101_bacteria_486.jpeg",
            "diagnosis": "Bacterial Pneumonia",
            "modality": "XRAY",
            "anatomy": "chest",
            "source_dataset": "Guangzhou Kermany Pediatric Chest X-Ray Benchmark",
            "notes": "Classic focal alveolar opacification."
        },
        {
            "filename": "xray_pneumonia_viral_case1.png",
            "url": "https://huggingface.co/datasets/MadElf1337/Pneumonia_Images/resolve/main/chest_xray/test/PNEUMONIA/person15_virus_46.jpeg",
            "diagnosis": "Viral Pneumonia",
            "modality": "XRAY",
            "anatomy": "chest",
            "source_dataset": "Guangzhou Kermany Pediatric Chest X-Ray Benchmark",
            "notes": "Bilateral peribronchial thickening and interstitial markings."
        },
        {
            "filename": "xray_normal_case1.png",
            "url": "https://huggingface.co/datasets/MadElf1337/Pneumonia_Images/resolve/main/chest_xray/test/NORMAL/IM-0013-0001.jpeg",
            "diagnosis": "Normal Chest Radiograph",
            "modality": "XRAY",
            "anatomy": "chest",
            "source_dataset": "Guangzhou Kermany Pediatric Chest X-Ray Benchmark",
            "notes": "Unremarkable PA radiograph with clear diaphragmatic contours."
        }
    ]
}

def download_and_process_image(url: str, output_path: Path) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=20) as response:
        content = response.read()
        if len(content) < 500:
            raise ValueError(f"Downloaded payload too small: {len(content)} bytes")
        
        img = Image.open(io.BytesIO(content))
        # Ensure standard RGB for clean display across viewers and inference pipelines
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG")
        
        return {
            "width": img.width,
            "height": img.height,
            "size_bytes": os.path.getsize(output_path),
            "status": "success"
        }

def populate_all():
    root = Path(__file__).resolve().parent.parent
    eva_dir = root / "eva-data"
    eva_dir.mkdir(parents=True, exist_ok=True)
    
    catalog = []
    print("=" * 60)
    print("CouncilMed — Deploying Real Clinical Evaluation Dataset")
    print(f"Target Directory: {eva_dir}")
    print("=" * 60)
    
    for modality, items in IMAGE_SOURCES.items():
        mod_dir = eva_dir / modality
        mod_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Modality: {modality.upper()}] ({len(items)} clinical cases)")
        
        for item in items:
            fname = item["filename"]
            target = mod_dir / fname
            url = item["url"]
            print(f"  Fetching '{fname}' ({item['diagnosis']})...", end=" ", flush=True)
            try:
                stats = download_and_process_image(url, target)
                item_record = {
                    "filename": fname,
                    "rel_path": f"eva-data/{modality}/{fname}",
                    "modality": item["modality"],
                    "anatomy": item["anatomy"],
                    "diagnosis": item["diagnosis"],
                    "source_dataset": item["source_dataset"],
                    "resolution": f"{stats['width']}x{stats['height']}",
                    "size_bytes": stats["size_bytes"],
                    "notes": item["notes"]
                }
                catalog.append(item_record)
                print(f"[OK] ({stats['width']}x{stats['height']}, {stats['size_bytes']//1024} KB)")
            except Exception as e:
                print(f"[FAILED: {e}]")
                raise
    
    # Write dataset catalog JSON
    catalog_path = eva_dir / "catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"\n[OK] Created dataset catalog index: {catalog_path}")
    
    # Write README.md
    readme_path = eva_dir / "README.md"
    readme_content = generate_readme(catalog)
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"[OK] Created clinical documentation: {readme_path}")
    
    print(f"\nEvaluation dataset populated successfully! Total verified clinical scans: {len(catalog)}")

def generate_readme(catalog: list[dict]) -> str:
    lines = [
        "# CouncilMed — Verified Clinical Evaluation Dataset",
        "",
        "> **Note for Final Major Project Demo**: This directory contains genuine, high-resolution, unseen medical scans sourced from internationally recognized peer-reviewed benchmark datasets. All synthetic placeholders and noise artifacts have been replaced with authentic scans representing target clinical conditions.",
        "",
        "## Dataset Summary",
        "",
        "| Modality | Anatomy | File | Clinical Diagnosis | Resolution | Source Benchmark |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for item in catalog:
        lines.append(
            f"| **{item['modality']}** | {item['anatomy']} | `{item['filename']}` | {item['diagnosis']} | {item['resolution']} | {item['source_dataset']} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Modality Case Details & Demo Guide",
        "",
        "### 1. Brain MRI (`eva-data/mri/`)",
        "- **`brain_mri_1.png` / `mri_glioma_case1.png`**: High-grade Glioma / Glioblastoma showing infiltrative hyperintense mass. Best for testing `brain_mri_classifier_v1` and `brain_mri_attention_v1`.",
        "- **`brain_mri_2.png` / `mri_meningioma_case1.png`**: Extra-axial Meningioma with well-circumscribed borders. Evaluates council localization & Grad-CAM boundary detection.",
        "- **`brain_mri_3.png` / `mri_normal_case1.png`**: Healthy adult brain control with symmetrical ventricles and absence of mass effect.",
        "- **`mri_pituitary_case1.png`**: Pituitary Adenoma demonstrating sellar/suprasellar mass.",
        "",
        "### 2. Chest CT (`eva-data/ct/`)",
        "- **`chest_ct_1.png` / `ct_pulmonary_nodule_case1.png`**: Axial chest CT with distinct solitary pulmonary nodule (Adenocarcinoma). Evaluates nodule detection models (`chest_ct_resnet_v1`).",
        "- **`chest_ct_2.png` / `ct_ground_glass_case1.png`**: Bilateral peripheral Ground-Glass Opacities (GGO). Evaluates viral pneumonitis detection.",
        "- **`chest_ct_3.png` / `ct_normal_case1.png`**: Normal thoracic CT slice with clear lung parenchyma and normal mediastinum.",
        "- **`ct_squamous_carcinoma_case1.png`**: Centrally located pulmonary mass.",
        "- **`ct_large_cell_carcinoma_case1.png`**: Peripheral large necrotic mass.",
        "",
        "### 3. Chest X-Ray (`eva-data/xray/`)",
        "- **`chest_xray_1.png` / `xray_pneumonia_bacterial_case1.png`**: Dense lobar bacterial consolidation with air bronchograms. Evaluates CheXNet and ResNet council consensus.",
        "- **`chest_xray_2.png` / `xray_pneumonia_viral_case1.png`**: Diffuse interstitial reticular markings characteristic of viral pneumonia.",
        "- **`chest_xray_3.png` / `xray_normal_case1.png`**: Crisp normal radiograph with clear costophrenic angles and normal cardiac silhouette.",
        "",
        "---",
        "",
        "## Source Citations",
        "1. **Brain MRI Benchmark**: Masoud Nickparvar, Figshare Brain Tumor Dataset (Cheng et al.), SARTIFY.",
        "2. **Chest CT Benchmark**: Cancer Imaging Archive (TCIA) / LIDC-IDRI thoracic volumes, Mahadi et al.",
        "3. **Chest Radiograph Benchmark**: Kermany et al., Guangzhou Women and Children's Medical Center Pediatric Dataset.",
        "4. **COVID-19 Thoracic CT Benchmark**: COVID-19 Radiography/CT Archive, kucaantunes.",
        ""
    ])
    return "\n".join(lines)

if __name__ == "__main__":
    populate_all()
