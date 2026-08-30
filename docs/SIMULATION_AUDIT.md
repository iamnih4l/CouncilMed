# CouncilMed — Simulation Audit

> **Audit Date:** 2026-08-10
> **Purpose:** Catalog every instance of simulated, hardcoded, placeholder, or fabricated data in the production inference path.

---

## Critical Simulations (Inference Path)

These are the primary simulations that make CouncilMed's diagnostic output scientifically invalid.

### 1. DenseNet-121: Model Output Fabrication

**File:** `src/services/inference/models/DenseNet121.ts`
**Lines:** 68-133

The loaded model (`_model`) is explicitly ignored. Diagnostic probabilities are constructed from pixel brightness heuristics and `Math.random()`:

```
Line 88:  const probabilities = new Array(4).fill(0.01);
Line 94:  probabilities[0] = 0.85 + (Math.random() * 0.1);
Line 100: probabilities[1] = 0.75 + (Math.random() * 0.1);
Line 105: probabilities[2] = 0.92 + (Math.random() * 0.05);
Line 123: const inferenceTimeMs = performance.now() - startTime + 80; // +80ms fake latency
```

**Impact:** The classifier never runs. All diagnoses are made by checking if pixels are bright.

---

### 2. Attention-Net: Spatial Localization Fabrication

**File:** `src/services/inference/models/AttentionNet.ts`
**Lines:** 146-249

The loaded model is ignored. The "attention heatmap" is created by resizing the input to 14x14 and averaging RGB channels. Confidence is random:

```
Line 191: const confidence = stats.isAbnormal ? 80 + (Math.random() * 15) : 10 + (Math.random() * 20);
```

**Impact:** No learned spatial attention exists. The "focus region" is determined by which bright pixel cluster is nearest to a predefined anatomical center.

---

### 3. Swin-UNETR: Segmentation Mask Fabrication

**File:** `src/services/inference/models/SwinUNETR.ts`
**Lines:** 175-286

The loaded model is ignored. The "3D segmentation mask" is fabricated by:
1. Resizing image to 8x8
2. Extruding it into 3D with depth falloff: `const depthWeight = 1 - Math.abs(z - 4) / 4;`
3. Generating random max probability: `0.75 + (Math.random() * 0.2)`
4. Computing fake tumor volume: `solidVoxels * 15.6` (arbitrary mm³ per voxel)

Hardcoded segmentation labels:
```
Line 255-265:
  { label: 'Enhancing Tumor', volumePercentage: maxProb * 45 },
  { label: 'Tumor Core', volumePercentage: maxProb * 30 },
  { label: 'Peritumoral Edema', volumePercentage: maxProb * 25 },
```

**Impact:** No volumetric segmentation occurs. A 2D image is extruded and presented as 3D analysis.

---

## Secondary Simulations (UI / Dashboard)

### 4. Dashboard Hardcoded Statistics

**File:** `src/pages/Dashboard.tsx`

```
Line 38: const scansProcessed = 1284 + (result ? 1 : 0);
Line 39: const anomaliesDetected = 347 + (result && result.primaryDiagnosis !== 'Normal' ? 1 : 0);
Line 45: : 'ResNet-Diag-v4';   // Fake model name
Line 128: <h3>42</h3>           // Fake active radiologist count
```

Pie chart with static values:
```
Line 11-15:
  { name: 'Clear', value: 65, color: '#10b981' },
  { name: 'Anomalies', value: 25, color: '#f59e0b' },
  { name: 'Critical', value: 10, color: '#ef4444' },
```

---

### 5. Mock Scan Logs

**File:** `src/components/dashboard/ScanLogsTable.tsx`

```
Line 19-25:
  { id: 'PT-8801', modality: 'MRI', finding: 'Glioblastoma Suspicion', models: 'viT-Med-Alpha, ResNet-Diagnostic-v4', conf: 92.4 },
  { id: 'PT-8802', modality: 'CT', finding: 'Clear / No Anomalies', models: 'DenseNet-Abd, ResNet-Diagnostic-v4', conf: 98.1 },
  { id: 'PT-8803', modality: 'X-Ray', finding: 'Hairline Fracture - Tibia', models: 'viT-Med-Alpha', conf: 76.5 },
  { id: 'PT-8804', modality: 'MRI', finding: 'Meningioma Detected', models: 'Swin-UNETR, Attention-Net', conf: 88.2 },
  { id: 'PT-8805', modality: 'CT', finding: 'Moderate Edema', models: 'DenseNet-121', conf: 81.0 },
```

These reference model names that don't exist in the system (`viT-Med-Alpha`, `ResNet-Diagnostic-v4`, `DenseNet-Abd`).

---

### 6. Backend Hardcoded Values

**File:** `backend/main.py`

```
Line 72: severity="Calculated",              # Never actually calculated
Line 75: overall_agreement_score=1.0,         # Hardcoded to 100% agreement
Line 76: requires_manual_review=False          # Always false
```

---

### 7. Report Generator Hardcoded Values

**File:** `src/services/reporting/RadiologyReportGenerator.ts`

```
Line 19: const studyType = 'FLAIR';           // Hardcoded, always FLAIR
Line 66: modality: 'MRI Brain',               // Hardcoded, ignores actual modality
Line 69: technique: 'Axial MRI sequences reviewed (FLAIR/T2 weighted reconstructions).',
```

**File:** `src/utils/reportGenerator.ts`

```
Line 73: pdf.text(`REPORT ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()}`, ...);
```
Report IDs are random, not traceable.

---

### 8. Random Patient ID Generation

**File:** `src/hooks/usePersistentReports.ts`
```
Line 34: id: `PT-${Math.floor(Math.random() * 9000) + 1000}`,
```

**File:** `src/components/dashboard/ScanLogsTable.tsx`
```
Line 40: id: `PT-${Math.floor(Math.random() * 9000) + 1000}`,
```

---

## Summary Table

| # | File | Line(s) | Simulation Type | Severity |
|---|---|---|---|---|
| 1 | DenseNet121.ts | 88-123 | Model output fabrication | CRITICAL |
| 2 | AttentionNet.ts | 158-248 | Attention/localization fabrication | CRITICAL |
| 3 | SwinUNETR.ts | 186-285 | Segmentation mask fabrication | CRITICAL |
| 4 | Dashboard.tsx | 38-39, 128 | Hardcoded dashboard stats | MEDIUM |
| 5 | Dashboard.tsx | 11-15 | Hardcoded pie chart | MEDIUM |
| 6 | ScanLogsTable.tsx | 19-25 | Mock scan log entries | MEDIUM |
| 7 | main.py | 72, 75-76 | Hardcoded backend values | HIGH |
| 8 | RadiologyReportGenerator.ts | 19, 66, 69 | Hardcoded report metadata | MEDIUM |
| 9 | reportGenerator.ts | 73 | Random report IDs | LOW |
| 10 | usePersistentReports.ts | 34 | Random patient IDs | LOW |
| 11 | ScanLogsTable.tsx | 40 | Random patient IDs | LOW |
| 12 | DenseNet121.ts | 123 | Fake latency (+80ms) | LOW |

---

## Remediation Plan

1. **CRITICAL:** Connect `runDenseNet121Inference()` to actually call the loaded TF.js model
2. **CRITICAL:** Connect `runAttentionNetInference()` to actually call the loaded TF.js model
3. **CRITICAL:** Connect `runSwinUNETRInference()` to actually call the loaded TF.js model (or replace with proper segmentation)
4. **CRITICAL:** Implement genuine Grad-CAM for explainability
5. **HIGH:** Replace backend hardcoded values with computed values from actual inference
6. **MEDIUM:** Replace dashboard mock data with real statistics from the database
7. **MEDIUM:** Replace mock scan logs with real audit trail data
8. **MEDIUM:** Generate deterministic report/patient IDs (UUID)
9. **LOW:** Remove fake latency additions
