import type { 
  CouncilConsensusResult, 
  RadiologyReportData,
} from '../inference/types';
import { BRAIN_REGIONS } from '../inference/types';

/**
 * Radiology Report Generator — CouncilAI
 * Generates structured, RSNA-style clinical reports based on multi-model AI inference.
 */
export class RadiologyReportGenerator {
  /**
   * Main entry point for report synthesis
   */
  static generate(result: CouncilConsensusResult): RadiologyReportData {
    const { primaryDiagnosis, overallConfidence, modelAgreement, affectedRegion, swinResult } = result;

    // 1. Determine Study Type & Quality (Defaults for now)
    const studyType: RadiologyReportData['studyType'] = 'FLAIR'; // Common for tumor detection
    const dataQuality: RadiologyReportData['dataQuality'] = 'Adequate';

    // 2. Bone Parenchyma
    const regionLabel = BRAIN_REGIONS[affectedRegion].label;
    let brainParenchyma = 'No definitive abnormal signal identified within limits of available data.';
    if (primaryDiagnosis !== 'Normal' && overallConfidence >= 60) {
      brainParenchyma = `Abnormal signal pattern detected within the ${regionLabel}.`;
    }

    // 3. Lesion Assessment (Confidence gating: ≥60%)
    const lesionIncluded = primaryDiagnosis !== 'Normal' && overallConfidence >= 60;
    const lesionAssessment = {
      included: lesionIncluded,
      location: lesionIncluded ? regionLabel : null,
      sizeVolume: lesionIncluded ? `${swinResult.tumorVolumeMm3.toFixed(0)} mm³` : null,
      characteristics: lesionIncluded 
        ? `Hyperintense signal on ${studyType} suggestive of ${primaryDiagnosis}.`
        : 'Inconclusive features.',
    };

    // 4. Edema / Mass Effect (Heuristic based on Swin segmentation)
    const hasEdema = swinResult.segmentedRegions.some(r => r.label.toLowerCase().includes('edema') && r.volumePercentage > 10);
    const edemaMassEffect: RadiologyReportData['findings']['edemaMassEffect'] = 
      hasEdema ? 'Present' : (overallConfidence < 50 ? 'Uncertain' : 'Absent');

    // 5. Ventricular System & Midline (Heuristics)
    const ventricularSystem: RadiologyReportData['findings']['ventricularSystem'] = 
      swinResult.tumorVolumeMm3 > 500 ? 'Compressed' : 'Normal';
    const midlineShift: RadiologyReportData['findings']['midlineStructures']['shift'] = 
      swinResult.tumorVolumeMm3 > 1000 ? 'Yes' : 'No';

    // 6. Impression (STRICT CONSERVATIVE LOGIC)
    let impression = '';
    if (overallConfidence >= 75 && modelAgreement >= 0.8) {
      impression = `Findings are suggestive of ${primaryDiagnosis}, however correlation with clinical and radiological expertise is required.`;
    } else if (overallConfidence >= 60) {
      impression = `There is a suspicious pattern that may represent ${primaryDiagnosis}, but findings are not definitive.`;
    } else {
      impression = 'No reliable abnormality detected. Current AI analysis is inconclusive.';
    }

    // Hard Rule: NEVER output definitive diagnosis under 75% confidence
    // We already handled this in the IF/ELSE above.

    return {
      modality: 'MRI Brain',
      studyType,
      dataQuality,
      clinicalIndication: 'Not specified',
      technique: 'Axial MRI sequences reviewed (FLAIR/T2 weighted reconstructions).',
      findings: {
        brainParenchyma,
        lesionAssessment,
        edemaMassEffect,
        ventricularSystem,
        midlineStructures: {
          shift: midlineShift,
        },
      },
      impression,
      limitations: [
        'This analysis is based on limited AI interpretation.',
        'Single-image or incomplete scan reduces diagnostic reliability.',
        'Subtle pathologies may not be detected.',
      ],
      recommendations: [
        'Correlation with full MRI study is advised.',
        'Consultation with a certified radiologist is strongly recommended.',
      ],
    };
  }
}
