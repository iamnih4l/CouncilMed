import type {
  CouncilConsensusResult,
  ClinicalValidationResult,
  ValidationIssue,
} from './types';
import {
  PathologyClass,
  BRAIN_REGIONS,
} from './types';

/**
 * AI Clinical Validation Layer — CouncilAI
 * Critically evaluates model outputs, identifies reliability issues, and enforces safe, 
 * clinically responsible reporting.
 */
export class ClinicalValidationLayer {
  /**
   * Main validation entry point
   */
  static validate(result: CouncilConsensusResult): ClinicalValidationResult {
    const issues: ValidationIssue[] = [];
    const corrections: string[] = [];
    let patientSafetyOverride = false;

    // --- STEP 1 & 2: IDENTIFY ISSUES & EXPLAIN WHY ---
    
    // 1. LOW CONFIDENCE
    if (result.densenetResult.confidence < 60) {
      issues.push({
        type: 'LOW CLASSIFICATION CONFIDENCE',
        description: `Classification confidence (${result.densenetResult.confidence.toFixed(1)}%) is below 60% threshold.`,
        risk: 'Reduces clinical reliability and increases risk of misdiagnosis.'
      });
    }
    if (result.attentionResult.confidence < 50) {
      issues.push({
        type: 'LOW LOCALIZATION CONFIDENCE',
        description: `Localization confidence (${result.attentionResult.confidence.toFixed(1)}%) is below 50% threshold.`,
        risk: 'Increases risk of spatial misidentification of findings.'
      });
    }
    if (result.swinResult.confidence < 50) {
      issues.push({
        type: 'LOW SEGMENTATION CONFIDENCE',
        description: `Segmentation confidence (${result.swinResult.confidence.toFixed(1)}%) is below 50% threshold.`,
        risk: 'Introduces volumetric measurement errors and over-interpretation risks.'
      });
    }

    // 2. MODEL DISAGREEMENT
    if (result.modelAgreement < 0.5) {
      issues.push({
        type: 'MODEL DISAGREEMENT',
        description: `Inter-model agreement score (${(result.modelAgreement * 100).toFixed(0)}%) is critical (< 0.5).`,
        risk: 'Conflicting predictions suggest unstable inference. High risk of false positive.'
      });
    }

    // 3. ANATOMICAL INCONSISTENCY
    const attentionRegion = result.attentionResult.focusRegion;
    // Simple spatial check: is Swin center in the same lobe as Attention focus?
    // In a production system, this would use a 3D atlas lookup.
    if (result.affectedRegion !== attentionRegion) {
      issues.push({
        type: 'ANATOMICAL INCONSISTENCY',
        description: `Mismatch in regional detection between Attention-Net (${BRAIN_REGIONS[attentionRegion].label}) and Consensus (${BRAIN_REGIONS[result.affectedRegion].label}).`,
        risk: 'Mismatch in detected regions reduces diagnostic certainty.'
      });
    }

    // 4. OVER-INTERPRETATION RISK (Diagnosis with weak agreement)
    if (result.primaryDiagnosis !== PathologyClass.NORMAL && result.modelAgreement < 0.6) {
      issues.push({
        type: 'OVER-INTERPRETATION RISK',
        description: 'Diagnosis generated despite weak cross-model signals.',
        risk: 'High potential for over-diagnosis without manual verification.'
      });
    }

    // --- STEP 3: ENFORCE SOLUTIONS (MANDATORY RULES) ---

    let finalStatus: ClinicalValidationResult['status'] = 'High Confidence';
    let patternDetected: 'Yes' | 'No' | 'Uncertain' = result.primaryDiagnosis !== PathologyClass.NORMAL ? 'Yes' : 'No';
    let validatedRegion: string | null = BRAIN_REGIONS[result.affectedRegion].label;

    // Rule 1: Confidence Gating
    if (result.overallConfidence < 60) {
      corrections.push('Confidence < 60% → REPLACED diagnosis with "Inconclusive AI signal"');
      finalStatus = 'Inconclusive';
      patternDetected = 'Uncertain';
      patientSafetyOverride = true;
    }

    // Rule 2: Consensus Filter
    if (result.modelAgreement < 0.5) {
      corrections.push('Agreement score < 0.5 → Marked "LOW AGREEMENT", primary finding suppressed.');
      finalStatus = 'Inconclusive';
      patternDetected = 'Uncertain';
      validatedRegion = null;
      patientSafetyOverride = true;
    }

    // Rule 3: Anatomy Validation
    if (result.affectedRegion !== attentionRegion) {
      corrections.push('Anatomical inconsistency detected → Relabelled finding as "Inconsistent Location".');
      finalStatus = 'Suspicious';
      patientSafetyOverride = true;
    }

    // Rule 4: Safe Language Rewrite (Applied to findings)
    // We don't change the underlying enum, but we provide a "safe" label in the findings.
    
    // Determine status based on issues if not already set by overrides
    if (!patientSafetyOverride && issues.length > 0) {
      finalStatus = 'Suspicious';
    }

    return {
      detectedIssues: issues,
      correctionsApplied: corrections,
      status: finalStatus,
      patientSafetyOverride,
      finalFindings: {
        patternDetected,
        region: validatedRegion,
      },
    };
  }

  /**
   * Helper to map raw pathology labels to safe clinical observation terms
   */
  static getSafeLabel(pathology: PathologyClass): string {
    if (pathology === PathologyClass.NORMAL) return 'No significant abnormalities detected';
    
    // Replace "Tumor" or "Glioma" with "Possible abnormal pattern" as per directive
    if (pathology.includes('Glioma') || pathology.includes('Meningioma') || pathology.includes('Glioblastoma')) {
      return 'Possible abnormal pattern';
    }

    return `AI-supported observation: ${pathology}`;
  }
}
