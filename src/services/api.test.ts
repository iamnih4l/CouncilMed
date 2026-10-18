import { describe, it, expect } from 'vitest';
import type { ReportResponse, CouncilResultResponse } from './api';

describe('CouncilMed Clinical Types & Reporting', () => {
  it('validates structured report structure', () => {
    const mockReport: ReportResponse = {
      report_id: 'test-report-1234',
      version: 1,
      study_type: 'Chest X-Ray',
      modality: 'xray',
      anatomy: 'chest',
      study_date: '2026-10-18',
      sections: [
        {
          title: 'Study Information',
          content: ['Modality: xray', 'Anatomy: chest'],
          subsections: [],
          severity: '',
        },
        {
          title: 'AI Council Consensus',
          content: ['Primary Finding: Normal', 'Confidence: 96.5%'],
          subsections: [],
          severity: 'clear',
        },
      ],
      impression: 'No acute cardiopulmonary process identified.',
      recommendations: ['Routine follow-up.'],
      limitations: ['AI-assisted analysis; clinical review required.'],
      generated_at: new Date().toISOString(),
      report_status: 'final',
      report_text: 'COUNCILMED CLINICAL REPORT\nStudy: Chest X-Ray\nFinding: Normal\n',
    };

    expect(mockReport.report_id).toBe('test-report-1234');
    expect(mockReport.modality).toBe('xray');
    expect(mockReport.sections.length).toBe(2);
    expect(mockReport.report_status).toBe('final');
  });

  it('validates council consensus structure', () => {
    const mockCouncil: CouncilResultResponse = {
      council_run_id: 'run-5678',
      consensus_level: 'strong',
      evidence_strength: 'high',
      primary_finding: 'Right Middle Lobe Infiltrate',
      primary_finding_confidence: 0.94,
      severity: 'warning',
      models_agreeing: 3,
      models_total: 3,
      agreement_score: 1.0,
      ensemble_uncertainty: 0.05,
      requires_human_review: true,
      human_review_reasons: ['Focal opacity detected'],
      member_evidence: [
        {
          model_id: 'xray_resnet_v1',
          model_name: 'Chest ResNet-50',
          model_version: '1.0.0',
          model_architecture: 'resnet50',
          prediction: 'Infiltrate',
          probabilities: { Infiltrate: 0.94, Normal: 0.06 },
          confidence: 0.94,
          uncertainty: 0.05,
          processing_time_ms: 45,
          status: 'success',
          error_message: null,
          device: 'cpu',
        },
      ],
      disagreements: [],
      spatial_agreement: 0.88,
      total_inference_time_ms: 135,
    };

    expect(mockCouncil.models_agreeing).toBe(3);
    expect(mockCouncil.consensus_level).toBe('strong');
    expect(mockCouncil.member_evidence[0].device).toBe('cpu');
  });
});
