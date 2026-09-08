/**
 * CouncilMed — Backend API Client
 * =================================
 * Type-safe client for the Python backend.
 * Replaces the previous simulated inference with real API calls.
 */

// Types matching backend/api/schemas.py

export interface ModelEvidenceResponse {
  model_id: string;
  model_name: string;
  model_version: string;
  model_architecture: string;
  prediction: string;
  probabilities: Record<string, number>;
  confidence: number;
  uncertainty: number;
  processing_time_ms: number;
  status: 'success' | 'error' | 'unavailable' | 'timeout';
  error_message: string | null;
  bounding_box?: { x: number; y: number; width: number; height: number } | null;
}

export interface CouncilResultResponse {
  council_run_id: string;
  consensus_level: 'strong' | 'moderate' | 'weak' | 'none' | 'insufficient';
  evidence_strength: 'high' | 'moderate' | 'low' | 'insufficient';
  primary_finding: string;
  primary_finding_confidence: number;
  severity: 'clear' | 'info' | 'warning' | 'critical';
  models_agreeing: number;
  models_total: number;
  agreement_score: number;
  ensemble_uncertainty: number;
  requires_human_review: boolean;
  human_review_reasons: string[];
  member_evidence: ModelEvidenceResponse[];
  disagreements: Array<{
    model_id: string;
    model_prediction: string;
    model_confidence: number;
    majority_prediction: string;
    confidence_gap: number;
  }>;
  spatial_agreement: number;
  total_inference_time_ms: number;
}

export interface AgentInsightResponse {
  key_insights: string[];
  clinical_significance: string[];
  recommended_actions: string[];
  missing_information: string[];
  safety_flags: string[];
  evidence_references: string[];
  confidence: number;
  agent_model: string;
  processing_time_ms: number;
}

export interface AnalyzeResponse {
  request_id: string;
  council_result: CouncilResultResponse;
  agent_insight: AgentInsightResponse | null;
  modality: string;
  anatomy: string;
  image_shape: number[];
  total_time_ms: number;
  hardware: Record<string, unknown>;
  errors: string[];
  warnings: string[];
}

export interface ModelInfo {
  model_id: string;
  name: string;
  version: string;
  architecture: string;
  architecture_family: string;
  framework: string;
  modality: string;
  anatomy: string;
  num_classes: number;
  class_names: string[];
  installed: boolean;
  validated: boolean;
  accuracy: number | null;
  auroc: number | null;
}

export interface RegistryResponse {
  total_models: number;
  installed: number;
  validated: number;
  modalities: string[];
  anatomies: string[];
  models: ModelInfo[];
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: number;
  device: string;
  hardware: Record<string, unknown>;
}

export interface AnalysisHistoryItem {
  id: string;
  request_id: string;
  modality: string;
  anatomy: string;
  status: string;
  total_time_ms: number;
  created_at: string;
}

// ==============================
// API Client
// ==============================

const API_BASE = 'http://localhost:8000';

class CouncilMedAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  // Health check
  async health(): Promise<HealthResponse> {
    const res = await fetch(`${this.baseUrl}/api/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return res.json();
  }

  // Analyze via file path (for desktop/Tauri use)
  async analyzeByPath(params: {
    file_path: string;
    study_id?: string;
    modality_override?: string;
    anatomy_override?: string;
    enable_explainability?: boolean;
    enable_agent?: boolean;
  }): Promise<AnalyzeResponse> {
    const res = await fetch(`${this.baseUrl}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_path: params.file_path,
        study_id: params.study_id || null,
        modality_override: params.modality_override || null,
        anatomy_override: params.anatomy_override || null,
        enable_explainability: params.enable_explainability ?? true,
        enable_agent: params.enable_agent ?? false,
      }),
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `Analysis failed: ${res.status}`);
    }

    return res.json();
  }

  // Analyze via file upload (for browser use)
  async analyzeUpload(params: {
    file: File;
    modality?: string;
    anatomy?: string;
    enable_explainability?: boolean;
    enable_agent?: boolean;
  }): Promise<AnalyzeResponse> {
    const formData = new FormData();
    formData.append('file', params.file);

    const queryParams = new URLSearchParams();
    if (params.modality) queryParams.set('modality', params.modality);
    if (params.anatomy) queryParams.set('anatomy', params.anatomy);
    if (params.enable_explainability !== undefined)
      queryParams.set('enable_explainability', String(params.enable_explainability));
    if (params.enable_agent !== undefined)
      queryParams.set('enable_agent', String(params.enable_agent));

    const url = `${this.baseUrl}/api/analyze/upload?${queryParams}`;
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `Upload analysis failed: ${res.status}`);
    }

    return res.json();
  }

  // List registered models
  async getModels(): Promise<RegistryResponse> {
    const res = await fetch(`${this.baseUrl}/api/models`);
    if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
    return res.json();
  }

  // List analysis history
  async getAnalyses(limit = 20, offset = 0): Promise<AnalysisHistoryItem[]> {
    const res = await fetch(
      `${this.baseUrl}/api/analyses?limit=${limit}&offset=${offset}`
    );
    if (!res.ok) throw new Error(`Failed to fetch analyses: ${res.status}`);
    return res.json();
  }

  // Submit human review
  async submitReview(params: {
    analysis_id: string;
    decision: 'accepted' | 'rejected' | 'modified';
    reviewer_name?: string;
    notes?: string;
    modified_finding?: string;
  }): Promise<{ status: string; decision: string }> {
    const res = await fetch(`${this.baseUrl}/api/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    if (!res.ok) throw new Error(`Review submission failed: ${res.status}`);
    return res.json();
  }

  // Generate a report from an analysis
  async generateReport(params: {
    analysis_id: string;
    study_date?: string;
  }): Promise<ReportResponse> {
    const res = await fetch(`${this.baseUrl}/api/reports/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `Report generation failed: ${res.status}`);
    }

    return res.json();
  }

  // Get an existing report
  async getReport(reportId: string): Promise<ReportResponse> {
    const res = await fetch(`${this.baseUrl}/api/reports/${reportId}`);
    if (!res.ok) throw new Error(`Failed to fetch report: ${res.status}`);
    return res.json();
  }
}

// Report types
export interface ReportSectionResponse {
  title: string;
  content: string[];
  subsections: ReportSectionResponse[];
  severity: string;
}

export interface ReportResponse {
  report_id: string;
  version: number;
  study_type: string;
  modality: string;
  anatomy: string;
  study_date: string;
  sections: ReportSectionResponse[];
  impression: string;
  recommendations: string[];
  limitations: string[];
  generated_at: string;
  report_status: string;
  report_text: string;
}

export const api = new CouncilMedAPI();
export default CouncilMedAPI;

