export interface RecommendedAction {
  priority: 'High' | 'Medium' | 'Low';
  action: string;
  reason: string;
  expected_impact: string;
}

export interface AdvisorReport {
  summary: string;
  key_findings: string[];
  risk_analysis: string;
  recommended_actions: RecommendedAction[];
  resource_recommendations: string[];
  confidence: 'High' | 'Medium' | 'Low';
  limitations: string[];
}

export interface NaturalLanguageQueryResponse {
  answer: string;
  confidence: 'High' | 'Medium' | 'Low';
  supporting_evidence: string[];
}

const API_BASE_URL = 'http://localhost:8000/api/advisor';

export const advisorApi = {
  async generateReport(scanData: Record<string, unknown>): Promise<AdvisorReport> {
    const response = await fetch(`${API_BASE_URL}/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ scan_data: scanData }),
    });

    if (!response.ok) {
      throw new Error('Failed to generate advisor report');
    }

    return response.json();
  },

  async queryAdvisor(query: string, contextData: Record<string, unknown>): Promise<NaturalLanguageQueryResponse> {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, context_data: contextData }),
    });

    if (!response.ok) {
      throw new Error('Failed to query advisor');
    }

    return response.json();
  }
};
