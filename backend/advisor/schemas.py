from pydantic import BaseModel, Field
from typing import List, Literal

class RecommendedAction(BaseModel):
    priority: Literal["High", "Medium", "Low"] = Field(..., description="Priority level of the action")
    action: str = Field(..., description="The recommended action to take")
    reason: str = Field(..., description="Why this action was suggested")
    expected_impact: str = Field(..., description="The potential benefits and impact of the action")

class AdvisorReportSchema(BaseModel):
    summary: str = Field(..., description="Executive summary of the situation suitable for policymakers")
    key_findings: List[str] = Field(..., description="List of key findings, anomalies, and identified trends")
    risk_analysis: str = Field(..., description="Detailed risk analysis and current trend overview")
    recommended_actions: List[RecommendedAction] = Field(..., description="Practical actions ranked by priority")
    resource_recommendations: List[str] = Field(..., description="Recommendations for resource allocation")
    confidence: Literal["High", "Medium", "Low"] = Field(..., description="Confidence level in this report")
    limitations: List[str] = Field(..., description="Known limitations or potential risks in the assessment")

class NaturalLanguageQueryResponse(BaseModel):
    answer: str = Field(..., description="The direct answer to the user's natural language query")
    confidence: Literal["High", "Medium", "Low"] = Field(..., description="Confidence level of the answer")
    supporting_evidence: List[str] = Field(..., description="Evidence from the data supporting the answer")
