from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

# --- Incoming Schemas ---
class DiagnosticRequest(BaseModel):
    metadata: Dict[str, Any]
    densenet_outputs: Dict[str, Any]
    attention_outputs: Dict[str, Any]
    swin_unetr_outputs: Dict[str, Any]

# --- Response Schemas ---
class MLInferenceSchema(BaseModel):
    id: str
    model_name: str
    raw_output: Dict[str, Any]
    confidence_score: float
    execution_time_ms: int

    class Config:
        from_attributes = True

class CouncilReportSchema(BaseModel):
    id: str
    primary_diagnosis: str
    severity: str
    spatial_findings: Dict[str, Any]
    agent_reasoning_transcript: str
    overall_agreement_score: float
    requires_manual_review: bool
    generated_at: datetime

    class Config:
        from_attributes = True

class ScanResponseSchema(BaseModel):
    id: str
    patient_id: Optional[str]
    modality: str
    scan_date: datetime
    status: str
    inferences: List[MLInferenceSchema] = []
    report: Optional[CouncilReportSchema] = None

    class Config:
        from_attributes = True
