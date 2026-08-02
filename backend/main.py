from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
import uvicorn
from typing import List

from agents.orchestrator import run_council_debate
from database import init_db, get_db
from models import Scan, MLInference, CouncilReport
from schemas import DiagnosticRequest, ScanResponseSchema

app = FastAPI(title="CouncilMed Multi-Agent Backend with Database")

# Allow requests from the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    await init_db()

@app.post("/api/analyze", response_model=ScanResponseSchema)
async def analyze_scan(request: DiagnosticRequest, db: AsyncSession = Depends(get_db)):
    try:
        # 1. Trigger the LangGraph multi-agent workflow
        llm_report_text = run_council_debate(
            metadata=request.metadata,
            densenet=request.densenet_outputs,
            attention=request.attention_outputs,
            swin=request.swin_unetr_outputs
        )
        
        # 2. Persist the Scan
        new_scan = Scan(modality=request.metadata.get("modality", "MRI"))
        db.add(new_scan)
        await db.flush() # Get scan ID

        # 3. Persist individual ML outputs
        inferences = [
            MLInference(
                scan_id=new_scan.id,
                model_name="DenseNet-121",
                raw_output=request.densenet_outputs,
                confidence_score=request.densenet_outputs.get('confidence', 0)
            ),
            MLInference(
                scan_id=new_scan.id,
                model_name="Attention-Net",
                raw_output=request.attention_outputs,
                confidence_score=request.attention_outputs.get('confidence', 0)
            ),
            MLInference(
                scan_id=new_scan.id,
                model_name="Swin-UNETR",
                raw_output=request.swin_unetr_outputs,
                confidence_score=request.swin_unetr_outputs.get('confidence', 0)
            )
        ]
        db.add_all(inferences)

        # 4. Persist the Council Report
        report = CouncilReport(
            scan_id=new_scan.id,
            primary_diagnosis=request.densenet_outputs.get('primaryDiagnosis', 'Unknown'),
            severity="Calculated", # This would map to the logic
            spatial_findings={"focus": request.attention_outputs.get('focusCenter')},
            agent_reasoning_transcript=llm_report_text,
            overall_agreement_score=1.0, # Placeholder
            requires_manual_review=False
        )
        db.add(report)
        
        await db.commit()
        await db.refresh(new_scan)
        
        # Explicit load of relations for the response model
        stmt = select(Scan).where(Scan.id == new_scan.id).options(
            selectinload(Scan.inferences),
            selectinload(Scan.report)
        )
        result = await db.execute(stmt)
        return result.scalar_one()

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scans", response_model=List[ScanResponseSchema])
async def get_scans(db: AsyncSession = Depends(get_db)):
    stmt = select(Scan).order_by(Scan.scan_date.desc()).options(
        selectinload(Scan.inferences),
        selectinload(Scan.report)
    ).limit(50)
    result = await db.execute(stmt)
    return result.scalars().all()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
