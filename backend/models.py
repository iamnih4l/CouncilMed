import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True)
    role = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    scans = relationship("Scan", back_populates="uploaded_by_user")

class Patient(Base):
    __tablename__ = 'patients'
    id = Column(String, primary_key=True, default=generate_uuid)
    full_name = Column(String)
    medical_record_number = Column(String, unique=True, index=True)
    
    scans = relationship("Scan", back_populates="patient")

class Scan(Base):
    __tablename__ = 'scans'
    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey('patients.id'))
    uploaded_by = Column(String, ForeignKey('users.id'))
    modality = Column(String)
    study_type = Column(String, nullable=True)
    storage_path = Column(String, nullable=True)
    scan_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String, default="COMPLETED")
    
    patient = relationship("Patient", back_populates="scans")
    uploaded_by_user = relationship("User", back_populates="scans")
    inferences = relationship("MLInference", back_populates="scan")
    report = relationship("CouncilReport", back_populates="scan", uselist=False)

class MLInference(Base):
    __tablename__ = 'ml_inferences'
    id = Column(String, primary_key=True, default=generate_uuid)
    scan_id = Column(String, ForeignKey('scans.id'))
    model_name = Column(String)
    raw_output = Column(JSON)
    confidence_score = Column(Float)
    execution_time_ms = Column(Integer)
    completed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    scan = relationship("Scan", back_populates="inferences")

class CouncilReport(Base):
    __tablename__ = 'council_reports'
    id = Column(String, primary_key=True, default=generate_uuid)
    scan_id = Column(String, ForeignKey('scans.id'))
    primary_diagnosis = Column(String)
    severity = Column(String)
    spatial_findings = Column(JSON)
    agent_reasoning_transcript = Column(String)
    overall_agreement_score = Column(Float)
    requires_manual_review = Column(Boolean, default=False)
    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    scan = relationship("Scan", back_populates="report")
