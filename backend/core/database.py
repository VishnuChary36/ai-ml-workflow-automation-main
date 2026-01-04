"""Database models and connection management."""
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import settings

Base = declarative_base()


class Dataset(Base):
    """Dataset model."""
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    rows = Column(Integer)
    columns = Column(Integer)
    profile = Column(JSON)  # Store profiling results
    file_path = Column(String)


class Task(Base):
    """Task model for tracking background jobs."""
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True)
    task_type = Column(String, nullable=False)  # preprocess, train, deploy, etc.
    status = Column(String, default="pending")  # pending, running, completed, failed, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    dataset_id = Column(String, nullable=True)
    model_id = Column(String, nullable=True)
    config = Column(JSON)  # Task configuration
    result = Column(JSON, nullable=True)  # Task result
    error = Column(Text, nullable=True)  # Error message if failed


class Model(Base):
    """Model registry."""
    __tablename__ = "models"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    algorithm = Column(String, nullable=False)
    dataset_id = Column(String)
    task_id = Column(String)  # Training task ID
    created_at = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)  # Training metrics
    hyperparams = Column(JSON)  # Hyperparameters
    artifact_path = Column(String)  # Path to model file
    deployed = Column(Boolean, default=False)
    deployment_url = Column(String, nullable=True)


class Deployment(Base):
    """Deployment tracking."""
    __tablename__ = "deployments"
    
    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)
    task_id = Column(String)  # Deployment task ID
    platform = Column(String, nullable=False)  # docker, hf, aws, gcp
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")
    url = Column(String, nullable=True)
    config = Column(JSON)


class DriftAlert(Base):
    """Drift detection alerts."""
    __tablename__ = "drift_alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, nullable=False)
    metric = Column(String, nullable=False)  # PSI, KL, ADWIN
    value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    severity = Column(String)  # LOW, MEDIUM, HIGH
    column_name = Column(String, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    message = Column(Text)


# Database engine and session
engine = None
SessionLocal = None


def init_db():
    """Initialize database connection."""
    global engine, SessionLocal
    
    engine = create_engine(settings.database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    if SessionLocal is None:
        init_db()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
