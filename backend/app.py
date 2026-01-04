"""Main FastAPI application with WebSocket log streaming."""
import os
import uuid
import asyncio
import json
import joblib
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from config import settings
from core.database import init_db, get_db, Dataset, Task, Model, Deployment
from core.log_emitter import log_emitter, create_log_entry
from core.log_store import log_store
from core.auth import (
    AuthMiddleware,
    require_auth,
    require_permissions,
    require_role,
    get_current_user,
    authenticate_user,
    create_access_token,
    create_api_key,
    User,
    Role,
    Permission,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from core.rate_limiter import (
    RateLimitMiddleware,
    RateLimitConfig,
    create_rate_limit_config,
)
from ws_log_broker import manager
from services.task_manager import TaskManager, TaskStatus, TaskType
from services.profiler import DataProfiler
from services.suggestor import SuggestionEngine
from services.preprocess import PreprocessingService
from services.training import TrainingService
from services.drift_monitoring import DriftMonitoringService
from services.visualization import VisualizationService, generate_model_visualizations
from services.explainability import ExplainabilityService
from services.deployment import DeploymentService
from services.dashboard import DashboardService

# Import schemas for type-safe API contracts
from schemas import (
    LoginRequest,
    LoginResponse,
    TokenResponse,
    UserResponse,
    APIKeyCreateRequest,
    APIKeyResponse,
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
)

# Initialize app
app = FastAPI(
    title="AI-ML Workflow Automation",
    description="Production-ready ML lifecycle automation with live console streaming",
    version="1.0.0"
)

# ============================================================================
# Middleware Setup
# ============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
rate_limit_config = create_rate_limit_config(redis_url=settings.redis_url)
app.add_middleware(RateLimitMiddleware, config=rate_limit_config)

# Authentication middleware for protected routes
# Note: This enforces auth on specific route prefixes
app.add_middleware(
    AuthMiddleware,
    protected_prefixes=[
        "/api/predict",
        "/api/deploy_model",
        "/api/deployment",
        "/api/start_drift_monitoring",
    ]
)

# Create storage directories
Path(settings.artifact_storage_path).mkdir(parents=True, exist_ok=True)
Path(settings.model_storage_path).mkdir(parents=True, exist_ok=True)
Path("./uploads").mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()
    print("✓ Database initialized")
    print("✓ Authentication middleware enabled")
    print("✓ Rate limiting middleware enabled")
    print(f"✓ Artifact storage: {settings.artifact_storage_path}")
    print(f"✓ Model storage: {settings.model_storage_path}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/api/auth/login", response_model=LoginResponse, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT access token.
    
    Use username and password to obtain an access token for API access.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": user.permissions,
        }
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role.value,
            permissions=user.permissions,
        ),
    )


@app.post("/api/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def get_token(request: LoginRequest):
    """
    Get JWT token using JSON request body.
    Alternative to form-based login.
    """
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
        )
    
    access_token = create_access_token(
        data={
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": user.permissions,
        }
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.get("/api/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(user: User = Depends(require_auth)):
    """Get current authenticated user information."""
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.value,
        permissions=user.permissions,
    )


@app.post("/api/auth/api-key", response_model=APIKeyResponse, tags=["Authentication"])
async def create_new_api_key(
    request: APIKeyCreateRequest,
    user: User = Depends(require_permissions(Permission.ADMIN_FULL)),
):
    """
    Create a new API key for programmatic access.
    Requires admin permissions.
    """
    api_key = create_api_key(
        user_id=user.id,
        name=request.name,
        permissions=request.permissions,
        rate_limit=request.rate_limit,
        expires_days=request.expires_days,
    )
    
    from core.auth import API_KEYS
    key_data = API_KEYS[api_key]
    
    return APIKeyResponse(
        key=api_key,
        key_id=key_data.key_id,
        name=key_data.name,
        permissions=key_data.permissions,
        rate_limit=key_data.rate_limit,
        created_at=key_data.created_at,
        expires_at=key_data.expires_at,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# Dataset Upload & Profiling
# ============================================================================

@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and profile a dataset."""
    # Generate dataset ID
    dataset_id = f"ds-{uuid.uuid4().hex[:12]}"

    # Save file
    file_path = f"./uploads/{dataset_id}_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Load dataset
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            df = pd.read_excel(file_path)
        elif file.filename.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file format")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error reading file: {str(e)}")

    # Profile dataset
    profile = DataProfiler.profile_dataset(df)
    target_column = DataProfiler.detect_target_column(df, profile)

    # Save to database
    dataset = Dataset(
        id=dataset_id,
        filename=file.filename,
        rows=len(df),
        columns=len(df.columns),
        profile=profile,
        file_path=file_path
    )
    db.add(dataset)
    db.commit()

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "profile": profile,
        "suggested_target": target_column
    }


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Get dataset information."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "uploaded_at": dataset.uploaded_at.isoformat(),
        "rows": dataset.rows,
        "columns": dataset.columns,
        "profile": dataset.profile
    }


# ============================================================================
# AI Suggestions
# ============================================================================

@app.get("/api/suggest/pipeline")
async def suggest_pipeline(
    dataset_id: str,
    target_column: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get AI-suggested preprocessing pipeline."""
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load dataframe
    df = pd.read_csv(dataset.file_path)

    # Generate suggestions
    suggestions = SuggestionEngine.suggest_pipeline(
        df, dataset.profile, target_column)

    return {
        "dataset_id": dataset_id,
        "suggestions": suggestions,
        "target_column": target_column
    }


@app.get("/api/suggest/models")
async def suggest_models(
    dataset_id: str,
    target_column: str,
    problem_type: str = "auto",
    db: Session = Depends(get_db)
):
    """Get AI-suggested models."""
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load dataframe
    df = pd.read_csv(dataset.file_path)

    # Generate suggestions
    suggestions = SuggestionEngine.suggest_models(
        df, dataset.profile, target_column, problem_type)

    return {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "problem_type": problem_type,
        "suggestions": suggestions
    }


# ============================================================================
# Pipeline Execution
# ============================================================================

@app.post("/api/run_pipeline")
async def run_pipeline(
    dataset_id: str = Query(...),
    steps: List[dict] = Body(...),
    db: Session = Depends(get_db)
):
    """Execute preprocessing pipeline."""
    # Create task
    task = TaskManager.create_task(
        db,
        TaskType.PREPROCESS,
        config={"steps": steps},
        dataset_id=dataset_id
    )

    # Run async
    asyncio.create_task(execute_pipeline_task(task.id, dataset_id, steps))

    return {
        "task_id": task.id,
        "status": "started",
        "message": "Pipeline execution started"
    }


async def execute_pipeline_task(task_id: str, dataset_id: str, steps: List[dict]):
    """Execute pipeline task in background."""
    from core.database import SessionLocal

    db = SessionLocal()

    try:
        # Update status
        TaskManager.update_status(db, task_id, TaskStatus.RUNNING)

        # Load dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        df = pd.read_csv(dataset.file_path)

        await TaskManager.emit_log(
            task_id, "INFO",
            f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns",
            source="pipeline.load"
        )

        # Execute steps
        preprocessor = PreprocessingService(task_id, TaskManager.emit_log)

        for step in steps:
            df = await preprocessor.execute_step(df, step)

        # Save processed dataset
        output_path = f"./artifacts/{task_id}_processed.csv"
        df.to_csv(output_path, index=False)

        await TaskManager.emit_log(
            task_id, "INFO",
            f"Pipeline completed. Processed dataset saved: {output_path}",
            source="pipeline.complete",
            meta={"output_path": output_path, "rows": len(
                df), "columns": len(df.columns)}
        )

        # Update task
        TaskManager.update_status(
            db, task_id, TaskStatus.COMPLETED,
            result={"output_path": output_path, "rows": len(
                df), "columns": len(df.columns)}
        )

    except Exception as e:
        await TaskManager.emit_log(
            task_id, "ERROR",
            f"Pipeline failed: {str(e)}",
            source="pipeline.error"
        )
        TaskManager.update_status(db, task_id, TaskStatus.FAILED, error=str(e))

    finally:
        db.close()


# ============================================================================
# Task Management
# ============================================================================

@app.get("/api/task/{task_id}/status")
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """Get task status."""
    task = TaskManager.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.id,
        "task_type": task.task_type,
        "status": task.status,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "config": task.config,
        "result": task.result,
        "error": task.error
    }


@app.get("/api/tasks")
async def list_tasks(
    limit: int = 50,
    task_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List tasks."""
    query = db.query(Task)

    if task_type:
        query = query.filter(Task.task_type == task_type)

    tasks = query.order_by(Task.created_at.desc()).limit(limit).all()

    return {
        "tasks": [
            {
                "task_id": t.id,
                "task_type": t.task_type,
                "status": t.status,
                "created_at": t.created_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None
            }
            for t in tasks
        ]
    }


@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str, db: Session = Depends(get_db)):
    """Cancel a running task."""
    task = TaskManager.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")

    TaskManager.update_status(db, task_id, TaskStatus.CANCELLED)

    await TaskManager.emit_log(
        task_id, "WARN",
        "Task cancelled by user",
        source="task.cancel"
    )

    return {"message": "Task cancelled", "task_id": task_id}


# ============================================================================
# Log Streaming (WebSocket)
# ============================================================================

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket, task_id: str = Query(...)):
    """WebSocket endpoint for streaming logs."""
    await manager.connect(websocket, task_id)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)


# ============================================================================
# Data Analytics Dashboard
# ============================================================================

@app.get("/api/dashboard/{task_id}")
async def get_data_dashboard(
    task_id: str,
    target_column: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Generate a comprehensive data analytics dashboard for the processed dataset.
    
    This endpoint creates a Power BI/Excel-style storytelling dashboard with:
    - Executive Summary with narrative
    - Key Performance Indicators (KPIs)
    - Data Quality Analysis
    - Distribution Analysis
    - Correlation Heatmaps
    - Categorical Breakdowns
    - Trend Analysis (if datetime columns exist)
    - Insights and Recommendations
    - Chart data for visualizations
    
    Parameters:
    - task_id: The preprocessing task ID
    - target_column: Optional target column for analysis
    """
    # Find the processed dataset
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get the output path from task result
    output_path = None
    if task.result:
        output_path = task.result.get("output_path")
    
    if not output_path:
        # Try default naming convention
        output_path = f"./artifacts/{task_id}_processed.csv"
    
    if not os.path.exists(output_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Processed dataset not found at {output_path}. Run preprocessing first."
        )
    
    try:
        # Load the processed dataset
        df = pd.read_csv(output_path)
        
        # Auto-detect target column if not provided
        if not target_column:
            # Try common target column names
            common_targets = ['target', 'label', 'class', 'y', 'outcome', 'result', 'status']
            for col in df.columns:
                if col.lower() in common_targets:
                    target_column = col
                    break
            # If not found, use the last column as target
            if not target_column:
                target_column = df.columns[-1]
        
        # Generate dashboard
        dashboard_service = DashboardService(task_id=task_id)
        dashboard = dashboard_service.generate_dashboard(df, target_column)
        
        # Add task and dataset info
        dashboard["task_id"] = task_id
        dashboard["target_column"] = target_column
        dashboard["file_path"] = output_path
        
        return dashboard
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate dashboard: {str(e)}"
        )


@app.get("/api/dashboard/{task_id}/summary")
async def get_dashboard_summary(
    task_id: str,
    db: Session = Depends(get_db)
):
    """Get a quick summary for the dashboard - lighter version for initial load."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    output_path = task.result.get("output_path") if task.result else f"./artifacts/{task_id}_processed.csv"
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Processed dataset not found")
    
    try:
        df = pd.read_csv(output_path)
        
        # Quick summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        summary = {
            "task_id": task_id,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "columns": df.columns.tolist(),
            "data_completeness": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/processed-datasets")
async def list_processed_datasets(db: Session = Depends(get_db)):
    """List all processed datasets available for dashboard analysis."""
    # Find all preprocessing tasks
    tasks = db.query(Task).filter(
        Task.task_type == "preprocess",
        Task.status == "completed"
    ).order_by(Task.completed_at.desc()).all()
    
    datasets = []
    for task in tasks:
        output_path = task.result.get("output_path") if task.result else None
        if output_path and os.path.exists(output_path):
            try:
                # Get basic file info
                df = pd.read_csv(output_path, nrows=0)  # Just get columns
                file_size = os.path.getsize(output_path)
                
                datasets.append({
                    "task_id": task.id,
                    "dataset_id": task.config.get("dataset_id") if task.config else None,
                    "file_path": output_path,
                    "columns": len(df.columns),
                    "file_size_mb": round(file_size / 1024 / 1024, 2),
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "rows": task.result.get("rows") if task.result else None
                })
            except:
                continue
    
    return {"datasets": datasets}


# ============================================================================
# Log Retrieval (HTTP)
# ============================================================================

@app.get("/api/logs/{task_id}")
async def get_logs_json(task_id: str):
    """Get logs as JSON."""
    logs = await log_store.get_logs(task_id, from_memory=False)
    return {"task_id": task_id, "logs": logs}


@app.get("/api/logs/{task_id}.txt")
async def get_logs_text(task_id: str):
    """Get logs as plain text."""
    logs_text = await log_store.get_logs_text(task_id)
    return StreamingResponse(
        iter([logs_text]),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={task_id}.log"}
    )


# ============================================================================
# Models
# ============================================================================

@app.get("/api/models")
async def list_models(db: Session = Depends(get_db)):
    """List trained models."""
    models = db.query(Model).order_by(Model.created_at.desc()).limit(50).all()

    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "algorithm": m.algorithm,
                "created_at": m.created_at.isoformat(),
                "metrics": m.metrics,
                "deployed": m.deployed,
                "deployment_url": m.deployment_url
            }
            for m in models
        ]
    }


# ============================================================================
# Model Training
# ============================================================================


@app.post("/api/train_model")
async def train_model(
    dataset_id: str,
    model_config: dict = Body(...),
    target_column: str = Query(...),
    db: Session = Depends(get_db)
):
    """Train a model with the specified configuration."""
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create training task
    task = TaskManager.create_task(
        db,
        TaskType.TRAIN,
        config={"model_config": model_config, "target_column": target_column},
        dataset_id=dataset_id
    )

    # Run training async
    asyncio.create_task(execute_training_task(
        task.id, dataset_id, model_config, target_column))

    return {
        "task_id": task.id,
        "status": "started",
        "message": "Model training started"
    }


async def execute_training_task(task_id: str, dataset_id: str, model_config: dict, target_column: str):
    """Execute training task in background."""
    from core.database import SessionLocal

    db = SessionLocal()

    try:
        # Update status
        TaskManager.update_status(db, task_id, TaskStatus.RUNNING)

        # Load dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        df = pd.read_csv(dataset.file_path)

        await TaskManager.emit_log(
            task_id, "INFO",
            f"Loaded dataset for training: {len(df)} rows, {len(df.columns)} columns",
            source="training.load"
        )

        # Execute training
        trainer = TrainingService(task_id, TaskManager.emit_log)
        result = await trainer.train_model(df, model_config, target_column)

        # Create model record
        model_id = "mdl-" + uuid.uuid4().hex[:12]
        model = Model(
            id=model_id,
            name=f"{model_config['model']}_{dataset.filename.split('.')[0]}",
            algorithm=model_config['model'],
            dataset_id=dataset_id,
            task_id=task_id,
            metrics=result['metrics'],
            hyperparams=model_config['params'],
            artifact_path=result['model_path']
        )
        db.add(model)

        await TaskManager.emit_log(
            task_id, "INFO",
            f"Model training completed. Model saved with ID: {model_id}",
            source="training.complete",
            meta={"model_id": model_id, "model_path": result['model_path']}
        )

        # Update task
        TaskManager.update_status(
            db, task_id, TaskStatus.COMPLETED,
            result={"model_id": model_id,
                    "model_path": result['model_path'], "metrics": result['metrics']}
        )

        # Generate visualizations after training
        viz_task_id = f"viz-{uuid.uuid4().hex[:12]}"
        viz_task = TaskManager.create_task(
            db,
            TaskType.VISUALIZATION,  # Use proper visualization task type
            config={"model_id": model_id, "model_path": result['model_path'],
                    "dataset_id": dataset_id, "target_column": target_column,
                    "model_name": result.get('model_name', model_config['model'])},
            model_id=model_id
        )
        asyncio.create_task(generate_model_visualizations(
            viz_task.id, dataset.file_path, result['model_path'],
            target_column, result['metrics'], result['model_type'], TaskManager.emit_log, model_id,
            model_name=result.get('model_name', model_config['model'])
        ))

    except Exception as e:
        await TaskManager.emit_log(
            task_id, "ERROR",
            f"Training failed: {str(e)}",
            source="training.error"
        )
        TaskManager.update_status(db, task_id, TaskStatus.FAILED, error=str(e))

    finally:
        db.close()


# ============================================================================
# Model Deployment
# ============================================================================


@app.post("/api/deploy_model")
async def deploy_model(
    model_id: str,
    platform: str = Query(..., description="Deployment platform: local, docker, or cloud"),
    db: Session = Depends(get_db)
):
    """
    Deploy a trained model to specified platform.
    
    Platforms:
    - local: Create deployment package for local serving
    - docker: Create Docker container for deployment
    - cloud: Prepare for cloud deployment (AWS/GCP/Azure)
    """
    # Get model
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Create deployment task
    task = TaskManager.create_task(
        db,
        TaskType.DEPLOY,
        config={"platform": platform, "model_id": model_id},
        model_id=model_id
    )

    # Run deployment async
    asyncio.create_task(execute_deployment_task(task.id, model_id, platform))

    return {
        "task_id": task.id,
        "status": "started",
        "message": f"Model deployment to {platform} started"
    }


async def execute_deployment_task(task_id: str, model_id: str, platform: str):
    """Execute deployment task in background - creates full deployment package."""
    from core.database import SessionLocal

    db = SessionLocal()

    try:
        # Update status
        TaskManager.update_status(db, task_id, TaskStatus.RUNNING)

        await TaskManager.emit_log(
            task_id, "INFO",
            f"🚀 Starting deployment to {platform}",
            source="deployment.init"
        )

        # Get model and related data
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise Exception("Model not found")
        
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == model.dataset_id).first()
        if not dataset:
            raise Exception("Dataset not found for model")
        
        # Get task for target column
        training_task = db.query(Task).filter(Task.id == model.task_id).first()
        
        await TaskManager.emit_log(
            task_id, "INFO",
            f"📦 Creating deployment package...",
            source="deployment.package"
        )
        
        # Load dataset
        df = None
        if training_task:
            preprocessed_patterns = [
                f"./artifacts/{training_task.id}_processed.csv",
                f"../artifacts/{training_task.id}_processed.csv",
                f"{settings.artifact_storage_path}/{training_task.id}_processed.csv",
            ]
            for pattern in preprocessed_patterns:
                if os.path.exists(pattern):
                    try:
                        df = pd.read_csv(pattern)
                        break
                    except Exception:
                        pass
        
        if df is None:
            df = pd.read_csv(dataset.file_path)
        
        # Determine target column
        target_column = None
        if training_task and training_task.config:
            target_column = training_task.config.get("target_column")
        if not target_column:
            target_column = df.columns[-1]
        
        # Determine model type
        model_type = "classification"
        if model.metrics:
            if "mse" in model.metrics or "r2" in model.metrics or "mae" in model.metrics:
                model_type = "regression"
        
        # Create deployment service
        deploy_service = DeploymentService(task_id, TaskManager.emit_log)
        
        # Create deployment package
        deployments_dir = "./deployments"
        os.makedirs(deployments_dir, exist_ok=True)
        
        # Use async method for real-time progress
        package_result = await deploy_service.package_model_async(
            model_path=model.artifact_path,
            df=df,
            target_column=target_column,
            model_name=model.algorithm or "Unknown",
            model_type=model_type,
            metrics=model.metrics or {},
            output_dir=deployments_dir
        )
        
        await TaskManager.emit_log(
            task_id, "INFO",
            f"✅ Deployment package created with {len(package_result.get('files', []))} files",
            source="deployment.package"
        )
        
        # Generate deployment URL based on platform
        deployment_id = "dep-" + uuid.uuid4().hex[:12]
        
        if platform == "local":
            deployment_url = f"http://localhost:8080"
            await TaskManager.emit_log(
                task_id, "INFO",
                f"📁 Local deployment package ready at: {package_result['package_dir']}",
                source="deployment.local"
            )
            await TaskManager.emit_log(
                task_id, "INFO",
                f"   To start: cd {package_result['package_dir']} && python app.py",
                source="deployment.local"
            )
            
        elif platform == "docker":
            deployment_url = f"http://localhost:8080"
            await TaskManager.emit_log(
                task_id, "INFO",
                f"🐳 Docker deployment ready. Build with:",
                source="deployment.docker"
            )
            await TaskManager.emit_log(
                task_id, "INFO",
                f"   cd {package_result['package_dir']} && docker build -t ml-model-{model_id[:8]} .",
                source="deployment.docker"
            )
            await TaskManager.emit_log(
                task_id, "INFO",
                f"   docker run -p 8080:8080 ml-model-{model_id[:8]}",
                source="deployment.docker"
            )
            
        elif platform == "cloud":
            deployment_url = f"https://api.example.com/models/{model_id}"
            await TaskManager.emit_log(
                task_id, "INFO",
                f"☁️ Cloud deployment package ready. Upload {package_result['package_path']} to your cloud provider.",
                source="deployment.cloud"
            )
        else:
            deployment_url = f"http://localhost:8080/model/{model_id}"
        
        # Live prediction URL (always available through main backend)
        live_prediction_url = f"http://localhost:8000/api/predict/{model_id}"
        
        await TaskManager.emit_log(
            task_id, "INFO",
            f"🎯 Live prediction endpoint ready: {live_prediction_url}",
            source="deployment.live"
        )
        
        # Create deployment record
        deployment = Deployment(
            id=deployment_id,
            model_id=model_id,
            task_id=task_id,
            platform=platform,
            url=live_prediction_url,  # Use live prediction URL
            config={
                "platform": platform,
                "package_path": package_result.get("package_path"),
                "package_dir": package_result.get("package_dir"),
                "files": package_result.get("files", []),
                "live_prediction_url": live_prediction_url,
                "standalone_url": deployment_url
            }
        )
        db.add(deployment)

        await TaskManager.emit_log(
            task_id, "INFO",
            f"✅ Deployment completed successfully!",
            source="deployment.complete",
            meta={"deployment_url": live_prediction_url, "deployment_id": deployment_id}
        )

        # Update model record
        model = db.query(Model).filter(Model.id == model_id).first()
        if model:
            model.deployed = True
            model.deployment_url = live_prediction_url

        # Update task
        TaskManager.update_status(
            db, task_id, TaskStatus.COMPLETED,
            result={
                "deployment_url": live_prediction_url,
                "live_prediction_url": live_prediction_url,
                "standalone_url": deployment_url,
                "deployment_id": deployment_id,
                "package_path": package_result.get("package_path"),
                "package_dir": package_result.get("package_dir"),
                "files": package_result.get("files", []),
                "platform": platform
            }
        )

    except Exception as e:
        await TaskManager.emit_log(
            task_id, "ERROR",
            f"❌ Deployment failed: {str(e)}",
            source="deployment.error"
        )
        TaskManager.update_status(db, task_id, TaskStatus.FAILED, error=str(e))
        import traceback
        traceback.print_exc()

    finally:
        db.close()


@app.get("/api/deployment/{deployment_id}")
async def get_deployment(deployment_id: str, db: Session = Depends(get_db)):
    """Get deployment details."""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    return {
        "id": deployment.id,
        "model_id": deployment.model_id,
        "platform": deployment.platform,
        "url": deployment.url,
        "config": deployment.config,
        "created_at": deployment.created_at.isoformat() if deployment.created_at else None
    }


@app.get("/api/deployment/{deployment_id}/download")
async def download_deployment_package(deployment_id: str, db: Session = Depends(get_db)):
    """Download deployment package as zip file."""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    package_path = deployment.config.get("package_path") if deployment.config else None
    if not package_path or not os.path.exists(package_path):
        raise HTTPException(status_code=404, detail="Deployment package not found")
    
    return FileResponse(
        package_path,
        media_type="application/zip",
        filename=f"deployment_{deployment.model_id}.zip"
    )


@app.get("/api/model/{model_id}/deployments")
async def list_model_deployments(model_id: str, db: Session = Depends(get_db)):
    """List all deployments for a model."""
    deployments = db.query(Deployment).filter(Deployment.model_id == model_id).all()
    
    return {
        "model_id": model_id,
        "deployments": [
            {
                "id": d.id,
                "platform": d.platform,
                "url": d.url,
                "created_at": d.created_at.isoformat() if d.created_at else None
            }
            for d in deployments
        ]
    }


# ============================================================================
# Drift Monitoring
# ============================================================================


@app.post("/api/start_drift_monitoring")
async def start_drift_monitoring(
    model_id: str,
    reference_dataset_id: str,
    db: Session = Depends(get_db)
):
    """Start drift monitoring for a deployed model."""
    # Verify model exists
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Get reference dataset
    dataset = db.query(Dataset).filter(
        Dataset.id == reference_dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=404, detail="Reference dataset not found")

    # Create monitoring task
    task = TaskManager.create_task(
        db,
        TaskType.TRAIN,  # Using TRAIN type as placeholder
        config={"model_id": model_id,
                "reference_dataset_id": reference_dataset_id},
        model_id=model_id
    )

    # In a real implementation, this would start a background monitoring process
    # For now, we'll just simulate it
    asyncio.create_task(execute_drift_monitoring_task(
        task.id, model_id, dataset.file_path))

    return {
        "task_id": task.id,
        "status": "started",
        "message": "Drift monitoring started"
    }


async def execute_drift_monitoring_task(task_id: str, model_id: str, reference_dataset_path: str):
    """Execute drift monitoring task in background."""
    from core.database import SessionLocal

    db = SessionLocal()

    try:
        # Update status
        TaskManager.update_status(db, task_id, TaskStatus.RUNNING)

        await TaskManager.emit_log(
            task_id, "INFO",
            f"Starting drift monitoring for model {model_id}",
            source="drift.init"
        )

        # Initialize drift monitoring service
        drift_service = DriftMonitoringService(task_id, TaskManager.emit_log)

        # In a real implementation, this would continuously monitor incoming data
        # For now, we'll just record that monitoring started
        await TaskManager.emit_log(
            task_id, "INFO",
            "Drift monitoring service initialized and running",
            source="drift.service"
        )

        # Update task
        TaskManager.update_status(
            db, task_id, TaskStatus.COMPLETED,
            result={"monitoring_started": True, "model_id": model_id}
        )

    except Exception as e:
        await TaskManager.emit_log(
            task_id, "ERROR",
            f"Drift monitoring failed: {str(e)}",
            source="drift.error"
        )
        TaskManager.update_status(db, task_id, TaskStatus.FAILED, error=str(e))

    finally:
        db.close()


# ============================================================================
# Visualization
# ============================================================================


def find_visualization_file(model_id: str) -> Optional[str]:
    """Find visualization file for a given model ID in multiple locations."""
    import glob
    
    # List of directories to search for visualization files
    search_dirs = [
        "./artifacts",
        "../artifacts",
        settings.artifact_storage_path,
    ]
    
    for search_dir in search_dirs:
        # Try multiple patterns
        patterns = [
            f"{search_dir}/*_{model_id}_visualizations.json",
            f"{search_dir}/*{model_id}*_visualizations.json",
            f"{search_dir}/*{model_id}*.json",
        ]
        
        for pattern in patterns:
            viz_files = glob.glob(pattern)
            if viz_files:
                return viz_files[0]
    
    return None


async def generate_visualizations_sync(model: Model, db: Session) -> dict:
    """Generate visualizations synchronously for a model."""
    import json
    import glob
    
    # Get the dataset
    dataset = db.query(Dataset).filter(Dataset.id == model.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found for model")
    
    # Try to find preprocessed data first (from pipeline runs)
    # Look for preprocessed CSV files related to this model's training task
    task = db.query(Task).filter(Task.id == model.task_id).first()
    df = None
    
    # Check for preprocessed data in artifacts
    if task:
        preprocessed_patterns = [
            f"./artifacts/{task.id}_processed.csv",
            f"../artifacts/{task.id}_processed.csv",
            f"{settings.artifact_storage_path}/{task.id}_processed.csv",
        ]
        for pattern in preprocessed_patterns:
            if os.path.exists(pattern):
                try:
                    df = pd.read_csv(pattern)
                    break
                except Exception:
                    pass
    
    # If no preprocessed data found, use original dataset
    if df is None:
        df = pd.read_csv(dataset.file_path)
    
    # Initialize visualization service (without task logging)
    viz_service = VisualizationService(model.task_id or "viz-sync", lambda *args, **kwargs: None)
    
    # Determine model type from metrics
    model_type = "classification"
    if model.metrics:
        if "mse" in model.metrics or "r2" in model.metrics or "mae" in model.metrics:
            model_type = "regression"
    
    # Get target column from task config or try to infer
    target_column = None
    if task and task.config:
        target_column = task.config.get("target_column")
    
    if not target_column:
        # Try to get from model hyperparams or use the last column
        target_column = df.columns[-1]
    
    # Get model name from algorithm field
    model_name = model.algorithm if model.algorithm else None
    
    # Generate visualizations
    try:
        result = viz_service.generate_visualizations(
            df, model.artifact_path, target_column, 
            model.metrics or {}, model_type, model_name
        )
        
        viz_result = {
            "model_id": model.id,
            "generated_at": datetime.utcnow().isoformat(),
            "visualizations": result.get("visualizations", {}),
            "metrics": result.get("metrics", model.metrics or {}),
            "model_type": result.get("model_type", model_type),
            "model_name": result.get("model_name", model_name),
            "target_column": result.get("target_column", target_column),
            "dataset_info": result.get("dataset_info", {})
        }
        
        # Save to file for future use
        viz_result_path = f"./artifacts/{model.task_id}_{model.id}_visualizations.json"
        os.makedirs(os.path.dirname(viz_result_path), exist_ok=True)
        
        with open(viz_result_path, 'w') as f:
            json.dump(viz_result, f, indent=2)
        
        return viz_result
        
    except Exception as e:
        # Return minimal visualizations on error
        return {
            "model_id": model.id,
            "generated_at": datetime.utcnow().isoformat(),
            "visualizations": {},
            "metrics": model.metrics or {},
            "model_type": model_type,
            "target_column": target_column,
            "error": str(e)
        }


@app.get("/api/visualizations/{model_id}")
async def get_model_visualizations(model_id: str, db: Session = Depends(get_db)):
    """Get visualizations for a trained model."""
    import json

    # First check if model exists
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Try to find existing visualization file
    viz_file = find_visualization_file(model_id)
    
    if viz_file:
        try:
            with open(viz_file, 'r') as f:
                visualizations = json.load(f)
            return visualizations
        except json.JSONDecodeError:
            # File corrupted, regenerate
            pass
        except Exception:
            # Other error, try to regenerate
            pass
    
    # No visualization file found or error reading it - generate on demand
    try:
        visualizations = await generate_visualizations_sync(model, db)
        return visualizations
    except Exception as e:
        # Return a minimal response with model metrics if visualization generation fails
        return {
            "model_id": model_id,
            "generated_at": datetime.utcnow().isoformat(),
            "visualizations": {},
            "metrics": model.metrics or {},
            "model_type": "unknown",
            "target_column": None,
            "status": "generation_failed",
            "error": str(e)
        }


# ============================================================================
# Model Explainability
# ============================================================================


@app.get("/api/explainability/{model_id}")
async def get_model_explainability(model_id: str, db: Session = Depends(get_db)):
    """Get comprehensive model explainability analysis using SHAP, LIME, and other techniques."""
    import json
    import glob
    
    # Check if model exists
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check for cached explainability file
    explain_patterns = [
        f"./artifacts/*{model_id}*_explainability.json",
        f"../artifacts/*{model_id}*_explainability.json",
        f"{settings.artifact_storage_path}/*{model_id}*_explainability.json",
    ]
    
    for pattern in explain_patterns:
        for file_path in glob.glob(pattern):
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        return json.load(f)
                except:
                    pass
    
    # Generate explainability on demand
    try:
        result = await generate_explainability_sync(model, db)
        return result
    except Exception as e:
        return {
            "model_id": model_id,
            "generated_at": datetime.utcnow().isoformat(),
            "explanations": {},
            "metadata": {"error": str(e)},
            "status": "generation_failed",
            "error": str(e)
        }


async def generate_explainability_sync(model: Model, db: Session) -> dict:
    """Generate explainability analysis synchronously for a model."""
    import json
    
    # Get the dataset
    dataset = db.query(Dataset).filter(Dataset.id == model.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found for model")
    
    # Try to find preprocessed data first
    task = db.query(Task).filter(Task.id == model.task_id).first()
    df = None
    
    if task:
        preprocessed_patterns = [
            f"./artifacts/{task.id}_processed.csv",
            f"../artifacts/{task.id}_processed.csv",
            f"{settings.artifact_storage_path}/{task.id}_processed.csv",
        ]
        for pattern in preprocessed_patterns:
            if os.path.exists(pattern):
                try:
                    df = pd.read_csv(pattern)
                    break
                except Exception:
                    pass
    
    if df is None:
        df = pd.read_csv(dataset.file_path)
    
    # Initialize explainability service
    explain_service = ExplainabilityService(model.task_id or "explain-sync")
    
    # Determine model type from metrics
    model_type = "classification"
    if model.metrics:
        if "mse" in model.metrics or "r2" in model.metrics or "mae" in model.metrics:
            model_type = "regression"
    
    # Get target column - try multiple sources
    target_column = None
    
    # 1. Try from task config
    if task and task.config:
        target_column = task.config.get("target_column")
    
    # 2. Try from model config
    if not target_column and model.config:
        target_column = model.config.get("target_column")
    
    # 3. Try from dataset suggested target
    if not target_column and dataset.columns:
        # Dataset columns might contain suggested target
        if isinstance(dataset.columns, dict) and "suggested_target" in dataset.columns:
            target_column = dataset.columns.get("suggested_target")
    
    # 4. Infer from column names - look for common target column names
    if not target_column:
        common_targets = ['target', 'label', 'class', 'category', 'y', 'outcome', 
                         'result', 'status', 'type', 'response', 'classification']
        for col in df.columns:
            if col.lower() in common_targets or any(t in col.lower() for t in common_targets):
                target_column = col
                break
    
    # 5. For classification, find column with few unique values relative to dataset size
    if not target_column and model_type == "classification":
        for col in df.columns:
            n_unique = df[col].nunique()
            # Good target candidates: categorical with reasonable number of classes
            if 2 <= n_unique <= 50 and n_unique < len(df) * 0.1:
                if df[col].dtype == 'object' or n_unique <= 20:
                    target_column = col
                    break
    
    # 6. Fallback: last column
    if not target_column:
        target_column = df.columns[-1]
    
    # Get model display name
    MODEL_DISPLAY_NAMES = {
        'RandomForestClassifier': 'Random Forest Classifier',
        'RandomForestRegressor': 'Random Forest Regressor',
        'XGBClassifier': 'XGBoost Classifier',
        'XGBRegressor': 'XGBoost Regressor',
        'LogisticRegression': 'Logistic Regression',
        'LinearRegression': 'Linear Regression',
        'GradientBoostingClassifier': 'Gradient Boosting Classifier',
        'GradientBoostingRegressor': 'Gradient Boosting Regressor',
    }
    model_name = MODEL_DISPLAY_NAMES.get(model.algorithm, model.algorithm) if model.algorithm else None
    
    # Generate explainability
    result = explain_service.generate_explainability(
        df, model.artifact_path, target_column, model_type, model_name
    )
    
    # Add model info
    result["model_id"] = model.id
    result["model_name"] = model_name
    result["target_column"] = target_column
    
    # Save to file for caching
    explain_path = f"./artifacts/{model.task_id}_{model.id}_explainability.json"
    os.makedirs(os.path.dirname(explain_path), exist_ok=True)
    
    with open(explain_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


# ============================================================================
# Live Model Prediction API - Deployed Models
# ============================================================================

# In-memory store for deployed model metadata
deployed_models_cache = {}


def load_deployed_model(model_id: str, db: Session):
    """Load a deployed model and its preprocessing artifacts."""
    if model_id in deployed_models_cache:
        return deployed_models_cache[model_id]
    
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model or not model.artifact_path:
        return None
    
    try:
        # Load the model
        loaded_model = joblib.load(model.artifact_path)
        
        # Try to load preprocessing artifacts
        deployment = db.query(Deployment).filter(Deployment.model_id == model_id).first()
        encoders = None
        preprocessing_config = None
        target_encoder = None
        
        if deployment and deployment.config:
            package_dir = deployment.config.get("package_dir")
            if package_dir:
                # Load encoders
                encoders_path = os.path.join(package_dir, "encoders.joblib")
                if os.path.exists(encoders_path):
                    encoders = joblib.load(encoders_path)
                
                # Load preprocessing config
                preprocess_path = os.path.join(package_dir, "preprocessing.json")
                if os.path.exists(preprocess_path):
                    with open(preprocess_path, 'r') as f:
                        preprocessing_config = json.load(f)
                
                # Load target encoder
                target_encoder_path = os.path.join(package_dir, "target_encoder.joblib")
                if os.path.exists(target_encoder_path):
                    target_encoder = joblib.load(target_encoder_path)
        
        # Determine model type
        model_type = "classification"
        if model.metrics:
            if "mse" in model.metrics or "r2" in model.metrics or "mae" in model.metrics:
                model_type = "regression"
        
        cache_entry = {
            "model": loaded_model,
            "encoders": encoders,
            "preprocessing_config": preprocessing_config,
            "target_encoder": target_encoder,
            "model_type": model_type,
            "model_name": model.algorithm,
            "metrics": model.metrics
        }
        
        deployed_models_cache[model_id] = cache_entry
        return cache_entry
        
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None


@app.post("/api/predict/{model_id}")
async def predict(
    model_id: str,
    features: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Make predictions using a deployed model.
    
    This is the LIVE prediction endpoint that frontend applications can call.
    
    Example request body:
    {
        "features": {
            "feature1": "value1",
            "feature2": 123,
            ...
        }
    }
    
    Or for batch predictions:
    {
        "features": [
            {"feature1": "value1", "feature2": 123},
            {"feature1": "value2", "feature2": 456}
        ]
    }
    """
    # Load the deployed model
    model_data = load_deployed_model(model_id, db)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found or not deployed")
    
    model = model_data["model"]
    encoders = model_data["encoders"]
    preprocessing_config = model_data["preprocessing_config"]
    target_encoder = model_data["target_encoder"]
    model_type = model_data["model_type"]
    
    try:
        # Get features from request
        input_features = features.get("features", features)
        
        # Handle single prediction or batch
        is_batch = isinstance(input_features, list)
        if not is_batch:
            input_features = [input_features]
        
        # Convert to DataFrame
        df = pd.DataFrame(input_features)
        
        # Apply preprocessing if available
        if preprocessing_config:
            feature_order = preprocessing_config.get("feature_order", [])
            numeric_columns = preprocessing_config.get("numeric_columns", [])
            categorical_columns = preprocessing_config.get("categorical_columns", [])
            fill_values = preprocessing_config.get("fill_values", {})
            
            # Reorder columns if needed
            if feature_order:
                missing_cols = [c for c in feature_order if c not in df.columns]
                for col in missing_cols:
                    df[col] = None
                df = df[feature_order]
            
            # Fill missing numeric values
            for col, fill_val in fill_values.items():
                if col in df.columns:
                    df[col] = df[col].fillna(fill_val)
            
            # Encode categorical columns
            if encoders:
                for col, encoder in encoders.items():
                    if col in df.columns:
                        df[col] = df[col].fillna('_MISSING_').astype(str)
                        # Handle unseen categories
                        df[col] = df[col].apply(
                            lambda x: x if x in encoder.classes_ else '_MISSING_' if '_MISSING_' in encoder.classes_ else encoder.classes_[0]
                        )
                        df[col] = encoder.transform(df[col])
        
        # Make prediction
        predictions = model.predict(df)
        
        # Get probabilities for classification
        probabilities = None
        if model_type == "classification" and hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(df).tolist()
            except:
                pass
        
        # Decode predictions if target encoder exists
        decoded_predictions = predictions.tolist()
        if target_encoder is not None and model_type == "classification":
            try:
                decoded_predictions = target_encoder.inverse_transform(predictions.astype(int)).tolist()
            except:
                pass
        
        # Format response
        if is_batch:
            results = []
            for i, pred in enumerate(decoded_predictions):
                result = {"prediction": pred}
                if probabilities:
                    result["probabilities"] = probabilities[i]
                    if target_encoder is not None:
                        result["class_labels"] = target_encoder.classes_.tolist()
                results.append(result)
            return {"predictions": results, "model_id": model_id}
        else:
            result = {"prediction": decoded_predictions[0], "model_id": model_id}
            if probabilities:
                result["probabilities"] = probabilities[0]
                if target_encoder is not None:
                    result["class_labels"] = target_encoder.classes_.tolist()
            return result
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/api/predict/{model_id}/info")
async def get_prediction_info(model_id: str, db: Session = Depends(get_db)):
    """
    Get information about a deployed model's prediction endpoint.
    
    Returns the expected input format, feature names, and example code.
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    deployment = db.query(Deployment).filter(Deployment.model_id == model_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Model not deployed")
    
    # Load preprocessing config for feature info
    preprocessing_config = None
    if deployment.config:
        package_dir = deployment.config.get("package_dir")
        if package_dir:
            preprocess_path = os.path.join(package_dir, "preprocessing.json")
            if os.path.exists(preprocess_path):
                with open(preprocess_path, 'r') as f:
                    preprocessing_config = json.load(f)
    
    # Build feature info
    features = []
    example_input = {}
    
    if preprocessing_config:
        for col in preprocessing_config.get("feature_order", []):
            feature_info = {"name": col}
            if col in preprocessing_config.get("numeric_columns", []):
                feature_info["type"] = "number"
                example_input[col] = 0
            else:
                feature_info["type"] = "string"
                encoder_classes = preprocessing_config.get("encoder_classes", {})
                if col in encoder_classes:
                    feature_info["allowed_values"] = encoder_classes[col][:10]  # First 10 values
                    example_input[col] = encoder_classes[col][0] if encoder_classes[col] else ""
                else:
                    example_input[col] = ""
            features.append(feature_info)
    
    # Determine model type
    model_type = "classification"
    if model.metrics:
        if "mse" in model.metrics or "r2" in model.metrics:
            model_type = "regression"
    
    # Generate example code
    api_url = f"http://localhost:8000/api/predict/{model_id}"
    
    javascript_code = f'''// JavaScript/React Example
const response = await fetch("{api_url}", {{
  method: "POST",
  headers: {{ "Content-Type": "application/json" }},
  body: JSON.stringify({{ features: {json.dumps(example_input, indent=4)} }})
}});
const result = await response.json();
console.log(result.prediction);'''

    python_code = f'''# Python Example
import requests

response = requests.post(
    "{api_url}",
    json={{"features": {json.dumps(example_input)}}}
)
result = response.json()
print(result["prediction"])'''

    curl_code = f'''# cURL Example
curl -X POST "{api_url}" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps({"features": example_input})}\''''

    return {
        "model_id": model_id,
        "model_name": model.algorithm,
        "model_type": model_type,
        "endpoint_url": api_url,
        "method": "POST",
        "features": features,
        "example_input": example_input,
        "code_examples": {
            "javascript": javascript_code,
            "python": python_code,
            "curl": curl_code
        }
    }


@app.get("/api/deployed-models")
async def list_deployed_models(db: Session = Depends(get_db)):
    """List all deployed models with their prediction endpoints."""
    deployments = db.query(Deployment).all()
    
    deployed = []
    for dep in deployments:
        model = db.query(Model).filter(Model.id == dep.model_id).first()
        if model:
            deployed.append({
                "model_id": model.id,
                "model_name": model.name,
                "algorithm": model.algorithm,
                "prediction_url": f"http://localhost:8000/api/predict/{model.id}",
                "deployed_at": dep.created_at.isoformat() if dep.created_at else None,
                "platform": dep.platform,
                "metrics": model.metrics
            })
    
    return {"deployed_models": deployed}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
