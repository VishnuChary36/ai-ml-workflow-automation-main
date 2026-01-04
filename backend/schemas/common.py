"""Common schemas used across the application."""
from datetime import datetime
from typing import Optional, List, Any, Dict, TypeVar, Generic
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: Optional[str] = Field(None, description="Service version")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-01T00:00:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }
    }


class ReadyResponse(BaseModel):
    """Readiness check response schema."""
    ready: bool = Field(..., description="Whether service is ready to accept traffic")
    status: str = Field(..., description="Readiness status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual readiness checks")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "ready": True,
                "status": "ready",
                "timestamp": "2025-01-01T00:00:00Z",
                "checks": {
                    "model_loaded": True,
                    "preprocessing_ready": True,
                    "database_connected": True
                }
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "detail": "Field 'model_id' is required",
                "code": "VALIDATION_001",
                "timestamp": "2025-01-01T00:00:00Z",
                "request_id": "req-abc123"
            }
        }
    }


class PaginatedResponse(BaseModel):
    """Paginated response schema."""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there's a next page")
    has_prev: bool = Field(..., description="Whether there's a previous page")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [{"id": "1"}, {"id": "2"}],
                "total": 100,
                "page": 1,
                "page_size": 20,
                "pages": 5,
                "has_next": True,
                "has_prev": False
            }
        }
    }


class TaskResponse(BaseModel):
    """Task response schema."""
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "task-abc123",
                "status": "started",
                "message": "Task started successfully",
                "created_at": "2025-01-01T00:00:00Z"
            }
        }
    }
