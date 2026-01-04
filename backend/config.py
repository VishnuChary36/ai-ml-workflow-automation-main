"""Application configuration management."""
import os
from typing import Optional
from pydantic_settings import BaseSettings


def _get_secret(key: str, default: str = None) -> Optional[str]:
    """
    Get value from secrets manager or environment variable.
    Lazy imports to avoid circular dependencies.
    """
    # First try environment variable (fast path)
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    # Try secrets manager if configured
    secrets_backend = os.getenv("SECRETS_BACKEND", "env")
    if secrets_backend != "env":
        try:
            from core.secrets import secrets
            value = secrets.get(key)
            if value:
                return value
        except ImportError:
            pass
    
    return default


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "AI-ML Workflow Automation"
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    secret_key: str = _get_secret("SECRET_KEY", "development-secret-key-change-in-production")
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Database - prefer secrets manager for credentials
    database_url: str = _get_secret("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mlworkflow")
    
    # Redis
    redis_url: str = _get_secret("REDIS_URL", "redis://localhost:6379/0")
    
    # Celery
    celery_broker_url: str = _get_secret("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_result_backend: str = _get_secret("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    # Storage
    artifact_storage_path: str = os.getenv("ARTIFACT_STORAGE_PATH", "./artifacts")
    model_storage_path: str = os.getenv("MODEL_STORAGE_PATH", "./models")
    
    # AI/LLM - always use secrets manager for API keys
    openai_api_key: Optional[str] = _get_secret("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = _get_secret("ANTHROPIC_API_KEY")
    use_llm_suggestions: bool = os.getenv("USE_LLM_SUGGESTIONS", "false").lower() == "true"
    
    # Cloud - use secrets manager for credentials
    aws_access_key_id: Optional[str] = _get_secret("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = _get_secret("AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_ecr_repository: Optional[str] = os.getenv("AWS_ECR_REPOSITORY")
    
    gcp_project_id: Optional[str] = os.getenv("GCP_PROJECT_ID")
    gcp_service_account: Optional[str] = _get_secret("GCP_SERVICE_ACCOUNT")
    gcp_region: str = os.getenv("GCP_REGION", "us-central1")
    
    hf_api_token: Optional[str] = _get_secret("HF_API_TOKEN")
    hf_space_name: Optional[str] = os.getenv("HF_SPACE_NAME")
    
    # Inference Service
    inference_image_name: str = os.getenv("INFERENCE_IMAGE_NAME", "ml-inference")
    inference_image_registry: Optional[str] = os.getenv("INFERENCE_IMAGE_REGISTRY")
    
    # Monitoring
    drift_check_interval_seconds: int = int(os.getenv("DRIFT_CHECK_INTERVAL_SECONDS", "3600"))
    enable_drift_monitoring: bool = os.getenv("ENABLE_DRIFT_MONITORING", "true").lower() == "true"
    
    # Security
    auth_enabled: bool = os.getenv("AUTH_ENABLED", "true").lower() == "true"
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
