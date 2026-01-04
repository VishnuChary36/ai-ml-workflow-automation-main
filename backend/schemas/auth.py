"""Authentication and authorization schemas."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str = Field(..., min_length=1, max_length=50, description="Username")
    password: str = Field(..., min_length=6, max_length=100, description="Password")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "admin",
                "password": "admin123"
            }
        }
    }


class LoginResponse(BaseModel):
    """Login response schema."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: "UserResponse" = Field(..., description="User information")


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class UserResponse(BaseModel):
    """User response schema."""
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    role: str = Field(..., description="User role")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "usr-001",
                "username": "admin",
                "email": "admin@example.com",
                "role": "admin",
                "permissions": ["dataset:read", "model:train", "deploy:create"]
            }
        }
    }


class APIKeyCreateRequest(BaseModel):
    """API Key creation request schema."""
    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    permissions: List[str] = Field(default_factory=list, description="Permissions for this key")
    rate_limit: int = Field(default=100, ge=1, le=10000, description="Rate limit (requests per minute)")
    expires_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Production API Key",
                "permissions": ["predict", "predict:batch"],
                "rate_limit": 1000,
                "expires_days": 90
            }
        }
    }


class APIKeyResponse(BaseModel):
    """API Key response schema."""
    key: str = Field(..., description="The API key (only shown once)")
    key_id: str = Field(..., description="API key identifier")
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(..., description="Permissions")
    rate_limit: int = Field(..., description="Rate limit")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")


# Forward reference update
LoginResponse.model_rebuild()
