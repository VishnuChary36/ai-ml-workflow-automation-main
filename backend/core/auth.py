"""Authentication, Authorization and RBAC utilities."""
from datetime import datetime, timedelta
from typing import Optional, List, Callable
from enum import Enum
from functools import wraps

from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# Role-Based Access Control (RBAC)
# ============================================================================

class Role(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(str, Enum):
    """Permissions for fine-grained access control."""
    # Dataset permissions
    DATASET_READ = "dataset:read"
    DATASET_WRITE = "dataset:write"
    DATASET_DELETE = "dataset:delete"
    
    # Model permissions
    MODEL_READ = "model:read"
    MODEL_TRAIN = "model:train"
    MODEL_DELETE = "model:delete"
    
    # Deployment permissions
    DEPLOY_READ = "deployment:read"
    DEPLOY_CREATE = "deployment:create"
    DEPLOY_DELETE = "deployment:delete"
    
    # Prediction permissions
    PREDICT = "predict"
    PREDICT_BATCH = "predict:batch"
    
    # Monitoring permissions
    MONITORING_READ = "monitoring:read"
    MONITORING_CONFIGURE = "monitoring:configure"
    
    # Admin permissions
    ADMIN_FULL = "admin:full"


# Role to permissions mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [p for p in Permission],  # Admin has all permissions
    Role.DATA_SCIENTIST: [
        Permission.DATASET_READ,
        Permission.DATASET_WRITE,
        Permission.MODEL_READ,
        Permission.MODEL_TRAIN,
        Permission.DEPLOY_READ,
        Permission.MONITORING_READ,
    ],
    Role.ML_ENGINEER: [
        Permission.DATASET_READ,
        Permission.MODEL_READ,
        Permission.MODEL_TRAIN,
        Permission.DEPLOY_READ,
        Permission.DEPLOY_CREATE,
        Permission.MONITORING_READ,
        Permission.MONITORING_CONFIGURE,
    ],
    Role.VIEWER: [
        Permission.DATASET_READ,
        Permission.MODEL_READ,
        Permission.DEPLOY_READ,
        Permission.MONITORING_READ,
    ],
    Role.API_USER: [
        Permission.PREDICT,
        Permission.PREDICT_BATCH,
        Permission.MODEL_READ,
    ],
}


# Protected routes configuration
PROTECTED_ROUTES = {
    # Prediction endpoints - require authentication
    "/api/predict": [Permission.PREDICT],
    "/api/predict/batch": [Permission.PREDICT_BATCH],
    
    # Deployment endpoints
    "/api/deploy_model": [Permission.DEPLOY_CREATE],
    "/api/deployment": [Permission.DEPLOY_READ],
    
    # Drift monitoring
    "/api/start_drift_monitoring": [Permission.MONITORING_CONFIGURE],
}


# ============================================================================
# User and Token Models
# ============================================================================

class User(BaseModel):
    """User model for authentication."""
    id: str
    username: str
    email: Optional[str] = None
    role: Role = Role.VIEWER
    permissions: List[str] = []
    api_key: Optional[str] = None
    disabled: bool = False


class TokenData(BaseModel):
    """Token payload data."""
    sub: str  # user_id
    username: str
    role: str
    permissions: List[str] = []
    exp: datetime
    iat: datetime
    type: str = "access"


class APIKeyData(BaseModel):
    """API Key data."""
    key_id: str
    user_id: str
    name: str
    permissions: List[str] = []
    rate_limit: int = 100  # requests per minute
    created_at: datetime
    expires_at: Optional[datetime] = None


# ============================================================================
# In-memory stores (replace with database in production)
# ============================================================================

# Sample users for demo - in production, load from database
DEMO_USERS = {
    "admin": User(
        id="usr-admin-001",
        username="admin",
        email="admin@example.com",
        role=Role.ADMIN,
        permissions=[p.value for p in Permission],
    ),
    "scientist": User(
        id="usr-scientist-001",
        username="scientist",
        email="scientist@example.com",
        role=Role.DATA_SCIENTIST,
        permissions=[p.value for p in ROLE_PERMISSIONS[Role.DATA_SCIENTIST]],
    ),
    "api_user": User(
        id="usr-api-001",
        username="api_user",
        email="api@example.com",
        role=Role.API_USER,
        permissions=[p.value for p in ROLE_PERMISSIONS[Role.API_USER]],
        api_key="mlwf-api-key-demo-12345",
    ),
}

# Password hashes for demo users - lazily initialized
_DEMO_PASSWORD_HASHES = None

def _get_demo_password_hashes():
    """Lazily initialize password hashes to avoid bcrypt issues at import time."""
    global _DEMO_PASSWORD_HASHES
    if _DEMO_PASSWORD_HASHES is None:
        try:
            _DEMO_PASSWORD_HASHES = {
                "admin": pwd_context.hash("admin123"),
                "scientist": pwd_context.hash("scientist123"),
                "api_user": pwd_context.hash("api123"),
            }
        except Exception:
            # Fallback for environments without bcrypt
            import hashlib
            _DEMO_PASSWORD_HASHES = {
                "admin": hashlib.sha256("admin123".encode()).hexdigest(),
                "scientist": hashlib.sha256("scientist123".encode()).hexdigest(),
                "api_user": hashlib.sha256("api123".encode()).hexdigest(),
            }
    return _DEMO_PASSWORD_HASHES

# API Keys store
API_KEYS = {
    "mlwf-api-key-demo-12345": APIKeyData(
        key_id="key-001",
        user_id="usr-api-001",
        name="Demo API Key",
        permissions=[Permission.PREDICT.value, Permission.PREDICT_BATCH.value, Permission.MODEL_READ.value],
        rate_limit=100,
        created_at=datetime.utcnow(),
    ),
}


# ============================================================================
# Password Utilities
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# ============================================================================
# Token Utilities
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    now = datetime.utcnow()
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": "access"
    })
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def create_api_key(user_id: str, name: str, permissions: List[str], 
                   rate_limit: int = 100, expires_days: Optional[int] = None) -> str:
    """Create a new API key."""
    import secrets
    
    key = f"mlwf-{secrets.token_urlsafe(32)}"
    expires_at = None
    if expires_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_days)
    
    API_KEYS[key] = APIKeyData(
        key_id=f"key-{secrets.token_hex(6)}",
        user_id=user_id,
        name=name,
        permissions=permissions,
        rate_limit=rate_limit,
        created_at=datetime.utcnow(),
        expires_at=expires_at,
    )
    
    return key


# ============================================================================
# Authentication Functions
# ============================================================================

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    if username not in DEMO_USERS:
        return None
    
    password_hashes = _get_demo_password_hashes()
    if username not in password_hashes:
        return None
    
    if not verify_password(password, password_hashes[username]):
        return None
    
    return DEMO_USERS[username]


def get_user_from_token(token: str) -> Optional[User]:
    """Get user from JWT token."""
    payload = decode_token(token)
    if not payload:
        return None
    
    username = payload.get("username")
    if not username or username not in DEMO_USERS:
        return None
    
    return DEMO_USERS[username]


def get_user_from_api_key(api_key: str) -> Optional[tuple[User, APIKeyData]]:
    """Get user from API key."""
    if api_key not in API_KEYS:
        return None
    
    key_data = API_KEYS[api_key]
    
    # Check expiration
    if key_data.expires_at and datetime.utcnow() > key_data.expires_at:
        return None
    
    # Find user
    for user in DEMO_USERS.values():
        if user.id == key_data.user_id:
            return user, key_data
    
    return None


# ============================================================================
# Dependency Injection Functions
# ============================================================================

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[User]:
    """
    Get the current authenticated user from JWT token or API key.
    Returns None if not authenticated (for optional auth).
    """
    # Try Bearer token first
    if credentials and credentials.credentials:
        user = get_user_from_token(credentials.credentials)
        if user:
            request.state.auth_method = "bearer"
            request.state.user = user
            return user
    
    # Try API key
    if api_key:
        result = get_user_from_api_key(api_key)
        if result:
            user, key_data = result
            request.state.auth_method = "api_key"
            request.state.api_key_data = key_data
            request.state.user = user
            return user
    
    return None


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
) -> User:
    """
    Require authentication - raises 401 if not authenticated.
    """
    user = await get_current_user(request, credentials, api_key)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    
    return user


def require_permissions(*required_permissions: Permission):
    """
    Dependency factory that requires specific permissions.
    
    Usage:
        @app.get("/admin", dependencies=[Depends(require_permissions(Permission.ADMIN_FULL))])
        async def admin_endpoint():
            ...
    """
    async def permission_checker(
        request: Request,
        user: User = Depends(require_auth),
    ) -> User:
        user_permissions = set(user.permissions)
        
        # Check if user has all required permissions
        for perm in required_permissions:
            if perm.value not in user_permissions and Permission.ADMIN_FULL.value not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied. Required: {perm.value}",
                )
        
        return user
    
    return permission_checker


def require_role(*allowed_roles: Role):
    """
    Dependency factory that requires specific roles.
    
    Usage:
        @app.get("/admin", dependencies=[Depends(require_role(Role.ADMIN))])
        async def admin_endpoint():
            ...
    """
    async def role_checker(
        user: User = Depends(require_auth),
    ) -> User:
        if user.role not in allowed_roles and user.role != Role.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role not authorized. Required one of: {[r.value for r in allowed_roles]}",
            )
        return user
    
    return role_checker


# ============================================================================
# Auth Middleware
# ============================================================================

class AuthMiddleware:
    """
    Middleware to enforce authentication on protected routes.
    """
    
    def __init__(self, app, protected_prefixes: List[str] = None):
        self.app = app
        self.protected_prefixes = protected_prefixes or [
            "/api/predict",
            "/api/deploy_model",
            "/api/deployment",
            "/api/start_drift_monitoring",
        ]
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        path = scope.get("path", "")
        
        # Check if route is protected
        is_protected = any(path.startswith(prefix) for prefix in self.protected_prefixes)
        
        if not is_protected:
            await self.app(scope, receive, send)
            return
        
        # Extract auth headers
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()
        api_key = headers.get(b"x-api-key", b"").decode()
        
        # Try to authenticate
        user = None
        
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user = get_user_from_token(token)
        elif api_key:
            result = get_user_from_api_key(api_key)
            if result:
                user, _ = result
        
        if not user:
            # Send 401 response
            response = {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"www-authenticate", b"Bearer"],
                ],
            }
            await send(response)
            
            body = b'{"detail": "Authentication required for this endpoint"}'
            await send({
                "type": "http.response.body",
                "body": body,
            })
            return
        
        # User authenticated - continue
        await self.app(scope, receive, send)


# ============================================================================
# Utility Functions for Route Protection
# ============================================================================

def check_route_permission(path: str, user: User) -> bool:
    """Check if user has permission for a specific route."""
    for route_prefix, required_perms in PROTECTED_ROUTES.items():
        if path.startswith(route_prefix):
            user_perms = set(user.permissions)
            for perm in required_perms:
                if perm.value not in user_perms and Permission.ADMIN_FULL.value not in user_perms:
                    return False
    return True


def get_protected_routes() -> dict:
    """Get the list of protected routes and their required permissions."""
    return {k: [p.value for p in v] for k, v in PROTECTED_ROUTES.items()}
