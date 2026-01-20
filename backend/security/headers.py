"""
Security Headers Middleware

Adds security headers to HTTP responses.
"""

from typing import Dict, List, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all responses.
    
    Headers added:
    - Content-Security-Policy
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Strict-Transport-Security
    - Referrer-Policy
    - Permissions-Policy
    """
    
    def __init__(
        self,
        app,
        csp_policy: Optional[str] = None,
        hsts_max_age: int = 31536000,  # 1 year
        frame_options: str = "DENY",
        content_type_options: str = "nosniff",
        referrer_policy: str = "strict-origin-when-cross-origin",
        custom_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize middleware.
        
        Args:
            app: ASGI application
            csp_policy: Content Security Policy
            hsts_max_age: HSTS max age in seconds
            frame_options: X-Frame-Options value
            content_type_options: X-Content-Type-Options value
            referrer_policy: Referrer-Policy value
            custom_headers: Additional custom headers
        """
        super().__init__(app)
        
        self.headers = {
            "X-Content-Type-Options": content_type_options,
            "X-Frame-Options": frame_options,
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": referrer_policy,
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }
        
        # CSP
        if csp_policy:
            self.headers["Content-Security-Policy"] = csp_policy
        else:
            self.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' https:; "
                "connect-src 'self' https: wss:; "
                "frame-ancestors 'none';"
            )
        
        # HSTS
        if hsts_max_age > 0:
            self.headers["Strict-Transport-Security"] = (
                f"max-age={hsts_max_age}; includeSubDomains; preload"
            )
        
        # Custom headers
        if custom_headers:
            self.headers.update(custom_headers)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        for name, value in self.headers.items():
            response.headers[name] = value
        
        return response


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware with security focus.
    """
    
    def __init__(
        self,
        app,
        allowed_origins: Optional[List[str]] = None,
        allowed_methods: Optional[List[str]] = None,
        allowed_headers: Optional[List[str]] = None,
        allow_credentials: bool = False,
        max_age: int = 86400,  # 24 hours
    ):
        super().__init__(app)
        
        self.allowed_origins = set(allowed_origins or [])
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or [
            "Accept",
            "Accept-Language",
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Request-ID",
        ]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle CORS."""
        origin = request.headers.get("origin")
        
        # Handle preflight
        if request.method == "OPTIONS":
            return self._preflight_response(origin)
        
        response = await call_next(request)
        
        # Add CORS headers
        if origin and self._is_allowed_origin(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
    
    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins
    
    def _preflight_response(self, origin: Optional[str]) -> Response:
        """Generate preflight response."""
        headers = {}
        
        if origin and self._is_allowed_origin(origin):
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
            headers["Access-Control-Max-Age"] = str(self.max_age)
            
            if self.allow_credentials:
                headers["Access-Control-Allow-Credentials"] = "true"
        
        return Response(status_code=204, headers=headers)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds request ID for tracing.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add request ID."""
        import uuid
        
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state
        request.state.request_id = request_id
        
        response = await call_next(request)
        
        # Add to response
        response.headers["X-Request-ID"] = request_id
        
        return response
