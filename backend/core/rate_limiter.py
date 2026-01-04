"""
Rate Limiting Middleware

Redis-based rate limiting for API endpoints with per-API-key limits.
"""

import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Tuple, Callable
from functools import wraps

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window: int, retry_after: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {limit} requests per {window} seconds.",
            headers={"Retry-After": str(retry_after)},
        )
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


class InMemoryRateLimiter:
    """
    In-memory rate limiter using sliding window algorithm.
    Use for development or single-instance deployments.
    """
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
        self._lock = asyncio.Lock()
    
    async def is_rate_limited(
        self,
        key: str,
        limit: int,
        window: int,
    ) -> Tuple[bool, int, int, int]:
        """
        Check if request should be rate limited.
        
        Returns:
            Tuple of (is_limited, remaining, reset_time, retry_after)
        """
        async with self._lock:
            now = time.time()
            window_start = now - window
            
            # Get or create request list for this key
            if key not in self.requests:
                self.requests[key] = []
            
            # Remove old requests outside the window
            self.requests[key] = [
                ts for ts in self.requests[key] if ts > window_start
            ]
            
            # Count requests in current window
            current_count = len(self.requests[key])
            
            if current_count >= limit:
                # Rate limited
                oldest = min(self.requests[key]) if self.requests[key] else now
                retry_after = int(oldest + window - now) + 1
                return True, 0, int(now + retry_after), retry_after
            
            # Add current request
            self.requests[key].append(now)
            remaining = limit - current_count - 1
            reset_time = int(now + window)
            
            return False, remaining, reset_time, 0
    
    async def cleanup(self):
        """Cleanup old entries."""
        async with self._lock:
            now = time.time()
            # Keep only keys with recent activity (last hour)
            cutoff = now - 3600
            self.requests = {
                k: [ts for ts in v if ts > cutoff]
                for k, v in self.requests.items()
                if any(ts > cutoff for ts in v)
            }


class RedisRateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.
    Use for distributed/multi-instance deployments.
    """
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None
    
    async def get_redis(self) -> aioredis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis
    
    async def is_rate_limited(
        self,
        key: str,
        limit: int,
        window: int,
    ) -> Tuple[bool, int, int, int]:
        """
        Check if request should be rate limited using Redis.
        Uses a sliding window counter algorithm.
        
        Returns:
            Tuple of (is_limited, remaining, reset_time, retry_after)
        """
        redis = await self.get_redis()
        now = time.time()
        window_key = f"ratelimit:{key}:{int(now // window)}"
        
        pipe = redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, window + 1)
        results = await pipe.execute()
        
        current_count = results[0]
        
        if current_count > limit:
            # Rate limited
            retry_after = int(window - (now % window)) + 1
            reset_time = int(now + retry_after)
            return True, 0, reset_time, retry_after
        
        remaining = limit - current_count
        reset_time = int((int(now // window) + 1) * window)
        
        return False, remaining, reset_time, 0
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


class RateLimitConfig:
    """Rate limit configuration."""
    
    def __init__(
        self,
        default_limit: int = 100,
        default_window: int = 60,
        redis_url: Optional[str] = None,
        enabled: bool = True,
        key_prefix: str = "ratelimit",
    ):
        self.default_limit = default_limit
        self.default_window = default_window
        self.redis_url = redis_url
        self.enabled = enabled
        self.key_prefix = key_prefix
        
        # Route-specific limits (route_pattern -> (limit, window))
        self.route_limits: Dict[str, Tuple[int, int]] = {}
        
        # API key specific limits (api_key -> (limit, window))
        self.api_key_limits: Dict[str, Tuple[int, int]] = {}
    
    def add_route_limit(self, route: str, limit: int, window: int = 60):
        """Add a route-specific rate limit."""
        self.route_limits[route] = (limit, window)
    
    def add_api_key_limit(self, api_key: str, limit: int, window: int = 60):
        """Add an API key specific rate limit."""
        self.api_key_limits[api_key] = (limit, window)
    
    def get_limit_for_request(
        self,
        path: str,
        api_key: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Get the rate limit for a specific request."""
        # Check API key specific limit first
        if api_key and api_key in self.api_key_limits:
            return self.api_key_limits[api_key]
        
        # Check route-specific limits
        for route, limit_config in self.route_limits.items():
            if path.startswith(route):
                return limit_config
        
        return self.default_limit, self.default_window


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI.
    
    Supports:
    - In-memory rate limiting (single instance)
    - Redis-based rate limiting (distributed)
    - Per-route limits
    - Per-API-key limits
    """
    
    def __init__(self, app, config: RateLimitConfig):
        super().__init__(app)
        self.config = config
        
        # Initialize rate limiter
        if config.redis_url and REDIS_AVAILABLE:
            self.limiter = RedisRateLimiter(config.redis_url)
        else:
            self.limiter = InMemoryRateLimiter()
        
        # Excluded paths (health checks, etc.)
        self.excluded_paths = {"/health", "/ready", "/", "/docs", "/openapi.json"}
    
    def _get_client_key(self, request: Request) -> str:
        """Get the client identifier for rate limiting."""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key}"
        
        # Try user from auth
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiter."""
        # Skip if disabled
        if not self.config.enabled:
            return await call_next(request)
        
        # Skip excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Get client key and limits
        client_key = self._get_client_key(request)
        api_key = request.headers.get("X-API-Key")
        limit, window = self.config.get_limit_for_request(request.url.path, api_key)
        
        # Build rate limit key
        rate_key = f"{self.config.key_prefix}:{client_key}"
        
        # Check rate limit
        is_limited, remaining, reset_time, retry_after = await self.limiter.is_rate_limited(
            rate_key, limit, window
        )
        
        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "TooManyRequests",
                    "message": f"Rate limit exceeded. Maximum {limit} requests per {window} seconds.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


def rate_limit(limit: int = 100, window: int = 60):
    """
    Decorator for route-specific rate limiting.
    
    Usage:
        @app.get("/api/heavy")
        @rate_limit(limit=10, window=60)
        async def heavy_endpoint():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from kwargs or args
            request = kwargs.get("request")
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                return await func(*args, **kwargs)
            
            # Simple in-memory check for decorator-based limiting
            client_key = request.headers.get("X-API-Key") or (
                request.client.host if request.client else "unknown"
            )
            rate_key = f"decorator:{func.__name__}:{client_key}"
            
            # Use a simple in-memory check (would need to be more sophisticated in production)
            now = time.time()
            if not hasattr(wrapper, "_requests"):
                wrapper._requests = {}
            
            # Cleanup old entries
            cutoff = now - window
            wrapper._requests = {
                k: [t for t in v if t > cutoff]
                for k, v in wrapper._requests.items()
            }
            
            # Check limit
            if rate_key not in wrapper._requests:
                wrapper._requests[rate_key] = []
            
            if len(wrapper._requests[rate_key]) >= limit:
                raise RateLimitExceeded(limit, window, int(window - (now % window)) + 1)
            
            wrapper._requests[rate_key].append(now)
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Default configuration
def create_rate_limit_config(redis_url: Optional[str] = None) -> RateLimitConfig:
    """Create default rate limit configuration."""
    config = RateLimitConfig(
        default_limit=100,
        default_window=60,
        redis_url=redis_url,
        enabled=True,
    )
    
    # Add route-specific limits
    config.add_route_limit("/api/predict", 200, 60)  # Higher limit for predictions
    config.add_route_limit("/api/predict/batch", 50, 60)  # Lower for batch
    config.add_route_limit("/api/upload", 10, 60)  # Low limit for uploads
    config.add_route_limit("/api/train", 5, 60)  # Very low for training
    config.add_route_limit("/api/deploy", 5, 60)  # Very low for deployment
    
    return config
