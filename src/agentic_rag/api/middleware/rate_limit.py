"""
Rate limiting middleware for Agentic RAG API.

This middleware implements basic rate limiting to prevent abuse.
"""

import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import structlog

from ..exceptions import RateLimitError

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60, **kwargs):
        super().__init__(app, **kwargs)
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute window
        self.request_counts: Dict[str, list] = defaultdict(list)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use X-Forwarded-For if available, otherwise use client IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_id: str) -> Tuple[bool, int]:
        """Check if client is rate limited."""
        now = time.time()
        window_start = now - self.window_size
        
        # Clean old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > window_start
        ]
        
        # Check if rate limit exceeded
        current_requests = len(self.request_counts[client_id])
        if current_requests >= self.requests_per_minute:
            return True, 0
        
        # Add current request
        self.request_counts[client_id].append(now)
        
        # Calculate remaining requests
        remaining = self.requests_per_minute - (current_requests + 1)
        return False, remaining
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        is_limited, remaining = self._is_rate_limited(client_id)
        
        if is_limited:
            logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "error_details": {
                        "requests_per_minute": self.requests_per_minute,
                        "window_size_seconds": self.window_size
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + self.window_size))
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + self.window_size))
        
        return response
