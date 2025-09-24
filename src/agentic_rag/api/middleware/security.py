"""
Security headers middleware for Agentic RAG API.

This middleware adds comprehensive security headers to all responses
following OWASP security best practices.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from agentic_rag.config import get_settings

logger = structlog.get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding comprehensive security headers.

    Implements OWASP-recommended security headers for web application security.
    """

    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next):
        """Add comprehensive security headers to response."""

        response = await call_next(request)

        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
            "img-src 'self' data: https:",
            "font-src 'self' https://cdn.jsdelivr.net",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]

        if self.settings.environment == "development":
            # More permissive CSP for development
            csp_directives = [
                "default-src 'self'",
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:",
                "style-src 'self' 'unsafe-inline' https:",
                "img-src 'self' data: https:",
                "font-src 'self' https:",
                "connect-src 'self' ws: wss:",
                "frame-ancestors 'none'",
                "base-uri 'self'",
                "form-action 'self'"
            ]

        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # XSS Protection (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HSTS (only in production with HTTPS)
        if self.settings.environment == "production" or request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions Policy (Feature Policy replacement)
        permissions_policy = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "gyroscope=()",
            "speaker=()",
            "vibrate=()",
            "fullscreen=(self)",
            "sync-xhr=()"
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions_policy)

        # Cross-Origin Policies
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"

        # Cache Control for sensitive endpoints
        if self._is_sensitive_endpoint(request.url.path):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        # Server header removal (don't reveal server info)
        if "Server" in response.headers:
            del response.headers["Server"]

        # Add custom security headers
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Security-Headers"] = "enabled"

        logger.debug(
            "Security headers applied",
            path=request.url.path,
            method=request.method,
            sensitive=self._is_sensitive_endpoint(request.url.path)
        )

        return response

    def _is_sensitive_endpoint(self, path: str) -> bool:
        """Check if endpoint contains sensitive data that shouldn't be cached."""
        sensitive_patterns = [
            "/auth/",
            "/users/",
            "/admin/",
            "/api/v1/users",
            "/api/v1/auth"
        ]
        return any(pattern in path for pattern in sensitive_patterns)
