"""
Authentication middleware for Agentic RAG API.

This middleware handles JWT token validation and user authentication.
"""

import uuid
from typing import Optional

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import structlog

from agentic_rag.services.auth import get_auth_service
from agentic_rag.api.exceptions import AuthenticationError

logger = structlog.get_logger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware."""

    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.public_paths = {
            "/health",
            "/health/",
            "/health/ready",
            "/health/live",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/login",
            "/auth/refresh",
            "/demo/public",
            "/demo/optional-auth",
        }
        self.auth_service = get_auth_service()

    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from Authorization header."""
        authorization = request.headers.get("Authorization")
        if not authorization:
            return None

        if not authorization.startswith("Bearer "):
            return None

        return authorization[7:]  # Remove "Bearer " prefix

    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""

        # Skip authentication for public paths
        if request.url.path in self.public_paths or request.url.path.startswith("/health"):
            return await call_next(request)

        # Extract token
        token = self._extract_token(request)
        if not token:
            logger.warning("Missing authorization token", path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "success": False,
                    "message": "Authorization token required",
                    "error_code": "MISSING_TOKEN"
                }
            )

        try:
            # Verify token
            payload = self.auth_service.verify_token(token)

            # Get user information
            user_id = uuid.UUID(payload.get("sub"))
            user = self.auth_service.get_user_by_id(user_id)

            if not user or not user.is_active:
                logger.warning("User not found or inactive", user_id=str(user_id))
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "success": False,
                        "message": "User not found or inactive",
                        "error_code": "INVALID_USER"
                    }
                )

            # Add user context to request state
            request.state.user = user
            request.state.user_id = user.id
            request.state.tenant_id = user.tenant_id
            request.state.user_role = user.role.value

            logger.debug(
                "User authenticated",
                user_id=str(user.id),
                email=user.email,
                role=user.role.value,
                path=request.url.path
            )

        except AuthenticationError as e:
            logger.warning("Authentication failed", error=str(e), path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "success": False,
                    "message": str(e),
                    "error_code": "AUTHENTICATION_FAILED"
                }
            )
        except Exception as e:
            logger.error("Authentication error", error=str(e), path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "message": "Internal server error",
                    "error_code": "INTERNAL_ERROR"
                }
            )

        response = await call_next(request)
        return response
        
        return await call_next(request)
