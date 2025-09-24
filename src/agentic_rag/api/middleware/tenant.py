"""
Tenant context middleware for Agentic RAG API.

This middleware handles multi-tenant context extraction and validation.
"""

import uuid
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette import status
import structlog

from agentic_rag.adapters.database import get_database_adapter
from agentic_rag.models.database import Tenant

logger = structlog.get_logger(__name__)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Middleware for handling multi-tenant context."""

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
        self.db = get_database_adapter()

    def _extract_tenant_id(self, request: Request) -> Optional[uuid.UUID]:
        """
        Extract tenant ID from request.

        Tries multiple sources in order:
        1. X-Tenant-ID header
        2. tenant_id query parameter
        3. User's tenant_id from authentication context (if available)
        """
        # Try header first
        tenant_header = request.headers.get("X-Tenant-ID")
        if tenant_header:
            try:
                return uuid.UUID(tenant_header)
            except ValueError:
                logger.warning("Invalid tenant ID in header", tenant_id=tenant_header)

        # Try query parameter
        tenant_param = request.query_params.get("tenant_id")
        if tenant_param:
            try:
                return uuid.UUID(tenant_param)
            except ValueError:
                logger.warning("Invalid tenant ID in query param", tenant_id=tenant_param)

        # Try user context (set by auth middleware)
        user_tenant_id = getattr(request.state, "tenant_id", None)
        if user_tenant_id:
            return user_tenant_id

        return None

    def _validate_tenant(self, tenant_id: uuid.UUID) -> Optional[Tenant]:
        """Validate that tenant exists and is active."""
        try:
            with self.db.get_session() as session:
                tenant = session.query(Tenant).filter(
                    Tenant.id == tenant_id,
                    Tenant.is_active == True
                ).first()
                return tenant
        except Exception as e:
            logger.error("Error validating tenant", error=str(e), tenant_id=str(tenant_id))
            return None

    async def dispatch(self, request: Request, call_next):
        """Process tenant context for incoming requests."""

        # Skip tenant validation for public paths
        if request.url.path in self.public_paths or request.url.path.startswith("/health"):
            return await call_next(request)

        # Extract tenant ID
        tenant_id = self._extract_tenant_id(request)

        if not tenant_id:
            # For authenticated endpoints, tenant should be available from user context
            # For now, we'll allow requests without explicit tenant ID if user context exists
            user_id = getattr(request.state, "user_id", None)
            if user_id:
                # User is authenticated, tenant context should be available
                logger.debug("No explicit tenant ID, using user's tenant context", path=request.url.path)
                return await call_next(request)
            else:
                logger.warning("Missing tenant context", path=request.url.path)
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "message": "Tenant context required. Provide X-Tenant-ID header or tenant_id parameter.",
                        "error_code": "MISSING_TENANT_CONTEXT"
                    }
                )

        # Validate tenant
        tenant = self._validate_tenant(tenant_id)
        if not tenant:
            logger.warning("Invalid or inactive tenant", tenant_id=str(tenant_id), path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "success": False,
                    "message": "Invalid or inactive tenant",
                    "error_code": "INVALID_TENANT"
                }
            )

        # Add tenant context to request state
        request.state.tenant = tenant
        request.state.explicit_tenant_id = tenant_id

        # Validate tenant isolation if user is authenticated
        user_tenant_id = getattr(request.state, "tenant_id", None)
        if user_tenant_id and user_tenant_id != tenant_id:
            logger.warning(
                "Tenant isolation violation",
                user_tenant_id=str(user_tenant_id),
                requested_tenant_id=str(tenant_id),
                path=request.url.path
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "success": False,
                    "message": "Access denied: tenant isolation violation",
                    "error_code": "TENANT_ISOLATION_VIOLATION"
                }
            )

        logger.debug(
            "Tenant context validated",
            tenant_id=str(tenant_id),
            tenant_name=tenant.name,
            path=request.url.path
        )

        response = await call_next(request)

        # Add tenant context to response headers
        response.headers["X-Tenant-ID"] = str(tenant_id)
        response.headers["X-Tenant-Name"] = tenant.name

        return response
