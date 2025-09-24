"""
Demo endpoints for testing authorization and permissions.

This module provides demonstration endpoints to test different
authorization levels and permission requirements.
"""

from fastapi import APIRouter, Depends, Request
import structlog

from agentic_rag.models.database import User, UserRole, Tenant
from agentic_rag.api.models.responses import SuccessResponse
from agentic_rag.api.dependencies.auth import (
    get_current_user,
    require_permission,
    require_any_permission,
    require_user_role,
    get_optional_current_user,
    get_current_tenant,
    get_effective_tenant_id
)
from agentic_rag.services.authorization import Permission

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/public", response_model=SuccessResponse[dict])
async def public_endpoint():
    """
    Public endpoint that doesn't require authentication.
    
    This endpoint is accessible to anyone without authentication.
    """
    return SuccessResponse(
        message="Public endpoint accessed successfully",
        data={"access_level": "public", "authentication_required": False}
    )


@router.get("/authenticated", response_model=SuccessResponse[dict])
async def authenticated_endpoint(
    current_user: User = Depends(get_current_user)
):
    """
    Authenticated endpoint that requires any valid user.
    
    This endpoint requires authentication but no specific permissions.
    """
    return SuccessResponse(
        message="Authenticated endpoint accessed successfully",
        data={
            "access_level": "authenticated",
            "user_id": str(current_user.id),
            "user_role": current_user.role.value,
            "tenant_id": str(current_user.tenant_id)
        }
    )


@router.get("/viewer-level", response_model=SuccessResponse[dict])
async def viewer_level_endpoint(
    current_user: User = Depends(require_user_role(UserRole.VIEWER))
):
    """
    Viewer level endpoint (accessible to viewer, analyst, admin).
    
    This endpoint requires at least viewer role.
    """
    return SuccessResponse(
        message="Viewer level endpoint accessed successfully",
        data={
            "access_level": "viewer",
            "minimum_role": "viewer",
            "user_role": current_user.role.value
        }
    )


@router.get("/analyst-level", response_model=SuccessResponse[dict])
async def analyst_level_endpoint(
    current_user: User = Depends(require_user_role(UserRole.ANALYST))
):
    """
    Analyst level endpoint (accessible to analyst, admin).
    
    This endpoint requires at least analyst role.
    """
    return SuccessResponse(
        message="Analyst level endpoint accessed successfully",
        data={
            "access_level": "analyst",
            "minimum_role": "analyst",
            "user_role": current_user.role.value
        }
    )


@router.get("/admin-level", response_model=SuccessResponse[dict])
async def admin_level_endpoint(
    current_user: User = Depends(require_user_role(UserRole.ADMIN))
):
    """
    Admin level endpoint (accessible to admin only).
    
    This endpoint requires admin role.
    """
    return SuccessResponse(
        message="Admin level endpoint accessed successfully",
        data={
            "access_level": "admin",
            "minimum_role": "admin",
            "user_role": current_user.role.value
        }
    )


@router.get("/document-read", response_model=SuccessResponse[dict])
async def document_read_endpoint(
    current_user: User = Depends(require_permission(Permission.DOCUMENT_READ))
):
    """
    Document read endpoint (requires DOCUMENT_READ permission).
    
    This endpoint requires specific permission rather than role.
    """
    return SuccessResponse(
        message="Document read endpoint accessed successfully",
        data={
            "access_level": "permission-based",
            "required_permission": "document:read",
            "user_role": current_user.role.value
        }
    )


@router.get("/document-write", response_model=SuccessResponse[dict])
async def document_write_endpoint(
    current_user: User = Depends(require_any_permission([
        Permission.DOCUMENT_CREATE,
        Permission.DOCUMENT_UPDATE
    ]))
):
    """
    Document write endpoint (requires DOCUMENT_CREATE or DOCUMENT_UPDATE).
    
    This endpoint requires any of multiple permissions.
    """
    return SuccessResponse(
        message="Document write endpoint accessed successfully",
        data={
            "access_level": "permission-based",
            "required_permissions": ["document:create", "document:update"],
            "user_role": current_user.role.value
        }
    )


@router.get("/user-management", response_model=SuccessResponse[dict])
async def user_management_endpoint(
    current_user: User = Depends(require_permission(Permission.USER_READ))
):
    """
    User management endpoint (requires USER_READ permission).
    
    This endpoint requires admin-only permission.
    """
    return SuccessResponse(
        message="User management endpoint accessed successfully",
        data={
            "access_level": "admin-only",
            "required_permission": "user:read",
            "user_role": current_user.role.value
        }
    )


@router.get("/optional-auth", response_model=SuccessResponse[dict])
async def optional_auth_endpoint(
    current_user: User = Depends(get_optional_current_user)
):
    """
    Optional authentication endpoint.
    
    This endpoint works for both authenticated and unauthenticated users.
    """
    if current_user:
        return SuccessResponse(
            message="Optional auth endpoint accessed by authenticated user",
            data={
                "access_level": "optional-auth",
                "authenticated": True,
                "user_id": str(current_user.id),
                "user_role": current_user.role.value
            }
        )
    else:
        return SuccessResponse(
            message="Optional auth endpoint accessed by anonymous user",
            data={
                "access_level": "optional-auth",
                "authenticated": False
            }
        )


@router.get("/tenant-context", response_model=SuccessResponse[dict])
async def tenant_context_endpoint(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Tenant context demonstration endpoint.

    Shows how tenant context is extracted and validated.
    """
    # Get tenant information
    current_tenant = get_current_tenant(request)
    effective_tenant_id = get_effective_tenant_id(request)

    tenant_info = None
    if current_tenant:
        tenant_info = {
            "id": str(current_tenant.id),
            "name": current_tenant.name,
            "is_active": current_tenant.is_active,
            "created_at": current_tenant.created_at.isoformat()
        }

    return SuccessResponse(
        message="Tenant context information retrieved",
        data={
            "access_level": "tenant-aware",
            "user_tenant_id": str(current_user.tenant_id),
            "effective_tenant_id": str(effective_tenant_id),
            "tenant_info": tenant_info,
            "tenant_isolation_active": current_tenant is not None
        }
    )
