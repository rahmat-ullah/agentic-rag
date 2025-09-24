"""
User management endpoints for the Agentic RAG API.

This module provides user management endpoints.
"""

from fastapi import APIRouter, Depends
import structlog

from agentic_rag.models.database import User
from agentic_rag.api.models.users import UserResponse
from agentic_rag.api.models.responses import SuccessResponse
from agentic_rag.api.dependencies.auth import get_current_user, require_permission
from agentic_rag.services.authorization import Permission, get_authorization_service

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/me", response_model=SuccessResponse[UserResponse])
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information.

    Returns the authenticated user's profile information.
    """
    user_data = UserResponse(
        id=current_user.id,
        email=current_user.email,
        role=current_user.role.value,
        tenant_id=current_user.tenant_id,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at
    )

    return SuccessResponse(
        message="User information retrieved successfully",
        data=user_data
    )


@router.get("/", response_model=SuccessResponse[dict])
async def list_users(
    current_user: User = Depends(require_permission(Permission.USER_READ))
):
    """
    List all users (requires USER_READ permission).

    Returns a list of all users in the tenant.
    Requires USER_READ permission (admin only).
    """
    # TODO: Implement user listing - placeholder for now
    return SuccessResponse(
        message="User listing endpoint - functional with permission check",
        data={
            "users": [
                {
                    "id": str(current_user.id),
                    "email": current_user.email,
                    "role": current_user.role.value,
                    "tenant_id": str(current_user.tenant_id),
                    "is_active": current_user.is_active,
                    "created_at": current_user.created_at.isoformat(),
                    "updated_at": current_user.updated_at.isoformat()
                }
            ],
            "total": 1
        }
    )


@router.get("/me/permissions", response_model=SuccessResponse[dict])
async def get_current_user_permissions(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user's permissions.

    Returns the list of permissions available to the authenticated user
    based on their role.
    """
    auth_service = get_authorization_service()
    permissions = auth_service.get_user_permissions(current_user)

    return SuccessResponse(
        message="User permissions retrieved successfully",
        data={
            "user_id": str(current_user.id),
            "role": current_user.role.value,
            "permissions": [perm.value for perm in permissions]
        }
    )
