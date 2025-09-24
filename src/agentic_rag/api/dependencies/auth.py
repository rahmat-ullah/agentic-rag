"""
Authentication dependencies for FastAPI endpoints.

This module provides dependency functions for authentication and authorization.
"""

import uuid
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from agentic_rag.models.database import User, UserRole, Tenant
from agentic_rag.services.auth import get_auth_service, AuthService
from agentic_rag.services.authorization import get_authorization_service, AuthorizationService, Permission
from agentic_rag.api.exceptions import AuthenticationError, AuthorizationError

security = HTTPBearer()


def get_current_user(request: Request) -> User:
    """
    Get current authenticated user from request state.
    
    This dependency assumes the AuthenticationMiddleware has already
    validated the token and set the user in request.state.
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user


def get_current_user_id(request: Request) -> uuid.UUID:
    """Get current user ID from request state."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user_id


def get_current_tenant_id(request: Request) -> uuid.UUID:
    """Get current tenant ID from request state."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return tenant_id


def get_current_user_role(request: Request) -> str:
    """Get current user role from request state."""
    role = getattr(request.state, "user_role", None)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return role


def require_role(required_role: str):
    """
    Dependency factory for role-based authorization.
    
    Args:
        required_role: The role required to access the endpoint
        
    Returns:
        A dependency function that checks user role
    """
    def check_role(current_role: str = Depends(get_current_user_role)) -> str:
        # Define role hierarchy
        role_hierarchy = {
            "viewer": 1,
            "analyst": 2,
            "admin": 3
        }
        
        current_level = role_hierarchy.get(current_role, 0)
        required_level = role_hierarchy.get(required_role, 999)
        
        if current_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' or higher required"
            )
        
        return current_role
    
    return check_role


def require_admin(current_role: str = Depends(get_current_user_role)) -> str:
    """Dependency that requires admin role."""
    if current_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    return current_role


def require_analyst_or_admin(current_role: str = Depends(get_current_user_role)) -> str:
    """Dependency that requires analyst or admin role."""
    if current_role not in ["analyst", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analyst or admin role required"
        )
    return current_role


def require_permission(permission: Permission):
    """
    Dependency factory for permission-based authorization.

    Args:
        permission: The permission required to access the endpoint

    Returns:
        A dependency function that checks user permission
    """
    def check_permission(
        current_user: User = Depends(get_current_user),
        auth_service: AuthorizationService = Depends(get_authorization_service)
    ) -> User:
        try:
            auth_service.require_permission(current_user, permission)
            return current_user
        except AuthorizationError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )

    return check_permission


def require_any_permission(permissions: list[Permission]):
    """
    Dependency factory for requiring any of multiple permissions.

    Args:
        permissions: List of permissions, user needs at least one

    Returns:
        A dependency function that checks user permissions
    """
    def check_permissions(
        current_user: User = Depends(get_current_user),
        auth_service: AuthorizationService = Depends(get_authorization_service)
    ) -> User:
        try:
            auth_service.require_any_permission(current_user, permissions)
            return current_user
        except AuthorizationError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )

    return check_permissions


def require_all_permissions(permissions: list[Permission]):
    """
    Dependency factory for requiring all of multiple permissions.

    Args:
        permissions: List of permissions, user needs all of them

    Returns:
        A dependency function that checks user permissions
    """
    def check_permissions(
        current_user: User = Depends(get_current_user),
        auth_service: AuthorizationService = Depends(get_authorization_service)
    ) -> User:
        try:
            auth_service.require_all_permissions(current_user, permissions)
            return current_user
        except AuthorizationError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )

    return check_permissions


def require_user_role(required_role: UserRole):
    """
    Dependency factory for role-based authorization with hierarchy.

    Args:
        required_role: The minimum role required

    Returns:
        A dependency function that checks user role
    """
    def check_role(
        current_user: User = Depends(get_current_user),
        auth_service: AuthorizationService = Depends(get_authorization_service)
    ) -> User:
        try:
            auth_service.require_role(current_user, required_role)
            return current_user
        except AuthorizationError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )

    return check_role


def require_same_tenant(resource_tenant_id: uuid.UUID):
    """
    Dependency factory for tenant isolation.

    Args:
        resource_tenant_id: The tenant ID of the resource being accessed

    Returns:
        A dependency function that checks tenant access
    """
    def check_tenant(
        current_user: User = Depends(get_current_user),
        auth_service: AuthorizationService = Depends(get_authorization_service)
    ) -> User:
        try:
            auth_service.require_same_tenant(current_user, resource_tenant_id)
            return current_user
        except AuthorizationError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )

    return check_tenant


# Alternative approach using direct token validation (bypasses middleware)
def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """
    Get current user by validating JWT token directly.
    
    This is an alternative to using the middleware approach.
    Use this when you need more control over authentication.
    """
    try:
        # Verify token
        payload = auth_service.verify_token(credentials.credentials)
        
        # Get user
        user_id = uuid.UUID(payload.get("sub"))
        user = auth_service.get_user_by_id(user_id)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        return user
        
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication error"
        )


def get_optional_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    
    This is useful for endpoints that work for both authenticated
    and unauthenticated users.
    """
    # Try to get user from middleware first
    user = getattr(request.state, "user", None)
    if user:
        return user
    
    # If no middleware user and no credentials, return None
    if not credentials:
        return None
    
    # Try to authenticate with token
    try:
        auth_service = get_auth_service()
        payload = auth_service.verify_token(credentials.credentials)
        user_id = uuid.UUID(payload.get("sub"))
        user = auth_service.get_user_by_id(user_id)
        
        if user and user.is_active:
            return user
        
    except Exception:
        pass  # Ignore authentication errors for optional auth

    return None


def get_current_tenant(request: Request) -> Optional[Tenant]:
    """
    Get current tenant from request state.

    This dependency assumes the TenantContextMiddleware has already
    validated the tenant and set it in request.state.
    """
    tenant = getattr(request.state, "tenant", None)
    return tenant


def require_tenant_context(request: Request) -> Tenant:
    """
    Require tenant context to be present.

    Raises HTTPException if no tenant context is available.
    """
    tenant = get_current_tenant(request)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required"
        )
    return tenant


def get_effective_tenant_id(request: Request) -> uuid.UUID:
    """
    Get the effective tenant ID for the current request.

    Returns the tenant ID from either:
    1. Explicit tenant context (from middleware)
    2. User's tenant ID (from authentication)
    """
    # Try explicit tenant first
    explicit_tenant_id = getattr(request.state, "explicit_tenant_id", None)
    if explicit_tenant_id:
        return explicit_tenant_id

    # Fall back to user's tenant
    user_tenant_id = getattr(request.state, "tenant_id", None)
    if user_tenant_id:
        return user_tenant_id

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No tenant context available"
    )


def require_tenant_admin(
    current_user: User = Depends(get_current_user),
    auth_service: AuthorizationService = Depends(get_authorization_service)
) -> User:
    """
    Require user to have tenant admin permissions.

    This checks for TENANT_ADMIN permission.
    """
    try:
        auth_service.require_permission(current_user, Permission.TENANT_ADMIN)
        return current_user
    except AuthorizationError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


def require_tenant_isolation(resource_tenant_id: uuid.UUID):
    """
    Dependency factory for enforcing tenant isolation.

    Ensures that the current user can only access resources
    from their own tenant.
    """
    def check_tenant_isolation(
        current_user: User = Depends(get_current_user),
        auth_service: AuthorizationService = Depends(get_authorization_service)
    ) -> User:
        try:
            auth_service.require_same_tenant(current_user, resource_tenant_id)
            return current_user
        except AuthorizationError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )

    return check_tenant_isolation
