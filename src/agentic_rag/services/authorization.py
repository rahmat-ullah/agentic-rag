"""
Authorization service for Agentic RAG System.

This module provides role-based authorization and permission checking.
"""

import uuid
from enum import Enum
from typing import List, Set, Dict, Any, Optional

import structlog

from agentic_rag.models.database import User, UserRole
from agentic_rag.api.exceptions import AuthorizationError

logger = structlog.get_logger(__name__)


class Permission(str, Enum):
    """System permissions enumeration."""
    
    # User management
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Document management
    DOCUMENT_READ = "document:read"
    DOCUMENT_CREATE = "document:create"
    DOCUMENT_UPDATE = "document:update"
    DOCUMENT_DELETE = "document:delete"
    
    # Offer management
    OFFER_READ = "offer:read"
    OFFER_CREATE = "offer:create"
    OFFER_UPDATE = "offer:update"
    OFFER_DELETE = "offer:delete"
    
    # Feedback management
    FEEDBACK_READ = "feedback:read"
    FEEDBACK_CREATE = "feedback:create"
    FEEDBACK_UPDATE = "feedback:update"
    FEEDBACK_DELETE = "feedback:delete"
    
    # System administration
    SYSTEM_ADMIN = "system:admin"
    TENANT_ADMIN = "tenant:admin"
    
    # Analytics and reporting
    ANALYTICS_READ = "analytics:read"
    REPORTS_GENERATE = "reports:generate"


class AuthorizationService:
    """Service for handling authorization and permissions."""
    
    def __init__(self):
        # Define role-based permissions
        self.role_permissions: Dict[UserRole, Set[Permission]] = {
            UserRole.VIEWER: {
                Permission.DOCUMENT_READ,
                Permission.OFFER_READ,
                Permission.FEEDBACK_READ,
                Permission.ANALYTICS_READ,
            },
            UserRole.ANALYST: {
                Permission.DOCUMENT_READ,
                Permission.DOCUMENT_CREATE,
                Permission.DOCUMENT_UPDATE,
                Permission.OFFER_READ,
                Permission.OFFER_CREATE,
                Permission.OFFER_UPDATE,
                Permission.FEEDBACK_READ,
                Permission.FEEDBACK_CREATE,
                Permission.FEEDBACK_UPDATE,
                Permission.ANALYTICS_READ,
                Permission.REPORTS_GENERATE,
            },
            UserRole.ADMIN: {
                # Admins have all permissions
                Permission.USER_READ,
                Permission.USER_CREATE,
                Permission.USER_UPDATE,
                Permission.USER_DELETE,
                Permission.DOCUMENT_READ,
                Permission.DOCUMENT_CREATE,
                Permission.DOCUMENT_UPDATE,
                Permission.DOCUMENT_DELETE,
                Permission.OFFER_READ,
                Permission.OFFER_CREATE,
                Permission.OFFER_UPDATE,
                Permission.OFFER_DELETE,
                Permission.FEEDBACK_READ,
                Permission.FEEDBACK_CREATE,
                Permission.FEEDBACK_UPDATE,
                Permission.FEEDBACK_DELETE,
                Permission.SYSTEM_ADMIN,
                Permission.TENANT_ADMIN,
                Permission.ANALYTICS_READ,
                Permission.REPORTS_GENERATE,
            }
        }
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for a user based on their role."""
        return self.role_permissions.get(user.role, set())
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
    
    def has_any_permission(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        user_permissions = self.get_user_permissions(user)
        return any(perm in user_permissions for perm in permissions)
    
    def has_all_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions."""
        user_permissions = self.get_user_permissions(user)
        return all(perm in user_permissions for perm in permissions)
    
    def require_permission(self, user: User, permission: Permission) -> None:
        """Require user to have a specific permission, raise exception if not."""
        if not self.has_permission(user, permission):
            logger.warning(
                "Permission denied",
                user_id=str(user.id),
                user_role=user.role.value,
                required_permission=permission.value
            )
            raise AuthorizationError(f"Permission '{permission.value}' required")
    
    def require_any_permission(self, user: User, permissions: List[Permission]) -> None:
        """Require user to have any of the specified permissions."""
        if not self.has_any_permission(user, permissions):
            permission_names = [p.value for p in permissions]
            logger.warning(
                "Permission denied - none of required permissions",
                user_id=str(user.id),
                user_role=user.role.value,
                required_permissions=permission_names
            )
            raise AuthorizationError(f"One of these permissions required: {', '.join(permission_names)}")
    
    def require_all_permissions(self, user: User, permissions: List[Permission]) -> None:
        """Require user to have all of the specified permissions."""
        if not self.has_all_permissions(user, permissions):
            permission_names = [p.value for p in permissions]
            logger.warning(
                "Permission denied - missing required permissions",
                user_id=str(user.id),
                user_role=user.role.value,
                required_permissions=permission_names
            )
            raise AuthorizationError(f"All of these permissions required: {', '.join(permission_names)}")
    
    def require_role(self, user: User, required_role: UserRole) -> None:
        """Require user to have a specific role or higher."""
        role_hierarchy = {
            UserRole.VIEWER: 1,
            UserRole.ANALYST: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 999)
        
        if user_level < required_level:
            logger.warning(
                "Role requirement not met",
                user_id=str(user.id),
                user_role=user.role.value,
                required_role=required_role.value
            )
            raise AuthorizationError(f"Role '{required_role.value}' or higher required")
    
    def require_same_tenant(self, user: User, resource_tenant_id: uuid.UUID) -> None:
        """Require user to be in the same tenant as the resource."""
        if user.tenant_id != resource_tenant_id:
            logger.warning(
                "Tenant access denied",
                user_id=str(user.id),
                user_tenant_id=str(user.tenant_id),
                resource_tenant_id=str(resource_tenant_id)
            )
            raise AuthorizationError("Access denied: resource belongs to different tenant")
    
    def require_resource_owner_or_permission(
        self, 
        user: User, 
        resource_owner_id: uuid.UUID, 
        permission: Permission
    ) -> None:
        """Require user to be the resource owner or have specific permission."""
        if user.id == resource_owner_id:
            return  # User owns the resource
        
        if self.has_permission(user, permission):
            return  # User has required permission
        
        logger.warning(
            "Resource access denied",
            user_id=str(user.id),
            resource_owner_id=str(resource_owner_id),
            required_permission=permission.value
        )
        raise AuthorizationError("Access denied: must be resource owner or have required permission")
    
    def can_access_user_data(self, current_user: User, target_user_id: uuid.UUID) -> bool:
        """Check if current user can access another user's data."""
        # Users can always access their own data
        if current_user.id == target_user_id:
            return True
        
        # Admins can access any user data in their tenant
        if current_user.role == UserRole.ADMIN:
            return True
        
        return False
    
    def can_modify_user(self, current_user: User, target_user: User) -> bool:
        """Check if current user can modify another user."""
        # Users cannot modify themselves (except through specific endpoints)
        if current_user.id == target_user.id:
            return False
        
        # Only admins can modify other users
        if current_user.role != UserRole.ADMIN:
            return False
        
        # Admins can only modify users in their tenant
        if current_user.tenant_id != target_user.tenant_id:
            return False
        
        # Admins cannot modify other admins (prevent privilege escalation)
        if target_user.role == UserRole.ADMIN:
            return False
        
        return True


# Global authorization service instance
authorization_service = AuthorizationService()


def get_authorization_service() -> AuthorizationService:
    """Get authorization service instance."""
    return authorization_service
