"""
Dependencies package for FastAPI.

This package contains dependency functions for authentication,
authorization, and other cross-cutting concerns.
"""

from .auth import (
    get_current_user,
    get_current_user_id,
    get_current_tenant_id,
    get_current_user_role,
    require_role,
    require_admin,
    require_analyst_or_admin,
    get_current_user_from_token,
    get_optional_current_user,
    require_permission,
    require_any_permission,
    require_all_permissions,
    require_user_role,
    require_same_tenant,
    get_current_tenant,
    require_tenant_context,
    get_effective_tenant_id,
    require_tenant_admin,
    require_tenant_isolation,
)

__all__ = [
    "get_current_user",
    "get_current_user_id",
    "get_current_tenant_id",
    "get_current_user_role",
    "require_role",
    "require_admin",
    "require_analyst_or_admin",
    "get_current_user_from_token",
    "get_optional_current_user",
    "require_permission",
    "require_any_permission",
    "require_all_permissions",
    "require_user_role",
    "require_same_tenant",
    "get_current_tenant",
    "require_tenant_context",
    "get_effective_tenant_id",
    "require_tenant_admin",
    "require_tenant_isolation",
]
