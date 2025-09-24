"""
Database dependencies for FastAPI routes.

This module provides dependency functions for database session management
and tenant context handling in API endpoints.
"""

from typing import Generator
from uuid import UUID

from fastapi import Depends
from sqlalchemy.orm import Session

from agentic_rag.adapters.database import DatabaseAdapter
from agentic_rag.api.dependencies.auth import get_current_user
from agentic_rag.models.database import User


# Global database adapter instance
_db_adapter: DatabaseAdapter = None


def get_database_adapter() -> DatabaseAdapter:
    """Get database adapter instance."""
    global _db_adapter
    if _db_adapter is None:
        _db_adapter = DatabaseAdapter()
    return _db_adapter


def get_db_session() -> Generator[Session, None, None]:
    """Get database session dependency."""
    db_adapter = get_database_adapter()
    with db_adapter.get_session() as session:
        yield session


def get_tenant_db_session(
    current_user: User = Depends(get_current_user)
) -> Generator[Session, None, None]:
    """Get database session with tenant context."""
    db_adapter = get_database_adapter()
    with db_adapter.get_tenant_session(
        tenant_id=current_user.tenant_id,
        user_role=current_user.role.value
    ) as session:
        yield session
