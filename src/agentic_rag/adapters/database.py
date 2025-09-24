"""
Database adapter for Agentic RAG System.

This module provides database connection management, session handling,
and tenant context management for the multi-tenant system.
"""

import uuid
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from agentic_rag.config import get_settings
from agentic_rag.models.database import Base


class DatabaseAdapter:
    """Database adapter for managing connections and sessions."""
    
    def __init__(self):
        """Initialize database adapter."""
        self.settings = get_settings()
        self._engine = None
        self._session_factory = None
    
    @property
    def engine(self):
        """Get database engine."""
        if self._engine is None:
            self._engine = create_engine(
                str(self.settings.database.postgres_url),
                pool_size=self.settings.database.db_pool_size,
                max_overflow=self.settings.database.db_max_overflow,
                pool_timeout=self.settings.database.db_pool_timeout,
                pool_recycle=self.settings.database.db_pool_recycle,
                echo=self.settings.database.db_echo,
                # Use StaticPool for testing
                poolclass=StaticPool if self.settings.is_testing else None,
            )
        return self._engine
    
    @property
    def session_factory(self):
        """Get session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_tenant_session(
        self, 
        tenant_id: uuid.UUID, 
        user_role: str = "viewer"
    ) -> Generator[Session, None, None]:
        """Get database session with tenant context set."""
        session = self.session_factory()
        try:
            # Set tenant context
            self.set_tenant_context(session, tenant_id, user_role)
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def set_tenant_context(
        self, 
        session: Session, 
        tenant_id: uuid.UUID, 
        user_role: str = "viewer"
    ):
        """Set tenant context for Row-Level Security."""
        session.execute(
            text("SELECT set_current_tenant_id(:tenant_id)"),
            {"tenant_id": str(tenant_id)}
        )
        session.execute(
            text("SELECT set_current_user_role(:user_role)"),
            {"user_role": user_role}
        )
    
    def clear_tenant_context(self, session: Session):
        """Clear tenant context."""
        session.execute(
            text("SELECT set_current_tenant_id('00000000-0000-0000-0000-000000000000'::UUID)")
        )
        session.execute(
            text("SELECT set_current_user_role('viewer')")
        )
    
    def get_document_stats(self, session: Session, tenant_id: uuid.UUID) -> list:
        """Get document statistics for a tenant."""
        result = session.execute(
            text("SELECT * FROM get_document_stats(:tenant_id)"),
            {"tenant_id": str(tenant_id)}
        )
        return result.fetchall()
    
    def get_chunk_stats(self, session: Session, tenant_id: uuid.UUID) -> list:
        """Get chunk statistics for a tenant."""
        result = session.execute(
            text("SELECT * FROM get_chunk_stats(:tenant_id)"),
            {"tenant_id": str(tenant_id)}
        )
        return result.fetchall()
    
    def get_feedback_summary(self, session: Session, tenant_id: uuid.UUID) -> list:
        """Get feedback summary for a tenant."""
        result = session.execute(
            text("SELECT * FROM get_feedback_summary(:tenant_id)"),
            {"tenant_id": str(tenant_id)}
        )
        return result.fetchall()
    
    def cleanup_retired_chunks(
        self, 
        session: Session, 
        tenant_id: uuid.UUID, 
        days_old: int = 30
    ) -> int:
        """Clean up retired chunks older than specified days."""
        result = session.execute(
            text("SELECT cleanup_retired_chunks(:tenant_id, :days_old)"),
            {"tenant_id": str(tenant_id), "days_old": days_old}
        )
        return result.scalar()
    
    def update_document_link_confidence(
        self,
        session: Session,
        tenant_id: uuid.UUID,
        rfq_id: uuid.UUID,
        offer_id: uuid.UUID,
        offer_type: str,
        new_confidence: float
    ) -> bool:
        """Update document link confidence."""
        result = session.execute(
            text("""
                SELECT update_document_link_confidence(
                    :tenant_id, :rfq_id, :offer_id, :offer_type, :new_confidence
                )
            """),
            {
                "tenant_id": str(tenant_id),
                "rfq_id": str(rfq_id),
                "offer_id": str(offer_id),
                "offer_type": offer_type,
                "new_confidence": new_confidence
            }
        )
        return result.scalar()
    
    def get_next_document_version(
        self, 
        session: Session, 
        tenant_id: uuid.UUID, 
        sha256: str
    ) -> int:
        """Get next version number for a document."""
        result = session.execute(
            text("SELECT get_next_document_version(:tenant_id, :sha256)"),
            {"tenant_id": str(tenant_id), "sha256": sha256}
        )
        return result.scalar()
    
    def health_check(self) -> bool:
        """Check database health."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception:
            return False


# Global database adapter instance
db_adapter = DatabaseAdapter()


def get_database_adapter() -> DatabaseAdapter:
    """Get database adapter instance."""
    return db_adapter
