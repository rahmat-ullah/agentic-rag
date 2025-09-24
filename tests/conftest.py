"""
Global pytest configuration and fixtures for Agentic RAG System tests.

This module provides shared fixtures, test configuration, and utilities
that are available to all test modules.
"""

import asyncio
import os
import pytest
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import Mock

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from agentic_rag.config import get_settings, Settings
from agentic_rag.models.database import Base
from agentic_rag.adapters.database import get_database_adapter
from agentic_rag.api.app import create_app


# Test configuration - will be updated if a working connection is found
TEST_DATABASE_URL = "sqlite:///./test_agentic_rag.db"
TEST_DATABASE_URL_ASYNC = "sqlite+aiosqlite:///./test_agentic_rag.db"

# Check if database is available
def is_database_available():
    """Check if test database is available."""
    global TEST_DATABASE_URL, TEST_DATABASE_URL_ASYNC

    try:
        # First try PostgreSQL connections
        import psycopg2
        connection_options = [
            "postgresql://agentic_rag_app:app_password@localhost:5432/agentic_rag_test",
            "postgresql://postgres@localhost:5432/agentic_rag_test",  # No password
            "postgresql://postgres:postgres@localhost:5432/agentic_rag_test",  # With password
        ]

        for url in connection_options:
            try:
                conn = psycopg2.connect(url)
                conn.close()
                # Update the global URLs to use the working PostgreSQL connection
                TEST_DATABASE_URL = url
                TEST_DATABASE_URL_ASYNC = url.replace("postgresql://", "postgresql+asyncpg://")
                return True
            except Exception:
                continue

        # If PostgreSQL fails, fall back to SQLite (always available)
        import sqlite3
        TEST_DATABASE_URL = "sqlite:///./test_agentic_rag.db"
        TEST_DATABASE_URL_ASYNC = "sqlite+aiosqlite:///./test_agentic_rag.db"
        return True

    except Exception:
        # Final fallback to SQLite
        TEST_DATABASE_URL = "sqlite:///./test_agentic_rag.db"
        TEST_DATABASE_URL_ASYNC = "sqlite+aiosqlite:///./test_agentic_rag.db"
        return True

DATABASE_AVAILABLE = is_database_available()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings with overrides for testing environment."""
    # Override environment variables for testing
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["POSTGRES_URL"] = TEST_DATABASE_URL
    os.environ["POSTGRES_DB"] = "agentic_rag_test"
    
    # Create settings instance
    settings = get_settings()
    settings.environment = "testing"
    settings.debug = True
    settings.database.postgres_url = TEST_DATABASE_URL
    settings.database.postgres_db = "agentic_rag_test"
    
    return settings


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    if not DATABASE_AVAILABLE:
        pytest.skip("Database not available for integration tests")

    engine = create_engine(
        TEST_DATABASE_URL,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True,
        pool_recycle=300,
    )
    return engine


@pytest.fixture(scope="session")
def test_async_engine():
    """Create test async database engine."""
    if not DATABASE_AVAILABLE:
        pytest.skip("Database not available for integration tests")

    engine = create_async_engine(
        TEST_DATABASE_URL_ASYNC,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True,
        pool_recycle=300,
    )
    return engine


@pytest.fixture(scope="session")
def setup_test_database(test_engine):
    """Set up test database schema."""
    if not DATABASE_AVAILABLE:
        pytest.skip("Database not available for integration tests")

    # Create all tables
    Base.metadata.create_all(bind=test_engine)

    # Run any additional setup SQL (PostgreSQL only)
    if "postgresql" in str(test_engine.url):
        with test_engine.connect() as conn:
            # Enable necessary extensions
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"pg_stat_statements\""))
            conn.commit()

    yield

    # Cleanup: Drop all tables
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session(test_engine, setup_test_database):
    """Create a database session for testing with automatic rollback."""
    connection = test_engine.connect()
    transaction = connection.begin()
    
    # Create session
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    
    yield session
    
    # Rollback transaction and close connection
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
async def async_db_session(test_async_engine, setup_test_database):
    """Create an async database session for testing with automatic rollback."""
    async with test_async_engine.connect() as connection:
        async with connection.begin() as transaction:
            # Create async session
            AsyncSessionLocal = sessionmaker(
                bind=connection,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            async with AsyncSessionLocal() as session:
                yield session
                
            # Transaction will be rolled back automatically


@pytest.fixture
def mock_database_adapter():
    """Create a mock database adapter for unit tests."""
    mock_adapter = Mock()
    mock_adapter.health_check.return_value = True
    mock_adapter.get_session.return_value = Mock()
    return mock_adapter


@pytest.fixture
def test_app(test_settings, mock_database_adapter):
    """Create FastAPI test application."""
    app = create_app()
    
    # Override dependencies for testing
    from agentic_rag.adapters.database import get_database_adapter
    app.dependency_overrides[get_database_adapter] = lambda: mock_database_adapter
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create FastAPI test client."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
async def async_test_client(test_app):
    """Create async FastAPI test client."""
    async with httpx.AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_env_vars():
    """Provide sample environment variables for testing."""
    return {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "SECRET_KEY": "test-secret-key",
        "POSTGRES_URL": TEST_DATABASE_URL,
        "JWT_SECRET_KEY": "test-jwt-secret",
        "JWT_ALGORITHM": "HS256",
    }


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.slow = pytest.mark.slow
pytest.mark.database = pytest.mark.database
pytest.mark.api = pytest.mark.api


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_response_success(response, expected_status=200):
        """Assert that a response is successful."""
        assert response.status_code == expected_status
        assert response.json().get("success") is True
    
    @staticmethod
    def assert_response_error(response, expected_status=400):
        """Assert that a response contains an error."""
        assert response.status_code == expected_status
        assert response.json().get("success") is False
    
    @staticmethod
    def create_auth_headers(token: str) -> dict:
        """Create authorization headers for API requests."""
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils


# Test Data Factories and Utilities
class TestDataFactory:
    """Factory for creating test data objects."""

    @staticmethod
    def create_test_user(db_session, tenant_id=None, email="test@example.com", role="viewer"):
        """Create a test user."""
        from agentic_rag.models.database import User, UserRole
        import uuid

        if tenant_id is None:
            tenant_id = uuid.uuid4()

        user = User(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            email=email,
            password_hash="$2b$12$test_hash",  # bcrypt hash for "password"
            role=UserRole(role),
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        return user

    @staticmethod
    def create_test_tenant(db_session, name="Test Tenant", description="Test tenant description"):
        """Create a test tenant."""
        from agentic_rag.models.database import Tenant
        import uuid

        tenant = Tenant(
            id=uuid.uuid4(),
            name=name,
            description=description,
            is_active=True
        )
        db_session.add(tenant)
        db_session.commit()
        return tenant

    @staticmethod
    def create_test_document(db_session, tenant_id, title="Test Document", kind="RFQ", created_by=None):
        """Create a test document."""
        from agentic_rag.models.database import Document, DocumentKind
        import uuid

        document = Document(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            kind=DocumentKind(kind),
            title=title,
            source_uri="s3://test-bucket/test-doc.pdf",
            sha256="test_hash_" + str(uuid.uuid4())[:8],
            version=1,
            pages=10,
            created_by=created_by
        )
        db_session.add(document)
        db_session.commit()
        return document

    @staticmethod
    def create_test_chunk(db_session, tenant_id, document_id, page_from=1, page_to=1):
        """Create a test chunk."""
        from agentic_rag.models.database import ChunkMeta
        import uuid

        chunk = ChunkMeta(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            document_id=document_id,
            page_from=page_from,
            page_to=page_to,
            section_path=["Section 1", "Introduction"],
            token_count=100,
            hash="chunk_hash_" + str(uuid.uuid4())[:8],
            is_table=False,
            retired=False,
            embedding_model_version="text-embedding-ada-002"
        )
        db_session.add(chunk)
        db_session.commit()
        return chunk

    @staticmethod
    def create_test_feedback(db_session, tenant_id, query="Test query", label="UP", created_by=None):
        """Create test feedback."""
        from agentic_rag.models.database import Feedback, FeedbackLabel
        import uuid

        feedback = Feedback(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            query=query,
            label=FeedbackLabel(label),
            notes="Test feedback notes",
            created_by=created_by
        )
        db_session.add(feedback)
        db_session.commit()
        return feedback


@pytest.fixture
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory


@pytest.fixture
def test_tenant(db_session, test_data_factory):
    """Create a test tenant."""
    return test_data_factory.create_test_tenant(db_session)


@pytest.fixture
def test_user(db_session, test_tenant, test_data_factory):
    """Create a test user."""
    return test_data_factory.create_test_user(db_session, tenant_id=test_tenant.id)


@pytest.fixture
def test_admin_user(db_session, test_tenant, test_data_factory):
    """Create a test admin user."""
    return test_data_factory.create_test_user(
        db_session,
        tenant_id=test_tenant.id,
        email="admin@example.com",
        role="admin"
    )


@pytest.fixture
def test_document(db_session, test_tenant, test_user, test_data_factory):
    """Create a test document."""
    return test_data_factory.create_test_document(
        db_session,
        tenant_id=test_tenant.id,
        created_by=test_user.id
    )


@pytest.fixture
def test_chunk(db_session, test_tenant, test_document, test_data_factory):
    """Create a test chunk."""
    return test_data_factory.create_test_chunk(
        db_session,
        tenant_id=test_tenant.id,
        document_id=test_document.id
    )


# JWT Authentication Fixtures
@pytest.fixture
def jwt_token_user(test_user):
    """Create JWT token for test user."""
    from agentic_rag.services.auth import AuthService
    from agentic_rag.config import get_settings

    settings = get_settings()
    auth_service = AuthService(settings)

    # Create token payload
    token_data = {
        "sub": str(test_user.id),
        "email": test_user.email,
        "tenant_id": str(test_user.tenant_id),
        "role": test_user.role.value,
        "type": "access"
    }

    return auth_service.create_access_token(token_data)


@pytest.fixture
def jwt_token_admin(test_admin_user):
    """Create JWT token for test admin user."""
    from agentic_rag.services.auth import AuthService
    from agentic_rag.config import get_settings

    settings = get_settings()
    auth_service = AuthService(settings)

    # Create token payload
    token_data = {
        "sub": str(test_admin_user.id),
        "email": test_admin_user.email,
        "tenant_id": str(test_admin_user.tenant_id),
        "role": test_admin_user.role.value,
        "type": "access"
    }

    return auth_service.create_access_token(token_data)


@pytest.fixture
def auth_headers_user(jwt_token_user):
    """Create authorization headers for test user."""
    return {"Authorization": f"Bearer {jwt_token_user}"}


@pytest.fixture
def auth_headers_admin(jwt_token_admin):
    """Create authorization headers for test admin."""
    return {"Authorization": f"Bearer {jwt_token_admin}"}


# Database Seeding Utilities
@pytest.fixture
def seed_test_data(db_session, test_data_factory):
    """Seed database with comprehensive test data."""
    # Create multiple tenants
    tenant1 = test_data_factory.create_test_tenant(db_session, "Tenant 1", "First test tenant")
    tenant2 = test_data_factory.create_test_tenant(db_session, "Tenant 2", "Second test tenant")

    # Create users for each tenant
    user1 = test_data_factory.create_test_user(db_session, tenant1.id, "user1@tenant1.com", "viewer")
    admin1 = test_data_factory.create_test_user(db_session, tenant1.id, "admin1@tenant1.com", "admin")
    user2 = test_data_factory.create_test_user(db_session, tenant2.id, "user2@tenant2.com", "analyst")

    # Create documents
    doc1 = test_data_factory.create_test_document(db_session, tenant1.id, "RFQ Document 1", "RFQ", user1.id)
    doc2 = test_data_factory.create_test_document(db_session, tenant1.id, "Offer Document 1", "OFFER_TECH", admin1.id)
    doc3 = test_data_factory.create_test_document(db_session, tenant2.id, "RFQ Document 2", "RFQ", user2.id)

    # Create chunks
    chunk1 = test_data_factory.create_test_chunk(db_session, tenant1.id, doc1.id, 1, 2)
    chunk2 = test_data_factory.create_test_chunk(db_session, tenant1.id, doc2.id, 1, 1)
    chunk3 = test_data_factory.create_test_chunk(db_session, tenant2.id, doc3.id, 3, 4)

    # Create feedback
    feedback1 = test_data_factory.create_test_feedback(db_session, tenant1.id, "Test query 1", "UP", user1.id)
    feedback2 = test_data_factory.create_test_feedback(db_session, tenant2.id, "Test query 2", "DOWN", user2.id)

    return {
        "tenants": [tenant1, tenant2],
        "users": [user1, admin1, user2],
        "documents": [doc1, doc2, doc3],
        "chunks": [chunk1, chunk2, chunk3],
        "feedback": [feedback1, feedback2]
    }
