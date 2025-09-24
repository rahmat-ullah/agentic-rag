"""
Base test classes and utilities for Agentic RAG System tests.

This module provides base test classes that can be inherited by specific test modules
to provide common functionality and setup.
"""

import pytest
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from agentic_rag.config import Settings


class BaseTest(ABC):
    """Base test class with common functionality."""
    
    def setup_method(self):
        """Set up method called before each test method."""
        self.setup_test_data()
    
    def teardown_method(self):
        """Tear down method called after each test method."""
        self.cleanup_test_data()
    
    def setup_test_data(self):
        """Set up test data. Override in subclasses."""
        pass
    
    def cleanup_test_data(self):
        """Clean up test data. Override in subclasses."""
        pass


class BaseUnitTest(BaseTest):
    """Base class for unit tests."""
    
    def setup_method(self):
        """Set up unit test environment."""
        super().setup_method()
        self.setup_mocks()
    
    def setup_mocks(self):
        """Set up mocks for unit tests. Override in subclasses."""
        pass


class BaseDatabaseTest(BaseTest):
    """Base class for database tests."""
    
    @pytest.fixture(autouse=True)
    def setup_database(self, db_session: Session):
        """Set up database session for tests."""
        self.db_session = db_session
    
    def create_test_tenant(self, **kwargs) -> Dict[str, Any]:
        """Create a test tenant."""
        from agentic_rag.models.database import Tenant
        import uuid
        
        tenant_data = {
            "id": uuid.uuid4(),
            "name": "Test Tenant",
            **kwargs
        }
        
        tenant = Tenant(**tenant_data)
        self.db_session.add(tenant)
        self.db_session.commit()
        self.db_session.refresh(tenant)
        
        return {
            "id": str(tenant.id),
            "name": tenant.name,
            "created_at": tenant.created_at.isoformat(),
            "updated_at": tenant.updated_at.isoformat(),
        }
    
    def create_test_user(self, tenant_id: str = None, **kwargs) -> Dict[str, Any]:
        """Create a test user."""
        from agentic_rag.models.database import User, UserRole
        import uuid

        if not tenant_id:
            tenant = self.create_test_tenant()
            tenant_id = tenant["id"]

        user_data = {
            "id": uuid.uuid4(),
            "tenant_id": uuid.UUID(tenant_id),
            "email": "test@example.com",
            "password_hash": "$2b$12$test.hash.value",
            "role": UserRole.ANALYST,
            "is_active": True,
            **kwargs
        }

        user = User(**user_data)
        self.db_session.add(user)
        self.db_session.commit()
        self.db_session.refresh(user)

        return {
            "id": str(user.id),
            "tenant_id": str(user.tenant_id),
            "email": user.email,
            "role": user.role.value,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat(),
        }


class BaseAPITest(BaseTest):
    """Base class for API tests."""
    
    @pytest.fixture(autouse=True)
    def setup_api(self, test_client: TestClient, test_settings: Settings):
        """Set up API test environment."""
        self.client = test_client
        self.settings = test_settings
        self.base_url = "http://testserver"
    
    def get(self, url: str, headers: Optional[Dict[str, str]] = None, **kwargs):
        """Make GET request."""
        return self.client.get(url, headers=headers, **kwargs)
    
    def post(self, url: str, json: Optional[Dict[str, Any]] = None, 
             headers: Optional[Dict[str, str]] = None, **kwargs):
        """Make POST request."""
        return self.client.post(url, json=json, headers=headers, **kwargs)
    
    def put(self, url: str, json: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None, **kwargs):
        """Make PUT request."""
        return self.client.put(url, json=json, headers=headers, **kwargs)
    
    def delete(self, url: str, headers: Optional[Dict[str, str]] = None, **kwargs):
        """Make DELETE request."""
        return self.client.delete(url, headers=headers, **kwargs)
    
    def assert_success_response(self, response, expected_status: int = 200):
        """Assert successful API response."""
        assert response.status_code == expected_status
        data = response.json()
        assert data.get("success") is True
        return data
    
    def assert_error_response(self, response, expected_status: int = 400):
        """Assert error API response."""
        assert response.status_code == expected_status
        data = response.json()
        assert data.get("success") is False
        return data
    
    def create_auth_token(self, user_data: Optional[Dict[str, Any]] = None) -> str:
        """Create authentication token for testing."""
        from agentic_rag.services.auth import AuthService
        
        if not user_data:
            user_data = {
                "id": "test-user-id",
                "email": "test@example.com",
                "tenant_id": "test-tenant-id",
                "role": "analyst"
            }
        
        auth_service = AuthService()
        return auth_service.create_access_token(user_data)
    
    def get_auth_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        """Get authorization headers."""
        if not token:
            token = self.create_auth_token()
        return {"Authorization": f"Bearer {token}"}


class BaseIntegrationTest(BaseDatabaseTest, BaseAPITest):
    """Base class for integration tests that need both database and API."""
    
    @pytest.fixture(autouse=True)
    def setup_integration(self, db_session: Session, test_client: TestClient, 
                         test_settings: Settings):
        """Set up integration test environment."""
        self.db_session = db_session
        self.client = test_client
        self.settings = test_settings
        self.base_url = "http://testserver"


class MockFactory:
    """Factory for creating mock objects."""
    
    @staticmethod
    def create_mock_database_adapter():
        """Create mock database adapter."""
        mock_adapter = Mock()
        mock_adapter.health_check.return_value = True
        mock_adapter.get_session.return_value = Mock()
        return mock_adapter
    
    @staticmethod
    def create_mock_auth_service():
        """Create mock authentication service."""
        mock_service = Mock()
        mock_service.authenticate_user.return_value = {
            "id": "test-user-id",
            "email": "test@example.com",
            "tenant_id": "test-tenant-id",
            "role": "analyst"
        }
        mock_service.create_access_token.return_value = "test-token"
        mock_service.verify_token.return_value = {
            "user_id": "test-user-id",
            "tenant_id": "test-tenant-id",
            "role": "analyst"
        }
        return mock_service
    
    @staticmethod
    def create_mock_vector_service():
        """Create mock vector service."""
        mock_service = Mock()
        mock_service.search.return_value = [
            {"id": "doc1", "score": 0.9, "content": "Test content 1"},
            {"id": "doc2", "score": 0.8, "content": "Test content 2"},
        ]
        mock_service.add_documents.return_value = True
        return mock_service


# Test data generators
class TestDataGenerator:
    """Generate test data for various entities."""
    
    @staticmethod
    def generate_tenant_data(**overrides) -> Dict[str, Any]:
        """Generate tenant test data."""
        import uuid
        data = {
            "id": str(uuid.uuid4()),
            "name": "Test Tenant",
        }
        data.update(overrides)
        return data
    
    @staticmethod
    def generate_user_data(**overrides) -> Dict[str, Any]:
        """Generate user test data."""
        import uuid
        data = {
            "id": str(uuid.uuid4()),
            "tenant_id": str(uuid.uuid4()),
            "email": "test@example.com",
            "password": "password123",
            "role": "analyst",
            "is_active": True,
        }
        data.update(overrides)
        return data
    
    @staticmethod
    def generate_document_data(**overrides) -> Dict[str, Any]:
        """Generate document test data."""
        import uuid
        data = {
            "id": str(uuid.uuid4()),
            "tenant_id": str(uuid.uuid4()),
            "title": "Test Document",
            "doc_kind": "RFQ",
            "file_path": "/test/path/document.pdf",
            "file_size": 1024,
            "mime_type": "application/pdf",
        }
        data.update(overrides)
        return data
