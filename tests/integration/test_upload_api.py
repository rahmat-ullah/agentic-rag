"""
Integration tests for upload API endpoints.

This module contains integration tests for file upload functionality,
testing the complete upload flow from API endpoint to storage.
"""

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agentic_rag.api.app import create_app
from agentic_rag.config import get_settings


class TestUploadAPI:
    """Test upload API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create test app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_storage_service(self):
        """Mock storage service."""
        with patch('agentic_rag.services.storage.get_storage_service') as mock:
            storage_service = AsyncMock()
            storage_service.store_file = AsyncMock(return_value="s3://bucket/path/file.pdf")
            mock.return_value = storage_service
            yield storage_service
    
    @pytest.fixture
    def mock_auth(self):
        """Mock authentication."""
        with patch('agentic_rag.api.dependencies.auth.get_current_user') as mock:
            from agentic_rag.models.database import User, UserRole
            from uuid import uuid4
            
            user = User(
                id=uuid4(),
                tenant_id=uuid4(),
                email="test@example.com",
                password_hash="hashed",
                role=UserRole.ADMIN,
                is_active=True
            )
            mock.return_value = user
            yield user
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch('agentic_rag.api.dependencies.database.get_db_session') as mock:
            session = AsyncMock()
            session.query.return_value.filter.return_value.first.return_value = None
            session.add = AsyncMock()
            session.commit = AsyncMock()
            mock.return_value = session
            yield session
    
    def test_upload_endpoint_exists(self, client):
        """Test that upload endpoint exists."""
        # This will fail with authentication error, but endpoint should exist
        response = client.post("/api/v1/upload/")
        # Should not be 404 (not found)
        assert response.status_code != 404
    
    def test_upload_quota_endpoint_exists(self, client):
        """Test that upload quota endpoint exists."""
        response = client.get("/api/v1/upload/quota")
        # Should not be 404 (not found)
        assert response.status_code != 404
    
    def test_upload_stats_endpoint_exists(self, client):
        """Test that upload stats endpoint exists."""
        response = client.get("/api/v1/upload/stats")
        # Should not be 404 (not found)
        assert response.status_code != 404
    
    @pytest.mark.skip(reason="Requires full authentication setup")
    def test_upload_file_success(self, client, mock_auth, mock_db_session, mock_storage_service):
        """Test successful file upload."""
        # Create test file
        file_content = b"This is a test PDF file content"
        file_data = {
            "file": ("test.pdf", io.BytesIO(file_content), "application/pdf")
        }
        
        form_data = {
            "title": "Test Document",
            "description": "Test upload",
            "kind": "RFQ",
            "extract_text": True,
            "generate_thumbnail": True
        }
        
        response = client.post(
            "/api/v1/upload/",
            files=file_data,
            data=form_data
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "upload_id" in data["data"]
        assert data["data"]["status"] == "complete"
    
    @pytest.mark.skip(reason="Requires full authentication setup")
    def test_upload_file_validation_error(self, client, mock_auth, mock_db_session):
        """Test file upload with validation error."""
        # Create test file with invalid extension
        file_content = b"This is not a valid file"
        file_data = {
            "file": ("test.exe", io.BytesIO(file_content), "application/x-executable")
        }
        
        response = client.post(
            "/api/v1/upload/",
            files=file_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "validation failed" in data["detail"]["message"].lower()
    
    @pytest.mark.skip(reason="Requires full authentication setup")
    def test_create_upload_session(self, client, mock_auth):
        """Test creating upload session."""
        form_data = {
            "filename": "large_file.pdf",
            "content_type": "application/pdf",
            "file_size": 50 * 1024 * 1024,  # 50MB
            "title": "Large Document"
        }
        
        response = client.post(
            "/api/v1/upload/session",
            data=form_data
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "upload_id" in data["data"]["id"]
        assert data["data"]["status"] == "pending"
    
    @pytest.mark.skip(reason="Requires full authentication setup")
    def test_get_upload_quota(self, client, mock_auth, mock_db_session):
        """Test getting upload quota."""
        response = client.get("/api/v1/upload/quota")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_quota" in data["data"]
        assert "used_quota" in data["data"]
        assert "available_quota" in data["data"]
    
    @pytest.mark.skip(reason="Requires full authentication setup")
    def test_get_upload_stats(self, client, mock_auth, mock_db_session):
        """Test getting upload statistics."""
        response = client.get("/api/v1/upload/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_uploads" in data["data"]
        assert "successful_uploads" in data["data"]
        assert "quota_info" in data["data"]


class TestUploadValidation:
    """Test upload validation functionality."""
    
    def test_file_size_validation(self):
        """Test file size validation."""
        from agentic_rag.services.upload import FileValidator
        from agentic_rag.config import get_settings
        from unittest.mock import MagicMock
        from fastapi import UploadFile
        
        settings = get_settings()
        validator = FileValidator(settings)
        
        # Create oversized file
        content = b"x" * (settings.upload.max_file_size + 1)
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "large.pdf"
        upload_file.content_type = "application/pdf"
        
        errors = validator.validate_file(upload_file, content)
        
        assert len(errors) > 0
        assert any(error.code == "FILE_TOO_LARGE" for error in errors)
    
    def test_mime_type_validation(self):
        """Test MIME type validation."""
        from agentic_rag.services.upload import FileValidator
        from agentic_rag.config import get_settings
        from unittest.mock import MagicMock
        from fastapi import UploadFile
        
        settings = get_settings()
        validator = FileValidator(settings)
        
        # Create file with invalid MIME type
        content = b"This is test content"
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.pdf"
        upload_file.content_type = "application/x-malware"
        
        errors = validator.validate_file(upload_file, content)
        
        assert len(errors) > 0
        assert any(error.code == "INVALID_MIME_TYPE" for error in errors)
    
    def test_extension_validation(self):
        """Test file extension validation."""
        from agentic_rag.services.upload import FileValidator
        from agentic_rag.config import get_settings
        from unittest.mock import MagicMock
        from fastapi import UploadFile
        
        settings = get_settings()
        validator = FileValidator(settings)
        
        # Create file with invalid extension
        content = b"This is test content"
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "malware.exe"
        upload_file.content_type = "application/pdf"
        
        errors = validator.validate_file(upload_file, content)
        
        assert len(errors) > 0
        assert any(error.code == "INVALID_EXTENSION" for error in errors)


class TestUploadConfiguration:
    """Test upload configuration."""
    
    def test_upload_settings_loaded(self):
        """Test that upload settings are properly loaded."""
        settings = get_settings()
        
        assert hasattr(settings, 'upload')
        assert settings.upload.max_file_size > 0
        assert len(settings.upload.allowed_mime_types) > 0
        assert len(settings.upload.allowed_extensions) > 0
        assert settings.upload.tenant_upload_quota > 0
    
    def test_storage_settings_loaded(self):
        """Test that storage settings are properly loaded."""
        settings = get_settings()
        
        assert hasattr(settings, 'storage')
        assert settings.storage.minio_endpoint
        assert settings.storage.minio_access_key
        assert settings.storage.minio_secret_key
        assert settings.storage.minio_bucket_documents
    
    def test_allowed_file_types(self):
        """Test allowed file types configuration."""
        settings = get_settings()
        
        # Check that common document types are allowed
        assert "application/pdf" in settings.upload.allowed_mime_types
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in settings.upload.allowed_mime_types
        assert ".pdf" in settings.upload.allowed_extensions
        assert ".docx" in settings.upload.allowed_extensions
