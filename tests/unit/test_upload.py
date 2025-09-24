"""
Unit tests for upload functionality.

This module contains unit tests for file upload services, validation,
and related functionality.
"""

import hashlib
import io
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import UploadFile

from agentic_rag.api.models.upload import (
    FileValidationError,
    UploadRequest,
    UploadStatus,
)
from agentic_rag.config import Settings
from agentic_rag.services.upload import FileValidator, UploadService


class TestFileValidator:
    """Test file validation functionality."""
    
    def test_file_validator_initialization(self, test_settings):
        """Test FileValidator initialization."""
        validator = FileValidator(test_settings)
        
        assert validator.settings == test_settings
        assert validator.max_file_size == test_settings.upload.max_file_size
        assert len(validator.allowed_mime_types) > 0
        assert len(validator.allowed_extensions) > 0
    
    def test_validate_file_success(self, test_settings):
        """Test successful file validation."""
        validator = FileValidator(test_settings)
        
        # Create mock upload file
        content = b"This is a test PDF content"
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.pdf"
        upload_file.content_type = "application/pdf"
        
        errors = validator.validate_file(upload_file, content)
        
        assert len(errors) == 0
    
    def test_validate_file_too_large(self, test_settings):
        """Test file size validation failure."""
        validator = FileValidator(test_settings)
        
        # Create content larger than max size
        content = b"x" * (test_settings.upload.max_file_size + 1)
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "large.pdf"
        upload_file.content_type = "application/pdf"
        
        errors = validator.validate_file(upload_file, content)
        
        assert len(errors) == 1
        assert errors[0].code == "FILE_TOO_LARGE"
        assert "exceeds maximum allowed size" in errors[0].message
    
    def test_validate_file_invalid_extension(self, test_settings):
        """Test file extension validation failure."""
        validator = FileValidator(test_settings)

        content = b"This is test content"
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.exe"  # Not allowed extension
        upload_file.content_type = "application/pdf"

        errors = validator.validate_file(upload_file, content)

        # Should have at least one error for invalid extension
        assert len(errors) >= 1
        error_codes = [error.code for error in errors]
        assert "INVALID_EXTENSION" in error_codes
    
    def test_validate_file_invalid_mime_type(self, test_settings):
        """Test MIME type validation failure."""
        validator = FileValidator(test_settings)

        content = b"This is test content"
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.pdf"
        upload_file.content_type = "application/x-executable"  # Not allowed MIME type

        errors = validator.validate_file(upload_file, content)

        # Should have at least one error for invalid MIME type
        assert len(errors) >= 1
        error_codes = [error.code for error in errors]
        assert "INVALID_MIME_TYPE" in error_codes
    
    def test_validate_file_multiple_errors(self, test_settings):
        """Test multiple validation errors."""
        validator = FileValidator(test_settings)

        # Create content with multiple issues
        content = b"x" * (test_settings.upload.max_file_size + 1)
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.exe"  # Invalid extension
        upload_file.content_type = "application/x-executable"  # Invalid MIME type

        errors = validator.validate_file(upload_file, content)

        # Should have at least 3 errors (size, extension, MIME type)
        assert len(errors) >= 3
        error_codes = [error.code for error in errors]
        assert "FILE_TOO_LARGE" in error_codes
        assert "INVALID_EXTENSION" in error_codes
        assert "INVALID_MIME_TYPE" in error_codes


class TestUploadService:
    """Test upload service functionality."""
    
    @pytest.fixture
    def mock_storage_service(self):
        """Create mock storage service."""
        storage_service = AsyncMock()
        storage_service.store_file = AsyncMock(return_value="s3://bucket/path/file.pdf")
        return storage_service
    
    @pytest.fixture
    def upload_service(self, test_settings, mock_storage_service):
        """Create upload service with mocked dependencies."""
        return UploadService(test_settings, mock_storage_service)
    
    @pytest.mark.asyncio
    async def test_create_upload_session(self, upload_service):
        """Test upload session creation."""
        tenant_id = uuid4()
        user_id = uuid4()
        filename = "test.pdf"
        content_type = "application/pdf"
        file_size = 1024
        
        upload_request = UploadRequest(
            title="Test Document",
            chunk_upload=False
        )
        
        session = await upload_service.create_upload_session(
            tenant_id=tenant_id,
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            upload_request=upload_request
        )
        
        assert session.tenant_id == tenant_id
        assert session.user_id == user_id
        assert session.filename == filename
        assert session.content_type == content_type
        assert session.total_size == file_size
        assert session.status == UploadStatus.PENDING
        assert session.upload_options == upload_request
    
    @pytest.mark.asyncio
    async def test_create_chunked_upload_session(self, upload_service):
        """Test chunked upload session creation."""
        tenant_id = uuid4()
        user_id = uuid4()
        filename = "large.pdf"
        content_type = "application/pdf"
        file_size = 50 * 1024 * 1024  # 50MB
        
        upload_request = UploadRequest(
            title="Large Document",
            chunk_upload=True
        )
        
        session = await upload_service.create_upload_session(
            tenant_id=tenant_id,
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            upload_request=upload_request
        )
        
        assert session.chunk_size is not None
        assert session.total_chunks is not None
        assert session.total_chunks > 1  # Should be chunked
        assert session.upload_options.chunk_upload is True
    
    @pytest.mark.asyncio
    async def test_check_duplicate_no_duplicate(self, upload_service):
        """Test duplicate detection when no duplicate exists."""
        tenant_id = uuid4()
        sha256_hash = hashlib.sha256(b"unique content").hexdigest()

        # Mock database session
        mock_db_session = MagicMock()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_filter.first.return_value = None
        mock_query.filter.return_value = mock_filter
        mock_db_session.query.return_value = mock_query

        duplicate_info = await upload_service._check_duplicate(tenant_id, sha256_hash, mock_db_session)

        assert duplicate_info.is_duplicate is False
        assert duplicate_info.existing_document_id is None
        assert duplicate_info.action_taken == "uploaded"
    
    @pytest.mark.asyncio
    async def test_check_duplicate_found(self, upload_service):
        """Test duplicate detection when duplicate exists."""
        tenant_id = uuid4()
        sha256_hash = hashlib.sha256(b"duplicate content").hexdigest()

        # Mock existing document
        existing_doc = MagicMock()
        existing_doc.id = uuid4()
        existing_doc.title = "existing.pdf"
        existing_doc.created_at = "2024-01-01T00:00:00Z"

        # Mock database session
        mock_db_session = MagicMock()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_filter.first.return_value = existing_doc
        mock_query.filter.return_value = mock_filter
        mock_db_session.query.return_value = mock_query

        duplicate_info = await upload_service._check_duplicate(tenant_id, sha256_hash, mock_db_session)

        assert duplicate_info.is_duplicate is True
        assert duplicate_info.existing_document_id == existing_doc.id
        assert duplicate_info.existing_filename == existing_doc.title
        assert duplicate_info.action_taken == "skipped"
    
    @pytest.mark.asyncio
    async def test_virus_scan_success(self, upload_service):
        """Test successful virus scan."""
        content = b"Clean file content"
        
        # Should not raise any exception
        await upload_service._virus_scan(content)
    
    @pytest.mark.asyncio
    async def test_virus_scan_empty_file(self, upload_service):
        """Test virus scan with empty file."""
        content = b""

        with pytest.raises(Exception) as exc_info:
            await upload_service._virus_scan(content)

        # Check that the exception contains the expected message
        assert "Empty file detected" in str(exc_info.value) or "Empty file detected" in getattr(exc_info.value, 'detail', '')
    
    @pytest.mark.asyncio
    async def test_store_file(self, upload_service, mock_storage_service):
        """Test file storage."""
        tenant_id = uuid4()
        sha256_hash = "abcd1234"
        content = b"File content"
        filename = "test.pdf"
        
        storage_path = await upload_service._store_file(tenant_id, sha256_hash, content, filename)
        
        # Verify storage service was called
        mock_storage_service.store_file.assert_called_once()
        
        # Verify path format
        assert str(tenant_id) in storage_path
        assert sha256_hash in storage_path
        assert filename in storage_path
    
    @pytest.mark.asyncio
    async def test_create_document_record(self, upload_service):
        """Test document record creation."""
        tenant_id = uuid4()
        user_id = uuid4()

        from agentic_rag.api.models.upload import FileMetadata
        file_metadata = FileMetadata(
            filename="test.pdf",
            content_type="application/pdf",
            size=1024,
            sha256="abcd1234",
            extension=".pdf"
        )

        storage_path = "tenant/2024/01/01/hash/test.pdf"
        upload_request = UploadRequest(title="Test Document", kind="RFQ")

        # Mock database session
        mock_db_session = MagicMock()

        document_id = await upload_service._create_document_record(
            tenant_id=tenant_id,
            user_id=user_id,
            file_metadata=file_metadata,
            storage_path=storage_path,
            upload_request=upload_request,
            db_session=mock_db_session
        )

        # Verify document was added to session
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # Verify document ID is UUID
        assert isinstance(document_id, type(uuid4()))
