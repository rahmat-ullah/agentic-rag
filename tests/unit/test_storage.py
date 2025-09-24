"""
Unit tests for storage service.

This module contains comprehensive tests for the storage service including
encryption, security features, and health monitoring.
"""

import io
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from cryptography.fernet import Fernet
from minio.error import S3Error

from agentic_rag.config import Settings
from agentic_rag.services.storage import StorageService


class TestStorageService:
    """Test cases for StorageService."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            storage={
                "minio_endpoint": "localhost:9000",
                "minio_access_key": "testkey",
                "minio_secret_key": "testsecret",
                "minio_secure": False,
                "minio_region": "us-east-1",
                "minio_bucket_documents": "test-documents",
                "minio_bucket_thumbnails": "test-thumbnails",
                "minio_bucket_exports": "test-exports",
                "storage_encryption_enabled": True,
                "storage_encryption_key": None
            }
        )
    
    @pytest.fixture
    def mock_minio_client(self):
        """Create mock MinIO client."""
        with patch('agentic_rag.services.storage.Minio') as mock_minio:
            client = MagicMock()
            mock_minio.return_value = client
            yield client
    
    @pytest.fixture
    def storage_service(self, settings, mock_minio_client):
        """Create storage service instance."""
        return StorageService(settings)
    
    def test_initialization(self, storage_service, settings):
        """Test storage service initialization."""
        assert storage_service.settings == settings
        assert storage_service.bucket_documents == "test-documents"
        assert storage_service.bucket_thumbnails == "test-thumbnails"
        assert storage_service.bucket_exports == "test-exports"
        assert storage_service._encryption_enabled is True
        assert storage_service._fernet is not None
    
    def test_generate_secure_object_name(self, storage_service):
        """Test secure object name generation."""
        tenant_id = uuid4()
        file_hash = "abcdef1234567890" * 4  # 64 char hash
        filename = "test.pdf"
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        
        object_name = storage_service.generate_secure_object_name(
            tenant_id=tenant_id,
            file_hash=file_hash,
            filename=filename,
            timestamp=timestamp
        )
        
        expected_parts = [
            str(tenant_id),
            "2024",
            "01",
            "15",
            file_hash[:8],
            f"{file_hash}_{filename}"
        ]
        expected = "/".join(expected_parts)
        
        assert object_name == expected
    
    @pytest.mark.asyncio
    async def test_store_file_with_encryption(self, storage_service, mock_minio_client):
        """Test file storage with encryption."""
        object_name = "test/file.txt"
        content = b"test content"
        
        # Mock successful upload
        mock_minio_client.put_object.return_value = MagicMock()
        
        result = await storage_service.store_file(object_name, content)
        
        # Verify MinIO client was called
        assert mock_minio_client.put_object.called
        call_args = mock_minio_client.put_object.call_args
        
        # Check that content was encrypted (should be different from original)
        uploaded_content = call_args[1]['data'].read()
        assert uploaded_content != content
        
        # Check metadata includes encryption info
        metadata = call_args[1]['metadata']
        assert metadata['encrypted'] == 'true'
        assert metadata['encryption_method'] == 'fernet'
        
        assert result == f"s3://test-documents/{object_name}"
    
    @pytest.mark.asyncio
    async def test_store_file_without_encryption(self, settings, mock_minio_client):
        """Test file storage without encryption."""
        settings.storage.storage_encryption_enabled = False
        storage_service = StorageService(settings)
        
        object_name = "test/file.txt"
        content = b"test content"
        
        # Mock successful upload
        mock_minio_client.put_object.return_value = MagicMock()
        
        result = await storage_service.store_file(object_name, content)
        
        # Verify MinIO client was called
        assert mock_minio_client.put_object.called
        call_args = mock_minio_client.put_object.call_args
        
        # Check that content was not encrypted
        uploaded_content = call_args[1]['data'].read()
        assert uploaded_content == content
        
        # Check metadata does not include encryption info
        metadata = call_args[1]['metadata']
        assert 'encrypted' not in metadata
        
        assert result == f"s3://test-documents/{object_name}"
    
    @pytest.mark.asyncio
    async def test_retrieve_file_with_decryption(self, storage_service, mock_minio_client):
        """Test file retrieval with decryption."""
        object_name = "test/file.txt"
        original_content = b"test content"
        
        # Encrypt content manually for test
        encrypted_content = storage_service._fernet.encrypt(original_content)
        
        # Mock response with encrypted content and metadata
        mock_response = MagicMock()
        mock_response.read.return_value = encrypted_content
        mock_response.headers = {
            'x-amz-meta-encrypted': 'true',
            'x-amz-meta-encryption_method': 'fernet'
        }
        mock_minio_client.get_object.return_value = mock_response
        
        content, metadata = await storage_service.retrieve_file(object_name)
        
        assert content == original_content
        assert metadata['encrypted'] == 'true'
        assert metadata['encryption_method'] == 'fernet'
    
    @pytest.mark.asyncio
    async def test_delete_file_success(self, storage_service, mock_minio_client):
        """Test successful file deletion."""
        object_name = "test/file.txt"
        
        # Mock file exists
        mock_minio_client.stat_object.return_value = MagicMock()
        
        result = await storage_service.delete_file(object_name)
        
        assert result is True
        assert mock_minio_client.remove_object.called
        assert storage_service._operation_stats["deletions"] == 1
    
    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, storage_service, mock_minio_client):
        """Test deletion of non-existent file."""
        object_name = "test/nonexistent.txt"
        
        # Mock file does not exist
        mock_minio_client.stat_object.side_effect = S3Error("NoSuchKey", "", "", "", "", "", "")
        
        result = await storage_service.delete_file(object_name)
        
        assert result is False
        assert not mock_minio_client.remove_object.called
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, storage_service, mock_minio_client):
        """Test health check when storage is healthy."""
        # Mock successful bucket listing with all required buckets
        mock_bucket1 = MagicMock()
        mock_bucket1.name = "test-documents"
        mock_bucket2 = MagicMock()
        mock_bucket2.name = "test-thumbnails"
        mock_bucket3 = MagicMock()
        mock_bucket3.name = "test-exports"

        mock_minio_client.list_buckets.return_value = [mock_bucket1, mock_bucket2, mock_bucket3]

        # Mock successful bucket operations
        mock_minio_client.list_objects.return_value = []
        mock_minio_client.put_object.return_value = MagicMock()
        mock_minio_client.remove_object.return_value = None

        health = await storage_service.health_check()

        assert health["status"] == "healthy"
        assert "connectivity" in health["checks"]
        assert health["checks"]["connectivity"]["status"] == "pass"
        assert "buckets" in health["checks"]
        assert health["checks"]["buckets"]["status"] == "pass"
        assert health["encryption_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, storage_service, mock_minio_client):
        """Test health check when storage is unhealthy."""
        # Mock connection failure
        mock_minio_client.list_buckets.side_effect = Exception("Connection failed")
        
        health = await storage_service.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
        assert health["checks"]["connectivity"]["status"] == "fail"
    
    @pytest.mark.asyncio
    async def test_get_storage_statistics(self, storage_service, mock_minio_client):
        """Test storage statistics retrieval."""
        # Mock objects in bucket
        mock_obj1 = MagicMock()
        mock_obj1.size = 1024
        mock_obj1.last_modified = datetime.utcnow()
        
        mock_obj2 = MagicMock()
        mock_obj2.size = 2048
        mock_obj2.last_modified = datetime.utcnow()
        
        mock_minio_client.list_objects.return_value = [mock_obj1, mock_obj2]
        
        stats = await storage_service.get_storage_statistics()
        
        assert "operations" in stats
        assert "buckets" in stats
        assert stats["total_objects"] == 6  # 2 objects * 3 buckets
        assert stats["total_size_bytes"] == 9216  # (1024 + 2048) * 3 buckets
    
    def test_operation_stats_tracking(self, storage_service):
        """Test operation statistics tracking."""
        initial_stats = storage_service.get_operation_stats()
        
        # Simulate operations
        storage_service._operation_stats["uploads"] += 1
        storage_service._operation_stats["downloads"] += 2
        storage_service._operation_stats["errors"] += 1
        
        updated_stats = storage_service.get_operation_stats()
        
        assert updated_stats["uploads"] == initial_stats["uploads"] + 1
        assert updated_stats["downloads"] == initial_stats["downloads"] + 2
        assert updated_stats["errors"] == initial_stats["errors"] + 1
        
        # Test reset
        storage_service.reset_operation_stats()
        reset_stats = storage_service.get_operation_stats()
        
        assert reset_stats["uploads"] == 0
        assert reset_stats["downloads"] == 0
        assert reset_stats["errors"] == 0
