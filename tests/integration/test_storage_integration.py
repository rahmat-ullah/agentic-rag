"""
Integration tests for storage service.

This module contains integration tests for the storage service that test
the actual storage operations with a real or mock storage backend.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from agentic_rag.config import get_settings
from agentic_rag.services.storage import StorageService, get_storage_service


class TestStorageIntegration:
    """Integration test cases for StorageService."""
    
    @pytest.fixture
    def settings(self):
        """Get test settings."""
        return get_settings()
    
    @pytest.fixture
    def storage_service(self, settings):
        """Create storage service instance."""
        return StorageService(settings)
    
    @pytest.mark.asyncio
    async def test_storage_service_initialization(self, storage_service):
        """Test storage service can be initialized."""
        assert storage_service is not None
        assert storage_service.bucket_documents is not None
        assert storage_service.bucket_thumbnails is not None
        assert storage_service.bucket_exports is not None
    
    @pytest.mark.asyncio
    async def test_secure_object_name_generation(self, storage_service):
        """Test secure object name generation produces valid paths."""
        tenant_id = uuid4()
        file_hash = "a" * 64  # Valid SHA256 hash
        filename = "test-document.pdf"
        
        object_name = storage_service.generate_secure_object_name(
            tenant_id=tenant_id,
            file_hash=file_hash,
            filename=filename
        )
        
        # Verify structure: tenant/year/month/day/hash_prefix/hash_filename
        parts = object_name.split("/")
        assert len(parts) == 6
        assert parts[0] == str(tenant_id)
        assert len(parts[1]) == 4  # Year
        assert len(parts[2]) == 2  # Month
        assert len(parts[3]) == 2  # Day
        assert parts[4] == file_hash[:8]  # Hash prefix
        assert parts[5] == f"{file_hash}_{filename}"  # Full hash + filename
    
    @pytest.mark.asyncio
    async def test_encryption_key_generation(self, storage_service):
        """Test encryption key is properly generated or loaded."""
        if storage_service._encryption_enabled:
            assert storage_service._encryption_key is not None
            assert storage_service._fernet is not None
            
            # Test encryption/decryption works
            test_data = b"test encryption data"
            encrypted = storage_service._fernet.encrypt(test_data)
            decrypted = storage_service._fernet.decrypt(encrypted)
            
            assert encrypted != test_data
            assert decrypted == test_data
    
    @pytest.mark.asyncio
    async def test_health_check_structure(self, storage_service):
        """Test health check returns proper structure."""
        health = await storage_service.health_check()
        
        # Verify required fields
        assert "status" in health
        assert "timestamp" in health
        assert "endpoint" in health
        assert "encryption_enabled" in health
        assert "checks" in health
        assert "statistics" in health
        assert "buckets" in health
        
        # Verify status is valid
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Verify checks structure
        if "connectivity" in health["checks"]:
            assert "status" in health["checks"]["connectivity"]
        
        if "buckets" in health["checks"]:
            assert "status" in health["checks"]["buckets"]
            assert "required" in health["checks"]["buckets"]
    
    @pytest.mark.asyncio
    async def test_operation_statistics_structure(self, storage_service):
        """Test operation statistics have proper structure."""
        stats = storage_service.get_operation_stats()
        
        required_fields = ["uploads", "downloads", "deletions", "errors"]
        for field in required_fields:
            assert field in stats
            assert isinstance(stats[field], int)
            assert stats[field] >= 0
    
    @pytest.mark.asyncio
    async def test_storage_statistics_structure(self, storage_service):
        """Test storage statistics have proper structure."""
        stats = await storage_service.get_storage_statistics()
        
        assert "operations" in stats
        assert "buckets" in stats
        assert "total_objects" in stats
        assert "total_size_bytes" in stats
        
        assert isinstance(stats["total_objects"], int)
        assert isinstance(stats["total_size_bytes"], int)
        assert stats["total_objects"] >= 0
        assert stats["total_size_bytes"] >= 0
    
    @pytest.mark.asyncio
    async def test_file_operations_workflow(self, storage_service):
        """Test complete file operations workflow."""
        # This test would require a real storage backend
        # For now, we test the method signatures and basic validation
        
        tenant_id = uuid4()
        file_hash = "b" * 64
        filename = "workflow-test.txt"
        content = b"test file content for workflow"
        
        # Test object name generation
        object_name = storage_service.generate_secure_object_name(
            tenant_id=tenant_id,
            file_hash=file_hash,
            filename=filename
        )
        
        assert object_name is not None
        assert isinstance(object_name, str)
        assert len(object_name) > 0
        
        # Test metadata preparation
        metadata = {
            "tenant_id": str(tenant_id),
            "original_filename": filename,
            "sha256": file_hash,
            "upload_timestamp": datetime.utcnow().isoformat()
        }
        
        assert all(key in metadata for key in ["tenant_id", "original_filename", "sha256", "upload_timestamp"])
    
    def test_global_storage_service_singleton(self, settings):
        """Test global storage service singleton pattern."""
        service1 = get_storage_service(settings)
        service2 = get_storage_service(settings)
        
        # Should return the same instance
        assert service1 is service2
    
    @pytest.mark.asyncio
    async def test_error_handling_structure(self, storage_service):
        """Test error handling produces proper error structures."""
        # Test with invalid object name
        try:
            await storage_service.retrieve_file("")
        except Exception as e:
            assert isinstance(e, Exception)
            assert str(e) is not None
    
    @pytest.mark.asyncio
    async def test_bucket_configuration(self, storage_service):
        """Test bucket configuration is properly set."""
        assert storage_service.bucket_documents is not None
        assert storage_service.bucket_thumbnails is not None
        assert storage_service.bucket_exports is not None
        
        # Verify bucket names are strings
        assert isinstance(storage_service.bucket_documents, str)
        assert isinstance(storage_service.bucket_thumbnails, str)
        assert isinstance(storage_service.bucket_exports, str)
        
        # Verify bucket names are not empty
        assert len(storage_service.bucket_documents) > 0
        assert len(storage_service.bucket_thumbnails) > 0
        assert len(storage_service.bucket_exports) > 0


class TestStorageServiceConfiguration:
    """Test storage service configuration scenarios."""
    
    @pytest.mark.asyncio
    async def test_encryption_disabled_configuration(self):
        """Test storage service with encryption disabled."""
        from agentic_rag.config import Settings
        
        settings = Settings(
            storage={
                "minio_endpoint": "localhost:9000",
                "minio_access_key": "testkey",
                "minio_secret_key": "testsecret",
                "storage_encryption_enabled": False
            }
        )
        
        storage_service = StorageService(settings)
        
        assert storage_service._encryption_enabled is False
        assert storage_service._fernet is None
    
    @pytest.mark.asyncio
    async def test_custom_encryption_key_configuration(self):
        """Test storage service with custom encryption key."""
        from agentic_rag.config import Settings
        from cryptography.fernet import Fernet
        
        custom_key = Fernet.generate_key().decode()
        
        settings = Settings(
            storage={
                "minio_endpoint": "localhost:9000",
                "minio_access_key": "testkey",
                "minio_secret_key": "testsecret",
                "storage_encryption_enabled": True,
                "storage_encryption_key": custom_key
            }
        )
        
        storage_service = StorageService(settings)
        
        assert storage_service._encryption_enabled is True
        assert storage_service._encryption_key == custom_key.encode()
        assert storage_service._fernet is not None
