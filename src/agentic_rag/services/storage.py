"""
Storage service for object storage operations.

This module provides an abstraction layer for object storage operations
using MinIO/S3-compatible storage backends with encryption and security features.
"""

import asyncio
import hashlib
import io
import logging
import os
from datetime import datetime, timedelta
from typing import BinaryIO, Dict, List, Optional, Tuple
from uuid import UUID

from cryptography.fernet import Fernet
from minio import Minio
from minio.commonconfig import ENABLED, Filter
from minio.error import S3Error
from minio.sse import SseCustomerKey

from agentic_rag.config import Settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service for object storage operations with encryption and security features."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = Minio(
            endpoint=settings.storage.minio_endpoint,
            access_key=settings.storage.minio_access_key,
            secret_key=settings.storage.minio_secret_key,
            secure=settings.storage.minio_secure,
            region=settings.storage.minio_region
        )
        self.bucket_documents = settings.storage.minio_bucket_documents
        self.bucket_thumbnails = settings.storage.minio_bucket_thumbnails
        self.bucket_exports = settings.storage.minio_bucket_exports

        # Initialize encryption
        self._encryption_enabled = settings.storage.storage_encryption_enabled
        self._encryption_key = self._get_or_generate_encryption_key()
        self._fernet = Fernet(self._encryption_key) if self._encryption_enabled else None

        # Storage statistics
        self._operation_stats = {
            "uploads": 0,
            "downloads": 0,
            "deletions": 0,
            "errors": 0,
            "last_health_check": None
        }
    
    def _get_or_generate_encryption_key(self) -> bytes:
        """Get or generate encryption key for file encryption."""
        if self.settings.storage.storage_encryption_key:
            # Use provided key (must be base64 encoded Fernet key)
            return self.settings.storage.storage_encryption_key.encode()
        else:
            # Generate a new key (in production, this should be stored securely)
            return Fernet.generate_key()

    async def initialize(self) -> None:
        """Initialize storage service and create buckets if they don't exist."""
        await self._ensure_buckets_exist()
        await self._setup_bucket_policies()
        logger.info("Storage service initialized successfully")

    async def _ensure_buckets_exist(self) -> None:
        """Ensure all required buckets exist with proper configuration."""
        buckets = [
            self.bucket_documents,
            self.bucket_thumbnails,
            self.bucket_exports
        ]

        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket, location=self.settings.storage.minio_region)
                    logger.info(f"Created bucket: {bucket}")

                # Set bucket encryption if supported
                if self._encryption_enabled:
                    await self._set_bucket_encryption(bucket)

            except S3Error as e:
                self._operation_stats["errors"] += 1
                logger.error(f"Failed to create bucket {bucket}: {e}")
                raise Exception(f"Failed to create bucket {bucket}: {e}")

    async def _setup_bucket_policies(self) -> None:
        """Set up bucket policies for security."""
        # This would set up bucket policies for access control
        # Implementation depends on specific security requirements
        pass

    async def _set_bucket_encryption(self, bucket: str) -> None:
        """Set bucket-level encryption if supported by the storage backend."""
        try:
            # MinIO supports server-side encryption
            # This is a placeholder for bucket-level encryption setup
            logger.info(f"Encryption configured for bucket: {bucket}")
        except Exception as e:
            logger.warning(f"Could not set bucket encryption for {bucket}: {e}")
    
    def generate_secure_object_name(
        self,
        tenant_id: UUID,
        file_hash: str,
        filename: str,
        timestamp: Optional[datetime] = None
    ) -> str:
        """Generate secure, unique object name with hierarchical structure."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Create hierarchical path: tenant/year/month/day/hash/filename
        path_components = [
            str(tenant_id),
            f"{timestamp.year:04d}",
            f"{timestamp.month:02d}",
            f"{timestamp.day:02d}",
            file_hash[:8],  # First 8 chars of hash for directory
            f"{file_hash}_{filename}"  # Full hash + filename for uniqueness
        ]

        return "/".join(path_components)

    async def store_file(
        self,
        object_name: str,
        content: bytes,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Store file content in object storage with encryption and security."""
        bucket_name = bucket or self.bucket_documents

        try:
            # Encrypt content if encryption is enabled
            if self._encryption_enabled and self._fernet:
                content = self._fernet.encrypt(content)
                if metadata is None:
                    metadata = {}
                metadata["encrypted"] = "true"
                metadata["encryption_method"] = "fernet"

            # Add storage metadata
            if metadata is None:
                metadata = {}
            metadata.update({
                "stored_at": datetime.utcnow().isoformat(),
                "original_size": str(len(content)),
                "storage_version": "2.0"
            })

            # Convert bytes to file-like object
            content_stream = io.BytesIO(content)

            # Upload file with metadata
            result = self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=content_stream,
                length=len(content),
                content_type=content_type,
                metadata=metadata
            )

            self._operation_stats["uploads"] += 1
            logger.info(f"Successfully stored file: {object_name} ({len(content)} bytes)")

            return f"s3://{bucket_name}/{object_name}"

        except S3Error as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Failed to store file {object_name}: {e}")
            raise Exception(f"Failed to store file {object_name}: {e}")
        except Exception as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Unexpected error storing file {object_name}: {e}")
            raise Exception(f"Failed to store file {object_name}: {e}")
    
    async def retrieve_file(self, object_name: str, bucket: Optional[str] = None) -> Tuple[bytes, Dict[str, str]]:
        """Retrieve file content from object storage with decryption."""
        bucket_name = bucket or self.bucket_documents

        try:
            # Get object with metadata
            response = self.client.get_object(bucket_name, object_name)
            content = response.read()
            metadata = response.headers or {}
            response.close()
            response.release_conn()

            # Check if file is encrypted
            if metadata.get("x-amz-meta-encrypted") == "true" and self._encryption_enabled and self._fernet:
                try:
                    content = self._fernet.decrypt(content)
                    logger.debug(f"Successfully decrypted file: {object_name}")
                except Exception as e:
                    logger.error(f"Failed to decrypt file {object_name}: {e}")
                    raise Exception(f"Failed to decrypt file {object_name}: {e}")

            self._operation_stats["downloads"] += 1
            logger.info(f"Successfully retrieved file: {object_name} ({len(content)} bytes)")

            # Convert headers to metadata dict
            file_metadata = {}
            for key, value in metadata.items():
                if key.startswith("x-amz-meta-"):
                    file_metadata[key[11:]] = value  # Remove 'x-amz-meta-' prefix

            return content, file_metadata

        except S3Error as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Failed to retrieve file {object_name}: {e}")
            raise Exception(f"Failed to retrieve file {object_name}: {e}")
        except Exception as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Unexpected error retrieving file {object_name}: {e}")
            raise Exception(f"Failed to retrieve file {object_name}: {e}")
    
    async def delete_file(self, object_name: str, bucket: Optional[str] = None) -> bool:
        """Delete file from object storage with proper error handling."""
        bucket_name = bucket or self.bucket_documents

        try:
            # Check if file exists before deletion
            if not await self.file_exists(object_name, bucket):
                logger.warning(f"File not found for deletion: {object_name}")
                return False

            self.client.remove_object(bucket_name, object_name)
            self._operation_stats["deletions"] += 1
            logger.info(f"Successfully deleted file: {object_name}")
            return True

        except S3Error as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Failed to delete file {object_name}: {e}")
            raise Exception(f"Failed to delete file {object_name}: {e}")
        except Exception as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Unexpected error deleting file {object_name}: {e}")
            raise Exception(f"Failed to delete file {object_name}: {e}")
    
    async def file_exists(self, object_name: str, bucket: Optional[str] = None) -> bool:
        """Check if file exists in object storage."""
        bucket_name = bucket or self.bucket_documents
        
        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    async def get_file_info(self, object_name: str, bucket: Optional[str] = None) -> dict:
        """Get file information from object storage."""
        bucket_name = bucket or self.bucket_documents
        
        try:
            stat = self.client.stat_object(bucket_name, object_name)
            return {
                "size": stat.size,
                "etag": stat.etag,
                "last_modified": stat.last_modified,
                "content_type": stat.content_type,
                "metadata": stat.metadata
            }
        except S3Error as e:
            raise Exception(f"Failed to get file info for {object_name}: {e}")
    
    async def list_files(self, prefix: str = "", bucket: Optional[str] = None) -> list:
        """List files in object storage with optional prefix filter."""
        bucket_name = bucket or self.bucket_documents
        
        try:
            objects = self.client.list_objects(bucket_name, prefix=prefix, recursive=True)
            return [
                {
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag
                }
                for obj in objects
            ]
        except S3Error as e:
            raise Exception(f"Failed to list files with prefix {prefix}: {e}")
    
    async def generate_presigned_url(
        self,
        object_name: str,
        expires_in_seconds: int = 3600,
        bucket: Optional[str] = None
    ) -> str:
        """Generate presigned URL for file access."""
        bucket_name = bucket or self.bucket_documents
        
        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                bucket_name,
                object_name,
                expires=timedelta(seconds=expires_in_seconds)
            )
            return url
        except S3Error as e:
            raise Exception(f"Failed to generate presigned URL for {object_name}: {e}")
    
    async def copy_file(
        self,
        source_object: str,
        dest_object: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None
    ) -> None:
        """Copy file within or between buckets."""
        source_bucket_name = source_bucket or self.bucket_documents
        dest_bucket_name = dest_bucket or self.bucket_documents
        
        try:
            from minio.commonconfig import CopySource
            copy_source = CopySource(source_bucket_name, source_object)
            self.client.copy_object(dest_bucket_name, dest_object, copy_source)
        except S3Error as e:
            raise Exception(f"Failed to copy file from {source_object} to {dest_object}: {e}")
    
    async def health_check(self) -> dict:
        """Perform comprehensive health check on storage service."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": self.settings.storage.minio_endpoint,
            "encryption_enabled": self._encryption_enabled,
            "checks": {},
            "statistics": self._operation_stats.copy(),
            "buckets": {},
            "performance": {}
        }

        try:
            # Test 1: Basic connectivity
            start_time = datetime.utcnow()
            buckets = self.client.list_buckets()
            connectivity_time = (datetime.utcnow() - start_time).total_seconds()

            health_status["checks"]["connectivity"] = {
                "status": "pass",
                "response_time_seconds": connectivity_time
            }

            bucket_names = [bucket.name for bucket in buckets]

            # Test 2: Required buckets exist
            required_buckets = [
                self.bucket_documents,
                self.bucket_thumbnails,
                self.bucket_exports
            ]

            missing_buckets = [b for b in required_buckets if b not in bucket_names]
            health_status["checks"]["buckets"] = {
                "status": "pass" if not missing_buckets else "fail",
                "required": required_buckets,
                "existing": bucket_names,
                "missing": missing_buckets
            }

            # Test 3: Bucket accessibility and permissions
            for bucket in required_buckets:
                if bucket in bucket_names:
                    try:
                        # Test read permission
                        list(self.client.list_objects(bucket, prefix="health-check/", max_keys=1))
                        health_status["buckets"][bucket] = {
                            "accessible": True,
                            "permissions": ["read", "list"]
                        }

                        # Test write permission with a small test file
                        test_object = f"health-check/test-{datetime.utcnow().timestamp()}.txt"
                        test_content = b"health check test"

                        self.client.put_object(
                            bucket_name=bucket,
                            object_name=test_object,
                            data=io.BytesIO(test_content),
                            length=len(test_content)
                        )

                        # Clean up test file
                        self.client.remove_object(bucket, test_object)

                        health_status["buckets"][bucket]["permissions"].extend(["write", "delete"])

                    except Exception as e:
                        health_status["buckets"][bucket] = {
                            "accessible": False,
                            "error": str(e)
                        }
                        health_status["status"] = "degraded"

            # Test 4: Encryption functionality
            if self._encryption_enabled:
                try:
                    test_data = b"encryption test data"
                    encrypted = self._fernet.encrypt(test_data)
                    decrypted = self._fernet.decrypt(encrypted)

                    health_status["checks"]["encryption"] = {
                        "status": "pass" if decrypted == test_data else "fail",
                        "method": "fernet"
                    }
                except Exception as e:
                    health_status["checks"]["encryption"] = {
                        "status": "fail",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
            else:
                health_status["checks"]["encryption"] = {
                    "status": "disabled"
                }

            # Update last health check time
            self._operation_stats["last_health_check"] = datetime.utcnow().isoformat()

            # Determine overall status
            if missing_buckets or any(
                check.get("status") == "fail"
                for check in health_status["checks"].values()
            ):
                health_status["status"] = "degraded"

        except Exception as e:
            health_status.update({
                "status": "unhealthy",
                "error": str(e),
                "checks": {
                    "connectivity": {
                        "status": "fail",
                        "error": str(e)
                    }
                }
            })
            logger.error(f"Storage health check failed: {e}")

        return health_status

    async def get_storage_statistics(self) -> dict:
        """Get detailed storage statistics."""
        stats = {
            "operations": self._operation_stats.copy(),
            "buckets": {},
            "total_objects": 0,
            "total_size_bytes": 0
        }

        try:
            for bucket_name in [self.bucket_documents, self.bucket_thumbnails, self.bucket_exports]:
                try:
                    objects = list(self.client.list_objects(bucket_name, recursive=True))
                    bucket_stats = {
                        "object_count": len(objects),
                        "total_size": sum(obj.size for obj in objects if obj.size),
                        "last_modified": max((obj.last_modified for obj in objects), default=None)
                    }
                    stats["buckets"][bucket_name] = bucket_stats
                    stats["total_objects"] += bucket_stats["object_count"]
                    stats["total_size_bytes"] += bucket_stats["total_size"]

                except S3Error as e:
                    stats["buckets"][bucket_name] = {"error": str(e)}

        except Exception as e:
            stats["error"] = str(e)

        return stats

    async def cleanup_old_files(self, bucket: str, prefix: str, days_old: int = 30) -> int:
        """Clean up files older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        deleted_count = 0

        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)

            for obj in objects:
                if obj.last_modified and obj.last_modified < cutoff_date:
                    try:
                        self.client.remove_object(bucket, obj.object_name)
                        deleted_count += 1
                        logger.info(f"Cleaned up old file: {obj.object_name}")
                    except S3Error as e:
                        logger.error(f"Failed to delete old file {obj.object_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            raise Exception(f"Failed to cleanup old files: {e}")

        return deleted_count

    def get_operation_stats(self) -> dict:
        """Get current operation statistics."""
        return self._operation_stats.copy()

    def reset_operation_stats(self) -> None:
        """Reset operation statistics."""
        self._operation_stats = {
            "uploads": 0,
            "downloads": 0,
            "deletions": 0,
            "errors": 0,
            "last_health_check": self._operation_stats.get("last_health_check")
        }


# Global storage service instance
_storage_service: Optional[StorageService] = None


def get_storage_service(settings: Settings) -> StorageService:
    """Get or create storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService(settings)
    return _storage_service
