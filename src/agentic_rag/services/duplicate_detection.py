"""
Duplicate Detection Service for Task 4: SHA256-based duplicate detection.

This module implements comprehensive duplicate detection with tenant isolation,
version handling, and statistics reporting.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from agentic_rag.config import Settings
from agentic_rag.models.database import Document, DocumentKind

logger = logging.getLogger(__name__)


class DuplicateAction(str, Enum):
    """Actions taken when duplicates are detected."""
    
    SKIPPED = "skipped"
    OVERWRITTEN = "overwritten"
    VERSIONED = "versioned"
    UPLOADED = "uploaded"


class DuplicateDetectionResult:
    """Result of duplicate detection analysis."""
    
    def __init__(
        self,
        is_duplicate: bool,
        sha256_hash: str,
        action_taken: DuplicateAction,
        existing_document_id: Optional[UUID] = None,
        existing_filename: Optional[str] = None,
        existing_upload_date: Optional[datetime] = None,
        existing_version: Optional[int] = None,
        new_version: Optional[int] = None,
        duplicate_count: int = 0,
        tenant_duplicate_count: int = 0
    ):
        self.is_duplicate = is_duplicate
        self.sha256_hash = sha256_hash
        self.action_taken = action_taken
        self.existing_document_id = existing_document_id
        self.existing_filename = existing_filename
        self.existing_upload_date = existing_upload_date
        self.existing_version = existing_version
        self.new_version = new_version
        self.duplicate_count = duplicate_count
        self.tenant_duplicate_count = tenant_duplicate_count


class DuplicateStatistics:
    """Statistics about duplicate detection for a tenant."""
    
    def __init__(
        self,
        tenant_id: UUID,
        total_documents: int,
        unique_documents: int,
        duplicate_documents: int,
        duplicate_percentage: float,
        most_duplicated_hash: Optional[str] = None,
        most_duplicated_count: int = 0,
        recent_duplicates: int = 0,
        storage_saved_bytes: int = 0
    ):
        self.tenant_id = tenant_id
        self.total_documents = total_documents
        self.unique_documents = unique_documents
        self.duplicate_documents = duplicate_documents
        self.duplicate_percentage = duplicate_percentage
        self.most_duplicated_hash = most_duplicated_hash
        self.most_duplicated_count = most_duplicated_count
        self.recent_duplicates = recent_duplicates
        self.storage_saved_bytes = storage_saved_bytes


class DuplicateDetectionService:
    """Service for SHA256-based duplicate detection with tenant isolation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enable_versioning = getattr(settings.upload, 'enable_document_versioning', True)
        self.max_versions = getattr(settings.upload, 'max_document_versions', 10)
    
    def calculate_sha256(self, content: bytes) -> str:
        """Calculate SHA256 hash for file content."""
        return hashlib.sha256(content).hexdigest()
    
    async def detect_duplicate(
        self,
        tenant_id: UUID,
        sha256_hash: str,
        filename: str,
        content_type: str,
        file_size: int,
        db_session: Session,
        overwrite_existing: bool = False,
        create_version: bool = False
    ) -> DuplicateDetectionResult:
        """
        Detect duplicate files and determine appropriate action.
        
        Args:
            tenant_id: Tenant ID for isolation
            sha256_hash: SHA256 hash of the file content
            filename: Original filename
            content_type: MIME type of the file
            file_size: Size of the file in bytes
            db_session: Database session
            overwrite_existing: Whether to overwrite existing files
            create_version: Whether to create a new version
            
        Returns:
            DuplicateDetectionResult with detection results and recommended action
        """
        # Check for existing documents with same hash in tenant
        existing_docs = db_session.query(Document).filter(
            and_(
                Document.tenant_id == tenant_id,
                Document.sha256 == sha256_hash
            )
        ).order_by(Document.version.desc()).all()
        
        if not existing_docs:
            # No duplicates found
            return DuplicateDetectionResult(
                is_duplicate=False,
                sha256_hash=sha256_hash,
                action_taken=DuplicateAction.UPLOADED,
                duplicate_count=0,
                tenant_duplicate_count=0
            )
        
        # Get the latest version
        latest_doc = existing_docs[0]
        duplicate_count = len(existing_docs)
        
        # Get tenant-wide duplicate statistics
        tenant_duplicate_count = self._get_tenant_duplicate_count(tenant_id, sha256_hash, db_session)
        
        # Determine action based on parameters
        if overwrite_existing:
            action = DuplicateAction.OVERWRITTEN
            new_version = latest_doc.version
        elif create_version and self.enable_versioning:
            action = DuplicateAction.VERSIONED
            new_version = latest_doc.version + 1
            
            # Check version limits
            if new_version > self.max_versions:
                logger.warning(f"Maximum versions ({self.max_versions}) reached for document {latest_doc.id}")
                action = DuplicateAction.OVERWRITTEN
                new_version = self.max_versions
        else:
            action = DuplicateAction.SKIPPED
            new_version = None
        
        return DuplicateDetectionResult(
            is_duplicate=True,
            sha256_hash=sha256_hash,
            action_taken=action,
            existing_document_id=latest_doc.id,
            existing_filename=latest_doc.title,
            existing_upload_date=latest_doc.created_at,
            existing_version=latest_doc.version,
            new_version=new_version,
            duplicate_count=duplicate_count,
            tenant_duplicate_count=tenant_duplicate_count
        )
    
    def _get_tenant_duplicate_count(self, tenant_id: UUID, sha256_hash: str, db_session: Session) -> int:
        """Get count of documents with same hash across tenant."""
        return db_session.query(Document).filter(
            and_(
                Document.tenant_id == tenant_id,
                Document.sha256 == sha256_hash
            )
        ).count()
    
    async def get_duplicate_statistics(
        self,
        tenant_id: UUID,
        db_session: Session,
        days_back: int = 30
    ) -> DuplicateStatistics:
        """
        Get comprehensive duplicate statistics for a tenant.
        
        Args:
            tenant_id: Tenant ID
            db_session: Database session
            days_back: Number of days to look back for recent duplicates
            
        Returns:
            DuplicateStatistics with comprehensive duplicate information
        """
        # Get total document count
        total_docs = db_session.query(Document).filter(
            Document.tenant_id == tenant_id
        ).count()
        
        # Get unique document count (distinct SHA256 hashes)
        unique_docs = db_session.query(Document.sha256).filter(
            Document.tenant_id == tenant_id
        ).distinct().count()
        
        # Calculate duplicate documents
        duplicate_docs = total_docs - unique_docs
        duplicate_percentage = (duplicate_docs / total_docs * 100) if total_docs > 0 else 0
        
        # Find most duplicated hash
        most_duplicated = db_session.query(
            Document.sha256,
            func.count(Document.id).label('count')
        ).filter(
            Document.tenant_id == tenant_id
        ).group_by(Document.sha256).order_by(
            func.count(Document.id).desc()
        ).first()
        
        most_duplicated_hash = most_duplicated.sha256 if most_duplicated else None
        most_duplicated_count = most_duplicated.count if most_duplicated else 0
        
        # Get recent duplicates (last N days)
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        recent_duplicates = db_session.query(Document).filter(
            and_(
                Document.tenant_id == tenant_id,
                Document.created_at >= cutoff_date,
                Document.sha256.in_(
                    db_session.query(Document.sha256).filter(
                        Document.tenant_id == tenant_id
                    ).group_by(Document.sha256).having(
                        func.count(Document.id) > 1
                    )
                )
            )
        ).count()
        
        # Calculate storage saved (approximate)
        # This would need file size information to be accurate
        storage_saved_bytes = self._estimate_storage_saved(tenant_id, db_session)
        
        return DuplicateStatistics(
            tenant_id=tenant_id,
            total_documents=total_docs,
            unique_documents=unique_docs,
            duplicate_documents=duplicate_docs,
            duplicate_percentage=duplicate_percentage,
            most_duplicated_hash=most_duplicated_hash,
            most_duplicated_count=most_duplicated_count,
            recent_duplicates=recent_duplicates,
            storage_saved_bytes=storage_saved_bytes
        )
    
    def _estimate_storage_saved(self, tenant_id: UUID, db_session: Session) -> int:
        """Estimate storage saved by duplicate detection."""
        # This is a simplified estimation
        # In a real implementation, you'd need file size information
        duplicate_count = db_session.query(Document).filter(
            and_(
                Document.tenant_id == tenant_id,
                Document.sha256.in_(
                    db_session.query(Document.sha256).filter(
                        Document.tenant_id == tenant_id
                    ).group_by(Document.sha256).having(
                        func.count(Document.id) > 1
                    )
                )
            )
        ).count()
        
        # Estimate average file size (this should be stored in metadata)
        estimated_avg_size = 1024 * 1024  # 1MB average
        return duplicate_count * estimated_avg_size
    
    async def get_document_versions(
        self,
        tenant_id: UUID,
        sha256_hash: str,
        db_session: Session
    ) -> List[Document]:
        """Get all versions of a document by SHA256 hash."""
        return db_session.query(Document).filter(
            and_(
                Document.tenant_id == tenant_id,
                Document.sha256 == sha256_hash
            )
        ).order_by(Document.version.desc()).all()
    
    async def cleanup_old_versions(
        self,
        tenant_id: UUID,
        sha256_hash: str,
        db_session: Session,
        keep_versions: int = None
    ) -> int:
        """
        Clean up old versions of a document, keeping only the specified number.
        
        Returns:
            Number of versions deleted
        """
        if keep_versions is None:
            keep_versions = self.max_versions
        
        versions = await self.get_document_versions(tenant_id, sha256_hash, db_session)
        
        if len(versions) <= keep_versions:
            return 0
        
        # Delete oldest versions
        versions_to_delete = versions[keep_versions:]
        deleted_count = 0
        
        for version in versions_to_delete:
            db_session.delete(version)
            deleted_count += 1
        
        db_session.commit()
        logger.info(f"Cleaned up {deleted_count} old versions for hash {sha256_hash}")
        
        return deleted_count
