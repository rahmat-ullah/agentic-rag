"""
Document Cleanup Service.

This service handles comprehensive cleanup of documents and associated data,
including chunks, metadata, files, and vector embeddings.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from agentic_rag.models.database import Document, DocumentStatus, ChunkMeta, DocumentLink
from agentic_rag.services.storage import get_storage_service
from agentic_rag.services.document_status_tracker import get_document_status_tracker

logger = logging.getLogger(__name__)


class CleanupResult:
    """Result of document cleanup operation."""
    
    def __init__(
        self,
        document_id: UUID,
        success: bool,
        chunks_removed: int = 0,
        links_removed: int = 0,
        file_removed: bool = False,
        error: Optional[str] = None,
        cleanup_time: float = 0.0
    ):
        self.document_id = document_id
        self.success = success
        self.chunks_removed = chunks_removed
        self.links_removed = links_removed
        self.file_removed = file_removed
        self.error = error
        self.cleanup_time = cleanup_time


class DocumentCleanupService:
    """
    Service for comprehensive document cleanup.
    
    This service handles:
    - Soft delete with retention period
    - Hard delete with complete cleanup
    - Chunk metadata removal
    - Document link cleanup
    - File removal from object storage
    - Vector embedding cleanup (if applicable)
    - Bulk cleanup operations
    """
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.storage_service = None
        self.status_tracker = get_document_status_tracker()
    
    async def initialize(self):
        """Initialize async components."""
        if not self.storage_service:
            self.storage_service = get_storage_service()
        logger.info("Document cleanup service initialized")
    
    async def soft_delete_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
        db_session: Session,
        user_id: Optional[UUID] = None
    ) -> CleanupResult:
        """
        Perform soft delete of a document.
        
        Args:
            document_id: Document ID to delete
            tenant_id: Tenant ID for security
            db_session: Database session
            user_id: User performing the deletion
            
        Returns:
            CleanupResult with deletion outcome
        """
        import time
        start_time = time.time()
        
        try:
            # Get document with tenant isolation
            document = db_session.query(Document).filter(
                and_(
                    Document.id == document_id,
                    Document.tenant_id == tenant_id,
                    Document.deleted_at.is_(None)
                )
            ).first()
            
            if not document:
                return CleanupResult(
                    document_id=document_id,
                    success=False,
                    error="Document not found or already deleted"
                )
            
            # Update document status tracking
            await self.status_tracker.update_status(
                db_session=db_session,
                document_id=document_id,
                tenant_id=tenant_id,
                status=DocumentStatus.DELETED,
                progress=1.0,
                message="Document soft deleted",
                metadata={
                    "deletion_type": "soft",
                    "deleted_by": str(user_id) if user_id else None,
                    "retention_until": (datetime.now(timezone.utc) + timedelta(days=self.retention_days)).isoformat()
                }
            )
            
            # Soft delete the document
            document.deleted_at = datetime.now(timezone.utc)
            document.status = DocumentStatus.DELETED
            db_session.commit()
            
            cleanup_time = time.time() - start_time
            
            logger.info(f"Document {document_id} soft deleted successfully")
            
            return CleanupResult(
                document_id=document_id,
                success=True,
                cleanup_time=cleanup_time
            )
            
        except Exception as e:
            db_session.rollback()
            cleanup_time = time.time() - start_time
            error_message = str(e)
            
            logger.error(f"Failed to soft delete document {document_id}: {error_message}")
            
            return CleanupResult(
                document_id=document_id,
                success=False,
                error=error_message,
                cleanup_time=cleanup_time
            )
    
    async def hard_delete_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
        db_session: Session,
        remove_file: bool = True
    ) -> CleanupResult:
        """
        Perform hard delete of a document with complete cleanup.
        
        Args:
            document_id: Document ID to delete
            tenant_id: Tenant ID for security
            db_session: Database session
            remove_file: Whether to remove file from storage
            
        Returns:
            CleanupResult with deletion outcome
        """
        import time
        start_time = time.time()
        
        try:
            await self.initialize()
            
            # Get document with tenant isolation
            document = db_session.query(Document).filter(
                and_(
                    Document.id == document_id,
                    Document.tenant_id == tenant_id
                )
            ).first()
            
            if not document:
                return CleanupResult(
                    document_id=document_id,
                    success=False,
                    error="Document not found"
                )
            
            source_uri = document.source_uri
            chunks_removed = 0
            links_removed = 0
            file_removed = False
            
            # Step 1: Remove chunk metadata
            chunks = db_session.query(ChunkMeta).filter(
                and_(
                    ChunkMeta.document_id == document_id,
                    ChunkMeta.tenant_id == tenant_id
                )
            ).all()
            
            for chunk in chunks:
                db_session.delete(chunk)
                chunks_removed += 1
            
            # Step 2: Remove document links
            # Remove links where this document is the RFQ
            rfq_links = db_session.query(DocumentLink).filter(
                and_(
                    DocumentLink.rfq_id == document_id,
                    DocumentLink.tenant_id == tenant_id
                )
            ).all()
            
            for link in rfq_links:
                db_session.delete(link)
                links_removed += 1
            
            # Remove links where this document is the offer
            offer_links = db_session.query(DocumentLink).filter(
                and_(
                    DocumentLink.offer_id == document_id,
                    DocumentLink.tenant_id == tenant_id
                )
            ).all()
            
            for link in offer_links:
                db_session.delete(link)
                links_removed += 1
            
            # Step 3: Remove document record
            db_session.delete(document)
            db_session.commit()
            
            # Step 4: Remove file from storage
            if remove_file and source_uri:
                try:
                    # Extract object name from S3 URI
                    if source_uri.startswith("s3://"):
                        parts = source_uri[5:].split("/", 1)
                        if len(parts) == 2:
                            bucket, object_name = parts
                            await self.storage_service.delete_file(object_name)
                            file_removed = True
                except Exception as e:
                    logger.warning(f"Failed to remove file {source_uri}: {e}")
            
            # Step 5: Clean up status tracking
            self.status_tracker.cleanup_history(document_id)
            
            cleanup_time = time.time() - start_time
            
            logger.info(
                f"Document {document_id} hard deleted: "
                f"{chunks_removed} chunks, {links_removed} links, file: {file_removed}"
            )
            
            return CleanupResult(
                document_id=document_id,
                success=True,
                chunks_removed=chunks_removed,
                links_removed=links_removed,
                file_removed=file_removed,
                cleanup_time=cleanup_time
            )
            
        except Exception as e:
            db_session.rollback()
            cleanup_time = time.time() - start_time
            error_message = str(e)
            
            logger.error(f"Failed to hard delete document {document_id}: {error_message}")
            
            return CleanupResult(
                document_id=document_id,
                success=False,
                error=error_message,
                cleanup_time=cleanup_time
            )
    
    async def cleanup_expired_documents(
        self,
        tenant_id: UUID,
        db_session: Session,
        batch_size: int = 100
    ) -> List[CleanupResult]:
        """
        Clean up documents that have exceeded the retention period.
        
        Args:
            tenant_id: Tenant ID
            db_session: Database session
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of CleanupResult for each processed document
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        
        # Find expired documents
        expired_documents = db_session.query(Document).filter(
            and_(
                Document.tenant_id == tenant_id,
                Document.deleted_at.is_not(None),
                Document.deleted_at <= cutoff_date
            )
        ).limit(batch_size).all()
        
        results = []
        
        for document in expired_documents:
            result = await self.hard_delete_document(
                document_id=document.id,
                tenant_id=tenant_id,
                db_session=db_session,
                remove_file=True
            )
            results.append(result)
        
        logger.info(f"Cleaned up {len(results)} expired documents for tenant {tenant_id}")
        return results
    
    async def bulk_delete_documents(
        self,
        document_ids: List[UUID],
        tenant_id: UUID,
        db_session: Session,
        hard_delete: bool = False,
        user_id: Optional[UUID] = None
    ) -> List[CleanupResult]:
        """
        Perform bulk deletion of documents.
        
        Args:
            document_ids: List of document IDs to delete
            tenant_id: Tenant ID for security
            db_session: Database session
            hard_delete: Whether to perform hard delete
            user_id: User performing the deletion
            
        Returns:
            List of CleanupResult for each document
        """
        results = []
        
        for document_id in document_ids:
            if hard_delete:
                result = await self.hard_delete_document(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    db_session=db_session
                )
            else:
                result = await self.soft_delete_document(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    db_session=db_session,
                    user_id=user_id
                )
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Bulk deleted {successful}/{len(document_ids)} documents")
        
        return results


# Global instance
_cleanup_service: Optional[DocumentCleanupService] = None


def get_document_cleanup_service(retention_days: int = 30) -> DocumentCleanupService:
    """Get the global document cleanup service instance."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = DocumentCleanupService(retention_days=retention_days)
    return _cleanup_service
