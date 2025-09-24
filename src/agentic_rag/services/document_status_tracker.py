"""
Document Status Tracking Service.

This module provides comprehensive status tracking for documents throughout
the processing pipeline, including parsing, chunking, and error handling.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import and_

from agentic_rag.models.database import Document, DocumentStatus
from agentic_rag.services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class DocumentStatusUpdate:
    """Represents a document status update."""
    
    def __init__(
        self,
        document_id: UUID,
        status: DocumentStatus,
        progress: float = 0.0,
        message: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.document_id = document_id
        self.status = status
        self.progress = progress
        self.message = message
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)


class DocumentStatusTracker:
    """
    Service for tracking document processing status throughout the pipeline.
    
    This service provides:
    - Status updates during processing stages
    - Progress tracking for long-running operations
    - Error handling and reporting
    - Real-time notifications via WebSocket
    - Status change history and auditing
    """
    
    def __init__(self):
        self._status_history: Dict[UUID, List[DocumentStatusUpdate]] = {}
        self._active_operations: Dict[UUID, str] = {}
        self._locks: Dict[UUID, asyncio.Lock] = {}
    
    def _get_lock(self, document_id: UUID) -> asyncio.Lock:
        """Get or create a lock for a document."""
        if document_id not in self._locks:
            self._locks[document_id] = asyncio.Lock()
        return self._locks[document_id]
    
    async def update_status(
        self,
        db_session: Session,
        document_id: UUID,
        tenant_id: UUID,
        status: DocumentStatus,
        progress: float = 0.0,
        message: Optional[str] = None,
        error: Optional[str] = None,
        chunk_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update document status and progress.
        
        Args:
            db_session: Database session
            document_id: Document ID
            tenant_id: Tenant ID for security
            status: New document status
            progress: Processing progress (0.0-1.0)
            message: Status message
            error: Error message if failed
            chunk_count: Number of chunks created
            metadata: Additional metadata
            
        Returns:
            True if update successful, False otherwise
        """
        async with self._get_lock(document_id):
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
                    logger.warning(f"Document {document_id} not found for status update")
                    return False
                
                # Update document status
                old_status = document.status
                document.status = status
                document.processing_progress = max(0.0, min(1.0, progress))
                document.updated_at = datetime.now(timezone.utc)
                
                if error:
                    document.processing_error = error
                elif status == DocumentStatus.READY:
                    document.processing_error = None  # Clear error on success
                
                if chunk_count is not None:
                    document.chunk_count = chunk_count
                
                # Commit changes
                db_session.commit()
                
                # Create status update record
                status_update = DocumentStatusUpdate(
                    document_id=document_id,
                    status=status,
                    progress=progress,
                    message=message,
                    error=error,
                    metadata=metadata
                )
                
                # Store in history
                if document_id not in self._status_history:
                    self._status_history[document_id] = []
                self._status_history[document_id].append(status_update)
                
                # Send real-time notification
                await self._send_status_notification(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    status_update=status_update,
                    old_status=old_status
                )
                
                logger.info(
                    f"Updated document {document_id} status: {old_status} -> {status} "
                    f"(progress: {progress:.1%})"
                )
                
                return True
                
            except Exception as e:
                db_session.rollback()
                logger.error(f"Failed to update document {document_id} status: {e}")
                return False
    
    async def start_processing(
        self,
        db_session: Session,
        document_id: UUID,
        tenant_id: UUID,
        operation: str,
        message: Optional[str] = None
    ) -> bool:
        """
        Mark document as starting processing.
        
        Args:
            db_session: Database session
            document_id: Document ID
            tenant_id: Tenant ID
            operation: Processing operation name
            message: Optional status message
            
        Returns:
            True if successful
        """
        self._active_operations[document_id] = operation
        
        return await self.update_status(
            db_session=db_session,
            document_id=document_id,
            tenant_id=tenant_id,
            status=DocumentStatus.PROCESSING,
            progress=0.0,
            message=message or f"Starting {operation}",
            metadata={"operation": operation}
        )
    
    async def update_progress(
        self,
        db_session: Session,
        document_id: UUID,
        tenant_id: UUID,
        progress: float,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update processing progress without changing status.
        
        Args:
            db_session: Database session
            document_id: Document ID
            tenant_id: Tenant ID
            progress: Processing progress (0.0-1.0)
            message: Progress message
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        return await self.update_status(
            db_session=db_session,
            document_id=document_id,
            tenant_id=tenant_id,
            status=DocumentStatus.PROCESSING,
            progress=progress,
            message=message,
            metadata=metadata
        )
    
    async def complete_processing(
        self,
        db_session: Session,
        document_id: UUID,
        tenant_id: UUID,
        chunk_count: int,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark document processing as complete.
        
        Args:
            db_session: Database session
            document_id: Document ID
            tenant_id: Tenant ID
            chunk_count: Number of chunks created
            message: Completion message
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        # Remove from active operations
        self._active_operations.pop(document_id, None)
        
        return await self.update_status(
            db_session=db_session,
            document_id=document_id,
            tenant_id=tenant_id,
            status=DocumentStatus.READY,
            progress=1.0,
            message=message or "Processing completed successfully",
            chunk_count=chunk_count,
            metadata=metadata
        )
    
    async def fail_processing(
        self,
        db_session: Session,
        document_id: UUID,
        tenant_id: UUID,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark document processing as failed.
        
        Args:
            db_session: Database session
            document_id: Document ID
            tenant_id: Tenant ID
            error: Error message
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        # Remove from active operations
        operation = self._active_operations.pop(document_id, "processing")
        
        return await self.update_status(
            db_session=db_session,
            document_id=document_id,
            tenant_id=tenant_id,
            status=DocumentStatus.FAILED,
            progress=0.0,
            message=f"Processing failed: {error}",
            error=error,
            metadata={**(metadata or {}), "failed_operation": operation}
        )
    
    async def _send_status_notification(
        self,
        document_id: UUID,
        tenant_id: UUID,
        status_update: DocumentStatusUpdate,
        old_status: DocumentStatus
    ):
        """Send real-time status notification via WebSocket."""
        try:
            notification_data = {
                "type": "document_status_update",
                "document_id": str(document_id),
                "old_status": old_status.value,
                "new_status": status_update.status.value,
                "progress": status_update.progress,
                "message": status_update.message,
                "error": status_update.error,
                "timestamp": status_update.timestamp.isoformat(),
                "metadata": status_update.metadata
            }
            
            # Send to all tenant users
            await websocket_manager.send_to_tenant(tenant_id, notification_data)
            
        except Exception as e:
            logger.error(f"Failed to send status notification for document {document_id}: {e}")
    
    def get_status_history(self, document_id: UUID) -> List[DocumentStatusUpdate]:
        """Get status history for a document."""
        return self._status_history.get(document_id, [])
    
    def get_active_operations(self) -> Dict[UUID, str]:
        """Get currently active processing operations."""
        return self._active_operations.copy()
    
    def cleanup_history(self, document_id: UUID):
        """Clean up status history for a document."""
        self._status_history.pop(document_id, None)
        self._active_operations.pop(document_id, None)
        self._locks.pop(document_id, None)


# Global instance
_document_status_tracker: Optional[DocumentStatusTracker] = None


def get_document_status_tracker() -> DocumentStatusTracker:
    """Get the global document status tracker instance."""
    global _document_status_tracker
    if _document_status_tracker is None:
        _document_status_tracker = DocumentStatusTracker()
    return _document_status_tracker
