"""
Indexing Trigger Service

This module provides automatic triggering of vector indexing
after document processing completion.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session

from agentic_rag.services.vector_indexing_pipeline import (
    get_indexing_pipeline,
    IndexingRequest,
    IndexingPriority
)
from agentic_rag.services.chunking.deduplication_chunker import DeduplicatedChunk
from agentic_rag.services.chunking.pipeline import ChunkingResult
from agentic_rag.models.database import Document, DocumentChunk
from agentic_rag.config import get_settings

logger = logging.getLogger(__name__)


class IndexingTrigger:
    """Service for triggering automatic vector indexing after document processing."""
    
    def __init__(self):
        self._indexing_pipeline = None
        self._settings = get_settings()
        
        logger.info("Indexing trigger service initialized")
    
    async def initialize(self) -> None:
        """Initialize the indexing trigger service."""
        self._indexing_pipeline = await get_indexing_pipeline()
        logger.info("Indexing trigger service ready")
    
    async def trigger_document_indexing(
        self,
        document_id: UUID,
        tenant_id: UUID,
        chunking_result: ChunkingResult,
        db_session: Session,
        priority: IndexingPriority = IndexingPriority.NORMAL
    ) -> str:
        """
        Trigger automatic vector indexing for a processed document.
        
        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier
            chunking_result: Result from document chunking
            db_session: Database session
            priority: Indexing priority level
            
        Returns:
            Indexing request ID for tracking
        """
        try:
            # Get document information
            document = db_session.query(Document).filter(
                Document.id == document_id,
                Document.tenant_id == tenant_id
            ).first()
            
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Prepare chunk data for indexing
            chunks_data = []
            
            for chunk in chunking_result.chunks:
                chunk_metadata = {
                    "document_kind": document.document_kind.value if document.document_kind else "UNKNOWN",
                    "document_name": document.name,
                    "document_type": document.document_type.value if document.document_type else "UNKNOWN",
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "is_table": getattr(chunk, 'is_table', False),
                    "section_path": getattr(chunk, 'section_path', []),
                    "page_from": getattr(chunk, 'page_from', None),
                    "page_to": getattr(chunk, 'page_to', None),
                    "quality_score": getattr(chunk, 'quality_score', None),
                    "is_duplicate": chunk.deduplication_metadata.is_duplicate if hasattr(chunk, 'deduplication_metadata') else False,
                    "similarity_score": chunk.deduplication_metadata.similarity_score if hasattr(chunk, 'deduplication_metadata') else None,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                chunk_data = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk_metadata
                }
                
                chunks_data.append(chunk_data)
            
            # Create indexing request
            indexing_request = IndexingRequest(
                document_id=str(document_id),
                tenant_id=str(tenant_id),
                chunks=chunks_data,
                priority=priority,
                metadata={
                    "document_name": document.name,
                    "document_type": document.document_type.value if document.document_type else "UNKNOWN",
                    "document_kind": document.document_kind.value if document.document_kind else "UNKNOWN",
                    "total_chunks": len(chunks_data),
                    "triggered_at": datetime.utcnow().isoformat()
                }
            )
            
            # Submit for indexing
            request_id = await self._indexing_pipeline.submit_indexing_request(indexing_request)
            
            logger.info(
                f"Triggered indexing for document {document_id}: "
                f"{len(chunks_data)} chunks, request_id: {request_id}"
            )
            
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to trigger indexing for document {document_id}: {e}")
            raise
    
    async def trigger_chunk_reindexing(
        self,
        document_id: UUID,
        tenant_id: UUID,
        chunk_ids: List[str],
        db_session: Session,
        priority: IndexingPriority = IndexingPriority.HIGH
    ) -> str:
        """
        Trigger reindexing for specific document chunks.
        
        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier
            chunk_ids: List of chunk IDs to reindex
            db_session: Database session
            priority: Indexing priority level
            
        Returns:
            Indexing request ID for tracking
        """
        try:
            # Get document information
            document = db_session.query(Document).filter(
                Document.id == document_id,
                Document.tenant_id == tenant_id
            ).first()
            
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Get chunk data from database
            chunks = db_session.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.chunk_id.in_(chunk_ids)
            ).all()
            
            if not chunks:
                raise ValueError(f"No chunks found for document {document_id}")
            
            # Prepare chunk data for reindexing
            chunks_data = []
            
            for chunk in chunks:
                chunk_metadata = {
                    "document_kind": document.document_kind.value if document.document_kind else "UNKNOWN",
                    "document_name": document.name,
                    "document_type": document.document_type.value if document.document_type else "UNKNOWN",
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "is_table": chunk.is_table or False,
                    "section_path": chunk.section_path or [],
                    "page_from": chunk.page_from,
                    "page_to": chunk.page_to,
                    "quality_score": chunk.quality_score,
                    "reindexing": True,
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                chunk_data = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk_metadata
                }
                
                chunks_data.append(chunk_data)
            
            # Create reindexing request
            indexing_request = IndexingRequest(
                document_id=str(document_id),
                tenant_id=str(tenant_id),
                chunks=chunks_data,
                priority=priority,
                metadata={
                    "document_name": document.name,
                    "document_type": document.document_type.value if document.document_type else "UNKNOWN",
                    "document_kind": document.document_kind.value if document.document_kind else "UNKNOWN",
                    "total_chunks": len(chunks_data),
                    "reindexing": True,
                    "triggered_at": datetime.utcnow().isoformat()
                }
            )
            
            # Submit for reindexing
            request_id = await self._indexing_pipeline.submit_indexing_request(indexing_request)
            
            logger.info(
                f"Triggered reindexing for document {document_id}: "
                f"{len(chunks_data)} chunks, request_id: {request_id}"
            )
            
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to trigger reindexing for document {document_id}: {e}")
            raise
    
    async def get_indexing_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an indexing request."""
        try:
            result = await self._indexing_pipeline.get_indexing_status(request_id)
            
            if result:
                return {
                    "request_id": result.request_id,
                    "document_id": result.document_id,
                    "status": result.status.value,
                    "total_chunks": result.total_chunks,
                    "indexed_chunks": result.indexed_chunks,
                    "failed_chunks": result.failed_chunks,
                    "processing_time": result.processing_time,
                    "errors": result.errors,
                    "vector_ids": result.vector_ids,
                    "created_at": result.created_at.isoformat(),
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get indexing status for request {request_id}: {e}")
            return None
    
    async def cancel_indexing(self, request_id: str) -> bool:
        """Cancel a pending or processing indexing request."""
        try:
            return await self._indexing_pipeline.cancel_indexing_request(request_id)
            
        except Exception as e:
            logger.error(f"Failed to cancel indexing request {request_id}: {e}")
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current indexing queue status."""
        try:
            return await self._indexing_pipeline.get_queue_status()
            
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        try:
            return self._indexing_pipeline.get_statistics()
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on indexing trigger service."""
        try:
            if not self._indexing_pipeline:
                return {
                    "status": "unhealthy",
                    "error": "Indexing pipeline not initialized",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            pipeline_health = await self._indexing_pipeline.health_check()
            
            return {
                "status": "healthy" if pipeline_health.get("status") == "healthy" else "unhealthy",
                "pipeline_health": pipeline_health,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Indexing trigger health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global indexing trigger instance
_indexing_trigger: Optional[IndexingTrigger] = None


async def get_indexing_trigger() -> IndexingTrigger:
    """Get or create the global indexing trigger instance."""
    global _indexing_trigger
    
    if _indexing_trigger is None:
        _indexing_trigger = IndexingTrigger()
        await _indexing_trigger.initialize()
    
    return _indexing_trigger


async def close_indexing_trigger() -> None:
    """Close the global indexing trigger instance."""
    global _indexing_trigger
    
    if _indexing_trigger:
        _indexing_trigger = None


# Convenience functions for automatic triggering
async def trigger_document_indexing_after_processing(
    document_id: UUID,
    tenant_id: UUID,
    chunking_result: ChunkingResult,
    db_session: Session,
    priority: IndexingPriority = IndexingPriority.NORMAL
) -> str:
    """
    Convenience function to trigger document indexing after processing.
    
    Args:
        document_id: Document identifier
        tenant_id: Tenant identifier
        chunking_result: Result from document chunking
        db_session: Database session
        priority: Indexing priority level
        
    Returns:
        Indexing request ID for tracking
    """
    trigger = await get_indexing_trigger()
    return await trigger.trigger_document_indexing(
        document_id=document_id,
        tenant_id=tenant_id,
        chunking_result=chunking_result,
        db_session=db_session,
        priority=priority
    )


async def get_document_indexing_status(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get document indexing status.
    
    Args:
        request_id: Indexing request ID
        
    Returns:
        Indexing status information
    """
    trigger = await get_indexing_trigger()
    return await trigger.get_indexing_status(request_id)
