"""
Document Processing Orchestrator Service.

This service orchestrates the complete document processing pipeline
with comprehensive status tracking, including parsing, chunking,
and metadata extraction.
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from uuid import UUID

from sqlalchemy.orm import Session

from agentic_rag.models.database import Document, DocumentStatus
from agentic_rag.services.document_parsing import DocumentParsingService
from agentic_rag.services.chunking.pipeline import ChunkingPipeline, get_chunking_pipeline
from agentic_rag.services.document_status_tracker import get_document_status_tracker
from agentic_rag.services.storage import get_storage_service

logger = logging.getLogger(__name__)


class DocumentProcessingResult:
    """Result of complete document processing."""
    
    def __init__(
        self,
        success: bool,
        document_id: UUID,
        status: DocumentStatus,
        chunk_count: int = 0,
        error: Optional[str] = None,
        processing_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.document_id = document_id
        self.status = status
        self.chunk_count = chunk_count
        self.error = error
        self.processing_time = processing_time
        self.metadata = metadata or {}


class DocumentProcessor:
    """
    Orchestrates complete document processing with status tracking.
    
    This service coordinates:
    - Document parsing with Granite-Docling
    - Content extraction and chunking
    - Metadata enrichment
    - Status tracking throughout the pipeline
    - Error handling and recovery
    """
    
    def __init__(self):
        self.status_tracker = get_document_status_tracker()
        self.parsing_service = None
        self.chunking_pipeline = None
        self.storage_service = None
    
    async def initialize(self):
        """Initialize async components."""
        if not self.parsing_service:
            self.parsing_service = DocumentParsingService()
            await self.parsing_service.initialize()
        
        if not self.chunking_pipeline:
            self.chunking_pipeline = get_chunking_pipeline()
        
        if not self.storage_service:
            self.storage_service = get_storage_service()
        
        logger.info("Document processor initialized")
    
    async def process_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
        user_id: UUID,
        db_session: Session,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> DocumentProcessingResult:
        """
        Process a document through the complete pipeline with status tracking.
        
        Args:
            document_id: Document ID to process
            tenant_id: Tenant ID
            user_id: User ID
            db_session: Database session
            processing_options: Optional processing configuration
            
        Returns:
            DocumentProcessingResult with processing outcome
        """
        import time
        start_time = time.time()
        
        try:
            # Ensure components are initialized
            await self.initialize()
            
            # Get document from database
            document = db_session.query(Document).filter(
                Document.id == document_id,
                Document.tenant_id == tenant_id
            ).first()
            
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            if not document.source_uri:
                raise ValueError(f"Document {document_id} has no source file")
            
            # Start processing
            await self.status_tracker.start_processing(
                db_session=db_session,
                document_id=document_id,
                tenant_id=tenant_id,
                operation="document_processing",
                message="Starting document processing pipeline"
            )
            
            # Step 1: Load file content
            await self.status_tracker.update_progress(
                db_session=db_session,
                document_id=document_id,
                tenant_id=tenant_id,
                progress=0.1,
                message="Loading document file",
                metadata={"step": "file_loading"}
            )
            
            file_content = await self._load_document_file(document.source_uri)
            
            # Step 2: Parse document
            await self.status_tracker.update_progress(
                db_session=db_session,
                document_id=document_id,
                tenant_id=tenant_id,
                progress=0.2,
                message="Parsing document with Granite-Docling",
                metadata={"step": "parsing"}
            )
            
            parsing_result = await self.parsing_service.parse_document(
                file_content=file_content,
                filename=document.title,
                tenant_id=tenant_id,
                user_id=user_id,
                parse_options=processing_options
            )
            
            if not parsing_result.success:
                raise Exception(f"Document parsing failed: {parsing_result.error_message}")
            
            # Step 3: Extract and chunk content
            await self.status_tracker.update_progress(
                db_session=db_session,
                document_id=document_id,
                tenant_id=tenant_id,
                progress=0.6,
                message="Extracting and chunking content",
                metadata={"step": "chunking"}
            )
            
            chunking_result = await self.chunking_pipeline.process_document_async(
                parsing_result.extracted_content
            )
            
            # Step 4: Store chunks and update metadata
            await self.status_tracker.update_progress(
                db_session=db_session,
                document_id=document_id,
                tenant_id=tenant_id,
                progress=0.8,
                message="Storing chunks and updating metadata",
                metadata={"step": "storage"}
            )
            
            # Update document with processing results
            document.chunk_count = chunking_result.total_chunks
            document.file_size = len(file_content)
            if hasattr(parsing_result, 'enriched_metadata') and parsing_result.enriched_metadata:
                if hasattr(parsing_result.enriched_metadata, 'page_count'):
                    document.pages = parsing_result.enriched_metadata.page_count
            
            db_session.commit()
            
            # Step 5: Trigger vector indexing
            await self.status_tracker.update_progress(
                db_session=db_session,
                document_id=document_id,
                tenant_id=tenant_id,
                progress=0.9,
                message="Triggering vector indexing",
                metadata={"step": "indexing"}
            )

            # Trigger automatic vector indexing
            try:
                from agentic_rag.services.indexing_trigger import trigger_document_indexing_after_processing

                indexing_request_id = await trigger_document_indexing_after_processing(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    chunking_result=chunking_result,
                    db_session=db_session
                )

                logger.info(f"Triggered vector indexing for document {document_id}, request_id: {indexing_request_id}")

            except Exception as e:
                logger.error(f"Failed to trigger vector indexing for document {document_id}: {e}")
                # Don't fail the entire processing if indexing trigger fails

            # Step 6: Complete processing
            processing_time = time.time() - start_time

            await self.status_tracker.complete_processing(
                db_session=db_session,
                document_id=document_id,
                tenant_id=tenant_id,
                chunk_count=chunking_result.total_chunks,
                message=f"Processing completed in {processing_time:.1f}s",
                metadata={
                    "processing_time": processing_time,
                    "total_chunks": chunking_result.total_chunks,
                    "unique_chunks": chunking_result.unique_chunks,
                    "duplicate_chunks": chunking_result.duplicate_chunks,
                    "indexing_request_id": indexing_request_id if 'indexing_request_id' in locals() else None
                }
            )
            
            logger.info(
                f"Document {document_id} processed successfully: "
                f"{chunking_result.total_chunks} chunks in {processing_time:.1f}s"
            )
            
            return DocumentProcessingResult(
                success=True,
                document_id=document_id,
                status=DocumentStatus.READY,
                chunk_count=chunking_result.total_chunks,
                processing_time=processing_time,
                metadata={
                    "parsing_metrics": parsing_result.performance_metrics.__dict__ if parsing_result.performance_metrics else {},
                    "chunking_metrics": {
                        "total_chunks": chunking_result.total_chunks,
                        "unique_chunks": chunking_result.unique_chunks,
                        "duplicate_chunks": chunking_result.duplicate_chunks,
                        "processing_time": chunking_result.processing_time_seconds
                    }
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            
            logger.error(f"Document {document_id} processing failed: {error_message}")
            
            # Mark as failed
            await self.status_tracker.fail_processing(
                db_session=db_session,
                document_id=document_id,
                tenant_id=tenant_id,
                error=error_message,
                metadata={
                    "processing_time": processing_time,
                    "error_type": type(e).__name__
                }
            )
            
            return DocumentProcessingResult(
                success=False,
                document_id=document_id,
                status=DocumentStatus.FAILED,
                error=error_message,
                processing_time=processing_time
            )
    
    async def _load_document_file(self, source_uri: str) -> bytes:
        """Load document file content from storage."""
        # Extract object name from S3 URI
        if source_uri.startswith("s3://"):
            # Parse S3 URI: s3://bucket/path/to/file
            parts = source_uri[5:].split("/", 1)
            if len(parts) == 2:
                bucket, object_name = parts
                return await self.storage_service.get_file(object_name)
        
        raise ValueError(f"Unsupported source URI format: {source_uri}")


# Global instance
_document_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """Get the global document processor instance."""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor


async def process_document_async(
    document_id: UUID,
    tenant_id: UUID,
    user_id: UUID,
    db_session: Session,
    processing_options: Optional[Dict[str, Any]] = None
) -> DocumentProcessingResult:
    """
    Convenience function to process a document asynchronously.
    
    Args:
        document_id: Document ID to process
        tenant_id: Tenant ID
        user_id: User ID
        db_session: Database session
        processing_options: Optional processing configuration
        
    Returns:
        DocumentProcessingResult with processing outcome
    """
    processor = get_document_processor()
    return await processor.process_document(
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
        db_session=db_session,
        processing_options=processing_options
    )
