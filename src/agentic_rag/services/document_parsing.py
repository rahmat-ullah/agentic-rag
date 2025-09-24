"""
Integrated Document Parsing Service

This module provides the main document parsing service that integrates
Granite-Docling parsing with content extraction, metadata enrichment,
error handling, and performance optimization.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
from uuid import UUID, uuid4

import redis
from sqlalchemy.orm import Session

from agentic_rag.config import Settings
from agentic_rag.models.database import Document, DocumentKind
from agentic_rag.services.content_extraction import ExtractedContent, get_content_extraction_pipeline
from agentic_rag.services.docling_client import (
    DoclingClient, DoclingParsingError, ParseRequest, ParseResponse, DocumentMetadata, get_docling_client
)
from agentic_rag.services.metadata_extraction import EnrichedMetadata, get_metadata_extractor
from agentic_rag.services.parsing_fallbacks import (
    FallbackResult, ParseQuality, get_parsing_fallback_service
)
from agentic_rag.services.parsing_optimization import (
    PerformanceMetrics, get_performance_optimizer
)
from agentic_rag.services.document_status_tracker import get_document_status_tracker

logger = logging.getLogger(__name__)


class DocumentParsingResult:
    """Complete result of document parsing operation."""
    
    def __init__(
        self,
        success: bool,
        document_id: str,
        extracted_content: Optional[ExtractedContent] = None,
        enriched_metadata: Optional[EnrichedMetadata] = None,
        parse_response: Optional[ParseResponse] = None,
        fallback_result: Optional[FallbackResult] = None,
        performance_metrics: Optional[PerformanceMetrics] = None,
        error_message: Optional[str] = None,
        warnings: Optional[list] = None
    ):
        self.success = success
        self.document_id = document_id
        self.extracted_content = extracted_content
        self.enriched_metadata = enriched_metadata
        self.parse_response = parse_response
        self.fallback_result = fallback_result
        self.performance_metrics = performance_metrics
        self.error_message = error_message
        self.warnings = warnings or []
    
    @property
    def quality(self) -> ParseQuality:
        """Get the overall parsing quality."""
        if not self.success:
            return ParseQuality.FAILED
        
        if self.fallback_result:
            return self.fallback_result.quality
        
        # Assess quality based on content completeness
        if self.extracted_content:
            if len(self.extracted_content.text_content) > 1000 and self.extracted_content.tables:
                return ParseQuality.EXCELLENT
            elif len(self.extracted_content.text_content) > 500:
                return ParseQuality.GOOD
            elif len(self.extracted_content.text_content) > 100:
                return ParseQuality.FAIR
            else:
                return ParseQuality.POOR
        
        return ParseQuality.POOR
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "document_id": self.document_id,
            "quality": self.quality.value,
            "has_content": self.extracted_content is not None,
            "has_metadata": self.enriched_metadata is not None,
            "used_fallback": self.fallback_result is not None,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "performance": {
                "total_time": self.performance_metrics.total_time if self.performance_metrics else 0,
                "cache_hit": self.performance_metrics.cache_hit if self.performance_metrics else False
            } if self.performance_metrics else None
        }


class DocumentParsingService:
    """Main service for parsing documents with full integration."""
    
    def __init__(self, settings: Settings, redis_client: redis.Redis):
        self.settings = settings
        self.redis_client = redis_client
        
        # Initialize components
        self.content_pipeline = get_content_extraction_pipeline()
        self.metadata_extractor = get_metadata_extractor()
        self.fallback_service = get_parsing_fallback_service()
        
        # These will be initialized async
        self.docling_client: Optional[DoclingClient] = None
        self.performance_optimizer = None
        
        logger.info("Document parsing service initialized")
    
    async def initialize(self):
        """Initialize async components."""
        if not self.docling_client:
            self.docling_client = await get_docling_client(self.settings)
        
        if not self.performance_optimizer:
            self.performance_optimizer = await get_performance_optimizer(self.settings, self.redis_client)
        
        logger.info("Document parsing service async initialization completed")
    
    async def parse_document(
        self,
        file_content: bytes,
        filename: str,
        tenant_id: UUID,
        user_id: UUID,
        parse_options: Optional[Dict] = None
    ) -> DocumentParsingResult:
        """
        Parse a document with full integration of all services.
        
        Args:
            file_content: The document file content
            filename: Original filename
            tenant_id: Tenant identifier
            user_id: User identifier
            parse_options: Optional parsing configuration
            
        Returns:
            DocumentParsingResult: Complete parsing result
        """
        document_id = str(uuid4())
        start_time = time.time()
        
        logger.info(f"Starting document parsing for {filename} (ID: {document_id})")
        
        # Ensure async components are initialized
        await self.initialize()
        
        # Create parse request
        parse_request = self._create_parse_request(parse_options)
        
        try:
            # Performance optimization and caching
            cache_entry, perf_metrics = await self.performance_optimizer.optimize_parsing(
                file_content, filename, parse_request
            )
            
            # Check if we have a cache hit
            if cache_entry:
                logger.info(f"Cache hit for document {filename}")
                return DocumentParsingResult(
                    success=True,
                    document_id=document_id,
                    extracted_content=cache_entry.extracted_content,
                    enriched_metadata=cache_entry.enriched_metadata,
                    parse_response=cache_entry.parse_response,
                    performance_metrics=perf_metrics
                )
            
            # Primary parsing with Granite-Docling
            parse_response = None
            extracted_content = None
            enriched_metadata = None
            fallback_result = None
            warnings = []
            
            try:
                # Health check first
                if not await self.docling_client.health_check():
                    raise DoclingParsingError("Granite-Docling service is not healthy")
                
                # Parse with Granite-Docling
                parse_response = await self.docling_client.parse_document(
                    file_content, filename, parse_request
                )
                
                if not parse_response.success:
                    raise DoclingParsingError(parse_response.error_message or "Parsing failed")
                
                # Extract content
                extracted_content = self.content_pipeline.extract_content(parse_response, document_id)
                
                # Extract metadata
                enriched_metadata = self.metadata_extractor.extract_metadata(
                    extracted_content, parse_response, filename
                )
                
                logger.info(f"Primary parsing successful for {filename}")
                
            except Exception as primary_error:
                logger.warning(f"Primary parsing failed for {filename}: {str(primary_error)}")
                
                # Attempt fallback parsing
                fallback_result = self.fallback_service.handle_parsing_failure(
                    file_content, filename, primary_error
                )
                
                if fallback_result.success:
                    extracted_content = fallback_result.content
                    
                    # Create minimal parse response for fallback
                    parse_response = ParseResponse(
                        success=True,
                        document_type=fallback_result.method_used.value,
                        content=[],
                        tables=[],
                        metadata=DocumentMetadata(page_count=1),
                        processing_time=fallback_result.processing_time,
                        pages_processed=1
                    )
                    
                    # Extract metadata from fallback content
                    enriched_metadata = self.metadata_extractor.extract_metadata(
                        extracted_content, parse_response, filename
                    )
                    
                    warnings.extend(fallback_result.warnings)
                    logger.info(f"Fallback parsing successful for {filename} using {fallback_result.method_used.value}")
                else:
                    # Complete failure
                    error_message = f"Both primary and fallback parsing failed: {str(primary_error)}"
                    logger.error(error_message)
                    
                    return DocumentParsingResult(
                        success=False,
                        document_id=document_id,
                        error_message=error_message,
                        performance_metrics=perf_metrics
                    )
            
            # Cache successful results
            if extracted_content and enriched_metadata and parse_response:
                await self.performance_optimizer.cache_parsing_result(
                    file_content, filename, parse_request, parse_response,
                    extracted_content, enriched_metadata
                )
            
            # Update performance metrics
            total_time = time.time() - start_time
            perf_metrics.total_time = total_time
            
            logger.info(
                f"Document parsing completed for {filename}",
                extra={
                    "document_id": document_id,
                    "total_time": total_time,
                    "quality": fallback_result.quality.value if fallback_result else "excellent",
                    "used_fallback": fallback_result is not None
                }
            )
            
            return DocumentParsingResult(
                success=True,
                document_id=document_id,
                extracted_content=extracted_content,
                enriched_metadata=enriched_metadata,
                parse_response=parse_response,
                fallback_result=fallback_result,
                performance_metrics=perf_metrics,
                warnings=warnings
            )
            
        except Exception as e:
            error_message = f"Unexpected error during document parsing: {str(e)}"
            logger.error(error_message, exc_info=True)
            
            return DocumentParsingResult(
                success=False,
                document_id=document_id,
                error_message=error_message,
                performance_metrics=PerformanceMetrics(
                    total_time=time.time() - start_time,
                    parsing_time=0,
                    extraction_time=0,
                    metadata_time=0,
                    cache_hit=False,
                    memory_usage_mb=0,
                    pages_per_second=0,
                    throughput_mb_per_second=0
                )
            )
    
    async def store_parsing_result(
        self,
        parsing_result: DocumentParsingResult,
        tenant_id: UUID,
        user_id: UUID,
        db_session: Session,
        source_uri: Optional[str] = None
    ) -> Optional[Document]:
        """
        Store parsing result in the database.
        
        Args:
            parsing_result: The parsing result to store
            tenant_id: Tenant identifier
            user_id: User identifier
            db_session: Database session
            source_uri: Optional source URI for the document
            
        Returns:
            Document: Created document record or None if failed
        """
        if not parsing_result.success or not parsing_result.enriched_metadata:
            logger.warning(f"Cannot store unsuccessful parsing result for document {parsing_result.document_id}")
            return None
        
        try:
            metadata = parsing_result.enriched_metadata
            
            # Determine document kind
            document_kind = self._determine_document_kind(metadata.classification.format)
            
            # Create document record
            document = Document(
                id=UUID(parsing_result.document_id),
                tenant_id=tenant_id,
                title=metadata.properties.title or "Untitled Document",
                kind=document_kind,
                source_uri=source_uri,
                sha256="",  # Would be calculated from file content
                version=1,
                pages=metadata.structural.page_count,
                created_by=user_id
            )
            
            db_session.add(document)
            db_session.commit()
            
            logger.info(f"Stored parsing result for document {parsing_result.document_id}")
            return document
            
        except Exception as e:
            db_session.rollback()
            logger.error(f"Failed to store parsing result: {str(e)}")
            return None
    
    def _create_parse_request(self, parse_options: Optional[Dict] = None) -> ParseRequest:
        """Create parse request from options."""
        if parse_options is None:
            parse_options = {}
        
        return ParseRequest(
            extract_text=parse_options.get("extract_text", True),
            extract_tables=parse_options.get("extract_tables", True),
            extract_images=parse_options.get("extract_images", False),
            extract_metadata=parse_options.get("extract_metadata", True),
            max_pages=parse_options.get("max_pages", self.settings.ai.docling_max_pages),
            ocr_enabled=parse_options.get("ocr_enabled", True)
        )
    
    def _determine_document_kind(self, file_format: str) -> DocumentKind:
        """Determine document kind from file format."""
        # For now, map all formats to RFQ as the default
        # In a real implementation, this would be determined by content analysis
        # or user input during upload
        return DocumentKind.RFQ
    
    async def get_service_health(self) -> Dict:
        """Get health status of all parsing services."""
        await self.initialize()
        
        health_status = {
            "docling_service": await self.docling_client.health_check(),
            "cache_status": "unknown",
            "performance_stats": {}
        }
        
        # Get cache stats
        try:
            cache_stats = await self.performance_optimizer.cache.get_stats()
            health_status["cache_status"] = "healthy"
            health_status["cache_stats"] = cache_stats
        except Exception as e:
            health_status["cache_status"] = f"error: {str(e)}"
        
        # Get performance stats
        try:
            perf_stats = self.performance_optimizer.get_performance_stats()
            health_status["performance_stats"] = perf_stats
        except Exception as e:
            health_status["performance_stats"] = {"error": str(e)}
        
        return health_status
    
    async def cleanup(self):
        """Cleanup service resources."""
        if self.docling_client:
            await self.docling_client.stop()
        
        if self.performance_optimizer:
            await self.performance_optimizer.cleanup()
        
        logger.info("Document parsing service cleanup completed")


# Global service instance
_document_parsing_service: Optional[DocumentParsingService] = None


async def get_document_parsing_service(settings: Settings, redis_client: redis.Redis) -> DocumentParsingService:
    """Get or create the global document parsing service instance."""
    global _document_parsing_service
    
    if _document_parsing_service is None:
        _document_parsing_service = DocumentParsingService(settings, redis_client)
        await _document_parsing_service.initialize()
    
    return _document_parsing_service


async def cleanup_document_parsing_service():
    """Cleanup the global document parsing service instance."""
    global _document_parsing_service
    
    if _document_parsing_service:
        await _document_parsing_service.cleanup()
        _document_parsing_service = None
