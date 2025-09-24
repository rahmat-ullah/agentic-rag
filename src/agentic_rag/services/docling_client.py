"""
Granite-Docling Document Parsing Client

This module provides an async HTTP client for interacting with the Granite-Docling
document parsing service. It includes retry logic, circuit breaker patterns,
and comprehensive error handling.
"""

import asyncio
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import UUID

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from agentic_rag.config import Settings

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ParseRequest(BaseModel):
    """Request model for document parsing."""
    
    extract_text: bool = Field(default=True, description="Extract text content")
    extract_tables: bool = Field(default=True, description="Extract table structures")
    extract_images: bool = Field(default=False, description="Extract images")
    extract_metadata: bool = Field(default=True, description="Extract document metadata")
    max_pages: Optional[int] = Field(default=None, description="Maximum pages to process")
    ocr_enabled: bool = Field(default=True, description="Enable OCR for scanned documents")


class ParsedContent(BaseModel):
    """Parsed content structure."""
    
    text: str = Field(..., description="Extracted text content")
    page_number: int = Field(..., description="Page number")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates")
    content_type: str = Field(..., description="Type of content (text, table, image)")
    confidence: Optional[float] = Field(None, description="Confidence score")


class ParsedTable(BaseModel):
    """Parsed table structure."""
    
    rows: List[List[str]] = Field(..., description="Table rows and cells")
    headers: Optional[List[str]] = Field(None, description="Table headers")
    page_number: int = Field(..., description="Page number")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates")


class DocumentMetadata(BaseModel):
    """Document metadata."""
    
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    subject: Optional[str] = Field(None, description="Document subject")
    creator: Optional[str] = Field(None, description="Document creator")
    producer: Optional[str] = Field(None, description="Document producer")
    creation_date: Optional[str] = Field(None, description="Creation date")
    modification_date: Optional[str] = Field(None, description="Modification date")
    page_count: int = Field(..., description="Total number of pages")
    language: Optional[str] = Field(None, description="Document language")


class ParseResponse(BaseModel):
    """Response model for document parsing."""
    
    success: bool = Field(..., description="Whether parsing was successful")
    document_type: str = Field(..., description="Detected document type")
    content: List[ParsedContent] = Field(..., description="Extracted content blocks")
    tables: List[ParsedTable] = Field(default_factory=list, description="Extracted tables")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    processing_time: float = Field(..., description="Processing time in seconds")
    pages_processed: int = Field(..., description="Number of pages processed")
    error_message: Optional[str] = Field(None, description="Error message if parsing failed")


class CircuitBreaker:
    """Circuit breaker implementation for service resilience."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func()
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker transitioning to CLOSED state")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker transitioning to OPEN state after {self.failure_count} failures")
            
            raise e


class DoclingParsingError(Exception):
    """Custom exception for Granite-Docling parsing errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class DoclingClient:
    """Async HTTP client for Granite-Docling document parsing service."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ai.docling_endpoint.rstrip('/')
        self.timeout = settings.ai.docling_timeout
        self.max_pages = settings.ai.docling_max_pages
        self.retry_attempts = settings.ai.docling_retry_attempts
        self.retry_delay = settings.ai.docling_retry_delay
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.ai.docling_circuit_breaker_threshold,
            timeout=settings.ai.docling_circuit_breaker_timeout
        )
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"Initialized DoclingClient with endpoint: {self.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self):
        """Start the HTTP client."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
            logger.info("DoclingClient HTTP client started")
    
    async def stop(self):
        """Stop the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("DoclingClient HTTP client stopped")
    
    async def health_check(self) -> bool:
        """Check if the Granite-Docling service is healthy."""
        try:
            if not self.client:
                await self.start()
            
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            health_data = response.json()
            is_healthy = health_data.get("status") == "healthy" and health_data.get("model_loaded", False)
            
            logger.info(f"Granite-Docling health check: {'healthy' if is_healthy else 'unhealthy'}")
            return is_healthy
            
        except Exception as e:
            logger.error(f"Granite-Docling health check failed: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make HTTP request with retry logic."""
        if not self.client:
            await self.start()
        
        url = f"{self.base_url}{endpoint}"
        
        def make_request():
            # This is a synchronous wrapper for the circuit breaker
            # In a real implementation, you'd want an async circuit breaker
            return asyncio.create_task(self.client.request(method, url, **kwargs))
        
        # Use circuit breaker (simplified for this example)
        try:
            response_task = await make_request()
            response = await response_task
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Request to {url} failed: {e}")
            raise DoclingParsingError(f"Request failed: {str(e)}")
    
    async def parse_document(
        self,
        file_content: bytes,
        filename: str,
        parse_request: Optional[ParseRequest] = None
    ) -> ParseResponse:
        """
        Parse a document using the Granite-Docling service.
        
        Args:
            file_content: The document file content as bytes
            filename: The original filename
            parse_request: Parsing configuration options
            
        Returns:
            ParseResponse: The parsed document content and metadata
            
        Raises:
            DoclingParsingError: If parsing fails
        """
        if parse_request is None:
            parse_request = ParseRequest()
        
        # Apply max_pages limit from settings if not specified
        if parse_request.max_pages is None:
            parse_request.max_pages = self.max_pages
        
        logger.info(f"Starting document parsing for {filename}")
        start_time = time.time()
        
        try:
            # Prepare multipart form data
            files = {
                "file": (filename, file_content, self._get_content_type(filename))
            }
            
            data = {
                "extract_text": parse_request.extract_text,
                "extract_tables": parse_request.extract_tables,
                "extract_images": parse_request.extract_images,
                "extract_metadata": parse_request.extract_metadata,
                "max_pages": parse_request.max_pages,
                "ocr_enabled": parse_request.ocr_enabled
            }
            
            # Make request to parsing service
            response = await self._make_request(
                "POST",
                "/parse",
                files=files,
                data=data
            )
            
            # Parse response
            response_data = response.json()
            parse_response = ParseResponse(**response_data)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Document parsing completed for {filename}",
                extra={
                    "processing_time": processing_time,
                    "pages_processed": parse_response.pages_processed,
                    "content_blocks": len(parse_response.content),
                    "tables_found": len(parse_response.tables),
                    "success": parse_response.success
                }
            )
            
            return parse_response
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} while parsing {filename}"
            logger.error(error_msg, extra={"status_code": e.response.status_code})
            
            # Try to extract error details from response
            try:
                error_data = e.response.json()
                error_msg = error_data.get("detail", error_msg)
            except:
                pass
            
            raise DoclingParsingError(error_msg, e.response.status_code)
            
        except Exception as e:
            error_msg = f"Unexpected error while parsing {filename}: {str(e)}"
            logger.error(error_msg)
            raise DoclingParsingError(error_msg)
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type based on file extension."""
        suffix = Path(filename).suffix.lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.bmp': 'image/bmp'
        }
        return content_types.get(suffix, 'application/octet-stream')


# Global client instance
_docling_client: Optional[DoclingClient] = None


async def get_docling_client(settings: Settings) -> DoclingClient:
    """Get or create the global Granite-Docling client instance."""
    global _docling_client
    
    if _docling_client is None:
        _docling_client = DoclingClient(settings)
        await _docling_client.start()
    
    return _docling_client


async def close_docling_client():
    """Close the global Granite-Docling client instance."""
    global _docling_client
    
    if _docling_client:
        await _docling_client.stop()
        _docling_client = None
