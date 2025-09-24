"""
IBM Granite-Docling Document Parsing Service

This service provides document parsing capabilities using the Granite-Docling model.
It exposes a REST API for parsing various document formats including PDF, DOCX, PPTX, and images.
"""

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import structlog
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
PARSE_REQUESTS = Counter('docling_parse_requests_total', 'Total parse requests', ['document_type', 'status'])
PARSE_DURATION = Histogram('docling_parse_duration_seconds', 'Parse duration in seconds', ['document_type'])
ACTIVE_REQUESTS = Counter('docling_active_requests', 'Currently active requests')

# Global variables for model and service state
docling_parser = None
service_ready = False


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


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    uptime: float = Field(..., description="Service uptime in seconds")


# Service startup time
startup_time = time.time()


async def load_docling_model():
    """Load the Granite-Docling model."""
    global docling_parser, service_ready
    
    try:
        logger.info("Loading Granite-Docling model...")
        
        # Import docling here to avoid import errors during container build
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Initialize the document converter
        docling_parser = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options,
            }
        )
        
        service_ready = True
        logger.info("Granite-Docling model loaded successfully")
        
    except Exception as e:
        logger.error("Failed to load Granite-Docling model", error=str(e))
        service_ready = False
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await load_docling_model()
    yield
    # Shutdown
    logger.info("Shutting down Granite-Docling service")


# Create FastAPI application
app = FastAPI(
    title="Granite-Docling Document Parsing Service",
    description="Document parsing service using IBM Granite-Docling-258M model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if service_ready else "loading",
        version="1.0.0",
        model_loaded=service_ready,
        uptime=time.time() - startup_time
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.post("/parse", response_model=ParseResponse)
async def parse_document(
    file: UploadFile = File(...),
    extract_text: bool = True,
    extract_tables: bool = True,
    extract_images: bool = False,
    extract_metadata: bool = True,
    max_pages: Optional[int] = None,
    ocr_enabled: bool = True
):
    """Parse a document and extract structured content."""
    if not service_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready. Model is still loading."
        )
    
    start_time = time.time()
    document_type = "unknown"
    
    try:
        # Increment active requests counter
        ACTIVE_REQUESTS.inc()
        
        # Detect document type from filename
        if file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix == '.pdf':
                document_type = 'pdf'
            elif suffix in ['.docx', '.doc']:
                document_type = 'docx'
            elif suffix in ['.pptx', '.ppt']:
                document_type = 'pptx'
            elif suffix in ['.xlsx', '.xls']:
                document_type = 'xlsx'
            elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                document_type = 'image'
        
        logger.info("Starting document parsing", 
                   filename=file.filename, 
                   document_type=document_type,
                   file_size=file.size)
        
        # Read file content
        file_content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Parse document with Granite-Docling
            result = docling_parser.convert(temp_file_path)
            
            # Extract content blocks
            content_blocks = []
            tables = []
            
            # Process the parsed document
            for page_num, page in enumerate(result.document.pages, 1):
                if max_pages and page_num > max_pages:
                    break
                
                # Extract text content
                if extract_text and hasattr(page, 'text'):
                    content_blocks.append(ParsedContent(
                        text=page.text,
                        page_number=page_num,
                        content_type="text",
                        confidence=1.0
                    ))
                
                # Extract tables
                if extract_tables and hasattr(page, 'tables'):
                    for table in page.tables:
                        table_data = []
                        headers = None
                        
                        if hasattr(table, 'data') and table.data:
                            table_data = table.data
                            if table_data and len(table_data) > 0:
                                headers = table_data[0]
                                table_data = table_data[1:]
                        
                        tables.append(ParsedTable(
                            rows=table_data,
                            headers=headers,
                            page_number=page_num
                        ))
            
            # Extract metadata
            metadata = DocumentMetadata(
                page_count=len(result.document.pages),
                title=getattr(result.document, 'title', None),
                language=getattr(result.document, 'language', None)
            )
            
            processing_time = time.time() - start_time
            
            # Record metrics
            PARSE_REQUESTS.labels(document_type=document_type, status='success').inc()
            PARSE_DURATION.labels(document_type=document_type).observe(processing_time)
            
            logger.info("Document parsing completed successfully",
                       filename=file.filename,
                       processing_time=processing_time,
                       pages_processed=len(result.document.pages),
                       content_blocks=len(content_blocks),
                       tables_found=len(tables))
            
            return ParseResponse(
                success=True,
                document_type=document_type,
                content=content_blocks,
                tables=tables,
                metadata=metadata,
                processing_time=processing_time,
                pages_processed=len(result.document.pages)
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
    
    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        
        # Record error metrics
        PARSE_REQUESTS.labels(document_type=document_type, status='error').inc()
        
        logger.error("Document parsing failed",
                    filename=file.filename,
                    error=error_message,
                    processing_time=processing_time)
        
        return ParseResponse(
            success=False,
            document_type=document_type,
            content=[],
            tables=[],
            metadata=DocumentMetadata(page_count=0),
            processing_time=processing_time,
            pages_processed=0,
            error_message=error_message
        )
    
    finally:
        # Decrement active requests counter
        ACTIVE_REQUESTS.dec()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Start the service
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_config=None,  # Use structlog configuration
        access_log=False  # Disable default access log
    )
