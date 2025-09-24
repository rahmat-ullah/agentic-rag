"""
Integration tests for Granite-Docling document parsing.

These tests validate the complete integration of all Granite-Docling
components including parsing, content extraction, metadata enrichment,
error handling, and performance optimization.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agentic_rag.config import get_settings
from agentic_rag.services.document_parsing import DocumentParsingService, DocumentParsingResult
from agentic_rag.services.docling_client import ParseRequest, ParseResponse, DocumentMetadata, ParsedContent
from agentic_rag.services.content_extraction import ExtractedContent, LayoutElement, ContentType, DocumentStructure
from agentic_rag.services.metadata_extraction import (
    EnrichedMetadata, DocumentProperties, StructuralMetadata, ContentMetrics,
    QualityMetrics, DocumentClassification
)
from agentic_rag.services.parsing_fallbacks import FallbackResult, ParseQuality, FallbackMethod


class TestGraniteDoclingIntegration:
    """Test suite for Granite-Docling integration."""
    
    @pytest.fixture
    def settings(self):
        """Get test settings."""
        return get_settings()
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.setex = AsyncMock()
        redis_mock.delete = AsyncMock()
        redis_mock.keys = AsyncMock(return_value=[])
        return redis_mock
    
    @pytest.fixture
    async def parsing_service(self, settings, mock_redis):
        """Create document parsing service for testing."""
        service = DocumentParsingService(settings, mock_redis)
        yield service
        await service.cleanup()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing."""
        # This would be actual PDF bytes in a real test
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    
    @pytest.fixture
    def mock_parse_response(self):
        """Mock successful parse response."""
        return ParseResponse(
            success=True,
            document_type="pdf",
            content=[
                ParsedContent(
                    text="Sample document content with important information.",
                    page_number=1,
                    content_type="text",
                    confidence=0.95
                ),
                ParsedContent(
                    text="Chapter 1: Introduction",
                    page_number=1,
                    content_type="heading",
                    confidence=0.98
                )
            ],
            tables=[],
            metadata=DocumentMetadata(
                title="Sample Document",
                author="Test Author",
                page_count=1,
                language="en"
            ),
            processing_time=1.5,
            pages_processed=1
        )
    
    @pytest.mark.asyncio
    async def test_successful_document_parsing(self, parsing_service, sample_pdf_content, mock_parse_response):
        """Test successful document parsing with all components."""
        tenant_id = uuid4()
        user_id = uuid4()
        filename = "test_document.pdf"
        
        # Mock the Granite-Docling client
        with patch.object(parsing_service, 'docling_client') as mock_client:
            mock_client.health_check = AsyncMock(return_value=True)
            mock_client.parse_document = AsyncMock(return_value=mock_parse_response)
            
            # Mock performance optimizer to skip caching
            with patch.object(parsing_service, 'performance_optimizer') as mock_optimizer:
                mock_optimizer.optimize_parsing = AsyncMock(return_value=(None, MagicMock()))
                mock_optimizer.cache_parsing_result = AsyncMock()
                
                # Parse document
                result = await parsing_service.parse_document(
                    sample_pdf_content,
                    filename,
                    tenant_id,
                    user_id
                )
                
                # Validate result
                assert result.success is True
                assert result.document_id is not None
                assert result.extracted_content is not None
                assert result.enriched_metadata is not None
                assert result.parse_response is not None
                assert result.fallback_result is None
                assert result.error_message is None
                
                # Validate extracted content
                content = result.extracted_content
                assert len(content.elements) > 0
                assert content.text_content.strip() != ""
                assert content.structure.page_count == 1
                
                # Validate metadata
                metadata = result.enriched_metadata
                assert metadata.properties.title == "Sample Document"
                assert metadata.properties.author == "Test Author"
                assert metadata.structural.page_count == 1
                
                # Verify service calls
                mock_client.health_check.assert_called_once()
                mock_client.parse_document.assert_called_once()
                mock_optimizer.cache_parsing_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_parsing_on_primary_failure(self, parsing_service, sample_pdf_content):
        """Test fallback parsing when primary parsing fails."""
        tenant_id = uuid4()
        user_id = uuid4()
        filename = "test_document.pdf"
        
        # Mock the Granite-Docling client to fail
        with patch.object(parsing_service, 'docling_client') as mock_client:
            mock_client.health_check = AsyncMock(return_value=True)
            mock_client.parse_document = AsyncMock(side_effect=Exception("Parsing failed"))
            
            # Mock performance optimizer
            with patch.object(parsing_service, 'performance_optimizer') as mock_optimizer:
                mock_optimizer.optimize_parsing = AsyncMock(return_value=(None, MagicMock()))
                mock_optimizer.cache_parsing_result = AsyncMock()
                
                # Mock fallback service to succeed
                with patch.object(parsing_service, 'fallback_service') as mock_fallback:
                    # Create mock fallback content
                    fallback_content = ExtractedContent(
                        document_id=str(uuid4()),
                        structure=DocumentStructure(page_count=1, document_type="pdf"),
                        elements=[
                            LayoutElement(
                                id="fallback_element",
                                type=ContentType.TEXT,
                                content="Fallback extracted text",
                                page_number=1,
                                confidence=0.7
                            )
                        ],
                        tables=[],
                        text_content="Fallback extracted text",
                        processing_metadata={"extraction_method": "fallback"}
                    )
                    
                    mock_fallback_result = FallbackResult(
                        success=True,
                        method_used=FallbackMethod.PYPDF2,
                        quality=ParseQuality.FAIR,
                        content=fallback_content,
                        processing_time=0.5,
                        warnings=["Limited formatting preservation"]
                    )
                    
                    mock_fallback.handle_parsing_failure = MagicMock(return_value=mock_fallback_result)
                    
                    # Parse document
                    result = await parsing_service.parse_document(
                        sample_pdf_content,
                        filename,
                        tenant_id,
                        user_id
                    )
                    
                    # Validate result
                    assert result.success is True
                    assert result.fallback_result is not None
                    assert result.fallback_result.method_used == FallbackMethod.PYPDF2
                    assert result.quality == ParseQuality.FAIR
                    assert len(result.warnings) > 0
                    
                    # Verify fallback was called
                    mock_fallback.handle_parsing_failure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self, parsing_service, sample_pdf_content, mock_parse_response):
        """Test cache hit scenario."""
        tenant_id = uuid4()
        user_id = uuid4()
        filename = "test_document.pdf"
        
        # Create mock cache entry
        from agentic_rag.services.parsing_optimization import CacheEntry
        
        mock_extracted_content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=[],
            tables=[],
            text_content="Cached content",
            processing_metadata={}
        )
        
        mock_metadata = EnrichedMetadata(
            document_id=str(uuid4()),
            properties=DocumentProperties(title="Cached Document"),
            structural=StructuralMetadata(page_count=1),
            content_metrics=ContentMetrics(
                total_characters=100,
                total_words=20,
                total_sentences=5,
                average_words_per_sentence=4.0,
                reading_time_minutes=0.1,
                complexity_score=0.5,
                unique_words=15,
                vocabulary_richness=0.75
            ),
            quality_metrics=QualityMetrics(
                extraction_confidence=0.9,
                text_quality_score=0.8,
                table_quality_score=0.7
            ),
            classification=DocumentClassification(
                document_type="pdf",
                format="pdf",
                confidence=0.9
            )
        )
        
        cache_entry = CacheEntry(
            content_hash="test_hash",
            filename=filename,
            parse_response=mock_parse_response,
            extracted_content=mock_extracted_content,
            enriched_metadata=mock_metadata,
            cache_timestamp=1234567890,
            last_access=1234567890
        )
        
        # Mock performance optimizer to return cache hit
        with patch.object(parsing_service, 'performance_optimizer') as mock_optimizer:
            mock_optimizer.optimize_parsing = AsyncMock(return_value=(cache_entry, MagicMock(cache_hit=True)))
            
            # Parse document
            result = await parsing_service.parse_document(
                sample_pdf_content,
                filename,
                tenant_id,
                user_id
            )
            
            # Validate cache hit
            assert result.success is True
            assert result.performance_metrics.cache_hit is True
            assert result.extracted_content.text_content == "Cached content"
            assert result.enriched_metadata.properties.title == "Cached Document"
    
    @pytest.mark.asyncio
    async def test_complete_parsing_failure(self, parsing_service, sample_pdf_content):
        """Test complete parsing failure (both primary and fallback fail)."""
        tenant_id = uuid4()
        user_id = uuid4()
        filename = "test_document.pdf"
        
        # Mock all services to fail
        with patch.object(parsing_service, 'docling_client') as mock_client:
            mock_client.health_check = AsyncMock(return_value=False)
            
            with patch.object(parsing_service, 'performance_optimizer') as mock_optimizer:
                mock_optimizer.optimize_parsing = AsyncMock(return_value=(None, MagicMock()))
                
                with patch.object(parsing_service, 'fallback_service') as mock_fallback:
                    mock_fallback_result = FallbackResult(
                        success=False,
                        method_used=FallbackMethod.BASIC_TEXT,
                        quality=ParseQuality.FAILED,
                        error_message="All fallback methods failed",
                        processing_time=0.1
                    )
                    
                    mock_fallback.handle_parsing_failure = MagicMock(return_value=mock_fallback_result)
                    
                    # Parse document
                    result = await parsing_service.parse_document(
                        sample_pdf_content,
                        filename,
                        tenant_id,
                        user_id
                    )
                    
                    # Validate failure
                    assert result.success is False
                    assert result.error_message is not None
                    assert result.extracted_content is None
                    assert result.enriched_metadata is None
    
    @pytest.mark.asyncio
    async def test_service_health_check(self, parsing_service):
        """Test service health check functionality."""
        # Mock all components
        with patch.object(parsing_service, 'docling_client') as mock_client:
            mock_client.health_check = AsyncMock(return_value=True)
            
            with patch.object(parsing_service, 'performance_optimizer') as mock_optimizer:
                mock_optimizer.cache.get_stats = AsyncMock(return_value={"total_entries": 10})
                mock_optimizer.get_performance_stats = MagicMock(return_value={"avg_time": 1.5})
                
                # Get health status
                health = await parsing_service.get_service_health()
                
                # Validate health response
                assert health["docling_service"] is True
                assert health["cache_status"] == "healthy"
                assert "cache_stats" in health
                assert "performance_stats" in health
    
    @pytest.mark.asyncio
    async def test_parse_request_creation(self, parsing_service):
        """Test parse request creation with various options."""
        # Test default options
        request = parsing_service._create_parse_request()
        assert request.extract_text is True
        assert request.extract_tables is True
        assert request.extract_images is False
        assert request.extract_metadata is True
        assert request.ocr_enabled is True
        
        # Test custom options
        custom_options = {
            "extract_text": False,
            "extract_images": True,
            "max_pages": 50,
            "ocr_enabled": False
        }
        
        request = parsing_service._create_parse_request(custom_options)
        assert request.extract_text is False
        assert request.extract_images is True
        assert request.max_pages == 50
        assert request.ocr_enabled is False
    
    @pytest.mark.asyncio
    async def test_document_kind_determination(self, parsing_service):
        """Test document kind determination from file format."""
        from agentic_rag.models.database import DocumentKind

        # Test various formats - all return RFQ for now
        assert parsing_service._determine_document_kind("pdf") == DocumentKind.RFQ
        assert parsing_service._determine_document_kind("docx") == DocumentKind.RFQ
        assert parsing_service._determine_document_kind("pptx") == DocumentKind.RFQ
        assert parsing_service._determine_document_kind("xlsx") == DocumentKind.RFQ
        assert parsing_service._determine_document_kind("png") == DocumentKind.RFQ
        assert parsing_service._determine_document_kind("unknown") == DocumentKind.RFQ  # Default
    
    def test_parsing_result_quality_assessment(self):
        """Test parsing result quality assessment."""
        from agentic_rag.services.content_extraction import ExtractedTable

        # Test excellent quality
        content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=[],
            tables=[ExtractedTable(
                id="test_table",
                headers=["Col1", "Col2"],
                rows=[["A", "B"], ["C", "D"]],
                row_count=2,
                column_count=2,
                page_number=1,
                confidence=0.9
            )],  # Has tables
            text_content="A" * 1500,  # Long text
            processing_metadata={}
        )
        
        result = DocumentParsingResult(
            success=True,
            document_id=str(uuid4()),
            extracted_content=content
        )
        
        assert result.quality == ParseQuality.EXCELLENT
        
        # Test poor quality
        poor_content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=[],
            tables=[],
            text_content="Short",  # Very short text
            processing_metadata={}
        )
        
        poor_result = DocumentParsingResult(
            success=True,
            document_id=str(uuid4()),
            extracted_content=poor_content
        )
        
        assert poor_result.quality == ParseQuality.POOR
    
    def test_parsing_result_serialization(self):
        """Test parsing result serialization to dictionary."""
        result = DocumentParsingResult(
            success=True,
            document_id="test-id",
            error_message=None,
            warnings=["Test warning"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["document_id"] == "test-id"
        assert result_dict["warnings"] == ["Test warning"]
        assert "quality" in result_dict
        assert "performance" in result_dict
