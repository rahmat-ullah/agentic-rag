"""
Unit tests for chunking pipeline integration.
"""

import asyncio
import pytest
from uuid import uuid4

from agentic_rag.services.chunking.pipeline import (
    ChunkingPipeline,
    ChunkingPipelineConfig,
    ChunkingResult,
    get_chunking_pipeline,
    process_document_chunks_async,
    process_document_chunks_sync
)
from agentic_rag.services.content_extraction import (
    ExtractedContent,
    LayoutElement,
    ContentType,
    DocumentStructure
)


class TestChunkingPipelineConfig:
    """Test suite for chunking pipeline configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingPipelineConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100
        assert config.language == "en"
        assert config.enable_deduplication is True
        assert config.similarity_threshold == 0.85
        assert config.enable_async is True
        assert config.max_concurrent_chunks == 10
        assert config.batch_size == 50
        assert config.min_quality_score == 0.5
        assert config.filter_low_quality is True
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 3600
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChunkingPipelineConfig(
            chunk_size=500,
            chunk_overlap=100,
            enable_deduplication=False,
            similarity_threshold=0.9,
            enable_async=False
        )
        
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.enable_deduplication is False
        assert config.similarity_threshold == 0.9
        assert config.enable_async is False


class TestChunkingPipeline:
    """Test suite for chunking pipeline."""
    
    @pytest.fixture
    def sample_content(self):
        """Create sample extracted content for testing."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.PARAGRAPH,
                content="This is the first paragraph with some content that should be chunked properly.",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_2",
                type=ContentType.PARAGRAPH,
                content="This is the second paragraph with different content that provides more text for chunking.",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_3",
                type=ContentType.TABLE,
                content="Name | Age\nJohn | 25\nJane | 30",
                page_number=1,
                confidence=0.9
            )
        ]
        
        return ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Sample content for chunking pipeline testing",
            processing_metadata={}
        )
    
    @pytest.fixture
    def pipeline_config(self):
        """Create a test pipeline configuration."""
        return ChunkingPipelineConfig(
            chunk_size=200,
            chunk_overlap=50,
            enable_deduplication=True,
            similarity_threshold=0.8,
            enable_async=True,
            enable_caching=False  # Disable caching for tests
        )
    
    @pytest.fixture
    def pipeline(self, pipeline_config):
        """Create a chunking pipeline for testing."""
        return ChunkingPipeline(pipeline_config)
    
    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initialization."""
        pipeline = ChunkingPipeline(pipeline_config)
        
        assert pipeline.config == pipeline_config
        assert hasattr(pipeline.chunker, 'chunk_document')
        assert pipeline._cache == {}
    
    def test_pipeline_initialization_default_config(self):
        """Test pipeline initialization with default config."""
        pipeline = ChunkingPipeline()
        
        assert isinstance(pipeline.config, ChunkingPipelineConfig)
        assert pipeline.config.chunk_size == 1000
        assert pipeline.config.enable_deduplication is True
    
    def test_pipeline_initialization_without_deduplication(self):
        """Test pipeline initialization without deduplication."""
        config = ChunkingPipelineConfig(enable_deduplication=False)
        pipeline = ChunkingPipeline(config)
        
        assert hasattr(pipeline.chunker, 'chunk_document')
        # Should use table-aware chunker as fallback
    
    @pytest.mark.asyncio
    async def test_async_document_processing(self, pipeline, sample_content):
        """Test asynchronous document processing."""
        result = await pipeline.process_document_async(sample_content)
        
        assert isinstance(result, ChunkingResult)
        assert result.document_id == sample_content.document_id
        assert result.total_chunks > 0
        assert result.processing_time_seconds > 0
        assert len(result.chunks) == result.total_chunks
        assert result.processing_metadata is not None
    
    def test_sync_document_processing(self, pipeline, sample_content):
        """Test synchronous document processing."""
        result = pipeline.process_document_sync(sample_content)
        
        assert isinstance(result, ChunkingResult)
        assert result.document_id == sample_content.document_id
        assert result.total_chunks > 0
        assert result.processing_time_seconds > 0
        assert len(result.chunks) == result.total_chunks
    
    def test_quality_filtering(self, sample_content):
        """Test quality filtering functionality."""
        config = ChunkingPipelineConfig(
            filter_low_quality=True,
            min_quality_score=0.9,  # High threshold
            enable_caching=False
        )
        pipeline = ChunkingPipeline(config)
        
        result = pipeline.process_document_sync(sample_content)
        
        # Should filter some chunks due to high quality threshold
        assert result.quality_filtered_chunks >= 0
        assert all(chunk.metadata.quality_score >= 0.9 for chunk in result.chunks)
    
    def test_quality_filtering_disabled(self, sample_content):
        """Test with quality filtering disabled."""
        config = ChunkingPipelineConfig(
            filter_low_quality=False,
            enable_caching=False
        )
        pipeline = ChunkingPipeline(config)
        
        result = pipeline.process_document_sync(sample_content)
        
        assert result.quality_filtered_chunks == 0
    
    def test_caching_functionality(self, sample_content):
        """Test caching functionality."""
        config = ChunkingPipelineConfig(enable_caching=True)
        pipeline = ChunkingPipeline(config)
        
        # First processing
        result1 = pipeline.process_document_sync(sample_content)
        
        # Second processing (should use cache)
        result2 = pipeline.process_document_sync(sample_content)
        
        assert result1.document_id == result2.document_id
        assert result1.total_chunks == result2.total_chunks
        
        # Clear cache
        pipeline.clear_cache()
        assert len(pipeline._cache) == 0
    
    def test_pipeline_stats(self, pipeline):
        """Test pipeline statistics."""
        stats = pipeline.get_pipeline_stats()
        
        assert "config" in stats
        assert "cache_stats" in stats
        assert "chunker_type" in stats
        
        assert stats["config"]["chunk_size"] == 200
        assert stats["config"]["deduplication_enabled"] is True
        assert stats["cache_stats"]["cached_documents"] == 0
    
    def test_convert_to_deduplicated_chunks(self, pipeline):
        """Test conversion of regular chunks to deduplicated format."""
        # This tests the fallback functionality when deduplication is disabled
        from agentic_rag.services.chunking.basic_chunker import TextChunk, ChunkMetadata
        
        # Create mock regular chunks
        regular_chunks = [
            TextChunk(
                content="Test chunk content",
                metadata=ChunkMetadata(
                    chunk_id=str(uuid4()),
                    document_id=str(uuid4()),
                    chunk_index=0,
                    start_char=0,
                    end_char=18,
                    chunk_size=18,
                    word_count=3,
                    sentence_count=1,
                    quality_score=0.8
                )
            )
        ]
        
        deduplicated_chunks = pipeline._convert_to_deduplicated_chunks(regular_chunks)
        
        assert len(deduplicated_chunks) == 1
        assert hasattr(deduplicated_chunks[0], 'deduplication_metadata')
        assert deduplicated_chunks[0].deduplication_metadata.is_duplicate is False
        assert deduplicated_chunks[0].content == "Test chunk content"


class TestChunkingResult:
    """Test suite for chunking result model."""
    
    def test_chunking_result_creation(self):
        """Test chunking result creation."""
        from agentic_rag.services.chunking.deduplication_chunker import DeduplicatedChunk, DeduplicationMetadata
        from agentic_rag.services.chunking.basic_chunker import ChunkMetadata
        from agentic_rag.services.chunking.section_aware_chunker import SectionMetadata
        from agentic_rag.services.chunking.table_aware_chunker import TableMetadata
        
        # Create sample chunks
        chunks = [
            DeduplicatedChunk(
                content="Test chunk",
                metadata=ChunkMetadata(
                    chunk_id=str(uuid4()),
                    document_id=str(uuid4()),
                    chunk_index=0,
                    start_char=0,
                    end_char=10,
                    chunk_size=10,
                    word_count=2,
                    sentence_count=1,
                    quality_score=0.8
                ),
                section_metadata=SectionMetadata(
                    section_id=str(uuid4()),
                    section_title="Test Section",
                    section_level=1
                ),
                table_metadata=TableMetadata(),
                deduplication_metadata=DeduplicationMetadata(
                    content_hash="test_hash",
                    normalized_content="test chunk",
                    is_duplicate=False
                )
            )
        ]
        
        result = ChunkingResult(
            document_id=str(uuid4()),
            total_chunks=1,
            unique_chunks=1,
            duplicate_chunks=0,
            processing_time_seconds=1.5,
            chunks=chunks
        )
        
        assert result.total_chunks == 1
        assert result.unique_chunks == 1
        assert result.duplicate_chunks == 0
        assert result.processing_time_seconds == 1.5
        assert len(result.chunks) == 1


class TestGlobalPipelineFunctions:
    """Test the global pipeline functions."""
    
    def test_get_chunking_pipeline_default(self):
        """Test getting chunking pipeline with default config."""
        pipeline = get_chunking_pipeline()
        assert isinstance(pipeline, ChunkingPipeline)
        assert pipeline.config.chunk_size == 1000
    
    def test_get_chunking_pipeline_custom_config(self):
        """Test getting chunking pipeline with custom config."""
        config = ChunkingPipelineConfig(chunk_size=500)
        pipeline = get_chunking_pipeline(config)
        assert isinstance(pipeline, ChunkingPipeline)
        assert pipeline.config.chunk_size == 500
    
    def test_pipeline_singleton_behavior(self):
        """Test that the same instance is returned when no config is provided."""
        pipeline1 = get_chunking_pipeline()
        pipeline2 = get_chunking_pipeline()
        assert pipeline1 is pipeline2
    
    @pytest.mark.asyncio
    async def test_process_document_chunks_async_convenience(self):
        """Test the async convenience function."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.PARAGRAPH,
                content="Test content for async processing",
                page_number=1,
                confidence=0.9
            )
        ]
        
        content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Test content",
            processing_metadata={}
        )
        
        result = await process_document_chunks_async(content)
        assert isinstance(result, ChunkingResult)
        assert result.document_id == content.document_id
    
    def test_process_document_chunks_sync_convenience(self):
        """Test the sync convenience function."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.PARAGRAPH,
                content="Test content for sync processing",
                page_number=1,
                confidence=0.9
            )
        ]
        
        content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Test content",
            processing_metadata={}
        )
        
        result = process_document_chunks_sync(content)
        assert isinstance(result, ChunkingResult)
        assert result.document_id == content.document_id
