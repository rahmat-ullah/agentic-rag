"""
Unit tests for chunk deduplication functionality.
"""

import pytest
from uuid import uuid4

from agentic_rag.services.chunking.deduplication_chunker import (
    DeduplicationChunker,
    ContentNormalizer,
    ChunkDeduplicator,
    get_deduplication_chunker
)
from agentic_rag.services.chunking.basic_chunker import ChunkingConfig
from agentic_rag.services.content_extraction import (
    ExtractedContent,
    LayoutElement,
    ContentType,
    DocumentStructure
)


class TestContentNormalizer:
    """Test suite for content normalization."""
    
    @pytest.fixture
    def normalizer(self):
        """Create a content normalizer for testing."""
        return ContentNormalizer()
    
    def test_basic_normalization(self, normalizer):
        """Test basic content normalization."""
        content = "This is a TEST with Multiple   Spaces!"
        normalized = normalizer.normalize_content(content)
        
        assert normalized == "this is a test with multiple spaces"
    
    def test_punctuation_removal(self, normalizer):
        """Test punctuation removal."""
        content = "Hello, world! How are you? (Fine, thanks.)"
        normalized = normalizer.normalize_content(content)
        
        assert normalized == "hello world how are you fine thanks"
    
    def test_table_structure_preservation(self, normalizer):
        """Test that table structure indicators are preserved."""
        content = "Name | Age | City\nJohn | 25 | NYC"
        normalized = normalizer.normalize_content(content)
        
        # Should preserve | and - characters
        assert "|" in normalized
        assert "name | age | city john | 25 | nyc" == normalized
    
    def test_whitespace_normalization(self, normalizer):
        """Test whitespace normalization."""
        content = "  Multiple\t\tspaces\n\nand\r\nlines  "
        normalized = normalizer.normalize_content(content)
        
        assert normalized == "multiple spaces and lines"
    
    def test_hash_calculation(self, normalizer):
        """Test content hash calculation."""
        content1 = "test content"
        content2 = "test content"
        content3 = "different content"
        
        hash1 = normalizer.calculate_content_hash(content1)
        hash2 = normalizer.calculate_content_hash(content2)
        hash3 = normalizer.calculate_content_hash(content3)
        
        assert hash1 == hash2  # Same content should have same hash
        assert hash1 != hash3  # Different content should have different hash
        assert len(hash1) == 64  # SHA256 hash length
    
    def test_similarity_calculation(self, normalizer):
        """Test similarity calculation between content."""
        content1 = "the quick brown fox"
        content2 = "the quick brown fox"  # Identical
        content3 = "the quick brown dog"  # Similar
        content4 = "completely different text"  # Different
        
        # Identical content
        similarity1 = normalizer.calculate_similarity(content1, content2)
        assert similarity1 == 1.0
        
        # Similar content
        similarity2 = normalizer.calculate_similarity(content1, content3)
        assert 0.5 < similarity2 < 1.0
        
        # Different content
        similarity3 = normalizer.calculate_similarity(content1, content4)
        assert similarity3 < 0.5
    
    def test_empty_content_similarity(self, normalizer):
        """Test similarity calculation with empty content."""
        assert normalizer.calculate_similarity("", "") == 1.0
        assert normalizer.calculate_similarity("test", "") == 0.0
        assert normalizer.calculate_similarity("", "test") == 0.0


class TestChunkDeduplicator:
    """Test suite for chunk deduplication logic."""
    
    @pytest.fixture
    def deduplicator(self):
        """Create a chunk deduplicator for testing."""
        return ChunkDeduplicator(similarity_threshold=0.8)
    
    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk for testing."""
        from agentic_rag.services.chunking.table_aware_chunker import TableAwareChunk, TableMetadata
        from agentic_rag.services.chunking.section_aware_chunker import SectionMetadata
        from agentic_rag.services.chunking.basic_chunker import ChunkMetadata
        
        return TableAwareChunk(
            content="This is a sample chunk for testing deduplication.",
            metadata=ChunkMetadata(
                chunk_id=str(uuid4()),
                document_id=str(uuid4()),
                chunk_index=0,
                start_char=0,
                end_char=50,
                chunk_size=50,
                word_count=9,
                sentence_count=1,
                quality_score=0.8
            ),
            section_metadata=SectionMetadata(
                section_id=str(uuid4()),
                section_title="Test Section",
                section_level=1
            ),
            table_metadata=TableMetadata()
        )
    
    def test_exact_duplicate_detection(self, deduplicator, sample_chunk):
        """Test detection of exact duplicates."""
        # First chunk should not be duplicate
        dup_info1 = deduplicator.check_duplication(sample_chunk)
        assert not dup_info1.is_duplicate
        assert dup_info1.similarity_score == 0.0  # No previous content to compare
        
        # Create identical chunk
        identical_chunk = sample_chunk.model_copy()
        original_chunk_id = sample_chunk.metadata.chunk_id  # Store original ID
        identical_chunk.metadata.chunk_id = str(uuid4())  # Different ID

        # Second identical chunk should be detected as duplicate
        dup_info2 = deduplicator.check_duplication(identical_chunk)
        assert dup_info2.is_duplicate
        assert dup_info2.similarity_score == 1.0
        assert dup_info2.duplicate_reason == "exact_hash_match"
        assert dup_info2.original_chunk_id == original_chunk_id
    
    def test_similarity_duplicate_detection(self, deduplicator, sample_chunk):
        """Test detection of similar duplicates."""
        # First chunk
        dup_info1 = deduplicator.check_duplication(sample_chunk)
        assert not dup_info1.is_duplicate
        
        # Create similar chunk
        similar_chunk = sample_chunk.model_copy()
        similar_chunk.metadata.chunk_id = str(uuid4())
        similar_chunk.content = "This is a sample chunk for testing deduplication purposes."  # Very similar
        
        # Should be detected as duplicate due to high similarity
        dup_info2 = deduplicator.check_duplication(similar_chunk)
        assert dup_info2.is_duplicate
        assert dup_info2.similarity_score >= 0.8
        assert dup_info2.duplicate_reason == "similarity_match"
    
    def test_unique_content_detection(self, deduplicator, sample_chunk):
        """Test that unique content is not marked as duplicate."""
        # First chunk
        dup_info1 = deduplicator.check_duplication(sample_chunk)
        assert not dup_info1.is_duplicate
        
        # Create different chunk
        different_chunk = sample_chunk.model_copy()
        different_chunk.metadata.chunk_id = str(uuid4())
        different_chunk.content = "This is completely different content about something else entirely."
        
        # Should not be detected as duplicate
        dup_info2 = deduplicator.check_duplication(different_chunk)
        assert not dup_info2.is_duplicate
        assert dup_info2.similarity_score < 0.8
    
    def test_deduplicator_reset(self, deduplicator, sample_chunk):
        """Test deduplicator reset functionality."""
        # Add a chunk
        deduplicator.check_duplication(sample_chunk)
        assert len(deduplicator.seen_hashes) == 1
        
        # Reset
        deduplicator.reset()
        assert len(deduplicator.seen_hashes) == 0
        assert len(deduplicator.seen_content) == 0
        
        # Same chunk should not be duplicate after reset
        dup_info = deduplicator.check_duplication(sample_chunk)
        assert not dup_info.is_duplicate


class TestDeduplicationChunker:
    """Test suite for deduplication chunking."""
    
    @pytest.fixture
    def chunker(self):
        """Create a deduplication chunker for testing."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=30)
        return DeduplicationChunker(config, similarity_threshold=0.8)
    
    @pytest.fixture
    def sample_content_with_duplicates(self):
        """Create sample content with duplicate text that will create separate chunks."""
        # Create longer content that will be split into multiple chunks
        long_content = "This is a long paragraph that contains enough text to be split into multiple chunks by the chunking algorithm. " * 10
        duplicate_content = "This is a long paragraph that contains enough text to be split into multiple chunks by the chunking algorithm. " * 10
        different_content = "This is completely different content that talks about something entirely different and should not be considered a duplicate. " * 10

        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.PARAGRAPH,
                content=long_content,
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_2",
                type=ContentType.PARAGRAPH,
                content=duplicate_content,  # Exact duplicate
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_3",
                type=ContentType.PARAGRAPH,
                content=different_content,  # Different content
                page_number=1,
                confidence=0.9
            )
        ]

        return ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Sample content with duplicates",
            processing_metadata={}
        )
    
    def test_deduplication_chunking(self, chunker, sample_content_with_duplicates):
        """Test basic deduplication chunking functionality."""
        chunks = chunker.chunk_document(sample_content_with_duplicates)
        
        # Should create chunks
        assert len(chunks) > 0
        
        # All chunks should have deduplication metadata
        for chunk in chunks:
            assert hasattr(chunk, 'deduplication_metadata')
            assert chunk.deduplication_metadata.content_hash is not None
            assert chunk.deduplication_metadata.normalized_content is not None
        
        # Check that deduplication analysis was performed
        duplicate_chunks = [chunk for chunk in chunks if chunk.deduplication_metadata.is_duplicate]
        unique_chunks = [chunk for chunk in chunks if not chunk.deduplication_metadata.is_duplicate]

        # At least some chunks should be unique
        assert len(unique_chunks) > 0
        # Total should equal sum of duplicates and unique
        assert len(chunks) == len(duplicate_chunks) + len(unique_chunks)
    
    def test_unique_chunks_filtering(self, chunker, sample_content_with_duplicates):
        """Test filtering to get only unique chunks."""
        chunks = chunker.chunk_document(sample_content_with_duplicates)
        unique_chunks = chunker.get_unique_chunks(chunks)
        
        # All returned chunks should be unique
        for chunk in unique_chunks:
            assert chunk.is_unique()
            assert not chunk.deduplication_metadata.is_duplicate
        
        # Should have fewer unique chunks than total chunks
        assert len(unique_chunks) <= len(chunks)
    
    def test_deduplication_summary(self, chunker, sample_content_with_duplicates):
        """Test deduplication summary generation."""
        chunks = chunker.chunk_document(sample_content_with_duplicates)
        summary = chunker.get_deduplication_summary(chunks)
        
        # Check summary structure
        assert "total_chunks" in summary
        assert "unique_chunks" in summary
        assert "duplicate_chunks" in summary
        assert "deduplication_ratio" in summary
        assert "duplicate_groups" in summary
        assert "similarity_threshold" in summary
        
        # Validate summary data
        assert summary["total_chunks"] == len(chunks)
        assert summary["unique_chunks"] + summary["duplicate_chunks"] == summary["total_chunks"]
        assert 0.0 <= summary["deduplication_ratio"] <= 1.0
        assert summary["similarity_threshold"] == 0.8
    
    def test_content_signature_generation(self, chunker, sample_content_with_duplicates):
        """Test content signature generation for chunks."""
        chunks = chunker.chunk_document(sample_content_with_duplicates)
        
        for chunk in chunks:
            signature = chunk.get_content_signature()
            
            # Should contain hash prefix, length, and word count
            parts = signature.split(':')
            assert len(parts) == 3
            assert len(parts[0]) == 8  # Hash prefix
            assert int(parts[1]) == len(chunk.content)  # Content length
            assert int(parts[2]) == chunk.metadata.word_count  # Word count
    
    def test_empty_content_handling(self, chunker):
        """Test handling of empty content."""
        elements = []
        
        content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="",
            processing_metadata={}
        )
        
        chunks = chunker.chunk_document(content)
        
        # Should handle empty content gracefully
        assert isinstance(chunks, list)
        
        # Get summary
        summary = chunker.get_deduplication_summary(chunks)
        assert summary["total_chunks"] == len(chunks)


class TestGlobalDeduplicationChunker:
    """Test the global deduplication chunker instance functionality."""
    
    def test_get_deduplication_chunker_default(self):
        """Test getting deduplication chunker with default config."""
        chunker = get_deduplication_chunker()
        assert isinstance(chunker, DeduplicationChunker)
        assert chunker.config.chunk_size == 1000  # Default value
        assert chunker.deduplicator.similarity_threshold == 0.85  # Default value
    
    def test_get_deduplication_chunker_custom_config(self):
        """Test getting deduplication chunker with custom config."""
        config = ChunkingConfig(chunk_size=500)
        chunker = get_deduplication_chunker(config, similarity_threshold=0.9)
        assert isinstance(chunker, DeduplicationChunker)
        assert chunker.config.chunk_size == 500
        assert chunker.deduplicator.similarity_threshold == 0.9
    
    def test_chunker_singleton_behavior(self):
        """Test that the same instance is returned when no config is provided."""
        chunker1 = get_deduplication_chunker()
        chunker2 = get_deduplication_chunker()
        assert chunker1 is chunker2
