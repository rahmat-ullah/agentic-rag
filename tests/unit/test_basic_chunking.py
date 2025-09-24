"""
Unit tests for basic text chunking functionality.
"""

import pytest
from uuid import uuid4

from agentic_rag.services.chunking.basic_chunker import (
    BasicTextChunker,
    ChunkingConfig,
    SentenceDetector,
    TextChunk,
    get_basic_chunker
)


class TestSentenceDetector:
    """Test suite for sentence detection."""
    
    def test_simple_sentence_detection(self):
        """Test basic sentence boundary detection."""
        detector = SentenceDetector()
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        
        boundaries = detector.find_sentence_boundaries(text)
        
        # Should find boundaries after each sentence
        assert len(boundaries) >= 2
        assert boundaries[0] > text.find("one.")
        assert boundaries[1] > text.find("two!")
    
    def test_abbreviation_handling(self):
        """Test that abbreviations don't create false sentence boundaries."""
        detector = SentenceDetector()
        text = "Dr. Smith went to the store. He bought milk."
        
        boundaries = detector.find_sentence_boundaries(text)
        
        # Should not break at "Dr." but should break after "store."
        assert len(boundaries) >= 1
        first_boundary = boundaries[0]
        assert first_boundary > text.find("store.")
    
    def test_empty_text(self):
        """Test sentence detection with empty text."""
        detector = SentenceDetector()
        boundaries = detector.find_sentence_boundaries("")
        assert boundaries == []
    
    def test_no_sentences(self):
        """Test text with no clear sentence boundaries."""
        detector = SentenceDetector()
        text = "just some words without proper punctuation"
        boundaries = detector.find_sentence_boundaries(text)
        assert len(boundaries) == 0


class TestChunkingConfig:
    """Test suite for chunking configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 2000
        assert config.respect_sentence_boundaries is True
        assert config.respect_paragraph_boundaries is True
        assert config.language == "en"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
            respect_sentence_boundaries=False
        )
        
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.min_chunk_size == 50
        assert config.respect_sentence_boundaries is False


class TestBasicTextChunker:
    """Test suite for basic text chunking."""
    
    @pytest.fixture
    def chunker(self):
        """Create a basic text chunker for testing."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=30)
        return BasicTextChunker(config)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return (
            "This is the first sentence of our test document. "
            "It contains multiple sentences to test chunking. "
            "The chunker should break this text into appropriate chunks. "
            "Each chunk should maintain context through overlap. "
            "This helps ensure that information is not lost at boundaries."
        )
    
    def test_basic_chunking(self, chunker, sample_text):
        """Test basic text chunking functionality."""
        document_id = str(uuid4())
        chunks = chunker.chunk_text(sample_text, document_id)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.metadata.document_id == document_id
        
        # Chunks should be ordered
        for i in range(len(chunks) - 1):
            assert chunks[i].metadata.chunk_index < chunks[i + 1].metadata.chunk_index
    
    def test_chunk_overlap(self, chunker, sample_text):
        """Test that chunks have proper overlap."""
        document_id = str(uuid4())
        chunks = chunker.chunk_text(sample_text, document_id)
        
        if len(chunks) > 1:
            # Check overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                
                # Should have some overlap
                assert next_chunk.metadata.overlap_with_previous > 0
                assert current_chunk.metadata.overlap_with_next > 0
    
    def test_chunk_metadata(self, chunker, sample_text):
        """Test chunk metadata creation."""
        document_id = str(uuid4())
        chunks = chunker.chunk_text(sample_text, document_id)
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            
            # Basic metadata checks
            assert metadata.chunk_id is not None
            assert metadata.document_id == document_id
            assert metadata.chunk_index == i
            assert metadata.start_char >= 0
            assert metadata.end_char > metadata.start_char
            assert metadata.chunk_size == len(chunk.content)
            assert metadata.word_count > 0
            assert 0 <= metadata.quality_score <= 1.0
    
    def test_empty_text(self, chunker):
        """Test chunking empty text."""
        document_id = str(uuid4())
        chunks = chunker.chunk_text("", document_id)
        assert chunks == []
        
        chunks = chunker.chunk_text("   ", document_id)
        assert chunks == []
    
    def test_short_text(self, chunker):
        """Test chunking very short text."""
        document_id = str(uuid4())
        short_text = "Short text."
        chunks = chunker.chunk_text(short_text, document_id)
        
        # Should create at least one chunk
        assert len(chunks) >= 1
        assert chunks[0].content.strip() == short_text.strip()
    
    def test_long_text(self):
        """Test chunking very long text."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=50)
        chunker = BasicTextChunker(config)
        
        # Create long text
        long_text = "This is a sentence. " * 100  # 2000 characters
        document_id = str(uuid4())
        
        chunks = chunker.chunk_text(long_text, document_id)
        
        # Should create multiple chunks
        assert len(chunks) > 5
        
        # No chunk should be excessively long
        for chunk in chunks:
            assert len(chunk.content) <= config.max_chunk_size
    
    def test_sentence_boundary_respect(self):
        """Test that sentence boundaries are respected."""
        config = ChunkingConfig(
            chunk_size=50, 
            chunk_overlap=10, 
            respect_sentence_boundaries=True
        )
        chunker = BasicTextChunker(config)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        document_id = str(uuid4())
        
        chunks = chunker.chunk_text(text, document_id)
        
        # Chunks should generally end at sentence boundaries
        for chunk in chunks[:-1]:  # Exclude last chunk
            content = chunk.content.strip()
            if len(content) > 20:  # Only check substantial chunks
                assert content.endswith('.') or content.endswith('!') or content.endswith('?')
    
    def test_paragraph_boundary_respect(self):
        """Test that paragraph boundaries are respected."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            respect_paragraph_boundaries=True
        )
        chunker = BasicTextChunker(config)
        
        text = "First paragraph.\n\nSecond paragraph with more text.\n\nThird paragraph."
        document_id = str(uuid4())
        
        chunks = chunker.chunk_text(text, document_id)
        
        # Should create chunks that respect paragraph boundaries
        assert len(chunks) >= 1
        
        # Check that chunks don't unnecessarily break within paragraphs
        for chunk in chunks:
            # Chunk shouldn't start or end with paragraph breaks unless necessary
            content = chunk.content.strip()
            assert not content.startswith('\n\n')
    
    def test_chunking_stats(self, chunker, sample_text):
        """Test chunking statistics generation."""
        document_id = str(uuid4())
        chunks = chunker.chunk_text(sample_text, document_id)
        
        stats = chunker.get_chunking_stats(chunks)
        
        # Check required statistics
        assert "total_chunks" in stats
        assert "total_characters" in stats
        assert "total_words" in stats
        assert "average_chunk_size" in stats
        assert "min_chunk_size" in stats
        assert "max_chunk_size" in stats
        assert "average_quality_score" in stats
        
        # Validate statistics
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_characters"] > 0
        assert stats["total_words"] > 0
        assert stats["average_chunk_size"] > 0
        assert 0 <= stats["average_quality_score"] <= 1.0
    
    def test_quality_score_calculation(self, chunker):
        """Test chunk quality score calculation."""
        document_id = str(uuid4())
        
        # Test high quality chunk
        good_text = "This is a well-formed sentence with proper punctuation."
        good_chunks = chunker.chunk_text(good_text, document_id)
        assert good_chunks[0].metadata.quality_score > 0.8
        
        # Test lower quality chunk
        poor_text = "a"  # Very short
        poor_chunks = chunker.chunk_text(poor_text, document_id)
        if poor_chunks:  # Might be filtered out
            assert poor_chunks[0].metadata.quality_score < 0.8
    
    def test_chunk_serialization(self, chunker, sample_text):
        """Test chunk serialization to dictionary."""
        document_id = str(uuid4())
        chunks = chunker.chunk_text(sample_text, document_id)
        
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            
            # Check required fields
            assert "id" in chunk_dict
            assert "content" in chunk_dict
            assert "metadata" in chunk_dict
            assert "size" in chunk_dict
            
            # Validate data
            assert chunk_dict["id"] == chunk.id
            assert chunk_dict["content"] == chunk.content
            assert chunk_dict["size"] == chunk.size


class TestGlobalChunkerInstance:
    """Test the global chunker instance functionality."""
    
    def test_get_basic_chunker_default(self):
        """Test getting basic chunker with default config."""
        chunker = get_basic_chunker()
        assert isinstance(chunker, BasicTextChunker)
        assert chunker.config.chunk_size == 1000  # Default value
    
    def test_get_basic_chunker_custom_config(self):
        """Test getting basic chunker with custom config."""
        config = ChunkingConfig(chunk_size=500)
        chunker = get_basic_chunker(config)
        assert isinstance(chunker, BasicTextChunker)
        assert chunker.config.chunk_size == 500
    
    def test_chunker_singleton_behavior(self):
        """Test that the same instance is returned when no config is provided."""
        chunker1 = get_basic_chunker()
        chunker2 = get_basic_chunker()
        assert chunker1 is chunker2
