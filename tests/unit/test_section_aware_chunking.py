"""
Unit tests for section-aware text chunking functionality.
"""

import pytest
from uuid import uuid4

from agentic_rag.services.chunking.section_aware_chunker import (
    SectionAwareChunker,
    DocumentSectionAnalyzer,
    SectionInfo,
    SectionMetadata,
    get_section_aware_chunker
)
from agentic_rag.services.chunking.basic_chunker import ChunkingConfig
from agentic_rag.services.content_extraction import (
    ExtractedContent,
    LayoutElement,
    ContentType,
    DocumentStructure
)


class TestDocumentSectionAnalyzer:
    """Test suite for document section analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a document section analyzer for testing."""
        return DocumentSectionAnalyzer()
    
    @pytest.fixture
    def sample_extracted_content(self):
        """Create sample extracted content with sections."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.HEADING,
                content="1. Introduction",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_2",
                type=ContentType.PARAGRAPH,
                content="This is the introduction paragraph with some content.",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_3",
                type=ContentType.HEADING,
                content="2. Methodology",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_4",
                type=ContentType.PARAGRAPH,
                content="This section describes the methodology used in the study.",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_5",
                type=ContentType.HEADING,
                content="2.1 Data Collection",
                page_number=2,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_6",
                type=ContentType.PARAGRAPH,
                content="Data was collected using various methods.",
                page_number=2,
                confidence=0.9
            )
        ]
        
        return ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=2, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Sample document content",
            processing_metadata={}
        )
    
    def test_analyze_document_structure(self, analyzer, sample_extracted_content):
        """Test document structure analysis."""
        sections = analyzer.analyze_document_structure(sample_extracted_content)
        
        # Should identify 3 sections
        assert len(sections) == 3
        
        # Check section titles
        section_titles = [s.title for s in sections]
        assert "1. Introduction" in section_titles
        assert "2. Methodology" in section_titles
        assert "2.1 Data Collection" in section_titles
    
    def test_heading_level_detection(self, analyzer):
        """Test heading level detection."""
        # Test numbered headings
        assert analyzer._determine_heading_level("1. Introduction") == 1
        assert analyzer._determine_heading_level("1.1 Subsection") == 2
        assert analyzer._determine_heading_level("1.1.1 Sub-subsection") == 3
        
        # Test all caps
        assert analyzer._determine_heading_level("INTRODUCTION") == 1
        
        # Test short headings
        assert analyzer._determine_heading_level("Overview") == 2
        
        # Test long headings
        assert analyzer._determine_heading_level("This is a very long heading that should be level 3") == 3
    
    def test_section_hierarchy_building(self, analyzer, sample_extracted_content):
        """Test section hierarchy building."""
        sections = analyzer.analyze_document_structure(sample_extracted_content)
        
        # Find sections by title
        intro_section = next(s for s in sections if "Introduction" in s.title)
        method_section = next(s for s in sections if s.title == "2. Methodology")
        data_section = next(s for s in sections if "Data Collection" in s.title)
        
        # Check hierarchy
        assert intro_section.parent_section_id is None  # Top level
        assert method_section.parent_section_id is None  # Top level
        assert data_section.parent_section_id == method_section.section_id  # Subsection
        
        # Check subsections
        assert data_section.section_id in method_section.subsections
    
    def test_document_without_headings(self, analyzer):
        """Test analysis of document without explicit headings."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.PARAGRAPH,
                content="Just some paragraph content without headings.",
                page_number=1,
                confidence=0.9
            )
        ]
        
        content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Sample content",
            processing_metadata={}
        )
        
        sections = analyzer.analyze_document_structure(content)
        
        # Should create a default section
        assert len(sections) == 1
        assert sections[0].title == "Document Content"
        assert sections[0].level == 1


class TestSectionAwareChunker:
    """Test suite for section-aware chunking."""
    
    @pytest.fixture
    def chunker(self):
        """Create a section-aware chunker for testing."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=30)
        return SectionAwareChunker(config)
    
    @pytest.fixture
    def sample_extracted_content(self):
        """Create sample extracted content for testing."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.HEADING,
                content="Introduction",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_2",
                type=ContentType.PARAGRAPH,
                content="This is a long introduction paragraph that should be chunked appropriately. " * 3,
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_3",
                type=ContentType.HEADING,
                content="Methodology",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_4",
                type=ContentType.PARAGRAPH,
                content="This methodology section also contains substantial content that needs chunking. " * 3,
                page_number=1,
                confidence=0.9
            )
        ]
        
        return ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Sample document content",
            processing_metadata={}
        )
    
    def test_section_aware_chunking(self, chunker, sample_extracted_content):
        """Test basic section-aware chunking functionality."""
        chunks = chunker.chunk_document(sample_extracted_content)
        
        # Should create multiple chunks
        assert len(chunks) > 0
        
        # All chunks should have section metadata
        for chunk in chunks:
            assert hasattr(chunk, 'section_metadata')
            assert chunk.section_metadata.section_id is not None
            assert chunk.section_metadata.section_title is not None
            assert chunk.section_metadata.section_level > 0
    
    def test_section_boundaries_preserved(self, chunker, sample_extracted_content):
        """Test that section boundaries are preserved."""
        chunks = chunker.chunk_document(sample_extracted_content)
        
        # Group chunks by section
        sections = {}
        for chunk in chunks:
            section_id = chunk.section_metadata.section_id
            if section_id not in sections:
                sections[section_id] = []
            sections[section_id].append(chunk)
        
        # Should have chunks for each section
        assert len(sections) >= 2  # At least Introduction and Methodology
        
        # Check section titles
        section_titles = set()
        for chunk in chunks:
            section_titles.add(chunk.section_metadata.section_title)
        
        assert "Introduction" in section_titles
        assert "Methodology" in section_titles
    
    def test_section_start_and_end_markers(self, chunker, sample_extracted_content):
        """Test section start and end markers."""
        chunks = chunker.chunk_document(sample_extracted_content)
        
        # Group chunks by section
        sections = {}
        for chunk in chunks:
            section_id = chunk.section_metadata.section_id
            if section_id not in sections:
                sections[section_id] = []
            sections[section_id].append(chunk)
        
        # Check start and end markers for each section
        for section_chunks in sections.values():
            if len(section_chunks) > 0:
                # First chunk should be marked as section start
                assert section_chunks[0].section_metadata.is_section_start is True
                
                # Last chunk should be marked as section end
                assert section_chunks[-1].section_metadata.is_section_end is True
                
                # Middle chunks should not be start or end
                for chunk in section_chunks[1:-1]:
                    assert chunk.section_metadata.is_section_start is False
                    assert chunk.section_metadata.is_section_end is False
    
    def test_heading_detection_in_chunks(self, chunker, sample_extracted_content):
        """Test detection of headings in chunks."""
        chunks = chunker.chunk_document(sample_extracted_content)
        
        # Find chunks that should contain headings
        heading_chunks = [
            chunk for chunk in chunks 
            if chunk.section_metadata.contains_heading
        ]
        
        # Should have at least some chunks with headings
        assert len(heading_chunks) > 0
        
        # Check heading text
        for chunk in heading_chunks:
            assert chunk.section_metadata.heading_text is not None
            assert len(chunk.section_metadata.heading_text.strip()) > 0
    
    def test_section_path_building(self, chunker):
        """Test section path building for nested sections."""
        # Create content with nested sections
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.HEADING,
                content="1. Main Section",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_2",
                type=ContentType.PARAGRAPH,
                content="Main section content.",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_3",
                type=ContentType.HEADING,
                content="1.1 Subsection",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_4",
                type=ContentType.PARAGRAPH,
                content="Subsection content.",
                page_number=1,
                confidence=0.9
            )
        ]
        
        content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Sample content",
            processing_metadata={}
        )
        
        chunks = chunker.chunk_document(content)
        
        # Find subsection chunks
        subsection_chunks = [
            chunk for chunk in chunks 
            if "Subsection" in chunk.section_metadata.section_title
        ]
        
        # Check section path
        for chunk in subsection_chunks:
            path = chunk.section_metadata.section_path
            assert len(path) >= 2
            assert "Main Section" in path[0]
            assert "Subsection" in path[1]
    
    def test_full_context_generation(self, chunker, sample_extracted_content):
        """Test full context generation for chunks."""
        chunks = chunker.chunk_document(sample_extracted_content)
        
        for chunk in chunks:
            full_context = chunk.get_full_context()
            
            # Should contain the chunk content
            assert chunk.content in full_context
            
            # Should contain section information if path exists
            if chunk.section_metadata.section_path:
                path_str = " > ".join(chunk.section_metadata.section_path)
                assert path_str in full_context
    
    def test_section_summary_generation(self, chunker, sample_extracted_content):
        """Test section summary generation."""
        chunks = chunker.chunk_document(sample_extracted_content)
        summary = chunker.get_section_summary(chunks)
        
        # Check summary structure
        assert "total_sections" in summary
        assert "sections" in summary
        assert "total_chunks" in summary
        
        # Validate summary data
        assert summary["total_sections"] > 0
        assert summary["total_chunks"] == len(chunks)
        
        # Check section details
        for section_id, section_info in summary["sections"].items():
            assert "title" in section_info
            assert "level" in section_info
            assert "chunk_count" in section_info
            assert "total_characters" in section_info
            assert "total_words" in section_info
            
            assert section_info["chunk_count"] > 0
            assert section_info["total_characters"] > 0
    
    def test_empty_content_handling(self, chunker):
        """Test handling of empty or minimal content."""
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
        # May be empty or have a default section


class TestGlobalSectionAwareChunker:
    """Test the global section-aware chunker instance functionality."""
    
    def test_get_section_aware_chunker_default(self):
        """Test getting section-aware chunker with default config."""
        chunker = get_section_aware_chunker()
        assert isinstance(chunker, SectionAwareChunker)
        assert chunker.config.chunk_size == 1000  # Default value
    
    def test_get_section_aware_chunker_custom_config(self):
        """Test getting section-aware chunker with custom config."""
        config = ChunkingConfig(chunk_size=500)
        chunker = get_section_aware_chunker(config)
        assert isinstance(chunker, SectionAwareChunker)
        assert chunker.config.chunk_size == 500
    
    def test_chunker_singleton_behavior(self):
        """Test that the same instance is returned when no config is provided."""
        chunker1 = get_section_aware_chunker()
        chunker2 = get_section_aware_chunker()
        assert chunker1 is chunker2
