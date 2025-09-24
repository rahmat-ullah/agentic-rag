"""
Unit tests for table-aware text chunking functionality.
"""

import pytest
from uuid import uuid4

from agentic_rag.services.chunking.table_aware_chunker import (
    TableAwareChunker,
    TableDetector,
    TableInfo,
    TableMetadata,
    get_table_aware_chunker
)
from agentic_rag.services.chunking.basic_chunker import ChunkingConfig
from agentic_rag.services.content_extraction import (
    ExtractedContent,
    LayoutElement,
    ContentType,
    DocumentStructure,
    ExtractedTable
)


class TestTableDetector:
    """Test suite for table detection."""
    
    @pytest.fixture
    def detector(self):
        """Create a table detector for testing."""
        return TableDetector()
    
    @pytest.fixture
    def sample_content_with_tables(self):
        """Create sample content with tables."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.HEADING,
                content="Data Analysis Results",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_2",
                type=ContentType.TABLE,
                content="Name | Age | City\nJohn | 25 | NYC\nJane | 30 | LA",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_3",
                type=ContentType.PARAGRAPH,
                content="The table above shows demographic data.",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_4",
                type=ContentType.TABLE,
                content="Product\tPrice\tStock\nLaptop\t$999\t15\nMouse\t$25\t50",
                page_number=1,
                confidence=0.9
            )
        ]
        
        tables = [
            ExtractedTable(
                id="table_1",
                page_number=1,
                headers=["Name", "Age", "City"],
                rows=[
                    ["John", "25", "NYC"],
                    ["Jane", "30", "LA"]
                ],
                row_count=2,
                column_count=3,
                caption="Demographics Table"
            )
        ]
        
        return ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=tables,
            text_content="Sample document with tables",
            processing_metadata={}
        )
    
    def test_detect_explicit_tables(self, detector, sample_content_with_tables):
        """Test detection of explicit table elements."""
        tables = detector.detect_tables(sample_content_with_tables)
        
        # Should detect tables from both elements and table data
        assert len(tables) >= 2
        
        # Check table properties
        table_contents = []
        for table in tables:
            assert table.table_id is not None
            assert table.row_count > 0
            assert table.column_count > 0
            table_contents.append(table.table_id)
        
        # Should have detected the explicit table data
        table_titles = [t.table_title for t in tables if t.table_title]
        assert "Demographics Table" in table_titles
    
    def test_detect_implicit_tables(self, detector):
        """Test detection of implicit tables from structured text."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.PARAGRAPH,
                content="Product    Price    Stock\nLaptop     $999     15\nMouse      $25      50\nKeyboard   $75      25",
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
        
        tables = detector.detect_tables(content)
        
        # Should detect the implicit table
        assert len(tables) >= 1
        
        table = tables[0]
        assert table.row_count >= 3  # Header + 3 data rows
        assert table.column_count >= 3  # Product, Price, Stock
    
    def test_table_pattern_recognition(self, detector):
        """Test table pattern recognition."""
        # Test pipe-separated table
        pipe_table = "Name | Age | City\nJohn | 25 | NYC\nJane | 30 | LA"
        assert detector._looks_like_table(pipe_table) is True
        
        # Test tab-separated table
        tab_table = "Name\tAge\tCity\nJohn\t25\tNYC\nJane\t30\tLA"
        assert detector._looks_like_table(tab_table) is True
        
        # Test space-separated table
        space_table = "Name    Age    City\nJohn    25     NYC\nJane    30     LA"
        assert detector._looks_like_table(space_table) is True
        
        # Test non-table content
        regular_text = "This is just regular paragraph text without any tabular structure."
        assert detector._looks_like_table(regular_text) is False
    
    def test_column_estimation(self, detector):
        """Test column count estimation."""
        # Pipe-separated
        pipe_content = "A | B | C\n1 | 2 | 3"
        assert detector._estimate_columns(pipe_content) == 3
        
        # Tab-separated
        tab_content = "A\tB\tC\tD\n1\t2\t3\t4"
        assert detector._estimate_columns(tab_content) == 4
        
        # Space-separated
        space_content = "A    B    C\n1    2    3"
        assert detector._estimate_columns(space_content) >= 2


class TestTableAwareChunker:
    """Test suite for table-aware chunking."""
    
    @pytest.fixture
    def chunker(self):
        """Create a table-aware chunker for testing."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=30, min_chunk_size=50)
        return TableAwareChunker(config)
    
    @pytest.fixture
    def sample_content_with_tables(self):
        """Create sample content with tables for testing."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.HEADING,
                content="Sales Report",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_2",
                type=ContentType.PARAGRAPH,
                content="This report contains sales data for Q1 2024.",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_3",
                type=ContentType.TABLE,
                content="Month | Sales | Growth\nJan | $10K | 5%\nFeb | $12K | 20%\nMar | $15K | 25%",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_4",
                type=ContentType.PARAGRAPH,
                content="The table shows consistent growth throughout the quarter.",
                page_number=1,
                confidence=0.9
            )
        ]
        
        tables = [
            ExtractedTable(
                id="sales_table",
                page_number=1,
                headers=["Month", "Sales", "Growth"],
                rows=[
                    ["Jan", "$10K", "5%"],
                    ["Feb", "$12K", "20%"],
                    ["Mar", "$15K", "25%"]
                ],
                row_count=3,
                column_count=3,
                caption="Q1 Sales Data"
            )
        ]
        
        return ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=tables,
            text_content="Sales report content",
            processing_metadata={}
        )
    
    def test_table_aware_chunking(self, chunker, sample_content_with_tables):
        """Test basic table-aware chunking functionality."""
        chunks = chunker.chunk_document(sample_content_with_tables)
        
        # Should create chunks
        assert len(chunks) > 0
        
        # All chunks should have table metadata
        for chunk in chunks:
            assert hasattr(chunk, 'table_metadata')
            assert isinstance(chunk.table_metadata, TableMetadata)
        
        # At least one chunk should contain table content
        table_chunks = [chunk for chunk in chunks if chunk.table_metadata.contains_table]
        assert len(table_chunks) > 0
    
    def test_table_detection_in_chunks(self, chunker, sample_content_with_tables):
        """Test table detection within chunks."""
        chunks = chunker.chunk_document(sample_content_with_tables)
        
        # Find chunks with table content
        table_chunks = [chunk for chunk in chunks if chunk.table_metadata.contains_table]
        
        # Verify table metadata
        for chunk in table_chunks:
            assert chunk.table_metadata.table_id is not None
            
            # Check if table structure is detected
            if chunk.table_metadata.table_rows:
                assert len(chunk.table_metadata.table_rows) > 0
            
            if chunk.table_metadata.table_columns:
                assert len(chunk.table_metadata.table_columns) > 0
    
    def test_table_context_generation(self, chunker, sample_content_with_tables):
        """Test table context generation."""
        chunks = chunker.chunk_document(sample_content_with_tables)
        
        # Find table chunks
        table_chunks = [chunk for chunk in chunks if chunk.table_metadata.contains_table]
        
        for chunk in table_chunks:
            table_context = chunk.get_table_context()
            
            # Should contain the chunk content
            assert chunk.content in table_context
            
            # Should contain table information if available
            if chunk.table_metadata.table_title:
                assert chunk.table_metadata.table_title in table_context
    
    def test_table_chunk_merging(self, chunker):
        """Test merging of small table chunks."""
        # Create content with small table chunks
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.TABLE,
                content="A | B\n1 | 2",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_2",
                type=ContentType.TABLE,
                content="C | D\n3 | 4",
                page_number=1,
                confidence=0.9
            )
        ]
        
        content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Small table content",
            processing_metadata={}
        )
        
        chunks = chunker.chunk_document(content)
        
        # Should handle small table chunks appropriately
        assert len(chunks) > 0
        
        # Check that table chunks are properly identified
        table_chunks = [chunk for chunk in chunks if chunk.table_metadata.contains_table]
        assert len(table_chunks) > 0
    
    def test_table_summary_generation(self, chunker, sample_content_with_tables):
        """Test table summary generation."""
        chunks = chunker.chunk_document(sample_content_with_tables)
        summary = chunker.get_table_summary(chunks)
        
        # Check summary structure
        assert "total_tables" in summary
        assert "total_table_chunks" in summary
        assert "total_chunks" in summary
        assert "tables" in summary
        
        # Validate summary data
        assert summary["total_chunks"] == len(chunks)
        
        table_chunks = [chunk for chunk in chunks if chunk.table_metadata.contains_table]
        assert summary["total_table_chunks"] == len(table_chunks)
        
        # Check table details
        if summary["tables"]:
            for table_id, table_info in summary["tables"].items():
                assert "title" in table_info
                assert "chunk_count" in table_info
                assert "total_rows" in table_info
                assert "total_columns" in table_info
    
    def test_mixed_content_handling(self, chunker):
        """Test handling of mixed content with tables and regular text."""
        elements = [
            LayoutElement(
                id="elem_1",
                type=ContentType.PARAGRAPH,
                content="Introduction paragraph with regular text content.",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_2",
                type=ContentType.TABLE,
                content="Name | Value\nItem1 | 100\nItem2 | 200",
                page_number=1,
                confidence=0.9
            ),
            LayoutElement(
                id="elem_3",
                type=ContentType.PARAGRAPH,
                content="Conclusion paragraph discussing the table results.",
                page_number=1,
                confidence=0.9
            )
        ]
        
        content = ExtractedContent(
            document_id=str(uuid4()),
            structure=DocumentStructure(page_count=1, document_type="pdf"),
            elements=elements,
            tables=[],
            text_content="Mixed content",
            processing_metadata={}
        )
        
        chunks = chunker.chunk_document(content)
        
        # Should create chunks for both table and non-table content
        assert len(chunks) > 0
        
        # Should have both table and non-table chunks
        table_chunks = [chunk for chunk in chunks if chunk.table_metadata.contains_table]
        non_table_chunks = [chunk for chunk in chunks if not chunk.table_metadata.contains_table]
        
        # Both types should exist
        assert len(table_chunks) > 0
        assert len(non_table_chunks) > 0
    
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


class TestGlobalTableAwareChunker:
    """Test the global table-aware chunker instance functionality."""
    
    def test_get_table_aware_chunker_default(self):
        """Test getting table-aware chunker with default config."""
        chunker = get_table_aware_chunker()
        assert isinstance(chunker, TableAwareChunker)
        assert chunker.config.chunk_size == 1000  # Default value
    
    def test_get_table_aware_chunker_custom_config(self):
        """Test getting table-aware chunker with custom config."""
        config = ChunkingConfig(chunk_size=500)
        chunker = get_table_aware_chunker(config)
        assert isinstance(chunker, TableAwareChunker)
        assert chunker.config.chunk_size == 500
    
    def test_chunker_singleton_behavior(self):
        """Test that the same instance is returned when no config is provided."""
        chunker1 = get_table_aware_chunker()
        chunker2 = get_table_aware_chunker()
        assert chunker1 is chunker2
