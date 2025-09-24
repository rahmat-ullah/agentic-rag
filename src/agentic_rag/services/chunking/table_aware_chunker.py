"""
Table-Aware Text Chunking

This module implements intelligent chunking that detects and handles tables
and structured content specially, preserving table structure and relationships.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rag.services.chunking.section_aware_chunker import (
    SectionAwareChunker, SectionAwareChunk, SectionMetadata
)
from agentic_rag.services.chunking.basic_chunker import ChunkingConfig, ChunkMetadata
from agentic_rag.services.content_extraction import (
    ExtractedContent, LayoutElement, ContentType, DocumentStructure, ExtractedTable
)

logger = logging.getLogger(__name__)


@dataclass
class TableInfo:
    """Information about a detected table."""
    
    table_id: str
    start_element_index: int
    end_element_index: int
    row_count: int
    column_count: int
    has_header: bool = False
    table_title: Optional[str] = None
    table_caption: Optional[str] = None
    table_data: Optional[ExtractedTable] = None


class TableMetadata(BaseModel):
    """Extended metadata for table-aware chunks."""
    
    contains_table: bool = Field(default=False, description="Whether chunk contains table content")
    table_id: Optional[str] = Field(None, description="ID of the table if chunk contains table")
    table_position: Optional[str] = Field(None, description="Position in table: header, body, footer")
    table_rows: List[int] = Field(default_factory=list, description="Row indices included in chunk")
    table_columns: List[int] = Field(default_factory=list, description="Column indices included in chunk")
    is_table_complete: bool = Field(default=False, description="Whether chunk contains complete table")
    table_title: Optional[str] = Field(None, description="Title of the table")
    table_caption: Optional[str] = Field(None, description="Caption of the table")
    related_tables: List[str] = Field(default_factory=list, description="IDs of related tables")


class TableAwareChunk(SectionAwareChunk):
    """Text chunk with table awareness."""
    
    table_metadata: TableMetadata = Field(..., description="Table-specific metadata")
    
    def get_table_context(self) -> str:
        """Get chunk content with table context."""
        context_parts = []
        
        # Add table title if present
        if self.table_metadata.table_title:
            context_parts.append(f"Table: {self.table_metadata.table_title}")
        
        # Add table position context
        if self.table_metadata.table_position:
            context_parts.append(f"Table Position: {self.table_metadata.table_position}")
        
        # Add section context
        if self.section_metadata.section_path:
            context_parts.append("Section: " + " > ".join(self.section_metadata.section_path))
        
        # Add the chunk content
        context_parts.append(self.content)
        
        return "\n\n".join(context_parts)


class TableDetector:
    """Detects tables and structured content in document elements."""
    
    def __init__(self):
        self.table_indicators = {
            ContentType.TABLE,
            # Add other table-related content types as needed
        }
    
    def detect_tables(self, extracted_content: ExtractedContent) -> List[TableInfo]:
        """
        Detect tables in the document content.
        
        Args:
            extracted_content: The extracted content from document parsing
            
        Returns:
            List of table information objects
        """
        logger.info(f"Detecting tables in document {extracted_content.document_id}")
        
        tables = []
        current_table = None
        
        # First, process explicit table elements
        for i, element in enumerate(extracted_content.elements):
            if element.type == ContentType.TABLE:
                if current_table is None:
                    # Start new table
                    table_id = str(uuid4())
                    current_table = TableInfo(
                        table_id=table_id,
                        start_element_index=i,
                        end_element_index=i,
                        row_count=0,
                        column_count=0
                    )
                else:
                    # Extend current table
                    current_table.end_element_index = i
                
                # Extract table information from element
                self._extract_table_info_from_element(current_table, element)
            else:
                # End current table if we hit non-table content
                if current_table is not None:
                    tables.append(current_table)
                    current_table = None
        
        # Don't forget the last table
        if current_table is not None:
            tables.append(current_table)
        
        # Process tables from extracted table data
        for table_data in extracted_content.tables:
            table_info = self._create_table_info_from_data(table_data, extracted_content.elements)
            if table_info:
                tables.append(table_info)
        
        # Detect implicit tables (structured text that looks like tables)
        implicit_tables = self._detect_implicit_tables(extracted_content.elements)
        tables.extend(implicit_tables)
        
        logger.info(f"Detected {len(tables)} tables in document")
        return tables
    
    def _extract_table_info_from_element(self, table_info: TableInfo, element: LayoutElement):
        """Extract table information from a table element."""
        content = element.content.strip()
        
        # Simple heuristics to estimate table dimensions
        lines = content.split('\n')
        table_info.row_count = max(table_info.row_count, len(lines))
        
        # Estimate columns by looking for common separators
        for line in lines:
            if '|' in line:
                cols = len(line.split('|'))
            elif '\t' in line:
                cols = len(line.split('\t'))
            else:
                cols = len(line.split())
            
            table_info.column_count = max(table_info.column_count, cols)
        
        # Check for header indicators
        if any(indicator in content.lower() for indicator in ['header', 'column', 'field']):
            table_info.has_header = True
    
    def _create_table_info_from_data(self, table_data: ExtractedTable, elements: List[LayoutElement]) -> Optional[TableInfo]:
        """Create table info from extracted table data."""
        if not table_data.rows:
            return None
        
        # Find corresponding elements
        start_idx = None
        end_idx = None
        
        for i, element in enumerate(elements):
            if element.type == ContentType.TABLE and table_data.id in element.id:
                if start_idx is None:
                    start_idx = i
                end_idx = i
        
        if start_idx is None:
            start_idx = 0
            end_idx = 0
        
        return TableInfo(
            table_id=table_data.id,
            start_element_index=start_idx,
            end_element_index=end_idx,
            row_count=table_data.row_count,
            column_count=table_data.column_count,
            has_header=bool(table_data.headers),
            table_title=table_data.caption,  # Use caption as title
            table_caption=table_data.caption,
            table_data=table_data
        )
    
    def _detect_implicit_tables(self, elements: List[LayoutElement]) -> List[TableInfo]:
        """Detect implicit tables from structured text patterns."""
        implicit_tables = []
        
        # Look for patterns that suggest tabular data
        for i, element in enumerate(elements):
            if element.type == ContentType.PARAGRAPH:
                content = element.content.strip()
                
                # Check for table-like patterns
                if self._looks_like_table(content):
                    table_id = str(uuid4())
                    table_info = TableInfo(
                        table_id=table_id,
                        start_element_index=i,
                        end_element_index=i,
                        row_count=len(content.split('\n')),
                        column_count=self._estimate_columns(content)
                    )
                    implicit_tables.append(table_info)
        
        return implicit_tables
    
    def _looks_like_table(self, content: str) -> bool:
        """Check if content looks like a table."""
        lines = content.split('\n')
        
        if len(lines) < 2:
            return False
        
        # Check for consistent separators
        separator_counts = []
        for line in lines:
            if '|' in line:
                separator_counts.append(line.count('|'))
            elif '\t' in line:
                separator_counts.append(line.count('\t'))
            else:
                # Check for multiple spaces (potential column separator)
                import re
                spaces = len(re.findall(r'\s{2,}', line))
                separator_counts.append(spaces)
        
        # If most lines have similar separator counts, it's likely a table
        if len(set(separator_counts)) <= 2 and max(separator_counts) >= 1:
            return True
        
        return False
    
    def _estimate_columns(self, content: str) -> int:
        """Estimate number of columns in table-like content."""
        lines = content.split('\n')
        max_cols = 0
        
        for line in lines:
            if '|' in line:
                cols = len(line.split('|'))
            elif '\t' in line:
                cols = len(line.split('\t'))
            else:
                import re
                cols = len(re.split(r'\s{2,}', line))
            
            max_cols = max(max_cols, cols)
        
        return max_cols


class TableAwareChunker:
    """Chunker that detects and handles tables specially."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.section_aware_chunker = SectionAwareChunker(self.config)
        self.table_detector = TableDetector()
        
        logger.info("Table-aware chunker initialized")
    
    def chunk_document(self, extracted_content: ExtractedContent) -> List[TableAwareChunk]:
        """
        Chunk document while detecting and handling tables specially.
        
        Args:
            extracted_content: The extracted content from document parsing
            
        Returns:
            List of table-aware chunks
        """
        logger.info(f"Starting table-aware chunking for document {extracted_content.document_id}")
        
        # Detect tables first
        tables = self.table_detector.detect_tables(extracted_content)
        
        # Get section-aware chunks as base
        section_chunks = self.section_aware_chunker.chunk_document(extracted_content)
        
        # Convert to table-aware chunks and add table metadata
        table_aware_chunks = []
        
        for chunk in section_chunks:
            # Determine if chunk contains table content
            table_metadata = self._analyze_chunk_for_tables(chunk, tables, extracted_content)
            
            # Create table-aware chunk
            table_chunk = TableAwareChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                section_metadata=chunk.section_metadata,
                table_metadata=table_metadata
            )
            
            # Update chunk metadata with table context
            if table_metadata.contains_table:
                table_chunk.metadata.processing_metadata.update({
                    "table_aware": True,
                    "table_id": table_metadata.table_id,
                    "contains_table": True
                })
            
            table_aware_chunks.append(table_chunk)
        
        # Post-process table chunks for better handling
        table_aware_chunks = self._post_process_table_chunks(table_aware_chunks, tables)
        
        logger.info(f"Created {len(table_aware_chunks)} table-aware chunks")
        return table_aware_chunks
    
    def _analyze_chunk_for_tables(
        self,
        chunk: SectionAwareChunk,
        tables: List[TableInfo],
        extracted_content: ExtractedContent
    ) -> TableMetadata:
        """Analyze a chunk to determine table content."""
        table_metadata = TableMetadata()

        # Check if chunk content contains table indicators
        content = chunk.content

        # Handle escaped newlines
        if '\\n' in content:
            content = content.replace('\\n', '\n')

        # Look for table-like patterns in content
        has_pipe_separators = '|' in content and content.count('|') >= 2
        has_tab_separators = '\t' in content
        has_multiple_lines = len(content.split('\n')) >= 2

        # Check for table patterns
        if has_pipe_separators or has_tab_separators:
            if has_multiple_lines:
                table_metadata.contains_table = True
                table_metadata.table_id = str(uuid4())

                # Analyze table structure
                lines = content.split('\n')
                table_metadata.table_rows = list(range(len(lines)))

                # Estimate columns
                max_cols = 0
                for line in lines:
                    if '|' in line:
                        cols = len([col.strip() for col in line.split('|') if col.strip()])
                    elif '\t' in line:
                        cols = len(line.split('\t'))
                    else:
                        cols = len(line.split())
                    max_cols = max(max_cols, cols)

                table_metadata.table_columns = list(range(max_cols))
                table_metadata.is_table_complete = len(lines) >= 2 and max_cols >= 2

        # Check against detected tables
        for table in tables:
            # If chunk contains table title or matches table content
            if table.table_title and table.table_title.lower() in content.lower():
                table_metadata.contains_table = True
                table_metadata.table_id = table.table_id
                table_metadata.table_title = table.table_title
                table_metadata.table_caption = table.table_caption
                break

        return table_metadata
    
    def _post_process_table_chunks(
        self, 
        chunks: List[TableAwareChunk], 
        tables: List[TableInfo]
    ) -> List[TableAwareChunk]:
        """Post-process chunks to improve table handling."""
        # Group consecutive table chunks
        processed_chunks = []
        current_table_group = []
        
        for chunk in chunks:
            if chunk.table_metadata.contains_table:
                current_table_group.append(chunk)
            else:
                # Process any accumulated table group
                if current_table_group:
                    processed_chunks.extend(self._merge_table_chunks_if_needed(current_table_group))
                    current_table_group = []
                
                processed_chunks.append(chunk)
        
        # Don't forget the last group
        if current_table_group:
            processed_chunks.extend(self._merge_table_chunks_if_needed(current_table_group))
        
        return processed_chunks
    
    def _merge_table_chunks_if_needed(self, table_chunks: List[TableAwareChunk]) -> List[TableAwareChunk]:
        """Merge table chunks if they belong to the same table and are small."""
        if len(table_chunks) <= 1:
            return table_chunks
        
        # Check if all chunks belong to the same table
        table_ids = set(chunk.table_metadata.table_id for chunk in table_chunks if chunk.table_metadata.table_id)
        
        if len(table_ids) == 1 and all(len(chunk.content) < self.config.chunk_size // 2 for chunk in table_chunks):
            # Merge small chunks from the same table
            merged_content = '\n'.join(chunk.content for chunk in table_chunks)
            
            if len(merged_content) <= self.config.chunk_size * 1.5:  # Allow some flexibility
                # Create merged chunk
                first_chunk = table_chunks[0]
                merged_chunk = TableAwareChunk(
                    content=merged_content,
                    metadata=first_chunk.metadata,
                    section_metadata=first_chunk.section_metadata,
                    table_metadata=first_chunk.table_metadata
                )
                
                # Update metadata
                merged_chunk.metadata.chunk_size = len(merged_content)
                merged_chunk.metadata.word_count = len(merged_content.split())
                merged_chunk.table_metadata.is_table_complete = True
                
                return [merged_chunk]
        
        return table_chunks
    
    def get_table_summary(self, chunks: List[TableAwareChunk]) -> Dict:
        """Get summary information about tables in the chunks."""
        table_chunks = [chunk for chunk in chunks if chunk.table_metadata.contains_table]
        
        tables_info = {}
        for chunk in table_chunks:
            table_id = chunk.table_metadata.table_id
            
            if table_id and table_id not in tables_info:
                tables_info[table_id] = {
                    "title": chunk.table_metadata.table_title,
                    "chunk_count": 0,
                    "total_rows": len(chunk.table_metadata.table_rows),
                    "total_columns": len(chunk.table_metadata.table_columns),
                    "is_complete": chunk.table_metadata.is_table_complete
                }
            
            if table_id:
                tables_info[table_id]["chunk_count"] += 1
        
        return {
            "total_tables": len(tables_info),
            "total_table_chunks": len(table_chunks),
            "total_chunks": len(chunks),
            "tables": tables_info
        }


# Global table-aware chunker instance
_table_aware_chunker: Optional[TableAwareChunker] = None


def get_table_aware_chunker(config: Optional[ChunkingConfig] = None) -> TableAwareChunker:
    """Get or create the global table-aware chunker instance."""
    global _table_aware_chunker
    
    if _table_aware_chunker is None or config is not None:
        _table_aware_chunker = TableAwareChunker(config)
    
    return _table_aware_chunker
