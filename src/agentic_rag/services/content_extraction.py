"""
Content Extraction Pipeline

This module processes Granite-Docling output to extract structured content,
including text, tables, layout information, and document hierarchy.
It provides content cleaning, normalization, and enrichment capabilities.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentic_rag.services.docling_client import ParsedContent, ParsedTable, ParseResponse

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of extracted content."""
    TEXT = "text"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    IMAGE = "image"
    FOOTER = "footer"
    HEADER = "header"
    CAPTION = "caption"
    FORMULA = "formula"


class LayoutElement(BaseModel):
    """Represents a layout element in the document."""
    
    id: str = Field(..., description="Unique identifier for the element")
    type: ContentType = Field(..., description="Type of content element")
    content: str = Field(..., description="Text content of the element")
    page_number: int = Field(..., description="Page number where element appears")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence: Optional[float] = Field(None, description="Confidence score for the extraction")
    parent_id: Optional[str] = Field(None, description="ID of parent element in hierarchy")
    children_ids: List[str] = Field(default_factory=list, description="IDs of child elements")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class ExtractedTable(BaseModel):
    """Represents an extracted table with structure."""
    
    id: str = Field(..., description="Unique identifier for the table")
    page_number: int = Field(..., description="Page number where table appears")
    headers: Optional[List[str]] = Field(None, description="Table column headers")
    rows: List[List[str]] = Field(..., description="Table rows and cells")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates")
    caption: Optional[str] = Field(None, description="Table caption if available")
    row_count: int = Field(..., description="Number of data rows")
    column_count: int = Field(..., description="Number of columns")
    metadata: Dict = Field(default_factory=dict, description="Additional table metadata")


class DocumentStructure(BaseModel):
    """Represents the hierarchical structure of a document."""
    
    title: Optional[str] = Field(None, description="Document title")
    sections: List[Dict] = Field(default_factory=list, description="Document sections hierarchy")
    page_count: int = Field(..., description="Total number of pages")
    language: Optional[str] = Field(None, description="Detected document language")
    document_type: str = Field(..., description="Type of document (pdf, docx, etc.)")


class ExtractedContent(BaseModel):
    """Complete extracted content from a document."""
    
    document_id: str = Field(..., description="Unique document identifier")
    structure: DocumentStructure = Field(..., description="Document structure and hierarchy")
    elements: List[LayoutElement] = Field(..., description="All layout elements")
    tables: List[ExtractedTable] = Field(..., description="Extracted tables")
    text_content: str = Field(..., description="Full text content")
    processing_metadata: Dict = Field(default_factory=dict, description="Processing metadata")


class ContentExtractionPipeline:
    """Pipeline for extracting and processing content from Granite-Docling output."""
    
    def __init__(self):
        self.text_cleaners = [
            self._remove_excessive_whitespace,
            self._normalize_line_breaks,
            self._fix_encoding_issues,
            self._remove_control_characters
        ]
        
        logger.info("Content extraction pipeline initialized")
    
    def extract_content(self, parse_response: ParseResponse, document_id: Optional[str] = None) -> ExtractedContent:
        """
        Extract structured content from Granite-Docling parse response.
        
        Args:
            parse_response: The response from Granite-Docling parsing
            document_id: Optional document identifier
            
        Returns:
            ExtractedContent: Structured extracted content
        """
        if document_id is None:
            document_id = str(uuid4())
        
        logger.info(f"Starting content extraction for document {document_id}")
        
        # Extract document structure
        structure = self._extract_document_structure(parse_response)
        
        # Process layout elements
        elements = self._process_layout_elements(parse_response.content)
        
        # Process tables
        tables = self._process_tables(parse_response.tables)
        
        # Extract and clean full text content
        text_content = self._extract_full_text(elements)
        
        # Build processing metadata
        processing_metadata = {
            "processing_time": parse_response.processing_time,
            "pages_processed": parse_response.pages_processed,
            "elements_count": len(elements),
            "tables_count": len(tables),
            "success": parse_response.success,
            "original_document_type": parse_response.document_type
        }
        
        extracted_content = ExtractedContent(
            document_id=document_id,
            structure=structure,
            elements=elements,
            tables=tables,
            text_content=text_content,
            processing_metadata=processing_metadata
        )
        
        logger.info(
            f"Content extraction completed for document {document_id}",
            extra={
                "elements_extracted": len(elements),
                "tables_extracted": len(tables),
                "text_length": len(text_content),
                "pages": structure.page_count
            }
        )
        
        return extracted_content
    
    def _extract_document_structure(self, parse_response: ParseResponse) -> DocumentStructure:
        """Extract document structure and hierarchy."""
        return DocumentStructure(
            title=parse_response.metadata.title,
            sections=[],  # Will be populated by analyzing headings
            page_count=parse_response.metadata.page_count,
            language=parse_response.metadata.language,
            document_type=parse_response.document_type
        )
    
    def _process_layout_elements(self, content_blocks: List[ParsedContent]) -> List[LayoutElement]:
        """Process and classify layout elements."""
        elements = []
        
        for i, block in enumerate(content_blocks):
            # Determine content type based on content analysis
            content_type = self._classify_content_type(block.text, block.content_type)
            
            # Clean and normalize text content
            cleaned_content = self._clean_text(block.text)
            
            # Create layout element
            element = LayoutElement(
                id=f"element_{i}",
                type=content_type,
                content=cleaned_content,
                page_number=block.page_number,
                bbox=block.bbox,
                confidence=block.confidence,
                metadata={
                    "original_type": block.content_type,
                    "character_count": len(cleaned_content),
                    "word_count": len(cleaned_content.split())
                }
            )
            
            elements.append(element)
        
        # Build hierarchy relationships
        self._build_element_hierarchy(elements)
        
        return elements
    
    def _process_tables(self, parsed_tables: List[ParsedTable]) -> List[ExtractedTable]:
        """Process and structure extracted tables."""
        tables = []
        
        for i, table in enumerate(parsed_tables):
            # Clean table content
            cleaned_rows = []
            for row in table.rows:
                cleaned_row = [self._clean_text(cell) for cell in row]
                cleaned_rows.append(cleaned_row)
            
            cleaned_headers = None
            if table.headers:
                cleaned_headers = [self._clean_text(header) for header in table.headers]
            
            # Calculate table dimensions
            row_count = len(cleaned_rows)
            column_count = len(cleaned_rows[0]) if cleaned_rows else 0
            
            extracted_table = ExtractedTable(
                id=f"table_{i}",
                page_number=table.page_number,
                headers=cleaned_headers,
                rows=cleaned_rows,
                bbox=table.bbox,
                row_count=row_count,
                column_count=column_count,
                metadata={
                    "has_headers": cleaned_headers is not None,
                    "total_cells": row_count * column_count,
                    "empty_cells": self._count_empty_cells(cleaned_rows)
                }
            )
            
            tables.append(extracted_table)
        
        return tables
    
    def _classify_content_type(self, text: str, original_type: str) -> ContentType:
        """Classify content type based on text analysis."""
        text_stripped = text.strip()
        
        # Check for headings (simple heuristics)
        if self._is_heading(text_stripped):
            return ContentType.HEADING
        
        # Check for lists
        if self._is_list_item(text_stripped):
            return ContentType.LIST
        
        # Check for captions
        if self._is_caption(text_stripped):
            return ContentType.CAPTION
        
        # Check for headers/footers (based on position and content)
        if self._is_header_footer(text_stripped):
            return ContentType.FOOTER
        
        # Default to paragraph for regular text
        if original_type == "text":
            return ContentType.PARAGRAPH
        
        # Map original types
        type_mapping = {
            "table": ContentType.TABLE,
            "image": ContentType.IMAGE,
            "text": ContentType.TEXT
        }
        
        return type_mapping.get(original_type, ContentType.TEXT)
    
    def _is_heading(self, text: str) -> bool:
        """Determine if text is likely a heading."""
        # Simple heuristics for heading detection
        if len(text) > 200:  # Too long to be a heading
            return False
        
        # Check for heading patterns
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS
            r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*$',  # Title Case
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_list_item(self, text: str) -> bool:
        """Determine if text is a list item."""
        list_patterns = [
            r'^\s*[-•*]\s+',  # Bullet points
            r'^\s*\d+\.?\s+',  # Numbered lists
            r'^\s*[a-zA-Z]\.?\s+',  # Lettered lists
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_caption(self, text: str) -> bool:
        """Determine if text is a caption."""
        caption_patterns = [
            r'^(Figure|Table|Chart|Image)\s+\d+',
            r'^(Fig\.|Tab\.)\s+\d+',
        ]
        
        for pattern in caption_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_header_footer(self, text: str) -> bool:
        """Determine if text is header or footer content."""
        # Simple heuristics for headers/footers
        if len(text) < 5:  # Too short
            return False
        
        # Check for page numbers
        if re.match(r'^\s*\d+\s*$', text):
            return True
        
        # Check for common header/footer patterns
        header_footer_patterns = [
            r'page\s+\d+',
            r'\d+\s+of\s+\d+',
            r'copyright',
            r'confidential',
        ]
        
        for pattern in header_footer_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _build_element_hierarchy(self, elements: List[LayoutElement]):
        """Build hierarchical relationships between elements."""
        # Simple hierarchy building based on headings and proximity
        current_parent = None
        
        for element in elements:
            if element.type == ContentType.HEADING:
                current_parent = element.id
            elif current_parent and element.type in [ContentType.PARAGRAPH, ContentType.LIST]:
                element.parent_id = current_parent
                # Add to parent's children
                for parent_element in elements:
                    if parent_element.id == current_parent:
                        parent_element.children_ids.append(element.id)
                        break
    
    def _extract_full_text(self, elements: List[LayoutElement]) -> str:
        """Extract full text content from all elements."""
        text_parts = []
        
        for element in elements:
            if element.type in [ContentType.TEXT, ContentType.PARAGRAPH, ContentType.HEADING, ContentType.LIST]:
                text_parts.append(element.content)
        
        return "\n\n".join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Apply all text cleaning operations."""
        cleaned = text
        
        for cleaner in self.text_cleaners:
            cleaned = cleaner(cleaned)
        
        return cleaned.strip()
    
    def _remove_excessive_whitespace(self, text: str) -> str:
        """Remove excessive whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text
    
    def _normalize_line_breaks(self, text: str) -> str:
        """Normalize line breaks."""
        # Convert Windows/Mac line endings to Unix
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues."""
        # Fix common Unicode issues
        replacements = {
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Horizontal ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters."""
        # Remove control characters except newlines and tabs
        return ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    def _count_empty_cells(self, rows: List[List[str]]) -> int:
        """Count empty cells in table rows."""
        empty_count = 0
        for row in rows:
            for cell in row:
                if not cell.strip():
                    empty_count += 1
        return empty_count


# Global pipeline instance
_content_pipeline: Optional[ContentExtractionPipeline] = None


def get_content_extraction_pipeline() -> ContentExtractionPipeline:
    """Get or create the global content extraction pipeline instance."""
    global _content_pipeline
    
    if _content_pipeline is None:
        _content_pipeline = ContentExtractionPipeline()
    
    return _content_pipeline
