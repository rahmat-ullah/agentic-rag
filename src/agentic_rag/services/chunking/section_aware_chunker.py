"""
Section-Aware Text Chunking

This module implements intelligent chunking that respects document structure
and preserves section boundaries using the document hierarchy from Granite-Docling parsing.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rag.services.chunking.basic_chunker import (
    BasicTextChunker, ChunkingConfig, ChunkMetadata, TextChunk
)
from agentic_rag.services.content_extraction import (
    ExtractedContent, LayoutElement, ContentType, DocumentStructure
)

logger = logging.getLogger(__name__)


@dataclass
class SectionInfo:
    """Information about a document section."""
    
    section_id: str
    title: str
    level: int  # Heading level (1, 2, 3, etc.)
    start_element_index: int
    end_element_index: int
    parent_section_id: Optional[str] = None
    subsections: List[str] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []


class SectionMetadata(BaseModel):
    """Extended metadata for section-aware chunks."""
    
    section_id: str = Field(..., description="ID of the section this chunk belongs to")
    section_title: str = Field(..., description="Title of the section")
    section_level: int = Field(..., description="Heading level of the section")
    parent_section_id: Optional[str] = Field(None, description="ID of parent section")
    section_path: List[str] = Field(default_factory=list, description="Path from root to this section")
    is_section_start: bool = Field(default=False, description="Whether chunk starts a new section")
    is_section_end: bool = Field(default=False, description="Whether chunk ends a section")
    contains_heading: bool = Field(default=False, description="Whether chunk contains a heading")
    heading_text: Optional[str] = Field(None, description="Text of heading if present")


class SectionAwareChunk(TextChunk):
    """Text chunk with section awareness."""
    
    section_metadata: SectionMetadata = Field(..., description="Section-specific metadata")
    
    def get_full_context(self) -> str:
        """Get chunk content with section context."""
        context_parts = []
        
        # Add section path for context
        if self.section_metadata.section_path:
            context_parts.append(" > ".join(self.section_metadata.section_path))
        
        # Add the chunk content
        context_parts.append(self.content)
        
        return "\n\n".join(context_parts)


class DocumentSectionAnalyzer:
    """Analyzes document structure to identify sections and hierarchy."""
    
    def __init__(self):
        self.heading_patterns = {
            ContentType.HEADING: 1,  # Default heading level
        }
    
    def analyze_document_structure(self, extracted_content: ExtractedContent) -> List[SectionInfo]:
        """
        Analyze document structure to identify sections and their hierarchy.
        
        Args:
            extracted_content: The extracted content from document parsing
            
        Returns:
            List of section information objects
        """
        logger.info(f"Analyzing document structure for {extracted_content.document_id}")
        
        sections = []
        current_section = None
        section_stack = []  # Stack to track nested sections
        
        for i, element in enumerate(extracted_content.elements):
            if element.type == ContentType.HEADING:
                # Determine heading level
                level = self._determine_heading_level(element.content)
                
                # Close previous sections if necessary
                self._close_sections_at_level(sections, section_stack, level)
                
                # Create new section
                section_id = str(uuid4())
                section = SectionInfo(
                    section_id=section_id,
                    title=element.content.strip(),
                    level=level,
                    start_element_index=i,
                    end_element_index=len(extracted_content.elements) - 1,  # Will be updated
                    parent_section_id=section_stack[-1].section_id if section_stack else None
                )
                
                # Update parent section's end index
                if current_section:
                    current_section.end_element_index = i - 1
                
                sections.append(section)
                section_stack.append(section)
                current_section = section
                
                logger.debug(f"Found section: {section.title} (level {level})")
        
        # Handle document without explicit sections
        if not sections:
            # Create a default section for the entire document
            default_section = SectionInfo(
                section_id=str(uuid4()),
                title="Document Content",
                level=1,
                start_element_index=0,
                end_element_index=len(extracted_content.elements) - 1
            )
            sections.append(default_section)
        
        # Build section hierarchy
        self._build_section_hierarchy(sections)
        
        logger.info(f"Identified {len(sections)} sections in document")
        return sections
    
    def _determine_heading_level(self, heading_text: str) -> int:
        """Determine the level of a heading based on its text."""
        # Simple heuristics for heading level detection
        text = heading_text.strip()

        # Check for numbered headings (1., 1.1., 1.1.1., etc.)
        import re
        numbered_match = re.match(r'^(\d+(?:\.\d+)*)', text)
        if numbered_match:
            number_part = numbered_match.group(1)
            dots = number_part.count('.')
            return dots + 1  # 1. = level 1, 1.1 = level 2, etc.

        # Check for all caps (likely higher level)
        if text.isupper() and len(text) > 3:
            return 1

        # Check for length (shorter headings are often higher level)
        if len(text) < 30:
            return 2

        return 3  # Default level
    
    def _close_sections_at_level(self, sections: List[SectionInfo], section_stack: List[SectionInfo], new_level: int):
        """Close sections at or below the specified level."""
        while section_stack and section_stack[-1].level >= new_level:
            closed_section = section_stack.pop()
            # The end index will be set when the next section starts
    
    def _build_section_hierarchy(self, sections: List[SectionInfo]):
        """Build parent-child relationships between sections."""
        for i, section in enumerate(sections):
            # Find parent section (previous section with lower level)
            for j in range(i - 1, -1, -1):
                potential_parent = sections[j]
                if potential_parent.level < section.level:
                    section.parent_section_id = potential_parent.section_id
                    potential_parent.subsections.append(section.section_id)
                    break


class SectionAwareChunker:
    """Chunker that respects document structure and section boundaries."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.basic_chunker = BasicTextChunker(self.config)
        self.section_analyzer = DocumentSectionAnalyzer()
        
        logger.info("Section-aware chunker initialized")
    
    def chunk_document(self, extracted_content: ExtractedContent) -> List[SectionAwareChunk]:
        """
        Chunk document while respecting section boundaries.
        
        Args:
            extracted_content: The extracted content from document parsing
            
        Returns:
            List of section-aware chunks
        """
        logger.info(f"Starting section-aware chunking for document {extracted_content.document_id}")
        
        # Analyze document structure
        sections = self.section_analyzer.analyze_document_structure(extracted_content)
        
        # Create section-aware chunks
        all_chunks = []
        
        for section in sections:
            section_chunks = self._chunk_section(extracted_content, section, sections)
            all_chunks.extend(section_chunks)
        
        # Update chunk indices
        for i, chunk in enumerate(all_chunks):
            chunk.metadata.chunk_index = i
        
        logger.info(f"Created {len(all_chunks)} section-aware chunks")
        return all_chunks
    
    def _chunk_section(
        self, 
        extracted_content: ExtractedContent, 
        section: SectionInfo, 
        all_sections: List[SectionInfo]
    ) -> List[SectionAwareChunk]:
        """Chunk a specific section while preserving its structure."""
        # Extract text content for this section
        section_elements = extracted_content.elements[
            section.start_element_index:section.end_element_index + 1
        ]
        
        section_text = self._build_section_text(section_elements, section)
        
        if not section_text.strip():
            return []
        
        # Use basic chunker for the section text
        basic_chunks = self.basic_chunker.chunk_text(section_text, extracted_content.document_id)
        
        # Convert to section-aware chunks
        section_chunks = []
        section_path = self._build_section_path(section, all_sections)
        
        for i, basic_chunk in enumerate(basic_chunks):
            # Create section metadata
            section_metadata = SectionMetadata(
                section_id=section.section_id,
                section_title=section.title,
                section_level=section.level,
                parent_section_id=section.parent_section_id,
                section_path=section_path,
                is_section_start=(i == 0),
                is_section_end=(i == len(basic_chunks) - 1),
                contains_heading=(i == 0 and section_elements and 
                                section_elements[0].type == ContentType.HEADING),
                heading_text=section.title if i == 0 else None
            )
            
            # Create section-aware chunk
            section_chunk = SectionAwareChunk(
                content=basic_chunk.content,
                metadata=basic_chunk.metadata,
                section_metadata=section_metadata
            )
            
            # Update chunk metadata with section context
            section_chunk.metadata.processing_metadata.update({
                "section_aware": True,
                "section_id": section.section_id,
                "section_title": section.title,
                "section_level": section.level
            })
            
            section_chunks.append(section_chunk)
        
        return section_chunks
    
    def _build_section_text(self, elements: List[LayoutElement], section: SectionInfo) -> str:
        """Build text content for a section from its elements."""
        text_parts = []
        
        for element in elements:
            if element.content.strip():
                # Add appropriate spacing based on element type
                if element.type == ContentType.HEADING:
                    text_parts.append(f"\n\n{element.content}\n")
                elif element.type == ContentType.PARAGRAPH:
                    text_parts.append(f"{element.content}\n")
                elif element.type == ContentType.LIST:
                    text_parts.append(f"{element.content}\n")
                else:
                    text_parts.append(element.content)
        
        return "".join(text_parts).strip()
    
    def _build_section_path(self, section: SectionInfo, all_sections: List[SectionInfo]) -> List[str]:
        """Build the path from root to the current section."""
        path = []
        current_section = section
        
        # Build path by traversing up the hierarchy
        while current_section:
            path.insert(0, current_section.title)
            
            # Find parent section
            parent_section = None
            if current_section.parent_section_id:
                for s in all_sections:
                    if s.section_id == current_section.parent_section_id:
                        parent_section = s
                        break
            
            current_section = parent_section
        
        return path
    
    def get_section_summary(self, chunks: List[SectionAwareChunk]) -> Dict:
        """Get summary information about sections in the chunks."""
        sections_info = {}
        
        for chunk in chunks:
            section_id = chunk.section_metadata.section_id
            
            if section_id not in sections_info:
                sections_info[section_id] = {
                    "title": chunk.section_metadata.section_title,
                    "level": chunk.section_metadata.section_level,
                    "path": chunk.section_metadata.section_path,
                    "chunk_count": 0,
                    "total_characters": 0,
                    "total_words": 0
                }
            
            sections_info[section_id]["chunk_count"] += 1
            sections_info[section_id]["total_characters"] += len(chunk.content)
            sections_info[section_id]["total_words"] += chunk.metadata.word_count
        
        return {
            "total_sections": len(sections_info),
            "sections": sections_info,
            "total_chunks": len(chunks)
        }


# Global section-aware chunker instance
_section_aware_chunker: Optional[SectionAwareChunker] = None


def get_section_aware_chunker(config: Optional[ChunkingConfig] = None) -> SectionAwareChunker:
    """Get or create the global section-aware chunker instance."""
    global _section_aware_chunker
    
    if _section_aware_chunker is None or config is not None:
        _section_aware_chunker = SectionAwareChunker(config)
    
    return _section_aware_chunker
