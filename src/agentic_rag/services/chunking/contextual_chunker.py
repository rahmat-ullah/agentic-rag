"""
Contextual Chunking Implementation

This module provides contextual chunking capabilities that include surrounding context
for more accurate and meaningful search results. It extracts local and global context
to enhance chunk understanding while maintaining semantic coherence.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rag.services.chunking.basic_chunker import ChunkingConfig, TextChunk, ChunkMetadata
from agentic_rag.services.chunking.section_aware_chunker import SectionInfo, DocumentSectionAnalyzer
from agentic_rag.services.content_extraction import ExtractedContent, LayoutElement, ContentType

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context that can be extracted."""
    
    PREVIOUS_SPAN = "previous_span"
    NEXT_SPAN = "next_span"
    SIBLING_HEADING = "sibling_heading"
    PARENT_SECTION = "parent_section"
    DOCUMENT_TITLE = "document_title"
    SECTION_TRAIL = "section_trail"
    KEY_DEFINITION = "key_definition"


@dataclass
class ContextElement:
    """Represents a single context element."""
    
    type: ContextType
    content: str
    relevance_score: float = 1.0
    source_element_id: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LocalContext(BaseModel):
    """Local context extracted around a chunk."""
    
    prev_spans: List[ContextElement] = Field(default_factory=list, description="Previous text spans")
    next_spans: List[ContextElement] = Field(default_factory=list, description="Next text spans")
    sibling_headings: List[ContextElement] = Field(default_factory=list, description="Sibling section headings")
    parent_context: Optional[ContextElement] = Field(None, description="Parent section context")
    window_size: int = Field(default=2, description="Context window size used")


class GlobalContext(BaseModel):
    """Global context extracted from document."""
    
    title: Optional[ContextElement] = Field(None, description="Document title")
    section_trail: List[ContextElement] = Field(default_factory=list, description="Hierarchical section path")
    doc_type: Optional[str] = Field(None, description="Document classification")
    key_definitions: List[ContextElement] = Field(default_factory=list, description="Relevant definitions")
    document_metadata: Dict = Field(default_factory=dict, description="Additional document metadata")


class ContextualMetadata(BaseModel):
    """Extended metadata for contextual chunks."""
    
    local_context: LocalContext = Field(..., description="Local context information")
    global_context: GlobalContext = Field(..., description="Global context information")
    context_token_count: int = Field(default=0, description="Total tokens in contextual text")
    context_quality_score: float = Field(default=0.0, description="Quality score of context extraction")
    fusion_strategy: str = Field(default="standard", description="Strategy used for context fusion")


class ContextualChunk(TextChunk):
    """Text chunk with contextual information."""
    
    contextual_text: str = Field(..., description="Context-enhanced text for embeddings")
    original_text: str = Field(..., description="Original chunk text for citations")
    contextual_metadata: ContextualMetadata = Field(..., description="Contextual metadata")
    
    def get_embedding_text(self) -> str:
        """Get the text to use for embedding generation."""
        return self.contextual_text
    
    def get_citation_text(self) -> str:
        """Get the text to use for citations."""
        return self.original_text


@dataclass
class ContextExtractionConfig:
    """Configuration for contextual chunking."""
    
    # Local context settings
    local_window_size: int = 2
    max_local_context_chars: int = 1000
    include_sibling_headings: bool = True
    include_parent_context: bool = True
    
    # Global context settings
    include_document_title: bool = True
    include_section_trail: bool = True
    max_section_trail_depth: int = 5
    include_key_definitions: bool = True
    max_key_definitions: int = 3
    
    # Context fusion settings
    max_context_tokens: int = 1024
    core_text_priority: float = 1.0
    section_trail_priority: float = 0.8
    title_priority: float = 0.7
    neighbor_priority: float = 0.6
    sibling_priority: float = 0.5
    definition_priority: float = 0.4
    
    # Performance settings
    enable_caching: bool = True
    cache_global_context: bool = True
    parallel_processing: bool = True


class LocalContextExtractor:
    """Extracts local context around document spans."""
    
    def __init__(self, config: ContextExtractionConfig):
        self.config = config
        logger.info(f"Local context extractor initialized with window_size={config.local_window_size}")
    
    def collect_neighbors(
        self, 
        span_index: int, 
        elements: List[LayoutElement], 
        window_size: Optional[int] = None
    ) -> Dict[str, List[ContextElement]]:
        """
        Collect neighboring spans around the current span.
        
        Args:
            span_index: Index of current span in elements list
            elements: List of all layout elements
            window_size: Number of spans to include before/after (defaults to config)
        
        Returns:
            dict: {
                'prev_spans': List of previous spans,
                'next_spans': List of next spans
            }
        """
        window_size = window_size or self.config.local_window_size
        
        prev_spans = []
        next_spans = []
        
        # Collect previous spans
        start_idx = max(0, span_index - window_size)
        for i in range(start_idx, span_index):
            element = elements[i]
            if self._is_valid_context_element(element):
                context_element = ContextElement(
                    type=ContextType.PREVIOUS_SPAN,
                    content=element.content.strip(),
                    relevance_score=self._calculate_proximity_score(span_index, i),
                    source_element_id=element.id,
                    metadata={
                        "element_type": element.type.value,
                        "page_number": element.page_number,
                        "distance": span_index - i
                    }
                )
                prev_spans.append(context_element)
        
        # Collect next spans
        end_idx = min(len(elements), span_index + window_size + 1)
        for i in range(span_index + 1, end_idx):
            element = elements[i]
            if self._is_valid_context_element(element):
                context_element = ContextElement(
                    type=ContextType.NEXT_SPAN,
                    content=element.content.strip(),
                    relevance_score=self._calculate_proximity_score(span_index, i),
                    source_element_id=element.id,
                    metadata={
                        "element_type": element.type.value,
                        "page_number": element.page_number,
                        "distance": i - span_index
                    }
                )
                next_spans.append(context_element)
        
        logger.debug(f"Collected {len(prev_spans)} previous and {len(next_spans)} next spans for element {span_index}")
        
        return {
            'prev_spans': prev_spans,
            'next_spans': next_spans
        }
    
    def extract_sibling_headings(
        self, 
        current_section: SectionInfo, 
        all_sections: List[SectionInfo]
    ) -> List[ContextElement]:
        """Extract headings at the same hierarchical level."""
        if not self.config.include_sibling_headings:
            return []
        
        sibling_headings = []
        
        for section in all_sections:
            # Check if it's a sibling (same level, same parent)
            if (section.level == current_section.level and 
                section.parent_section_id == current_section.parent_section_id and
                section.section_id != current_section.section_id):
                
                context_element = ContextElement(
                    type=ContextType.SIBLING_HEADING,
                    content=section.title,
                    relevance_score=0.8,  # High relevance for structural context
                    source_element_id=section.section_id,
                    metadata={
                        "section_level": section.level,
                        "section_id": section.section_id
                    }
                )
                sibling_headings.append(context_element)
        
        logger.debug(f"Found {len(sibling_headings)} sibling headings for section {current_section.title}")
        return sibling_headings
    
    def _is_valid_context_element(self, element: LayoutElement) -> bool:
        """Check if an element is valid for context extraction."""
        # Skip empty content
        if not element.content.strip():
            return False
        
        # Include text and headings, exclude images and other non-textual content
        valid_types = {ContentType.TEXT, ContentType.HEADING, ContentType.PARAGRAPH, ContentType.LIST}
        return element.type in valid_types
    
    def _calculate_proximity_score(self, center_idx: int, target_idx: int) -> float:
        """Calculate relevance score based on proximity."""
        distance = abs(center_idx - target_idx)
        # Closer elements get higher scores
        return max(0.1, 1.0 - (distance * 0.2))


class GlobalContextExtractor:
    """Extracts global context from document structure."""

    def __init__(self, config: ContextExtractionConfig):
        self.config = config
        self._global_context_cache: Dict[str, GlobalContext] = {}

        # Initialize performance optimizer if available
        self._performance_optimizer = None
        try:
            from agentic_rag.services.chunking.contextual_performance import get_performance_optimizer
            self._performance_optimizer = get_performance_optimizer()
        except ImportError:
            logger.warning("Performance optimizer not available")

        logger.info("Global context extractor initialized")

    def collect_global_context(
        self,
        extracted_content: ExtractedContent,
        current_section: Optional[SectionInfo] = None
    ) -> GlobalContext:
        """
        Collect global document context.

        Args:
            extracted_content: Parsed document structure
            current_section: Current section being processed

        Returns:
            GlobalContext with document-level information
        """
        # Check performance optimizer cache first
        cache_key = extracted_content.document_id
        if self._performance_optimizer and self._performance_optimizer.enable_caching:
            cached_context = self._performance_optimizer.context_cache.get_global_context(cache_key)
            if cached_context:
                self._performance_optimizer.metrics.cache_hits += 1
                return cached_context
            else:
                self._performance_optimizer.metrics.cache_misses += 1

        # Check local cache as fallback
        if self.config.cache_global_context and cache_key in self._global_context_cache:
            cached_context = self._global_context_cache[cache_key]
            # Update section trail for current section
            if current_section:
                cached_context.section_trail = self._build_section_trail(current_section, extracted_content)
            return cached_context

        global_context = GlobalContext()

        # Extract document title
        if self.config.include_document_title and extracted_content.structure.title:
            global_context.title = ContextElement(
                type=ContextType.DOCUMENT_TITLE,
                content=extracted_content.structure.title,
                relevance_score=self.config.title_priority,
                metadata={"source": "document_structure"}
            )

        # Build section trail for current section
        if self.config.include_section_trail and current_section:
            global_context.section_trail = self._build_section_trail(current_section, extracted_content)

        # Extract document type
        global_context.doc_type = extracted_content.structure.document_type

        # Extract key definitions
        if self.config.include_key_definitions:
            global_context.key_definitions = self._extract_key_definitions(extracted_content)

        # Store document metadata
        global_context.document_metadata = {
            "page_count": extracted_content.structure.page_count,
            "language": extracted_content.structure.language,
            "processing_metadata": extracted_content.processing_metadata
        }

        # Cache the result in both caches
        if self.config.cache_global_context:
            self._global_context_cache[cache_key] = global_context

        if self._performance_optimizer and self._performance_optimizer.enable_caching:
            self._performance_optimizer.context_cache.put_global_context(cache_key, global_context)

        logger.debug(f"Collected global context for document {extracted_content.document_id}")
        return global_context

    def _build_section_trail(
        self,
        current_section: SectionInfo,
        extracted_content: ExtractedContent
    ) -> List[ContextElement]:
        """Build hierarchical section path for current section."""
        if not current_section:
            return []

        # Build section hierarchy by traversing up the parent chain
        section_trail = []
        sections_by_id = {}

        # First, analyze document structure to get all sections
        analyzer = DocumentSectionAnalyzer()
        all_sections = analyzer.analyze_document_structure(extracted_content)

        # Create lookup dictionary
        for section in all_sections:
            sections_by_id[section.section_id] = section

        # Build trail from current section up to root
        current = current_section
        trail_sections = []

        while current and len(trail_sections) < self.config.max_section_trail_depth:
            trail_sections.append(current)
            if current.parent_section_id and current.parent_section_id in sections_by_id:
                current = sections_by_id[current.parent_section_id]
            else:
                break

        # Reverse to get root-to-current order
        trail_sections.reverse()

        # Convert to context elements
        for i, section in enumerate(trail_sections):
            context_element = ContextElement(
                type=ContextType.SECTION_TRAIL,
                content=section.title,
                relevance_score=self.config.section_trail_priority * (1.0 - i * 0.1),  # Deeper sections get lower scores
                source_element_id=section.section_id,
                metadata={
                    "section_level": section.level,
                    "trail_position": i,
                    "is_current_section": (section.section_id == current_section.section_id)
                }
            )
            section_trail.append(context_element)

        return section_trail

    def _extract_key_definitions(self, extracted_content: ExtractedContent) -> List[ContextElement]:
        """Extract relevant definitions from the document."""
        definitions = []

        # Simple pattern-based definition extraction
        definition_patterns = [
            r'(.+?)\s+(?:is|means|refers to|defined as)\s+(.+?)(?:\.|$)',
            r'(.+?):\s*(.+?)(?:\.|$)',
            r'Definition\s*:\s*(.+?)(?:\.|$)',
        ]

        for element in extracted_content.elements:
            if element.type in {ContentType.TEXT, ContentType.PARAGRAPH, ContentType.LIST}:
                content = element.content.strip()

                for pattern in definition_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        if len(definitions) >= self.config.max_key_definitions:
                            break

                        if len(match.groups()) >= 2:
                            term = match.group(1).strip()
                            definition = match.group(2).strip()
                        else:
                            term = "Definition"
                            definition = match.group(1).strip()

                        # Filter out very short or very long definitions
                        if 10 <= len(definition) <= 200 and len(term) <= 50:
                            context_element = ContextElement(
                                type=ContextType.KEY_DEFINITION,
                                content=f"{term}: {definition}",
                                relevance_score=self.config.definition_priority,
                                source_element_id=element.id,
                                metadata={
                                    "term": term,
                                    "definition": definition,
                                    "page_number": element.page_number
                                }
                            )
                            definitions.append(context_element)

                if len(definitions) >= self.config.max_key_definitions:
                    break

        logger.debug(f"Extracted {len(definitions)} key definitions")
        return definitions


# Singleton instances
_local_context_extractor: Optional[LocalContextExtractor] = None
_global_context_extractor: Optional[GlobalContextExtractor] = None


def get_local_context_extractor(config: Optional[ContextExtractionConfig] = None) -> LocalContextExtractor:
    """Get or create a local context extractor instance."""
    global _local_context_extractor

    if _local_context_extractor is None or config is not None:
        _local_context_extractor = LocalContextExtractor(config or ContextExtractionConfig())

    return _local_context_extractor


def get_global_context_extractor(config: Optional[ContextExtractionConfig] = None) -> GlobalContextExtractor:
    """Get or create a global context extractor instance."""
    global _global_context_extractor

    if _global_context_extractor is None or config is not None:
        _global_context_extractor = GlobalContextExtractor(config or ContextExtractionConfig())

    return _global_context_extractor


class ContextFusionEngine:
    """Fuses local and global context with core text within token limits."""

    def __init__(self, config: ContextExtractionConfig):
        self.config = config
        logger.info(f"Context fusion engine initialized with max_tokens={config.max_context_tokens}")

    def fuse_context(
        self,
        global_ctx: GlobalContext,
        local_ctx: LocalContext,
        core_text: str,
        limit_tokens: Optional[int] = None
    ) -> str:
        """
        Fuse global, local, and core text within token limits.

        Args:
            global_ctx: Global document context
            local_ctx: Local neighboring context
            core_text: Core chunk text
            limit_tokens: Maximum token limit for final text

        Returns:
            str: Fused contextual text for embedding
        """
        limit_tokens = limit_tokens or self.config.max_context_tokens

        # Prioritize context elements
        prioritized_elements = self._prioritize_context_elements(global_ctx, local_ctx, core_text)

        # Build contextual text within token limits
        contextual_text = self._build_contextual_text(prioritized_elements, core_text, limit_tokens)

        logger.debug(f"Fused context: {len(contextual_text)} characters, estimated {self._estimate_tokens(contextual_text)} tokens")

        return contextual_text

    def _prioritize_context_elements(
        self,
        global_ctx: GlobalContext,
        local_ctx: LocalContext,
        core_text: str
    ) -> List[Tuple[ContextElement, float]]:
        """Prioritize context elements by relevance and importance."""
        elements = []

        # Add global context elements
        if global_ctx.title:
            elements.append((global_ctx.title, self.config.title_priority))

        for trail_element in global_ctx.section_trail:
            elements.append((trail_element, self.config.section_trail_priority))

        for definition in global_ctx.key_definitions:
            elements.append((definition, self.config.definition_priority))

        # Add local context elements
        for prev_span in local_ctx.prev_spans:
            elements.append((prev_span, self.config.neighbor_priority * prev_span.relevance_score))

        for next_span in local_ctx.next_spans:
            elements.append((next_span, self.config.neighbor_priority * next_span.relevance_score))

        for sibling in local_ctx.sibling_headings:
            elements.append((sibling, self.config.sibling_priority))

        if local_ctx.parent_context:
            elements.append((local_ctx.parent_context, self.config.sibling_priority))

        # Sort by priority (highest first)
        elements.sort(key=lambda x: x[1], reverse=True)

        return elements

    def _build_contextual_text(
        self,
        prioritized_elements: List[Tuple[ContextElement, float]],
        core_text: str,
        limit_tokens: int
    ) -> str:
        """Build contextual text within token limits."""
        # Start with core text (always included)
        parts = [f"Content: {core_text}"]
        current_tokens = self._estimate_tokens(parts[0])

        # Add context elements in priority order
        for element, priority in prioritized_elements:
            element_text = self._format_context_element(element)
            element_tokens = self._estimate_tokens(element_text)

            # Check if adding this element would exceed the limit
            if current_tokens + element_tokens <= limit_tokens:
                parts.append(element_text)
                current_tokens += element_tokens
            else:
                # Try to truncate the element to fit
                available_tokens = limit_tokens - current_tokens
                if available_tokens > 20:  # Minimum useful size
                    truncated_text = self._truncate_to_token_limit(element_text, available_tokens)
                    if truncated_text:
                        parts.append(truncated_text)
                        break
                else:
                    break

        return "\n\n".join(parts)

    def _format_context_element(self, element: ContextElement) -> str:
        """Format a context element for inclusion in contextual text."""
        if element.type == ContextType.DOCUMENT_TITLE:
            return f"Document: {element.content}"
        elif element.type == ContextType.SECTION_TRAIL:
            return f"Section: {element.content}"
        elif element.type == ContextType.KEY_DEFINITION:
            return f"Definition: {element.content}"
        elif element.type == ContextType.SIBLING_HEADING:
            return f"Related Section: {element.content}"
        elif element.type == ContextType.PREVIOUS_SPAN:
            return f"Previous: {element.content}"
        elif element.type == ContextType.NEXT_SPAN:
            return f"Following: {element.content}"
        elif element.type == ContextType.PARENT_SECTION:
            return f"Parent Section: {element.content}"
        else:
            return f"Context: {element.content}"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token for English text
        return max(1, len(text) // 4)

    def _truncate_to_token_limit(self, text: str, limit_tokens: int) -> str:
        """Intelligently truncate text to fit token limit."""
        if self._estimate_tokens(text) <= limit_tokens:
            return text

        # Calculate approximate character limit
        char_limit = limit_tokens * 4

        if len(text) <= char_limit:
            return text

        # Try to truncate at sentence boundaries
        sentences = text.split('. ')
        truncated = ""

        for sentence in sentences:
            test_text = truncated + sentence + ". " if truncated else sentence + ". "
            if self._estimate_tokens(test_text) <= limit_tokens:
                truncated = test_text
            else:
                break

        # If no complete sentences fit, truncate at word boundaries
        if not truncated:
            words = text.split()
            truncated = ""

            for word in words:
                test_text = truncated + " " + word if truncated else word
                if self._estimate_tokens(test_text) <= limit_tokens:
                    truncated = test_text
                else:
                    break

        # Add ellipsis if truncated
        if truncated and len(truncated) < len(text):
            truncated = truncated.rstrip() + "..."

        return truncated or text[:char_limit] + "..."


# Singleton instance
_context_fusion_engine: Optional[ContextFusionEngine] = None


def get_context_fusion_engine(config: Optional[ContextExtractionConfig] = None) -> ContextFusionEngine:
    """Get or create a context fusion engine instance."""
    global _context_fusion_engine

    if _context_fusion_engine is None or config is not None:
        _context_fusion_engine = ContextFusionEngine(config or ContextExtractionConfig())

    return _context_fusion_engine


class ContextualChunker:
    """Main contextual chunking processor that orchestrates context extraction and fusion."""

    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        context_config: Optional[ContextExtractionConfig] = None
    ):
        self.chunking_config = chunking_config or ChunkingConfig()
        self.context_config = context_config or ContextExtractionConfig()

        # Initialize performance optimizer
        self._performance_optimizer = None
        try:
            from agentic_rag.services.chunking.contextual_performance import get_performance_optimizer
            self._performance_optimizer = get_performance_optimizer()
        except ImportError:
            logger.warning("Performance optimizer not available")

        # Initialize components
        self.local_extractor = get_local_context_extractor(self.context_config)
        self.global_extractor = get_global_context_extractor(self.context_config)
        self.fusion_engine = get_context_fusion_engine(self.context_config)
        self.section_analyzer = DocumentSectionAnalyzer()

        # Initialize basic chunker for core text processing
        from agentic_rag.services.chunking.basic_chunker import BasicTextChunker
        self.basic_chunker = BasicTextChunker(self.chunking_config)

        # Performance tracking
        self._stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "context_extractions": 0,
            "cache_hits": 0
        }

        logger.info("Contextual chunker initialized with performance optimization")

    def chunk_document(self, extracted_content: ExtractedContent) -> List[ContextualChunk]:
        """
        Process a document into contextual chunks.

        Args:
            extracted_content: The extracted content from document parsing

        Returns:
            List of contextual chunks with enhanced context
        """
        logger.info(f"Starting contextual chunking for document {extracted_content.document_id}")

        # Start performance tracking
        if self._performance_optimizer:
            self._performance_optimizer.start_document_processing(
                extracted_content.document_id,
                len(extracted_content.elements)
            )
            self._performance_optimizer.check_memory_usage()

        # Analyze document structure
        sections = self.section_analyzer.analyze_document_structure(extracted_content)

        # Extract global context (cached per document)
        global_context = self.global_extractor.collect_global_context(extracted_content)

        # Process each section
        all_chunks = []

        for section in sections:
            section_chunks = self._process_section(extracted_content, section, sections, global_context)
            all_chunks.extend(section_chunks)

        # Update statistics
        self._stats["documents_processed"] += 1
        self._stats["chunks_created"] += len(all_chunks)

        logger.info(f"Created {len(all_chunks)} contextual chunks for document {extracted_content.document_id}")
        return all_chunks

    def _process_section(
        self,
        extracted_content: ExtractedContent,
        section: SectionInfo,
        all_sections: List[SectionInfo],
        global_context: GlobalContext
    ) -> List[ContextualChunk]:
        """Process a single section into contextual chunks."""
        # Get section elements
        section_elements = extracted_content.elements[section.start_element_index:section.end_element_index + 1]

        # Extract section text
        section_text = " ".join(element.content for element in section_elements if element.content.strip())

        if not section_text.strip():
            return []

        # Create basic chunks from section text
        basic_chunks = self.basic_chunker.chunk_text(section_text, extracted_content.document_id)

        # Convert to contextual chunks
        contextual_chunks = []

        for i, basic_chunk in enumerate(basic_chunks):
            # Calculate element index for this chunk
            chunk_element_index = section.start_element_index + i

            # Extract local context
            local_context = self._extract_local_context(
                chunk_element_index,
                extracted_content.elements,
                section,
                all_sections
            )

            # Update global context with section trail for this chunk
            chunk_global_context = GlobalContext(**global_context.dict())
            chunk_global_context.section_trail = self.global_extractor._build_section_trail(section, extracted_content)

            # Fuse context
            contextual_text = self.fusion_engine.fuse_context(
                chunk_global_context,
                local_context,
                basic_chunk.content
            )

            # Calculate context quality score
            quality_score = self._calculate_context_quality(local_context, chunk_global_context, basic_chunk.content)

            # Create contextual metadata
            contextual_metadata = ContextualMetadata(
                local_context=local_context,
                global_context=chunk_global_context,
                context_token_count=self.fusion_engine._estimate_tokens(contextual_text),
                context_quality_score=quality_score,
                fusion_strategy="standard"
            )

            # Create contextual chunk
            contextual_chunk = ContextualChunk(
                content=basic_chunk.content,
                metadata=basic_chunk.metadata,
                contextual_text=contextual_text,
                original_text=basic_chunk.content,
                contextual_metadata=contextual_metadata
            )

            contextual_chunks.append(contextual_chunk)

            # Update progress tracking
            if self._performance_optimizer:
                self._performance_optimizer.update_chunk_progress(
                    extracted_content.document_id,
                    len(contextual_chunks)
                )

        # Complete performance tracking
        if self._performance_optimizer:
            self._performance_optimizer.complete_document_processing(extracted_content.document_id)
            self._performance_optimizer.check_memory_usage()

        # Update statistics
        self._stats["documents_processed"] += 1
        self._stats["chunks_created"] += len(contextual_chunks)
        self._stats["context_extractions"] += len(contextual_chunks)

        logger.info(f"Completed contextual chunking for document {extracted_content.document_id}: {len(contextual_chunks)} chunks")
        return contextual_chunks

    def _extract_local_context(
        self,
        element_index: int,
        elements: List[LayoutElement],
        current_section: SectionInfo,
        all_sections: List[SectionInfo]
    ) -> LocalContext:
        """Extract local context for a specific element."""
        # Collect neighboring spans
        neighbors = self.local_extractor.collect_neighbors(element_index, elements)

        # Extract sibling headings
        sibling_headings = self.local_extractor.extract_sibling_headings(current_section, all_sections)

        # Extract parent context if available
        parent_context = None
        if (self.context_config.include_parent_context and
            current_section.parent_section_id):

            # Find parent section
            parent_section = next(
                (s for s in all_sections if s.section_id == current_section.parent_section_id),
                None
            )
            if parent_section:
                parent_context = ContextElement(
                    type=ContextType.PARENT_SECTION,
                    content=parent_section.title,
                    relevance_score=0.7,
                    source_element_id=parent_section.section_id,
                    metadata={"section_level": parent_section.level}
                )

        local_context = LocalContext(
            prev_spans=neighbors['prev_spans'],
            next_spans=neighbors['next_spans'],
            sibling_headings=sibling_headings,
            parent_context=parent_context,
            window_size=self.context_config.local_window_size
        )

        self._stats["context_extractions"] += 1
        return local_context

    def _calculate_context_quality(
        self,
        local_context: LocalContext,
        global_context: GlobalContext,
        core_text: str
    ) -> float:
        """Calculate quality score for context extraction."""
        score = 0.0
        max_score = 0.0

        # Local context quality
        if local_context.prev_spans or local_context.next_spans:
            score += 0.3
        max_score += 0.3

        if local_context.sibling_headings:
            score += 0.2
        max_score += 0.2

        # Global context quality
        if global_context.title:
            score += 0.2
        max_score += 0.2

        if global_context.section_trail:
            score += 0.2
        max_score += 0.2

        if global_context.key_definitions:
            score += 0.1
        max_score += 0.1

        return score / max_score if max_score > 0 else 0.0

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self._stats.copy()


# Singleton instance
_contextual_chunker: Optional[ContextualChunker] = None


def get_contextual_chunker(
    chunking_config: Optional[ChunkingConfig] = None,
    context_config: Optional[ContextExtractionConfig] = None
) -> ContextualChunker:
    """Get or create a contextual chunker instance."""
    global _contextual_chunker

    if _contextual_chunker is None or chunking_config is not None or context_config is not None:
        _contextual_chunker = ContextualChunker(chunking_config, context_config)

    return _contextual_chunker
