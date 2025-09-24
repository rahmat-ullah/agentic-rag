"""
Basic Text Chunking Algorithm

This module implements the basic text chunking algorithm with configurable parameters,
sliding window approach, and sentence boundary detection for clean breaks.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    
    chunk_size: int = 1000  # Target chunk size in characters
    chunk_overlap: int = 200  # Overlap between chunks in characters
    min_chunk_size: int = 100  # Minimum chunk size to avoid tiny chunks
    max_chunk_size: int = 2000  # Maximum chunk size to prevent oversized chunks
    respect_sentence_boundaries: bool = True  # Whether to break at sentence boundaries
    respect_paragraph_boundaries: bool = True  # Whether to break at paragraph boundaries
    preserve_whitespace: bool = False  # Whether to preserve original whitespace
    language: str = "en"  # Language for sentence detection


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""
    
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="Document this chunk belongs to")
    chunk_index: int = Field(..., description="Index of chunk within document")
    start_char: int = Field(..., description="Start character position in original text")
    end_char: int = Field(..., description="End character position in original text")
    chunk_size: int = Field(..., description="Size of chunk in characters")
    word_count: int = Field(..., description="Number of words in chunk")
    sentence_count: int = Field(..., description="Number of sentences in chunk")
    overlap_with_previous: int = Field(default=0, description="Characters overlapping with previous chunk")
    overlap_with_next: int = Field(default=0, description="Characters overlapping with next chunk")
    chunk_type: str = Field(default="text", description="Type of chunk (text, table, etc.)")
    quality_score: float = Field(default=1.0, description="Quality score for the chunk")
    processing_metadata: Dict = Field(default_factory=dict, description="Additional processing metadata")


class TextChunk(BaseModel):
    """Represents a chunk of text with metadata."""
    
    content: str = Field(..., description="The actual text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Metadata about the chunk")
    
    @property
    def id(self) -> str:
        """Get chunk ID."""
        return self.metadata.chunk_id
    
    @property
    def size(self) -> int:
        """Get chunk size in characters."""
        return len(self.content)
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.model_dump(),
            "size": self.size
        }


class SentenceDetector:
    """Simple sentence boundary detection."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        
        # Simple sentence ending patterns for English
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.abbreviations = {
            'dr', 'mr', 'mrs', 'ms', 'prof', 'inc', 'ltd', 'corp', 'co',
            'vs', 'etc', 'ie', 'eg', 'al', 'st', 'ave', 'blvd'
        }
    
    def find_sentence_boundaries(self, text: str) -> List[int]:
        """
        Find sentence boundaries in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of character positions where sentences end
        """
        boundaries = []
        
        for match in self.sentence_endings.finditer(text):
            end_pos = match.end()
            
            # Check if this is likely a real sentence boundary
            if self._is_sentence_boundary(text, match.start(), end_pos):
                boundaries.append(end_pos)
        
        return boundaries
    
    def _is_sentence_boundary(self, text: str, start_pos: int, end_pos: int) -> bool:
        """Check if a potential boundary is a real sentence boundary."""
        # Get text before the punctuation
        before_text = text[max(0, start_pos - 10):start_pos].lower().strip()
        
        # Check for common abbreviations
        words_before = before_text.split()
        if words_before and words_before[-1].rstrip('.') in self.abbreviations:
            return False
        
        # Check if next character is uppercase (likely new sentence)
        if end_pos < len(text):
            next_char = text[end_pos:end_pos + 1]
            if next_char and next_char.isupper():
                return True
        
        return True


class BasicTextChunker:
    """Basic text chunking implementation with configurable parameters."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.sentence_detector = SentenceDetector(self.config.language)
        
        logger.info(f"Basic text chunker initialized with chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
    
    def chunk_text(self, text: str, document_id: str) -> List[TextChunk]:
        """
        Chunk text into overlapping segments with configurable parameters.
        
        Args:
            text: Text to chunk
            document_id: ID of the document being chunked
            
        Returns:
            List of text chunks with metadata
        """
        if not text.strip():
            logger.warning(f"Empty text provided for chunking in document {document_id}")
            return []
        
        logger.info(f"Starting text chunking for document {document_id}, text length: {len(text)}")
        
        # Normalize text if needed
        normalized_text = self._normalize_text(text)
        
        # Find potential break points
        break_points = self._find_break_points(normalized_text)
        
        # Create chunks using sliding window approach
        chunks = self._create_chunks_with_overlap(normalized_text, break_points, document_id)
        
        # Post-process chunks
        processed_chunks = self._post_process_chunks(chunks)
        
        logger.info(f"Created {len(processed_chunks)} chunks for document {document_id}")
        
        return processed_chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for chunking."""
        if not self.config.preserve_whitespace:
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    def _find_break_points(self, text: str) -> List[int]:
        """Find potential break points in text."""
        break_points = [0]  # Start of text
        
        if self.config.respect_sentence_boundaries:
            # Add sentence boundaries
            sentence_boundaries = self.sentence_detector.find_sentence_boundaries(text)
            break_points.extend(sentence_boundaries)
        
        if self.config.respect_paragraph_boundaries:
            # Add paragraph boundaries (double newlines)
            paragraph_boundaries = [m.end() for m in re.finditer(r'\n\s*\n', text)]
            break_points.extend(paragraph_boundaries)
        
        # Add end of text
        break_points.append(len(text))
        
        # Remove duplicates and sort
        break_points = sorted(set(break_points))
        
        return break_points
    
    def _create_chunks_with_overlap(self, text: str, break_points: List[int], document_id: str) -> List[TextChunk]:
        """Create chunks with overlap using sliding window approach."""
        chunks = []
        chunk_index = 0
        
        start_pos = 0
        
        while start_pos < len(text):
            # Find the end position for this chunk
            target_end = start_pos + self.config.chunk_size
            
            # Find the best break point near the target end
            end_pos = self._find_best_break_point(break_points, target_end, start_pos)
            
            # Ensure we don't exceed text length
            end_pos = min(end_pos, len(text))
            
            # Extract chunk content
            chunk_content = text[start_pos:end_pos].strip()
            
            # Skip empty chunks
            if not chunk_content:
                start_pos = end_pos
                continue
            
            # Check minimum chunk size
            if len(chunk_content) < self.config.min_chunk_size and end_pos < len(text):
                # Try to extend the chunk
                extended_end = self._find_best_break_point(
                    break_points, 
                    start_pos + self.config.min_chunk_size, 
                    start_pos
                )
                extended_end = min(extended_end, len(text))
                chunk_content = text[start_pos:extended_end].strip()
                end_pos = extended_end
            
            # Create chunk metadata
            metadata = self._create_chunk_metadata(
                chunk_content, document_id, chunk_index, start_pos, end_pos
            )
            
            # Calculate overlap with previous chunk
            if chunks:
                prev_chunk = chunks[-1]
                overlap = max(0, prev_chunk.metadata.end_char - start_pos)
                metadata.overlap_with_previous = overlap
                prev_chunk.metadata.overlap_with_next = overlap
            
            # Create chunk
            chunk = TextChunk(content=chunk_content, metadata=metadata)
            chunks.append(chunk)
            
            chunk_index += 1
            
            # Calculate next start position with overlap
            if end_pos >= len(text):
                break

            # Move start position forward, accounting for overlap
            overlap_size = min(self.config.chunk_overlap, len(chunk_content) // 2)
            next_start = max(start_pos + 1, end_pos - overlap_size)

            # Ensure we have actual overlap if possible
            if next_start >= end_pos:
                next_start = max(start_pos + 1, end_pos - min(self.config.chunk_overlap, len(chunk_content) // 3))

            start_pos = next_start
        
        return chunks
    
    def _find_best_break_point(self, break_points: List[int], target_pos: int, min_pos: int) -> int:
        """Find the best break point near the target position."""
        if not break_points:
            return target_pos
        
        # Find break points within acceptable range
        acceptable_points = [
            bp for bp in break_points 
            if min_pos <= bp <= target_pos + self.config.chunk_size // 4
        ]
        
        if not acceptable_points:
            # No good break points, use target position
            return target_pos
        
        # Find the break point closest to target
        best_point = min(acceptable_points, key=lambda x: abs(x - target_pos))
        
        return best_point
    
    def _create_chunk_metadata(
        self, 
        content: str, 
        document_id: str, 
        chunk_index: int, 
        start_char: int, 
        end_char: int
    ) -> ChunkMetadata:
        """Create metadata for a chunk."""
        # Count words and sentences
        words = content.split()
        word_count = len(words)
        
        sentence_count = len(self.sentence_detector.find_sentence_boundaries(content + "."))
        if sentence_count == 0 and content.strip():
            sentence_count = 1  # At least one sentence if there's content
        
        # Calculate quality score based on various factors
        quality_score = self._calculate_quality_score(content, word_count, sentence_count)
        
        return ChunkMetadata(
            chunk_id=str(uuid4()),
            document_id=document_id,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            chunk_size=len(content),
            word_count=word_count,
            sentence_count=sentence_count,
            quality_score=quality_score,
            processing_metadata={
                "chunking_method": "basic_sliding_window",
                "config": {
                    "chunk_size": self.config.chunk_size,
                    "overlap": self.config.chunk_overlap,
                    "respect_sentences": self.config.respect_sentence_boundaries
                }
            }
        )
    
    def _calculate_quality_score(self, content: str, word_count: int, sentence_count: int) -> float:
        """Calculate a quality score for the chunk."""
        score = 1.0
        
        # Penalize very short chunks
        if len(content) < self.config.min_chunk_size:
            score *= 0.7
        
        # Penalize chunks with very few words
        if word_count < 10:
            score *= 0.8
        
        # Penalize chunks that are mostly whitespace
        if len(content.strip()) / len(content) < 0.8:
            score *= 0.6
        
        # Bonus for chunks with complete sentences
        if sentence_count > 0 and content.strip().endswith(('.', '!', '?')):
            score *= 1.1
        
        return min(score, 1.0)
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Post-process chunks to improve quality."""
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for chunk in chunks:
            # Skip chunks that are too small unless they're the last chunk
            if (len(chunk.content) < self.config.min_chunk_size and 
                chunk != chunks[-1] and 
                len(chunks) > 1):
                
                # Try to merge with previous chunk if possible
                if processed_chunks:
                    last_chunk = processed_chunks[-1]
                    merged_content = last_chunk.content + " " + chunk.content
                    
                    if len(merged_content) <= self.config.max_chunk_size:
                        # Update last chunk with merged content
                        last_chunk.content = merged_content
                        last_chunk.metadata.end_char = chunk.metadata.end_char
                        last_chunk.metadata.chunk_size = len(merged_content)
                        last_chunk.metadata.word_count += chunk.metadata.word_count
                        last_chunk.metadata.sentence_count += chunk.metadata.sentence_count
                        continue
            
            processed_chunks.append(chunk)
        
        # Update chunk indices
        for i, chunk in enumerate(processed_chunks):
            chunk.metadata.chunk_index = i
        
        return processed_chunks
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict:
        """Get statistics about the chunking results."""
        if not chunks:
            return {"total_chunks": 0}
        
        sizes = [chunk.size for chunk in chunks]
        word_counts = [chunk.metadata.word_count for chunk in chunks]
        quality_scores = [chunk.metadata.quality_score for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(sizes),
            "total_words": sum(word_counts),
            "average_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "average_word_count": sum(word_counts) / len(word_counts),
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "chunks_below_min_size": sum(1 for size in sizes if size < self.config.min_chunk_size),
            "chunks_above_max_size": sum(1 for size in sizes if size > self.config.max_chunk_size)
        }


# Global chunker instance
_basic_chunker: Optional[BasicTextChunker] = None


def get_basic_chunker(config: Optional[ChunkingConfig] = None) -> BasicTextChunker:
    """Get or create the global basic chunker instance."""
    global _basic_chunker
    
    if _basic_chunker is None or config is not None:
        _basic_chunker = BasicTextChunker(config)
    
    return _basic_chunker
