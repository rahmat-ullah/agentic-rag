"""
Chunk Deduplication

This module implements deduplication to prevent redundant chunks within documents
using content hashing and similarity detection.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rag.services.chunking.table_aware_chunker import (
    TableAwareChunker, TableAwareChunk, TableMetadata
)
from agentic_rag.services.chunking.basic_chunker import ChunkingConfig
from agentic_rag.services.content_extraction import ExtractedContent

logger = logging.getLogger(__name__)


@dataclass
class DuplicationInfo:
    """Information about chunk duplication."""
    
    content_hash: str
    normalized_content: str
    similarity_score: float = 0.0
    is_duplicate: bool = False
    original_chunk_id: Optional[str] = None
    duplicate_reason: Optional[str] = None


class DeduplicationMetadata(BaseModel):
    """Extended metadata for deduplication-aware chunks."""
    
    content_hash: str = Field(..., description="SHA256 hash of normalized content")
    normalized_content: str = Field(..., description="Normalized content used for comparison")
    is_duplicate: bool = Field(default=False, description="Whether this chunk is a duplicate")
    original_chunk_id: Optional[str] = Field(None, description="ID of original chunk if this is duplicate")
    duplicate_count: int = Field(default=0, description="Number of duplicates found for this content")
    similarity_score: float = Field(default=1.0, description="Similarity score with original")
    deduplication_reason: Optional[str] = Field(None, description="Reason for deduplication decision")


class DeduplicatedChunk(TableAwareChunk):
    """Text chunk with deduplication awareness."""
    
    deduplication_metadata: DeduplicationMetadata = Field(..., description="Deduplication-specific metadata")
    
    def is_unique(self) -> bool:
        """Check if this chunk is unique (not a duplicate)."""
        return not self.deduplication_metadata.is_duplicate
    
    def get_content_signature(self) -> str:
        """Get a signature for this chunk's content."""
        return f"{self.deduplication_metadata.content_hash[:8]}:{len(self.content)}:{self.metadata.word_count}"


class ContentNormalizer:
    """Normalizes content for deduplication comparison."""
    
    def __init__(self):
        self.whitespace_chars = {' ', '\t', '\n', '\r'}
        self.punctuation_chars = {'.', ',', ';', ':', '!', '?', '"', "'", '(', ')', '[', ']', '{', '}'}
    
    def normalize_content(self, content: str) -> str:
        """
        Normalize content for deduplication comparison.
        
        Args:
            content: Raw content to normalize
            
        Returns:
            Normalized content string
        """
        # Convert to lowercase
        normalized = content.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common punctuation (but keep structure indicators like |)
        for char in self.punctuation_chars:
            if char not in ['|', '-']:  # Keep table structure indicators
                normalized = normalized.replace(char, '')
        
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def calculate_content_hash(self, normalized_content: str) -> str:
        """Calculate SHA256 hash of normalized content."""
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
    
    def calculate_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two content strings.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Handle empty content cases
        if not content1 and not content2:
            return 1.0  # Both empty = identical
        if not content1 or not content2:
            return 0.0  # One empty, one not = different

        # Simple Jaccard similarity based on words
        words1 = set(content1.split())
        words2 = set(content2.split())

        if not words1 and not words2:
            return 1.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


class ChunkDeduplicator:
    """Handles chunk deduplication logic."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.normalizer = ContentNormalizer()
        self.seen_hashes: Dict[str, str] = {}  # hash -> original_chunk_id
        self.seen_content: Dict[str, str] = {}  # hash -> normalized_content
        
        logger.info(f"Chunk deduplicator initialized with similarity threshold: {similarity_threshold}")
    
    def reset(self):
        """Reset the deduplicator state for a new document."""
        self.seen_hashes.clear()
        self.seen_content.clear()
    
    def check_duplication(self, chunk: TableAwareChunk) -> DuplicationInfo:
        """
        Check if a chunk is a duplicate of previously seen content.
        
        Args:
            chunk: Chunk to check for duplication
            
        Returns:
            DuplicationInfo with duplication analysis
        """
        # Normalize content
        normalized_content = self.normalizer.normalize_content(chunk.content)
        content_hash = self.normalizer.calculate_content_hash(normalized_content)
        
        # Check for exact hash match
        if content_hash in self.seen_hashes:
            return DuplicationInfo(
                content_hash=content_hash,
                normalized_content=normalized_content,
                similarity_score=1.0,
                is_duplicate=True,
                original_chunk_id=self.seen_hashes[content_hash],
                duplicate_reason="exact_hash_match"
            )
        
        # Check for high similarity with existing content
        best_similarity = 0.0
        best_match_hash = None
        
        for existing_hash, existing_content in self.seen_content.items():
            similarity = self.normalizer.calculate_similarity(normalized_content, existing_content)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_hash = existing_hash
        
        # Determine if it's a duplicate based on similarity
        is_duplicate = best_similarity >= self.similarity_threshold
        original_chunk_id = self.seen_hashes.get(best_match_hash) if is_duplicate else None
        
        # Record this content if it's not a duplicate
        if not is_duplicate:
            self.seen_hashes[content_hash] = chunk.metadata.chunk_id
            self.seen_content[content_hash] = normalized_content
        
        return DuplicationInfo(
            content_hash=content_hash,
            normalized_content=normalized_content,
            similarity_score=best_similarity,
            is_duplicate=is_duplicate,
            original_chunk_id=original_chunk_id,
            duplicate_reason="similarity_match" if is_duplicate else None
        )


class DeduplicationChunker:
    """Chunker that performs deduplication to prevent redundant chunks."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None, similarity_threshold: float = 0.85):
        self.config = config or ChunkingConfig()
        self.table_aware_chunker = TableAwareChunker(self.config)
        self.deduplicator = ChunkDeduplicator(similarity_threshold)
        
        logger.info("Deduplication chunker initialized")
    
    def chunk_document(self, extracted_content: ExtractedContent) -> List[DeduplicatedChunk]:
        """
        Chunk document while performing deduplication.
        
        Args:
            extracted_content: The extracted content from document parsing
            
        Returns:
            List of deduplicated chunks
        """
        logger.info(f"Starting deduplication chunking for document {extracted_content.document_id}")
        
        # Reset deduplicator for new document
        self.deduplicator.reset()
        
        # Get table-aware chunks as base
        table_chunks = self.table_aware_chunker.chunk_document(extracted_content)
        
        # Process chunks for deduplication
        deduplicated_chunks = []
        duplicate_count = 0
        
        for chunk in table_chunks:
            # Check for duplication
            dup_info = self.deduplicator.check_duplication(chunk)
            
            # Create deduplication metadata
            dedup_metadata = DeduplicationMetadata(
                content_hash=dup_info.content_hash,
                normalized_content=dup_info.normalized_content,
                is_duplicate=dup_info.is_duplicate,
                original_chunk_id=dup_info.original_chunk_id,
                similarity_score=dup_info.similarity_score,
                deduplication_reason=dup_info.duplicate_reason
            )
            
            # Create deduplicated chunk
            dedup_chunk = DeduplicatedChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                section_metadata=chunk.section_metadata,
                table_metadata=chunk.table_metadata,
                deduplication_metadata=dedup_metadata
            )
            
            # Update chunk metadata with deduplication info
            dedup_chunk.metadata.processing_metadata.update({
                "deduplicated": True,
                "is_duplicate": dup_info.is_duplicate,
                "content_hash": dup_info.content_hash,
                "similarity_score": dup_info.similarity_score
            })
            
            # Add chunk to results
            deduplicated_chunks.append(dedup_chunk)
            
            if dup_info.is_duplicate:
                duplicate_count += 1
                logger.debug(f"Duplicate chunk detected: {dup_info.duplicate_reason}, similarity: {dup_info.similarity_score:.3f}")
        
        # Update duplicate counts
        self._update_duplicate_counts(deduplicated_chunks)
        
        logger.info(f"Created {len(deduplicated_chunks)} chunks ({duplicate_count} duplicates) for document")
        return deduplicated_chunks
    
    def _update_duplicate_counts(self, chunks: List[DeduplicatedChunk]):
        """Update duplicate counts for all chunks."""
        hash_counts = {}
        
        # Count occurrences of each hash
        for chunk in chunks:
            content_hash = chunk.deduplication_metadata.content_hash
            hash_counts[content_hash] = hash_counts.get(content_hash, 0) + 1
        
        # Update duplicate counts
        for chunk in chunks:
            content_hash = chunk.deduplication_metadata.content_hash
            chunk.deduplication_metadata.duplicate_count = hash_counts[content_hash] - 1
    
    def get_unique_chunks(self, chunks: List[DeduplicatedChunk]) -> List[DeduplicatedChunk]:
        """Get only unique chunks (filter out duplicates)."""
        return [chunk for chunk in chunks if chunk.is_unique()]
    
    def get_deduplication_summary(self, chunks: List[DeduplicatedChunk]) -> Dict:
        """Get summary information about deduplication results."""
        total_chunks = len(chunks)
        unique_chunks = len([chunk for chunk in chunks if chunk.is_unique()])
        duplicate_chunks = total_chunks - unique_chunks
        
        # Group by hash to find duplicate groups
        hash_groups = {}
        for chunk in chunks:
            content_hash = chunk.deduplication_metadata.content_hash
            if content_hash not in hash_groups:
                hash_groups[content_hash] = []
            hash_groups[content_hash].append(chunk)
        
        duplicate_groups = {h: chunks for h, chunks in hash_groups.items() if len(chunks) > 1}
        
        return {
            "total_chunks": total_chunks,
            "unique_chunks": unique_chunks,
            "duplicate_chunks": duplicate_chunks,
            "deduplication_ratio": duplicate_chunks / total_chunks if total_chunks > 0 else 0.0,
            "duplicate_groups": len(duplicate_groups),
            "largest_duplicate_group": max(len(chunks) for chunks in duplicate_groups.values()) if duplicate_groups else 0,
            "similarity_threshold": self.deduplicator.similarity_threshold
        }


# Global deduplication chunker instance
_deduplication_chunker: Optional[DeduplicationChunker] = None


def get_deduplication_chunker(
    config: Optional[ChunkingConfig] = None, 
    similarity_threshold: float = 0.85
) -> DeduplicationChunker:
    """Get or create the global deduplication chunker instance."""
    global _deduplication_chunker
    
    if _deduplication_chunker is None or config is not None:
        _deduplication_chunker = DeduplicationChunker(config, similarity_threshold)
    
    return _deduplication_chunker
