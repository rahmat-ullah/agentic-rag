"""
Chunking Pipeline Integration

This module integrates the chunking pipeline with the document processing workflow
and provides async processing capabilities.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rag.services.chunking.basic_chunker import ChunkingConfig
from agentic_rag.services.chunking.deduplication_chunker import (
    DeduplicationChunker, DeduplicatedChunk, get_deduplication_chunker
)
from agentic_rag.services.content_extraction import ExtractedContent

logger = logging.getLogger(__name__)


@dataclass
class ChunkingPipelineConfig:
    """Configuration for the chunking pipeline."""
    
    # Basic chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    language: str = "en"
    
    # Deduplication settings
    enable_deduplication: bool = True
    similarity_threshold: float = 0.85
    
    # Processing settings
    enable_async: bool = True
    max_concurrent_chunks: int = 10
    batch_size: int = 50
    
    # Quality settings
    min_quality_score: float = 0.5
    filter_low_quality: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


class ChunkingResult(BaseModel):
    """Result of chunking pipeline processing."""
    
    document_id: str = Field(..., description="Document ID that was processed")
    total_chunks: int = Field(..., description="Total number of chunks created")
    unique_chunks: int = Field(..., description="Number of unique chunks (after deduplication)")
    duplicate_chunks: int = Field(..., description="Number of duplicate chunks removed")
    processing_time_seconds: float = Field(..., description="Total processing time")
    quality_filtered_chunks: int = Field(default=0, description="Number of chunks filtered for low quality")
    deduplication_ratio: float = Field(default=0.0, description="Ratio of duplicates to total chunks")
    average_chunk_size: float = Field(default=0.0, description="Average chunk size in characters")
    chunks: List[DeduplicatedChunk] = Field(..., description="The processed chunks")
    processing_metadata: Dict = Field(default_factory=dict, description="Additional processing metadata")


class ChunkingPipeline:
    """Integrated chunking pipeline with async processing capabilities."""
    
    def __init__(self, config: Optional[ChunkingPipelineConfig] = None):
        self.config = config or ChunkingPipelineConfig()
        
        # Initialize chunking components
        chunking_config = ChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
            language=self.config.language
        )
        
        if self.config.enable_deduplication:
            self.chunker = get_deduplication_chunker(
                chunking_config, 
                self.config.similarity_threshold
            )
        else:
            # Fallback to table-aware chunker if deduplication is disabled
            from agentic_rag.services.chunking.table_aware_chunker import get_table_aware_chunker
            self.chunker = get_table_aware_chunker(chunking_config)
        
        # Processing state
        self._cache: Dict[str, ChunkingResult] = {}
        
        logger.info(f"Chunking pipeline initialized with config: {self.config}")
    
    async def process_document_async(self, extracted_content: ExtractedContent) -> ChunkingResult:
        """
        Process a document through the chunking pipeline asynchronously.
        
        Args:
            extracted_content: The extracted content from document parsing
            
        Returns:
            ChunkingResult with processed chunks and metadata
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting async chunking for document {extracted_content.document_id}")
        
        # Check cache if enabled
        if self.config.enable_caching:
            cached_result = self._get_cached_result(extracted_content.document_id)
            if cached_result:
                logger.info(f"Returning cached result for document {extracted_content.document_id}")
                return cached_result
        
        # Process chunks
        if self.config.enable_async:
            chunks = await self._process_chunks_async(extracted_content)
        else:
            chunks = self._process_chunks_sync(extracted_content)
        
        # Apply quality filtering
        if self.config.filter_low_quality:
            original_count = len(chunks)
            chunks = self._filter_low_quality_chunks(chunks)
            quality_filtered = original_count - len(chunks)
        else:
            quality_filtered = 0
        
        # Calculate metrics
        processing_time = time.time() - start_time
        result = self._create_result(
            extracted_content.document_id,
            chunks,
            processing_time,
            quality_filtered
        )
        
        # Cache result if enabled
        if self.config.enable_caching:
            self._cache_result(result)
        
        logger.info(f"Completed chunking for document {extracted_content.document_id} in {processing_time:.2f}s")
        return result
    
    def process_document_sync(self, extracted_content: ExtractedContent) -> ChunkingResult:
        """
        Process a document through the chunking pipeline synchronously.

        Args:
            extracted_content: The extracted content from document parsing

        Returns:
            ChunkingResult with processed chunks and metadata
        """
        # Run async method in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                return asyncio.create_task(self.process_document_async(extracted_content))
        except RuntimeError:
            # No event loop exists, create one
            pass

        # If we're in sync context, run the async method
        return asyncio.run(self.process_document_async(extracted_content))
    
    async def _process_chunks_async(self, extracted_content: ExtractedContent) -> List[DeduplicatedChunk]:
        """Process chunks asynchronously with concurrency control."""
        if hasattr(self.chunker, 'chunk_document'):
            # For now, chunking is CPU-bound, so we'll run it in a thread pool
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None, 
                self.chunker.chunk_document, 
                extracted_content
            )
            return chunks
        else:
            # Fallback to sync processing
            return self._process_chunks_sync(extracted_content)
    
    def _process_chunks_sync(self, extracted_content: ExtractedContent) -> List[DeduplicatedChunk]:
        """Process chunks synchronously."""
        if hasattr(self.chunker, 'chunk_document'):
            return self.chunker.chunk_document(extracted_content)
        else:
            # Handle non-deduplication chunkers
            chunks = self.chunker.chunk_document(extracted_content)
            # Convert to DeduplicatedChunk format for consistency
            return self._convert_to_deduplicated_chunks(chunks)
    
    def _convert_to_deduplicated_chunks(self, chunks) -> List[DeduplicatedChunk]:
        """Convert regular chunks to deduplicated chunk format."""
        from agentic_rag.services.chunking.deduplication_chunker import (
            DeduplicatedChunk, DeduplicationMetadata, ContentNormalizer
        )
        
        normalizer = ContentNormalizer()
        deduplicated_chunks = []
        
        for chunk in chunks:
            # Create basic deduplication metadata
            normalized_content = normalizer.normalize_content(chunk.content)
            content_hash = normalizer.calculate_content_hash(normalized_content)
            
            dedup_metadata = DeduplicationMetadata(
                content_hash=content_hash,
                normalized_content=normalized_content,
                is_duplicate=False,  # No deduplication performed
                similarity_score=1.0
            )
            
            # Get or create default metadata
            from agentic_rag.services.chunking.section_aware_chunker import SectionMetadata
            from agentic_rag.services.chunking.table_aware_chunker import TableMetadata

            section_metadata = getattr(chunk, 'section_metadata', None)
            if section_metadata is None:
                section_metadata = SectionMetadata(
                    section_id=str(uuid4()),
                    section_title="Default Section",
                    section_level=1
                )

            table_metadata = getattr(chunk, 'table_metadata', None)
            if table_metadata is None:
                table_metadata = TableMetadata()

            # Convert to DeduplicatedChunk
            dedup_chunk = DeduplicatedChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                section_metadata=section_metadata,
                table_metadata=table_metadata,
                deduplication_metadata=dedup_metadata
            )
            
            deduplicated_chunks.append(dedup_chunk)
        
        return deduplicated_chunks
    
    def _filter_low_quality_chunks(self, chunks: List[DeduplicatedChunk]) -> List[DeduplicatedChunk]:
        """Filter out low-quality chunks based on quality score."""
        filtered_chunks = []
        
        for chunk in chunks:
            if chunk.metadata.quality_score >= self.config.min_quality_score:
                filtered_chunks.append(chunk)
            else:
                logger.debug(f"Filtered low-quality chunk: {chunk.metadata.chunk_id} (score: {chunk.metadata.quality_score})")
        
        return filtered_chunks
    
    def _create_result(
        self, 
        document_id: str, 
        chunks: List[DeduplicatedChunk], 
        processing_time: float,
        quality_filtered: int
    ) -> ChunkingResult:
        """Create a chunking result with calculated metrics."""
        total_chunks = len(chunks)
        
        # Calculate deduplication metrics
        if self.config.enable_deduplication and hasattr(self.chunker, 'get_deduplication_summary'):
            dedup_summary = self.chunker.get_deduplication_summary(chunks)
            unique_chunks = dedup_summary.get('unique_chunks', total_chunks)
            duplicate_chunks = dedup_summary.get('duplicate_chunks', 0)
            deduplication_ratio = dedup_summary.get('deduplication_ratio', 0.0)
        else:
            unique_chunks = total_chunks
            duplicate_chunks = 0
            deduplication_ratio = 0.0
        
        # Calculate average chunk size
        if chunks:
            average_chunk_size = sum(chunk.metadata.chunk_size for chunk in chunks) / len(chunks)
        else:
            average_chunk_size = 0.0
        
        # Create processing metadata
        processing_metadata = {
            "pipeline_config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "deduplication_enabled": self.config.enable_deduplication,
                "similarity_threshold": self.config.similarity_threshold,
                "quality_filtering_enabled": self.config.filter_low_quality,
                "min_quality_score": self.config.min_quality_score
            },
            "processing_stats": {
                "async_processing": self.config.enable_async,
                "quality_filtered_count": quality_filtered,
                "average_chunk_size": average_chunk_size
            }
        }
        
        return ChunkingResult(
            document_id=document_id,
            total_chunks=total_chunks,
            unique_chunks=unique_chunks,
            duplicate_chunks=duplicate_chunks,
            processing_time_seconds=processing_time,
            quality_filtered_chunks=quality_filtered,
            deduplication_ratio=deduplication_ratio,
            average_chunk_size=average_chunk_size,
            chunks=chunks,
            processing_metadata=processing_metadata
        )
    
    def _get_cached_result(self, document_id: str) -> Optional[ChunkingResult]:
        """Get cached result if available and not expired."""
        if document_id in self._cache:
            # For simplicity, we're not implementing TTL expiration here
            # In production, you'd want to check timestamp and TTL
            return self._cache[document_id]
        return None
    
    def _cache_result(self, result: ChunkingResult):
        """Cache the chunking result."""
        self._cache[result.document_id] = result
        
        # Simple cache size management (keep last 100 results)
        if len(self._cache) > 100:
            # Remove oldest entries (in production, use LRU cache)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
    
    def clear_cache(self):
        """Clear the processing cache."""
        self._cache.clear()
        logger.info("Chunking pipeline cache cleared")
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "deduplication_enabled": self.config.enable_deduplication,
                "similarity_threshold": self.config.similarity_threshold,
                "async_enabled": self.config.enable_async,
                "caching_enabled": self.config.enable_caching
            },
            "cache_stats": {
                "cached_documents": len(self._cache),
                "cache_size_limit": 100
            },
            "chunker_type": type(self.chunker).__name__
        }


# Global pipeline instance
_chunking_pipeline: Optional[ChunkingPipeline] = None


def get_chunking_pipeline(config: Optional[ChunkingPipelineConfig] = None) -> ChunkingPipeline:
    """Get or create the global chunking pipeline instance."""
    global _chunking_pipeline
    
    if _chunking_pipeline is None or config is not None:
        _chunking_pipeline = ChunkingPipeline(config)
    
    return _chunking_pipeline


async def process_document_chunks_async(extracted_content: ExtractedContent) -> ChunkingResult:
    """
    Convenience function to process document chunks asynchronously.
    
    Args:
        extracted_content: The extracted content from document parsing
        
    Returns:
        ChunkingResult with processed chunks and metadata
    """
    pipeline = get_chunking_pipeline()
    return await pipeline.process_document_async(extracted_content)


def process_document_chunks_sync(extracted_content: ExtractedContent) -> ChunkingResult:
    """
    Convenience function to process document chunks synchronously.
    
    Args:
        extracted_content: The extracted content from document parsing
        
    Returns:
        ChunkingResult with processed chunks and metadata
    """
    pipeline = get_chunking_pipeline()
    return pipeline.process_document_sync(extracted_content)
