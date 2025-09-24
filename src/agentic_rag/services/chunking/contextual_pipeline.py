"""
Contextual Chunking Pipeline Integration

This module integrates contextual chunking with the embedding pipeline and vector storage,
providing enhanced chunk processing that uses contextual text for embeddings while
preserving original text for citations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rag.services.chunking.contextual_chunker import (
    ContextualChunker, ContextualChunk, ContextExtractionConfig, get_contextual_chunker
)
from agentic_rag.services.chunking.basic_chunker import ChunkingConfig
from agentic_rag.services.chunking.pipeline import ChunkingResult
from agentic_rag.services.content_extraction import ExtractedContent
from agentic_rag.services.embedding_pipeline import EmbeddingPipelineRequest, get_embedding_pipeline
from agentic_rag.services.vector_store import VectorMetadata
from agentic_rag.services.vector_operations import VectorData, get_vector_operations

logger = logging.getLogger(__name__)


@dataclass
class ContextualChunkingConfig:
    """Configuration for contextual chunking pipeline."""
    
    # Basic chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    language: str = "en"
    
    # Contextual settings
    enable_contextual_chunking: bool = True
    context_extraction_config: Optional[ContextExtractionConfig] = None
    
    # Embedding settings
    use_contextual_text_for_embeddings: bool = True
    preserve_original_text: bool = True
    validate_embedding_quality: bool = True
    
    # Performance settings
    enable_async: bool = True
    batch_size: int = 50
    max_concurrent_chunks: int = 10
    
    # Quality settings
    min_context_quality_score: float = 0.3
    filter_low_context_quality: bool = False


class ContextualChunkingResult(BaseModel):
    """Result of contextual chunking pipeline processing."""
    
    document_id: str = Field(..., description="Document identifier")
    total_chunks: int = Field(..., description="Total number of chunks created")
    contextual_chunks: int = Field(..., description="Number of chunks with context")
    embedding_chunks: int = Field(..., description="Number of chunks with embeddings")
    processing_time: float = Field(..., description="Total processing time in seconds")
    context_quality_avg: float = Field(default=0.0, description="Average context quality score")
    context_token_avg: int = Field(default=0, description="Average context token count")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    metadata: Dict = Field(default_factory=dict, description="Additional processing metadata")


class ContextualChunkProcessor:
    """Processes individual contextual chunks with embeddings."""
    
    def __init__(self, config: ContextualChunkingConfig):
        self.config = config
        self._embedding_pipeline = None
        self._vector_operations = None
        
    async def initialize(self):
        """Initialize the processor components."""
        if not self._embedding_pipeline:
            self._embedding_pipeline = await get_embedding_pipeline()
        if not self._vector_operations:
            self._vector_operations = await get_vector_operations()
    
    async def process_contextual_chunk(
        self, 
        chunk: ContextualChunk, 
        tenant_id: str, 
        document_id: str
    ) -> Dict:
        """
        Process a contextual chunk with embedding generation.
        
        Args:
            chunk: Contextual chunk to process
            tenant_id: Tenant identifier
            document_id: Document identifier
        
        Returns:
            dict: Processing result with embedding and metadata
        """
        await self.initialize()
        
        try:
            # Prepare vector metadata
            vector_metadata = self._create_vector_metadata(chunk, tenant_id, document_id)
            
            # Choose text for embedding
            embedding_text = chunk.get_embedding_text() if self.config.use_contextual_text_for_embeddings else chunk.get_citation_text()
            
            # Generate embedding
            embedding_request = EmbeddingPipelineRequest(
                texts=[embedding_text],
                metadata_list=[vector_metadata],
                tenant_id=tenant_id,
                document_id=document_id,
                store_vectors=False,  # We'll store manually with enhanced metadata
                validate_quality=self.config.validate_embedding_quality,
                optimize_cost=True
            )
            
            embedding_result = await self._embedding_pipeline.process_embeddings(embedding_request)
            
            if embedding_result.status.value != "completed" or not embedding_result.embeddings:
                raise Exception(f"Embedding generation failed: {embedding_result.errors}")
            
            # Validate contextual embedding quality
            if self.config.validate_embedding_quality:
                quality_valid = await self._validate_contextual_embedding(
                    chunk, embedding_result.embeddings[0]
                )
                if not quality_valid:
                    logger.warning(f"Low quality contextual embedding for chunk {chunk.metadata.chunk_id}")
            
            # Create enhanced vector data
            vector_data = self._create_enhanced_vector_data(
                chunk, embedding_result.embeddings[0], vector_metadata
            )
            
            return {
                "success": True,
                "chunk_id": chunk.metadata.chunk_id,
                "vector_data": vector_data,
                "embedding_quality": embedding_result.quality_report,
                "context_quality": chunk.contextual_metadata.context_quality_score,
                "context_tokens": chunk.contextual_metadata.context_token_count
            }
            
        except Exception as e:
            logger.error(f"Failed to process contextual chunk {chunk.metadata.chunk_id}: {e}")
            return {
                "success": False,
                "chunk_id": chunk.metadata.chunk_id,
                "error": str(e)
            }
    
    def _create_vector_metadata(
        self, 
        chunk: ContextualChunk, 
        tenant_id: str, 
        document_id: str
    ) -> VectorMetadata:
        """Create enhanced vector metadata for contextual chunk."""
        # Extract section path from contextual metadata
        section_path = []
        if chunk.contextual_metadata.global_context.section_trail:
            section_path = [
                element.content for element in chunk.contextual_metadata.global_context.section_trail
            ]
        
        return VectorMetadata(
            tenant_id=tenant_id,
            document_id=document_id,
            chunk_id=chunk.metadata.chunk_id,
            document_kind="UNKNOWN",  # Will be set by caller
            created_at=None,  # Will be set by vector operations
            section_path=section_path,
            page_from=None,  # Could be extracted from chunk metadata
            page_to=None,
            token_count=chunk.contextual_metadata.context_token_count,
            is_table=False  # Could be enhanced to detect tables
        )
    
    def _create_enhanced_vector_data(
        self, 
        chunk: ContextualChunk, 
        embedding: List[float], 
        vector_metadata: VectorMetadata
    ) -> VectorData:
        """Create enhanced vector data with contextual information."""
        # Enhance metadata with contextual information
        enhanced_metadata = vector_metadata.dict()
        enhanced_metadata.update({
            "contextual_chunking": True,
            "original_text": chunk.get_citation_text(),
            "contextual_text": chunk.get_embedding_text(),
            "context_quality_score": chunk.contextual_metadata.context_quality_score,
            "context_token_count": chunk.contextual_metadata.context_token_count,
            "fusion_strategy": chunk.contextual_metadata.fusion_strategy,
            "local_context_elements": len(chunk.contextual_metadata.local_context.prev_spans) + 
                                    len(chunk.contextual_metadata.local_context.next_spans),
            "global_context_elements": len(chunk.contextual_metadata.global_context.section_trail) +
                                     len(chunk.contextual_metadata.global_context.key_definitions),
            "has_document_title": chunk.contextual_metadata.global_context.title is not None,
            "has_section_trail": len(chunk.contextual_metadata.global_context.section_trail) > 0
        })
        
        return VectorData(
            id=f"{vector_metadata.document_id}_{chunk.metadata.chunk_id}",
            embedding=embedding,
            metadata=VectorMetadata(**{k: v for k, v in enhanced_metadata.items() 
                                     if k in VectorMetadata.__fields__}),
            document=chunk.get_citation_text()  # Store original text for citations
        )
    
    async def _validate_contextual_embedding(
        self, 
        chunk: ContextualChunk, 
        embedding: List[float], 
        quality_threshold: float = 0.8
    ) -> bool:
        """Validate quality of contextual embedding."""
        # Basic validation - could be enhanced with more sophisticated checks
        if not embedding or len(embedding) == 0:
            return False
        
        # Check if embedding has reasonable variance (not all zeros or all same values)
        embedding_variance = sum((x - sum(embedding)/len(embedding))**2 for x in embedding) / len(embedding)
        if embedding_variance < 1e-6:
            return False
        
        # Check context quality score
        if chunk.contextual_metadata.context_quality_score < quality_threshold:
            return False
        
        return True


class ContextualChunkingPipeline:
    """Enhanced chunking pipeline with contextual processing and embedding integration."""
    
    def __init__(self, config: Optional[ContextualChunkingConfig] = None):
        self.config = config or ContextualChunkingConfig()
        
        # Initialize components
        chunking_config = ChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
            language=self.config.language
        )
        
        context_config = self.config.context_extraction_config or ContextExtractionConfig()
        
        if self.config.enable_contextual_chunking:
            self.chunker = get_contextual_chunker(chunking_config, context_config)
        else:
            # Fallback to basic chunking
            from agentic_rag.services.chunking.basic_chunker import get_basic_chunker
            self.chunker = get_basic_chunker(chunking_config)
        
        self.chunk_processor = ContextualChunkProcessor(self.config)
        
        # Performance tracking
        self._stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "context_extractions": 0,
            "processing_time_total": 0.0
        }
        
        logger.info(f"Contextual chunking pipeline initialized (contextual={self.config.enable_contextual_chunking})")
    
    async def process_document_async(
        self, 
        extracted_content: ExtractedContent,
        tenant_id: str
    ) -> ContextualChunkingResult:
        """
        Process a document through the contextual chunking pipeline.
        
        Args:
            extracted_content: The extracted content from document parsing
            tenant_id: Tenant identifier for multi-tenancy
            
        Returns:
            ContextualChunkingResult with processing information
        """
        start_time = time.time()
        logger.info(f"Starting contextual chunking pipeline for document {extracted_content.document_id}")
        
        try:
            # Step 1: Create contextual chunks
            if self.config.enable_contextual_chunking:
                chunks = self.chunker.chunk_document(extracted_content)
            else:
                # Convert basic chunks to contextual format for consistency
                basic_chunks = self.chunker.chunk_text(extracted_content.text_content, extracted_content.document_id)
                chunks = self._convert_to_contextual_chunks(basic_chunks)
            
            # Step 2: Filter by context quality if enabled
            if self.config.filter_low_context_quality:
                original_count = len(chunks)
                chunks = [
                    chunk for chunk in chunks 
                    if chunk.contextual_metadata.context_quality_score >= self.config.min_context_quality_score
                ]
                filtered_count = original_count - len(chunks)
                logger.info(f"Filtered {filtered_count} chunks with low context quality")
            
            # Step 3: Process chunks with embeddings
            processing_results = []
            if self.config.enable_async:
                processing_results = await self._process_chunks_async(chunks, tenant_id, extracted_content.document_id)
            else:
                processing_results = await self._process_chunks_sync(chunks, tenant_id, extracted_content.document_id)
            
            # Step 4: Calculate metrics
            processing_time = time.time() - start_time
            result = self._create_result(
                extracted_content.document_id,
                chunks,
                processing_results,
                processing_time
            )
            
            # Update statistics
            self._update_stats(result)
            
            logger.info(f"Contextual chunking completed for document {extracted_content.document_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Contextual chunking pipeline failed for document {extracted_content.document_id}: {e}")
            processing_time = time.time() - start_time
            
            return ContextualChunkingResult(
                document_id=extracted_content.document_id,
                total_chunks=0,
                contextual_chunks=0,
                embedding_chunks=0,
                processing_time=processing_time,
                errors=[str(e)]
            )
    
    async def _process_chunks_async(
        self, 
        chunks: List[ContextualChunk], 
        tenant_id: str, 
        document_id: str
    ) -> List[Dict]:
        """Process chunks asynchronously with concurrency control."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_chunks)
        
        async def process_chunk_with_semaphore(chunk):
            async with semaphore:
                return await self.chunk_processor.process_contextual_chunk(chunk, tenant_id, document_id)
        
        # Process chunks in batches
        results = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            batch_tasks = [process_chunk_with_semaphore(chunk) for chunk in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Chunk processing failed: {result}")
                    results.append({
                        "success": False,
                        "chunk_id": batch[j].metadata.chunk_id,
                        "error": str(result)
                    })
                else:
                    results.append(result)
        
        return results
    
    async def _process_chunks_sync(
        self,
        chunks: List[ContextualChunk],
        tenant_id: str,
        document_id: str
    ) -> List[Dict]:
        """Process chunks synchronously."""
        results = []
        for chunk in chunks:
            result = await self.chunk_processor.process_contextual_chunk(chunk, tenant_id, document_id)
            results.append(result)
        return results

    def _convert_to_contextual_chunks(self, basic_chunks) -> List[ContextualChunk]:
        """Convert basic chunks to contextual format for consistency."""
        from agentic_rag.services.chunking.contextual_chunker import (
            ContextualChunk, ContextualMetadata, LocalContext, GlobalContext
        )

        contextual_chunks = []
        for chunk in basic_chunks:
            # Create minimal contextual metadata
            local_context = LocalContext()
            global_context = GlobalContext()

            contextual_metadata = ContextualMetadata(
                local_context=local_context,
                global_context=global_context,
                context_quality_score=0.0,  # No context extracted
                fusion_strategy="none"
            )

            contextual_chunk = ContextualChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                contextual_text=chunk.content,  # Same as original
                original_text=chunk.content,
                contextual_metadata=contextual_metadata
            )
            contextual_chunks.append(contextual_chunk)

        return contextual_chunks

    def _create_result(
        self,
        document_id: str,
        chunks: List[ContextualChunk],
        processing_results: List[Dict],
        processing_time: float
    ) -> ContextualChunkingResult:
        """Create processing result with metrics."""
        successful_results = [r for r in processing_results if r.get("success", False)]
        failed_results = [r for r in processing_results if not r.get("success", False)]

        # Calculate context quality metrics
        context_qualities = [
            chunk.contextual_metadata.context_quality_score
            for chunk in chunks
            if chunk.contextual_metadata.context_quality_score > 0
        ]
        context_quality_avg = sum(context_qualities) / len(context_qualities) if context_qualities else 0.0

        # Calculate context token metrics
        context_tokens = [
            chunk.contextual_metadata.context_token_count
            for chunk in chunks
        ]
        context_token_avg = sum(context_tokens) // len(context_tokens) if context_tokens else 0

        # Count contextual chunks (those with actual context)
        contextual_chunks = len([
            chunk for chunk in chunks
            if chunk.contextual_metadata.context_quality_score > 0
        ])

        return ContextualChunkingResult(
            document_id=document_id,
            total_chunks=len(chunks),
            contextual_chunks=contextual_chunks,
            embedding_chunks=len(successful_results),
            processing_time=processing_time,
            context_quality_avg=context_quality_avg,
            context_token_avg=context_token_avg,
            errors=[r.get("error", "") for r in failed_results],
            metadata={
                "successful_embeddings": len(successful_results),
                "failed_embeddings": len(failed_results),
                "avg_context_quality": context_quality_avg,
                "avg_context_tokens": context_token_avg
            }
        )

    def _update_stats(self, result: ContextualChunkingResult):
        """Update processing statistics."""
        self._stats["documents_processed"] += 1
        self._stats["chunks_created"] += result.total_chunks
        self._stats["embeddings_generated"] += result.embedding_chunks
        self._stats["context_extractions"] += result.contextual_chunks
        self._stats["processing_time_total"] += result.processing_time

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        stats = self._stats.copy()
        if stats["documents_processed"] > 0:
            stats["avg_processing_time"] = stats["processing_time_total"] / stats["documents_processed"]
            stats["avg_chunks_per_document"] = stats["chunks_created"] / stats["documents_processed"]
        return stats


# Singleton instance
_contextual_chunking_pipeline: Optional[ContextualChunkingPipeline] = None


def get_contextual_chunking_pipeline(config: Optional[ContextualChunkingConfig] = None) -> ContextualChunkingPipeline:
    """Get or create a contextual chunking pipeline instance."""
    global _contextual_chunking_pipeline

    if _contextual_chunking_pipeline is None or config is not None:
        _contextual_chunking_pipeline = ContextualChunkingPipeline(config)

    return _contextual_chunking_pipeline


async def process_document_contextual_chunks_async(
    extracted_content: ExtractedContent,
    tenant_id: str,
    config: Optional[ContextualChunkingConfig] = None
) -> ContextualChunkingResult:
    """
    Convenience function to process document with contextual chunking.

    Args:
        extracted_content: The extracted content from document parsing
        tenant_id: Tenant identifier for multi-tenancy
        config: Optional configuration for contextual chunking

    Returns:
        ContextualChunkingResult with processing information
    """
    pipeline = get_contextual_chunking_pipeline(config)
    return await pipeline.process_document_async(extracted_content, tenant_id)
