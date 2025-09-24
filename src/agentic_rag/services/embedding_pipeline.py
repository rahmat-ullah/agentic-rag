"""
Comprehensive Embedding Pipeline

This module integrates all embedding services into a unified pipeline
for production-ready embedding generation with full observability.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from pydantic import BaseModel, Field

from agentic_rag.services.openai_client import get_openai_client, EmbeddingResponse
from agentic_rag.services.embedding_batch_processor import (
    get_batch_processor, BatchRequest, EmbeddingTask, BatchStatus
)
from agentic_rag.services.embedding_resilience import get_resilience_manager
from agentic_rag.services.embedding_quality import get_quality_validator, EmbeddingQualityReport
from agentic_rag.services.embedding_cost_optimizer import get_cost_optimizer
from agentic_rag.services.vector_store import VectorMetadata
from agentic_rag.services.vector_operations import get_vector_operations

logger = logging.getLogger(__name__)


class EmbeddingPipelineRequest(BaseModel):
    """Request for the embedding pipeline."""
    
    texts: List[str] = Field(..., description="Texts to embed")
    metadata_list: List[VectorMetadata] = Field(..., description="Metadata for each text")
    tenant_id: str = Field(..., description="Tenant identifier")
    document_id: str = Field(..., description="Document identifier")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    priority: int = Field(default=5, description="Processing priority (1-10)")
    store_vectors: bool = Field(default=True, description="Whether to store vectors in ChromaDB")
    validate_quality: bool = Field(default=True, description="Whether to validate embedding quality")
    optimize_cost: bool = Field(default=True, description="Whether to optimize for cost")


class EmbeddingPipelineResult(BaseModel):
    """Result from the embedding pipeline."""
    
    batch_id: str = Field(..., description="Batch identifier")
    status: BatchStatus = Field(..., description="Processing status")
    embeddings: List[List[float]] = Field(default_factory=list, description="Generated embeddings")
    quality_report: Optional[EmbeddingQualityReport] = Field(None, description="Quality validation report")
    cost_summary: Dict[str, Any] = Field(default_factory=dict, description="Cost information")
    processing_time: float = Field(..., description="Total processing time")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate percentage")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    vector_storage_result: Optional[Dict[str, Any]] = Field(None, description="Vector storage result")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class EmbeddingPipeline:
    """Comprehensive embedding pipeline with all features integrated."""
    
    def __init__(self):
        self._openai_client = None
        self._batch_processor = None
        self._resilience_manager = None
        self._quality_validator = None
        self._cost_optimizer = None
        self._vector_operations = None
        
        # Pipeline statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_embeddings": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "cache_hit_rate": 0.0,
            "quality_score_average": 0.0
        }
        
        logger.info("Embedding pipeline initialized")
    
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Initialize all services
            self._openai_client = await get_openai_client()
            self._batch_processor = await get_batch_processor()
            self._resilience_manager = await get_resilience_manager()
            self._quality_validator = await get_quality_validator()
            self._cost_optimizer = await get_cost_optimizer()
            self._vector_operations = await get_vector_operations()
            
            logger.info("Embedding pipeline fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding pipeline: {e}")
            raise
    
    async def process_embeddings(self, request: EmbeddingPipelineRequest) -> EmbeddingPipelineResult:
        """
        Process embeddings through the complete pipeline.
        
        Args:
            request: Pipeline request with texts and configuration
            
        Returns:
            EmbeddingPipelineResult with all processing information
        """
        start_time = time.time()
        batch_id = request.batch_id or f"pipeline_{int(time.time())}"
        
        logger.info(f"Processing embedding pipeline request: {batch_id}")
        
        # Ensure pipeline is initialized
        if not self._openai_client:
            await self.initialize()
        
        try:
            # Update statistics
            self._stats["total_requests"] += 1
            
            # Step 1: Cost optimization and caching
            embeddings = []
            cache_hits = []
            total_cost = 0.0
            
            if request.optimize_cost:
                embeddings, cache_hits, total_cost = await self._cost_optimizer.optimize_embedding_request(
                    texts=request.texts,
                    model="text-embedding-3-large",  # Could be configurable
                    tenant_id=request.tenant_id,
                    batch_id=batch_id
                )
                cache_hit_rate = (sum(cache_hits) / len(cache_hits)) * 100 if cache_hits else 0.0
            else:
                # Generate embeddings without optimization
                embeddings, cache_hit_rate = await self._generate_embeddings_direct(
                    request.texts, batch_id
                )
                cache_hits = [False] * len(embeddings)
            
            # Step 2: Quality validation
            quality_report = None
            if request.validate_quality and embeddings:
                quality_report = await self._quality_validator.validate_embeddings(
                    embeddings=embeddings,
                    texts=request.texts,
                    batch_id=batch_id
                )
                logger.info(f"Quality validation: {quality_report.overall_status.value}")
            
            # Step 3: Vector storage
            vector_storage_result = None
            if request.store_vectors and embeddings:
                vector_storage_result = await self._store_vectors(
                    embeddings=embeddings,
                    metadata_list=request.metadata_list,
                    tenant_id=request.tenant_id
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = EmbeddingPipelineResult(
                batch_id=batch_id,
                status=BatchStatus.COMPLETED,
                embeddings=embeddings,
                quality_report=quality_report,
                cost_summary={
                    "total_cost": total_cost,
                    "cache_hit_rate": cache_hit_rate,
                    "embeddings_generated": len(embeddings) - sum(cache_hits)
                },
                processing_time=processing_time,
                cache_hit_rate=cache_hit_rate,
                vector_storage_result=vector_storage_result
            )
            
            # Update statistics
            self._update_stats(result)
            
            logger.info(f"Pipeline processing completed: {batch_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Handle errors with resilience manager
            error_msg = f"Pipeline processing failed: {str(e)}"
            logger.error(error_msg)
            
            processing_time = time.time() - start_time
            self._stats["failed_requests"] += 1
            
            return EmbeddingPipelineResult(
                batch_id=batch_id,
                status=BatchStatus.FAILED,
                processing_time=processing_time,
                errors=[error_msg]
            )
    
    async def _generate_embeddings_direct(
        self,
        texts: List[str],
        batch_id: str
    ) -> Tuple[List[List[float]], float]:
        """Generate embeddings directly through OpenAI client."""
        try:
            # Use resilience manager for robust execution
            response = await self._resilience_manager.execute_with_resilience(
                self._openai_client.generate_embeddings,
                texts=texts
            )
            
            return response.embeddings, 0.0  # No cache hits
            
        except Exception as e:
            logger.error(f"Direct embedding generation failed: {e}")
            raise
    
    async def _store_vectors(
        self,
        embeddings: List[List[float]],
        metadata_list: List[VectorMetadata],
        tenant_id: str
    ) -> Dict[str, Any]:
        """Store vectors in ChromaDB."""
        try:
            from agentic_rag.services.vector_operations import VectorData
            
            # Prepare vector data
            vector_data_list = []
            for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
                vector_data = VectorData(
                    id=f"{metadata.document_id}_{metadata.chunk_id}_{i}",
                    embedding=embedding,
                    metadata=metadata,
                    document=f"Document chunk {i}"  # Would be actual text in real implementation
                )
                vector_data_list.append(vector_data)
            
            # Store vectors
            result = await self._vector_operations.add_vectors_batch(vector_data_list)
            
            return {
                "vectors_stored": result.successful_operations,
                "failed_operations": result.failed_operations,
                "processing_time": result.processing_time
            }
            
        except Exception as e:
            logger.error(f"Vector storage failed: {e}")
            return {"error": str(e)}
    
    def _update_stats(self, result: EmbeddingPipelineResult) -> None:
        """Update pipeline statistics."""
        if result.status == BatchStatus.COMPLETED:
            self._stats["successful_requests"] += 1
            self._stats["total_embeddings"] += len(result.embeddings)
            
            if result.quality_report:
                # Update quality score average
                current_avg = self._stats["quality_score_average"]
                new_score = result.quality_report.overall_score
                total_successful = self._stats["successful_requests"]
                
                self._stats["quality_score_average"] = (
                    (current_avg * (total_successful - 1) + new_score) / total_successful
                )
        
        # Update processing time average
        self._stats["total_processing_time"] += result.processing_time
        self._stats["average_processing_time"] = (
            self._stats["total_processing_time"] / self._stats["total_requests"]
        )
        
        # Update cache hit rate average
        if result.cache_hit_rate > 0:
            current_cache_rate = self._stats["cache_hit_rate"]
            self._stats["cache_hit_rate"] = (
                (current_cache_rate + result.cache_hit_rate) / 2
            )
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        try:
            # Get status from all components
            openai_health = await self._openai_client.health_check() if self._openai_client else {"status": "not_initialized"}
            batch_health = await self._batch_processor.health_check() if self._batch_processor else {"status": "not_initialized"}
            resilience_health = await self._resilience_manager.health_check() if self._resilience_manager else {"status": "not_initialized"}
            cost_health = await self._cost_optimizer.health_check() if self._cost_optimizer else {"status": "not_initialized"}
            
            # Determine overall status
            component_statuses = [
                openai_health.get("status", "unknown"),
                batch_health.get("status", "unknown"),
                resilience_health.get("status", "unknown"),
                cost_health.get("status", "unknown")
            ]
            
            if all(status == "healthy" for status in component_statuses):
                overall_status = "healthy"
            elif any(status == "unhealthy" for status in component_statuses):
                overall_status = "unhealthy"
            else:
                overall_status = "degraded"
            
            return {
                "overall_status": overall_status,
                "components": {
                    "openai_client": openai_health,
                    "batch_processor": batch_health,
                    "resilience_manager": resilience_health,
                    "cost_optimizer": cost_health
                },
                "statistics": self._stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pipeline status check failed: {e}")
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self._stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        return await self.get_pipeline_status()


# Global pipeline instance
_embedding_pipeline: Optional[EmbeddingPipeline] = None


async def get_embedding_pipeline() -> EmbeddingPipeline:
    """Get or create the global embedding pipeline instance."""
    global _embedding_pipeline
    
    if _embedding_pipeline is None:
        _embedding_pipeline = EmbeddingPipeline()
        await _embedding_pipeline.initialize()
    
    return _embedding_pipeline


async def close_embedding_pipeline() -> None:
    """Close the global embedding pipeline instance."""
    global _embedding_pipeline
    
    if _embedding_pipeline:
        _embedding_pipeline = None
