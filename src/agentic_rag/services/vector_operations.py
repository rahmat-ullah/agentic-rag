"""
Vector Operations Service

This module provides a high-level interface for vector operations including
batch processing, error handling, retry logic, and operation monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field

from agentic_rag.services.vector_store import (
    ChromaDBClient, VectorMetadata, VectorSearchResult, VectorOperationResult,
    get_vector_store
)
from agentic_rag.services.collection_manager import CollectionManager, get_collection_manager

logger = logging.getLogger(__name__)


@dataclass
class VectorData:
    """Container for vector data."""
    
    id: str
    embedding: List[float]
    content: str
    metadata: VectorMetadata


class BatchOperationResult(BaseModel):
    """Result from batch vector operation."""
    
    operation: str = Field(..., description="Operation type")
    total_requested: int = Field(..., description="Total vectors requested")
    successful: int = Field(..., description="Successfully processed vectors")
    failed: int = Field(..., description="Failed vectors")
    duration: float = Field(..., description="Total operation duration")
    batch_results: List[VectorOperationResult] = Field(..., description="Individual batch results")
    errors: List[str] = Field(default_factory=list, description="Error messages")


class VectorSearchOptions(BaseModel):
    """Options for vector search operations."""
    
    n_results: int = Field(default=10, description="Number of results to return")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    include_documents: bool = Field(default=True, description="Include document content")
    where_filter: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters")
    score_threshold: Optional[float] = Field(None, description="Minimum similarity score threshold")


class VectorOperationsService:
    """High-level service for vector operations with batch processing and error handling."""
    
    def __init__(self):
        self._client: Optional[ChromaDBClient] = None
        self._collection_manager: Optional[CollectionManager] = None
        
        # Batch processing settings
        self._default_batch_size = 100
        self._max_batch_size = 500
        self._max_concurrent_batches = 5
        
        # Retry settings
        self._max_retries = 3
        self._retry_delay = 1.0
        
        # Operation statistics
        self._stats = {
            "operations_total": 0,
            "operations_successful": 0,
            "operations_failed": 0,
            "vectors_processed": 0,
            "average_batch_time": 0.0,
            "last_operation": None
        }
        
        logger.info("Vector operations service initialized")
    
    async def initialize(self) -> None:
        """Initialize the vector operations service."""
        self._client = await get_vector_store()
        self._collection_manager = await get_collection_manager()
        logger.info("Vector operations service connected to ChromaDB")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def add_vectors_batch(
        self,
        vectors: List[VectorData],
        batch_size: Optional[int] = None
    ) -> BatchOperationResult:
        """
        Add vectors in batches with retry logic.
        
        Args:
            vectors: List of VectorData objects to add
            batch_size: Size of each batch (default: 100)
            
        Returns:
            BatchOperationResult with operation details
        """
        if not self._client:
            await self.initialize()
        
        start_time = time.time()
        batch_size = batch_size or self._default_batch_size
        batch_size = min(batch_size, self._max_batch_size)
        
        logger.info(f"Starting batch add operation: {len(vectors)} vectors, batch size: {batch_size}")
        
        # Split vectors into batches
        batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
        
        batch_results = []
        successful = 0
        failed = 0
        errors = []
        
        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrent_batches)
        
        async def process_batch(batch: List[VectorData]) -> VectorOperationResult:
            async with semaphore:
                try:
                    # Convert VectorData to tuple format
                    vector_tuples = [
                        (v.id, v.embedding, v.content, v.metadata)
                        for v in batch
                    ]
                    
                    result = await self._client.add_vectors(vector_tuples)
                    return result
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    return VectorOperationResult(
                        success=False,
                        operation="add_vectors_batch",
                        count=0,
                        duration=0.0,
                        error=str(e)
                    )
        
        # Execute all batches concurrently
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                errors.append(f"Batch {i}: {str(result)}")
                failed += len(batches[i])
            elif isinstance(result, VectorOperationResult):
                if result.success:
                    successful += result.count
                else:
                    failed += len(batches[i])
                    if result.error:
                        errors.append(f"Batch {i}: {result.error}")
            else:
                errors.append(f"Batch {i}: Unexpected result type")
                failed += len(batches[i])
        
        # Update statistics
        self._stats["operations_total"] += 1
        if successful > 0:
            self._stats["operations_successful"] += 1
        if failed > 0:
            self._stats["operations_failed"] += 1
        self._stats["vectors_processed"] += successful
        
        duration = time.time() - start_time
        self._stats["average_batch_time"] = (
            (self._stats["average_batch_time"] * (self._stats["operations_total"] - 1) + duration)
            / self._stats["operations_total"]
        )
        self._stats["last_operation"] = time.time()
        
        logger.info(f"Batch add completed: {successful} successful, {failed} failed, {duration:.2f}s")
        
        return BatchOperationResult(
            operation="add_vectors_batch",
            total_requested=len(vectors),
            successful=successful,
            failed=failed,
            duration=duration,
            batch_results=[r for r in batch_results if isinstance(r, VectorOperationResult)],
            errors=errors
        )
    
    async def search_vectors(
        self,
        query_embedding: List[float],
        document_kind: str,
        tenant_id: str,
        options: Optional[VectorSearchOptions] = None
    ) -> List[VectorSearchResult]:
        """
        Search vectors with enhanced options and filtering.
        
        Args:
            query_embedding: Query vector embedding
            document_kind: Document type to search in
            tenant_id: Tenant identifier
            options: Search options and filters
            
        Returns:
            List of VectorSearchResult objects
        """
        if not self._client:
            await self.initialize()
        
        if options is None:
            options = VectorSearchOptions()
        
        try:
            results = await self._client.query_vectors(
                query_embedding=query_embedding,
                document_kind=document_kind,
                tenant_id=tenant_id,
                n_results=options.n_results,
                where_filter=options.where_filter
            )
            
            # Apply score threshold if specified
            if options.score_threshold is not None:
                results = [r for r in results if r.distance <= options.score_threshold]
            
            # Filter metadata/documents if requested
            if not options.include_metadata:
                for result in results:
                    result.metadata = None
            
            if not options.include_documents:
                for result in results:
                    result.document = ""
            
            logger.info(f"Vector search completed: {len(results)} results for {document_kind}")
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    async def delete_vectors_batch(
        self,
        vector_ids: List[str],
        document_kind: str,
        tenant_id: str,
        batch_size: Optional[int] = None
    ) -> BatchOperationResult:
        """
        Delete vectors in batches.
        
        Args:
            vector_ids: List of vector IDs to delete
            document_kind: Document type
            tenant_id: Tenant identifier
            batch_size: Size of each batch
            
        Returns:
            BatchOperationResult with operation details
        """
        if not self._client:
            await self.initialize()
        
        start_time = time.time()
        batch_size = batch_size or self._default_batch_size
        
        logger.info(f"Starting batch delete operation: {len(vector_ids)} vectors")
        
        # Split into batches
        batches = [vector_ids[i:i + batch_size] for i in range(0, len(vector_ids), batch_size)]
        
        batch_results = []
        successful = 0
        failed = 0
        errors = []
        
        # Process batches sequentially for delete operations
        for i, batch in enumerate(batches):
            try:
                result = await self._client.delete_vectors(batch, document_kind, tenant_id)
                batch_results.append(result)
                
                if result.success:
                    successful += result.count
                else:
                    failed += len(batch)
                    if result.error:
                        errors.append(f"Batch {i}: {result.error}")
                        
            except Exception as e:
                errors.append(f"Batch {i}: {str(e)}")
                failed += len(batch)
        
        duration = time.time() - start_time
        logger.info(f"Batch delete completed: {successful} successful, {failed} failed, {duration:.2f}s")
        
        return BatchOperationResult(
            operation="delete_vectors_batch",
            total_requested=len(vector_ids),
            successful=successful,
            failed=failed,
            duration=duration,
            batch_results=batch_results,
            errors=errors
        )
    
    async def update_vectors_batch(
        self,
        vectors: List[VectorData],
        batch_size: Optional[int] = None
    ) -> BatchOperationResult:
        """
        Update vectors in batches.
        
        Args:
            vectors: List of VectorData objects to update
            batch_size: Size of each batch
            
        Returns:
            BatchOperationResult with operation details
        """
        if not self._client:
            await self.initialize()
        
        start_time = time.time()
        batch_size = batch_size or self._default_batch_size
        
        logger.info(f"Starting batch update operation: {len(vectors)} vectors")
        
        # Split into batches
        batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
        
        batch_results = []
        successful = 0
        failed = 0
        errors = []
        
        # Process batches
        for i, batch in enumerate(batches):
            try:
                # Convert VectorData to tuple format
                vector_tuples = [
                    (v.id, v.embedding, v.content, v.metadata)
                    for v in batch
                ]
                
                result = await self._client.update_vectors(vector_tuples)
                batch_results.append(result)
                
                if result.success:
                    successful += result.count
                else:
                    failed += len(batch)
                    if result.error:
                        errors.append(f"Batch {i}: {result.error}")
                        
            except Exception as e:
                errors.append(f"Batch {i}: {str(e)}")
                failed += len(batch)
        
        duration = time.time() - start_time
        logger.info(f"Batch update completed: {successful} successful, {failed} failed, {duration:.2f}s")
        
        return BatchOperationResult(
            operation="update_vectors_batch",
            total_requested=len(vectors),
            successful=successful,
            failed=failed,
            duration=duration,
            batch_results=batch_results,
            errors=errors
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return self._stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector operations."""
        if not self._client:
            await self.initialize()
        
        try:
            client_health = await self._client.health_check()
            
            return {
                "status": "healthy" if client_health["status"] == "healthy" else "unhealthy",
                "client_health": client_health,
                "statistics": self.get_statistics(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Vector operations health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Global vector operations service instance
_vector_ops_service: Optional[VectorOperationsService] = None


async def get_vector_operations() -> VectorOperationsService:
    """Get or create the global vector operations service instance."""
    global _vector_ops_service
    
    if _vector_ops_service is None:
        _vector_ops_service = VectorOperationsService()
        await _vector_ops_service.initialize()
    
    return _vector_ops_service
