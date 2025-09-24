"""
Embedding Batch Processing System

This module provides efficient batch processing for embedding generation
with queue management, parallel processing, status tracking, and recovery.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from agentic_rag.services.openai_client import get_openai_client, EmbeddingResponse
from agentic_rag.services.vector_store import VectorMetadata

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """Batch processing status enumeration."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class EmbeddingTask:
    """Individual embedding task within a batch."""
    
    id: str
    text: str
    metadata: VectorMetadata
    embedding: Optional[List[float]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class BatchRequest(BaseModel):
    """Request for batch embedding processing."""
    
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique batch identifier")
    tasks: List[EmbeddingTask] = Field(..., description="List of embedding tasks")
    priority: int = Field(default=5, description="Batch priority (1-10, higher is more urgent)")
    tenant_id: str = Field(..., description="Tenant identifier")
    callback_url: Optional[str] = Field(None, description="Callback URL for completion notification")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=300, description="Timeout in seconds")


class BatchResult(BaseModel):
    """Result from batch processing."""
    
    batch_id: str = Field(..., description="Batch identifier")
    status: BatchStatus = Field(..., description="Processing status")
    total_tasks: int = Field(..., description="Total number of tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    processing_time: float = Field(..., description="Total processing time")
    embeddings: List[List[float]] = Field(default_factory=list, description="Generated embeddings")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    usage: Dict[str, Any] = Field(default_factory=dict, description="API usage information")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class BatchQueue:
    """Priority queue for batch processing."""
    
    def __init__(self, max_size: int = 1000):
        self._queue: List[BatchRequest] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def put(self, batch: BatchRequest) -> None:
        """Add batch to queue with priority ordering."""
        async with self._lock:
            if len(self._queue) >= self._max_size:
                raise Exception("Batch queue is full")
            
            # Insert batch in priority order (higher priority first)
            inserted = False
            for i, existing_batch in enumerate(self._queue):
                if batch.priority > existing_batch.priority:
                    self._queue.insert(i, batch)
                    inserted = True
                    break
            
            if not inserted:
                self._queue.append(batch)
            
            logger.info(f"Added batch {batch.batch_id} to queue (priority: {batch.priority})")
    
    async def get(self) -> Optional[BatchRequest]:
        """Get next batch from queue."""
        async with self._lock:
            if self._queue:
                batch = self._queue.pop(0)
                logger.info(f"Retrieved batch {batch.batch_id} from queue")
                return batch
            return None
    
    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return len(self._queue)
    
    async def clear(self) -> None:
        """Clear all batches from queue."""
        async with self._lock:
            self._queue.clear()
            logger.info("Batch queue cleared")


class EmbeddingBatchProcessor:
    """Batch processor for embedding generation with parallel processing and recovery."""
    
    def __init__(self, max_concurrent_batches: int = 3, max_batch_size: int = 100):
        self._queue = BatchQueue()
        self._results: Dict[str, BatchResult] = {}
        self._active_batches: Dict[str, BatchRequest] = {}
        
        # Processing configuration
        self._max_concurrent_batches = max_concurrent_batches
        self._max_batch_size = max_batch_size
        self._optimal_batch_size = 50  # Optimal size for OpenAI API
        
        # Worker management
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Statistics
        self._stats = {
            "batches_processed": 0,
            "batches_failed": 0,
            "total_embeddings": 0,
            "total_processing_time": 0.0,
            "average_batch_time": 0.0,
            "last_processed": None
        }
        
        logger.info(f"Embedding batch processor initialized (max_concurrent: {max_concurrent_batches})")
    
    async def start(self) -> None:
        """Start the batch processing workers."""
        if self._running:
            return
        
        self._running = True
        
        # Start worker tasks
        for i in range(self._max_concurrent_batches):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Started {len(self._workers)} batch processing workers")
    
    async def stop(self) -> None:
        """Stop the batch processing workers."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Batch processing workers stopped")
    
    async def submit_batch(self, batch: BatchRequest) -> str:
        """
        Submit a batch for processing.
        
        Args:
            batch: Batch request to process
            
        Returns:
            Batch ID for tracking
        """
        # Validate batch size
        if len(batch.tasks) > self._max_batch_size:
            raise ValueError(f"Batch size {len(batch.tasks)} exceeds maximum {self._max_batch_size}")
        
        # Initialize result
        result = BatchResult(
            batch_id=batch.batch_id,
            status=BatchStatus.PENDING,
            total_tasks=len(batch.tasks),
            completed_tasks=0,
            failed_tasks=0,
            processing_time=0.0
        )
        
        self._results[batch.batch_id] = result
        
        # Add to queue
        await self._queue.put(batch)
        
        logger.info(f"Submitted batch {batch.batch_id} with {len(batch.tasks)} tasks")
        
        return batch.batch_id
    
    async def get_batch_status(self, batch_id: str) -> Optional[BatchResult]:
        """Get status of a batch."""
        return self._results.get(batch_id)
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a pending or processing batch."""
        if batch_id in self._results:
            result = self._results[batch_id]
            if result.status in [BatchStatus.PENDING, BatchStatus.PROCESSING]:
                result.status = BatchStatus.CANCELLED
                result.completed_at = datetime.utcnow()
                
                # Remove from active batches if processing
                if batch_id in self._active_batches:
                    del self._active_batches[batch_id]
                
                logger.info(f"Cancelled batch {batch_id}")
                return True
        
        return False
    
    async def _worker(self, worker_name: str) -> None:
        """Worker task for processing batches."""
        logger.info(f"Batch worker {worker_name} started")
        
        while self._running:
            try:
                # Get next batch from queue
                batch = await self._queue.get()
                if not batch:
                    await asyncio.sleep(1)  # No batches available
                    continue
                
                # Check if batch was cancelled
                result = self._results.get(batch.batch_id)
                if result and result.status == BatchStatus.CANCELLED:
                    continue
                
                # Process batch with semaphore
                async with self._semaphore:
                    await self._process_batch(batch, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"Batch worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Batch worker {worker_name} error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info(f"Batch worker {worker_name} stopped")
    
    async def _process_batch(self, batch: BatchRequest, worker_name: str) -> None:
        """Process a single batch."""
        batch_id = batch.batch_id
        result = self._results[batch_id]
        
        logger.info(f"Worker {worker_name} processing batch {batch_id}")
        
        try:
            # Mark as processing
            result.status = BatchStatus.PROCESSING
            self._active_batches[batch_id] = batch
            
            start_time = time.time()
            
            # Split tasks into optimal sub-batches
            sub_batches = self._split_into_sub_batches(batch.tasks)
            
            # Process sub-batches
            all_embeddings = []
            total_usage = {"prompt_tokens": 0, "total_tokens": 0}
            
            for sub_batch in sub_batches:
                try:
                    # Extract texts for embedding
                    texts = [task.text for task in sub_batch]
                    
                    # Generate embeddings
                    openai_client = await get_openai_client()
                    response = await openai_client.generate_embeddings(texts)
                    
                    # Update tasks with embeddings
                    for i, task in enumerate(sub_batch):
                        if i < len(response.embeddings):
                            task.embedding = response.embeddings[i]
                            result.completed_tasks += 1
                        else:
                            task.error = "Missing embedding in response"
                            result.failed_tasks += 1
                    
                    all_embeddings.extend(response.embeddings)
                    
                    # Accumulate usage
                    total_usage["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                    total_usage["total_tokens"] += response.usage.get("total_tokens", 0)
                    
                except Exception as e:
                    logger.error(f"Sub-batch processing failed: {e}")
                    # Mark all tasks in sub-batch as failed
                    for task in sub_batch:
                        task.error = str(e)
                        result.failed_tasks += 1
                    
                    result.errors.append(f"Sub-batch error: {str(e)}")
            
            # Update result
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.embeddings = all_embeddings
            result.usage = total_usage
            result.completed_at = datetime.utcnow()
            
            # Determine final status
            if result.failed_tasks == 0:
                result.status = BatchStatus.COMPLETED
            elif result.completed_tasks > 0:
                result.status = BatchStatus.COMPLETED  # Partial success
            else:
                result.status = BatchStatus.FAILED
            
            # Update statistics
            self._update_stats(result)
            
            logger.info(f"Batch {batch_id} completed: {result.completed_tasks}/{result.total_tasks} successful")
            
        except Exception as e:
            logger.error(f"Batch {batch_id} processing failed: {e}")
            result.status = BatchStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.utcnow()
            result.processing_time = time.time() - start_time
            
            self._stats["batches_failed"] += 1
        
        finally:
            # Remove from active batches
            if batch_id in self._active_batches:
                del self._active_batches[batch_id]
    
    def _split_into_sub_batches(self, tasks: List[EmbeddingTask]) -> List[List[EmbeddingTask]]:
        """Split tasks into optimal sub-batches for API calls."""
        sub_batches = []
        
        for i in range(0, len(tasks), self._optimal_batch_size):
            sub_batch = tasks[i:i + self._optimal_batch_size]
            sub_batches.append(sub_batch)
        
        return sub_batches
    
    def _update_stats(self, result: BatchResult) -> None:
        """Update processing statistics."""
        self._stats["batches_processed"] += 1
        self._stats["total_embeddings"] += result.completed_tasks
        self._stats["total_processing_time"] += result.processing_time
        self._stats["average_batch_time"] = (
            self._stats["total_processing_time"] / self._stats["batches_processed"]
        )
        self._stats["last_processed"] = datetime.utcnow()
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        queue_size = await self._queue.size()
        
        return {
            "queue_size": queue_size,
            "active_batches": len(self._active_batches),
            "max_concurrent": self._max_concurrent_batches,
            "workers_running": len(self._workers),
            "is_running": self._running
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on batch processor."""
        try:
            queue_status = await self.get_queue_status()
            
            return {
                "status": "healthy" if self._running else "stopped",
                "queue_status": queue_status,
                "statistics": self.get_statistics(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Batch processor health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Global batch processor instance
_batch_processor: Optional[EmbeddingBatchProcessor] = None


async def get_batch_processor() -> EmbeddingBatchProcessor:
    """Get or create the global batch processor instance."""
    global _batch_processor
    
    if _batch_processor is None:
        _batch_processor = EmbeddingBatchProcessor()
        await _batch_processor.start()
    
    return _batch_processor


async def close_batch_processor() -> None:
    """Close the global batch processor instance."""
    global _batch_processor
    
    if _batch_processor:
        await _batch_processor.stop()
        _batch_processor = None
