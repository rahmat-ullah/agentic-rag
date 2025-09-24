"""
Vector Indexing Pipeline

This module provides automatic vector indexing for document chunks
after document processing completion with comprehensive monitoring.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agentic_rag.services.embedding_pipeline import get_embedding_pipeline, EmbeddingPipelineRequest
from agentic_rag.services.vector_operations import get_vector_operations, VectorData
from agentic_rag.services.vector_store import VectorMetadata
from agentic_rag.services.chunking.deduplication_chunker import DeduplicatedChunk
from agentic_rag.services.metadata_validator import validate_chunk_metadata, MetadataValidationLevel
from agentic_rag.services.metadata_indexing import prepare_metadata_for_vector_storage, IndexingStrategy
from agentic_rag.services.indexing_error_handler import get_error_handler, RecoveryAction
from agentic_rag.services.batch_indexing_optimizer import get_batch_optimizer, BatchItem, BatchConfiguration
from agentic_rag.services.indexing_monitor import get_indexing_monitor, log_indexing_event, MonitoringConfiguration
from agentic_rag.models.database import DocumentChunk, Document
from agentic_rag.config import get_settings

logger = logging.getLogger(__name__)


class IndexingStatus(str, Enum):
    """Status of vector indexing operations."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class IndexingPriority(int, Enum):
    """Priority levels for indexing operations."""
    
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10


@dataclass
class IndexingTask:
    """Individual indexing task for a document chunk."""
    
    task_id: str
    document_id: str
    chunk_id: str
    tenant_id: str
    chunk_text: str
    chunk_metadata: Dict[str, Any]
    priority: IndexingPriority = IndexingPriority.NORMAL
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class IndexingRequest(BaseModel):
    """Request for vector indexing of document chunks."""
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    document_id: str = Field(..., description="Document identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    chunks: List[Dict[str, Any]] = Field(..., description="List of chunks to index")
    priority: IndexingPriority = Field(default=IndexingPriority.NORMAL, description="Indexing priority")
    callback_url: Optional[str] = Field(None, description="Callback URL for completion notification")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")


class IndexingResult(BaseModel):
    """Result from vector indexing operation."""
    
    request_id: str = Field(..., description="Request identifier")
    document_id: str = Field(..., description="Document identifier")
    status: IndexingStatus = Field(..., description="Indexing status")
    total_chunks: int = Field(..., description="Total chunks to index")
    indexed_chunks: int = Field(..., description="Successfully indexed chunks")
    failed_chunks: int = Field(..., description="Failed chunk indexing")
    processing_time: float = Field(..., description="Total processing time")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    vector_ids: List[str] = Field(default_factory=list, description="Generated vector IDs")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class IndexingQueue:
    """Priority queue for indexing tasks."""
    
    def __init__(self, max_size: int = 10000):
        self._queue: List[IndexingTask] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()
        
        logger.info(f"Indexing queue initialized with max size: {max_size}")
    
    async def enqueue(self, task: IndexingTask) -> None:
        """Add task to queue with priority ordering."""
        async with self._lock:
            if len(self._queue) >= self._max_size:
                raise Exception("Indexing queue is full")
            
            # Insert task in priority order (higher priority first)
            inserted = False
            for i, existing_task in enumerate(self._queue):
                if task.priority.value > existing_task.priority.value:
                    self._queue.insert(i, task)
                    inserted = True
                    break
            
            if not inserted:
                self._queue.append(task)
            
            logger.debug(f"Enqueued indexing task {task.task_id} with priority {task.priority.value}")
    
    async def dequeue(self) -> Optional[IndexingTask]:
        """Get next task from queue."""
        async with self._lock:
            if self._queue:
                task = self._queue.pop(0)
                logger.debug(f"Dequeued indexing task {task.task_id}")
                return task
            return None
    
    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return len(self._queue)
    
    async def clear(self) -> None:
        """Clear all tasks from queue."""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            logger.info(f"Cleared {count} tasks from indexing queue")


class VectorIndexingPipeline:
    """Automatic vector indexing pipeline for document chunks."""
    
    def __init__(self, max_concurrent_workers: int = 3, batch_size: int = 50):
        self._queue = IndexingQueue()
        self._results: Dict[str, IndexingResult] = {}
        self._active_requests: Dict[str, IndexingRequest] = {}
        
        # Worker configuration
        self._max_concurrent_workers = max_concurrent_workers
        self._batch_size = batch_size
        self._workers: List[asyncio.Task] = []
        self._running = False
        
        # Service dependencies
        self._embedding_pipeline = None
        self._vector_operations = None
        self._error_handler = None
        self._batch_optimizer = None
        self._monitor = None
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "total_chunks_indexed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "queue_high_water_mark": 0
        }
        
        logger.info(f"Vector indexing pipeline initialized (workers: {max_concurrent_workers}, batch_size: {batch_size})")
    
    async def start(self) -> None:
        """Start the indexing pipeline workers."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize dependencies
        self._embedding_pipeline = await get_embedding_pipeline()
        self._vector_operations = await get_vector_operations()
        self._error_handler = await get_error_handler()

        # Initialize batch optimizer with configuration
        batch_config = BatchConfiguration(
            min_batch_size=5,
            max_batch_size=self._batch_size,
            optimal_batch_size=min(self._batch_size, 25),
            max_concurrent_batches=self._max_concurrent_workers,
            batch_timeout=300.0,
            adaptive_sizing=True
        )
        self._batch_optimizer = await get_batch_optimizer(batch_config)

        # Initialize monitoring
        monitor_config = MonitoringConfiguration(
            collection_interval=30.0,
            retention_period=3600,
            error_rate_threshold=0.1,
            latency_threshold=10.0,
            queue_size_threshold=1000,
            enable_alerting=True
        )
        self._monitor = await get_indexing_monitor(monitor_config)
        
        # Start worker tasks
        for i in range(self._max_concurrent_workers):
            worker = asyncio.create_task(self._worker(f"indexing-worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Started {len(self._workers)} indexing workers")
    
    async def stop(self) -> None:
        """Stop the indexing pipeline workers."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Indexing pipeline workers stopped")
    
    async def submit_indexing_request(self, request: IndexingRequest) -> str:
        """
        Submit a document for vector indexing.
        
        Args:
            request: Indexing request with document chunks
            
        Returns:
            Request ID for tracking
        """
        # Validate request
        if not request.chunks:
            raise ValueError("No chunks provided for indexing")
        
        # Create indexing tasks
        tasks = []
        for i, chunk_data in enumerate(request.chunks):
            task = IndexingTask(
                task_id=f"{request.request_id}_{i}",
                document_id=request.document_id,
                chunk_id=chunk_data.get("chunk_id", f"chunk_{i}"),
                tenant_id=request.tenant_id,
                chunk_text=chunk_data.get("text", ""),
                chunk_metadata=chunk_data.get("metadata", {}),
                priority=request.priority
            )
            tasks.append(task)
        
        # Initialize result
        result = IndexingResult(
            request_id=request.request_id,
            document_id=request.document_id,
            status=IndexingStatus.PENDING,
            total_chunks=len(tasks),
            indexed_chunks=0,
            failed_chunks=0,
            processing_time=0.0
        )
        
        self._results[request.request_id] = result
        self._active_requests[request.request_id] = request
        
        # Enqueue tasks
        for task in tasks:
            await self._queue.enqueue(task)
        
        # Update statistics
        self._stats["total_requests"] += 1
        queue_size = await self._queue.size()
        self._stats["queue_high_water_mark"] = max(self._stats["queue_high_water_mark"], queue_size)

        # Log monitoring event
        await log_indexing_event("request_started",
                                request_id=request.request_id,
                                chunk_count=len(tasks),
                                tenant_id=str(request.tenant_id),
                                queue_size=queue_size)

        logger.info(f"Submitted indexing request {request.request_id} with {len(tasks)} chunks")
        
        return request.request_id
    
    async def get_indexing_status(self, request_id: str) -> Optional[IndexingResult]:
        """Get status of an indexing request."""
        return self._results.get(request_id)
    
    async def cancel_indexing_request(self, request_id: str) -> bool:
        """Cancel a pending or processing indexing request."""
        if request_id in self._results:
            result = self._results[request_id]
            if result.status in [IndexingStatus.PENDING, IndexingStatus.PROCESSING]:
                result.status = IndexingStatus.CANCELLED
                result.completed_at = datetime.utcnow()
                
                # Remove from active requests
                if request_id in self._active_requests:
                    del self._active_requests[request_id]
                
                logger.info(f"Cancelled indexing request {request_id}")
                return True
        
        return False
    
    async def _worker(self, worker_name: str) -> None:
        """Worker task for processing indexing tasks."""
        logger.info(f"Indexing worker {worker_name} started")
        
        while self._running:
            try:
                # Get next task from queue
                task = await self._queue.dequeue()
                if not task:
                    await asyncio.sleep(1)  # No tasks available
                    continue
                
                # Process task
                await self._process_indexing_task(task, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"Indexing worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Indexing worker {worker_name} error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info(f"Indexing worker {worker_name} stopped")
    
    async def _process_indexing_task(self, task: IndexingTask, worker_name: str) -> None:
        """Process a single indexing task."""
        logger.debug(f"Worker {worker_name} processing task {task.task_id}")
        
        # Find the request this task belongs to
        request_id = task.task_id.split('_')[0]
        result = self._results.get(request_id)
        
        if not result:
            logger.error(f"No result found for task {task.task_id}")
            return
        
        try:
            # Mark as processing
            if result.status == IndexingStatus.PENDING:
                result.status = IndexingStatus.PROCESSING

            # Validate and prepare metadata
            metadata_preparation = await prepare_metadata_for_vector_storage(
                metadata=task.chunk_metadata,
                validate=True,
                strategy=IndexingStrategy.SELECTIVE
            )

            if metadata_preparation.errors:
                logger.warning(f"Metadata validation warnings for task {task.task_id}: {metadata_preparation.errors}")

            # Create vector metadata with validated data
            validated_metadata = task.chunk_metadata.copy()

            # Ensure required fields are present
            validated_metadata.update({
                "tenant_id": task.tenant_id,
                "document_id": task.document_id,
                "chunk_id": task.chunk_id,
                "created_at": datetime.utcnow().isoformat()
            })

            vector_metadata = VectorMetadata(
                tenant_id=task.tenant_id,
                document_id=task.document_id,
                chunk_id=task.chunk_id,
                document_kind=validated_metadata.get("document_kind", "UNKNOWN"),
                created_at=validated_metadata.get("created_at"),
                section_path=validated_metadata.get("section_path", []),
                page_from=validated_metadata.get("page_from"),
                page_to=validated_metadata.get("page_to"),
                token_count=validated_metadata.get("token_count"),
                is_table=validated_metadata.get("is_table", False)
            )
            
            # Generate embedding
            embedding_request = EmbeddingPipelineRequest(
                texts=[task.chunk_text],
                metadata_list=[vector_metadata],
                tenant_id=task.tenant_id,
                document_id=task.document_id,
                store_vectors=False,  # We'll store manually
                validate_quality=True,
                optimize_cost=True
            )
            
            embedding_result = await self._embedding_pipeline.process_embeddings(embedding_request)
            
            if embedding_result.status.value != "completed" or not embedding_result.embeddings:
                raise Exception(f"Embedding generation failed: {embedding_result.errors}")
            
            # Store vector
            vector_data = VectorData(
                id=f"{task.document_id}_{task.chunk_id}",
                embedding=embedding_result.embeddings[0],
                metadata=vector_metadata,
                document=task.chunk_text
            )
            
            vector_result = await self._vector_operations.add_vectors_batch([vector_data])
            
            if vector_result.failed_operations > 0:
                raise Exception(f"Vector storage failed: {vector_result.errors}")
            
            # Update result
            result.indexed_chunks += 1
            result.vector_ids.append(vector_data.id)
            
            logger.debug(f"Successfully indexed task {task.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to index task {task.task_id}: {e}")

            # Handle error with error handler
            task_data = {
                "task_id": task.task_id,
                "document_id": task.document_id,
                "chunk_id": task.chunk_id,
                "tenant_id": task.tenant_id,
                "chunk_text": task.chunk_text,
                "chunk_metadata": task.chunk_metadata
            }

            recovery_action = await self._error_handler.handle_error(
                exception=e,
                task_id=task.task_id,
                task_data=task_data,
                retry_count=0  # TODO: Track retry count per task
            )

            result.failed_chunks += 1
            result.errors.append(f"Task {task.task_id}: {str(e)} (Recovery: {recovery_action.value})")

            # Handle specific recovery actions
            if recovery_action == RecoveryAction.STOP_PROCESSING:
                logger.critical(f"Stopping indexing pipeline due to critical error in task {task.task_id}")
                # TODO: Implement pipeline shutdown
            elif recovery_action == RecoveryAction.RETRY_BACKOFF:
                # TODO: Implement task retry with backoff
                logger.info(f"Task {task.task_id} will be retried with backoff")
            elif recovery_action == RecoveryAction.SEND_TO_DLQ:
                logger.warning(f"Task {task.task_id} sent to dead letter queue")
        
        # Check if request is complete
        if result.indexed_chunks + result.failed_chunks >= result.total_chunks:
            await self._complete_indexing_request(request_id)
    
    async def _complete_indexing_request(self, request_id: str) -> None:
        """Complete an indexing request and update statistics."""
        result = self._results.get(request_id)
        if not result:
            return
        
        # Determine final status
        if result.failed_chunks == 0:
            result.status = IndexingStatus.COMPLETED
        elif result.indexed_chunks > 0:
            result.status = IndexingStatus.COMPLETED  # Partial success
        else:
            result.status = IndexingStatus.FAILED
        
        result.completed_at = datetime.utcnow()
        result.processing_time = (result.completed_at - result.created_at).total_seconds()
        
        # Update statistics
        if result.status == IndexingStatus.COMPLETED:
            self._stats["completed_requests"] += 1
        else:
            self._stats["failed_requests"] += 1

        self._stats["total_chunks_indexed"] += result.indexed_chunks
        self._stats["total_processing_time"] += result.processing_time

        # Log monitoring event
        event_type = "request_completed" if result.status == IndexingStatus.COMPLETED else "request_failed"
        await log_indexing_event(event_type,
                                request_id=request_id,
                                processing_time=result.processing_time,
                                indexed_chunks=result.indexed_chunks,
                                failed_chunks=result.failed_chunks,
                                success_rate=result.indexed_chunks / result.total_chunks if result.total_chunks > 0 else 0.0)
        
        if self._stats["completed_requests"] > 0:
            self._stats["average_processing_time"] = (
                self._stats["total_processing_time"] / self._stats["completed_requests"]
            )
        
        # Remove from active requests
        if request_id in self._active_requests:
            del self._active_requests[request_id]
        
        logger.info(f"Completed indexing request {request_id}: {result.indexed_chunks}/{result.total_chunks} successful")
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        queue_size = await self._queue.size()
        
        return {
            "queue_size": queue_size,
            "active_requests": len(self._active_requests),
            "max_concurrent_workers": self._max_concurrent_workers,
            "workers_running": len(self._workers),
            "is_running": self._running,
            "batch_size": self._batch_size
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        return self._stats.copy()
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        if self._error_handler:
            return self._error_handler.get_error_statistics()
        return {}

    async def get_dlq_items(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get items from the dead letter queue."""
        if self._error_handler:
            dlq_items = await self._error_handler.get_dlq_items(limit)
            return [item.dict() for item in dlq_items]
        return []

    async def retry_dlq_item(self, task_id: str) -> bool:
        """Retry a specific item from the dead letter queue."""
        if self._error_handler:
            task_data = await self._error_handler.retry_dlq_item(task_id)
            if task_data:
                # TODO: Re-enqueue the task for processing
                logger.info(f"Re-enqueuing DLQ task {task_id} for retry")
                return True
        return False

    async def clear_dlq(self) -> int:
        """Clear all items from the dead letter queue."""
        if self._error_handler:
            return await self._error_handler.clear_dlq()
        return 0

    async def submit_batch_indexing_request(self, request: IndexingRequest) -> str:
        """
        Submit a batch indexing request using optimized batch processing.

        Args:
            request: Indexing request with document chunks

        Returns:
            Request ID for tracking
        """
        # Convert chunks to batch items
        batch_items = []
        for i, chunk_data in enumerate(request.chunks):
            batch_item = BatchItem(
                item_id=f"{request.request_id}_{i}",
                data={
                    "task_id": f"{request.request_id}_{i}",
                    "document_id": request.document_id,
                    "chunk_id": chunk_data.get("chunk_id", f"chunk_{i}"),
                    "tenant_id": request.tenant_id,
                    "chunk_text": chunk_data.get("text", ""),
                    "chunk_metadata": chunk_data.get("metadata", {}),
                    "request_id": request.request_id
                },
                priority=request.priority.value,
                estimated_processing_time=2.0  # Estimate 2 seconds per chunk
            )
            batch_items.append(batch_item)

        # Add items to batch optimizer
        await self._batch_optimizer.add_items(batch_items)

        # Initialize result tracking
        result = IndexingResult(
            request_id=request.request_id,
            document_id=request.document_id,
            status=IndexingStatus.PENDING,
            total_chunks=len(batch_items),
            indexed_chunks=0,
            failed_chunks=0,
            processing_time=0.0
        )

        self._results[request.request_id] = result
        self._active_requests[request.request_id] = request

        # Update statistics
        self._stats["total_requests"] += 1

        logger.info(f"Submitted batch indexing request {request.request_id} with {len(batch_items)} items")

        return request.request_id

    async def process_batch_items(self) -> None:
        """Process items using batch optimization."""
        if not self._batch_optimizer:
            return

        # Create optimal batches
        batches = await self._batch_optimizer.create_optimal_batches()

        if not batches:
            return

        # Process batches
        for batch in batches:
            asyncio.create_task(self._process_optimized_batch(batch))

    async def _process_optimized_batch(self, batch) -> None:
        """Process a batch using the batch optimizer."""
        try:
            # Process the batch using the optimizer
            completed_batch = await self._batch_optimizer.process_batch(
                batch,
                self._process_batch_item
            )

            # Update request results based on batch completion
            await self._update_results_from_batch(completed_batch)

        except Exception as e:
            logger.error(f"Failed to process optimized batch {batch.batch_id}: {e}")

    async def _process_batch_item(self, batch_item: BatchItem) -> Any:
        """Process a single batch item (chunk indexing)."""
        data = batch_item.data

        try:
            # Extract task information
            task_id = data["task_id"]
            document_id = data["document_id"]
            chunk_id = data["chunk_id"]
            tenant_id = data["tenant_id"]
            chunk_text = data["chunk_text"]
            chunk_metadata = data["chunk_metadata"]

            # Validate and prepare metadata
            metadata_preparation = await prepare_metadata_for_vector_storage(
                metadata=chunk_metadata,
                validate=True,
                strategy=IndexingStrategy.SELECTIVE
            )

            # Create vector metadata
            validated_metadata = chunk_metadata.copy()
            validated_metadata.update({
                "tenant_id": tenant_id,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "created_at": datetime.utcnow().isoformat()
            })

            vector_metadata = VectorMetadata(
                tenant_id=tenant_id,
                document_id=document_id,
                chunk_id=chunk_id,
                document_kind=validated_metadata.get("document_kind", "UNKNOWN"),
                created_at=validated_metadata.get("created_at"),
                section_path=validated_metadata.get("section_path", []),
                page_from=validated_metadata.get("page_from"),
                page_to=validated_metadata.get("page_to"),
                token_count=validated_metadata.get("token_count"),
                is_table=validated_metadata.get("is_table", False)
            )

            # Generate embedding
            embedding_request = EmbeddingPipelineRequest(
                texts=[chunk_text],
                metadata_list=[vector_metadata],
                tenant_id=tenant_id,
                document_id=document_id,
                store_vectors=False,
                validate_quality=True,
                optimize_cost=True
            )

            embedding_result = await self._embedding_pipeline.process_embeddings(embedding_request)

            if embedding_result.status.value != "completed" or not embedding_result.embeddings:
                raise Exception(f"Embedding generation failed: {embedding_result.errors}")

            # Store vector
            vector_data = VectorData(
                id=f"{document_id}_{chunk_id}",
                embedding=embedding_result.embeddings[0],
                metadata=vector_metadata,
                document=chunk_text
            )

            vector_result = await self._vector_operations.add_vectors_batch([vector_data])

            if vector_result.failed_operations > 0:
                raise Exception(f"Vector storage failed: {vector_result.errors}")

            logger.debug(f"Successfully processed batch item {batch_item.item_id}")
            return {"success": True, "vector_id": vector_data.id}

        except Exception as e:
            logger.error(f"Failed to process batch item {batch_item.item_id}: {e}")

            # Handle error with error handler
            recovery_action = await self._error_handler.handle_error(
                exception=e,
                task_id=batch_item.item_id,
                task_data=batch_item.data,
                retry_count=batch_item.retry_count
            )

            raise e  # Re-raise to be handled by batch processor

    async def _update_results_from_batch(self, completed_batch) -> None:
        """Update request results based on completed batch."""
        # Group batch items by request ID
        request_updates = {}

        for item in completed_batch.items:
            request_id = item.data.get("request_id")
            if request_id and request_id in self._results:
                if request_id not in request_updates:
                    request_updates[request_id] = {"success": 0, "failed": 0, "vector_ids": []}

        # Update based on batch results
        for i, item in enumerate(completed_batch.items):
            request_id = item.data.get("request_id")
            if request_id in request_updates:
                if i < completed_batch.success_count:
                    request_updates[request_id]["success"] += 1
                    # TODO: Get actual vector ID from processing result
                    request_updates[request_id]["vector_ids"].append(f"vector_{item.item_id}")
                else:
                    request_updates[request_id]["failed"] += 1

        # Apply updates to results
        for request_id, updates in request_updates.items():
            result = self._results[request_id]
            result.indexed_chunks += updates["success"]
            result.failed_chunks += updates["failed"]
            result.vector_ids.extend(updates["vector_ids"])

            # Check if request is complete
            if result.indexed_chunks + result.failed_chunks >= result.total_chunks:
                await self._complete_indexing_request(request_id)

    async def get_batch_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        if self._batch_optimizer:
            return {
                "queue_status": await self._batch_optimizer.get_queue_status(),
                "performance_stats": self._batch_optimizer.get_performance_statistics(),
                "recent_metrics": self._batch_optimizer.get_recent_metrics(10)
            }
        return {}

    async def optimize_batch_configuration(self) -> Dict[str, Any]:
        """Get batch optimization recommendations."""
        if self._batch_optimizer:
            return await self._batch_optimizer.optimize_configuration()
        return {"status": "batch_optimizer_not_available"}

    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        if not self._monitor:
            return {"status": "monitoring_not_available"}

        # Get current metrics
        current_metrics = self._monitor.get_current_metrics()

        # Get health status
        health_status = self._monitor.get_health_status()

        # Get recent alerts
        recent_alerts = self._monitor.get_alerts(resolved=False)

        # Get performance summary
        performance_summary = self._monitor.get_performance_summary()

        # Update monitor with current pipeline statistics
        queue_size = await self._queue.size() if self._queue else 0
        processing_count = len(self._processing_batches) if hasattr(self, '_processing_batches') else 0

        self._monitor.update_metrics(
            queue_size=queue_size,
            processing_items=processing_count,
            total_requests=self._stats.get("total_requests", 0),
            completed_requests=self._stats.get("completed_requests", 0),
            failed_requests=self._stats.get("failed_requests", 0)
        )

        return {
            "status": "ok",
            "current_metrics": current_metrics.dict(),
            "health_status": health_status,
            "active_alerts": len(recent_alerts),
            "performance_summary": performance_summary,
            "pipeline_statistics": self.get_statistics()
        }

    async def get_indexing_alerts(self, severity: str = None) -> List[Dict[str, Any]]:
        """Get indexing alerts with optional severity filtering."""
        if not self._monitor:
            return []

        from agentic_rag.services.indexing_monitor import AlertSeverity

        severity_filter = None
        if severity:
            try:
                severity_filter = AlertSeverity(severity.lower())
            except ValueError:
                pass

        alerts = self._monitor.get_alerts(severity=severity_filter)

        return [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "threshold_value": alert.threshold_value,
                "current_value": alert.current_value,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in alerts
        ]

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a specific alert."""
        if self._monitor:
            return await self._monitor.resolve_alert(alert_id)
        return False

    async def export_monitoring_data(self, format: str = "json") -> str:
        """Export monitoring data in specified format."""
        if self._monitor:
            return await self._monitor.export_metrics(format)
        return "{\"error\": \"monitoring_not_available\"}"

    async def get_performance_metrics(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics over a specified duration."""
        if not self._monitor:
            return {"status": "monitoring_not_available"}

        from datetime import timedelta
        duration = timedelta(minutes=duration_minutes)

        # Get performance summary
        performance = self._monitor.get_performance_summary(duration)

        # Get metric history for key metrics
        latency_history = self._monitor.get_metric_history("average_latency", duration)
        throughput_history = self._monitor.get_metric_history("requests_per_second", duration)
        error_rate_history = self._monitor.get_metric_history("error_rate", duration)

        return {
            "status": "ok",
            "duration_minutes": duration_minutes,
            "performance_summary": performance,
            "metric_history": {
                "latency": [{"value": m.value, "timestamp": m.timestamp.isoformat()} for m in latency_history],
                "throughput": [{"value": m.value, "timestamp": m.timestamp.isoformat()} for m in throughput_history],
                "error_rate": [{"value": m.value, "timestamp": m.timestamp.isoformat()} for m in error_rate_history]
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on indexing pipeline."""
        try:
            queue_status = await self.get_queue_status()

            # Check if workers are healthy
            healthy_workers = sum(1 for worker in self._workers if not worker.done())

            # Get error handler health
            error_health = {}
            if self._error_handler:
                error_health = self._error_handler.get_health_status()

            overall_status = "healthy"
            if not self._running or healthy_workers == 0:
                overall_status = "unhealthy"
            elif error_health.get("status") in ["degraded", "unhealthy"]:
                overall_status = error_health["status"]

            return {
                "status": overall_status,
                "queue_status": queue_status,
                "healthy_workers": healthy_workers,
                "error_handler_health": error_health,
                "statistics": self.get_statistics(),
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Indexing pipeline health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Global indexing pipeline instance
_indexing_pipeline: Optional[VectorIndexingPipeline] = None


async def get_indexing_pipeline() -> VectorIndexingPipeline:
    """Get or create the global indexing pipeline instance."""
    global _indexing_pipeline
    
    if _indexing_pipeline is None:
        _indexing_pipeline = VectorIndexingPipeline()
        await _indexing_pipeline.start()
    
    return _indexing_pipeline


async def close_indexing_pipeline() -> None:
    """Close the global indexing pipeline instance."""
    global _indexing_pipeline
    
    if _indexing_pipeline:
        await _indexing_pipeline.stop()
        _indexing_pipeline = None
