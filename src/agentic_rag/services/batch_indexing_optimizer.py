"""
Batch Indexing Optimizer

This module provides optimized batch processing for vector indexing
with dynamic batch sizing and parallel processing capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from collections import deque

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BatchStrategy(str, Enum):
    """Strategies for batch processing."""
    
    FIXED_SIZE = "fixed_size"        # Fixed batch size
    ADAPTIVE = "adaptive"            # Adaptive based on performance
    LOAD_BALANCED = "load_balanced"  # Balance based on system load
    PRIORITY_BASED = "priority_based" # Batch by priority levels


class BatchStatus(str, Enum):
    """Status of batch processing."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance."""
    
    batch_id: str
    batch_size: int
    processing_time: float
    throughput: float  # items per second
    success_rate: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class BatchConfiguration(BaseModel):
    """Configuration for batch processing."""
    
    min_batch_size: int = Field(default=10, description="Minimum batch size")
    max_batch_size: int = Field(default=100, description="Maximum batch size")
    optimal_batch_size: int = Field(default=50, description="Optimal batch size")
    max_concurrent_batches: int = Field(default=3, description="Maximum concurrent batches")
    batch_timeout: float = Field(default=300.0, description="Batch processing timeout in seconds")
    adaptive_sizing: bool = Field(default=True, description="Enable adaptive batch sizing")
    performance_threshold: float = Field(default=0.8, description="Performance threshold for adaptation")


class BatchItem(BaseModel):
    """Individual item in a batch."""
    
    item_id: str = Field(..., description="Item identifier")
    data: Dict[str, Any] = Field(..., description="Item data")
    priority: int = Field(default=5, description="Item priority")
    estimated_processing_time: float = Field(default=1.0, description="Estimated processing time")
    retry_count: int = Field(default=0, description="Number of retries")


class ProcessingBatch(BaseModel):
    """A batch of items for processing."""
    
    batch_id: str = Field(..., description="Batch identifier")
    items: List[BatchItem] = Field(..., description="Items in batch")
    status: BatchStatus = Field(default=BatchStatus.PENDING, description="Batch status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    success_count: int = Field(default=0, description="Number of successful items")
    failure_count: int = Field(default=0, description="Number of failed items")
    errors: List[str] = Field(default_factory=list, description="Error messages")


class BatchIndexingOptimizer:
    """Optimizer for batch indexing operations."""
    
    def __init__(self, config: BatchConfiguration = None):
        self.config = config or BatchConfiguration()
        self._batch_queue: deque = deque()
        self._processing_batches: Dict[str, ProcessingBatch] = {}
        self._completed_batches: deque = deque(maxlen=1000)  # Keep last 1000 batches
        
        # Performance tracking
        self._metrics_history: deque = deque(maxlen=100)  # Keep last 100 batch metrics
        self._performance_stats = {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "average_throughput": 0.0,
            "average_batch_size": 0.0,
            "optimal_batch_size": self.config.optimal_batch_size
        }
        
        # Adaptive sizing state
        self._current_batch_size = self.config.optimal_batch_size
        self._last_adaptation = datetime.utcnow()
        self._adaptation_cooldown = 60.0  # seconds
        
        # Concurrency control
        self._batch_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        self._queue_lock = asyncio.Lock()
        
        logger.info(f"Batch indexing optimizer initialized with config: {self.config}")
    
    async def add_items(self, items: List[BatchItem]) -> None:
        """Add items to the batch queue."""
        async with self._queue_lock:
            for item in items:
                self._batch_queue.append(item)
            
            logger.debug(f"Added {len(items)} items to batch queue (total: {len(self._batch_queue)})")
    
    async def create_optimal_batches(self) -> List[ProcessingBatch]:
        """Create optimally sized batches from queued items."""
        batches = []
        
        async with self._queue_lock:
            if not self._batch_queue:
                return batches
            
            # Determine current optimal batch size
            current_batch_size = await self._get_current_batch_size()
            
            # Create batches
            while self._batch_queue and len(batches) < self.config.max_concurrent_batches:
                batch_items = []
                
                # Fill batch up to optimal size
                while len(batch_items) < current_batch_size and self._batch_queue:
                    item = self._batch_queue.popleft()
                    batch_items.append(item)
                
                if batch_items:
                    batch = ProcessingBatch(
                        batch_id=f"batch_{int(time.time() * 1000)}_{len(batches)}",
                        items=batch_items
                    )
                    batches.append(batch)
                    self._processing_batches[batch.batch_id] = batch
            
            logger.info(f"Created {len(batches)} batches with sizes: {[len(b.items) for b in batches]}")
        
        return batches
    
    async def _get_current_batch_size(self) -> int:
        """Get the current optimal batch size based on performance."""
        if not self.config.adaptive_sizing:
            return self.config.optimal_batch_size
        
        # Check if we should adapt
        time_since_adaptation = (datetime.utcnow() - self._last_adaptation).total_seconds()
        if time_since_adaptation < self._adaptation_cooldown:
            return self._current_batch_size
        
        # Analyze recent performance
        if len(self._metrics_history) >= 5:
            recent_metrics = list(self._metrics_history)[-5:]
            avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
            avg_success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
            
            # Adapt batch size based on performance
            if avg_success_rate < self.config.performance_threshold:
                # Reduce batch size if success rate is low
                new_size = max(self.config.min_batch_size, int(self._current_batch_size * 0.8))
                logger.info(f"Reducing batch size from {self._current_batch_size} to {new_size} (low success rate: {avg_success_rate:.2f})")
            elif avg_throughput > self._performance_stats["average_throughput"] * 1.2:
                # Increase batch size if throughput is high
                new_size = min(self.config.max_batch_size, int(self._current_batch_size * 1.2))
                logger.info(f"Increasing batch size from {self._current_batch_size} to {new_size} (high throughput: {avg_throughput:.2f})")
            else:
                new_size = self._current_batch_size
            
            self._current_batch_size = new_size
            self._last_adaptation = datetime.utcnow()
        
        return self._current_batch_size
    
    async def process_batch(
        self,
        batch: ProcessingBatch,
        processor_func: callable
    ) -> ProcessingBatch:
        """
        Process a batch of items using the provided processor function.
        
        Args:
            batch: Batch to process
            processor_func: Async function to process individual items
            
        Returns:
            Updated batch with results
        """
        async with self._batch_semaphore:
            start_time = time.time()
            batch.status = BatchStatus.PROCESSING
            batch.started_at = datetime.utcnow()
            
            logger.info(f"Processing batch {batch.batch_id} with {len(batch.items)} items")
            
            try:
                # Process items in parallel within the batch
                tasks = []
                for item in batch.items:
                    task = asyncio.create_task(self._process_item(item, processor_func))
                    tasks.append(task)
                
                # Wait for all items to complete with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.config.batch_timeout
                    )
                    
                    # Count successes and failures
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            batch.failure_count += 1
                            batch.errors.append(f"Item {batch.items[i].item_id}: {str(result)}")
                        else:
                            batch.success_count += 1
                    
                    # Determine final status
                    if batch.failure_count == 0:
                        batch.status = BatchStatus.COMPLETED
                    elif batch.success_count > 0:
                        batch.status = BatchStatus.PARTIAL
                    else:
                        batch.status = BatchStatus.FAILED
                
                except asyncio.TimeoutError:
                    batch.status = BatchStatus.FAILED
                    batch.errors.append(f"Batch processing timed out after {self.config.batch_timeout}s")
                    logger.error(f"Batch {batch.batch_id} timed out")
                
            except Exception as e:
                batch.status = BatchStatus.FAILED
                batch.errors.append(f"Batch processing error: {str(e)}")
                logger.error(f"Batch {batch.batch_id} failed: {e}")
            
            finally:
                batch.completed_at = datetime.utcnow()
                processing_time = time.time() - start_time
                
                # Record metrics
                await self._record_batch_metrics(batch, processing_time)
                
                # Move to completed batches
                self._completed_batches.append(batch)
                if batch.batch_id in self._processing_batches:
                    del self._processing_batches[batch.batch_id]
                
                logger.info(f"Completed batch {batch.batch_id}: {batch.success_count}/{len(batch.items)} successful in {processing_time:.2f}s")
        
        return batch
    
    async def _process_item(self, item: BatchItem, processor_func: callable) -> Any:
        """Process a single item within a batch."""
        try:
            return await processor_func(item)
        except Exception as e:
            logger.error(f"Failed to process item {item.item_id}: {e}")
            raise
    
    async def _record_batch_metrics(self, batch: ProcessingBatch, processing_time: float) -> None:
        """Record metrics for a completed batch."""
        success_rate = batch.success_count / len(batch.items) if batch.items else 0.0
        throughput = len(batch.items) / processing_time if processing_time > 0 else 0.0
        
        metrics = BatchMetrics(
            batch_id=batch.batch_id,
            batch_size=len(batch.items),
            processing_time=processing_time,
            throughput=throughput,
            success_rate=success_rate,
            memory_usage=0.0,  # TODO: Implement memory monitoring
            cpu_usage=0.0      # TODO: Implement CPU monitoring
        )
        
        self._metrics_history.append(metrics)
        
        # Update performance statistics
        self._performance_stats["total_batches"] += 1
        if batch.status == BatchStatus.COMPLETED:
            self._performance_stats["successful_batches"] += 1
        else:
            self._performance_stats["failed_batches"] += 1
        
        # Update averages
        if self._metrics_history:
            self._performance_stats["average_throughput"] = sum(
                m.throughput for m in self._metrics_history
            ) / len(self._metrics_history)
            
            self._performance_stats["average_batch_size"] = sum(
                m.batch_size for m in self._metrics_history
            ) / len(self._metrics_history)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue and processing status."""
        async with self._queue_lock:
            return {
                "queued_items": len(self._batch_queue),
                "processing_batches": len(self._processing_batches),
                "completed_batches": len(self._completed_batches),
                "current_batch_size": self._current_batch_size,
                "max_concurrent_batches": self.config.max_concurrent_batches,
                "available_batch_slots": self._batch_semaphore._value
            }
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self._performance_stats.copy()
    
    def get_recent_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent batch metrics."""
        recent = list(self._metrics_history)[-limit:]
        return [
            {
                "batch_id": m.batch_id,
                "batch_size": m.batch_size,
                "processing_time": m.processing_time,
                "throughput": m.throughput,
                "success_rate": m.success_rate,
                "timestamp": m.timestamp.isoformat()
            }
            for m in recent
        ]
    
    async def optimize_configuration(self) -> Dict[str, Any]:
        """Analyze performance and suggest configuration optimizations."""
        if len(self._metrics_history) < 10:
            return {
                "status": "insufficient_data",
                "message": "Need at least 10 batches for optimization analysis"
            }
        
        recent_metrics = list(self._metrics_history)[-20:]  # Last 20 batches
        
        # Analyze throughput by batch size
        size_performance = {}
        for metric in recent_metrics:
            size = metric.batch_size
            if size not in size_performance:
                size_performance[size] = []
            size_performance[size].append(metric.throughput)
        
        # Find optimal batch size
        best_size = self.config.optimal_batch_size
        best_throughput = 0.0
        
        for size, throughputs in size_performance.items():
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_size = size
        
        # Generate recommendations
        recommendations = []
        
        if best_size != self.config.optimal_batch_size:
            recommendations.append({
                "type": "batch_size",
                "current": self.config.optimal_batch_size,
                "recommended": best_size,
                "reason": f"Better throughput: {best_throughput:.2f} items/sec"
            })
        
        # Check success rates
        avg_success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        if avg_success_rate < 0.9:
            recommendations.append({
                "type": "reliability",
                "current_success_rate": avg_success_rate,
                "recommendation": "Consider reducing batch size or adding retry logic",
                "reason": "Low success rate detected"
            })
        
        # Check processing times
        avg_processing_time = sum(m.processing_time for m in recent_metrics) / len(recent_metrics)
        if avg_processing_time > self.config.batch_timeout * 0.8:
            recommendations.append({
                "type": "timeout",
                "current_avg_time": avg_processing_time,
                "recommendation": "Increase batch timeout or reduce batch size",
                "reason": "Processing times approaching timeout limit"
            })
        
        return {
            "status": "analysis_complete",
            "optimal_batch_size": best_size,
            "current_performance": {
                "average_throughput": sum(m.throughput for m in recent_metrics) / len(recent_metrics),
                "average_success_rate": avg_success_rate,
                "average_processing_time": avg_processing_time
            },
            "recommendations": recommendations
        }
    
    async def clear_completed_batches(self) -> int:
        """Clear completed batch history."""
        count = len(self._completed_batches)
        self._completed_batches.clear()
        logger.info(f"Cleared {count} completed batches from history")
        return count


# Global optimizer instance
_batch_optimizer: Optional[BatchIndexingOptimizer] = None


async def get_batch_optimizer(config: BatchConfiguration = None) -> BatchIndexingOptimizer:
    """Get or create the global batch optimizer instance."""
    global _batch_optimizer
    
    if _batch_optimizer is None:
        _batch_optimizer = BatchIndexingOptimizer(config)
    
    return _batch_optimizer


async def optimize_batch_processing(
    items: List[BatchItem],
    processor_func: callable,
    config: BatchConfiguration = None
) -> List[ProcessingBatch]:
    """
    Convenience function to optimize batch processing.
    
    Args:
        items: Items to process
        processor_func: Function to process individual items
        config: Batch configuration
        
    Returns:
        List of completed batches
    """
    optimizer = await get_batch_optimizer(config)
    
    # Add items to queue
    await optimizer.add_items(items)
    
    # Create and process batches
    batches = await optimizer.create_optimal_batches()
    completed_batches = []
    
    # Process batches concurrently
    tasks = [
        optimizer.process_batch(batch, processor_func)
        for batch in batches
    ]
    
    if tasks:
        completed_batches = await asyncio.gather(*tasks)
    
    return completed_batches
