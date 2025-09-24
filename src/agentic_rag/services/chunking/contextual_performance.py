"""
Contextual Chunking Performance Optimization

This module provides performance optimizations for contextual chunking including
caching, memory management, parallel processing, and progress tracking.
"""

import asyncio
import logging
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rag.services.chunking.contextual_chunker import GlobalContext, LocalContext
from agentic_rag.services.content_extraction import ExtractedContent

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for contextual chunking."""
    
    total_processing_time: float = 0.0
    context_extraction_time: float = 0.0
    fusion_time: float = 0.0
    embedding_time: float = 0.0
    
    documents_processed: int = 0
    chunks_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_requests = self.cache_hits + self.cache_misses
        return (self.cache_hits / total_requests * 100) if total_requests > 0 else 0.0
    
    def get_avg_processing_time(self) -> float:
        """Calculate average processing time per document."""
        return self.total_processing_time / self.documents_processed if self.documents_processed > 0 else 0.0


class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self._hits += 1
            return self.cache[key]
        else:
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self.cache:
            # Update existing item
            self.cache.move_to_end(key)
        else:
            # Add new item
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }


class ContextCache:
    """Specialized cache for context extraction results."""
    
    def __init__(self, max_size: int = 500):
        self.global_context_cache = LRUCache(max_size)
        self.definition_cache = LRUCache(max_size * 2)  # Definitions are smaller
        self.section_analysis_cache = LRUCache(max_size // 2)  # Section analysis is expensive
        
    def get_global_context(self, document_id: str) -> Optional[GlobalContext]:
        """Get cached global context for document."""
        return self.global_context_cache.get(document_id)
    
    def put_global_context(self, document_id: str, context: GlobalContext) -> None:
        """Cache global context for document."""
        self.global_context_cache.put(document_id, context)
    
    def get_definitions(self, document_id: str) -> Optional[List]:
        """Get cached definitions for document."""
        return self.definition_cache.get(document_id)
    
    def put_definitions(self, document_id: str, definitions: List) -> None:
        """Cache definitions for document."""
        self.definition_cache.put(document_id, definitions)
    
    def get_section_analysis(self, document_id: str) -> Optional[List]:
        """Get cached section analysis for document."""
        return self.section_analysis_cache.get(document_id)
    
    def put_section_analysis(self, document_id: str, sections: List) -> None:
        """Cache section analysis for document."""
        self.section_analysis_cache.put(document_id, sections)
    
    def clear_document(self, document_id: str) -> None:
        """Clear all cached data for a specific document."""
        # Note: LRUCache doesn't have direct key removal, but items will be evicted naturally
        pass
    
    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        return {
            "global_context": self.global_context_cache.get_stats(),
            "definitions": self.definition_cache.get_stats(),
            "section_analysis": self.section_analysis_cache.get_stats()
        }


class MemoryManager:
    """Memory usage monitoring and optimization."""
    
    def __init__(self):
        self.peak_memory = 0.0
        self.current_memory = 0.0
        self._weak_refs = weakref.WeakSet()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.current_memory = memory_mb
            self.peak_memory = max(self.peak_memory, memory_mb)
            return memory_mb
        except ImportError:
            # Fallback if psutil not available
            return 0.0
    
    def register_object(self, obj: Any) -> None:
        """Register object for memory tracking."""
        self._weak_refs.add(obj)
    
    def cleanup_memory(self) -> None:
        """Force garbage collection and cleanup."""
        import gc
        gc.collect()
        
        # Clear weak references to help with cleanup
        self._weak_refs.clear()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        current = self.get_memory_usage()
        return {
            "current_mb": current,
            "peak_mb": self.peak_memory,
            "tracked_objects": len(self._weak_refs)
        }


class ProgressTracker:
    """Progress tracking for long-running operations."""
    
    def __init__(self):
        self.operations = {}
        self.completed_operations = {}
    
    def start_operation(self, operation_id: str, total_items: int, description: str = "") -> None:
        """Start tracking an operation."""
        self.operations[operation_id] = {
            "total_items": total_items,
            "completed_items": 0,
            "start_time": time.time(),
            "description": description,
            "status": "running"
        }
        logger.info(f"Started operation {operation_id}: {description} ({total_items} items)")
    
    def update_progress(self, operation_id: str, completed_items: int) -> None:
        """Update progress for an operation."""
        if operation_id in self.operations:
            self.operations[operation_id]["completed_items"] = completed_items
            
            # Log progress at intervals
            total = self.operations[operation_id]["total_items"]
            if total > 0:
                progress = (completed_items / total) * 100
                if completed_items % max(1, total // 10) == 0:  # Log every 10%
                    logger.info(f"Operation {operation_id}: {progress:.1f}% complete ({completed_items}/{total})")
    
    def complete_operation(self, operation_id: str) -> None:
        """Mark operation as complete."""
        if operation_id in self.operations:
            operation = self.operations[operation_id]
            operation["status"] = "completed"
            operation["end_time"] = time.time()
            operation["duration"] = operation["end_time"] - operation["start_time"]
            
            # Move to completed operations
            self.completed_operations[operation_id] = operation
            del self.operations[operation_id]
            
            logger.info(f"Completed operation {operation_id} in {operation['duration']:.2f}s")
    
    def get_progress(self, operation_id: str) -> Optional[Dict]:
        """Get progress information for an operation."""
        if operation_id in self.operations:
            operation = self.operations[operation_id]
            total = operation["total_items"]
            completed = operation["completed_items"]
            progress = (completed / total) * 100 if total > 0 else 0.0
            
            return {
                "operation_id": operation_id,
                "description": operation["description"],
                "total_items": total,
                "completed_items": completed,
                "progress_percent": progress,
                "status": operation["status"],
                "elapsed_time": time.time() - operation["start_time"]
            }
        return None
    
    def get_all_operations(self) -> Dict:
        """Get status of all operations."""
        result = {}
        
        # Active operations
        for op_id in self.operations:
            result[op_id] = self.get_progress(op_id)
        
        # Completed operations (last 10)
        completed_items = list(self.completed_operations.items())[-10:]
        for op_id, operation in completed_items:
            result[op_id] = {
                "operation_id": op_id,
                "description": operation["description"],
                "total_items": operation["total_items"],
                "completed_items": operation["completed_items"],
                "progress_percent": 100.0,
                "status": operation["status"],
                "duration": operation["duration"]
            }
        
        return result


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, cache_size: int = 1000):
        self.context_cache = ContextCache(cache_size)
        self.memory_manager = MemoryManager()
        self.progress_tracker = ProgressTracker()
        self.metrics = PerformanceMetrics()
        
        # Performance settings
        self.enable_caching = True
        self.enable_memory_monitoring = True
        self.enable_progress_tracking = True
        self.memory_cleanup_threshold_mb = 1000  # Cleanup when memory exceeds this
        
        logger.info(f"Performance optimizer initialized with cache_size={cache_size}")
    
    def start_document_processing(self, document_id: str, total_chunks: int) -> None:
        """Start tracking document processing."""
        if self.enable_progress_tracking:
            self.progress_tracker.start_operation(
                f"doc_{document_id}",
                total_chunks,
                f"Processing document {document_id}"
            )
    
    def update_chunk_progress(self, document_id: str, completed_chunks: int) -> None:
        """Update chunk processing progress."""
        if self.enable_progress_tracking:
            self.progress_tracker.update_progress(f"doc_{document_id}", completed_chunks)
    
    def complete_document_processing(self, document_id: str) -> None:
        """Complete document processing tracking."""
        if self.enable_progress_tracking:
            self.progress_tracker.complete_operation(f"doc_{document_id}")
        
        self.metrics.documents_processed += 1
    
    def check_memory_usage(self) -> None:
        """Check memory usage and cleanup if needed."""
        if self.enable_memory_monitoring:
            current_memory = self.memory_manager.get_memory_usage()
            
            if current_memory > self.memory_cleanup_threshold_mb:
                logger.warning(f"Memory usage high ({current_memory:.1f}MB), performing cleanup")
                self.memory_manager.cleanup_memory()
                
                # Clear some cache if memory is still high
                new_memory = self.memory_manager.get_memory_usage()
                if new_memory > self.memory_cleanup_threshold_mb * 0.9:
                    self.context_cache.global_context_cache.clear()
                    logger.info("Cleared global context cache due to memory pressure")
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        return {
            "metrics": {
                "total_processing_time": self.metrics.total_processing_time,
                "documents_processed": self.metrics.documents_processed,
                "chunks_processed": self.metrics.chunks_processed,
                "avg_processing_time": self.metrics.get_avg_processing_time(),
                "cache_hit_rate": self.metrics.get_cache_hit_rate()
            },
            "cache_stats": self.context_cache.get_stats(),
            "memory_stats": self.memory_manager.get_stats(),
            "active_operations": len(self.progress_tracker.operations),
            "completed_operations": len(self.progress_tracker.completed_operations)
        }


# Singleton instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer(cache_size: int = 1000) -> PerformanceOptimizer:
    """Get or create a performance optimizer instance."""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(cache_size)
    
    return _performance_optimizer
