"""
Advanced Performance Optimization Service for Query Processing.

This module provides comprehensive performance optimization including enhanced caching,
preprocessing result caching, query optimization strategies, parallel processing,
and detailed performance monitoring.
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import concurrent.futures

import structlog

logger = structlog.get_logger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategy options."""
    NONE = "none"
    SIMPLE = "simple"
    LRU = "lru"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"


class OptimizationLevel(str, Enum):
    """Performance optimization levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    parallel_executions: int = 0
    optimization_savings_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    max_cache_size_mb: int = 100
    default_ttl_seconds: int = 300
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 4
    enable_query_optimization: bool = True
    enable_preprocessing_cache: bool = True
    enable_result_cache: bool = True
    performance_monitoring: bool = True


class AdvancedPerformanceOptimizer:
    """Advanced performance optimizer for query processing."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize the performance optimizer."""
        self.config = config or OptimizationConfig()
        
        # Cache storage
        self._query_cache: Dict[str, CacheEntry] = {}
        self._preprocessing_cache: Dict[str, CacheEntry] = {}
        self._result_cache: Dict[str, CacheEntry] = {}
        
        # Performance tracking
        self._metrics = PerformanceMetrics()
        self._response_times: List[float] = []
        self._max_response_times = 1000  # Keep last 1000 response times
        
        # Parallel processing
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_parallel_workers
        )
        
        # Optimization statistics
        self._optimization_stats = {
            "cache_evictions": 0,
            "query_optimizations": 0,
            "parallel_executions": 0,
            "memory_cleanups": 0
        }
        
        logger.info("Advanced performance optimizer initialized",
                   cache_strategy=self.config.cache_strategy.value,
                   optimization_level=self.config.optimization_level.value)
    
    async def optimize_query_execution(
        self,
        query_func: Callable,
        cache_key: str,
        cache_type: str = "query",
        ttl_seconds: Optional[int] = None,
        enable_parallel: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """
        Optimize query execution with caching and parallel processing.
        
        Args:
            query_func: Function to execute
            cache_key: Unique cache key
            cache_type: Type of cache (query, preprocessing, result)
            ttl_seconds: Time to live for cache entry
            enable_parallel: Whether to enable parallel execution
            *args, **kwargs: Arguments for query_func
            
        Returns:
            Query result with optimization applied
        """
        start_time = time.time()
        
        try:
            self._metrics.total_requests += 1
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key, cache_type)
            if cached_result is not None:
                self._metrics.cache_hits += 1
                response_time_ms = (time.time() - start_time) * 1000
                self._record_response_time(response_time_ms)
                
                logger.debug(f"Cache hit for key: {cache_key[:20]}...")
                return cached_result
            
            self._metrics.cache_misses += 1
            
            # Execute query with optimization
            if enable_parallel and self.config.enable_parallel_processing:
                result = await self._execute_parallel(query_func, *args, **kwargs)
                self._metrics.parallel_executions += 1
            else:
                if asyncio.iscoroutinefunction(query_func):
                    result = await query_func(*args, **kwargs)
                else:
                    result = query_func(*args, **kwargs)
            
            # Cache the result
            ttl = ttl_seconds or self.config.default_ttl_seconds
            self._store_in_cache(cache_key, result, cache_type, ttl)
            
            # Record performance metrics
            response_time_ms = (time.time() - start_time) * 1000
            self._record_response_time(response_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}", exc_info=True)
            raise
    
    async def optimize_preprocessing(
        self,
        preprocessing_func: Callable,
        query: str,
        *args,
        **kwargs
    ) -> Any:
        """Optimize query preprocessing with caching."""
        if not self.config.enable_preprocessing_cache:
            if asyncio.iscoroutinefunction(preprocessing_func):
                return await preprocessing_func(query, *args, **kwargs)
            else:
                return preprocessing_func(query, *args, **kwargs)
        
        # Generate cache key for preprocessing
        cache_key = self._generate_preprocessing_cache_key(query, args, kwargs)
        
        return await self.optimize_query_execution(
            preprocessing_func,
            cache_key,
            "preprocessing",  # cache_type
            600,  # ttl_seconds - Longer TTL for preprocessing
            False,  # enable_parallel - Preprocessing is usually fast
            query,
            *args,
            **kwargs
        )
    
    async def optimize_batch_processing(
        self,
        batch_func: Callable,
        items: List[Any],
        batch_size: int = 10,
        max_parallel_batches: int = 3
    ) -> List[Any]:
        """Optimize batch processing with parallel execution."""
        if not self.config.enable_parallel_processing or len(items) <= batch_size:
            # Process as single batch
            if asyncio.iscoroutinefunction(batch_func):
                return await batch_func(items)
            else:
                return batch_func(items)
        
        # Split into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches in parallel
        semaphore = asyncio.Semaphore(max_parallel_batches)
        
        async def process_batch(batch):
            async with semaphore:
                if asyncio.iscoroutinefunction(batch_func):
                    return await batch_func(batch)
                else:
                    return batch_func(batch)
        
        # Execute all batches
        batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        self._optimization_stats["parallel_executions"] += len(batches)
        
        return results
    
    def optimize_query_string(self, query: str) -> str:
        """Optimize query string for better performance."""
        if not self.config.enable_query_optimization:
            return query
        
        optimized = query.strip()
        
        # Remove excessive whitespace
        optimized = ' '.join(optimized.split())
        
        # Remove very short words (less than 2 characters) except important ones
        important_short_words = {'is', 'in', 'on', 'at', 'to', 'or', 'if'}
        words = optimized.split()
        filtered_words = [
            word for word in words 
            if len(word) >= 2 or word.lower() in important_short_words
        ]
        optimized = ' '.join(filtered_words)
        
        # Limit query length
        max_length = 500 if self.config.optimization_level == OptimizationLevel.AGGRESSIVE else 1000
        if len(optimized) > max_length:
            optimized = optimized[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
        
        if optimized != query:
            self._optimization_stats["query_optimizations"] += 1
            logger.debug(f"Query optimized: {len(query)} -> {len(optimized)} chars")
        
        return optimized
    
    async def _execute_parallel(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in parallel using thread pool."""
        loop = asyncio.get_event_loop()
        
        if asyncio.iscoroutinefunction(func):
            # For async functions, run in current event loop
            return await func(*args, **kwargs)
        else:
            # For sync functions, run in thread pool
            return await loop.run_in_executor(self._executor, func, *args, **kwargs)
    
    def _get_from_cache(self, cache_key: str, cache_type: str) -> Optional[Any]:
        """Get item from appropriate cache."""
        cache = self._get_cache_by_type(cache_type)
        
        if cache_key not in cache:
            return None
        
        entry = cache[cache_key]
        
        # Check if expired
        if entry.is_expired():
            del cache[cache_key]
            return None
        
        # Update access information
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        
        return entry.data
    
    def _store_in_cache(
        self, 
        cache_key: str, 
        data: Any, 
        cache_type: str, 
        ttl_seconds: int
    ) -> None:
        """Store item in appropriate cache."""
        cache = self._get_cache_by_type(cache_type)
        
        # Estimate size (rough approximation)
        size_bytes = len(str(data).encode('utf-8'))
        
        # Create cache entry
        entry = CacheEntry(
            data=data,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            ttl_seconds=ttl_seconds,
            size_bytes=size_bytes
        )
        
        cache[cache_key] = entry
        
        # Check if cache cleanup is needed
        self._cleanup_cache_if_needed(cache_type)
    
    def _get_cache_by_type(self, cache_type: str) -> Dict[str, CacheEntry]:
        """Get cache dictionary by type."""
        if cache_type == "preprocessing":
            return self._preprocessing_cache
        elif cache_type == "result":
            return self._result_cache
        else:
            return self._query_cache
    
    def _cleanup_cache_if_needed(self, cache_type: str) -> None:
        """Cleanup cache if it exceeds size limits."""
        cache = self._get_cache_by_type(cache_type)
        
        # Calculate current cache size
        total_size_mb = sum(entry.size_bytes for entry in cache.values()) / (1024 * 1024)
        
        if total_size_mb > self.config.max_cache_size_mb:
            self._evict_cache_entries(cache, cache_type)
    
    def _evict_cache_entries(self, cache: Dict[str, CacheEntry], cache_type: str) -> None:
        """Evict cache entries using LRU strategy."""
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            cache.items(),
            key=lambda x: (x[1].last_accessed, x[1].access_count)
        )
        
        # Remove oldest 25% of entries
        entries_to_remove = len(sorted_entries) // 4
        
        for i in range(entries_to_remove):
            cache_key, _ = sorted_entries[i]
            del cache[cache_key]
            self._optimization_stats["cache_evictions"] += 1
        
        logger.info(f"Evicted {entries_to_remove} entries from {cache_type} cache")
    
    def _generate_preprocessing_cache_key(
        self, 
        query: str, 
        args: Tuple, 
        kwargs: Dict[str, Any]
    ) -> str:
        """Generate cache key for preprocessing."""
        # Create a hash of query and parameters
        key_data = f"{query}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _record_response_time(self, response_time_ms: float) -> None:
        """Record response time for metrics."""
        self._response_times.append(response_time_ms)
        
        # Keep only recent response times
        if len(self._response_times) > self._max_response_times:
            self._response_times = self._response_times[-self._max_response_times:]
        
        # Update metrics
        self._update_performance_metrics()
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        if not self._response_times:
            return
        
        # Calculate average
        self._metrics.average_response_time_ms = sum(self._response_times) / len(self._response_times)
        
        # Calculate percentiles
        sorted_times = sorted(self._response_times)
        n = len(sorted_times)
        
        if n >= 20:  # Only calculate percentiles with sufficient data
            p95_index = int(0.95 * n)
            p99_index = int(0.99 * n)
            
            self._metrics.p95_response_time_ms = sorted_times[p95_index]
            self._metrics.p99_response_time_ms = sorted_times[p99_index]

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        # Calculate memory usage
        total_cache_size = 0
        cache_stats = {}

        for cache_name, cache in [
            ("query", self._query_cache),
            ("preprocessing", self._preprocessing_cache),
            ("result", self._result_cache)
        ]:
            cache_size = sum(entry.size_bytes for entry in cache.values())
            total_cache_size += cache_size

            cache_stats[f"{cache_name}_cache"] = {
                "entries": len(cache),
                "size_mb": cache_size / (1024 * 1024),
                "hit_rate": self._calculate_cache_hit_rate(cache)
            }

        self._metrics.memory_usage_mb = total_cache_size / (1024 * 1024)

        return {
            "performance_metrics": {
                "total_requests": self._metrics.total_requests,
                "cache_hit_rate": self._metrics.cache_hit_rate,
                "average_response_time_ms": self._metrics.average_response_time_ms,
                "p95_response_time_ms": self._metrics.p95_response_time_ms,
                "p99_response_time_ms": self._metrics.p99_response_time_ms,
                "parallel_executions": self._metrics.parallel_executions,
                "memory_usage_mb": self._metrics.memory_usage_mb
            },
            "cache_statistics": cache_stats,
            "optimization_statistics": self._optimization_stats,
            "configuration": {
                "cache_strategy": self.config.cache_strategy.value,
                "optimization_level": self.config.optimization_level.value,
                "max_cache_size_mb": self.config.max_cache_size_mb,
                "parallel_processing_enabled": self.config.enable_parallel_processing,
                "max_parallel_workers": self.config.max_parallel_workers
            }
        }

    def _calculate_cache_hit_rate(self, cache: Dict[str, CacheEntry]) -> float:
        """Calculate hit rate for a specific cache."""
        if not cache:
            return 0.0

        total_accesses = sum(entry.access_count for entry in cache.values())
        return len(cache) / max(total_accesses, 1)

    async def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """Clear cache(s)."""
        if cache_type is None:
            # Clear all caches
            self._query_cache.clear()
            self._preprocessing_cache.clear()
            self._result_cache.clear()
            logger.info("All caches cleared")
        else:
            cache = self._get_cache_by_type(cache_type)
            cache.clear()
            logger.info(f"{cache_type} cache cleared")

    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up caches."""
        initial_memory = self._metrics.memory_usage_mb

        # Remove expired entries
        expired_removed = 0
        for cache_name, cache in [
            ("query", self._query_cache),
            ("preprocessing", self._preprocessing_cache),
            ("result", self._result_cache)
        ]:
            expired_keys = [
                key for key, entry in cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del cache[key]
                expired_removed += 1

        # Force cache cleanup if still over limit
        for cache_type in ["query", "preprocessing", "result"]:
            self._cleanup_cache_if_needed(cache_type)

        # Update memory usage
        await self.get_performance_metrics()
        final_memory = self._metrics.memory_usage_mb

        self._optimization_stats["memory_cleanups"] += 1

        result = {
            "expired_entries_removed": expired_removed,
            "memory_freed_mb": initial_memory - final_memory,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory
        }

        logger.info("Memory optimization completed", **result)
        return result

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on performance optimizer."""
        health_status = "healthy"
        issues = []

        # Check memory usage
        if self._metrics.memory_usage_mb > self.config.max_cache_size_mb * 0.9:
            health_status = "warning"
            issues.append("High memory usage")

        # Check response times
        if self._metrics.p95_response_time_ms > 5000:  # 5 seconds
            health_status = "warning"
            issues.append("High response times")

        # Check cache hit rate
        if self._metrics.cache_hit_rate < 0.3:  # Less than 30%
            health_status = "warning"
            issues.append("Low cache hit rate")

        return {
            "status": health_status,
            "issues": issues,
            "uptime_requests": self._metrics.total_requests,
            "cache_hit_rate": self._metrics.cache_hit_rate,
            "memory_usage_mb": self._metrics.memory_usage_mb,
            "average_response_time_ms": self._metrics.average_response_time_ms
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Singleton instance
_performance_optimizer_instance: Optional[AdvancedPerformanceOptimizer] = None


async def get_advanced_performance_optimizer(
    config: Optional[OptimizationConfig] = None
) -> AdvancedPerformanceOptimizer:
    """Get the advanced performance optimizer instance."""
    global _performance_optimizer_instance

    if _performance_optimizer_instance is None:
        _performance_optimizer_instance = AdvancedPerformanceOptimizer(config)

    return _performance_optimizer_instance


def reset_advanced_performance_optimizer():
    """Reset the performance optimizer instance (for testing)."""
    global _performance_optimizer_instance
    if _performance_optimizer_instance is not None:
        _performance_optimizer_instance.__del__()
    _performance_optimizer_instance = None
