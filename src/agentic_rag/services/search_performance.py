"""
Search Performance Optimization Service

This module provides performance optimization capabilities for search operations
including caching strategies, query optimization, timeout handling, and
concurrent search management.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategies for search optimization."""
    NONE = "none"
    SIMPLE = "simple"
    LRU = "lru"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class OptimizationLevel(str, Enum):
    """Optimization levels for search performance."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class PerformanceMetrics:
    """Performance metrics for search operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    concurrent_requests: int = 0
    max_concurrent_requests: int = 0
    timeout_count: int = 0
    throttled_requests: int = 0


class SearchPerformanceConfig(BaseModel):
    """Configuration for search performance optimization."""
    
    cache_strategy: CacheStrategy = Field(default=CacheStrategy.TTL)
    cache_ttl_seconds: int = Field(default=300, ge=60)
    cache_max_size: int = Field(default=1000, ge=100)
    
    optimization_level: OptimizationLevel = Field(default=OptimizationLevel.STANDARD)
    
    # Timeout settings
    query_timeout_seconds: float = Field(default=30.0, ge=1.0)
    vector_search_timeout_seconds: float = Field(default=20.0, ge=1.0)
    ranking_timeout_seconds: float = Field(default=5.0, ge=1.0)
    
    # Concurrency settings
    max_concurrent_searches: int = Field(default=100, ge=1)
    max_queue_size: int = Field(default=500, ge=10)
    
    # Throttling settings
    enable_throttling: bool = Field(default=True)
    requests_per_minute: int = Field(default=1000, ge=10)
    burst_limit: int = Field(default=50, ge=5)
    
    # Query optimization
    enable_query_optimization: bool = Field(default=True)
    min_query_length: int = Field(default=3, ge=1)
    max_query_length: int = Field(default=1000, ge=10)
    
    # Result optimization
    enable_result_caching: bool = Field(default=True)
    max_results_to_cache: int = Field(default=100, ge=10)


class SearchPerformanceOptimizer:
    """Service for optimizing search performance."""
    
    def __init__(self, config: Optional[SearchPerformanceConfig] = None):
        self.config = config or SearchPerformanceConfig()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self._response_times: List[float] = []
        self._max_response_times = 1000  # Keep last 1000 response times
        
        # Cache implementation
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_access_count: Dict[str, int] = {}
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_searches)
        self._request_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Throttling
        self._request_timestamps: List[float] = []
        self._burst_tokens = self.config.burst_limit
        self._last_token_refill = time.time()
        
        logger.info(
            "Search performance optimizer initialized",
            cache_strategy=self.config.cache_strategy.value,
            optimization_level=self.config.optimization_level.value,
            max_concurrent=self.config.max_concurrent_searches
        )
    
    async def optimize_search_request(
        self,
        search_func: Callable,
        cache_key: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Optimize a search request with caching, throttling, and timeout handling.
        
        Args:
            search_func: The search function to execute
            cache_key: Cache key for the request
            *args, **kwargs: Arguments for the search function
            
        Returns:
            Search result with performance optimizations applied
        """
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.concurrent_requests += 1
            self.metrics.max_concurrent_requests = max(
                self.metrics.max_concurrent_requests,
                self.metrics.concurrent_requests
            )
            
            # Check throttling
            if self.config.enable_throttling:
                if not await self._check_rate_limit():
                    self.metrics.throttled_requests += 1
                    raise Exception("Rate limit exceeded")
            
            # Check cache
            if self.config.cache_strategy != CacheStrategy.NONE:
                cached_result = self._get_cached_result(cache_key)
                if cached_result is not None:
                    self.metrics.cache_hits += 1
                    self.metrics.concurrent_requests -= 1
                    return cached_result
                else:
                    self.metrics.cache_misses += 1
            
            # Execute search with concurrency control and timeout
            async with self._semaphore:
                try:
                    result = await asyncio.wait_for(
                        search_func(*args, **kwargs),
                        timeout=self.config.query_timeout_seconds
                    )
                    
                    # Cache the result
                    if self.config.enable_result_caching:
                        self._cache_result(cache_key, result)
                    
                    # Update success metrics
                    self.metrics.successful_requests += 1
                    
                    return result
                    
                except asyncio.TimeoutError:
                    self.metrics.timeout_count += 1
                    self.metrics.failed_requests += 1
                    raise Exception("Search request timed out")
                
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error(f"Search optimization failed: {e}")
            raise
            
        finally:
            # Update timing metrics
            response_time_ms = (time.time() - start_time) * 1000
            self._update_response_time_metrics(response_time_ms)
            self.metrics.concurrent_requests -= 1
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and valid."""
        if cache_key not in self._cache:
            return None
        
        # Check TTL for TTL-based caching
        if self.config.cache_strategy == CacheStrategy.TTL:
            timestamp = self._cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp > self.config.cache_ttl_seconds:
                self._remove_cache_entry(cache_key)
                return None
        
        # Update access count for LRU
        if self.config.cache_strategy == CacheStrategy.LRU:
            self._cache_access_count[cache_key] = self._cache_access_count.get(cache_key, 0) + 1
        
        return self._cache[cache_key]
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a search result."""
        # Check cache size limits
        if len(self._cache) >= self.config.cache_max_size:
            self._evict_cache_entries()
        
        # Store result
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
        self._cache_access_count[cache_key] = 1
    
    def _evict_cache_entries(self) -> None:
        """Evict cache entries based on strategy."""
        if self.config.cache_strategy == CacheStrategy.LRU:
            # Remove least recently used entries
            sorted_keys = sorted(
                self._cache_access_count.keys(),
                key=lambda k: self._cache_access_count[k]
            )
            keys_to_remove = sorted_keys[:len(sorted_keys) // 4]  # Remove 25%
            
        elif self.config.cache_strategy == CacheStrategy.TTL:
            # Remove expired entries
            current_time = time.time()
            keys_to_remove = [
                key for key, timestamp in self._cache_timestamps.items()
                if current_time - timestamp > self.config.cache_ttl_seconds
            ]
            
        else:
            # Simple FIFO eviction
            keys_to_remove = list(self._cache.keys())[:len(self._cache) // 4]
        
        for key in keys_to_remove:
            self._remove_cache_entry(key)
    
    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a cache entry."""
        self._cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)
        self._cache_access_count.pop(cache_key, None)
    
    async def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Refill burst tokens
        time_since_refill = current_time - self._last_token_refill
        tokens_to_add = int(time_since_refill * (self.config.requests_per_minute / 60))
        if tokens_to_add > 0:
            self._burst_tokens = min(
                self.config.burst_limit,
                self._burst_tokens + tokens_to_add
            )
            self._last_token_refill = current_time
        
        # Check burst limit
        if self._burst_tokens <= 0:
            return False
        
        # Check rate limit
        minute_ago = current_time - 60
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > minute_ago
        ]
        
        if len(self._request_timestamps) >= self.config.requests_per_minute:
            return False
        
        # Allow request
        self._burst_tokens -= 1
        self._request_timestamps.append(current_time)
        return True
    
    def _update_response_time_metrics(self, response_time_ms: float) -> None:
        """Update response time metrics."""
        # Add to response times list
        self._response_times.append(response_time_ms)
        
        # Keep only recent response times
        if len(self._response_times) > self._max_response_times:
            self._response_times = self._response_times[-self._max_response_times:]
        
        # Calculate metrics
        if self._response_times:
            self.metrics.average_response_time_ms = sum(self._response_times) / len(self._response_times)
            
            sorted_times = sorted(self._response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            
            self.metrics.p95_response_time_ms = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            self.metrics.p99_response_time_ms = sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
    
    def optimize_query(self, query: str) -> str:
        """Optimize a search query for better performance."""
        if not self.config.enable_query_optimization:
            return query
        
        # Basic query optimization
        optimized = query.strip()
        
        # Remove excessive whitespace
        optimized = ' '.join(optimized.split())
        
        # Truncate if too long
        if len(optimized) > self.config.max_query_length:
            optimized = optimized[:self.config.max_query_length].rsplit(' ', 1)[0]
        
        # Ensure minimum length
        if len(optimized) < self.config.min_query_length:
            return query  # Return original if too short after optimization
        
        return optimized
    
    def generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a hash of the arguments
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": self.metrics.successful_requests / max(self.metrics.total_requests, 1),
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": self.metrics.cache_hits / max(self.metrics.cache_hits + self.metrics.cache_misses, 1),
            "average_response_time_ms": self.metrics.average_response_time_ms,
            "p95_response_time_ms": self.metrics.p95_response_time_ms,
            "p99_response_time_ms": self.metrics.p99_response_time_ms,
            "concurrent_requests": self.metrics.concurrent_requests,
            "max_concurrent_requests": self.metrics.max_concurrent_requests,
            "timeout_count": self.metrics.timeout_count,
            "throttled_requests": self.metrics.throttled_requests,
            "cache_size": len(self._cache),
            "cache_max_size": self.config.cache_max_size
        }
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._cache_access_count.clear()
        logger.info("Search cache cleared")


# Global instance
_performance_optimizer: Optional[SearchPerformanceOptimizer] = None


def get_performance_optimizer() -> SearchPerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = SearchPerformanceOptimizer()
    
    return _performance_optimizer
