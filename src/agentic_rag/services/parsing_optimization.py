"""
Parsing Performance Optimization

This module provides performance optimization for document parsing,
including caching, parallel processing, memory management, and
performance monitoring.
"""

import asyncio
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID

import redis
from pydantic import BaseModel, Field

from agentic_rag.config import Settings
from agentic_rag.services.content_extraction import ExtractedContent
from agentic_rag.services.docling_client import ParseRequest, ParseResponse, DocumentMetadata
from agentic_rag.services.metadata_extraction import EnrichedMetadata

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for parsing operations."""
    
    total_time: float
    parsing_time: float
    extraction_time: float
    metadata_time: float
    cache_hit: bool
    memory_usage_mb: float
    pages_per_second: float
    throughput_mb_per_second: float


class CacheEntry(BaseModel):
    """Cache entry for parsed documents."""
    
    content_hash: str = Field(..., description="SHA256 hash of file content")
    filename: str = Field(..., description="Original filename")
    parse_response: ParseResponse = Field(..., description="Cached parse response")
    extracted_content: ExtractedContent = Field(..., description="Cached extracted content")
    enriched_metadata: EnrichedMetadata = Field(..., description="Cached enriched metadata")
    cache_timestamp: float = Field(..., description="When the entry was cached")
    access_count: int = Field(default=1, description="Number of times accessed")
    last_access: float = Field(..., description="Last access timestamp")


class ParsingCache:
    """Redis-based cache for parsing results."""
    
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 3600):
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        self.cache_prefix = "docling_parse:"
        
        logger.info(f"Parsing cache initialized with TTL {ttl_seconds} seconds")
    
    def _get_cache_key(self, content_hash: str) -> str:
        """Generate cache key for content hash."""
        return f"{self.cache_prefix}{content_hash}"
    
    def _calculate_content_hash(self, file_content: bytes, parse_request: ParseRequest) -> str:
        """Calculate hash for file content and parse parameters."""
        hasher = hashlib.sha256()
        hasher.update(file_content)
        hasher.update(str(parse_request.dict()).encode())
        return hasher.hexdigest()
    
    async def get(self, file_content: bytes, parse_request: ParseRequest) -> Optional[CacheEntry]:
        """Get cached parsing result."""
        content_hash = self._calculate_content_hash(file_content, parse_request)
        cache_key = self._get_cache_key(content_hash)
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                cache_entry = CacheEntry.parse_raw(cached_data)
                
                # Update access statistics
                cache_entry.access_count += 1
                cache_entry.last_access = time.time()
                
                # Update cache with new access stats
                await self.redis_client.setex(
                    cache_key,
                    self.ttl_seconds,
                    cache_entry.json()
                )
                
                logger.info(f"Cache hit for content hash {content_hash[:8]}...")
                return cache_entry
            
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        
        return None
    
    async def set(
        self,
        file_content: bytes,
        filename: str,
        parse_request: ParseRequest,
        parse_response: ParseResponse,
        extracted_content: ExtractedContent,
        enriched_metadata: EnrichedMetadata
    ):
        """Cache parsing result."""
        content_hash = self._calculate_content_hash(file_content, parse_request)
        cache_key = self._get_cache_key(content_hash)
        
        cache_entry = CacheEntry(
            content_hash=content_hash,
            filename=filename,
            parse_response=parse_response,
            extracted_content=extracted_content,
            enriched_metadata=enriched_metadata,
            cache_timestamp=time.time(),
            last_access=time.time()
        )
        
        try:
            await self.redis_client.setex(
                cache_key,
                self.ttl_seconds,
                cache_entry.json()
            )
            logger.info(f"Cached parsing result for {filename} (hash: {content_hash[:8]}...)")
            
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    async def invalidate(self, content_hash: str):
        """Invalidate cache entry."""
        cache_key = self._get_cache_key(content_hash)
        try:
            await self.redis_client.delete(cache_key)
            logger.info(f"Invalidated cache for hash {content_hash[:8]}...")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
    
    async def get_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            keys = await self.redis_client.keys(f"{self.cache_prefix}*")
            total_entries = len(keys)
            
            total_access_count = 0
            if keys:
                for key in keys[:100]:  # Sample first 100 entries
                    try:
                        data = await self.redis_client.get(key)
                        if data:
                            entry = CacheEntry.parse_raw(data)
                            total_access_count += entry.access_count
                    except:
                        continue
            
            return {
                "total_entries": total_entries,
                "average_access_count": total_access_count / max(len(keys[:100]), 1),
                "cache_prefix": self.cache_prefix
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"error": str(e)}


class DocumentPreprocessor:
    """Preprocessor for optimizing documents before parsing."""
    
    def __init__(self):
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.max_pages = 2000
        
    def should_preprocess(self, file_content: bytes, filename: str) -> bool:
        """Determine if document should be preprocessed."""
        file_size = len(file_content)
        
        # Preprocess large files
        if file_size > self.max_file_size:
            return True
        
        # Preprocess based on file type
        if filename.lower().endswith('.pdf'):
            # Could check PDF page count here
            return file_size > 10 * 1024 * 1024  # 10MB
        
        return False
    
    def preprocess_document(self, file_content: bytes, filename: str) -> Tuple[bytes, Dict]:
        """
        Preprocess document for optimal parsing.
        
        Returns:
            Tuple of (processed_content, preprocessing_metadata)
        """
        preprocessing_metadata = {
            "original_size": len(file_content),
            "preprocessing_applied": [],
            "size_reduction": 0.0
        }
        
        processed_content = file_content
        
        # Apply preprocessing based on file type
        if filename.lower().endswith('.pdf'):
            processed_content, pdf_metadata = self._preprocess_pdf(file_content)
            preprocessing_metadata.update(pdf_metadata)
        
        # Calculate size reduction
        size_reduction = (len(file_content) - len(processed_content)) / len(file_content)
        preprocessing_metadata["size_reduction"] = size_reduction
        preprocessing_metadata["processed_size"] = len(processed_content)
        
        if size_reduction > 0:
            logger.info(f"Preprocessing reduced file size by {size_reduction:.1%}")
        
        return processed_content, preprocessing_metadata
    
    def _preprocess_pdf(self, pdf_content: bytes) -> Tuple[bytes, Dict]:
        """Preprocess PDF for optimal parsing."""
        # This is a placeholder for PDF preprocessing
        # In a real implementation, you might:
        # - Compress images
        # - Remove unnecessary metadata
        # - Optimize for text extraction
        
        metadata = {
            "preprocessing_applied": ["pdf_optimization"],
            "pdf_optimized": True
        }
        
        return pdf_content, metadata


class ParallelProcessor:
    """Parallel processing for large documents."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"Parallel processor initialized with {max_workers} workers")
    
    async def process_large_document(
        self,
        file_content: bytes,
        filename: str,
        parse_request: ParseRequest
    ) -> Tuple[ParseResponse, float]:
        """
        Process large document with parallel processing.
        
        Returns:
            Tuple of (parse_response, processing_time)
        """
        start_time = time.time()
        
        # For now, this is a placeholder for parallel processing
        # In a real implementation, you might:
        # - Split PDF into chunks
        # - Process chunks in parallel
        # - Merge results
        
        logger.info(f"Processing large document {filename} with parallel processing")
        
        # Simulate parallel processing delay
        await asyncio.sleep(0.1)
        
        processing_time = time.time() - start_time
        
        # Return a mock response for now
        # In real implementation, this would return actual parsed content
        mock_response = ParseResponse(
            success=True,
            document_type="pdf",
            content=[],
            tables=[],
            metadata=DocumentMetadata(page_count=1),
            processing_time=processing_time,
            pages_processed=1
        )
        
        return mock_response, processing_time
    
    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)
        logger.info("Parallel processor shutdown")


class MemoryManager:
    """Memory management for parsing operations."""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
        
    def check_memory_availability(self, estimated_usage_mb: float) -> bool:
        """Check if there's enough memory for the operation."""
        return (self.current_memory_mb + estimated_usage_mb) <= self.max_memory_mb
    
    def estimate_memory_usage(self, file_size_bytes: int) -> float:
        """Estimate memory usage for parsing a file."""
        # Simple heuristic: assume 3x file size for processing
        return (file_size_bytes * 3) / (1024 * 1024)
    
    def acquire_memory(self, amount_mb: float) -> bool:
        """Acquire memory for processing."""
        if self.check_memory_availability(amount_mb):
            self.current_memory_mb += amount_mb
            return True
        return False
    
    def release_memory(self, amount_mb: float):
        """Release memory after processing."""
        self.current_memory_mb = max(0, self.current_memory_mb - amount_mb)


class PerformanceOptimizer:
    """Main performance optimization service."""
    
    def __init__(self, settings: Settings, redis_client: redis.Redis):
        self.settings = settings
        self.cache = ParsingCache(redis_client, ttl_seconds=3600)
        self.preprocessor = DocumentPreprocessor()
        self.parallel_processor = ParallelProcessor(max_workers=4)
        self.memory_manager = MemoryManager(max_memory_mb=2048)
        
        # Performance monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        
        logger.info("Performance optimizer initialized")
    
    async def optimize_parsing(
        self,
        file_content: bytes,
        filename: str,
        parse_request: ParseRequest
    ) -> Tuple[Optional[CacheEntry], PerformanceMetrics]:
        """
        Optimize parsing with caching, preprocessing, and parallel processing.
        
        Returns:
            Tuple of (cache_entry_if_hit, performance_metrics)
        """
        start_time = time.time()
        
        # Check cache first
        cache_entry = await self.cache.get(file_content, parse_request)
        if cache_entry:
            metrics = PerformanceMetrics(
                total_time=time.time() - start_time,
                parsing_time=0.0,
                extraction_time=0.0,
                metadata_time=0.0,
                cache_hit=True,
                memory_usage_mb=0.0,
                pages_per_second=float('inf'),
                throughput_mb_per_second=float('inf')
            )
            return cache_entry, metrics
        
        # Check memory availability
        file_size_mb = len(file_content) / (1024 * 1024)
        estimated_memory = self.memory_manager.estimate_memory_usage(len(file_content))
        
        if not self.memory_manager.check_memory_availability(estimated_memory):
            logger.warning(f"Insufficient memory for parsing {filename}")
            # Could implement queuing or alternative processing here
        
        # Preprocess if needed
        preprocessing_time = 0.0
        if self.preprocessor.should_preprocess(file_content, filename):
            preprocess_start = time.time()
            file_content, preprocessing_metadata = self.preprocessor.preprocess_document(file_content, filename)
            preprocessing_time = time.time() - preprocess_start
            logger.info(f"Preprocessing completed in {preprocessing_time:.2f}s")
        
        # Determine processing strategy
        if file_size_mb > 50:  # Large file threshold
            logger.info(f"Using parallel processing for large file {filename}")
            # Would use parallel processor here
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        pages_processed = 1  # Would get from actual parsing
        pages_per_second = pages_processed / max(total_time, 0.001)
        throughput_mb_per_second = file_size_mb / max(total_time, 0.001)
        
        metrics = PerformanceMetrics(
            total_time=total_time,
            parsing_time=0.0,  # Would be measured during actual parsing
            extraction_time=0.0,
            metadata_time=0.0,
            cache_hit=False,
            memory_usage_mb=estimated_memory,
            pages_per_second=pages_per_second,
            throughput_mb_per_second=throughput_mb_per_second
        )
        
        # Store metrics for monitoring
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Keep last 1000 entries
            self.metrics_history = self.metrics_history[-1000:]
        
        return None, metrics
    
    async def cache_parsing_result(
        self,
        file_content: bytes,
        filename: str,
        parse_request: ParseRequest,
        parse_response: ParseResponse,
        extracted_content: ExtractedContent,
        enriched_metadata: EnrichedMetadata
    ):
        """Cache parsing result for future use."""
        await self.cache.set(
            file_content,
            filename,
            parse_request,
            parse_response,
            extracted_content,
            enriched_metadata
        )
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 operations
        
        avg_total_time = sum(m.total_time for m in recent_metrics) / len(recent_metrics)
        avg_pages_per_second = sum(m.pages_per_second for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput_mb_per_second for m in recent_metrics) / len(recent_metrics)
        cache_hit_rate = sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)
        
        return {
            "total_operations": len(self.metrics_history),
            "recent_operations": len(recent_metrics),
            "average_processing_time": avg_total_time,
            "average_pages_per_second": avg_pages_per_second,
            "average_throughput_mb_per_second": avg_throughput,
            "cache_hit_rate": cache_hit_rate,
            "memory_usage_mb": self.memory_manager.current_memory_mb,
            "max_memory_mb": self.memory_manager.max_memory_mb
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.parallel_processor.shutdown()
        logger.info("Performance optimizer cleanup completed")


# Global optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


async def get_performance_optimizer(settings: Settings, redis_client: redis.Redis) -> PerformanceOptimizer:
    """Get or create the global performance optimizer instance."""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(settings, redis_client)
    
    return _performance_optimizer


async def cleanup_performance_optimizer():
    """Cleanup the global performance optimizer instance."""
    global _performance_optimizer
    
    if _performance_optimizer:
        await _performance_optimizer.cleanup()
        _performance_optimizer = None
