"""
Performance Monitoring and Optimization Service

This module provides performance monitoring, benchmarking, and optimization
for ChromaDB vector operations and overall system performance.
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from agentic_rag.services.vector_store import get_vector_store, VectorMetadata
from agentic_rag.services.vector_operations import get_vector_operations, VectorData
from agentic_rag.services.collection_manager import get_collection_manager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    
    operation: str
    duration: float
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmark(BaseModel):
    """Performance benchmark results."""
    
    operation: str = Field(..., description="Operation being benchmarked")
    total_operations: int = Field(..., description="Total operations performed")
    successful_operations: int = Field(..., description="Successful operations")
    failed_operations: int = Field(..., description="Failed operations")
    average_duration: float = Field(..., description="Average operation duration")
    min_duration: float = Field(..., description="Minimum operation duration")
    max_duration: float = Field(..., description="Maximum operation duration")
    median_duration: float = Field(..., description="Median operation duration")
    p95_duration: float = Field(..., description="95th percentile duration")
    p99_duration: float = Field(..., description="99th percentile duration")
    operations_per_second: float = Field(..., description="Operations per second")
    benchmark_duration: float = Field(..., description="Total benchmark duration")
    timestamp: float = Field(..., description="When benchmark was performed")


class SystemPerformanceReport(BaseModel):
    """Comprehensive system performance report."""
    
    vector_operations: Dict[str, PerformanceBenchmark] = Field(default_factory=dict)
    collection_stats: Dict[str, Any] = Field(default_factory=dict)
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    overall_health: str = Field(..., description="Overall system health")
    report_duration: float = Field(..., description="Time to generate report")
    timestamp: float = Field(..., description="When report was generated")


class PerformanceMonitor:
    """Service for monitoring and optimizing ChromaDB performance."""
    
    def __init__(self):
        self._client = None
        self._vector_ops = None
        self._collection_manager = None
        
        # Performance tracking
        self._metrics: List[PerformanceMetric] = []
        self._max_metrics = 10000  # Keep last 10k metrics
        
        # Performance thresholds (in seconds)
        self._thresholds = {
            "query_warning": 0.1,      # 100ms
            "query_critical": 0.5,     # 500ms
            "add_warning": 1.0,        # 1s
            "add_critical": 5.0,       # 5s
            "batch_warning": 10.0,     # 10s
            "batch_critical": 30.0     # 30s
        }
        
        # Optimization settings
        self._optimal_batch_sizes = {
            "add_vectors": 100,
            "query_vectors": 50,
            "delete_vectors": 200
        }
        
        logger.info("Performance monitor initialized")
    
    async def initialize(self) -> None:
        """Initialize the performance monitor."""
        self._client = await get_vector_store()
        self._vector_ops = await get_vector_operations()
        self._collection_manager = await get_collection_manager()
        logger.info("Performance monitor connected to services")
    
    def record_metric(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            operation=operation,
            duration=duration,
            timestamp=time.time(),
            success=success,
            metadata=metadata or {}
        )
        
        self._metrics.append(metric)
        
        # Keep only recent metrics
        if len(self._metrics) > self._max_metrics:
            self._metrics = self._metrics[-self._max_metrics:]
        
        # Check for performance issues
        self._check_performance_threshold(metric)
    
    def _check_performance_threshold(self, metric: PerformanceMetric) -> None:
        """Check if a metric exceeds performance thresholds."""
        operation_type = metric.operation.lower()
        
        # Determine threshold keys based on operation
        if "query" in operation_type:
            warning_key = "query_warning"
            critical_key = "query_critical"
        elif "batch" in operation_type:
            warning_key = "batch_warning"
            critical_key = "batch_critical"
        else:
            warning_key = "add_warning"
            critical_key = "add_critical"
        
        if metric.duration > self._thresholds[critical_key]:
            logger.error(f"CRITICAL: {metric.operation} took {metric.duration:.2f}s (threshold: {self._thresholds[critical_key]}s)")
        elif metric.duration > self._thresholds[warning_key]:
            logger.warning(f"WARNING: {metric.operation} took {metric.duration:.2f}s (threshold: {self._thresholds[warning_key]}s)")
    
    async def benchmark_vector_operations(
        self,
        operations_to_test: Optional[List[str]] = None,
        test_duration: int = 60
    ) -> Dict[str, PerformanceBenchmark]:
        """
        Run performance benchmarks on vector operations.
        
        Args:
            operations_to_test: List of operations to benchmark
            test_duration: Duration of each test in seconds
            
        Returns:
            Dictionary of benchmark results by operation
        """
        if not self._client:
            await self.initialize()
        
        if operations_to_test is None:
            operations_to_test = ["add_vectors", "query_vectors", "delete_vectors"]
        
        benchmarks = {}
        
        logger.info(f"Starting performance benchmarks for {len(operations_to_test)} operations")
        
        for operation in operations_to_test:
            try:
                benchmark = await self._benchmark_operation(operation, test_duration)
                benchmarks[operation] = benchmark
                logger.info(f"Completed benchmark for {operation}: {benchmark.operations_per_second:.2f} ops/sec")
            except Exception as e:
                logger.error(f"Benchmark failed for {operation}: {e}")
        
        return benchmarks
    
    async def _benchmark_operation(self, operation: str, duration: int) -> PerformanceBenchmark:
        """Benchmark a specific operation."""
        start_time = time.time()
        end_time = start_time + duration
        
        durations = []
        successful = 0
        failed = 0
        
        logger.info(f"Benchmarking {operation} for {duration} seconds")
        
        while time.time() < end_time:
            try:
                op_start = time.time()
                
                if operation == "add_vectors":
                    await self._benchmark_add_operation()
                elif operation == "query_vectors":
                    await self._benchmark_query_operation()
                elif operation == "delete_vectors":
                    await self._benchmark_delete_operation()
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                op_duration = time.time() - op_start
                durations.append(op_duration)
                successful += 1
                
            except Exception as e:
                logger.debug(f"Benchmark operation failed: {e}")
                failed += 1
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
        
        total_operations = successful + failed
        benchmark_duration = time.time() - start_time
        
        if not durations:
            # No successful operations
            return PerformanceBenchmark(
                operation=operation,
                total_operations=total_operations,
                successful_operations=successful,
                failed_operations=failed,
                average_duration=0.0,
                min_duration=0.0,
                max_duration=0.0,
                median_duration=0.0,
                p95_duration=0.0,
                p99_duration=0.0,
                operations_per_second=0.0,
                benchmark_duration=benchmark_duration,
                timestamp=time.time()
            )
        
        # Calculate statistics
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        median_duration = statistics.median(durations)
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        p95_index = int(0.95 * len(sorted_durations))
        p99_index = int(0.99 * len(sorted_durations))
        p95_duration = sorted_durations[p95_index] if p95_index < len(sorted_durations) else max_duration
        p99_duration = sorted_durations[p99_index] if p99_index < len(sorted_durations) else max_duration
        
        ops_per_second = successful / benchmark_duration if benchmark_duration > 0 else 0
        
        return PerformanceBenchmark(
            operation=operation,
            total_operations=total_operations,
            successful_operations=successful,
            failed_operations=failed,
            average_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            median_duration=median_duration,
            p95_duration=p95_duration,
            p99_duration=p99_duration,
            operations_per_second=ops_per_second,
            benchmark_duration=benchmark_duration,
            timestamp=time.time()
        )
    
    async def _benchmark_add_operation(self) -> None:
        """Benchmark vector add operation."""
        # Create test vector
        metadata = VectorMetadata(
            tenant_id="benchmark-tenant",
            document_id="benchmark-doc",
            chunk_id=f"benchmark-chunk-{time.time()}",
            document_kind="RFQ",
            created_at=datetime.utcnow().isoformat()
        )
        
        vector_data = VectorData(
            id=f"benchmark-{time.time()}",
            embedding=[0.1] * 1536,  # Standard embedding size
            content="Benchmark test content",
            metadata=metadata
        )
        
        # Add single vector (for benchmark purposes)
        result = await self._vector_ops.add_vectors_batch([vector_data], batch_size=1)
        
        if not result.successful:
            raise Exception("Add operation failed")
    
    async def _benchmark_query_operation(self) -> None:
        """Benchmark vector query operation."""
        # Create test query embedding
        query_embedding = [0.1] * 1536
        
        # Perform query
        results = await self._vector_ops.search_vectors(
            query_embedding=query_embedding,
            document_kind="RFQ",
            tenant_id="benchmark-tenant"
        )
        
        # Query is successful even if no results found
    
    async def _benchmark_delete_operation(self) -> None:
        """Benchmark vector delete operation."""
        # For benchmark purposes, we'll just simulate delete
        # In practice, you'd delete actual vectors
        await asyncio.sleep(0.001)  # Simulate delete operation
    
    async def generate_performance_report(self) -> SystemPerformanceReport:
        """Generate comprehensive performance report."""
        if not self._client:
            await self.initialize()
        
        start_time = time.time()
        
        logger.info("Generating performance report")
        
        # Run benchmarks
        benchmarks = await self.benchmark_vector_operations(test_duration=30)
        
        # Get collection statistics
        collection_stats = {}
        try:
            rfq_info = await self._collection_manager.get_collection_info("rfq")
            offer_info = await self._collection_manager.get_collection_info("offer")
            
            collection_stats = {
                "rfq": rfq_info.model_dump(),
                "offer": offer_info.model_dump()
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            collection_stats = {"error": str(e)}
        
        # Get resource usage (simplified)
        resource_usage = {
            "metrics_count": len(self._metrics),
            "memory_usage": "unknown",  # Would need psutil for actual memory usage
            "cpu_usage": "unknown"
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(benchmarks)
        
        # Determine overall health
        overall_health = self._assess_overall_health(benchmarks)
        
        report_duration = time.time() - start_time
        
        return SystemPerformanceReport(
            vector_operations=benchmarks,
            collection_stats=collection_stats,
            resource_usage=resource_usage,
            recommendations=recommendations,
            overall_health=overall_health,
            report_duration=report_duration,
            timestamp=time.time()
        )
    
    def _generate_recommendations(self, benchmarks: Dict[str, PerformanceBenchmark]) -> List[str]:
        """Generate performance recommendations based on benchmarks."""
        recommendations = []
        
        for operation, benchmark in benchmarks.items():
            if benchmark.operations_per_second < 10:
                recommendations.append(f"Low throughput for {operation}: {benchmark.operations_per_second:.2f} ops/sec")
            
            if benchmark.p95_duration > self._thresholds.get(f"{operation}_warning", 1.0):
                recommendations.append(f"High latency for {operation}: P95 = {benchmark.p95_duration:.2f}s")
            
            if benchmark.failed_operations > benchmark.successful_operations * 0.1:
                recommendations.append(f"High failure rate for {operation}: {benchmark.failed_operations} failures")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable thresholds")
        
        return recommendations
    
    def _assess_overall_health(self, benchmarks: Dict[str, PerformanceBenchmark]) -> str:
        """Assess overall system health based on benchmarks."""
        if not benchmarks:
            return "unknown"
        
        total_ops = sum(b.operations_per_second for b in benchmarks.values())
        avg_ops = total_ops / len(benchmarks)
        
        if avg_ops > 50:
            return "excellent"
        elif avg_ops > 20:
            return "good"
        elif avg_ops > 10:
            return "fair"
        else:
            return "poor"
    
    def get_recent_metrics(self, operation: Optional[str] = None, limit: int = 100) -> List[PerformanceMetric]:
        """Get recent performance metrics."""
        metrics = self._metrics
        
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        return metrics[-limit:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self._metrics:
            return {"error": "No metrics available"}
        
        recent_metrics = self._metrics[-1000:]  # Last 1000 metrics
        
        operations = {}
        for metric in recent_metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric.duration)
        
        summary = {}
        for operation, durations in operations.items():
            if durations:
                summary[operation] = {
                    "count": len(durations),
                    "average": statistics.mean(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "median": statistics.median(durations)
                }
        
        return summary


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


async def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        await _performance_monitor.initialize()
    
    return _performance_monitor
