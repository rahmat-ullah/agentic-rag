"""
Embedding Resilience and Error Handling

This module provides comprehensive error handling, circuit breaker patterns,
fallback mechanisms, and dead letter queue for embedding operations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Types of errors that can occur during embedding operations."""
    
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    NETWORK = "network"
    VALIDATION = "validation"
    QUOTA_EXCEEDED = "quota_exceeded"
    UNKNOWN = "unknown"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class ErrorRecord(BaseModel):
    """Record of an error occurrence."""
    
    error_type: ErrorType = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When error occurred")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    tenant_id: Optional[str] = Field(None, description="Tenant associated with error")


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""
    
    failure_threshold: int = Field(default=5, description="Number of failures to open circuit")
    success_threshold: int = Field(default=3, description="Number of successes to close circuit")
    timeout: int = Field(default=60, description="Timeout in seconds before trying half-open")
    reset_timeout: int = Field(default=300, description="Timeout to reset failure count")


class CircuitBreaker:
    """Circuit breaker implementation for embedding operations."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        logger.info(f"Circuit breaker initialized: {config.model_dump()}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        # Check if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker moved to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - rejecting request")
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout
    
    async def _record_success(self) -> None:
        """Record a successful operation."""
        self.last_success_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count after successful operation
            if self.last_success_time and self.last_failure_time:
                if self.last_success_time - self.last_failure_time > self.config.reset_timeout:
                    self.failure_count = 0
    
    async def _record_failure(self, error: Exception) -> None:
        """Record a failed operation."""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            logger.warning("Circuit breaker returned to OPEN state")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "config": self.config.model_dump()
        }


class DeadLetterQueue:
    """Dead letter queue for failed embedding requests."""
    
    def __init__(self, max_size: int = 10000):
        self._queue: deque = deque(maxlen=max_size)
        self._max_size = max_size
        self._lock = asyncio.Lock()
        
        logger.info(f"Dead letter queue initialized with max size: {max_size}")
    
    async def add(self, item: Dict[str, Any]) -> None:
        """Add failed item to dead letter queue."""
        async with self._lock:
            item["queued_at"] = datetime.utcnow().isoformat()
            self._queue.append(item)
            
            logger.warning(f"Added item to dead letter queue: {item.get('batch_id', 'unknown')}")
    
    async def get_all(self) -> List[Dict[str, Any]]:
        """Get all items from dead letter queue."""
        async with self._lock:
            return list(self._queue)
    
    async def clear(self) -> int:
        """Clear all items from dead letter queue."""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            logger.info(f"Cleared {count} items from dead letter queue")
            return count
    
    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return len(self._queue)


class EmbeddingResilienceManager:
    """Manager for embedding resilience and error handling."""
    
    def __init__(self, circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        # Circuit breaker
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_config)
        
        # Dead letter queue
        self.dead_letter_queue = DeadLetterQueue()
        
        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.error_counts: Dict[ErrorType, int] = {error_type: 0 for error_type in ErrorType}
        
        # Rate limiting
        self.rate_limit_reset_time = None
        self.rate_limit_remaining = None
        
        logger.info("Embedding resilience manager initialized")
    
    async def execute_with_resilience(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        **kwargs
    ) -> Any:
        """
        Execute function with comprehensive resilience patterns.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retries
            backoff_factor: Exponential backoff factor
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
        """
        
        @retry(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(multiplier=backoff_factor, min=1, max=60),
            retry=retry_if_exception_type((Exception,))
        )
        async def _execute_with_retry():
            return await self.circuit_breaker.call(func, *args, **kwargs)
        
        try:
            result = await _execute_with_retry()
            return result
            
        except Exception as e:
            # Record error
            error_record = await self._record_error(e, args, kwargs)
            
            # Add to dead letter queue if all retries failed
            await self.dead_letter_queue.add({
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
                "error": error_record.model_dump(),
                "max_retries": max_retries
            })
            
            raise
    
    async def _record_error(
        self,
        error: Exception,
        args: tuple,
        kwargs: dict
    ) -> ErrorRecord:
        """Record and classify an error."""
        
        # Classify error type
        error_type = self._classify_error(error)
        
        # Create error record
        error_record = ErrorRecord(
            error_type=error_type,
            message=str(error),
            context={
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
                "error_class": error.__class__.__name__
            }
        )
        
        # Update statistics
        self.error_counts[error_type] += 1
        self.error_history.append(error_record)
        
        logger.error(f"Recorded error: {error_type.value} - {str(error)}")
        
        return error_record
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error into appropriate type."""
        error_str = str(error).lower()
        error_class = error.__class__.__name__.lower()
        
        if "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "authentication" in error_str or "401" in error_str or "403" in error_str:
            return ErrorType.AUTHENTICATION
        elif "timeout" in error_str or "timeouterror" in error_class:
            return ErrorType.TIMEOUT
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK
        elif "validation" in error_str or "400" in error_str:
            return ErrorType.VALIDATION
        elif "quota" in error_str or "limit exceeded" in error_str:
            return ErrorType.QUOTA_EXCEEDED
        elif "api" in error_str or "500" in error_str:
            return ErrorType.API_ERROR
        else:
            return ErrorType.UNKNOWN
    
    async def handle_rate_limit(self, retry_after: Optional[int] = None) -> None:
        """Handle rate limit with appropriate delay."""
        delay = retry_after or 60  # Default 60 seconds
        
        logger.warning(f"Rate limit hit, waiting {delay} seconds")
        
        self.rate_limit_reset_time = time.time() + delay
        await asyncio.sleep(delay)
    
    async def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors and resilience status."""
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        return {
            "circuit_breaker": self.circuit_breaker.get_status(),
            "error_counts": self.error_counts,
            "recent_errors_count": len(recent_errors),
            "dead_letter_queue_size": await self.dead_letter_queue.size(),
            "rate_limit_reset_time": self.rate_limit_reset_time,
            "total_errors": len(self.error_history)
        }
    
    async def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        self.circuit_breaker.state = CircuitBreakerState.CLOSED
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.success_count = 0
        logger.info("Circuit breaker manually reset")
    
    async def clear_error_history(self) -> int:
        """Clear error history."""
        count = len(self.error_history)
        self.error_history.clear()
        self.error_counts = {error_type: 0 for error_type in ErrorType}
        logger.info(f"Cleared {count} error records")
        return count
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on resilience manager."""
        try:
            summary = await self.get_error_summary()
            
            # Determine health status
            circuit_state = summary["circuit_breaker"]["state"]
            recent_errors = summary["recent_errors_count"]
            dlq_size = summary["dead_letter_queue_size"]
            
            if circuit_state == "open":
                status = "unhealthy"
                reason = "Circuit breaker is open"
            elif recent_errors > 50:
                status = "degraded"
                reason = f"High error rate: {recent_errors} errors in last hour"
            elif dlq_size > 100:
                status = "degraded"
                reason = f"High dead letter queue size: {dlq_size}"
            else:
                status = "healthy"
                reason = "All systems operational"
            
            return {
                "status": status,
                "reason": reason,
                "summary": summary,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Resilience manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "reason": f"Health check failed: {str(e)}",
                "timestamp": time.time()
            }


# Global resilience manager instance
_resilience_manager: Optional[EmbeddingResilienceManager] = None


async def get_resilience_manager() -> EmbeddingResilienceManager:
    """Get or create the global resilience manager instance."""
    global _resilience_manager
    
    if _resilience_manager is None:
        _resilience_manager = EmbeddingResilienceManager()
    
    return _resilience_manager


async def close_resilience_manager() -> None:
    """Close the global resilience manager instance."""
    global _resilience_manager
    
    if _resilience_manager:
        _resilience_manager = None
