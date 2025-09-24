"""
Indexing Error Handler Service

This module provides comprehensive error handling and recovery
for vector indexing operations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Types of indexing errors."""
    
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_STORAGE = "vector_storage"
    METADATA_VALIDATION = "metadata_validation"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    
    LOW = "low"          # Recoverable, retry immediately
    MEDIUM = "medium"    # Recoverable, retry with backoff
    HIGH = "high"        # Requires intervention, send to DLQ
    CRITICAL = "critical" # System-level issue, stop processing


class RecoveryAction(str, Enum):
    """Recovery actions for different error types."""
    
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_BACKOFF = "retry_backoff"
    RETRY_LATER = "retry_later"
    SEND_TO_DLQ = "send_to_dlq"
    SKIP_TASK = "skip_task"
    STOP_PROCESSING = "stop_processing"


@dataclass
class ErrorInfo:
    """Information about an indexing error."""

    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    exception_type: Optional[str] = None
    task_id: Optional[str] = None
    timestamp: datetime = None
    retry_count: int = 0
    recovery_action: Optional[RecoveryAction] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class RetryPolicy(BaseModel):
    """Retry policy configuration."""
    
    max_attempts: int = Field(default=3, description="Maximum retry attempts")
    initial_wait: float = Field(default=1.0, description="Initial wait time in seconds")
    max_wait: float = Field(default=60.0, description="Maximum wait time in seconds")
    exponential_base: float = Field(default=2.0, description="Exponential backoff base")
    jitter: bool = Field(default=True, description="Add random jitter to wait times")


class DeadLetterQueueItem(BaseModel):
    """Item in the dead letter queue."""

    task_id: str = Field(..., description="Task identifier")
    original_task_data: Dict[str, Any] = Field(..., description="Original task data")
    error_type: str = Field(..., description="Error type")
    error_message: str = Field(..., description="Error message")
    attempts: int = Field(default=0, description="Number of retry attempts")
    first_failure: datetime = Field(default_factory=datetime.utcnow, description="First failure timestamp")
    last_failure: datetime = Field(default_factory=datetime.utcnow, description="Last failure timestamp")
    can_retry: bool = Field(default=True, description="Whether item can be retried")


class IndexingErrorHandler:
    """Comprehensive error handler for indexing operations."""
    
    def __init__(self, max_dlq_size: int = 10000):
        self._error_classifiers: Dict[type, ErrorType] = self._initialize_error_classifiers()
        self._retry_policies: Dict[ErrorType, RetryPolicy] = self._initialize_retry_policies()
        self._recovery_strategies: Dict[ErrorType, RecoveryAction] = self._initialize_recovery_strategies()
        
        # Dead letter queue
        self._dlq: deque = deque(maxlen=max_dlq_size)
        self._dlq_lock = asyncio.Lock()
        
        # Error statistics
        self._error_stats: Dict[str, int] = {}
        self._recovery_stats: Dict[str, int] = {}
        
        # Circuit breaker state
        self._circuit_breaker_state: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Indexing error handler initialized with DLQ size: {max_dlq_size}")
    
    def _initialize_error_classifiers(self) -> Dict[type, ErrorType]:
        """Initialize error type classifiers."""
        return {
            ConnectionError: ErrorType.NETWORK_ERROR,
            TimeoutError: ErrorType.TIMEOUT,
            asyncio.TimeoutError: ErrorType.TIMEOUT,
            PermissionError: ErrorType.AUTHENTICATION,
            ValueError: ErrorType.DATA_CORRUPTION,
            KeyError: ErrorType.DATA_CORRUPTION,
            MemoryError: ErrorType.RESOURCE_EXHAUSTION,
            OSError: ErrorType.NETWORK_ERROR,
        }
    
    def _initialize_retry_policies(self) -> Dict[ErrorType, RetryPolicy]:
        """Initialize retry policies for different error types."""
        return {
            ErrorType.EMBEDDING_GENERATION: RetryPolicy(max_attempts=3, initial_wait=2.0, max_wait=30.0),
            ErrorType.VECTOR_STORAGE: RetryPolicy(max_attempts=5, initial_wait=1.0, max_wait=60.0),
            ErrorType.METADATA_VALIDATION: RetryPolicy(max_attempts=2, initial_wait=0.5, max_wait=5.0),
            ErrorType.NETWORK_ERROR: RetryPolicy(max_attempts=5, initial_wait=2.0, max_wait=120.0),
            ErrorType.RATE_LIMIT: RetryPolicy(max_attempts=10, initial_wait=5.0, max_wait=300.0),
            ErrorType.AUTHENTICATION: RetryPolicy(max_attempts=2, initial_wait=1.0, max_wait=10.0),
            ErrorType.RESOURCE_EXHAUSTION: RetryPolicy(max_attempts=3, initial_wait=10.0, max_wait=180.0),
            ErrorType.DATA_CORRUPTION: RetryPolicy(max_attempts=1, initial_wait=1.0, max_wait=5.0),
            ErrorType.TIMEOUT: RetryPolicy(max_attempts=3, initial_wait=5.0, max_wait=60.0),
            ErrorType.UNKNOWN: RetryPolicy(max_attempts=2, initial_wait=2.0, max_wait=30.0),
        }
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorType, RecoveryAction]:
        """Initialize recovery strategies for different error types."""
        return {
            ErrorType.EMBEDDING_GENERATION: RecoveryAction.RETRY_BACKOFF,
            ErrorType.VECTOR_STORAGE: RecoveryAction.RETRY_BACKOFF,
            ErrorType.METADATA_VALIDATION: RecoveryAction.SEND_TO_DLQ,
            ErrorType.NETWORK_ERROR: RecoveryAction.RETRY_BACKOFF,
            ErrorType.RATE_LIMIT: RecoveryAction.RETRY_LATER,
            ErrorType.AUTHENTICATION: RecoveryAction.STOP_PROCESSING,
            ErrorType.RESOURCE_EXHAUSTION: RecoveryAction.RETRY_LATER,
            ErrorType.DATA_CORRUPTION: RecoveryAction.SEND_TO_DLQ,
            ErrorType.TIMEOUT: RecoveryAction.RETRY_BACKOFF,
            ErrorType.UNKNOWN: RecoveryAction.RETRY_BACKOFF,
        }
    
    async def handle_error(
        self,
        exception: Exception,
        task_id: str,
        task_data: Dict[str, Any],
        retry_count: int = 0
    ) -> RecoveryAction:
        """
        Handle an indexing error and determine recovery action.
        
        Args:
            exception: The exception that occurred
            task_id: Task identifier
            task_data: Original task data
            retry_count: Current retry count
            
        Returns:
            RecoveryAction to take
        """
        try:
            # Classify the error
            error_info = await self._classify_error(exception, task_id, retry_count)
            
            # Update error statistics
            self._update_error_stats(error_info)
            
            # Check circuit breaker
            if await self._should_circuit_break(error_info):
                error_info.recovery_action = RecoveryAction.STOP_PROCESSING
                logger.warning(f"Circuit breaker triggered for {error_info.error_type.value}")
            else:
                # Determine recovery action
                error_info.recovery_action = await self._determine_recovery_action(error_info, retry_count)
            
            # Execute recovery action
            await self._execute_recovery_action(error_info, task_id, task_data)
            
            logger.info(f"Handled error for task {task_id}: {error_info.error_type.value} -> {error_info.recovery_action.value}")
            
            return error_info.recovery_action
            
        except Exception as e:
            logger.error(f"Error handler failed for task {task_id}: {e}")
            return RecoveryAction.SEND_TO_DLQ
    
    async def _classify_error(self, exception: Exception, task_id: str, retry_count: int) -> ErrorInfo:
        """Classify an error and determine its type and severity."""
        error_message = str(exception)
        error_type = ErrorType.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        
        # Classify by exception type
        for exc_type, err_type in self._error_classifiers.items():
            if isinstance(exception, exc_type):
                error_type = err_type
                break
        
        # Classify by error message patterns
        if not error_type or error_type == ErrorType.UNKNOWN:
            error_message_lower = error_message.lower()
            
            if any(pattern in error_message_lower for pattern in ['rate limit', 'too many requests', 'quota exceeded']):
                error_type = ErrorType.RATE_LIMIT
                severity = ErrorSeverity.MEDIUM
            elif any(pattern in error_message_lower for pattern in ['unauthorized', 'authentication', 'api key']):
                error_type = ErrorType.AUTHENTICATION
                severity = ErrorSeverity.CRITICAL
            elif any(pattern in error_message_lower for pattern in ['timeout', 'timed out']):
                error_type = ErrorType.TIMEOUT
                severity = ErrorSeverity.MEDIUM
            elif any(pattern in error_message_lower for pattern in ['connection', 'network', 'dns']):
                error_type = ErrorType.NETWORK_ERROR
                severity = ErrorSeverity.MEDIUM
            elif any(pattern in error_message_lower for pattern in ['memory', 'resource', 'capacity']):
                error_type = ErrorType.RESOURCE_EXHAUSTION
                severity = ErrorSeverity.HIGH
            elif any(pattern in error_message_lower for pattern in ['embedding', 'openai']):
                error_type = ErrorType.EMBEDDING_GENERATION
                severity = ErrorSeverity.MEDIUM
            elif any(pattern in error_message_lower for pattern in ['vector', 'chromadb', 'storage']):
                error_type = ErrorType.VECTOR_STORAGE
                severity = ErrorSeverity.MEDIUM
            elif any(pattern in error_message_lower for pattern in ['metadata', 'validation']):
                error_type = ErrorType.METADATA_VALIDATION
                severity = ErrorSeverity.LOW
        
        # Adjust severity based on retry count
        if retry_count > 3:
            if severity == ErrorSeverity.LOW:
                severity = ErrorSeverity.MEDIUM
            elif severity == ErrorSeverity.MEDIUM:
                severity = ErrorSeverity.HIGH
        
        return ErrorInfo(
            error_type=error_type,
            severity=severity,
            message=error_message,
            exception_type=type(exception).__name__,
            task_id=task_id,
            retry_count=retry_count
        )
    
    async def _determine_recovery_action(self, error_info: ErrorInfo, retry_count: int) -> RecoveryAction:
        """Determine the appropriate recovery action for an error."""
        # Get default recovery strategy
        recovery_action = self._recovery_strategies.get(error_info.error_type, RecoveryAction.RETRY_BACKOFF)
        
        # Check retry limits
        retry_policy = self._retry_policies.get(error_info.error_type)
        if retry_policy and retry_count >= retry_policy.max_attempts:
            recovery_action = RecoveryAction.SEND_TO_DLQ
        
        # Override based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            recovery_action = RecoveryAction.STOP_PROCESSING
        elif error_info.severity == ErrorSeverity.HIGH and retry_count > 1:
            recovery_action = RecoveryAction.SEND_TO_DLQ
        
        return recovery_action
    
    async def _execute_recovery_action(
        self,
        error_info: ErrorInfo,
        task_id: str,
        task_data: Dict[str, Any]
    ) -> None:
        """Execute the determined recovery action."""
        action = error_info.recovery_action
        
        if action == RecoveryAction.SEND_TO_DLQ:
            await self._send_to_dlq(task_id, task_data, error_info)
        
        elif action == RecoveryAction.RETRY_LATER:
            # Schedule for later retry (implementation depends on task scheduler)
            logger.info(f"Scheduling task {task_id} for later retry")
        
        elif action == RecoveryAction.STOP_PROCESSING:
            # Signal to stop processing (implementation depends on pipeline)
            logger.critical(f"Stopping processing due to critical error: {error_info.message}")
        
        # Update recovery statistics
        self._recovery_stats[action.value] = self._recovery_stats.get(action.value, 0) + 1
    
    async def _send_to_dlq(self, task_id: str, task_data: Dict[str, Any], error_info: ErrorInfo) -> None:
        """Send a failed task to the dead letter queue."""
        async with self._dlq_lock:
            dlq_item = DeadLetterQueueItem(
                task_id=task_id,
                original_task_data=task_data,
                error_type=error_info.error_type.value,
                error_message=error_info.message,
                attempts=error_info.retry_count,
                first_failure=error_info.timestamp,
                last_failure=error_info.timestamp
            )
            
            self._dlq.append(dlq_item)
            
            logger.warning(f"Sent task {task_id} to dead letter queue: {error_info.error_type.value}")
    
    async def _should_circuit_break(self, error_info: ErrorInfo) -> bool:
        """Check if circuit breaker should be triggered."""
        error_type = error_info.error_type.value
        
        if error_type not in self._circuit_breaker_state:
            self._circuit_breaker_state[error_type] = {
                'failure_count': 0,
                'last_failure': None,
                'state': 'closed'  # closed, open, half-open
            }
        
        state = self._circuit_breaker_state[error_type]
        
        # Update failure count
        state['failure_count'] += 1
        state['last_failure'] = datetime.utcnow()
        
        # Check if we should open the circuit
        if state['state'] == 'closed' and state['failure_count'] >= 5:
            state['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {error_type}")
            return True
        
        # Check if we should try half-open
        if state['state'] == 'open':
            time_since_failure = datetime.utcnow() - state['last_failure']
            if time_since_failure > timedelta(minutes=5):
                state['state'] = 'half-open'
                state['failure_count'] = 0
                logger.info(f"Circuit breaker half-open for {error_type}")
        
        return state['state'] == 'open'
    
    def _update_error_stats(self, error_info: ErrorInfo) -> None:
        """Update error statistics."""
        error_key = f"{error_info.error_type.value}_{error_info.severity.value}"
        self._error_stats[error_key] = self._error_stats.get(error_key, 0) + 1
    
    async def get_dlq_items(self, limit: int = 100) -> List[DeadLetterQueueItem]:
        """Get items from the dead letter queue."""
        async with self._dlq_lock:
            return list(self._dlq)[-limit:]
    
    async def retry_dlq_item(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retry a specific item from the dead letter queue."""
        async with self._dlq_lock:
            for i, item in enumerate(self._dlq):
                if item.task_id == task_id and item.can_retry:
                    # Remove from DLQ and return task data
                    del self._dlq[i]
                    logger.info(f"Retrying DLQ item: {task_id}")
                    return item.original_task_data
            
            return None
    
    async def clear_dlq(self) -> int:
        """Clear all items from the dead letter queue."""
        async with self._dlq_lock:
            count = len(self._dlq)
            self._dlq.clear()
            logger.info(f"Cleared {count} items from dead letter queue")
            return count
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            'error_counts': self._error_stats.copy(),
            'recovery_counts': self._recovery_stats.copy(),
            'dlq_size': len(self._dlq),
            'circuit_breaker_states': {
                error_type: state['state'] 
                for error_type, state in self._circuit_breaker_state.items()
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of error handling system."""
        total_errors = sum(self._error_stats.values())
        dlq_size = len(self._dlq)
        
        # Determine health status
        if dlq_size > 1000 or any(state['state'] == 'open' for state in self._circuit_breaker_state.values()):
            status = 'unhealthy'
        elif dlq_size > 100 or total_errors > 100:
            status = 'degraded'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'total_errors': total_errors,
            'dlq_size': dlq_size,
            'circuit_breakers_open': sum(
                1 for state in self._circuit_breaker_state.values() 
                if state['state'] == 'open'
            ),
            'timestamp': datetime.utcnow().isoformat()
        }


# Global error handler instance
_error_handler: Optional[IndexingErrorHandler] = None


async def get_error_handler() -> IndexingErrorHandler:
    """Get or create the global error handler instance."""
    global _error_handler
    
    if _error_handler is None:
        _error_handler = IndexingErrorHandler()
    
    return _error_handler


async def handle_indexing_error(
    exception: Exception,
    task_id: str,
    task_data: Dict[str, Any],
    retry_count: int = 0
) -> RecoveryAction:
    """
    Convenience function to handle indexing errors.
    
    Args:
        exception: The exception that occurred
        task_id: Task identifier
        task_data: Original task data
        retry_count: Current retry count
        
    Returns:
        RecoveryAction to take
    """
    handler = await get_error_handler()
    return await handler.handle_error(exception, task_id, task_data, retry_count)
