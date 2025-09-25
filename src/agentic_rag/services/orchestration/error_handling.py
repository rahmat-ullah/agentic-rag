"""
Enhanced Error Handling and Recovery System

This module implements comprehensive error handling and recovery mechanisms
for the agent orchestration framework, including failure detection, retry
strategies, circuit breakers, error propagation, and recovery procedures.
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
from uuid import uuid4
import threading
import weakref

from pydantic import BaseModel, Field
import structlog

from .base import Agent, Task, Context, Result, WorkflowError, AgentCapability, TaskStatus
from .communication import Message, MessageType, MessagePriority

logger = structlog.get_logger(__name__)


class FailureType(str, Enum):
    """Types of agent failures."""
    
    TIMEOUT = "timeout"                    # Agent response timeout
    EXCEPTION = "exception"                # Unhandled exception
    PERFORMANCE_DEGRADATION = "performance_degradation"  # Slow response
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Out of resources
    NETWORK_ERROR = "network_error"        # Network connectivity issues
    AUTHENTICATION_ERROR = "auth_error"    # Authentication failures
    AUTHORIZATION_ERROR = "authz_error"    # Authorization failures
    VALIDATION_ERROR = "validation_error"  # Input validation failures
    BUSINESS_LOGIC_ERROR = "business_error"  # Business rule violations
    UNKNOWN_ERROR = "unknown_error"        # Unclassified errors


class FailureSeverity(str, Enum):
    """Severity levels for failures."""
    
    LOW = "low"                           # Minor issues, retry recommended
    MEDIUM = "medium"                     # Moderate issues, fallback recommended
    HIGH = "high"                         # Serious issues, circuit breaker activation
    CRITICAL = "critical"                 # System-threatening issues, immediate action


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"                     # Normal operation
    OPEN = "open"                         # Blocking requests due to failures
    HALF_OPEN = "half_open"              # Testing if service has recovered


class RetryStrategy(str, Enum):
    """Retry strategy types."""
    
    FIXED_DELAY = "fixed_delay"           # Fixed delay between retries
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential delay increase
    LINEAR_BACKOFF = "linear_backoff"     # Linear delay increase
    JITTERED_BACKOFF = "jittered_backoff"  # Exponential with random jitter
    NO_RETRY = "no_retry"                 # No retry attempts


class FailureEvent(BaseModel):
    """Represents a failure event."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str = Field(..., description="Agent that failed")
    task_id: Optional[str] = Field(None, description="Task being executed")
    failure_type: FailureType = Field(..., description="Type of failure")
    severity: FailureSeverity = Field(..., description="Failure severity")
    
    # Failure details
    error_message: str = Field(..., description="Error message")
    error_details: Dict[str, Any] = Field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    # Context
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metrics
    response_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True


class RetryPolicy(BaseModel):
    """Retry policy configuration."""
    
    strategy: RetryStrategy = Field(default=RetryStrategy.EXPONENTIAL_BACKOFF)
    max_attempts: int = Field(default=3, ge=0, le=10)
    initial_delay_ms: int = Field(default=1000, ge=0)
    max_delay_ms: int = Field(default=30000, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    jitter_factor: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Failure type specific settings
    failure_type_overrides: Dict[FailureType, Dict[str, Any]] = Field(default_factory=dict)
    
    def get_delay_ms(self, attempt: int, failure_type: Optional[FailureType] = None) -> int:
        """Calculate delay for given attempt number."""
        # Check for failure type specific overrides
        if failure_type and failure_type in self.failure_type_overrides:
            overrides = self.failure_type_overrides[failure_type]
            strategy = RetryStrategy(overrides.get('strategy', self.strategy))
            max_attempts = overrides.get('max_attempts', self.max_attempts)
            initial_delay = overrides.get('initial_delay_ms', self.initial_delay_ms)
            max_delay = overrides.get('max_delay_ms', self.max_delay_ms)
            multiplier = overrides.get('backoff_multiplier', self.backoff_multiplier)
            jitter = overrides.get('jitter_factor', self.jitter_factor)
        else:
            strategy = self.strategy
            max_attempts = self.max_attempts
            initial_delay = self.initial_delay_ms
            max_delay = self.max_delay_ms
            multiplier = self.backoff_multiplier
            jitter = self.jitter_factor
        
        if attempt >= max_attempts:
            return -1  # No more retries
        
        if strategy == RetryStrategy.NO_RETRY:
            return -1
        elif strategy == RetryStrategy.FIXED_DELAY:
            delay = initial_delay
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = initial_delay * (attempt + 1)
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = initial_delay * (multiplier ** attempt)
        elif strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = initial_delay * (multiplier ** attempt)
            jitter_amount = base_delay * jitter * (2 * time.time() % 1 - 0.5)  # Random jitter
            delay = base_delay + jitter_amount
        else:
            delay = initial_delay
        
        return min(int(delay), max_delay)


class CircuitBreaker(BaseModel):
    """Circuit breaker for agent protection."""
    
    agent_id: str = Field(..., description="Agent being protected")
    failure_threshold: int = Field(default=5, description="Failures before opening")
    recovery_timeout_ms: int = Field(default=60000, description="Time before testing recovery")
    success_threshold: int = Field(default=3, description="Successes needed to close")
    
    # State tracking
    state: CircuitBreakerState = Field(default=CircuitBreakerState.CLOSED)
    failure_count: int = Field(default=0)
    success_count: int = Field(default=0)
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metrics
    total_requests: int = Field(default=0)
    total_failures: int = Field(default=0)
    total_successes: int = Field(default=0)
    
    class Config:
        arbitrary_types_allowed = True
    
    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        now = datetime.now(timezone.utc)
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                (now - self.last_failure_time).total_seconds() * 1000 >= self.recovery_timeout_ms):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.last_state_change = now
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record a successful execution."""
        self.total_requests += 1
        self.total_successes += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.last_state_change = datetime.now(timezone.utc)
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed execution."""
        now = datetime.now(timezone.utc)
        self.total_requests += 1
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = now
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.last_state_change = now
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Return to open state on failure during testing
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            self.last_state_change = now
    
    def get_failure_rate(self) -> float:
        """Get current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests


class AgentHealthStatus(BaseModel):
    """Agent health status tracking."""
    
    agent_id: str = Field(..., description="Agent identifier")
    is_healthy: bool = Field(default=True)
    last_health_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Performance metrics
    average_response_time_ms: float = Field(default=0.0)
    success_rate: float = Field(default=1.0)
    current_load: int = Field(default=0)
    
    # Recent failures
    recent_failures: List[FailureEvent] = Field(default_factory=list)
    failure_count_24h: int = Field(default=0)
    
    # Health score (0.0 to 1.0)
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    
    class Config:
        arbitrary_types_allowed = True
    
    def update_health_score(self) -> None:
        """Update overall health score based on metrics."""
        # Base score from success rate
        score = self.success_rate
        
        # Penalize high response times
        if self.average_response_time_ms > 5000:  # 5 seconds
            score *= 0.8
        elif self.average_response_time_ms > 2000:  # 2 seconds
            score *= 0.9
        
        # Penalize high load
        if self.current_load > 10:
            score *= 0.9
        elif self.current_load > 5:
            score *= 0.95
        
        # Penalize recent failures
        if self.failure_count_24h > 10:
            score *= 0.7
        elif self.failure_count_24h > 5:
            score *= 0.85
        
        self.health_score = max(0.0, score)
        self.is_healthy = self.health_score > 0.7


class ErrorContext(BaseModel):
    """Error context for propagation and correlation."""
    
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    root_cause_id: Optional[str] = None
    error_chain: List[str] = Field(default_factory=list)
    
    # Context information
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    # Error details
    error_type: str = Field(..., description="Error type")
    error_message: str = Field(..., description="Error message")
    severity: FailureSeverity = Field(..., description="Error severity")
    
    # Timing
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class ErrorHandlingManager:
    """Comprehensive error handling and recovery manager."""

    def __init__(self, agent_registry=None, communication_framework=None):
        self._agent_registry = agent_registry
        self._communication_framework = communication_framework

        # Error tracking
        self._failure_events: Dict[str, List[FailureEvent]] = defaultdict(list)
        self._error_contexts: Dict[str, ErrorContext] = {}

        # Circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Health monitoring
        self._agent_health: Dict[str, AgentHealthStatus] = {}
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None

        # Retry policies
        self._default_retry_policy = RetryPolicy()
        self._agent_retry_policies: Dict[str, RetryPolicy] = {}

        # Recovery procedures
        self._recovery_procedures: Dict[str, Callable] = {}
        self._auto_recovery_enabled = True

        # Metrics
        self._error_metrics = {
            'total_errors': 0,
            'errors_by_type': defaultdict(int),
            'errors_by_severity': defaultdict(int),
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'circuit_breaker_activations': 0
        }

        # Event handlers
        self._error_handlers: Dict[str, List[Callable]] = defaultdict(list)

        self._logger = structlog.get_logger(__name__).bind(component="error_handling")

    async def initialize(self) -> None:
        """Initialize the error handling manager."""
        self._logger.info("Initializing error handling manager")

        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())

        # Initialize circuit breakers for known agents
        if self._agent_registry:
            agents = self._agent_registry.list_agents()
            for agent in agents:
                self._initialize_circuit_breaker(agent.agent.id)
                self._initialize_agent_health(agent.agent.id)

    async def shutdown(self) -> None:
        """Shutdown the error handling manager."""
        self._logger.info("Shutting down error handling manager")

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    # === Failure Detection ===

    async def detect_failure(self, agent_id: str, task: Optional[Task] = None,
                           exception: Optional[Exception] = None,
                           response_time_ms: Optional[float] = None,
                           context: Optional[Dict[str, Any]] = None) -> FailureEvent:
        """Detect and classify a failure."""
        failure_type = self._classify_failure(exception, response_time_ms)
        severity = self._determine_severity(failure_type, exception, response_time_ms)

        failure_event = FailureEvent(
            agent_id=agent_id,
            task_id=task.id if task else None,
            failure_type=failure_type,
            severity=severity,
            error_message=str(exception) if exception else "Performance degradation detected",
            error_details=self._extract_error_details(exception),
            stack_trace=self._extract_stack_trace(exception),
            context=context or {},
            response_time_ms=response_time_ms
        )

        # Record the failure
        await self._record_failure(failure_event)

        return failure_event

    def _classify_failure(self, exception: Optional[Exception],
                         response_time_ms: Optional[float]) -> FailureType:
        """Classify the type of failure."""
        if exception:
            if isinstance(exception, asyncio.TimeoutError):
                return FailureType.TIMEOUT
            elif isinstance(exception, ConnectionError):
                return FailureType.NETWORK_ERROR
            elif isinstance(exception, MemoryError):
                return FailureType.RESOURCE_EXHAUSTION
            elif isinstance(exception, ValueError):
                return FailureType.VALIDATION_ERROR
            elif isinstance(exception, PermissionError):
                return FailureType.AUTHORIZATION_ERROR
            else:
                return FailureType.EXCEPTION
        elif response_time_ms and response_time_ms > 10000:  # 10 seconds
            return FailureType.PERFORMANCE_DEGRADATION
        else:
            return FailureType.UNKNOWN_ERROR

    def _determine_severity(self, failure_type: FailureType,
                          exception: Optional[Exception],
                          response_time_ms: Optional[float]) -> FailureSeverity:
        """Determine the severity of the failure."""
        if failure_type in [FailureType.RESOURCE_EXHAUSTION, FailureType.NETWORK_ERROR]:
            return FailureSeverity.HIGH
        elif failure_type in [FailureType.TIMEOUT, FailureType.PERFORMANCE_DEGRADATION]:
            return FailureSeverity.MEDIUM
        elif failure_type in [FailureType.VALIDATION_ERROR, FailureType.BUSINESS_LOGIC_ERROR]:
            return FailureSeverity.LOW
        elif failure_type == FailureType.EXCEPTION:
            # Determine based on exception type
            if exception and isinstance(exception, (SystemError, RuntimeError)):
                return FailureSeverity.HIGH
            else:
                return FailureSeverity.MEDIUM
        else:
            return FailureSeverity.LOW

    def _extract_error_details(self, exception: Optional[Exception]) -> Dict[str, Any]:
        """Extract detailed error information."""
        if not exception:
            return {}

        return {
            'exception_type': type(exception).__name__,
            'exception_module': type(exception).__module__,
            'args': list(exception.args) if exception.args else []
        }

    def _extract_stack_trace(self, exception: Optional[Exception]) -> Optional[str]:
        """Extract stack trace from exception."""
        if not exception:
            return None

        import traceback
        return ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))

    async def _record_failure(self, failure_event: FailureEvent) -> None:
        """Record a failure event."""
        agent_id = failure_event.agent_id

        # Add to failure history
        self._failure_events[agent_id].append(failure_event)

        # Keep only recent failures (last 24 hours)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self._failure_events[agent_id] = [
            f for f in self._failure_events[agent_id]
            if f.timestamp > cutoff_time
        ]

        # Update metrics
        self._error_metrics['total_errors'] += 1
        self._error_metrics['errors_by_type'][failure_event.failure_type] += 1
        self._error_metrics['errors_by_severity'][failure_event.severity] += 1

        # Update circuit breaker
        circuit_breaker = self._get_circuit_breaker(agent_id)
        circuit_breaker.record_failure()

        if circuit_breaker.state == CircuitBreakerState.OPEN:
            self._error_metrics['circuit_breaker_activations'] += 1
            self._logger.warning(f"Circuit breaker opened for agent {agent_id}")

        # Update agent health
        await self._update_agent_health(agent_id, failure_event)

        # Trigger error handlers
        await self._trigger_error_handlers(failure_event)

        # Attempt recovery if enabled
        if self._auto_recovery_enabled:
            await self._attempt_recovery(failure_event)

    # === Circuit Breaker Management ===

    def _initialize_circuit_breaker(self, agent_id: str) -> None:
        """Initialize circuit breaker for an agent."""
        if agent_id not in self._circuit_breakers:
            self._circuit_breakers[agent_id] = CircuitBreaker(agent_id=agent_id)

    def _get_circuit_breaker(self, agent_id: str) -> CircuitBreaker:
        """Get circuit breaker for an agent."""
        if agent_id not in self._circuit_breakers:
            self._initialize_circuit_breaker(agent_id)
        return self._circuit_breakers[agent_id]

    async def can_execute_task(self, agent_id: str) -> bool:
        """Check if agent can execute tasks (circuit breaker check)."""
        circuit_breaker = self._get_circuit_breaker(agent_id)
        return circuit_breaker.can_execute()

    async def record_success(self, agent_id: str, response_time_ms: float) -> None:
        """Record successful task execution."""
        circuit_breaker = self._get_circuit_breaker(agent_id)
        circuit_breaker.record_success()

        # Update agent health
        await self._update_agent_health_success(agent_id, response_time_ms)

    def get_circuit_breaker_status(self, agent_id: str) -> Dict[str, Any]:
        """Get circuit breaker status for an agent."""
        circuit_breaker = self._get_circuit_breaker(agent_id)
        return {
            'state': circuit_breaker.state,
            'failure_count': circuit_breaker.failure_count,
            'success_count': circuit_breaker.success_count,
            'failure_rate': circuit_breaker.get_failure_rate(),
            'total_requests': circuit_breaker.total_requests,
            'last_state_change': circuit_breaker.last_state_change
        }

    # === Health Monitoring ===

    def _initialize_agent_health(self, agent_id: str) -> None:
        """Initialize health status for an agent."""
        if agent_id not in self._agent_health:
            self._agent_health[agent_id] = AgentHealthStatus(agent_id=agent_id)

    async def _update_agent_health(self, agent_id: str, failure_event: FailureEvent) -> None:
        """Update agent health after a failure."""
        if agent_id not in self._agent_health:
            self._initialize_agent_health(agent_id)

        health = self._agent_health[agent_id]
        health.recent_failures.append(failure_event)

        # Keep only recent failures (last 24 hours)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        health.recent_failures = [
            f for f in health.recent_failures
            if f.timestamp > cutoff_time
        ]

        health.failure_count_24h = len(health.recent_failures)
        health.last_health_check = datetime.now(timezone.utc)

        # Update success rate based on circuit breaker
        circuit_breaker = self._get_circuit_breaker(agent_id)
        health.success_rate = 1.0 - circuit_breaker.get_failure_rate()

        # Update health score
        health.update_health_score()

    async def _update_agent_health_success(self, agent_id: str, response_time_ms: float) -> None:
        """Update agent health after a successful execution."""
        if agent_id not in self._agent_health:
            self._initialize_agent_health(agent_id)

        health = self._agent_health[agent_id]

        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if health.average_response_time_ms == 0:
            health.average_response_time_ms = response_time_ms
        else:
            health.average_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * health.average_response_time_ms
            )

        health.last_health_check = datetime.now(timezone.utc)

        # Update success rate based on circuit breaker
        circuit_breaker = self._get_circuit_breaker(agent_id)
        health.success_rate = 1.0 - circuit_breaker.get_failure_rate()

        # Update health score
        health.update_health_score()

    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in health monitoring loop: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all agents."""
        for agent_id in list(self._agent_health.keys()):
            try:
                await self._check_agent_health(agent_id)
            except Exception as e:
                self._logger.error(f"Error checking health for agent {agent_id}: {e}")

    async def _check_agent_health(self, agent_id: str) -> None:
        """Check health of a specific agent."""
        if agent_id not in self._agent_health:
            return

        health = self._agent_health[agent_id]
        now = datetime.now(timezone.utc)

        # Check if agent has been inactive for too long
        time_since_check = (now - health.last_health_check).total_seconds()
        if time_since_check > 300:  # 5 minutes
            health.is_healthy = False
            health.health_score *= 0.9  # Penalize inactivity

        # Update health score
        health.update_health_score()

        # Log health status changes
        if not health.is_healthy and health.health_score < 0.5:
            self._logger.warning(f"Agent {agent_id} health degraded: score={health.health_score:.2f}")

    def get_agent_health(self, agent_id: str) -> Optional[AgentHealthStatus]:
        """Get health status for an agent."""
        return self._agent_health.get(agent_id)

    def get_all_agent_health(self) -> Dict[str, AgentHealthStatus]:
        """Get health status for all agents."""
        return self._agent_health.copy()

    # === Retry and Fallback Strategies ===

    def set_retry_policy(self, agent_id: str, policy: RetryPolicy) -> None:
        """Set retry policy for a specific agent."""
        self._agent_retry_policies[agent_id] = policy

    def get_retry_policy(self, agent_id: str) -> RetryPolicy:
        """Get retry policy for an agent."""
        return self._agent_retry_policies.get(agent_id, self._default_retry_policy)

    async def should_retry(self, agent_id: str, failure_event: FailureEvent,
                          attempt: int) -> Tuple[bool, int]:
        """Determine if a task should be retried and calculate delay."""
        policy = self.get_retry_policy(agent_id)
        delay_ms = policy.get_delay_ms(attempt, failure_event.failure_type)

        if delay_ms < 0:
            return False, 0

        # Don't retry if circuit breaker is open
        if not await self.can_execute_task(agent_id):
            return False, 0

        # Don't retry certain failure types
        if failure_event.failure_type in [
            FailureType.VALIDATION_ERROR,
            FailureType.AUTHORIZATION_ERROR,
            FailureType.BUSINESS_LOGIC_ERROR
        ]:
            return False, 0

        return True, delay_ms

    async def find_fallback_agent(self, original_agent_id: str,
                                 required_capability: AgentCapability) -> Optional[str]:
        """Find a fallback agent with the required capability."""
        if not self._agent_registry:
            return None

        # Get all agents with the required capability
        agents = await self._agent_registry.find_agents_by_capability(required_capability)

        # Filter out the original agent and unhealthy agents
        fallback_candidates = []
        for agent in agents:
            if agent.id == original_agent_id:
                continue

            # Check if agent is healthy
            health = self.get_agent_health(agent.id)
            if health and health.is_healthy and health.health_score > 0.7:
                # Check circuit breaker
                if await self.can_execute_task(agent.id):
                    fallback_candidates.append((agent, health.health_score))

        if not fallback_candidates:
            return None

        # Sort by health score (best first)
        fallback_candidates.sort(key=lambda x: x[1], reverse=True)
        return fallback_candidates[0][0].id

    # === Error Propagation and Context ===

    def create_error_context(self, error_type: str, error_message: str,
                           severity: FailureSeverity,
                           workflow_id: Optional[str] = None,
                           task_id: Optional[str] = None,
                           agent_id: Optional[str] = None,
                           user_id: Optional[str] = None,
                           tenant_id: Optional[str] = None,
                           root_cause_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Create error context for propagation."""
        context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            workflow_id=workflow_id,
            task_id=task_id,
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            root_cause_id=root_cause_id,
            metadata=metadata or {}
        )

        self._error_contexts[context.correlation_id] = context
        return context

    def propagate_error(self, parent_context: ErrorContext,
                       new_error_type: str, new_error_message: str,
                       agent_id: Optional[str] = None) -> ErrorContext:
        """Propagate error to create a new context in the error chain."""
        new_context = ErrorContext(
            error_type=new_error_type,
            error_message=new_error_message,
            severity=parent_context.severity,
            workflow_id=parent_context.workflow_id,
            task_id=parent_context.task_id,
            agent_id=agent_id or parent_context.agent_id,
            user_id=parent_context.user_id,
            tenant_id=parent_context.tenant_id,
            root_cause_id=parent_context.root_cause_id or parent_context.correlation_id,
            metadata=parent_context.metadata.copy()
        )

        # Add to error chain
        new_context.error_chain = parent_context.error_chain.copy()
        new_context.error_chain.append(parent_context.correlation_id)

        self._error_contexts[new_context.correlation_id] = new_context
        return new_context

    def get_error_context(self, correlation_id: str) -> Optional[ErrorContext]:
        """Get error context by correlation ID."""
        return self._error_contexts.get(correlation_id)

    # === Recovery Procedures ===

    def register_recovery_procedure(self, failure_type: str,
                                  procedure: Callable[[FailureEvent], None]) -> None:
        """Register a recovery procedure for a specific failure type."""
        self._recovery_procedures[failure_type] = procedure

    async def _attempt_recovery(self, failure_event: FailureEvent) -> bool:
        """Attempt to recover from a failure."""
        self._error_metrics['recovery_attempts'] += 1

        try:
            # Check for specific recovery procedure
            failure_type_str = failure_event.failure_type.value
            if failure_type_str in self._recovery_procedures:
                procedure = self._recovery_procedures[failure_type_str]
                await self._execute_recovery_procedure(procedure, failure_event)
                self._error_metrics['successful_recoveries'] += 1
                return True

            # Default recovery strategies
            if failure_event.failure_type == FailureType.RESOURCE_EXHAUSTION:
                await self._recover_from_resource_exhaustion(failure_event)
            elif failure_event.failure_type == FailureType.NETWORK_ERROR:
                await self._recover_from_network_error(failure_event)
            elif failure_event.failure_type == FailureType.TIMEOUT:
                await self._recover_from_timeout(failure_event)
            else:
                # Generic recovery
                await self._generic_recovery(failure_event)

            self._error_metrics['successful_recoveries'] += 1
            return True

        except Exception as e:
            self._logger.error(f"Recovery attempt failed for {failure_event.id}: {e}")
            return False

    async def _execute_recovery_procedure(self, procedure: Callable,
                                        failure_event: FailureEvent) -> None:
        """Execute a recovery procedure."""
        if asyncio.iscoroutinefunction(procedure):
            await procedure(failure_event)
        else:
            procedure(failure_event)

    async def _recover_from_resource_exhaustion(self, failure_event: FailureEvent) -> None:
        """Recover from resource exhaustion."""
        self._logger.info(f"Attempting recovery from resource exhaustion for agent {failure_event.agent_id}")

        # Trigger garbage collection
        import gc
        gc.collect()

        # Reduce agent load if possible
        if self._agent_registry:
            await self._agent_registry.reduce_agent_load(failure_event.agent_id)

    async def _recover_from_network_error(self, failure_event: FailureEvent) -> None:
        """Recover from network error."""
        self._logger.info(f"Attempting recovery from network error for agent {failure_event.agent_id}")

        # Wait a bit for network to recover
        await asyncio.sleep(5)

        # Test connectivity if possible
        # This would be implementation-specific

    async def _recover_from_timeout(self, failure_event: FailureEvent) -> None:
        """Recover from timeout."""
        self._logger.info(f"Attempting recovery from timeout for agent {failure_event.agent_id}")

        # Increase timeout for this agent temporarily
        # This would be implementation-specific

    async def _generic_recovery(self, failure_event: FailureEvent) -> None:
        """Generic recovery procedure."""
        self._logger.info(f"Attempting generic recovery for agent {failure_event.agent_id}")

        # Wait a bit before allowing new requests
        await asyncio.sleep(2)

    # === Event Handlers ===

    def add_error_handler(self, error_type: str, handler: Callable[[FailureEvent], None]) -> None:
        """Add an error event handler."""
        self._error_handlers[error_type].append(handler)

    async def _trigger_error_handlers(self, failure_event: FailureEvent) -> None:
        """Trigger error handlers for a failure event."""
        error_type = failure_event.failure_type.value

        # Trigger specific handlers
        for handler in self._error_handlers.get(error_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(failure_event)
                else:
                    handler(failure_event)
            except Exception as e:
                self._logger.error(f"Error handler failed: {e}")

        # Trigger global handlers
        for handler in self._error_handlers.get('*', []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(failure_event)
                else:
                    handler(failure_event)
            except Exception as e:
                self._logger.error(f"Global error handler failed: {e}")

    # === Metrics and Reporting ===

    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error handling metrics."""
        return {
            'total_errors': self._error_metrics['total_errors'],
            'errors_by_type': dict(self._error_metrics['errors_by_type']),
            'errors_by_severity': dict(self._error_metrics['errors_by_severity']),
            'recovery_attempts': self._error_metrics['recovery_attempts'],
            'successful_recoveries': self._error_metrics['successful_recoveries'],
            'recovery_success_rate': (
                self._error_metrics['successful_recoveries'] /
                max(1, self._error_metrics['recovery_attempts'])
            ),
            'circuit_breaker_activations': self._error_metrics['circuit_breaker_activations'],
            'active_circuit_breakers': len([
                cb for cb in self._circuit_breakers.values()
                if cb.state == CircuitBreakerState.OPEN
            ])
        }

    def get_failure_history(self, agent_id: str,
                          hours: int = 24) -> List[FailureEvent]:
        """Get failure history for an agent."""
        if agent_id not in self._failure_events:
            return []

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            f for f in self._failure_events[agent_id]
            if f.timestamp > cutoff_time
        ]


# Global error handling manager instance
_error_handling_manager: Optional[ErrorHandlingManager] = None


def get_error_handling_manager() -> ErrorHandlingManager:
    """Get the global error handling manager instance."""
    global _error_handling_manager
    if _error_handling_manager is None:
        _error_handling_manager = ErrorHandlingManager()
    return _error_handling_manager


def set_error_handling_manager(manager: ErrorHandlingManager) -> None:
    """Set the global error handling manager instance."""
    global _error_handling_manager
    _error_handling_manager = manager
