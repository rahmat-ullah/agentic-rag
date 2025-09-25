"""
Base classes and interfaces for the agent orchestration framework.

This module defines the foundational classes and interfaces that all agents
and orchestration components must implement.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class AgentCapability(str, Enum):
    """Enumeration of agent capabilities."""

    # Core capabilities
    QUERY_ANALYSIS = "query_analysis"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    ANSWER_SYNTHESIS = "answer_synthesis"
    CONTENT_REDACTION = "content_redaction"
    PRICING_ANALYSIS = "pricing_analysis"

    # Answer synthesis capabilities
    CITATION_GENERATION = "citation_generation"
    QUALITY_ASSESSMENT = "quality_assessment"

    # Pricing analysis capabilities
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    COST_MODELING = "cost_modeling"

    # Advanced analysis capabilities
    DOCUMENT_COMPARISON = "document_comparison"
    CONTENT_SUMMARIZATION = "content_summarization"
    TABLE_EXTRACTION = "table_extraction"
    COMPLIANCE_CHECKING = "compliance_checking"
    RISK_ASSESSMENT = "risk_assessment"

    # Privacy and security capabilities
    PII_DETECTION = "pii_detection"
    PRIVACY_PROTECTION = "privacy_protection"

    # Search and retrieval capabilities
    DOCUMENT_SEARCH = "document_search"
    SEMANTIC_SEARCH = "semantic_search"
    CONTEXTUAL_RETRIEVAL = "contextual_retrieval"

    # Meta capabilities
    WORKFLOW_PLANNING = "workflow_planning"
    TOOL_SELECTION = "tool_selection"
    RESULT_AGGREGATION = "result_aggregation"
    ERROR_RECOVERY = "error_recovery"


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
    SHUTTING_DOWN = "shutting_down"


class TaskStatus(str, Enum):
    """Task execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Task priority levels."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Task(BaseModel):
    """Represents a task to be executed by an agent."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    capability_required: AgentCapability = Field(..., description="Required capability")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    
    # Task data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution tracking
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Task IDs this task depends on")
    timeout_seconds: Optional[int] = Field(default=300, description="Task timeout in seconds")
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    class Config:
        use_enum_values = True


class Context(BaseModel):
    """Execution context shared between agents."""
    
    # Request context
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: UUID = Field(..., description="User identifier")
    tenant_id: UUID = Field(..., description="Tenant identifier")
    
    # Query context
    original_query: str = Field(..., description="Original user query")
    processed_query: Optional[str] = None
    query_intent: Optional[str] = None
    
    # Execution context
    workflow_id: Optional[str] = None
    session_data: Dict[str, Any] = Field(default_factory=dict)
    shared_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing and performance
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_at: Optional[datetime] = None
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class Result(BaseModel):
    """Result from agent task execution."""
    
    task_id: str = Field(..., description="Task identifier")
    agent_id: str = Field(..., description="Agent identifier")
    
    # Result data
    success: bool = Field(..., description="Whether task completed successfully")
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Error information
    error: Optional[str] = None
    error_code: Optional[str] = None
    
    # Timing
    execution_time_ms: Optional[int] = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


# Exception classes
class AgentError(Exception):
    """Base exception for agent-related errors."""
    
    def __init__(self, message: str, agent_id: str = None, error_code: str = None):
        super().__init__(message)
        self.agent_id = agent_id
        self.error_code = error_code


class AgentTimeoutError(AgentError):
    """Exception raised when agent execution times out."""
    pass


class AgentUnavailableError(AgentError):
    """Exception raised when agent is unavailable."""
    pass


class WorkflowError(Exception):
    """Exception raised for workflow-related errors."""
    pass


class Agent(ABC):
    """Base class for all agents in the orchestration framework."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        capabilities: List[AgentCapability],
        description: str = "",
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = set(capabilities)
        self.description = description
        self.config = config or {}
        
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = self.created_at
        
        # Performance tracking
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time_ms = 0
        
        self._logger = structlog.get_logger(__name__).bind(agent_id=agent_id, agent_name=name)
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def execute(self, task: Task, context: Context) -> Result:
        """Execute a task. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the agent. Must be implemented by subclasses."""
        pass
    
    def can_handle(self, task: Task) -> bool:
        """Check if this agent can handle the given task."""
        return task.capability_required in self.capabilities
    
    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        return self.status
    
    def get_capabilities(self) -> Set[AgentCapability]:
        """Get agent capabilities."""
        return self.capabilities.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        total_tasks = self.tasks_completed + self.tasks_failed
        success_rate = self.tasks_completed / total_tasks if total_tasks > 0 else 0.0
        avg_execution_time = (
            self.total_execution_time_ms / self.tasks_completed 
            if self.tasks_completed > 0 else 0.0
        )
        
        return {
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": success_rate,
            "average_execution_time_ms": avg_execution_time,
            "total_execution_time_ms": self.total_execution_time_ms,
            "uptime_seconds": (datetime.now(timezone.utc) - self.created_at).total_seconds(),
            "last_activity": self.last_activity.isoformat(),
        }
    
    async def health_check(self) -> bool:
        """Perform health check. Override in subclasses for custom checks."""
        return self.status in [AgentStatus.READY, AgentStatus.BUSY]
    
    def _update_performance_metrics(self, execution_time_ms: int, success: bool):
        """Update performance metrics after task execution."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        self.total_execution_time_ms += execution_time_ms
        self.last_activity = datetime.now(timezone.utc)
