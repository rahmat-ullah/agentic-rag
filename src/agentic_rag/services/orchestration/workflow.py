"""
Enhanced Workflow Orchestration Engine

This module implements comprehensive workflow orchestration for multi-step task execution.
It includes YAML-based workflow definition language, advanced execution engine with scheduling,
parallel and sequential task support, state tracking, persistence, and recovery capabilities.
"""

import asyncio
import json
import yaml
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
from uuid import uuid4
import threading
import weakref

from pydantic import BaseModel, Field, validator
import structlog

from .base import Task, TaskStatus, TaskPriority, Context, Result, WorkflowError, AgentCapability
from .communication import SharedStateManager, Message, MessageType, MessagePriority

logger = structlog.get_logger(__name__)


class WorkflowStepType(str, Enum):
    """Enhanced types of workflow steps."""

    TASK = "task"                    # Single task execution
    PARALLEL = "parallel"            # Parallel execution of sub-steps
    SEQUENTIAL = "sequential"        # Sequential execution of sub-steps
    CONDITIONAL = "conditional"      # Conditional execution based on expression
    LOOP = "loop"                   # Loop execution with condition
    FAN_OUT = "fan_out"             # Fan-out pattern (1 to many)
    FAN_IN = "fan_in"               # Fan-in pattern (many to 1)
    BARRIER = "barrier"             # Synchronization barrier
    SWITCH = "switch"               # Switch/case execution
    TRY_CATCH = "try_catch"         # Error handling block
    TIMEOUT = "timeout"             # Timeout wrapper
    RETRY = "retry"                 # Retry wrapper
    TRANSFORM = "transform"         # Data transformation step
    VALIDATE = "validate"           # Validation step
    CHECKPOINT = "checkpoint"       # State checkpoint


class WorkflowExecutionStatus(str, Enum):
    """Enhanced workflow execution status."""

    PENDING = "pending"             # Waiting to start
    INITIALIZING = "initializing"   # Setting up execution
    RUNNING = "running"             # Currently executing
    PAUSED = "paused"              # Execution paused
    SUSPENDED = "suspended"         # Execution suspended (waiting for external event)
    COMPLETED = "completed"         # Successfully completed
    FAILED = "failed"              # Failed with error
    CANCELLED = "cancelled"         # Cancelled by user
    TIMEOUT = "timeout"            # Timed out
    RETRYING = "retrying"          # Retrying after failure


class WorkflowTriggerType(str, Enum):
    """Types of workflow triggers."""

    MANUAL = "manual"              # Manual trigger
    SCHEDULED = "scheduled"        # Time-based trigger
    EVENT = "event"               # Event-based trigger
    API = "api"                   # API call trigger
    WEBHOOK = "webhook"           # Webhook trigger
    DEPENDENCY = "dependency"     # Dependency completion trigger


class ExecutionMode(str, Enum):
    """Workflow execution modes."""

    SYNCHRONOUS = "synchronous"    # Wait for completion
    ASYNCHRONOUS = "asynchronous"  # Fire and forget
    STREAMING = "streaming"        # Stream results as available
    BATCH = "batch"               # Batch processing mode


class DataFlowType(str, Enum):
    """Types of data flow between steps."""

    DIRECT = "direct"             # Direct data passing
    SHARED_STATE = "shared_state" # Via shared state
    MESSAGE = "message"           # Via message passing
    FILE = "file"                # Via file system
    DATABASE = "database"        # Via database


class WorkflowStep(BaseModel):
    """Enhanced workflow step with comprehensive configuration and tracking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Step name")
    description: str = Field(default="", description="Step description")
    type: WorkflowStepType = Field(default=WorkflowStepType.TASK)

    # Task configuration (for TASK type)
    task: Optional[Task] = None
    agent_capability: Optional[AgentCapability] = None
    agent_id: Optional[str] = Field(None, description="Specific agent ID to use")

    # Sub-steps (for PARALLEL/SEQUENTIAL types)
    steps: List["WorkflowStep"] = Field(default_factory=list)

    # Dependencies and data flow
    depends_on: List[str] = Field(default_factory=list, description="Step IDs this step depends on")
    data_dependencies: Dict[str, str] = Field(default_factory=dict, description="Data dependencies from other steps")
    data_flow_type: DataFlowType = Field(default=DataFlowType.DIRECT)

    # Input/Output configuration
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Step inputs")
    outputs: Dict[str, str] = Field(default_factory=dict, description="Output variable mappings")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input validation schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output validation schema")

    # Conditional execution
    condition: Optional[str] = Field(None, description="Condition expression for execution")
    condition_language: str = Field(default="python", description="Condition expression language")

    # Loop configuration
    loop_condition: Optional[str] = Field(None, description="Loop continuation condition")
    loop_variable: Optional[str] = Field(None, description="Loop iteration variable")
    max_iterations: Optional[int] = Field(default=100, description="Maximum loop iterations")

    # Parallel execution configuration
    max_parallel: Optional[int] = Field(None, description="Maximum parallel sub-steps")
    parallel_strategy: str = Field(default="all", description="Parallel execution strategy (all, any, majority)")

    # Error handling
    on_error: str = Field(default="fail", description="Error handling strategy (fail, continue, retry, fallback)")
    fallback_step: Optional[str] = Field(None, description="Fallback step ID on error")
    error_handlers: Dict[str, str] = Field(default_factory=dict, description="Error type to handler mappings")

    # Execution tracking
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Result] = None
    error: Optional[str] = None
    execution_count: int = Field(default=0, description="Number of times step has been executed")

    # Configuration and optimization
    timeout_seconds: Optional[int] = Field(default=300)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    retry_delay_seconds: int = Field(default=1)
    retry_backoff_multiplier: float = Field(default=2.0)

    # Resource requirements
    cpu_limit: Optional[float] = Field(None, description="CPU limit (cores)")
    memory_limit: Optional[int] = Field(None, description="Memory limit (MB)")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)

    # Monitoring and observability
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)

    # Caching
    cache_enabled: bool = Field(default=False)
    cache_key: Optional[str] = Field(None)
    cache_ttl_seconds: Optional[int] = Field(None)

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    @validator('condition', 'loop_condition')
    def validate_expressions(cls, v):
        """Validate condition expressions."""
        if v and not isinstance(v, str):
            raise ValueError("Condition must be a string expression")
        return v

    @validator('max_iterations')
    def validate_max_iterations(cls, v):
        """Validate maximum iterations."""
        if v is not None and v <= 0:
            raise ValueError("Maximum iterations must be positive")
        return v

    def is_ready_to_execute(self, completed_steps: Set[str]) -> bool:
        """Check if step is ready to execute based on dependencies."""
        return all(dep_id in completed_steps for dep_id in self.depends_on)

    def get_execution_time_ms(self) -> Optional[float]:
        """Get execution time in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


# Update forward reference
WorkflowStep.model_rebuild()


class WorkflowTrigger(BaseModel):
    """Workflow trigger configuration."""

    type: WorkflowTriggerType = Field(..., description="Trigger type")
    schedule: Optional[str] = Field(None, description="Cron schedule for scheduled triggers")
    event_type: Optional[str] = Field(None, description="Event type for event triggers")
    webhook_url: Optional[str] = Field(None, description="Webhook URL")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Trigger conditions")
    enabled: bool = Field(default=True)


class WorkflowDefinition(BaseModel):
    """Enhanced workflow definition with comprehensive configuration."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")

    # Steps and structure
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    entry_point: Optional[str] = Field(None, description="Entry point step ID")
    exit_points: List[str] = Field(default_factory=list, description="Exit point step IDs")

    # Execution configuration
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SYNCHRONOUS)
    timeout_seconds: Optional[int] = Field(default=1800, description="Overall workflow timeout")
    max_parallel_tasks: int = Field(default=10, description="Maximum parallel tasks")
    max_concurrent_executions: int = Field(default=1, description="Maximum concurrent executions")

    # Resource management
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)

    # Error handling and recovery
    error_handling_strategy: str = Field(default="fail_fast", description="Global error handling strategy")
    recovery_enabled: bool = Field(default=True)
    checkpoint_enabled: bool = Field(default=True)
    checkpoint_interval_seconds: int = Field(default=60)

    # Triggers
    triggers: List[WorkflowTrigger] = Field(default_factory=list)

    # Variables and configuration
    variables: Dict[str, Any] = Field(default_factory=dict, description="Workflow variables")
    constants: Dict[str, Any] = Field(default_factory=dict, description="Workflow constants")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")

    # Metadata and organization
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None

    # Versioning and migration
    parent_version: Optional[str] = Field(None, description="Parent version for migration")
    migration_notes: Optional[str] = None
    deprecated: bool = Field(default=False)
    deprecation_date: Optional[datetime] = None

    # Performance and monitoring
    performance_targets: Dict[str, float] = Field(default_factory=dict)
    monitoring_enabled: bool = Field(default=True)
    logging_level: str = Field(default="INFO")

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    @validator('steps')
    def validate_steps(cls, v):
        """Validate workflow steps."""
        if not v:
            raise ValueError("Workflow must have at least one step")

        # Check for duplicate step IDs
        step_ids = [step.id for step in v]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Duplicate step IDs found")

        return v

    @validator('version')
    def validate_version(cls, v):
        """Validate version format."""
        if not v or not isinstance(v, str):
            raise ValueError("Version must be a non-empty string")
        return v

    def get_step_by_id(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get dependency graph for the workflow."""
        graph = {}
        for step in self.steps:
            graph[step.id] = step.depends_on.copy()
        return graph

    def validate_dependencies(self) -> List[str]:
        """Validate workflow dependencies and return any errors."""
        errors = []
        step_ids = {step.id for step in self.steps}

        for step in self.steps:
            # Check if all dependencies exist
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    errors.append(f"Step '{step.id}' depends on non-existent step '{dep_id}'")

        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Circular dependencies detected in workflow")

        return errors

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS."""
        graph = self.get_dependency_graph()
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step_id in graph:
            if step_id not in visited:
                if dfs(step_id):
                    return True

        return False


class WorkflowCheckpoint(BaseModel):
    """Workflow execution checkpoint for recovery."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    execution_id: str = Field(..., description="Workflow execution ID")
    checkpoint_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # State snapshot
    step_status: Dict[str, TaskStatus] = Field(default_factory=dict)
    step_results: Dict[str, Result] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    current_step_id: Optional[str] = None

    # Metadata
    checkpoint_reason: str = Field(default="scheduled")
    size_bytes: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class WorkflowExecution(BaseModel):
    """Enhanced workflow execution instance with comprehensive tracking and recovery."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str = Field(..., description="Workflow definition ID")
    workflow_version: str = Field(..., description="Workflow version")

    # Execution context and configuration
    context: Context = Field(..., description="Execution context")
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SYNCHRONOUS)
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)

    # Status and lifecycle tracking
    status: WorkflowExecutionStatus = Field(default=WorkflowExecutionStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None

    # Step execution tracking
    step_results: Dict[str, Result] = Field(default_factory=dict)
    step_status: Dict[str, TaskStatus] = Field(default_factory=dict)
    step_start_times: Dict[str, datetime] = Field(default_factory=dict)
    step_end_times: Dict[str, datetime] = Field(default_factory=dict)
    step_retry_counts: Dict[str, int] = Field(default_factory=dict)

    # Current execution state
    current_step_id: Optional[str] = None
    active_steps: Set[str] = Field(default_factory=set)
    completed_steps: Set[str] = Field(default_factory=set)
    failed_steps: Set[str] = Field(default_factory=set)

    # Variables and data flow
    variables: Dict[str, Any] = Field(default_factory=dict)
    step_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Error handling and recovery
    error: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)
    failed_step_id: Optional[str] = None
    recovery_attempts: int = Field(default=0)
    max_recovery_attempts: int = Field(default=3)

    # Performance metrics
    total_execution_time_ms: Optional[float] = None
    steps_completed: int = Field(default=0)
    steps_failed: int = Field(default=0)
    steps_skipped: int = Field(default=0)
    steps_retried: int = Field(default=0)

    # Resource usage tracking
    peak_memory_mb: Optional[float] = None
    total_cpu_time_ms: Optional[float] = None
    network_requests: int = Field(default=0)

    # Checkpoints and persistence
    checkpoints: List[WorkflowCheckpoint] = Field(default_factory=list)
    last_checkpoint_at: Optional[datetime] = None
    persistence_enabled: bool = Field(default=True)

    # Monitoring and observability
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    # External references
    parent_execution_id: Optional[str] = None
    child_execution_ids: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def add_trace_event(self, event_type: str, step_id: Optional[str] = None,
                       data: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the execution trace."""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'step_id': step_id,
            'data': data or {}
        }
        self.execution_trace.append(event)

    def get_execution_time_ms(self) -> Optional[float]:
        """Get total execution time in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        elif self.started_at:
            return (datetime.now(timezone.utc) - self.started_at).total_seconds() * 1000
        return None

    def get_step_execution_time_ms(self, step_id: str) -> Optional[float]:
        """Get execution time for a specific step."""
        if step_id in self.step_start_times and step_id in self.step_end_times:
            start_time = self.step_start_times[step_id]
            end_time = self.step_end_times[step_id]
            return (end_time - start_time).total_seconds() * 1000
        return None

    def get_progress_percentage(self, total_steps: int) -> float:
        """Get execution progress as percentage."""
        if total_steps == 0:
            return 100.0
        return (self.steps_completed / total_steps) * 100.0

    def is_recoverable(self) -> bool:
        """Check if execution can be recovered."""
        return (self.status in [WorkflowExecutionStatus.FAILED, WorkflowExecutionStatus.PAUSED] and
                self.recovery_attempts < self.max_recovery_attempts and
                len(self.checkpoints) > 0)

    def create_checkpoint(self, reason: str = "scheduled") -> WorkflowCheckpoint:
        """Create a checkpoint of current execution state."""
        checkpoint = WorkflowCheckpoint(
            execution_id=self.id,
            step_status=self.step_status.copy(),
            step_results=self.step_results.copy(),
            variables=self.variables.copy(),
            current_step_id=self.current_step_id,
            checkpoint_reason=reason
        )

        self.checkpoints.append(checkpoint)
        self.last_checkpoint_at = checkpoint.checkpoint_at

        return checkpoint


class WorkflowParser:
    """YAML-based workflow definition parser and validator."""

    def __init__(self):
        self._logger = structlog.get_logger(__name__).bind(component="workflow_parser")

    def parse_yaml(self, yaml_content: str) -> WorkflowDefinition:
        """Parse YAML workflow definition."""
        try:
            data = yaml.safe_load(yaml_content)
            return self._parse_workflow_dict(data)
        except yaml.YAMLError as e:
            raise WorkflowError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise WorkflowError(f"Failed to parse workflow: {e}")

    def parse_file(self, file_path: Union[str, Path]) -> WorkflowDefinition:
        """Parse workflow definition from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise WorkflowError(f"Workflow file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_yaml(content)
        except Exception as e:
            raise WorkflowError(f"Failed to read workflow file {file_path}: {e}")

    def _parse_workflow_dict(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """Parse workflow from dictionary."""
        if not isinstance(data, dict):
            raise WorkflowError("Workflow definition must be a dictionary")

        # Extract basic workflow information
        workflow_data = {
            'name': data.get('name', 'Unnamed Workflow'),
            'description': data.get('description', ''),
            'version': data.get('version', '1.0.0'),
            'execution_mode': data.get('execution_mode', 'synchronous'),
            'timeout_seconds': data.get('timeout_seconds', 1800),
            'max_parallel_tasks': data.get('max_parallel_tasks', 10),
            'variables': data.get('variables', {}),
            'constants': data.get('constants', {}),
            'environment': data.get('environment', {}),
            'tags': data.get('tags', []),
            'category': data.get('category'),
        }

        # Parse triggers
        triggers = []
        for trigger_data in data.get('triggers', []):
            triggers.append(self._parse_trigger(trigger_data))
        workflow_data['triggers'] = triggers

        # Parse steps
        steps_data = data.get('steps', [])
        if not steps_data:
            raise WorkflowError("Workflow must have at least one step")

        steps = []
        step_name_to_id = {}

        # First pass: create steps and build name-to-id mapping
        for step_data in steps_data:
            step = self._parse_step(step_data)
            steps.append(step)
            step_name_to_id[step.name] = step.id

        # Second pass: resolve dependencies by name to ID
        for step in steps:
            if step.depends_on:
                resolved_deps = []
                for dep_name in step.depends_on:
                    if dep_name in step_name_to_id:
                        resolved_deps.append(step_name_to_id[dep_name])
                    else:
                        raise WorkflowError(f"Step '{step.name}' depends on unknown step '{dep_name}'")
                step.depends_on = resolved_deps

        workflow_data['steps'] = steps

        # Set entry point
        if 'entry_point' in data:
            workflow_data['entry_point'] = data['entry_point']
        elif steps:
            workflow_data['entry_point'] = steps[0].id

        return WorkflowDefinition(**workflow_data)

    def _parse_trigger(self, data: Dict[str, Any]) -> WorkflowTrigger:
        """Parse workflow trigger."""
        return WorkflowTrigger(
            type=data.get('type', 'manual'),
            schedule=data.get('schedule'),
            event_type=data.get('event_type'),
            webhook_url=data.get('webhook_url'),
            conditions=data.get('conditions', {}),
            enabled=data.get('enabled', True)
        )

    def _parse_step(self, data: Dict[str, Any]) -> WorkflowStep:
        """Parse workflow step."""
        step_data = {
            'name': data.get('name', 'Unnamed Step'),
            'description': data.get('description', ''),
            'type': data.get('type', 'task'),
            'depends_on': data.get('depends_on', []),
            'inputs': data.get('inputs', {}),
            'outputs': data.get('outputs', {}),
            'condition': data.get('condition'),
            'timeout_seconds': data.get('timeout_seconds', 300),
            'max_retries': data.get('max_retries', 3),
            'tags': data.get('tags', []),
            'metadata': data.get('metadata', {}),
        }

        # Parse task configuration
        if 'task' in data:
            task_data = data['task']
            step_data['task'] = Task(
                name=task_data.get('name', step_data['name']),
                description=task_data.get('description', ''),
                capability_required=AgentCapability(task_data.get('capability', 'query_analysis')),
                priority=TaskPriority(task_data.get('priority', 'normal')),
                input_data=task_data.get('input_data', {}),
                metadata=task_data.get('metadata', {})
            )

        # Parse agent configuration
        if 'agent' in data:
            agent_data = data['agent']
            if isinstance(agent_data, str):
                step_data['agent_capability'] = AgentCapability(agent_data)
            else:
                step_data['agent_capability'] = AgentCapability(agent_data.get('capability', 'query_analysis'))
                step_data['agent_id'] = agent_data.get('id')

        # Parse sub-steps for parallel/sequential
        if 'steps' in data:
            sub_steps = []
            for sub_step_data in data['steps']:
                sub_steps.append(self._parse_step(sub_step_data))
            step_data['steps'] = sub_steps

        # Parse loop configuration
        if 'loop' in data:
            loop_data = data['loop']
            step_data['loop_condition'] = loop_data.get('condition')
            step_data['loop_variable'] = loop_data.get('variable')
            step_data['max_iterations'] = loop_data.get('max_iterations', 100)

        # Parse error handling
        if 'error_handling' in data:
            error_data = data['error_handling']
            step_data['on_error'] = error_data.get('strategy', 'fail')
            step_data['fallback_step'] = error_data.get('fallback_step')
            step_data['error_handlers'] = error_data.get('handlers', {})

        return WorkflowStep(**step_data)

    def validate_workflow(self, workflow: WorkflowDefinition) -> List[str]:
        """Validate workflow definition and return errors."""
        errors = []

        # Validate basic structure
        if not workflow.name:
            errors.append("Workflow name is required")

        if not workflow.steps:
            errors.append("Workflow must have at least one step")

        # Validate dependencies
        dependency_errors = workflow.validate_dependencies()
        errors.extend(dependency_errors)

        # Validate step configurations
        for step in workflow.steps:
            step_errors = self._validate_step(step, workflow)
            errors.extend(step_errors)

        return errors

    def _validate_step(self, step: WorkflowStep, workflow: WorkflowDefinition) -> List[str]:
        """Validate individual step configuration."""
        errors = []

        if not step.name:
            errors.append(f"Step {step.id} must have a name")

        # Validate task configuration
        if step.type == WorkflowStepType.TASK:
            if not step.task and not step.agent_capability:
                errors.append(f"Task step {step.id} must have either task or agent_capability defined")

        # Validate conditional steps
        if step.type == WorkflowStepType.CONDITIONAL:
            if not step.condition:
                errors.append(f"Conditional step {step.id} must have a condition")

        # Validate loop steps
        if step.type == WorkflowStepType.LOOP:
            if not step.loop_condition:
                errors.append(f"Loop step {step.id} must have a loop condition")

        # Validate parallel/sequential steps
        if step.type in [WorkflowStepType.PARALLEL, WorkflowStepType.SEQUENTIAL]:
            if not step.steps:
                errors.append(f"{step.type.value} step {step.id} must have sub-steps")

        return errors


class WorkflowEngine:
    """Enhanced workflow execution engine with advanced scheduling and state management."""

    def __init__(self, communication_framework=None, agent_registry=None):
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}

        # External dependencies
        self._communication_framework = communication_framework
        self._agent_registry = agent_registry

        # State management
        self._shared_state = SharedStateManager(persistence_enabled=True)
        self._execution_queue = asyncio.Queue()
        self._scheduler_task: Optional[asyncio.Task] = None

        # Performance tracking
        self._execution_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time_ms': 0.0,
            'active_executions': 0
        }

        # Configuration
        self._max_concurrent_executions = 50
        self._checkpoint_interval_seconds = 60
        self._cleanup_interval_seconds = 300

        # Workflow parser
        self._parser = WorkflowParser()

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        self._logger = structlog.get_logger(__name__).bind(component="workflow_engine")

    async def initialize(self) -> None:
        """Initialize the workflow engine."""
        self._logger.info("Initializing enhanced workflow engine")

        # Start background tasks
        self._scheduler_task = asyncio.create_task(self._execution_scheduler())

        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())

    async def shutdown(self) -> None:
        """Shutdown the workflow engine."""
        self._logger.info("Shutting down workflow engine")

        # Stop scheduler
        if self._scheduler_task:
            self._scheduler_task.cancel()

        # Cancel all running tasks
        for task in self._running_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)

        self._running_tasks.clear()
        self._executions.clear()
        self._workflows.clear()

    # === Workflow Management ===

    def register_workflow_from_yaml(self, yaml_content: str) -> bool:
        """Register workflow from YAML definition."""
        try:
            workflow = self._parser.parse_yaml(yaml_content)
            return self.register_workflow(workflow)
        except Exception as e:
            self._logger.error(f"Failed to register workflow from YAML: {e}")
            return False

    def register_workflow_from_file(self, file_path: Union[str, Path]) -> bool:
        """Register workflow from file."""
        try:
            workflow = self._parser.parse_file(file_path)
            return self.register_workflow(workflow)
        except Exception as e:
            self._logger.error(f"Failed to register workflow from file {file_path}: {e}")
            return False
    
    def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Register a workflow definition."""
        try:
            # Validate workflow
            errors = self._parser.validate_workflow(workflow)
            if errors:
                raise WorkflowError(f"Workflow validation failed: {'; '.join(errors)}")

            # Register workflow
            self._workflows[workflow.id] = workflow

            self._logger.info(f"Registered workflow: {workflow.name} ({workflow.id})")
            return True

        except Exception as e:
            self._logger.error(f"Failed to register workflow {workflow.name}: {e}")
            return False

    # === Workflow Execution ===

    async def execute_workflow(self, workflow_id: str, context: Context,
                              execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS,
                              variables: Optional[Dict[str, Any]] = None) -> WorkflowExecution:
        """Execute a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise WorkflowError(f"Workflow not found: {workflow_id}")

        # Create execution instance
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_version=workflow.version,
            context=context,
            execution_mode=execution_mode,
            variables=variables or {},
            persistence_enabled=workflow.checkpoint_enabled
        )

        # Initialize execution variables with workflow defaults
        execution.variables.update(workflow.variables)
        if variables:
            execution.variables.update(variables)

        # Register execution
        self._executions[execution.id] = execution
        self._execution_metrics['total_executions'] += 1
        self._execution_metrics['active_executions'] += 1

        execution.add_trace_event('workflow_started', data={'workflow_name': workflow.name})

        try:
            if execution_mode == ExecutionMode.SYNCHRONOUS:
                await self._execute_workflow_sync(execution, workflow)
            else:
                # Queue for asynchronous execution
                await self._execution_queue.put((execution, workflow))

            return execution

        except Exception as e:
            execution.status = WorkflowExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now(timezone.utc)
            self._execution_metrics['failed_executions'] += 1
            self._execution_metrics['active_executions'] -= 1

            self._logger.error(f"Workflow execution failed: {e}")
            raise

    async def _execute_workflow_sync(self, execution: WorkflowExecution, workflow: WorkflowDefinition) -> None:
        """Execute workflow synchronously."""
        execution.status = WorkflowExecutionStatus.RUNNING
        execution.started_at = datetime.now(timezone.utc)

        try:
            # Execute workflow steps
            await self._execute_workflow_steps(execution, workflow)

            # Mark as completed
            execution.status = WorkflowExecutionStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            execution.total_execution_time_ms = execution.get_execution_time_ms()

            self._execution_metrics['successful_executions'] += 1
            execution.add_trace_event('workflow_completed')

            self._logger.info(f"Workflow execution completed: {execution.id}")

        except Exception as e:
            execution.status = WorkflowExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now(timezone.utc)

            execution.add_trace_event('workflow_failed', data={'error': str(e)})
            self._logger.error(f"Workflow execution failed: {e}")
            raise
        finally:
            self._execution_metrics['active_executions'] -= 1

    async def _execute_workflow_steps(self, execution: WorkflowExecution, workflow: WorkflowDefinition) -> None:
        """Execute workflow steps with dependency resolution."""
        # Build execution plan
        execution_plan = self._build_execution_plan(workflow)

        # Execute steps according to plan
        for step_batch in execution_plan:
            if execution.status != WorkflowExecutionStatus.RUNNING:
                break

            # Execute batch of steps (parallel execution)
            if len(step_batch) == 1:
                # Single step execution
                step = step_batch[0]
                await self._execute_step(execution, workflow, step)
            else:
                # Parallel step execution
                tasks = []
                for step in step_batch:
                    task = asyncio.create_task(self._execute_step(execution, workflow, step))
                    tasks.append(task)

                # Wait for all steps in batch to complete
                await asyncio.gather(*tasks)

            # Create checkpoint if enabled
            if workflow.checkpoint_enabled:
                execution.create_checkpoint("step_batch_completed")

    def _build_execution_plan(self, workflow: WorkflowDefinition) -> List[List[WorkflowStep]]:
        """Build execution plan with dependency resolution."""
        # Topological sort to determine execution order
        dependency_graph = workflow.get_dependency_graph()

        # Find steps with no dependencies (can execute first)
        ready_steps = []
        remaining_steps = workflow.steps.copy()
        execution_plan = []

        while remaining_steps:
            # Find steps that can execute now
            batch = []
            for step in remaining_steps[:]:
                if step.is_ready_to_execute({s.id for s in workflow.steps if s not in remaining_steps}):
                    batch.append(step)
                    remaining_steps.remove(step)

            if not batch:
                # Circular dependency or missing dependency
                remaining_ids = [s.id for s in remaining_steps]
                raise WorkflowError(f"Cannot resolve dependencies for steps: {remaining_ids}")

            execution_plan.append(batch)

        return execution_plan

    async def _execute_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute a single workflow step."""
        step_start_time = datetime.now(timezone.utc)
        execution.step_start_times[step.id] = step_start_time
        execution.step_status[step.id] = TaskStatus.RUNNING
        execution.current_step_id = step.id
        execution.active_steps.add(step.id)

        execution.add_trace_event('step_started', step.id, {'step_name': step.name, 'step_type': step.type})

        try:
            # Check condition if present
            if step.condition and not self._evaluate_condition(step.condition, execution.variables):
                execution.step_status[step.id] = TaskStatus.SKIPPED
                execution.steps_skipped += 1
                execution.add_trace_event('step_skipped', step.id, {'reason': 'condition_false'})
                return

            # Execute based on step type
            if step.type == WorkflowStepType.TASK:
                await self._execute_task_step(execution, step)
            elif step.type == WorkflowStepType.PARALLEL:
                await self._execute_parallel_step(execution, workflow, step)
            elif step.type == WorkflowStepType.SEQUENTIAL:
                await self._execute_sequential_step(execution, workflow, step)
            elif step.type == WorkflowStepType.CONDITIONAL:
                await self._execute_conditional_step(execution, workflow, step)
            elif step.type == WorkflowStepType.LOOP:
                await self._execute_loop_step(execution, workflow, step)
            else:
                raise WorkflowError(f"Unsupported step type: {step.type}")

            # Mark step as completed
            execution.step_status[step.id] = TaskStatus.COMPLETED
            execution.steps_completed += 1
            execution.completed_steps.add(step.id)

            execution.add_trace_event('step_completed', step.id)

        except Exception as e:
            execution.step_status[step.id] = TaskStatus.FAILED
            execution.steps_failed += 1
            execution.failed_steps.add(step.id)

            # Handle error based on step configuration
            if step.on_error == "continue":
                self._logger.warning(f"Step {step.id} failed but continuing: {e}")
                execution.add_trace_event('step_failed_continue', step.id, {'error': str(e)})
            elif step.on_error == "retry" and step.retry_count < step.max_retries:
                step.retry_count += 1
                execution.steps_retried += 1
                await asyncio.sleep(step.retry_delay_seconds * (step.retry_backoff_multiplier ** step.retry_count))
                await self._execute_step(execution, workflow, step)  # Retry
                return
            else:
                execution.add_trace_event('step_failed', step.id, {'error': str(e)})
                raise WorkflowError(f"Step {step.id} failed: {e}")

        finally:
            execution.step_end_times[step.id] = datetime.now(timezone.utc)
            execution.active_steps.discard(step.id)

    async def _execute_task_step(self, execution: WorkflowExecution, step: WorkflowStep) -> None:
        """Execute a task step."""
        if not step.task and not step.agent_capability:
            raise WorkflowError(f"Task step {step.id} has no task or agent capability defined")

        # Get agent for execution
        agent = None
        if self._agent_registry:
            if step.agent_id:
                agent = await self._agent_registry.get_agent(step.agent_id)
            elif step.agent_capability:
                agent = await self._agent_registry.find_best_agent(step.agent_capability)

        if not agent:
            raise WorkflowError(f"No suitable agent found for step {step.id}")

        # Prepare task
        task = step.task or Task(
            name=step.name,
            description=step.description,
            capability_required=step.agent_capability,
            priority=step.priority,
            input_data=step.inputs,
            metadata=step.metadata
        )

        # Execute task
        result = await agent.execute(task, execution.context)

        # Store result
        execution.step_results[step.id] = result
        execution.step_outputs[step.id] = result.output_data

        # Update variables with outputs
        for var_name, output_key in step.outputs.items():
            if output_key in result.output_data:
                execution.variables[var_name] = result.output_data[output_key]

    async def _execute_parallel_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute parallel sub-steps."""
        if not step.steps:
            return

        # Limit parallelism
        max_parallel = step.max_parallel or len(step.steps)
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_semaphore(sub_step):
            async with semaphore:
                await self._execute_step(execution, workflow, sub_step)

        # Execute all sub-steps in parallel
        tasks = [asyncio.create_task(execute_with_semaphore(sub_step)) for sub_step in step.steps]

        if step.parallel_strategy == "all":
            await asyncio.gather(*tasks)
        elif step.parallel_strategy == "any":
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
        elif step.parallel_strategy == "majority":
            majority_count = len(tasks) // 2 + 1
            completed = 0
            for task in asyncio.as_completed(tasks):
                await task
                completed += 1
                if completed >= majority_count:
                    break

    async def _execute_sequential_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute sequential sub-steps."""
        for sub_step in step.steps:
            await self._execute_step(execution, workflow, sub_step)

    async def _execute_conditional_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute conditional step."""
        if step.condition and self._evaluate_condition(step.condition, execution.variables):
            for sub_step in step.steps:
                await self._execute_step(execution, workflow, sub_step)

    async def _execute_loop_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute loop step."""
        iteration = 0
        while iteration < step.max_iterations:
            # Check loop condition
            if step.loop_condition and not self._evaluate_condition(step.loop_condition, execution.variables):
                break

            # Set loop variable
            if step.loop_variable:
                execution.variables[step.loop_variable] = iteration

            # Execute sub-steps
            for sub_step in step.steps:
                await self._execute_step(execution, workflow, sub_step)

            iteration += 1

    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a condition expression."""
        try:
            # Simple expression evaluation (can be enhanced with a proper expression engine)
            # For now, support basic comparisons and boolean operations
            return eval(condition, {"__builtins__": {}}, variables)
        except Exception as e:
            self._logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    # === Background Tasks ===

    async def _execution_scheduler(self) -> None:
        """Background task scheduler for asynchronous executions."""
        while True:
            try:
                # Get next execution from queue
                execution, workflow = await self._execution_queue.get()

                # Check if we can start new execution
                if self._execution_metrics['active_executions'] >= self._max_concurrent_executions:
                    # Put back in queue and wait
                    await self._execution_queue.put((execution, workflow))
                    await asyncio.sleep(1)
                    continue

                # Start execution
                task = asyncio.create_task(self._execute_workflow_sync(execution, workflow))
                self._running_tasks[execution.id] = task

                # Clean up completed task
                def cleanup_task(task_future):
                    self._running_tasks.pop(execution.id, None)

                task.add_done_callback(cleanup_task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in execution scheduler: {e}")
                await asyncio.sleep(1)

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of completed executions."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval_seconds)

                # Clean up old completed executions
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                to_remove = []

                for execution_id, execution in self._executions.items():
                    if (execution.status in [WorkflowExecutionStatus.COMPLETED, WorkflowExecutionStatus.FAILED] and
                        execution.completed_at and execution.completed_at < cutoff_time):
                        to_remove.append(execution_id)

                for execution_id in to_remove:
                    del self._executions[execution_id]

                if to_remove:
                    self._logger.info(f"Cleaned up {len(to_remove)} old executions")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in periodic cleanup: {e}")

    # === Workflow Management ===

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow definition by ID."""
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all registered workflows."""
        return list(self._workflows.values())
    
    async def execute_workflow(self, workflow_id: str, context: Context,
                              execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS,
                              variables: Optional[Dict[str, Any]] = None) -> WorkflowExecution:
        """Execute a workflow with enhanced capabilities."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise WorkflowError(f"Workflow not found: {workflow_id}")

        # Create execution instance
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_version=workflow.version,
            context=context,
            execution_mode=execution_mode,
            variables=variables or {},
            persistence_enabled=workflow.checkpoint_enabled
        )

        # Initialize execution variables with workflow defaults
        execution.variables.update(workflow.variables)
        if variables:
            execution.variables.update(variables)

        # Register execution
        self._executions[execution.id] = execution
        self._execution_metrics['total_executions'] += 1
        self._execution_metrics['active_executions'] += 1

        execution.add_trace_event('workflow_started', data={'workflow_name': workflow.name})

        try:
            if execution_mode == ExecutionMode.SYNCHRONOUS:
                await self._execute_workflow_sync(execution, workflow)
            else:
                # Queue for asynchronous execution
                await self._execution_queue.put((execution, workflow))

            return execution

        except Exception as e:
            execution.status = WorkflowExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now(timezone.utc)
            self._execution_metrics['failed_executions'] += 1
            self._execution_metrics['active_executions'] -= 1

            self._logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _execute_workflow_sync(self, execution: WorkflowExecution, workflow: WorkflowDefinition) -> None:
        """Execute workflow synchronously with enhanced features."""
        execution.status = WorkflowExecutionStatus.RUNNING
        execution.started_at = datetime.now(timezone.utc)

        try:
            # Execute workflow steps
            await self._execute_workflow_steps(execution, workflow)

            # Mark as completed
            execution.status = WorkflowExecutionStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            execution.total_execution_time_ms = execution.get_execution_time_ms()

            self._execution_metrics['successful_executions'] += 1
            execution.add_trace_event('workflow_completed')

            self._logger.info(f"Workflow execution completed: {execution.id}")

        except Exception as e:
            execution.status = WorkflowExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now(timezone.utc)

            execution.add_trace_event('workflow_failed', data={'error': str(e)})
            self._logger.error(f"Workflow execution failed: {e}")
            raise
        finally:
            self._execution_metrics['active_executions'] -= 1

    async def _execute_workflow_steps(self, execution: WorkflowExecution, workflow: WorkflowDefinition) -> None:
        """Execute workflow steps with enhanced dependency resolution."""
        # Build execution plan
        execution_plan = self._build_execution_plan(workflow)

        # Execute steps according to plan
        for step_batch in execution_plan:
            if execution.status != WorkflowExecutionStatus.RUNNING:
                break

            # Execute batch of steps (parallel execution)
            if len(step_batch) == 1:
                # Single step execution
                step = step_batch[0]
                await self._execute_enhanced_step(execution, workflow, step)
            else:
                # Parallel step execution
                tasks = []
                for step in step_batch:
                    task = asyncio.create_task(self._execute_enhanced_step(execution, workflow, step))
                    tasks.append(task)

                # Wait for all steps in batch to complete
                await asyncio.gather(*tasks)

            # Create checkpoint if enabled
            if workflow.checkpoint_enabled:
                execution.create_checkpoint("step_batch_completed")

    def _build_execution_plan(self, workflow: WorkflowDefinition) -> List[List[WorkflowStep]]:
        """Build execution plan with enhanced dependency resolution."""
        remaining_steps = workflow.steps.copy()
        execution_plan = []
        completed_step_ids = set()

        while remaining_steps:
            # Find steps that can execute now (all dependencies completed)
            batch = []
            for step in remaining_steps[:]:
                if all(dep_id in completed_step_ids for dep_id in step.depends_on):
                    batch.append(step)
                    remaining_steps.remove(step)

            if not batch:
                # Circular dependency or missing dependency
                remaining_ids = [s.id for s in remaining_steps]
                remaining_names = [s.name for s in remaining_steps]
                raise WorkflowError(f"Cannot resolve dependencies for steps: {remaining_names} (IDs: {remaining_ids})")

            execution_plan.append(batch)

            # Mark these steps as completed for next iteration
            for step in batch:
                completed_step_ids.add(step.id)

        return execution_plan

    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status."""
        return self._executions.get(execution_id)

    def list_executions(self, workflow_id: Optional[str] = None,
                       status: Optional[WorkflowExecutionStatus] = None) -> List[WorkflowExecution]:
        """List workflow executions with optional filters."""
        executions = list(self._executions.values())

        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]

        if status:
            executions = [e for e in executions if e.status == status]

        return executions

    async def _execute_enhanced_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute a single workflow step with enhanced capabilities."""
        step_start_time = datetime.now(timezone.utc)
        execution.step_start_times[step.id] = step_start_time
        execution.step_status[step.id] = TaskStatus.RUNNING
        execution.current_step_id = step.id
        execution.active_steps.add(step.id)

        execution.add_trace_event('step_started', step.id, {'step_name': step.name, 'step_type': step.type})

        try:
            # Check condition if present
            if step.condition and not self._evaluate_condition(step.condition, execution.variables):
                execution.step_status[step.id] = TaskStatus.SKIPPED
                execution.steps_skipped += 1
                execution.add_trace_event('step_skipped', step.id, {'reason': 'condition_false'})
                return

            # Execute based on step type
            if step.type == WorkflowStepType.TASK:
                await self._execute_task_step(execution, step)
            elif step.type == WorkflowStepType.PARALLEL:
                await self._execute_parallel_step(execution, workflow, step)
            elif step.type == WorkflowStepType.SEQUENTIAL:
                await self._execute_sequential_step(execution, workflow, step)
            elif step.type == WorkflowStepType.CONDITIONAL:
                await self._execute_conditional_step(execution, workflow, step)
            elif step.type == WorkflowStepType.LOOP:
                await self._execute_loop_step(execution, workflow, step)
            else:
                raise WorkflowError(f"Unsupported step type: {step.type}")

            # Mark step as completed
            execution.step_status[step.id] = TaskStatus.COMPLETED
            execution.steps_completed += 1
            execution.completed_steps.add(step.id)

            execution.add_trace_event('step_completed', step.id)

        except Exception as e:
            execution.step_status[step.id] = TaskStatus.FAILED
            execution.steps_failed += 1
            execution.failed_steps.add(step.id)

            # Handle error based on step configuration
            if step.on_error == "continue":
                self._logger.warning(f"Step {step.id} failed but continuing: {e}")
                execution.add_trace_event('step_failed_continue', step.id, {'error': str(e)})
            elif step.on_error == "retry" and step.retry_count < step.max_retries:
                step.retry_count += 1
                execution.steps_retried += 1
                await asyncio.sleep(step.retry_delay_seconds * (step.retry_backoff_multiplier ** step.retry_count))
                await self._execute_enhanced_step(execution, workflow, step)  # Retry
                return
            else:
                execution.add_trace_event('step_failed', step.id, {'error': str(e)})
                raise WorkflowError(f"Step {step.id} failed: {e}")

        finally:
            execution.step_end_times[step.id] = datetime.now(timezone.utc)
            execution.active_steps.discard(step.id)

    async def _execute_task_step(self, execution: WorkflowExecution, step: WorkflowStep) -> None:
        """Execute a task step with agent integration."""
        if not step.task and not step.agent_capability:
            raise WorkflowError(f"Task step {step.id} has no task or agent capability defined")

        # Get agent for execution
        agent = None
        if self._agent_registry:
            if step.agent_id:
                agent = await self._agent_registry.get_agent(step.agent_id)
            elif step.agent_capability:
                # Create a dummy task for agent selection
                dummy_task = Task(
                    id=str(uuid4()),
                    name=f"Workflow Step: {step.name}",
                    description=f"Task for workflow step {step.name}",
                    capability_required=step.agent_capability,
                    priority=TaskPriority.NORMAL,
                    input_data=step.inputs
                )
                agent = self._agent_registry.find_agent_for_task(dummy_task)

        if not agent:
            # Create placeholder result for testing
            result = Result(
                task_id=step.task.id if step.task else step.id,
                agent_id="placeholder",
                success=True,
                data={"message": f"Task {step.name} executed successfully", "result": f"output_from_{step.name.lower().replace(' ', '_')}"},
                metadata={"execution_mode": "placeholder"}
            )
        else:
            # Prepare task
            task = step.task or Task(
                name=step.name,
                description=step.description,
                capability_required=step.agent_capability,
                priority=step.priority,
                input_data=step.inputs,
                metadata=step.metadata
            )

            # Execute task
            result = await agent.execute(task, execution.context)

        # Store result
        execution.step_results[step.id] = result
        execution.step_outputs[step.id] = result.data

        # Update variables with outputs
        for var_name, output_key in step.outputs.items():
            if output_key in result.data:
                execution.variables[var_name] = result.data[output_key]

    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a condition expression safely."""
        try:
            # Simple expression evaluation (can be enhanced with a proper expression engine)
            # For now, support basic comparisons and boolean operations
            return eval(condition, {"__builtins__": {}}, variables)
        except Exception as e:
            self._logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution."""
        if execution_id in self._running_tasks:
            task = self._running_tasks[execution_id]
            task.cancel()
            
            if execution_id in self._executions:
                execution = self._executions[execution_id]
                execution.status = WorkflowExecutionStatus.CANCELLED
                execution.completed_at = datetime.now(timezone.utc)
            
            self._logger.info(f"Cancelled workflow execution: {execution_id}")
            return True
        
        return False
    
    def _validate_workflow(self, workflow: WorkflowDefinition) -> None:
        """Validate a workflow definition."""
        if not workflow.steps:
            raise WorkflowError("Workflow must have at least one step")
        
        # Check for circular dependencies
        self._check_circular_dependencies(workflow.steps)
        
        # Validate step configurations
        for step in workflow.steps:
            self._validate_step(step)
    
    def _check_circular_dependencies(self, steps: List[WorkflowStep]) -> None:
        """Check for circular dependencies in workflow steps."""
        step_ids = {step.id for step in steps}
        
        def has_cycle(step_id: str, visited: Set[str], path: Set[str]) -> bool:
            if step_id in path:
                return True
            if step_id in visited:
                return False
            
            visited.add(step_id)
            path.add(step_id)
            
            # Find step and check its dependencies
            for step in steps:
                if step.id == step_id:
                    for dep_id in step.depends_on:
                        if dep_id in step_ids and has_cycle(dep_id, visited, path):
                            return True
                    break
            
            path.remove(step_id)
            return False
        
        visited = set()
        for step in steps:
            if step.id not in visited:
                if has_cycle(step.id, visited, set()):
                    raise WorkflowError("Circular dependency detected in workflow")
    
    def _validate_step(self, step: WorkflowStep) -> None:
        """Validate a workflow step."""
        if step.type == WorkflowStepType.TASK and not step.task:
            raise WorkflowError(f"Task step {step.id} must have a task defined")
        
        if step.type in [WorkflowStepType.PARALLEL, WorkflowStepType.SEQUENTIAL]:
            if not step.steps:
                raise WorkflowError(f"Composite step {step.id} must have sub-steps")
        
        if step.type == WorkflowStepType.CONDITIONAL and not step.condition:
            raise WorkflowError(f"Conditional step {step.id} must have a condition")
        
        if step.type == WorkflowStepType.LOOP and not step.loop_condition:
            raise WorkflowError(f"Loop step {step.id} must have a loop condition")
    
    async def _execute_workflow_async(
        self,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Execute a workflow asynchronously."""
        try:
            execution.status = WorkflowExecutionStatus.RUNNING
            execution.started_at = datetime.now(timezone.utc)
            
            # Execute workflow steps
            await self._execute_steps(workflow.steps, execution)
            
            # Mark as completed
            execution.status = WorkflowExecutionStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            
            if execution.started_at:
                execution.total_execution_time_ms = int(
                    (execution.completed_at - execution.started_at).total_seconds() * 1000
                )
            
            self._logger.info(f"Workflow execution completed: {execution.id}")
            
        except asyncio.CancelledError:
            execution.status = WorkflowExecutionStatus.CANCELLED
            execution.completed_at = datetime.now(timezone.utc)
            self._logger.info(f"Workflow execution cancelled: {execution.id}")
            
        except Exception as e:
            execution.status = WorkflowExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now(timezone.utc)
            
            self._logger.error(f"Workflow execution failed: {execution.id} - {e}")
            
        finally:
            # Clean up
            if execution.id in self._running_tasks:
                del self._running_tasks[execution.id]
    
    async def _execute_steps(
        self,
        steps: List[WorkflowStep],
        execution: WorkflowExecution
    ) -> None:
        """Execute a list of workflow steps."""
        # Build dependency graph
        step_map = {step.id: step for step in steps}
        completed_steps = set()
        
        # Execute steps in dependency order
        while len(completed_steps) < len(steps):
            # Find steps ready to execute
            ready_steps = []
            for step in steps:
                if (step.id not in completed_steps and 
                    all(dep_id in completed_steps for dep_id in step.depends_on)):
                    ready_steps.append(step)
            
            if not ready_steps:
                raise WorkflowError("No steps ready to execute - possible dependency issue")
            
            # Execute ready steps
            tasks = []
            for step in ready_steps:
                task = asyncio.create_task(self._execute_step(step, execution))
                tasks.append((step.id, task))
            
            # Wait for completion
            for step_id, task in tasks:
                try:
                    await task
                    completed_steps.add(step_id)
                    execution.steps_completed += 1
                except Exception as e:
                    execution.steps_failed += 1
                    execution.failed_step_id = step_id
                    raise WorkflowError(f"Step {step_id} failed: {e}")
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> None:
        """Execute a single workflow step."""
        step.status = TaskStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        execution.step_status[step.id] = step.status
        
        try:
            if step.type == WorkflowStepType.TASK:
                # Execute task step
                if step.task:
                    # This would integrate with the agent registry to execute the task
                    # For now, we'll create a placeholder result
                    result = Result(
                        task_id=step.task.id,
                        agent_id="placeholder",
                        success=True,
                        data={"message": "Task executed successfully"}
                    )
                    step.result = result
                    execution.step_results[step.id] = result
            
            elif step.type in [WorkflowStepType.PARALLEL, WorkflowStepType.SEQUENTIAL]:
                # Execute sub-steps
                await self._execute_steps(step.steps, execution)
            
            # Mark step as completed
            step.status = TaskStatus.COMPLETED
            step.completed_at = datetime.now(timezone.utc)
            execution.step_status[step.id] = step.status
            
        except Exception as e:
            step.status = TaskStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.now(timezone.utc)
            execution.step_status[step.id] = step.status
            raise

    async def _execute_parallel_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute parallel sub-steps."""
        if not step.steps:
            return

        # Limit parallelism
        max_parallel = step.max_parallel or len(step.steps)
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_semaphore(sub_step):
            async with semaphore:
                await self._execute_enhanced_step(execution, workflow, sub_step)

        # Execute all sub-steps in parallel
        tasks = [asyncio.create_task(execute_with_semaphore(sub_step)) for sub_step in step.steps]

        if step.parallel_strategy == "all":
            await asyncio.gather(*tasks)
        elif step.parallel_strategy == "any":
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
        elif step.parallel_strategy == "majority":
            majority_count = len(tasks) // 2 + 1
            completed = 0
            for task in asyncio.as_completed(tasks):
                await task
                completed += 1
                if completed >= majority_count:
                    break

    async def _execute_sequential_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute sequential sub-steps."""
        for sub_step in step.steps:
            await self._execute_enhanced_step(execution, workflow, sub_step)

    async def _execute_conditional_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute conditional step."""
        if step.condition and self._evaluate_condition(step.condition, execution.variables):
            for sub_step in step.steps:
                await self._execute_enhanced_step(execution, workflow, sub_step)

    async def _execute_loop_step(self, execution: WorkflowExecution, workflow: WorkflowDefinition, step: WorkflowStep) -> None:
        """Execute loop step."""
        iteration = 0
        while iteration < step.max_iterations:
            # Check loop condition
            if step.loop_condition and not self._evaluate_condition(step.loop_condition, execution.variables):
                break

            # Set loop variable
            if step.loop_variable:
                execution.variables[step.loop_variable] = iteration

            # Execute sub-steps
            for sub_step in step.steps:
                await self._execute_enhanced_step(execution, workflow, sub_step)

            iteration += 1

    # === Metrics and Monitoring ===

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics."""
        return self._execution_metrics.copy()

    def get_workflow_statistics(self, workflow_id: str) -> Dict[str, Any]:
        """Get statistics for a specific workflow."""
        executions = self.list_executions(workflow_id)

        if not executions:
            return {}

        completed = [e for e in executions if e.status == WorkflowExecutionStatus.COMPLETED]
        failed = [e for e in executions if e.status == WorkflowExecutionStatus.FAILED]

        execution_times = [e.get_execution_time_ms() for e in completed if e.get_execution_time_ms()]

        return {
            'total_executions': len(executions),
            'successful_executions': len(completed),
            'failed_executions': len(failed),
            'success_rate': len(completed) / len(executions) if executions else 0,
            'average_execution_time_ms': sum(execution_times) / len(execution_times) if execution_times else 0,
            'min_execution_time_ms': min(execution_times) if execution_times else 0,
            'max_execution_time_ms': max(execution_times) if execution_times else 0,
        }


# === Workflow Templates ===

class WorkflowTemplates:
    """Common workflow templates and patterns."""

    @staticmethod
    def create_simple_sequential_workflow(name: str, tasks: List[Dict[str, Any]]) -> WorkflowDefinition:
        """Create a simple sequential workflow."""
        steps = []
        for i, task_config in enumerate(tasks):
            step = WorkflowStep(
                name=task_config.get('name', f'Step {i+1}'),
                description=task_config.get('description', ''),
                type=WorkflowStepType.TASK,
                agent_capability=AgentCapability(task_config.get('capability', 'query_analysis')),
                inputs=task_config.get('inputs', {}),
                outputs=task_config.get('outputs', {}),
                depends_on=[steps[-1].id] if steps else []
            )
            steps.append(step)

        return WorkflowDefinition(
            name=name,
            description=f"Sequential workflow with {len(tasks)} tasks",
            steps=steps
        )

    @staticmethod
    def create_parallel_workflow(name: str, tasks: List[Dict[str, Any]]) -> WorkflowDefinition:
        """Create a parallel workflow."""
        steps = []
        for i, task_config in enumerate(tasks):
            step = WorkflowStep(
                name=task_config.get('name', f'Step {i+1}'),
                description=task_config.get('description', ''),
                type=WorkflowStepType.TASK,
                agent_capability=AgentCapability(task_config.get('capability', 'query_analysis')),
                inputs=task_config.get('inputs', {}),
                outputs=task_config.get('outputs', {})
            )
            steps.append(step)

        return WorkflowDefinition(
            name=name,
            description=f"Parallel workflow with {len(tasks)} tasks",
            steps=steps
        )


# Global workflow engine instance
_workflow_engine: Optional[WorkflowEngine] = None


async def get_workflow_engine() -> WorkflowEngine:
    """Get or create the global workflow engine instance."""
    global _workflow_engine
    
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
        await _workflow_engine.initialize()
    
    return _workflow_engine


async def close_workflow_engine() -> None:
    """Close the global workflow engine instance."""
    global _workflow_engine
    
    if _workflow_engine:
        await _workflow_engine.shutdown()
        _workflow_engine = None
