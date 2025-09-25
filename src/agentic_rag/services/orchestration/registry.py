"""
Enhanced Agent Registry System

This module provides dynamic tool registry for agent discovery and selection.
It manages agent registration, capability-based discovery, tool metadata,
health monitoring, performance tracking, and intelligent load balancing.
"""

import asyncio
import statistics
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID

from pydantic import BaseModel, Field
import structlog

from .base import Agent, AgentCapability, AgentStatus, Task

logger = structlog.get_logger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies for agent selection."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    WEIGHTED_PERFORMANCE = "weighted_performance"
    RANDOM = "random"


class PerformanceMetrics(BaseModel):
    """Performance metrics for an agent."""

    # Execution metrics
    total_tasks_completed: int = Field(default=0, description="Total tasks completed")
    total_tasks_failed: int = Field(default=0, description="Total tasks failed")
    total_execution_time_ms: int = Field(default=0, description="Total execution time")

    # Response time metrics
    average_response_time_ms: float = Field(default=0.0, description="Average response time")
    min_response_time_ms: float = Field(default=float('inf'), description="Minimum response time")
    max_response_time_ms: float = Field(default=0.0, description="Maximum response time")
    response_time_p95_ms: float = Field(default=0.0, description="95th percentile response time")

    # Recent performance (last 100 tasks)
    recent_response_times: List[float] = Field(default_factory=list, description="Recent response times")
    recent_success_rate: float = Field(default=1.0, description="Recent success rate")

    # Resource utilization
    current_load: int = Field(default=0, description="Current number of active tasks")
    peak_load: int = Field(default=0, description="Peak concurrent tasks")
    memory_usage_mb: float = Field(default=0.0, description="Current memory usage")
    cpu_usage_percent: float = Field(default=0.0, description="Current CPU usage")

    # Quality metrics
    average_confidence_score: float = Field(default=0.0, description="Average result confidence")
    user_satisfaction_score: float = Field(default=0.0, description="User satisfaction score")

    # Availability metrics
    uptime_percentage: float = Field(default=100.0, description="Uptime percentage")
    last_failure_time: Optional[datetime] = Field(None, description="Last failure timestamp")
    consecutive_failures: int = Field(default=0, description="Consecutive failure count")

    # Update tracking
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def update_task_completion(self, execution_time_ms: float, success: bool, confidence: float = 0.0):
        """Update metrics after task completion."""
        if success:
            self.total_tasks_completed += 1
            self.consecutive_failures = 0
        else:
            self.total_tasks_failed += 1
            self.consecutive_failures += 1
            self.last_failure_time = datetime.now(timezone.utc)

        # Update response time metrics
        self.total_execution_time_ms += execution_time_ms
        self.recent_response_times.append(execution_time_ms)

        # Keep only last 100 response times
        if len(self.recent_response_times) > 100:
            self.recent_response_times = self.recent_response_times[-100:]

        # Recalculate metrics
        self._recalculate_metrics()

        # Update confidence
        if confidence > 0:
            total_tasks = self.total_tasks_completed + self.total_tasks_failed
            self.average_confidence_score = (
                (self.average_confidence_score * (total_tasks - 1) + confidence) / total_tasks
            )

        self.last_updated = datetime.now(timezone.utc)

    def _recalculate_metrics(self):
        """Recalculate derived metrics."""
        if self.recent_response_times:
            self.average_response_time_ms = statistics.mean(self.recent_response_times)
            self.min_response_time_ms = min(self.recent_response_times)
            self.max_response_time_ms = max(self.recent_response_times)

            # Calculate 95th percentile
            sorted_times = sorted(self.recent_response_times)
            p95_index = int(0.95 * len(sorted_times))
            self.response_time_p95_ms = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]

        # Calculate recent success rate
        total_recent = len(self.recent_response_times)
        if total_recent > 0:
            recent_successes = self.total_tasks_completed - max(0, self.total_tasks_completed - total_recent)
            self.recent_success_rate = recent_successes / total_recent


class ToolMetadata(BaseModel):
    """Enhanced metadata for registered tools/agents."""

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    version: str = Field(default="1.0.0", description="Agent version")

    # Capabilities and compatibility
    capabilities: List[AgentCapability] = Field(..., description="Agent capabilities")
    supported_input_types: List[str] = Field(default_factory=list, description="Supported input types")
    supported_output_types: List[str] = Field(default_factory=list, description="Supported output types")
    capability_scores: Dict[str, float] = Field(default_factory=dict, description="Capability performance scores")
    compatibility_matrix: Dict[str, List[str]] = Field(default_factory=dict, description="Agent compatibility info")

    # Performance characteristics
    max_concurrent_tasks: int = Field(default=1, description="Maximum concurrent tasks")
    preferred_batch_size: int = Field(default=1, description="Preferred batch size for optimal performance")
    timeout_seconds: int = Field(default=300, description="Default timeout for tasks")
    retry_attempts: int = Field(default=3, description="Default retry attempts")

    # Resource requirements
    min_memory_mb: int = Field(default=512, description="Minimum memory requirement")
    max_memory_mb: int = Field(default=2048, description="Maximum memory usage")
    cpu_cores_required: float = Field(default=1.0, description="CPU cores required")
    gpu_required: bool = Field(default=False, description="Whether GPU is required")

    # Quality and reliability
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Reliability score")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Quality score")
    user_rating: float = Field(default=5.0, ge=1.0, le=5.0, description="User rating")
    certification_level: str = Field(default="standard", description="Certification level")

    # Health and availability
    status: AgentStatus = Field(default=AgentStatus.READY)
    last_health_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    health_check_interval_seconds: int = Field(default=60)
    health_check_timeout_seconds: int = Field(default=30)

    # Configuration and customization
    config_schema: Dict = Field(default_factory=dict, description="Configuration schema")
    default_config: Dict = Field(default_factory=dict, description="Default configuration")
    customizable_parameters: List[str] = Field(default_factory=list, description="Customizable parameters")

    # Deployment and environment
    deployment_environment: str = Field(default="production", description="Deployment environment")
    container_image: Optional[str] = Field(None, description="Container image if containerized")
    service_endpoint: Optional[str] = Field(None, description="Service endpoint if remote")

    # Registration and lifecycle
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    registered_by: Optional[str] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deprecation_date: Optional[datetime] = Field(None, description="Planned deprecation date")

    # Tags and categorization
    tags: List[str] = Field(default_factory=list, description="Agent tags for categorization")
    category: str = Field(default="general", description="Agent category")
    priority: int = Field(default=100, description="Agent priority for selection")

    class Config:
        use_enum_values = True


class AgentRegistration(BaseModel):
    """Enhanced agent registration information with performance tracking."""

    agent: Agent = Field(..., description="Agent instance")
    metadata: ToolMetadata = Field(..., description="Agent metadata")
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics, description="Performance metrics")

    # Runtime state
    current_tasks: Set[str] = Field(default_factory=set, description="Currently executing task IDs")
    task_queue: List[str] = Field(default_factory=list, description="Queued task IDs")
    last_used: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Load balancing state
    selection_count: int = Field(default=0, description="Number of times selected")
    last_selected: Optional[datetime] = Field(None, description="Last selection timestamp")
    weight: float = Field(default=1.0, description="Load balancing weight")

    # Health monitoring
    consecutive_health_failures: int = Field(default=0, description="Consecutive health check failures")
    last_health_failure: Optional[datetime] = Field(None, description="Last health check failure")
    health_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall health score")

    # Compatibility tracking
    successful_collaborations: Dict[str, int] = Field(default_factory=dict, description="Successful collaborations with other agents")
    failed_collaborations: Dict[str, int] = Field(default_factory=dict, description="Failed collaborations with other agents")

    class Config:
        arbitrary_types_allowed = True

    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return (
            self.agent.status == AgentStatus.READY and
            len(self.current_tasks) < self.metadata.max_concurrent_tasks and
            self.consecutive_health_failures < 3
        )

    def get_load_factor(self) -> float:
        """Calculate current load factor (0.0 = no load, 1.0 = full capacity)."""
        if self.metadata.max_concurrent_tasks == 0:
            return 1.0
        return len(self.current_tasks) / self.metadata.max_concurrent_tasks

    def get_performance_score(self) -> float:
        """Calculate overall performance score."""
        if self.performance_metrics.total_tasks_completed == 0:
            return 0.5  # Default score for new agents

        # Combine multiple factors
        success_rate = (
            self.performance_metrics.total_tasks_completed /
            (self.performance_metrics.total_tasks_completed + self.performance_metrics.total_tasks_failed)
        )

        # Normalize response time (lower is better)
        avg_response = self.performance_metrics.average_response_time_ms
        response_score = max(0.0, 1.0 - (avg_response / 30000))  # 30 seconds as baseline

        # Combine scores
        return (
            success_rate * 0.4 +
            response_score * 0.3 +
            self.health_score * 0.2 +
            self.performance_metrics.average_confidence_score * 0.1
        )

    def update_selection(self):
        """Update selection tracking."""
        self.selection_count += 1
        self.last_selected = datetime.now(timezone.utc)
        self.last_used = datetime.now(timezone.utc)


class AgentRegistry:
    """Enhanced registry for managing agent discovery, selection, and load balancing."""

    def __init__(self, load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.WEIGHTED_PERFORMANCE):
        self._agents: Dict[str, AgentRegistration] = {}
        self._capability_index: Dict[AgentCapability, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._category_index: Dict[str, Set[str]] = {}

        # Load balancing
        self._load_balancing_strategy = load_balancing_strategy
        self._round_robin_counters: Dict[AgentCapability, int] = {}

        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_interval = 60  # seconds
        self._shutdown = False

        # Performance tracking
        self._performance_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._selection_history: List[Tuple[datetime, str, str]] = []  # (timestamp, agent_id, capability)

        # Configuration
        self._max_history_entries = 1000
        self._performance_window_hours = 24

        self._logger = structlog.get_logger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the agent registry."""
        self._logger.info("Initializing agent registry")
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self._logger.info("Agent registry initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the agent registry."""
        self._logger.info("Shutting down agent registry")
        
        self._shutdown = True
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown all registered agents
        for registration in self._agents.values():
            try:
                await registration.agent.shutdown()
            except Exception as e:
                self._logger.error(f"Error shutting down agent {registration.agent.agent_id}: {e}")
        
        self._agents.clear()
        self._capability_index.clear()
        
        self._logger.info("Agent registry shutdown complete")
    
    async def register_agent(
        self,
        agent: Agent,
        metadata: Optional[ToolMetadata] = None,
        validate_compatibility: bool = True
    ) -> bool:
        """Register an agent with enhanced validation and indexing."""
        try:
            # Validate agent
            if not await self._validate_agent(agent):
                self._logger.error(f"Agent validation failed: {agent.agent_id}")
                return False

            # Create metadata if not provided
            if metadata is None:
                metadata = ToolMetadata(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    description=agent.description,
                    capabilities=list(agent.capabilities)
                )

            # Validate metadata
            if not self._validate_metadata(metadata):
                self._logger.error(f"Metadata validation failed: {agent.agent_id}")
                return False

            # Check for conflicts
            if agent.agent_id in self._agents:
                self._logger.warning(f"Agent already registered, updating: {agent.agent_id}")
                await self.deregister_agent(agent.agent_id)

            # Initialize agent if not already done
            if agent.status == AgentStatus.INITIALIZING:
                await agent.initialize()

            # Perform compatibility check if requested
            if validate_compatibility:
                compatibility_issues = await self._check_compatibility(agent, metadata)
                if compatibility_issues:
                    self._logger.warning(f"Compatibility issues found for {agent.agent_id}: {compatibility_issues}")

            # Create registration with performance tracking
            registration = AgentRegistration(
                agent=agent,
                metadata=metadata
            )

            # Register agent
            self._agents[agent.agent_id] = registration

            # Update all indexes
            await self._update_indexes(agent.agent_id, metadata)

            # Initialize performance tracking
            self._performance_history[agent.agent_id] = []

            # Initialize round-robin counters for new capabilities
            for capability in agent.capabilities:
                if capability not in self._round_robin_counters:
                    self._round_robin_counters[capability] = 0

            self._logger.info(
                f"Agent registered successfully: {agent.agent_id} "
                f"with capabilities: {[cap.value for cap in agent.capabilities]}"
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False

    async def _validate_agent(self, agent: Agent) -> bool:
        """Validate agent before registration."""
        if not agent.agent_id:
            return False

        if not agent.name:
            return False

        if not agent.capabilities:
            return False

        # Check if agent responds to basic health check
        try:
            # This would be a basic ping/health check
            return agent.status != AgentStatus.ERROR
        except Exception:
            return False

    def _validate_metadata(self, metadata: ToolMetadata) -> bool:
        """Validate metadata before registration."""
        if not metadata.agent_id:
            return False

        if not metadata.capabilities:
            return False

        if metadata.max_concurrent_tasks <= 0:
            return False

        return True

    async def _check_compatibility(self, agent: Agent, metadata: ToolMetadata) -> List[str]:
        """Check compatibility with existing agents."""
        issues = []

        # Check for version conflicts
        for existing_id, existing_reg in self._agents.items():
            if existing_reg.metadata.name == metadata.name and existing_reg.metadata.version != metadata.version:
                issues.append(f"Version conflict with existing agent {existing_id}")

        # Check resource requirements
        if metadata.min_memory_mb > 8192:  # 8GB limit
            issues.append("High memory requirement may cause resource contention")

        # Check capability overlaps
        overlapping_agents = []
        for capability in metadata.capabilities:
            if capability in self._capability_index:
                overlapping_agents.extend(self._capability_index[capability])

        if len(set(overlapping_agents)) > 5:  # More than 5 agents with same capabilities
            issues.append("High capability overlap may indicate redundancy")

        return issues

    async def _update_indexes(self, agent_id: str, metadata: ToolMetadata):
        """Update all indexes for the agent."""
        # Update capability index
        for capability in metadata.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = set()
            self._capability_index[capability].add(agent_id)

        # Update tag index
        for tag in metadata.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(agent_id)

        # Update category index
        if metadata.category not in self._category_index:
            self._category_index[metadata.category] = set()
        self._category_index[metadata.category].add(agent_id)

    async def deregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry."""
        try:
            if agent_id not in self._agents:
                self._logger.warning(f"Agent {agent_id} not found in registry")
                return False
            
            registration = self._agents[agent_id]
            
            # Remove from capability index
            for capability in registration.agent.capabilities:
                if capability in self._capability_index:
                    self._capability_index[capability].discard(agent_id)
                    if not self._capability_index[capability]:
                        del self._capability_index[capability]
            
            # Shutdown agent
            await registration.agent.shutdown()
            
            # Remove from registry
            del self._agents[agent_id]
            
            self._logger.info(f"Unregistered agent {agent_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def find_agents_by_capability(
        self,
        capability: AgentCapability,
        status_filter: Optional[List[AgentStatus]] = None,
        include_performance_scores: bool = True
    ) -> List[AgentRegistration]:
        """Find agents by capability with enhanced filtering and scoring."""
        if capability not in self._capability_index:
            return []

        agent_ids = self._capability_index[capability]
        agents = []

        for agent_id in agent_ids:
            if agent_id in self._agents:
                registration = self._agents[agent_id]

                # Apply status filter
                if status_filter and registration.agent.status not in status_filter:
                    continue

                # Check availability
                if not registration.is_available():
                    continue

                agents.append(registration)

        # Sort by multiple criteria
        if include_performance_scores:
            agents.sort(key=lambda r: (
                -r.get_performance_score(),  # Higher performance score first
                r.get_load_factor(),         # Lower load factor first
                r.performance_metrics.average_response_time_ms,  # Faster response first
                -r.metadata.priority         # Higher priority first
            ))
        else:
            agents.sort(key=lambda r: (
                r.get_load_factor(),         # Lower load factor first
                len(r.current_tasks),        # Fewer current tasks first
                r.performance_metrics.average_response_time_ms or 0  # Faster response first
            ))

        return agents
    
    def find_agent_for_task(
        self,
        task: Task,
        strategy: Optional[LoadBalancingStrategy] = None,
        exclude_agents: Optional[Set[str]] = None
    ) -> Optional[AgentRegistration]:
        """Find the best agent for a specific task using intelligent load balancing."""
        strategy = strategy or self._load_balancing_strategy
        exclude_agents = exclude_agents or set()

        # Get candidates
        candidates = self.find_agents_by_capability(
            task.capability_required,
            status_filter=[AgentStatus.READY, AgentStatus.BUSY]
        )

        if not candidates:
            return None

        # Filter out excluded agents and check availability
        available_agents = []
        for registration in candidates:
            if (registration.agent.agent_id not in exclude_agents and
                registration.is_available()):
                available_agents.append(registration)

        if not available_agents:
            return None

        # Apply load balancing strategy
        selected_agent = self._apply_load_balancing_strategy(
            available_agents,
            strategy,
            task.capability_required
        )

        if selected_agent:
            # Update selection tracking
            selected_agent.update_selection()

            # Record selection in history
            capability_value = (
                task.capability_required.value
                if hasattr(task.capability_required, 'value')
                else str(task.capability_required)
            )
            self._selection_history.append((
                datetime.now(timezone.utc),
                selected_agent.agent.agent_id,
                capability_value
            ))

            # Trim history if needed
            if len(self._selection_history) > self._max_history_entries:
                self._selection_history = self._selection_history[-self._max_history_entries:]

        return selected_agent

    def _apply_load_balancing_strategy(
        self,
        agents: List[AgentRegistration],
        strategy: LoadBalancingStrategy,
        capability: AgentCapability
    ) -> Optional[AgentRegistration]:
        """Apply the specified load balancing strategy."""
        if not agents:
            return None

        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(agents, capability)

        elif strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(agents)

        elif strategy == LoadBalancingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_selection(agents)

        elif strategy == LoadBalancingStrategy.WEIGHTED_PERFORMANCE:
            return self._weighted_performance_selection(agents)

        elif strategy == LoadBalancingStrategy.RANDOM:
            import random
            return random.choice(agents)

        else:
            # Default to weighted performance
            return self._weighted_performance_selection(agents)

    def _round_robin_selection(
        self,
        agents: List[AgentRegistration],
        capability: AgentCapability
    ) -> AgentRegistration:
        """Round-robin selection among available agents."""
        if capability not in self._round_robin_counters:
            self._round_robin_counters[capability] = 0

        index = self._round_robin_counters[capability] % len(agents)
        self._round_robin_counters[capability] += 1

        return agents[index]

    def _least_loaded_selection(self, agents: List[AgentRegistration]) -> AgentRegistration:
        """Select the agent with the lowest current load."""
        return min(agents, key=lambda a: a.get_load_factor())

    def _fastest_response_selection(self, agents: List[AgentRegistration]) -> AgentRegistration:
        """Select the agent with the fastest average response time."""
        return min(agents, key=lambda a: a.performance_metrics.average_response_time_ms or float('inf'))

    def _weighted_performance_selection(self, agents: List[AgentRegistration]) -> AgentRegistration:
        """Select agent based on weighted performance score."""
        # Calculate weights based on performance scores
        weights = []
        for agent in agents:
            performance_score = agent.get_performance_score()
            load_factor = agent.get_load_factor()

            # Higher performance and lower load = higher weight
            weight = performance_score * (1.0 - load_factor) * agent.weight
            weights.append(max(weight, 0.1))  # Minimum weight to ensure all agents can be selected

        # Weighted random selection
        import random
        total_weight = sum(weights)
        if total_weight == 0:
            return agents[0]

        r = random.uniform(0, total_weight)
        cumulative_weight = 0

        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return agents[i]

        return agents[-1]  # Fallback

    def find_agents_by_tags(self, tags: List[str], match_all: bool = False) -> List[AgentRegistration]:
        """Find agents by tags."""
        if not tags:
            return []

        matching_agents = set()

        if match_all:
            # Find agents that have ALL specified tags
            for tag in tags:
                if tag in self._tag_index:
                    tag_agents = self._tag_index[tag]
                    if not matching_agents:
                        matching_agents = tag_agents.copy()
                    else:
                        matching_agents &= tag_agents
                else:
                    return []  # If any tag is not found, no agents match
        else:
            # Find agents that have ANY of the specified tags
            for tag in tags:
                if tag in self._tag_index:
                    matching_agents |= self._tag_index[tag]

        return [self._agents[agent_id] for agent_id in matching_agents if agent_id in self._agents]

    def find_agents_by_category(self, category: str) -> List[AgentRegistration]:
        """Find agents by category."""
        if category not in self._category_index:
            return []

        agent_ids = self._category_index[category]
        return [self._agents[agent_id] for agent_id in agent_ids if agent_id in self._agents]

    def find_compatible_agents(
        self,
        primary_agent_id: str,
        capability: AgentCapability
    ) -> List[AgentRegistration]:
        """Find agents compatible with a primary agent for collaboration."""
        if primary_agent_id not in self._agents:
            return []

        primary_agent = self._agents[primary_agent_id]
        candidates = self.find_agents_by_capability(capability)

        compatible_agents = []
        for candidate in candidates:
            if candidate.agent.agent_id == primary_agent_id:
                continue

            # Check collaboration history
            collaboration_score = self._calculate_collaboration_score(
                primary_agent, candidate
            )

            if collaboration_score > 0.5:  # Threshold for compatibility
                compatible_agents.append(candidate)

        # Sort by collaboration score
        compatible_agents.sort(
            key=lambda a: self._calculate_collaboration_score(primary_agent, a),
            reverse=True
        )

        return compatible_agents

    def _calculate_collaboration_score(
        self,
        agent1: AgentRegistration,
        agent2: AgentRegistration
    ) -> float:
        """Calculate collaboration compatibility score between two agents."""
        agent1_id = agent1.agent.agent_id
        agent2_id = agent2.agent.agent_id

        # Get collaboration history
        successful1 = agent1.successful_collaborations.get(agent2_id, 0)
        failed1 = agent1.failed_collaborations.get(agent2_id, 0)
        successful2 = agent2.successful_collaborations.get(agent1_id, 0)
        failed2 = agent2.failed_collaborations.get(agent1_id, 0)

        total_successful = successful1 + successful2
        total_failed = failed1 + failed2
        total_collaborations = total_successful + total_failed

        if total_collaborations == 0:
            return 0.7  # Default score for new collaborations

        success_rate = total_successful / total_collaborations

        # Adjust based on performance compatibility
        performance_diff = abs(
            agent1.get_performance_score() - agent2.get_performance_score()
        )
        performance_compatibility = 1.0 - min(performance_diff, 1.0)

        # Combine scores
        return (success_rate * 0.7) + (performance_compatibility * 0.3)

    def update_agent_performance(
        self,
        agent_id: str,
        execution_time_ms: float,
        success: bool,
        confidence: float = 0.0
    ) -> bool:
        """Update agent performance metrics after task completion."""
        if agent_id not in self._agents:
            return False

        registration = self._agents[agent_id]
        registration.performance_metrics.update_task_completion(
            execution_time_ms, success, confidence
        )

        # Update performance history
        if agent_id not in self._performance_history:
            self._performance_history[agent_id] = []

        self._performance_history[agent_id].append((
            datetime.now(timezone.utc),
            registration.get_performance_score()
        ))

        # Trim history
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self._performance_window_hours)
        self._performance_history[agent_id] = [
            (timestamp, score) for timestamp, score in self._performance_history[agent_id]
            if timestamp > cutoff_time
        ]

        return True

    def update_collaboration_result(
        self,
        agent1_id: str,
        agent2_id: str,
        success: bool
    ) -> bool:
        """Update collaboration results between two agents."""
        if agent1_id not in self._agents or agent2_id not in self._agents:
            return False

        agent1 = self._agents[agent1_id]
        agent2 = self._agents[agent2_id]

        if success:
            agent1.successful_collaborations[agent2_id] = (
                agent1.successful_collaborations.get(agent2_id, 0) + 1
            )
            agent2.successful_collaborations[agent1_id] = (
                agent2.successful_collaborations.get(agent1_id, 0) + 1
            )
        else:
            agent1.failed_collaborations[agent2_id] = (
                agent1.failed_collaborations.get(agent2_id, 0) + 1
            )
            agent2.failed_collaborations[agent1_id] = (
                agent2.failed_collaborations.get(agent1_id, 0) + 1
            )

        return True

    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent registration by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(
        self,
        capability_filter: Optional[AgentCapability] = None,
        status_filter: Optional[List[AgentStatus]] = None
    ) -> List[AgentRegistration]:
        """List all registered agents with optional filters."""
        agents = list(self._agents.values())
        
        if capability_filter:
            agents = [a for a in agents if capability_filter in a.agent.capabilities]
        
        if status_filter:
            agents = [a for a in agents if a.agent.status in status_filter]
        
        return agents
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        total_agents = len(self._agents)
        status_counts = {}
        capability_counts = {}
        category_counts = {}

        # Performance metrics
        total_tasks_completed = 0
        total_tasks_failed = 0
        total_execution_time = 0
        response_times = []

        for registration in self._agents.values():
            # Status distribution
            status = registration.agent.status
            status_counts[status] = status_counts.get(status, 0) + 1

            # Capability distribution
            for capability in registration.agent.capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1

            # Category distribution
            category = registration.metadata.category
            category_counts[category] = category_counts.get(category, 0) + 1

            # Performance aggregation
            metrics = registration.performance_metrics
            total_tasks_completed += metrics.total_tasks_completed
            total_tasks_failed += metrics.total_tasks_failed
            total_execution_time += metrics.total_execution_time_ms

            if metrics.recent_response_times:
                response_times.extend(metrics.recent_response_times)

        # Calculate aggregate performance metrics
        total_tasks = total_tasks_completed + total_tasks_failed
        success_rate = (total_tasks_completed / total_tasks) if total_tasks > 0 else 0.0
        avg_response_time = statistics.mean(response_times) if response_times else 0.0

        # Load balancing statistics
        load_distribution = {}
        for registration in self._agents.values():
            load_factor = registration.get_load_factor()
            load_bucket = f"{int(load_factor * 10) * 10}-{int(load_factor * 10) * 10 + 10}%"
            load_distribution[load_bucket] = load_distribution.get(load_bucket, 0) + 1

        # Recent selection statistics
        recent_selections = {}
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        for timestamp, agent_id, capability in self._selection_history:
            if timestamp > cutoff_time:
                recent_selections[agent_id] = recent_selections.get(agent_id, 0) + 1

        return {
            "total_agents": total_agents,
            "status_distribution": {k.value if hasattr(k, 'value') else str(k): v for k, v in status_counts.items()},
            "capability_distribution": {k.value if hasattr(k, 'value') else str(k): v for k, v in capability_counts.items()},
            "category_distribution": category_counts,
            "total_capabilities": len(self._capability_index),
            "total_tags": len(self._tag_index),
            "total_categories": len(self._category_index),
            "performance_metrics": {
                "total_tasks_completed": total_tasks_completed,
                "total_tasks_failed": total_tasks_failed,
                "overall_success_rate": success_rate,
                "average_response_time_ms": avg_response_time,
                "total_execution_time_ms": total_execution_time
            },
            "load_distribution": load_distribution,
            "recent_selections_last_hour": recent_selections,
            "load_balancing_strategy": self._load_balancing_strategy.value
        }
    
    async def _health_check_loop(self) -> None:
        """Enhanced background task for agent health checking."""
        while not self._shutdown:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _perform_health_checks(self) -> None:
        """Perform comprehensive health checks on all registered agents."""
        health_check_tasks = []

        for agent_id, registration in list(self._agents.items()):
            # Check if health check is due
            time_since_check = (
                datetime.now(timezone.utc) - registration.metadata.last_health_check
            ).total_seconds()

            if time_since_check >= registration.metadata.health_check_interval_seconds:
                # Create health check task
                task = asyncio.create_task(
                    self._perform_single_health_check(agent_id, registration)
                )
                health_check_tasks.append(task)

        # Execute health checks in parallel
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)

    async def _perform_single_health_check(
        self,
        agent_id: str,
        registration: AgentRegistration
    ) -> None:
        """Perform health check on a single agent."""
        try:
            # Perform health check with timeout
            is_healthy = await asyncio.wait_for(
                registration.agent.health_check(),
                timeout=registration.metadata.health_check_timeout_seconds
            )

            if is_healthy:
                # Reset failure counters on successful health check
                registration.consecutive_health_failures = 0
                registration.metadata.status = registration.agent.status

                # Update health score
                registration.health_score = min(registration.health_score + 0.1, 1.0)

                self._logger.debug(f"Health check passed for agent {agent_id}")
            else:
                await self._handle_health_check_failure(agent_id, registration, "Health check returned false")

        except asyncio.TimeoutError:
            await self._handle_health_check_failure(agent_id, registration, "Health check timeout")
        except Exception as e:
            await self._handle_health_check_failure(agent_id, registration, f"Health check exception: {e}")
        finally:
            registration.metadata.last_health_check = datetime.now(timezone.utc)

    async def _handle_health_check_failure(
        self,
        agent_id: str,
        registration: AgentRegistration,
        reason: str
    ) -> None:
        """Handle health check failure for an agent."""
        registration.consecutive_health_failures += 1
        registration.last_health_failure = datetime.now(timezone.utc)

        # Decrease health score
        registration.health_score = max(registration.health_score - 0.2, 0.0)

        self._logger.warning(
            f"Agent {agent_id} health check failed: {reason} "
            f"(consecutive failures: {registration.consecutive_health_failures})"
        )

        # Take action based on failure count
        if registration.consecutive_health_failures >= 3:
            registration.metadata.status = AgentStatus.ERROR
            self._logger.error(f"Agent {agent_id} marked as ERROR due to consecutive health check failures")

            # Optionally attempt to restart the agent
            if registration.consecutive_health_failures >= 5:
                self._logger.warning(f"Attempting to restart agent {agent_id}")
                await self._attempt_agent_restart(agent_id, registration)
        elif registration.consecutive_health_failures >= 1:
            # Mark as degraded but still usable
            if registration.agent.status == AgentStatus.READY:
                registration.metadata.status = AgentStatus.BUSY  # Temporary degradation

    async def _attempt_agent_restart(
        self,
        agent_id: str,
        registration: AgentRegistration
    ) -> bool:
        """Attempt to restart a failed agent."""
        try:
            self._logger.info(f"Attempting to restart agent {agent_id}")

            # Shutdown the agent
            await registration.agent.shutdown()

            # Wait a moment
            await asyncio.sleep(2)

            # Reinitialize the agent
            await registration.agent.initialize()

            # Reset failure counters
            registration.consecutive_health_failures = 0
            registration.health_score = 0.5  # Start with moderate health score

            self._logger.info(f"Successfully restarted agent {agent_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to restart agent {agent_id}: {e}")
            return False

    def get_agent_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report for all agents."""
        healthy_agents = 0
        degraded_agents = 0
        failed_agents = 0

        agent_details = {}

        for agent_id, registration in self._agents.items():
            health_info = {
                "status": registration.agent.status.value,
                "health_score": registration.health_score,
                "consecutive_failures": registration.consecutive_health_failures,
                "last_health_check": registration.metadata.last_health_check.isoformat(),
                "last_failure": registration.last_health_failure.isoformat() if registration.last_health_failure else None,
                "current_load": len(registration.current_tasks),
                "max_capacity": registration.metadata.max_concurrent_tasks,
                "load_factor": registration.get_load_factor(),
                "performance_score": registration.get_performance_score()
            }

            agent_details[agent_id] = health_info

            # Count by health status
            if registration.health_score >= 0.8 and registration.consecutive_health_failures == 0:
                healthy_agents += 1
            elif registration.health_score >= 0.5 or registration.consecutive_health_failures <= 2:
                degraded_agents += 1
            else:
                failed_agents += 1

        return {
            "summary": {
                "total_agents": len(self._agents),
                "healthy_agents": healthy_agents,
                "degraded_agents": degraded_agents,
                "failed_agents": failed_agents,
                "health_check_interval": self._health_check_interval
            },
            "agents": agent_details
        }


# Global registry instance
_agent_registry: Optional[AgentRegistry] = None


async def get_agent_registry() -> AgentRegistry:
    """Get or create the global agent registry instance."""
    global _agent_registry
    
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
        await _agent_registry.initialize()
    
    return _agent_registry


async def close_agent_registry() -> None:
    """Close the global agent registry instance."""
    global _agent_registry
    
    if _agent_registry:
        await _agent_registry.shutdown()
        _agent_registry = None
