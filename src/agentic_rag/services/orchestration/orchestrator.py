"""
Agent Orchestrator

This module implements the main agent orchestrator that coordinates
multiple specialized agents to handle complex queries. It integrates
all orchestration components and provides the main entry point.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
import structlog

from .base import AgentCapability, Task, TaskPriority, Context, Result
from .registry import get_agent_registry
from .communication import get_communication_framework
from .workflow import get_workflow_engine, WorkflowExecution
from .planner import get_planner_agent, QueryAnalysis, ExecutionPlan

logger = structlog.get_logger(__name__)


class OrchestrationConfig(BaseModel):
    """Configuration for agent orchestration."""
    
    # Execution settings
    max_concurrent_workflows: int = Field(default=10, description="Maximum concurrent workflows")
    default_timeout_seconds: int = Field(default=300, description="Default workflow timeout")
    enable_parallel_execution: bool = Field(default=True, description="Enable parallel task execution")
    
    # Quality settings
    min_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_fallback_strategies: bool = Field(default=True)
    
    # Monitoring settings
    enable_performance_monitoring: bool = Field(default=True)
    enable_detailed_logging: bool = Field(default=False)
    
    # Error handling
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts")
    enable_circuit_breaker: bool = Field(default=True)
    
    class Config:
        use_enum_values = True


class OrchestrationResult(BaseModel):
    """Result from agent orchestration."""
    
    # Request information
    request_id: str = Field(..., description="Request identifier")
    original_query: str = Field(..., description="Original user query")
    
    # Execution information
    workflow_execution_id: str = Field(..., description="Workflow execution ID")
    execution_plan: ExecutionPlan = Field(..., description="Execution plan used")
    
    # Results
    success: bool = Field(..., description="Whether orchestration succeeded")
    final_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Performance metrics
    total_execution_time_ms: int = Field(..., description="Total execution time")
    steps_completed: int = Field(default=0)
    steps_failed: int = Field(default=0)
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Timing
    started_at: datetime = Field(..., description="Orchestration start time")
    completed_at: datetime = Field(..., description="Orchestration completion time")
    
    class Config:
        use_enum_values = True


class AgentOrchestrator:
    """Main orchestrator for coordinating multiple agents."""
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config or OrchestrationConfig()
        
        # Component references (will be initialized)
        self._registry = None
        self._communication = None
        self._workflow_engine = None
        self._planner = None
        
        # State tracking
        self._active_workflows: Dict[str, WorkflowExecution] = {}
        self._orchestration_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_execution_time_ms": 0.0,
        }
        
        self._logger = structlog.get_logger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and all components."""
        self._logger.info("Initializing agent orchestrator")
        
        # Initialize components
        self._registry = await get_agent_registry()
        self._communication = await get_communication_framework()
        self._workflow_engine = await get_workflow_engine()
        self._planner = await get_planner_agent()
        
        # Register planner agent
        await self._registry.register_agent(self._planner)
        
        self._logger.info("Agent orchestrator initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator and all components."""
        self._logger.info("Shutting down agent orchestrator")
        
        # Cancel active workflows
        for workflow_id in list(self._active_workflows.keys()):
            await self._workflow_engine.cancel_execution(workflow_id)
        
        self._active_workflows.clear()
        
        self._logger.info("Agent orchestrator shutdown complete")
    
    async def process_query(
        self,
        query: str,
        user_id: UUID,
        tenant_id: UUID,
        context_data: Optional[Dict[str, Any]] = None
    ) -> OrchestrationResult:
        """Process a user query through the agent orchestration system."""
        start_time = datetime.now(timezone.utc)
        request_id = str(uuid4())
        
        self._logger.info(f"Processing query: {query[:100]}...", request_id=request_id)
        
        try:
            # Create execution context
            context = Context(
                request_id=request_id,
                user_id=user_id,
                tenant_id=tenant_id,
                original_query=query,
                config=context_data or {}
            )
            
            # Step 1: Analyze query
            query_analysis = await self._analyze_query(query, context)
            
            # Step 2: Plan workflow
            execution_plan = await self._plan_workflow(query_analysis, context)
            
            # Step 3: Execute workflow
            workflow_execution = await self._execute_workflow(execution_plan, context)
            
            # Step 4: Collect results
            final_result = await self._collect_results(workflow_execution)
            
            # Calculate metrics
            end_time = datetime.now(timezone.utc)
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Create orchestration result
            result = OrchestrationResult(
                request_id=request_id,
                original_query=query,
                workflow_execution_id=workflow_execution.id,
                execution_plan=execution_plan,
                success=True,
                final_result=final_result,
                total_execution_time_ms=execution_time_ms,
                steps_completed=workflow_execution.steps_completed,
                steps_failed=workflow_execution.steps_failed,
                confidence_score=self._calculate_confidence(workflow_execution),
                started_at=start_time,
                completed_at=end_time
            )
            
            # Update statistics
            self._update_stats(result)
            
            self._logger.info(
                f"Query processing completed successfully",
                request_id=request_id,
                execution_time_ms=execution_time_ms
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._logger.error(f"Query processing failed: {e}", request_id=request_id)
            
            # Create error result
            result = OrchestrationResult(
                request_id=request_id,
                original_query=query,
                workflow_execution_id="",
                execution_plan=ExecutionPlan(
                    query_analysis=QueryAnalysis(
                        original_query=query,
                        processed_query=query,
                        query_type="unknown",
                        intent="unknown",
                        complexity_score=0.0,
                        estimated_steps=0,
                        required_capabilities=[],
                        confidence_score=0.0
                    ),
                    workflow=None,
                    workflow_id="error_workflow",
                    estimated_execution_time_ms=0,
                    required_agents=[],
                    expected_confidence=0.0
                ),
                success=False,
                error=str(e),
                total_execution_time_ms=execution_time_ms,
                started_at=start_time,
                completed_at=end_time
            )
            
            self._update_stats(result)
            return result
    
    async def _analyze_query(self, query: str, context: Context) -> QueryAnalysis:
        """Analyze the user query."""
        analysis_task = Task(
            name="query_analysis",
            description="Analyze user query",
            capability_required=AgentCapability.QUERY_ANALYSIS,
            priority=TaskPriority.HIGH,
            input_data={"query": query}
        )
        
        result = await self._planner.execute(analysis_task, context)
        
        if not result.success:
            raise Exception(f"Query analysis failed: {result.error}")
        
        return QueryAnalysis(**result.data["analysis"])
    
    async def _plan_workflow(self, analysis: QueryAnalysis, context: Context) -> ExecutionPlan:
        """Plan workflow based on query analysis."""
        planning_task = Task(
            name="workflow_planning",
            description="Plan workflow execution",
            capability_required=AgentCapability.WORKFLOW_PLANNING,
            priority=TaskPriority.HIGH,
            input_data={"analysis": analysis.dict()}
        )
        
        result = await self._planner.execute(planning_task, context)
        
        if not result.success:
            raise Exception(f"Workflow planning failed: {result.error}")
        
        return ExecutionPlan(**result.data["plan"])
    
    async def _execute_workflow(
        self,
        plan: ExecutionPlan,
        context: Context
    ) -> WorkflowExecution:
        """Execute the planned workflow."""
        # Create workflow definition from plan data
        from .workflow import WorkflowDefinition, WorkflowStep

        # Convert workflow data back to WorkflowDefinition for the engine
        workflow_steps = []
        if plan.workflow:
            for step_data in plan.workflow.get("steps", []):
                # Create WorkflowStep from dict data
                step = WorkflowStep(**step_data)
                workflow_steps.append(step)

        workflow_def = WorkflowDefinition(
            id=plan.workflow_id,
            name=plan.workflow.get("name", "generated_workflow") if plan.workflow else "generated_workflow",
            description=plan.workflow.get("description", "") if plan.workflow else "",
            steps=workflow_steps,
            timeout_seconds=plan.workflow.get("timeout_seconds") if plan.workflow else self.config.default_timeout_seconds
        )

        # Register workflow with engine
        self._workflow_engine.register_workflow(workflow_def)

        # Execute workflow
        execution = await self._workflow_engine.execute_workflow(
            plan.workflow_id,
            context
        )
        
        # Track active workflow
        self._active_workflows[execution.id] = execution
        
        # Wait for completion (with timeout)
        timeout_seconds = workflow_def.timeout_seconds or self.config.default_timeout_seconds
        
        try:
            # Poll for completion
            while execution.status.value in ["pending", "running"]:
                await asyncio.sleep(1)
                execution = await self._workflow_engine.get_execution_status(execution.id)
                
                if not execution:
                    raise Exception("Workflow execution lost")
                
                # Check timeout
                if execution.started_at:
                    elapsed = (datetime.now(timezone.utc) - execution.started_at).total_seconds()
                    if elapsed > timeout_seconds:
                        await self._workflow_engine.cancel_execution(execution.id)
                        raise Exception("Workflow execution timed out")
            
            return execution
            
        finally:
            # Remove from active workflows
            if execution.id in self._active_workflows:
                del self._active_workflows[execution.id]
    
    async def _collect_results(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Collect and aggregate results from workflow execution."""
        if execution.status.value != "completed":
            raise Exception(f"Workflow execution failed: {execution.error}")
        
        # For now, return a simple aggregation
        # In a full implementation, this would intelligently combine results
        final_result = {
            "execution_id": execution.id,
            "status": execution.status,
            "steps_completed": execution.steps_completed,
            "step_results": execution.step_results,
            "total_time_ms": execution.total_execution_time_ms,
        }
        
        return final_result
    
    def _calculate_confidence(self, execution: WorkflowExecution) -> float:
        """Calculate overall confidence score for the execution."""
        if not execution.step_results:
            return 0.0
        
        # Average confidence from all step results
        confidence_scores = []
        for result in execution.step_results.values():
            if result.confidence_score is not None:
                confidence_scores.append(result.confidence_score)
        
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.5  # Default confidence
    
    def _update_stats(self, result: OrchestrationResult) -> None:
        """Update orchestration statistics."""
        self._orchestration_stats["total_requests"] += 1
        
        if result.success:
            self._orchestration_stats["successful_requests"] += 1
        else:
            self._orchestration_stats["failed_requests"] += 1
        
        # Update average execution time
        total_requests = self._orchestration_stats["total_requests"]
        current_avg = self._orchestration_stats["average_execution_time_ms"]
        new_avg = ((current_avg * (total_requests - 1)) + result.total_execution_time_ms) / total_requests
        self._orchestration_stats["average_execution_time_ms"] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        return {
            "orchestration": self._orchestration_stats.copy(),
            "active_workflows": len(self._active_workflows),
            "registry": self._registry.get_registry_stats() if self._registry else {},
            "communication": (
                self._communication.get_communication_stats() 
                if self._communication else {}
            ),
        }


# Global orchestrator instance
_agent_orchestrator: Optional[AgentOrchestrator] = None


async def get_agent_orchestrator(
    config: Optional[OrchestrationConfig] = None
) -> AgentOrchestrator:
    """Get or create the global agent orchestrator instance."""
    global _agent_orchestrator
    
    if _agent_orchestrator is None:
        _agent_orchestrator = AgentOrchestrator(config)
        await _agent_orchestrator.initialize()
    
    return _agent_orchestrator


async def close_agent_orchestrator() -> None:
    """Close the global agent orchestrator instance."""
    global _agent_orchestrator
    
    if _agent_orchestrator:
        await _agent_orchestrator.shutdown()
        _agent_orchestrator = None
