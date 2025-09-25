"""
API models for agent orchestration endpoints.

This module defines the request and response models for the agent
orchestration API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from agentic_rag.api.models.responses import BaseResponse


class QueryRequest(BaseModel):
    """Request model for query processing."""
    
    query: str = Field(..., description="User query to process", min_length=1, max_length=2000)
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context for query processing"
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration options for orchestration"
    )


class QueryAnalysisResponse(BaseModel):
    """Response model for query analysis."""
    
    original_query: str = Field(..., description="Original user query")
    processed_query: str = Field(..., description="Processed/normalized query")
    query_type: str = Field(..., description="Type of query")
    intent: str = Field(..., description="Query intent")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Query complexity (0-1)")
    estimated_steps: int = Field(..., description="Estimated number of steps required")
    required_capabilities: List[str] = Field(..., description="Required agent capabilities")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")


class WorkflowStepResult(BaseModel):
    """Result from a workflow step."""
    
    step_id: str = Field(..., description="Step identifier")
    step_name: str = Field(..., description="Step name")
    status: str = Field(..., description="Step status")
    agent_id: Optional[str] = Field(None, description="Agent that executed the step")
    execution_time_ms: Optional[int] = Field(None, description="Step execution time")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    error: Optional[str] = Field(None, description="Error message if step failed")


class QueryResponse(BaseResponse):
    """Response model for query processing."""
    
    request_id: str = Field(..., description="Request identifier")
    workflow_execution_id: str = Field(..., description="Workflow execution ID")
    
    # Query analysis
    query_analysis: QueryAnalysisResponse = Field(..., description="Query analysis results")
    
    # Execution results
    final_result: Optional[Dict[str, Any]] = Field(None, description="Final processed result")
    step_results: List[WorkflowStepResult] = Field(
        default_factory=list,
        description="Results from individual workflow steps"
    )
    
    # Performance metrics
    total_execution_time_ms: int = Field(..., description="Total execution time")
    steps_completed: int = Field(default=0, description="Number of steps completed")
    steps_failed: int = Field(default=0, description="Number of steps failed")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence")
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality metrics"
    )


class AgentStatusInfo(BaseModel):
    """Information about an agent's status."""
    
    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Current status")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    current_tasks: int = Field(default=0, description="Number of current tasks")
    tasks_completed: int = Field(default=0, description="Total tasks completed")
    tasks_failed: int = Field(default=0, description="Total tasks failed")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    average_execution_time_ms: float = Field(default=0.0, description="Average execution time")
    last_activity: datetime = Field(..., description="Last activity timestamp")


class OrchestrationStatsResponse(BaseResponse):
    """Response model for orchestration statistics."""
    
    # Overall statistics
    total_requests: int = Field(default=0, description="Total requests processed")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall success rate")
    average_execution_time_ms: float = Field(default=0.0, description="Average execution time")
    
    # Active state
    active_workflows: int = Field(default=0, description="Currently active workflows")
    registered_agents: int = Field(default=0, description="Number of registered agents")
    
    # Agent information
    agents: List[AgentStatusInfo] = Field(
        default_factory=list,
        description="Information about registered agents"
    )
    
    # Capability distribution
    capability_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of agent capabilities"
    )


class WorkflowExecutionInfo(BaseModel):
    """Information about a workflow execution."""
    
    execution_id: str = Field(..., description="Execution identifier")
    workflow_id: str = Field(..., description="Workflow definition ID")
    status: str = Field(..., description="Execution status")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    total_execution_time_ms: Optional[int] = Field(None, description="Total execution time")
    steps_completed: int = Field(default=0, description="Steps completed")
    steps_failed: int = Field(default=0, description="Steps failed")
    error: Optional[str] = Field(None, description="Error message if failed")


class WorkflowExecutionsResponse(BaseResponse):
    """Response model for listing workflow executions."""
    
    executions: List[WorkflowExecutionInfo] = Field(
        default_factory=list,
        description="List of workflow executions"
    )
    total_count: int = Field(default=0, description="Total number of executions")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=20, description="Page size")


class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration."""
    
    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    version: str = Field(default="1.0.0", description="Agent version")
    max_concurrent_tasks: int = Field(default=1, description="Maximum concurrent tasks")
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Agent configuration"
    )


class AgentRegistrationResponse(BaseResponse):
    """Response model for agent registration."""
    
    agent_id: str = Field(..., description="Registered agent identifier")
    registration_status: str = Field(..., description="Registration status")


class HealthCheckResponse(BaseResponse):
    """Response model for orchestration health check."""
    
    status: str = Field(..., description="Overall health status")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components"
    )
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    version: str = Field(default="1.0.0", description="System version")


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    component: Optional[str] = Field(None, description="Component that generated the error")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional error context")


class OrchestrationErrorResponse(BaseResponse):
    """Response model for orchestration errors."""
    
    error_details: List[ErrorDetail] = Field(
        default_factory=list,
        description="Detailed error information"
    )
    request_id: Optional[str] = Field(None, description="Request identifier")
    workflow_execution_id: Optional[str] = Field(None, description="Workflow execution ID")
    failed_step: Optional[str] = Field(None, description="Step that failed")
    recovery_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for error recovery"
    )
