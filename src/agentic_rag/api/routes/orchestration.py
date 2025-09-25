"""
API routes for agent orchestration.

This module provides REST endpoints for agent orchestration functionality,
including query processing, workflow management, and system monitoring.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
import structlog

from agentic_rag.api.dependencies.auth import get_current_user, get_current_tenant_id
from agentic_rag.api.dependencies.database import get_db_session
from agentic_rag.api.models.orchestration import (
    QueryRequest,
    QueryResponse,
    QueryAnalysisResponse,
    WorkflowStepResult,
    OrchestrationStatsResponse,
    AgentStatusInfo,
    WorkflowExecutionsResponse,
    WorkflowExecutionInfo,
    AgentRegistrationRequest,
    AgentRegistrationResponse,
    HealthCheckResponse,
    OrchestrationErrorResponse,
    ErrorDetail,
)
from agentic_rag.api.models.responses import BaseResponse
from agentic_rag.api.models.users import User
from agentic_rag.services.orchestration import (
    get_agent_orchestrator,
    get_agent_registry,
    OrchestrationConfig,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Process a query through agent orchestration",
    description="Process a user query using the agent orchestration system"
)
async def process_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    tenant_id: UUID = Depends(get_current_tenant_id),
    db_session: Session = Depends(get_db_session)
) -> QueryResponse:
    """Process a query through the agent orchestration system."""
    try:
        logger.info(
            f"Processing query for user {current_user.id}",
            query_length=len(request.query),
            tenant_id=str(tenant_id)
        )
        
        # Get orchestrator
        orchestrator = await get_agent_orchestrator()
        
        # Process query
        result = await orchestrator.process_query(
            query=request.query,
            user_id=current_user.id,
            tenant_id=tenant_id,
            context_data=request.context
        )
        
        # Convert to API response format
        query_analysis = QueryAnalysisResponse(
            original_query=result.execution_plan.query_analysis.original_query,
            processed_query=result.execution_plan.query_analysis.processed_query,
            query_type=result.execution_plan.query_analysis.query_type,
            intent=result.execution_plan.query_analysis.intent,
            complexity_score=result.execution_plan.query_analysis.complexity_score,
            estimated_steps=result.execution_plan.query_analysis.estimated_steps,
            required_capabilities=[cap.value for cap in result.execution_plan.query_analysis.required_capabilities],
            confidence_score=result.execution_plan.query_analysis.confidence_score
        )
        
        # Convert step results
        step_results = []
        if result.final_result and "step_results" in result.final_result:
            for step_id, step_result in result.final_result["step_results"].items():
                step_results.append(WorkflowStepResult(
                    step_id=step_id,
                    step_name=step_result.get("name", step_id),
                    status="completed" if step_result.success else "failed",
                    agent_id=step_result.agent_id,
                    execution_time_ms=step_result.execution_time_ms,
                    confidence_score=step_result.confidence_score,
                    error=step_result.error
                ))
        
        return QueryResponse(
            success=result.success,
            message="Query processed successfully" if result.success else "Query processing failed",
            request_id=result.request_id,
            workflow_execution_id=result.workflow_execution_id,
            query_analysis=query_analysis,
            final_result=result.final_result,
            step_results=step_results,
            total_execution_time_ms=result.total_execution_time_ms,
            steps_completed=result.steps_completed,
            steps_failed=result.steps_failed,
            confidence_score=result.confidence_score,
            quality_metrics=result.quality_metrics
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=OrchestrationStatsResponse,
    summary="Get orchestration statistics",
    description="Get statistics about the orchestration system"
)
async def get_orchestration_stats(
    current_user: User = Depends(get_current_user),
    tenant_id: UUID = Depends(get_current_tenant_id)
) -> OrchestrationStatsResponse:
    """Get orchestration system statistics."""
    try:
        # Get orchestrator and registry
        orchestrator = await get_agent_orchestrator()
        registry = await get_agent_registry()
        
        # Get stats
        stats = orchestrator.get_stats()
        
        # Convert agent information
        agents = []
        for registration in registry.list_agents():
            agent = registration.agent
            metrics = agent.get_performance_metrics()
            
            agents.append(AgentStatusInfo(
                agent_id=agent.agent_id,
                name=agent.name,
                status=agent.status.value,
                capabilities=[cap.value for cap in agent.capabilities],
                current_tasks=len(registration.current_tasks),
                tasks_completed=metrics["tasks_completed"],
                tasks_failed=metrics["tasks_failed"],
                success_rate=metrics["success_rate"],
                average_execution_time_ms=metrics["average_execution_time_ms"],
                last_activity=datetime.fromisoformat(metrics["last_activity"].replace('Z', '+00:00'))
            ))
        
        # Calculate success rate
        total_requests = stats["orchestration"]["total_requests"]
        successful_requests = stats["orchestration"]["successful_requests"]
        success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
        
        return OrchestrationStatsResponse(
            success=True,
            message="Statistics retrieved successfully",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=stats["orchestration"]["failed_requests"],
            success_rate=success_rate,
            average_execution_time_ms=stats["orchestration"]["average_execution_time_ms"],
            active_workflows=stats["active_workflows"],
            registered_agents=stats["registry"]["total_agents"],
            agents=agents,
            capability_distribution=stats["registry"]["capability_distribution"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get orchestration stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get(
    "/workflows",
    response_model=WorkflowExecutionsResponse,
    summary="List workflow executions",
    description="List recent workflow executions"
)
async def list_workflow_executions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_user),
    tenant_id: UUID = Depends(get_current_tenant_id)
) -> WorkflowExecutionsResponse:
    """List workflow executions with pagination."""
    try:
        # For now, return empty list as we don't have persistent storage
        # In a full implementation, this would query the database
        
        return WorkflowExecutionsResponse(
            success=True,
            message="Workflow executions retrieved successfully",
            executions=[],
            total_count=0,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list workflow executions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflow executions: {str(e)}"
        )


@router.get(
    "/workflows/{execution_id}",
    response_model=WorkflowExecutionInfo,
    summary="Get workflow execution details",
    description="Get details about a specific workflow execution"
)
async def get_workflow_execution(
    execution_id: str,
    current_user: User = Depends(get_current_user),
    tenant_id: UUID = Depends(get_current_tenant_id)
) -> WorkflowExecutionInfo:
    """Get details about a specific workflow execution."""
    try:
        # Get workflow engine
        from agentic_rag.services.orchestration.workflow import get_workflow_engine
        
        workflow_engine = await get_workflow_engine()
        execution = await workflow_engine.get_execution_status(execution_id)
        
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow execution {execution_id} not found"
            )
        
        return WorkflowExecutionInfo(
            execution_id=execution.id,
            workflow_id=execution.workflow_id,
            status=execution.status.value,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            total_execution_time_ms=execution.total_execution_time_ms,
            steps_completed=execution.steps_completed,
            steps_failed=execution.steps_failed,
            error=execution.error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow execution: {str(e)}"
        )


@router.post(
    "/agents/register",
    response_model=AgentRegistrationResponse,
    summary="Register a new agent",
    description="Register a new agent with the orchestration system"
)
async def register_agent(
    request: AgentRegistrationRequest,
    current_user: User = Depends(get_current_user),
    tenant_id: UUID = Depends(get_current_tenant_id)
) -> AgentRegistrationResponse:
    """Register a new agent with the orchestration system."""
    try:
        # This would be implemented when we have dynamic agent registration
        # For now, return a placeholder response
        
        return AgentRegistrationResponse(
            success=True,
            message="Agent registration not yet implemented",
            agent_id=request.agent_id,
            registration_status="pending"
        )
        
    except Exception as e:
        logger.error(f"Failed to register agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register agent: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check the health of the orchestration system"
)
async def health_check() -> HealthCheckResponse:
    """Check the health of the orchestration system."""
    try:
        # Check component health
        components = {}
        
        try:
            orchestrator = await get_agent_orchestrator()
            components["orchestrator"] = "healthy"
        except Exception:
            components["orchestrator"] = "unhealthy"
        
        try:
            registry = await get_agent_registry()
            components["registry"] = "healthy"
        except Exception:
            components["registry"] = "unhealthy"
        
        # Determine overall status
        overall_status = "healthy" if all(
            status == "healthy" for status in components.values()
        ) else "unhealthy"
        
        return HealthCheckResponse(
            success=True,
            message=f"System is {overall_status}",
            status=overall_status,
            components=components,
            uptime_seconds=0.0,  # Would be calculated from system start time
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            success=False,
            message=f"Health check failed: {str(e)}",
            status="unhealthy",
            components={},
            uptime_seconds=0.0,
            version="1.0.0"
        )
