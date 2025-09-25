"""
Integrated Query Processing API Endpoint

This endpoint demonstrates the complete Sprint 5 functionality by processing
queries through the enhanced orchestration framework with all integrated services.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from agentic_rag.api.dependencies.auth import get_current_user
from agentic_rag.api.models.responses import APIResponse
from agentic_rag.models.database import User
from agentic_rag.services.orchestration.integration import (
    get_enhanced_orchestrator,
    IntegratedWorkflowResult
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/integrated", tags=["Integrated Query Processing"])


class IntegratedQueryRequest(BaseModel):
    """Request for integrated query processing."""
    
    query: str = Field(..., description="User query to process")
    user_role: str = Field(default="viewer", description="User role for privacy policies")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Additional context data")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Compare the pricing strategies in the procurement documents and assess any compliance risks",
                "user_role": "analyst",
                "context_data": {
                    "department": "procurement",
                    "project": "vendor_analysis"
                }
            }
        }


class IntegratedQueryResponse(BaseModel):
    """Response from integrated query processing."""
    
    # Request information
    request_id: str = Field(..., description="Request identifier")
    original_query: str = Field(..., description="Original user query")
    query_intent: str = Field(..., description="Classified query intent")
    
    # Results
    success: bool = Field(..., description="Whether processing succeeded")
    final_answer: Optional[str] = Field(None, description="Final synthesized answer")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations")
    
    # Analysis results (when applicable)
    pricing_analysis: Optional[Dict[str, Any]] = Field(None, description="Pricing analysis results")
    document_comparison: Optional[Dict[str, Any]] = Field(None, description="Document comparison results")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk assessment results")
    compliance_check: Optional[Dict[str, Any]] = Field(None, description="Compliance check results")
    
    # Workflow information
    workflow_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Executed workflow steps")
    agents_used: List[str] = Field(default_factory=list, description="Agents used in processing")
    
    # Quality and performance metrics
    confidence_score: float = Field(..., description="Overall confidence score")
    execution_time_ms: int = Field(..., description="Total execution time")
    
    # Privacy and security
    pii_detected: int = Field(default=0, description="Number of PII items detected")
    redactions_applied: int = Field(default=0, description="Number of redactions applied")
    privacy_protected: bool = Field(default=False, description="Whether privacy protection was applied")
    
    # Metadata
    processed_at: datetime = Field(..., description="Processing timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "req_123456",
                "original_query": "Compare pricing strategies and assess compliance risks",
                "query_intent": "comparison",
                "success": True,
                "final_answer": "Based on the analysis of procurement documents...",
                "citations": [
                    {
                        "source": "procurement_doc_1.pdf",
                        "page": 5,
                        "text": "Pricing strategy overview..."
                    }
                ],
                "document_comparison": {
                    "differences_found": 12,
                    "key_changes": ["pricing_model", "terms"]
                },
                "confidence_score": 0.87,
                "execution_time_ms": 3450,
                "pii_detected": 2,
                "redactions_applied": 2,
                "privacy_protected": True,
                "processed_at": "2024-01-15T10:30:00Z"
            }
        }


@router.post(
    "/query",
    response_model=APIResponse[IntegratedQueryResponse],
    summary="Process Integrated Query",
    description="Process a query through the complete integrated Sprint 5 system with orchestration, analysis, synthesis, and privacy protection"
)
async def process_integrated_query(
    request: IntegratedQueryRequest,
    current_user: User = Depends(get_current_user)
) -> APIResponse[IntegratedQueryResponse]:
    """
    Process a query through the complete integrated system.
    
    This endpoint demonstrates the full Sprint 5 functionality:
    1. Query analysis and intent classification
    2. Document retrieval with semantic search
    3. Intent-specific processing (pricing, comparison, risk, compliance)
    4. Answer synthesis with citations
    5. Privacy protection and redaction
    
    The system automatically determines the appropriate workflow based on
    the query intent and applies the relevant specialized agents.
    """
    try:
        logger.info(f"Processing integrated query for user {current_user.id}")
        
        # Get the enhanced orchestrator
        orchestrator = await get_enhanced_orchestrator()
        
        # Process the query through the integrated system
        result = await orchestrator.process_integrated_query(
            query=request.query,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            user_role=request.user_role,
            context_data=request.context_data
        )
        
        # Convert to response format
        response_data = IntegratedQueryResponse(
            request_id=result.request_id,
            original_query=result.original_query,
            query_intent=result.query_intent.value,
            success=result.success,
            final_answer=result.final_answer,
            citations=result.citations,
            pricing_analysis=result.pricing_analysis,
            document_comparison=result.document_comparison,
            risk_assessment=result.risk_assessment,
            compliance_check=result.compliance_check,
            workflow_steps=result.workflow_steps,
            agents_used=result.agents_used,
            confidence_score=result.confidence_score,
            execution_time_ms=result.execution_time_ms,
            pii_detected=result.pii_detected,
            redactions_applied=result.redactions_applied,
            privacy_protected=result.redactions_applied > 0,
            processed_at=result.created_at
        )
        
        logger.info(f"Integrated query processing completed: {result.request_id}")
        
        return APIResponse(
            success=True,
            data=response_data,
            message="Query processed successfully through integrated system"
        )
        
    except Exception as e:
        logger.error(f"Integrated query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process integrated query: {str(e)}"
        )


@router.get(
    "/workflow/{request_id}",
    response_model=APIResponse[IntegratedQueryResponse],
    summary="Get Workflow Result",
    description="Retrieve the result of a previously processed integrated query"
)
async def get_workflow_result(
    request_id: str,
    current_user: User = Depends(get_current_user)
) -> APIResponse[IntegratedQueryResponse]:
    """Get the result of a previously processed integrated query."""
    try:
        orchestrator = await get_enhanced_orchestrator()
        
        # Get workflow result from orchestrator
        if request_id in orchestrator._active_workflows:
            result = orchestrator._active_workflows[request_id]
            
            response_data = IntegratedQueryResponse(
                request_id=result.request_id,
                original_query=result.original_query,
                query_intent=result.query_intent.value,
                success=result.success,
                final_answer=result.final_answer,
                citations=result.citations,
                pricing_analysis=result.pricing_analysis,
                document_comparison=result.document_comparison,
                risk_assessment=result.risk_assessment,
                compliance_check=result.compliance_check,
                workflow_steps=result.workflow_steps,
                agents_used=result.agents_used,
                confidence_score=result.confidence_score,
                execution_time_ms=result.execution_time_ms,
                pii_detected=result.pii_detected,
                redactions_applied=result.redactions_applied,
                privacy_protected=result.redactions_applied > 0,
                processed_at=result.created_at
            )
            
            return APIResponse(
                success=True,
                data=response_data,
                message="Workflow result retrieved successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow result not found for request ID: {request_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve workflow result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve workflow result: {str(e)}"
        )


@router.get(
    "/capabilities",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get System Capabilities",
    description="Get information about the integrated system capabilities and available agents"
)
async def get_system_capabilities(
    current_user: User = Depends(get_current_user)
) -> APIResponse[Dict[str, Any]]:
    """Get information about the integrated system capabilities."""
    try:
        orchestrator = await get_enhanced_orchestrator()
        registry = orchestrator._registry
        
        # Get registry statistics
        stats = registry.get_registry_stats()
        
        capabilities_info = {
            "total_agents": stats["total_agents"],
            "available_capabilities": [
                "query_analysis",
                "document_retrieval", 
                "answer_synthesis",
                "citation_generation",
                "pricing_analysis",
                "competitive_analysis",
                "document_comparison",
                "content_summarization",
                "table_extraction",
                "compliance_checking",
                "risk_assessment",
                "pii_detection",
                "privacy_protection"
            ],
            "supported_query_intents": [
                "information_seeking",
                "comparison",
                "analysis",
                "extraction",
                "summarization",
                "pricing_inquiry",
                "compliance_check",
                "risk_assessment",
                "document_search",
                "relationship_discovery"
            ],
            "workflow_features": [
                "automatic_intent_classification",
                "parallel_task_execution",
                "intelligent_agent_selection",
                "quality_assessment",
                "privacy_protection",
                "citation_generation",
                "performance_monitoring"
            ],
            "registry_stats": stats
        }
        
        return APIResponse(
            success=True,
            data=capabilities_info,
            message="System capabilities retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve system capabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system capabilities: {str(e)}"
        )
