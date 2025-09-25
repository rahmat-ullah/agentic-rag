"""
Enhanced Agent Integration Module

This module provides enhanced integration between the orchestration framework
and all completed Sprint 5 services, creating end-to-end workflows that
demonstrate the complete system functionality.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from .base import AgentCapability, Task, TaskPriority, Context, Result
from .registry import get_agent_registry
from .planner import get_planner_agent, QueryIntent, QueryComplexity
from .orchestrator import AgentOrchestrator, OrchestrationResult
from .specialized_agents import (
    create_synthesizer_agent,
    create_pricing_agent,
    create_analysis_agent,
    create_redaction_agent,
    create_retriever_agent
)

logger = structlog.get_logger(__name__)


class IntegratedWorkflowResult(BaseModel):
    """Result from an integrated end-to-end workflow."""
    
    # Request information
    request_id: str = Field(..., description="Request identifier")
    original_query: str = Field(..., description="Original user query")
    query_intent: QueryIntent = Field(..., description="Classified query intent")
    
    # Workflow execution
    workflow_steps: List[Dict[str, Any]] = Field(default_factory=list)
    agents_used: List[str] = Field(default_factory=list)
    
    # Final results
    success: bool = Field(..., description="Whether workflow succeeded")
    final_answer: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    redacted_content: Optional[str] = None
    
    # Analysis results (if applicable)
    pricing_analysis: Optional[Dict[str, Any]] = None
    document_comparison: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    compliance_check: Optional[Dict[str, Any]] = None
    
    # Quality and performance
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    execution_time_ms: int = Field(..., description="Total execution time")
    
    # Privacy and security
    pii_detected: int = Field(default=0, description="Number of PII items detected")
    redactions_applied: int = Field(default=0, description="Number of redactions applied")
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        use_enum_values = True


class EnhancedOrchestrator:
    """Enhanced orchestrator with integrated Sprint 5 functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._logger = structlog.get_logger(__name__)
        
        # Core components
        self._orchestrator = None
        self._registry = None
        self._planner = None
        
        # Specialized agents
        self._synthesizer_agent = None
        self._pricing_agent = None
        self._analysis_agent = None
        self._redaction_agent = None
        self._retriever_agent = None
        
        # State tracking
        self._active_workflows: Dict[str, IntegratedWorkflowResult] = {}
        
    async def initialize(self) -> None:
        """Initialize the enhanced orchestrator."""
        self._logger.info("Initializing enhanced orchestrator")
        
        # Initialize core components
        self._orchestrator = AgentOrchestrator()
        await self._orchestrator.initialize()
        
        self._registry = await get_agent_registry()
        self._planner = await get_planner_agent()
        
        # Create and register specialized agents
        await self._initialize_specialized_agents()
        
        self._logger.info("Enhanced orchestrator initialized")
    
    async def _initialize_specialized_agents(self) -> None:
        """Initialize and register all specialized agents."""
        self._logger.info("Initializing specialized agents")
        
        # Create agents
        self._synthesizer_agent = await create_synthesizer_agent()
        self._pricing_agent = await create_pricing_agent()
        self._analysis_agent = await create_analysis_agent()
        self._redaction_agent = await create_redaction_agent()
        self._retriever_agent = await create_retriever_agent()
        
        # Register agents with the registry
        agents = [
            self._synthesizer_agent,
            self._pricing_agent,
            self._analysis_agent,
            self._redaction_agent,
            self._retriever_agent
        ]
        
        for agent in agents:
            await self._registry.register_agent(agent)
            self._logger.info(f"Registered agent: {agent.name}")
    
    async def process_integrated_query(
        self,
        query: str,
        user_id: UUID,
        tenant_id: UUID,
        user_role: str = "viewer",
        context_data: Optional[Dict[str, Any]] = None
    ) -> IntegratedWorkflowResult:
        """Process a query through the complete integrated system."""
        start_time = datetime.now(timezone.utc)
        request_id = str(uuid4())
        
        self._logger.info(f"Processing integrated query: {request_id}")
        
        try:
            # Create context
            context = Context(
                user_id=user_id,
                tenant_id=tenant_id,
                original_query=query,
                user_context={
                    "role": user_role,
                    **(context_data or {})
                }
            )
            
            # Step 1: Query Analysis
            analysis_result = await self._analyze_query(query, context)
            query_intent = analysis_result["intent"]
            
            # Step 2: Document Retrieval
            retrieval_result = await self._retrieve_documents(query, context)
            
            # Step 3: Intent-specific Processing
            processing_result = await self._process_by_intent(
                query_intent, query, retrieval_result, context
            )
            
            # Step 4: Answer Synthesis
            synthesis_result = await self._synthesize_answer(
                query, retrieval_result, processing_result, context
            )
            
            # Step 5: Privacy Protection
            redaction_result = await self._apply_privacy_protection(
                synthesis_result, context
            )
            
            # Calculate execution time
            execution_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            
            # Create integrated result
            result = IntegratedWorkflowResult(
                request_id=request_id,
                original_query=query,
                query_intent=QueryIntent(query_intent),
                workflow_steps=[
                    {"step": "query_analysis", "result": analysis_result},
                    {"step": "document_retrieval", "result": {"count": len(retrieval_result.get("chunks", []))}},
                    {"step": "intent_processing", "result": processing_result},
                    {"step": "answer_synthesis", "result": {"answer_length": len(synthesis_result.get("answer", ""))}},
                    {"step": "privacy_protection", "result": {"redactions": redaction_result.get("redactions_applied", 0)}}
                ],
                agents_used=[
                    "planner_agent",
                    "retriever_agent",
                    self._get_processing_agent_name(query_intent),
                    "synthesizer_agent",
                    "redaction_agent"
                ],
                success=True,
                final_answer=redaction_result.get("protected_content", synthesis_result.get("answer")),
                citations=synthesis_result.get("citations", []),
                redacted_content=redaction_result.get("protected_content"),
                confidence_score=synthesis_result.get("quality_scores", {}).get("overall_score", 0.8),
                execution_time_ms=execution_time,
                pii_detected=redaction_result.get("pii_detected", 0),
                redactions_applied=redaction_result.get("redactions_applied", 0)
            )
            
            # Add intent-specific results
            self._add_intent_specific_results(result, query_intent, processing_result)
            
            self._active_workflows[request_id] = result
            self._logger.info(f"Integrated query processing completed: {request_id}")
            
            return result
            
        except Exception as e:
            execution_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self._logger.error(f"Integrated query processing failed: {e}")
            
            return IntegratedWorkflowResult(
                request_id=request_id,
                original_query=query,
                query_intent=QueryIntent.INFORMATION_SEEKING,
                success=False,
                execution_time_ms=execution_time
            )
    
    async def _analyze_query(self, query: str, context: Context) -> Dict[str, Any]:
        """Analyze query using the planner agent."""
        task = Task(
            id=str(uuid4()),
            capability_required=AgentCapability.QUERY_ANALYSIS,
            input_data={"query": query},
            priority=TaskPriority.HIGH
        )
        
        result = await self._planner.execute(task, context)
        if result.success:
            return result.data["analysis"]
        else:
            raise Exception(f"Query analysis failed: {result.error}")
    
    async def _retrieve_documents(self, query: str, context: Context) -> Dict[str, Any]:
        """Retrieve relevant documents."""
        task = Task(
            id=str(uuid4()),
            capability_required=AgentCapability.DOCUMENT_SEARCH,
            input_data={"query": query},
            priority=TaskPriority.HIGH
        )
        
        result = await self._retriever_agent.execute(task, context)
        if result.success:
            return result.data
        else:
            raise Exception(f"Document retrieval failed: {result.error}")
    
    async def _process_by_intent(
        self,
        intent: str,
        query: str,
        retrieval_result: Dict[str, Any],
        context: Context
    ) -> Dict[str, Any]:
        """Process query based on classified intent."""
        chunks = retrieval_result.get("chunks", [])
        content = "\n".join([chunk.get("content", "") for chunk in chunks])
        
        if intent == QueryIntent.PRICING_INQUIRY:
            return await self._process_pricing_query(content, context)
        elif intent == QueryIntent.COMPARISON:
            return await self._process_comparison_query(content, context)
        elif intent == QueryIntent.RISK_ASSESSMENT:
            return await self._process_risk_query(content, context)
        elif intent == QueryIntent.COMPLIANCE_CHECK:
            return await self._process_compliance_query(content, context)
        elif intent == QueryIntent.SUMMARIZATION:
            return await self._process_summarization_query(content, context)
        else:
            # Default processing - no special analysis needed
            return {"processed": True, "content": content}
    
    async def _process_pricing_query(self, content: str, context: Context) -> Dict[str, Any]:
        """Process pricing-related queries."""
        task = Task(
            id=str(uuid4()),
            capability_required=AgentCapability.PRICING_ANALYSIS,
            input_data={"content": content},
            priority=TaskPriority.HIGH
        )
        
        result = await self._pricing_agent.execute(task, context)
        return result.data if result.success else {}
    
    async def _process_comparison_query(self, content: str, context: Context) -> Dict[str, Any]:
        """Process document comparison queries."""
        # For simplicity, assume we're comparing the first two documents
        chunks = content.split("\n\n")
        if len(chunks) >= 2:
            task = Task(
                id=str(uuid4()),
                capability_required=AgentCapability.DOCUMENT_COMPARISON,
                input_data={"document1": chunks[0], "document2": chunks[1]},
                priority=TaskPriority.HIGH
            )
            
            result = await self._analysis_agent.execute(task, context)
            return result.data if result.success else {}
        return {}
    
    async def _process_risk_query(self, content: str, context: Context) -> Dict[str, Any]:
        """Process risk assessment queries."""
        task = Task(
            id=str(uuid4()),
            capability_required=AgentCapability.RISK_ASSESSMENT,
            input_data={"content": content},
            priority=TaskPriority.HIGH
        )
        
        result = await self._analysis_agent.execute(task, context)
        return result.data if result.success else {}
    
    async def _process_compliance_query(self, content: str, context: Context) -> Dict[str, Any]:
        """Process compliance checking queries."""
        task = Task(
            id=str(uuid4()),
            capability_required=AgentCapability.COMPLIANCE_CHECKING,
            input_data={"content": content},
            priority=TaskPriority.HIGH
        )
        
        result = await self._analysis_agent.execute(task, context)
        return result.data if result.success else {}
    
    async def _process_summarization_query(self, content: str, context: Context) -> Dict[str, Any]:
        """Process content summarization queries."""
        task = Task(
            id=str(uuid4()),
            capability_required=AgentCapability.CONTENT_SUMMARIZATION,
            input_data={"content": content},
            priority=TaskPriority.HIGH
        )
        
        result = await self._analysis_agent.execute(task, context)
        return result.data if result.success else {}
    
    async def _synthesize_answer(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        processing_result: Dict[str, Any],
        context: Context
    ) -> Dict[str, Any]:
        """Synthesize final answer with citations."""
        task = Task(
            id=str(uuid4()),
            capability_required=AgentCapability.ANSWER_SYNTHESIS,
            input_data={
                "query": query,
                "retrieved_chunks": retrieval_result.get("chunks", []),
                "processing_result": processing_result
            },
            priority=TaskPriority.HIGH
        )
        
        result = await self._synthesizer_agent.execute(task, context)
        return result.data if result.success else {}
    
    async def _apply_privacy_protection(
        self,
        synthesis_result: Dict[str, Any],
        context: Context
    ) -> Dict[str, Any]:
        """Apply privacy protection and redaction."""
        answer = synthesis_result.get("answer", "")
        
        task = Task(
            id=str(uuid4()),
            capability_required=AgentCapability.PRIVACY_PROTECTION,
            input_data={"content": answer},
            priority=TaskPriority.HIGH
        )
        
        result = await self._redaction_agent.execute(task, context)
        return result.data if result.success else {"protected_content": answer}
    
    def _get_processing_agent_name(self, intent: str) -> str:
        """Get the name of the agent used for intent-specific processing."""
        if intent in [QueryIntent.PRICING_INQUIRY]:
            return "pricing_agent"
        elif intent in [QueryIntent.COMPARISON, QueryIntent.RISK_ASSESSMENT, 
                       QueryIntent.COMPLIANCE_CHECK, QueryIntent.SUMMARIZATION]:
            return "analysis_agent"
        else:
            return "general_agent"
    
    def _add_intent_specific_results(
        self,
        result: IntegratedWorkflowResult,
        intent: str,
        processing_result: Dict[str, Any]
    ) -> None:
        """Add intent-specific results to the workflow result."""
        if intent == QueryIntent.PRICING_INQUIRY:
            result.pricing_analysis = processing_result
        elif intent == QueryIntent.COMPARISON:
            result.document_comparison = processing_result
        elif intent == QueryIntent.RISK_ASSESSMENT:
            result.risk_assessment = processing_result
        elif intent == QueryIntent.COMPLIANCE_CHECK:
            result.compliance_check = processing_result
    
    async def shutdown(self) -> None:
        """Shutdown the enhanced orchestrator."""
        self._logger.info("Shutting down enhanced orchestrator")
        if self._orchestrator:
            await self._orchestrator.shutdown()


# Global instance
_enhanced_orchestrator: Optional[EnhancedOrchestrator] = None


async def get_enhanced_orchestrator() -> EnhancedOrchestrator:
    """Get or create the global enhanced orchestrator instance."""
    global _enhanced_orchestrator
    if _enhanced_orchestrator is None:
        _enhanced_orchestrator = EnhancedOrchestrator()
        await _enhanced_orchestrator.initialize()
    return _enhanced_orchestrator


async def close_enhanced_orchestrator() -> None:
    """Close the global enhanced orchestrator instance."""
    global _enhanced_orchestrator
    if _enhanced_orchestrator is not None:
        await _enhanced_orchestrator.shutdown()
        _enhanced_orchestrator = None
