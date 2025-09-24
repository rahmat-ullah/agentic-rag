"""
Enhanced Query Processing API Routes

This module provides API endpoints for enhanced query processing with procurement-specific
terminology expansion, context injection, RFQ hint processing, intent classification,
and search strategy selection.
"""

from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from agentic_rag.api.dependencies.auth import get_current_user, require_permission
from agentic_rag.api.models.auth import User
from agentic_rag.api.models.common import APIResponse, PaginatedResponse
from agentic_rag.services.enhanced_query_processor import (
    get_enhanced_query_processor, EnhancedQueryConfig, EnhancedProcessedQuery
)
from agentic_rag.services.procurement_terminology import (
    get_procurement_terminology_service, TerminologyExpansionConfig
)
from agentic_rag.services.context_injection import (
    get_context_injection_service, ContextInjectionConfig, UserRole, ProcessStage
)
from agentic_rag.services.rfq_hint_processor import (
    get_rfq_hint_processor, RFQHintConfig
)
from agentic_rag.services.query_intent_classifier import (
    get_query_intent_classifier, QueryIntentConfig
)
from agentic_rag.services.search_strategy_selector import (
    get_search_strategy_selector, SearchStrategyConfig
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/enhanced-query", tags=["Enhanced Query Processing"])


# Request/Response Models
class EnhancedQueryRequest(BaseModel):
    """Request model for enhanced query processing."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Query to process")
    user_role: Optional[UserRole] = Field(None, description="User role for context injection")
    process_stage: Optional[ProcessStage] = Field(None, description="Current process stage")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    config: Optional[EnhancedQueryConfig] = Field(None, description="Processing configuration")


class TerminologyExpansionRequest(BaseModel):
    """Request model for terminology expansion."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Query to expand")
    config: Optional[TerminologyExpansionConfig] = Field(None, description="Expansion configuration")


class ContextInjectionRequest(BaseModel):
    """Request model for context injection."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Query for context injection")
    user_role: Optional[UserRole] = Field(None, description="User role")
    process_stage: Optional[ProcessStage] = Field(None, description="Process stage")
    config: Optional[ContextInjectionConfig] = Field(None, description="Injection configuration")


class RFQHintRequest(BaseModel):
    """Request model for RFQ hint processing."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Query for hint processing")
    config: Optional[RFQHintConfig] = Field(None, description="Hint processing configuration")


class IntentClassificationRequest(BaseModel):
    """Request model for intent classification."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Query for intent classification")
    config: Optional[QueryIntentConfig] = Field(None, description="Classification configuration")


class StrategySelectionRequest(BaseModel):
    """Request model for strategy selection."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Query for strategy selection")
    procurement_intent: Optional[str] = Field(None, description="Procurement intent")
    config: Optional[SearchStrategyConfig] = Field(None, description="Selection configuration")


# API Endpoints
@router.post("/process", response_model=APIResponse[EnhancedProcessedQuery])
async def process_enhanced_query(
    request: EnhancedQueryRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permission("search:read"))
):
    """
    Process query with enhanced procurement-specific capabilities.
    
    This endpoint provides comprehensive query enhancement including:
    - Procurement terminology expansion
    - Context injection based on user role and process stage
    - RFQ hint processing for targeted search strategies
    - Intent classification for procurement scenarios
    - Intelligent search strategy selection
    """
    
    try:
        logger.info(
            f"Processing enhanced query",
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            query_length=len(request.query)
        )
        
        # Get enhanced query processor
        processor = await get_enhanced_query_processor()
        
        # Process the query
        result = await processor.process_enhanced_query(
            query=request.query,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.id),
            config=request.config,
            context={
                "user_role": request.user_role.value if request.user_role else None,
                "process_stage": request.process_stage.value if request.process_stage else None,
                **(request.context or {})
            }
        )
        
        logger.info(
            f"Enhanced query processing complete",
            processing_time_ms=result.processing_time_ms,
            enhancement_confidence=result.enhancement_confidence,
            terminology_expansions=len(result.terminology_expansions),
            context_injections=len(result.context_injections),
            rfq_hints=len(result.rfq_hints)
        )
        
        return APIResponse(
            success=True,
            data=result,
            message="Query processed successfully with enhanced capabilities"
        )
        
    except Exception as e:
        logger.error(f"Enhanced query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced query processing failed: {str(e)}"
        )


@router.post("/terminology/expand", response_model=APIResponse[Dict[str, Any]])
async def expand_terminology(
    request: TerminologyExpansionRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permission("search:read"))
):
    """
    Expand query with procurement-specific terminology.
    
    This endpoint provides terminology expansion including:
    - Synonym expansion for procurement terms
    - Acronym expansion (RFQ, SLA, etc.)
    - Domain-specific term enhancement
    - Confidence scoring for expansions
    """
    
    try:
        logger.info(
            f"Expanding terminology for query",
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            query=request.query[:100]
        )
        
        # Get terminology service
        terminology_service = get_procurement_terminology_service()
        
        # Expand terminology
        expanded_query, expansions = terminology_service.expand_query_terms(
            request.query,
            request.config or TerminologyExpansionConfig()
        )
        
        result = {
            "original_query": request.query,
            "expanded_query": expanded_query,
            "expansions": [
                {
                    "original_term": exp.original_term,
                    "expanded_term": exp.expanded_term,
                    "expansion_type": exp.expansion_type.value,
                    "category": exp.category.value,
                    "confidence": exp.confidence,
                    "context": exp.context
                }
                for exp in expansions
            ],
            "expansion_count": len(expansions)
        }
        
        return APIResponse(
            success=True,
            data=result,
            message=f"Terminology expanded with {len(expansions)} enhancements"
        )
        
    except Exception as e:
        logger.error(f"Terminology expansion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terminology expansion failed: {str(e)}"
        )


@router.post("/context/inject", response_model=APIResponse[Dict[str, Any]])
async def inject_context(
    request: ContextInjectionRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permission("search:read"))
):
    """
    Inject context into query based on user role and process stage.
    
    This endpoint provides context injection including:
    - Role-based context (procurement manager, technical evaluator, etc.)
    - Process stage context (requirements gathering, vendor evaluation, etc.)
    - Document type context
    - Dynamic context selection
    """
    
    try:
        logger.info(
            f"Injecting context for query",
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            user_role=request.user_role.value if request.user_role else None,
            process_stage=request.process_stage.value if request.process_stage else None
        )
        
        # Get context injection service
        context_service = get_context_injection_service()
        
        # Create config with user context
        config = request.config or ContextInjectionConfig()
        if request.user_role:
            config.user_role = request.user_role
        if request.process_stage:
            config.process_stage = request.process_stage
        
        # Inject context (we need to simulate query type and intent for this endpoint)
        from agentic_rag.services.query_processor import QueryType, QueryIntent
        enhanced_query, injected_contexts = context_service.inject_context(
            query=request.query,
            query_type=QueryType.PHRASE,  # Default for standalone context injection
            query_intent=QueryIntent.SEARCH,  # Default for standalone context injection
            config=config
        )
        
        result = {
            "original_query": request.query,
            "enhanced_query": enhanced_query,
            "injected_contexts": [
                {
                    "content": ctx.content,
                    "context_type": ctx.context_type.value,
                    "source_template": ctx.source_template,
                    "confidence": ctx.confidence,
                    "relevance_score": ctx.relevance_score
                }
                for ctx in injected_contexts
            ],
            "context_count": len(injected_contexts)
        }
        
        return APIResponse(
            success=True,
            data=result,
            message=f"Context injected with {len(injected_contexts)} enhancements"
        )
        
    except Exception as e:
        logger.error(f"Context injection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context injection failed: {str(e)}"
        )


@router.post("/rfq-hints/process", response_model=APIResponse[Dict[str, Any]])
async def process_rfq_hints(
    request: RFQHintRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permission("search:read"))
):
    """
    Process RFQ hints to guide targeted search strategies.

    This endpoint provides RFQ hint processing including:
    - Document type detection
    - Process stage identification
    - Vendor and timeline extraction
    - Search strategy suggestions
    """

    try:
        logger.info(
            f"Processing RFQ hints for query",
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            query=request.query[:100]
        )

        # Get RFQ hint processor
        hint_processor = get_rfq_hint_processor()

        # Process hints
        hints = hint_processor.extract_rfq_hints(
            request.query,
            request.config or RFQHintConfig()
        )

        result = {
            "query": request.query,
            "hints": [
                {
                    "hint_type": hint.hint_type.value,
                    "content": hint.content,
                    "confidence": hint.confidence.value,
                    "relevance_score": hint.relevance_score,
                    "context": hint.context,
                    "search_strategy_suggestions": hint.search_strategy_suggestions
                }
                for hint in hints
            ],
            "hint_count": len(hints),
            "high_confidence_hints": len([h for h in hints if h.confidence.value in ["high", "very_high"]])
        }

        return APIResponse(
            success=True,
            data=result,
            message=f"RFQ hints processed with {len(hints)} hints extracted"
        )

    except Exception as e:
        logger.error(f"RFQ hint processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RFQ hint processing failed: {str(e)}"
        )


@router.post("/intent/classify", response_model=APIResponse[Dict[str, Any]])
async def classify_intent(
    request: IntentClassificationRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permission("search:read"))
):
    """
    Classify query intent for procurement scenarios.

    This endpoint provides intent classification including:
    - Procurement-specific intent detection
    - Confidence scoring
    - Alternative intent suggestions
    - Classification explanation
    """

    try:
        logger.info(
            f"Classifying intent for query",
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            query=request.query[:100]
        )

        # Get intent classifier
        intent_classifier = get_query_intent_classifier()

        # Classify intent
        classification = intent_classifier.classify_intent(
            request.query,
            request.config or QueryIntentConfig()
        )

        result = {
            "query": request.query,
            "intent": classification.intent.value,
            "confidence": classification.confidence,
            "confidence_level": classification.confidence_level.value,
            "features_matched": classification.features_matched,
            "alternative_intents": [
                {
                    "intent": intent.value,
                    "confidence": confidence
                }
                for intent, confidence in classification.alternative_intents
            ],
            "explanation": classification.explanation
        }

        return APIResponse(
            success=True,
            data=result,
            message=f"Intent classified as {classification.intent.value} with {classification.confidence:.1%} confidence"
        )

    except Exception as e:
        logger.error(f"Intent classification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intent classification failed: {str(e)}"
        )


@router.post("/strategy/select", response_model=APIResponse[Dict[str, Any]])
async def select_strategy(
    request: StrategySelectionRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permission("search:read"))
):
    """
    Select optimal search strategy based on query analysis.

    This endpoint provides strategy selection including:
    - Intent-based strategy recommendation
    - Performance prediction
    - Alternative strategy suggestions
    - Strategy explanation and parameters
    """

    try:
        logger.info(
            f"Selecting strategy for query",
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            query=request.query[:100],
            procurement_intent=request.procurement_intent
        )

        # Get strategy selector
        strategy_selector = get_search_strategy_selector()

        # We need to simulate query type and intent for this endpoint
        from agentic_rag.services.query_processor import QueryType, QueryIntent

        # Select strategy
        selection = strategy_selector.select_strategy(
            query=request.query,
            query_type=QueryType.PHRASE,  # Default for standalone strategy selection
            query_intent=QueryIntent.SEARCH,  # Default for standalone strategy selection
            procurement_intent=request.procurement_intent or "general_search",
            config=request.config or SearchStrategyConfig()
        )

        result = {
            "query": request.query,
            "strategy": selection.strategy.value,
            "confidence": selection.confidence,
            "confidence_level": selection.confidence_level.value,
            "reasoning": selection.reasoning,
            "alternative_strategies": [
                {
                    "strategy": strategy.value,
                    "confidence": confidence
                }
                for strategy, confidence in selection.alternative_strategies
            ],
            "parameters": selection.parameters,
            "performance_prediction": selection.performance_prediction
        }

        return APIResponse(
            success=True,
            data=result,
            message=f"Strategy selected: {selection.strategy.value} with {selection.confidence:.1%} confidence"
        )

    except Exception as e:
        logger.error(f"Strategy selection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy selection failed: {str(e)}"
        )


@router.get("/statistics", response_model=APIResponse[Dict[str, Any]])
async def get_statistics(
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permission("search:read"))
):
    """
    Get enhanced query processing statistics.

    This endpoint provides comprehensive statistics including:
    - Processing performance metrics
    - Enhancement effectiveness
    - Intent classification accuracy
    - Strategy selection distribution
    """

    try:
        logger.info(
            f"Getting enhanced query processing statistics",
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )

        # Get services
        processor = await get_enhanced_query_processor()
        terminology_service = get_procurement_terminology_service()
        context_service = get_context_injection_service()
        hint_processor = get_rfq_hint_processor()
        intent_classifier = get_query_intent_classifier()
        strategy_selector = get_search_strategy_selector()

        # Collect statistics
        result = {
            "enhanced_query_processor": processor.get_statistics(),
            "terminology_service": terminology_service.get_statistics(),
            "context_injection": context_service.get_statistics(),
            "rfq_hint_processor": hint_processor.get_statistics(),
            "intent_classifier": intent_classifier.get_statistics(),
            "strategy_selector": strategy_selector.get_statistics()
        }

        return APIResponse(
            success=True,
            data=result,
            message="Enhanced query processing statistics retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )
