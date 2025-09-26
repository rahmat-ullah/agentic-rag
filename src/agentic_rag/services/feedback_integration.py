"""
Feedback Integration Service for Sprint 6

This service integrates the feedback collection system with existing services
including search results, document linking, and agent orchestration framework.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy.orm import Session

from agentic_rag.services.feedback_service import FeedbackService, get_feedback_service
from agentic_rag.schemas.feedback import (
    FeedbackSubmissionRequest,
    FeedbackTypeEnum,
    FeedbackCategoryEnum
)
from agentic_rag.api.models.search import SearchResponse, SearchResultItem
from agentic_rag.services.orchestration.integration import IntegratedWorkflowResult
from agentic_rag.models.database import DocumentLink
from agentic_rag.database.connection import get_database_session

logger = structlog.get_logger(__name__)


@dataclass
class FeedbackContext:
    """Context information for feedback submission."""
    user_id: uuid.UUID
    tenant_id: uuid.UUID
    session_id: Optional[str] = None
    query: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FeedbackIntegrationService:
    """Service for integrating feedback with existing system components."""
    
    def __init__(self, db_session: Session, feedback_service: FeedbackService):
        self.db = db_session
        self.feedback_service = feedback_service
        self.logger = logger.bind(service="feedback_integration")
    
    async def collect_search_result_feedback(
        self,
        search_response: SearchResponse,
        result_item: SearchResultItem,
        rating: int,
        context: FeedbackContext,
        feedback_text: Optional[str] = None,
        category: Optional[FeedbackCategoryEnum] = None
    ) -> uuid.UUID:
        """Collect feedback on a specific search result."""
        try:
            # Create feedback submission
            feedback_data = FeedbackSubmissionRequest(
                feedback_type=FeedbackTypeEnum.SEARCH_RESULT,
                feedback_category=category,
                target_id=result_item.chunk_id,
                target_type="search_result",
                rating=rating,
                feedback_text=feedback_text,
                query=context.query,
                session_id=context.session_id,
                context_metadata={
                    "search_request_id": search_response.request_id,
                    "result_rank": result_item.rank,
                    "relevance_score": result_item.relevance_score,
                    "document_id": str(result_item.document_id),
                    "document_title": result_item.document_title,
                    "chunk_text_preview": result_item.text[:200] if result_item.text else None,
                    **(context.metadata or {})
                }
            )
            
            # Submit feedback
            result = await self.feedback_service.submit_feedback(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                feedback_data=feedback_data,
                session_id=context.session_id,
                user_agent=context.user_agent
            )
            
            self.logger.info(
                "search_result_feedback_collected",
                feedback_id=result.feedback_id,
                chunk_id=result_item.chunk_id,
                rating=rating,
                user_id=context.user_id
            )
            
            return result.feedback_id
            
        except Exception as e:
            self.logger.error("search_result_feedback_failed", error=str(e), chunk_id=result_item.chunk_id)
            raise
    
    async def collect_document_link_feedback(
        self,
        link_id: uuid.UUID,
        rating: int,
        context: FeedbackContext,
        feedback_text: Optional[str] = None,
        category: Optional[FeedbackCategoryEnum] = None
    ) -> uuid.UUID:
        """Collect feedback on document link quality."""
        try:
            # Get link information
            link = self.db.query(DocumentLink).filter(DocumentLink.id == link_id).first()
            if not link:
                raise ValueError(f"Document link {link_id} not found")
            
            # Create feedback submission
            feedback_data = FeedbackSubmissionRequest(
                feedback_type=FeedbackTypeEnum.LINK_QUALITY,
                feedback_category=category,
                target_id=link_id,
                target_type="document_link",
                rating=rating,
                feedback_text=feedback_text,
                query=context.query,
                session_id=context.session_id,
                context_metadata={
                    "rfq_id": str(link.rfq_id),
                    "offer_id": str(link.offer_id),
                    "link_type": link.link_type,
                    "confidence": link.confidence,
                    "quality_score": link.quality_score,
                    **(context.metadata or {})
                }
            )
            
            # Submit feedback
            result = await self.feedback_service.submit_feedback(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                feedback_data=feedback_data,
                session_id=context.session_id,
                user_agent=context.user_agent
            )
            
            self.logger.info(
                "document_link_feedback_collected",
                feedback_id=result.feedback_id,
                link_id=link_id,
                rating=rating,
                user_id=context.user_id
            )
            
            return result.feedback_id
            
        except Exception as e:
            self.logger.error("document_link_feedback_failed", error=str(e), link_id=link_id)
            raise
    
    async def collect_answer_quality_feedback(
        self,
        workflow_result: IntegratedWorkflowResult,
        rating: int,
        context: FeedbackContext,
        feedback_text: Optional[str] = None,
        category: Optional[FeedbackCategoryEnum] = None
    ) -> uuid.UUID:
        """Collect feedback on answer quality from agent orchestration."""
        try:
            # Create feedback submission
            feedback_data = FeedbackSubmissionRequest(
                feedback_type=FeedbackTypeEnum.ANSWER_QUALITY,
                feedback_category=category,
                target_id=uuid.UUID(workflow_result.request_id) if workflow_result.request_id else None,
                target_type="answer",
                rating=rating,
                feedback_text=feedback_text,
                query=workflow_result.original_query,
                session_id=context.session_id,
                context_metadata={
                    "query_intent": workflow_result.query_intent.value,
                    "agents_used": workflow_result.agents_used,
                    "confidence_score": workflow_result.confidence_score,
                    "execution_time_ms": workflow_result.execution_time_ms,
                    "citations_count": len(workflow_result.citations),
                    "pii_detected": workflow_result.pii_detected,
                    "redactions_applied": workflow_result.redactions_applied,
                    "has_pricing_analysis": workflow_result.pricing_analysis is not None,
                    "has_document_comparison": workflow_result.document_comparison is not None,
                    "has_risk_assessment": workflow_result.risk_assessment is not None,
                    "has_compliance_check": workflow_result.compliance_check is not None,
                    **(context.metadata or {})
                }
            )
            
            # Submit feedback
            result = await self.feedback_service.submit_feedback(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                feedback_data=feedback_data,
                session_id=context.session_id,
                user_agent=context.user_agent
            )
            
            self.logger.info(
                "answer_quality_feedback_collected",
                feedback_id=result.feedback_id,
                request_id=workflow_result.request_id,
                rating=rating,
                user_id=context.user_id
            )
            
            return result.feedback_id
            
        except Exception as e:
            self.logger.error("answer_quality_feedback_failed", error=str(e), request_id=workflow_result.request_id)
            raise
    
    async def collect_bulk_search_feedback(
        self,
        search_response: SearchResponse,
        feedback_items: List[Dict[str, Any]],
        context: FeedbackContext
    ) -> List[uuid.UUID]:
        """Collect feedback on multiple search results in bulk."""
        feedback_ids = []
        
        try:
            for feedback_item in feedback_items:
                chunk_id = feedback_item.get("chunk_id")
                rating = feedback_item.get("rating")
                feedback_text = feedback_item.get("feedback_text")
                category = feedback_item.get("category")
                
                if not chunk_id or rating is None:
                    continue
                
                # Find the result item
                result_item = None
                for item in search_response.results:
                    if item.chunk_id == chunk_id:
                        result_item = item
                        break
                
                if not result_item:
                    self.logger.warning("search_result_not_found", chunk_id=chunk_id)
                    continue
                
                # Collect feedback for this item
                feedback_id = await self.collect_search_result_feedback(
                    search_response=search_response,
                    result_item=result_item,
                    rating=rating,
                    context=context,
                    feedback_text=feedback_text,
                    category=FeedbackCategoryEnum(category) if category else None
                )
                
                feedback_ids.append(feedback_id)
            
            self.logger.info(
                "bulk_search_feedback_collected",
                feedback_count=len(feedback_ids),
                user_id=context.user_id
            )
            
            return feedback_ids
            
        except Exception as e:
            self.logger.error("bulk_search_feedback_failed", error=str(e), user_id=context.user_id)
            raise
    
    async def enhance_search_results_with_feedback(
        self,
        search_response: SearchResponse,
        tenant_id: uuid.UUID
    ) -> SearchResponse:
        """Enhance search results with aggregated feedback information."""
        try:
            # Get feedback aggregations for all result chunks
            chunk_ids = [result.chunk_id for result in search_response.results]
            
            # Query feedback aggregations
            from agentic_rag.models.feedback import FeedbackAggregation, FeedbackType
            aggregations = self.db.query(FeedbackAggregation).filter(
                FeedbackAggregation.tenant_id == tenant_id,
                FeedbackAggregation.target_id.in_(chunk_ids),
                FeedbackAggregation.target_type == "search_result",
                FeedbackAggregation.feedback_type == FeedbackType.SEARCH_RESULT
            ).all()
            
            # Create aggregation lookup
            aggregation_map = {agg.target_id: agg for agg in aggregations}
            
            # Enhance results with feedback data
            for result in search_response.results:
                agg = aggregation_map.get(result.chunk_id)
                if agg:
                    # Add feedback metadata to result
                    if not hasattr(result, 'feedback_metadata'):
                        result.feedback_metadata = {}
                    
                    result.feedback_metadata.update({
                        "total_feedback": agg.total_feedback_count,
                        "positive_feedback": agg.positive_count,
                        "negative_feedback": agg.negative_count,
                        "average_rating": agg.average_rating,
                        "quality_score": agg.quality_score,
                        "confidence_score": agg.confidence_score
                    })
            
            self.logger.info(
                "search_results_enhanced_with_feedback",
                results_count=len(search_response.results),
                enhanced_count=len([r for r in search_response.results if hasattr(r, 'feedback_metadata')])
            )
            
            return search_response
            
        except Exception as e:
            self.logger.error("search_enhancement_failed", error=str(e))
            return search_response  # Return original response on error
    
    async def get_feedback_insights_for_query(
        self,
        query: str,
        tenant_id: uuid.UUID,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get feedback insights for similar queries."""
        try:
            from agentic_rag.models.feedback import UserFeedbackSubmission
            
            # Find feedback for similar queries (simple text matching for now)
            similar_feedback = self.db.query(UserFeedbackSubmission).filter(
                UserFeedbackSubmission.tenant_id == tenant_id,
                UserFeedbackSubmission.query.ilike(f"%{query}%")
            ).limit(limit).all()
            
            if not similar_feedback:
                return {"insights": [], "patterns": {}}
            
            # Analyze patterns
            patterns = {
                "common_issues": {},
                "average_satisfaction": 0.0,
                "improvement_areas": []
            }
            
            total_rating = 0
            rating_count = 0
            
            for feedback in similar_feedback:
                if feedback.rating is not None:
                    total_rating += feedback.rating
                    rating_count += 1
                
                if feedback.feedback_category:
                    category = feedback.feedback_category.value
                    patterns["common_issues"][category] = patterns["common_issues"].get(category, 0) + 1
            
            if rating_count > 0:
                patterns["average_satisfaction"] = total_rating / rating_count
            
            # Identify improvement areas
            if patterns["common_issues"]:
                top_issues = sorted(patterns["common_issues"].items(), key=lambda x: x[1], reverse=True)
                patterns["improvement_areas"] = [issue[0] for issue in top_issues[:3]]
            
            insights = [
                {
                    "feedback_id": str(feedback.id),
                    "rating": feedback.rating,
                    "category": feedback.feedback_category.value if feedback.feedback_category else None,
                    "text": feedback.feedback_text[:200] if feedback.feedback_text else None,
                    "created_at": feedback.created_at.isoformat()
                }
                for feedback in similar_feedback
            ]
            
            return {
                "insights": insights,
                "patterns": patterns,
                "total_similar_feedback": len(similar_feedback)
            }
            
        except Exception as e:
            self.logger.error("feedback_insights_failed", error=str(e), query=query)
            return {"insights": [], "patterns": {}}


# Dependency injection
_feedback_integration_service = None


def get_feedback_integration_service(
    db_session: Session = None,
    feedback_service: FeedbackService = None
) -> FeedbackIntegrationService:
    """Get feedback integration service instance."""
    global _feedback_integration_service
    
    if _feedback_integration_service is None:
        if db_session is None:
            db_session = get_database_session()
        if feedback_service is None:
            feedback_service = get_feedback_service(db_session)
        
        _feedback_integration_service = FeedbackIntegrationService(db_session, feedback_service)
    
    return _feedback_integration_service
