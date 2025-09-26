"""
Feedback Service for Sprint 6 Story 6-01: Feedback Collection System

This service handles feedback submission, processing, validation, and aggregation
to support continuous learning and system improvement.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from agentic_rag.models.feedback import (
    UserFeedbackSubmission,
    FeedbackAggregation,
    FeedbackImpact,
    FeedbackSession,
    FeedbackType,
    FeedbackCategory,
    FeedbackStatus,
    FeedbackPriority
)
from agentic_rag.schemas.feedback import (
    FeedbackSubmissionRequest,
    ThumbsFeedbackRequest,
    DetailedFeedbackRequest,
    FeedbackDetails,
    FeedbackTypeEnum,
    FeedbackStatusEnum,
    FeedbackCategoryEnum
)
from agentic_rag.schemas.base import PaginationParams, SortParams
from agentic_rag.database.connection import get_database_session
from agentic_rag.api.exceptions import ValidationError, ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class FeedbackSubmissionResult:
    """Result of feedback submission."""
    feedback_id: uuid.UUID
    status: FeedbackStatusEnum
    estimated_processing_time: str
    confirmation_message: str


@dataclass
class FeedbackListResult:
    """Result of feedback list query."""
    items: List[FeedbackDetails]
    total_count: int
    total_pages: int
    has_next: bool
    has_previous: bool


@dataclass
class FeedbackStats:
    """Aggregated feedback statistics."""
    total_submissions: int
    pending_count: int
    processed_count: int
    average_rating: Optional[float]
    category_breakdown: Dict[str, int]
    recent_trends: Dict[str, Any]


class FeedbackService:
    """Service for managing feedback collection and processing."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logger.bind(service="feedback_service")
    
    async def submit_feedback(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        feedback_data: FeedbackSubmissionRequest,
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> FeedbackSubmissionResult:
        """Submit general feedback with validation and processing."""
        try:
            # Validate feedback data
            await self._validate_feedback_submission(feedback_data, tenant_id)
            
            # Create feedback submission
            feedback = UserFeedbackSubmission(
                tenant_id=tenant_id,
                user_id=user_id,
                feedback_type=FeedbackType(feedback_data.feedback_type.value),
                feedback_category=FeedbackCategory(feedback_data.feedback_category.value) if feedback_data.feedback_category else None,
                target_id=feedback_data.target_id,
                target_type=feedback_data.target_type,
                rating=feedback_data.rating,
                feedback_text=feedback_data.feedback_text,
                query=feedback_data.query,
                session_id=session_id,
                context_metadata=feedback_data.context_metadata,
                status=FeedbackStatus.PENDING,
                priority=self._determine_priority(feedback_data)
            )
            
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            
            # Update session tracking
            await self._update_feedback_session(tenant_id, user_id, session_id, user_agent)
            
            # Update aggregation (async)
            await self._update_feedback_aggregation(feedback)
            
            # Determine processing time estimate
            estimated_time = self._estimate_processing_time(feedback.priority, feedback.feedback_type)
            
            # Generate confirmation message
            confirmation_message = self._generate_confirmation_message(feedback)
            
            self.logger.info(
                "feedback_submitted_successfully",
                feedback_id=feedback.id,
                feedback_type=feedback.feedback_type,
                priority=feedback.priority,
                tenant_id=tenant_id,
                user_id=user_id
            )
            
            return FeedbackSubmissionResult(
                feedback_id=feedback.id,
                status=FeedbackStatusEnum(feedback.status.value),
                estimated_processing_time=estimated_time,
                confirmation_message=confirmation_message
            )
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("feedback_submission_failed", error=str(e), tenant_id=tenant_id, user_id=user_id)
            raise ServiceError(f"Failed to submit feedback: {str(e)}")
    
    async def submit_thumbs_feedback(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        feedback_data: FeedbackSubmissionRequest,
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> FeedbackSubmissionResult:
        """Submit simple thumbs up/down feedback optimized for speed."""
        try:
            # Determine feedback type based on target
            feedback_type = self._determine_feedback_type_from_target(feedback_data.target_type)
            
            # Create simplified feedback submission
            feedback = UserFeedbackSubmission(
                tenant_id=tenant_id,
                user_id=user_id,
                feedback_type=feedback_type,
                target_id=feedback_data.target_id,
                target_type=feedback_data.target_type,
                rating=feedback_data.rating,
                query=feedback_data.query,
                session_id=session_id,
                status=FeedbackStatus.PENDING,
                priority=FeedbackPriority.LOW  # Thumbs feedback is typically low priority
            )
            
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            
            # Update session tracking
            await self._update_feedback_session(tenant_id, user_id, session_id, user_agent)
            
            # Update aggregation immediately for thumbs feedback
            await self._update_feedback_aggregation(feedback)
            
            self.logger.info(
                "thumbs_feedback_submitted",
                feedback_id=feedback.id,
                rating=feedback.rating,
                target_type=feedback.target_type,
                tenant_id=tenant_id,
                user_id=user_id
            )
            
            return FeedbackSubmissionResult(
                feedback_id=feedback.id,
                status=FeedbackStatusEnum.PENDING,
                estimated_processing_time="Immediate",
                confirmation_message="Thank you for your feedback!"
            )
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("thumbs_feedback_failed", error=str(e), tenant_id=tenant_id, user_id=user_id)
            raise ServiceError(f"Failed to submit thumbs feedback: {str(e)}")
    
    async def submit_detailed_feedback(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        feedback_data: DetailedFeedbackRequest,
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> FeedbackSubmissionResult:
        """Submit detailed feedback with comprehensive information."""
        try:
            # Construct detailed feedback text
            detailed_text = self._construct_detailed_feedback_text(feedback_data)
            
            # Create detailed feedback submission
            feedback = UserFeedbackSubmission(
                tenant_id=tenant_id,
                user_id=user_id,
                feedback_type=FeedbackType(feedback_data.feedback_type.value),
                feedback_category=FeedbackCategory(feedback_data.feedback_category.value),
                target_id=feedback_data.target_id,
                target_type=feedback_data.target_type,
                feedback_text=detailed_text,
                query=feedback_data.query,
                session_id=session_id,
                context_metadata=feedback_data.context_metadata,
                status=FeedbackStatus.PENDING,
                priority=FeedbackPriority(feedback_data.priority.value)
            )
            
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            
            # Update session tracking
            await self._update_feedback_session(tenant_id, user_id, session_id, user_agent)
            
            # Determine processing time based on priority
            estimated_time = self._estimate_processing_time(feedback.priority, feedback.feedback_type)
            
            # Generate confirmation message
            confirmation_message = self._generate_detailed_confirmation_message(feedback_data)
            
            self.logger.info(
                "detailed_feedback_submitted",
                feedback_id=feedback.id,
                feedback_type=feedback.feedback_type,
                category=feedback.feedback_category,
                priority=feedback.priority,
                tenant_id=tenant_id,
                user_id=user_id
            )
            
            return FeedbackSubmissionResult(
                feedback_id=feedback.id,
                status=FeedbackStatusEnum.PENDING,
                estimated_processing_time=estimated_time,
                confirmation_message=confirmation_message
            )
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("detailed_feedback_failed", error=str(e), tenant_id=tenant_id, user_id=user_id)
            raise ServiceError(f"Failed to submit detailed feedback: {str(e)}")
    
    async def get_user_feedback(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        pagination: PaginationParams,
        sort: SortParams,
        filters: Dict[str, Any]
    ) -> FeedbackListResult:
        """Retrieve user's feedback history with filtering and pagination."""
        try:
            # Build query
            query = self.db.query(UserFeedbackSubmission).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.user_id == user_id
                )
            )
            
            # Apply filters
            if filters.get("feedback_type"):
                query = query.filter(UserFeedbackSubmission.feedback_type == FeedbackType(filters["feedback_type"].value))
            
            if filters.get("status"):
                query = query.filter(UserFeedbackSubmission.status == FeedbackStatus(filters["status"].value))
            
            if filters.get("category"):
                query = query.filter(UserFeedbackSubmission.feedback_category == FeedbackCategory(filters["category"].value))
            
            # Apply sorting
            if sort.sort_by:
                sort_column = getattr(UserFeedbackSubmission, sort.sort_by, None)
                if sort_column:
                    if sort.sort_order == "desc":
                        query = query.order_by(desc(sort_column))
                    else:
                        query = query.order_by(asc(sort_column))
            else:
                query = query.order_by(desc(UserFeedbackSubmission.created_at))
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            offset = (pagination.page - 1) * pagination.page_size
            items = query.offset(offset).limit(pagination.page_size).all()
            
            # Convert to response format
            feedback_details = [
                FeedbackDetails(
                    id=item.id,
                    feedback_type=FeedbackTypeEnum(item.feedback_type.value),
                    feedback_category=FeedbackCategoryEnum(item.feedback_category.value) if item.feedback_category else None,
                    target_id=item.target_id,
                    target_type=item.target_type,
                    rating=item.rating,
                    feedback_text=item.feedback_text,
                    query=item.query,
                    status=FeedbackStatusEnum(item.status.value),
                    priority=item.priority.value,
                    created_at=item.created_at,
                    processed_at=item.processed_at,
                    processing_notes=item.processing_notes
                )
                for item in items
            ]
            
            # Calculate pagination info
            total_pages = (total_count + pagination.page_size - 1) // pagination.page_size
            has_next = pagination.page < total_pages
            has_previous = pagination.page > 1
            
            return FeedbackListResult(
                items=feedback_details,
                total_count=total_count,
                total_pages=total_pages,
                has_next=has_next,
                has_previous=has_previous
            )
            
        except Exception as e:
            self.logger.error("get_user_feedback_failed", error=str(e), tenant_id=tenant_id, user_id=user_id)
            raise ServiceError(f"Failed to retrieve user feedback: {str(e)}")
    
    async def get_feedback_stats(self, tenant_id: uuid.UUID) -> FeedbackStats:
        """Get aggregated feedback statistics for the tenant."""
        try:
            # Get basic counts
            total_submissions = self.db.query(UserFeedbackSubmission).filter(
                UserFeedbackSubmission.tenant_id == tenant_id
            ).count()
            
            pending_count = self.db.query(UserFeedbackSubmission).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.status == FeedbackStatus.PENDING
                )
            ).count()
            
            processed_count = self.db.query(UserFeedbackSubmission).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.status.in_([
                        FeedbackStatus.REVIEWED,
                        FeedbackStatus.IMPLEMENTED,
                        FeedbackStatus.REJECTED
                    ])
                )
            ).count()
            
            # Calculate average rating
            avg_rating_result = self.db.query(func.avg(UserFeedbackSubmission.rating)).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.rating.isnot(None)
                )
            ).scalar()
            
            average_rating = float(avg_rating_result) if avg_rating_result else None
            
            # Get category breakdown
            category_breakdown = {}
            category_results = self.db.query(
                UserFeedbackSubmission.feedback_category,
                func.count(UserFeedbackSubmission.id)
            ).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.feedback_category.isnot(None)
                )
            ).group_by(UserFeedbackSubmission.feedback_category).all()
            
            for category, count in category_results:
                category_breakdown[category.value] = count
            
            # Calculate recent trends (last 7 days)
            seven_days_ago = datetime.now() - timedelta(days=7)
            recent_submissions = self.db.query(
                func.date(UserFeedbackSubmission.created_at),
                func.count(UserFeedbackSubmission.id)
            ).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.created_at >= seven_days_ago
                )
            ).group_by(func.date(UserFeedbackSubmission.created_at)).all()
            
            daily_submissions = [count for date, count in recent_submissions]
            
            recent_trends = {
                "daily_submissions": daily_submissions,
                "satisfaction_trend": "stable"  # TODO: Implement trend calculation
            }
            
            return FeedbackStats(
                total_submissions=total_submissions,
                pending_count=pending_count,
                processed_count=processed_count,
                average_rating=average_rating,
                category_breakdown=category_breakdown,
                recent_trends=recent_trends
            )
            
        except Exception as e:
            self.logger.error("get_feedback_stats_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to retrieve feedback statistics: {str(e)}")
    
    # Private helper methods
    
    async def _validate_feedback_submission(self, feedback_data: FeedbackSubmissionRequest, tenant_id: uuid.UUID):
        """Validate feedback submission data."""
        # Check if target exists (if target_id is provided)
        if feedback_data.target_id:
            # TODO: Implement target validation based on target_type
            pass
        
        # Validate rating range based on feedback type
        if feedback_data.rating is not None:
            if feedback_data.feedback_type in [FeedbackTypeEnum.SEARCH_RESULT, FeedbackTypeEnum.LINK_QUALITY]:
                if feedback_data.rating not in [-1, 1, 2, 3, 4, 5]:
                    raise ValidationError("Invalid rating for search result or link quality feedback")
            elif feedback_data.feedback_type == FeedbackTypeEnum.ANSWER_QUALITY:
                if feedback_data.rating < 1 or feedback_data.rating > 5:
                    raise ValidationError("Answer quality rating must be between 1 and 5")
    
    def _determine_priority(self, feedback_data: FeedbackSubmissionRequest) -> FeedbackPriority:
        """Determine feedback priority based on content and type."""
        if feedback_data.feedback_type == FeedbackTypeEnum.GENERAL:
            if feedback_data.feedback_category in [FeedbackCategoryEnum.BUG_REPORT, FeedbackCategoryEnum.PERFORMANCE_ISSUE]:
                return FeedbackPriority.HIGH
            elif feedback_data.feedback_category == FeedbackCategoryEnum.FEATURE_REQUEST:
                return FeedbackPriority.MEDIUM
        elif feedback_data.rating is not None and feedback_data.rating <= -1:
            return FeedbackPriority.MEDIUM
        
        return FeedbackPriority.LOW
    
    def _determine_feedback_type_from_target(self, target_type: Optional[str]) -> FeedbackType:
        """Determine feedback type based on target type."""
        if target_type == "search_result":
            return FeedbackType.SEARCH_RESULT
        elif target_type == "document_link":
            return FeedbackType.LINK_QUALITY
        elif target_type == "answer":
            return FeedbackType.ANSWER_QUALITY
        else:
            return FeedbackType.GENERAL
    
    def _estimate_processing_time(self, priority: FeedbackPriority, feedback_type: FeedbackType) -> str:
        """Estimate processing time based on priority and type."""
        if priority == FeedbackPriority.CRITICAL:
            return "4 hours"
        elif priority == FeedbackPriority.HIGH:
            return "24 hours"
        elif priority == FeedbackPriority.MEDIUM:
            return "3 days"
        else:
            return "1 week"
    
    def _generate_confirmation_message(self, feedback: UserFeedbackSubmission) -> str:
        """Generate appropriate confirmation message."""
        if feedback.feedback_type == FeedbackType.SEARCH_RESULT:
            return "Thank you for rating this search result! Your feedback helps improve our search quality."
        elif feedback.feedback_type == FeedbackType.LINK_QUALITY:
            return "Thank you for your feedback on document linking! We'll use this to improve our recommendations."
        elif feedback.feedback_type == FeedbackType.ANSWER_QUALITY:
            return "Thank you for rating this answer! Your feedback helps us provide better responses."
        else:
            return "Thank you for your feedback! We appreciate your input and will review it soon."
    
    def _generate_detailed_confirmation_message(self, feedback_data: DetailedFeedbackRequest) -> str:
        """Generate confirmation message for detailed feedback."""
        if feedback_data.feedback_category == FeedbackCategoryEnum.BUG_REPORT:
            return "Thank you for reporting this issue! Our team will investigate and provide updates."
        elif feedback_data.feedback_category == FeedbackCategoryEnum.FEATURE_REQUEST:
            return "Thank you for your feature suggestion! We'll consider it for future development."
        else:
            return "Thank you for your detailed feedback! We'll review it carefully and take appropriate action."
    
    def _construct_detailed_feedback_text(self, feedback_data: DetailedFeedbackRequest) -> str:
        """Construct comprehensive feedback text from detailed form."""
        parts = [f"Title: {feedback_data.title}"]
        parts.append(f"Description: {feedback_data.description}")
        
        if feedback_data.steps_to_reproduce:
            parts.append(f"Steps to Reproduce: {feedback_data.steps_to_reproduce}")
        
        if feedback_data.expected_behavior:
            parts.append(f"Expected Behavior: {feedback_data.expected_behavior}")
        
        if feedback_data.actual_behavior:
            parts.append(f"Actual Behavior: {feedback_data.actual_behavior}")
        
        return "\n\n".join(parts)
    
    async def _update_feedback_session(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: Optional[str],
        user_agent: Optional[str]
    ):
        """Update or create feedback session tracking."""
        if not session_id:
            return
        
        try:
            # Find or create session
            session = self.db.query(FeedbackSession).filter(
                and_(
                    FeedbackSession.tenant_id == tenant_id,
                    FeedbackSession.user_id == user_id,
                    FeedbackSession.session_id == session_id
                )
            ).first()
            
            if not session:
                session = FeedbackSession(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id,
                    user_agent=user_agent
                )
                self.db.add(session)
            
            # Update session metrics
            session.feedback_submissions += 1
            session.total_interactions += 1
            
            self.db.commit()
            
        except Exception as e:
            self.logger.warning("session_update_failed", error=str(e), session_id=session_id)
    
    async def _update_feedback_aggregation(self, feedback: UserFeedbackSubmission):
        """Update aggregated feedback statistics."""
        if not feedback.target_id:
            return
        
        try:
            # Find or create aggregation record
            aggregation = self.db.query(FeedbackAggregation).filter(
                and_(
                    FeedbackAggregation.tenant_id == feedback.tenant_id,
                    FeedbackAggregation.target_id == feedback.target_id,
                    FeedbackAggregation.target_type == feedback.target_type,
                    FeedbackAggregation.feedback_type == feedback.feedback_type
                )
            ).first()
            
            if not aggregation:
                aggregation = FeedbackAggregation(
                    tenant_id=feedback.tenant_id,
                    target_id=feedback.target_id,
                    target_type=feedback.target_type,
                    feedback_type=feedback.feedback_type,
                    first_feedback_at=feedback.created_at
                )
                self.db.add(aggregation)
            
            # Update aggregation metrics
            aggregation.total_feedback_count += 1
            aggregation.last_feedback_at = feedback.created_at
            
            if feedback.rating is not None:
                if feedback.rating > 0:
                    aggregation.positive_count += 1
                else:
                    aggregation.negative_count += 1
                
                # Recalculate average rating
                total_ratings = self.db.query(func.avg(UserFeedbackSubmission.rating)).filter(
                    and_(
                        UserFeedbackSubmission.tenant_id == feedback.tenant_id,
                        UserFeedbackSubmission.target_id == feedback.target_id,
                        UserFeedbackSubmission.target_type == feedback.target_type,
                        UserFeedbackSubmission.rating.isnot(None)
                    )
                ).scalar()
                
                aggregation.average_rating = float(total_ratings) if total_ratings else None
            
            # Update category counts
            if feedback.feedback_category:
                if not aggregation.category_counts:
                    aggregation.category_counts = {}
                
                category_key = feedback.feedback_category.value
                aggregation.category_counts[category_key] = aggregation.category_counts.get(category_key, 0) + 1
            
            # Calculate quality score (simple implementation)
            if aggregation.total_feedback_count > 0:
                positive_ratio = aggregation.positive_count / aggregation.total_feedback_count
                aggregation.quality_score = positive_ratio
                aggregation.confidence_score = min(1.0, aggregation.total_feedback_count / 10.0)  # Confidence increases with more feedback
            
            self.db.commit()
            
        except Exception as e:
            self.logger.warning("aggregation_update_failed", error=str(e), target_id=feedback.target_id)


# Dependency injection
_feedback_service_instance = None


def get_feedback_service(db_session: Session = Depends(get_database_session)) -> FeedbackService:
    """Get feedback service instance with dependency injection."""
    global _feedback_service_instance
    if _feedback_service_instance is None:
        _feedback_service_instance = FeedbackService(db_session)
    return _feedback_service_instance
