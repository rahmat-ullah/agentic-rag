"""
Feedback API endpoints for collecting user feedback and improving system quality.

This module provides endpoints for submitting feedback, retrieving feedback history,
and managing feedback processing as part of Sprint 6, Story 6-01.
"""

import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from fastapi.security import HTTPBearer
import structlog

from agentic_rag.models.database import User
from agentic_rag.schemas.feedback import (
    FeedbackSubmissionRequest,
    FeedbackSubmissionResponse,
    ThumbsFeedbackRequest,
    DetailedFeedbackRequest,
    FeedbackDetails,
    FeedbackListResponse,
    FeedbackStatsResponse,
    FeedbackTypeEnum,
    FeedbackStatusEnum,
    FeedbackCategoryEnum
)
from agentic_rag.schemas.base import BaseResponse, PaginationParams, SortParams, FilterParams
from agentic_rag.api.dependencies.auth import get_current_user, get_effective_tenant_id
from agentic_rag.api.exceptions import ValidationError, ServiceError
from agentic_rag.services.feedback_service import get_feedback_service, FeedbackService

logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/feedback", response_model=FeedbackSubmissionResponse)
async def submit_feedback(
    request: FeedbackSubmissionRequest,
    current_user: User = Depends(get_current_user),
    tenant_id: uuid.UUID = Depends(get_effective_tenant_id),
    feedback_service: FeedbackService = Depends(get_feedback_service),
    http_request: Request = None
):
    """
    Submit general feedback about search results, answers, or system functionality.
    
    This endpoint accepts comprehensive feedback submissions with ratings, categories,
    and detailed text feedback. It supports all feedback types defined in the system.
    
    **Features:**
    - Multiple feedback types (search results, link quality, answer quality, general)
    - Rating scales and thumbs up/down feedback
    - Categorized feedback for better organization
    - Context preservation (query, session, metadata)
    - Automatic status tracking and confirmation
    """
    try:
        start_time = datetime.now()
        
        # Extract session information
        session_id = request.session_id or getattr(http_request.state, "session_id", None)
        user_agent = http_request.headers.get("user-agent") if http_request else None
        
        # Submit feedback
        feedback_result = await feedback_service.submit_feedback(
            tenant_id=tenant_id,
            user_id=current_user.id,
            feedback_data=request,
            session_id=session_id,
            user_agent=user_agent
        )
        
        # Log feedback submission
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            "feedback_submitted",
            feedback_id=feedback_result.feedback_id,
            feedback_type=request.feedback_type,
            user_id=current_user.id,
            tenant_id=tenant_id,
            processing_time_seconds=processing_time
        )
        
        return FeedbackSubmissionResponse(
            feedback_id=feedback_result.feedback_id,
            status=feedback_result.status,
            estimated_processing_time=feedback_result.estimated_processing_time,
            confirmation_message=feedback_result.confirmation_message
        )
        
    except ValidationError as e:
        logger.warning("feedback_validation_error", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except ServiceError as e:
        logger.error("feedback_service_error", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service error: {str(e)}"
        )
    except Exception as e:
        logger.error("feedback_unexpected_error", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while submitting feedback"
        )


@router.post("/feedback/thumbs", response_model=FeedbackSubmissionResponse)
async def submit_thumbs_feedback(
    request: ThumbsFeedbackRequest,
    current_user: User = Depends(get_current_user),
    tenant_id: uuid.UUID = Depends(get_effective_tenant_id),
    feedback_service: FeedbackService = Depends(get_feedback_service),
    http_request: Request = None
):
    """
    Submit simple thumbs up/down feedback for quick user interactions.
    
    This endpoint provides a simplified interface for binary feedback on search results,
    document links, or answers. It's optimized for quick user interactions.
    
    **Features:**
    - Simple thumbs up/down interface
    - Automatic feedback type detection based on target
    - Fast response times (< 200ms)
    - Session tracking for analytics
    """
    try:
        start_time = datetime.now()
        
        # Convert thumbs feedback to standard format
        feedback_data = FeedbackSubmissionRequest(
            feedback_type=FeedbackTypeEnum.SEARCH_RESULT,  # Default, will be adjusted by service
            target_id=request.target_id,
            target_type=request.target_type,
            rating=1 if request.thumbs_up else -1,
            query=request.query,
            session_id=request.session_id
        )
        
        # Extract session information
        session_id = request.session_id or getattr(http_request.state, "session_id", None)
        user_agent = http_request.headers.get("user-agent") if http_request else None
        
        # Submit feedback
        feedback_result = await feedback_service.submit_thumbs_feedback(
            tenant_id=tenant_id,
            user_id=current_user.id,
            feedback_data=feedback_data,
            session_id=session_id,
            user_agent=user_agent
        )
        
        # Log feedback submission
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            "thumbs_feedback_submitted",
            feedback_id=feedback_result.feedback_id,
            thumbs_up=request.thumbs_up,
            target_type=request.target_type,
            user_id=current_user.id,
            processing_time_seconds=processing_time
        )
        
        return FeedbackSubmissionResponse(
            feedback_id=feedback_result.feedback_id,
            status=feedback_result.status,
            estimated_processing_time="Immediate",
            confirmation_message="Thank you for your feedback!"
        )
        
    except Exception as e:
        logger.error("thumbs_feedback_error", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while submitting thumbs feedback"
        )


@router.post("/feedback/detailed", response_model=FeedbackSubmissionResponse)
async def submit_detailed_feedback(
    request: DetailedFeedbackRequest,
    current_user: User = Depends(get_current_user),
    tenant_id: uuid.UUID = Depends(get_effective_tenant_id),
    feedback_service: FeedbackService = Depends(get_feedback_service),
    http_request: Request = None
):
    """
    Submit detailed feedback with comprehensive information for complex issues.
    
    This endpoint accepts detailed feedback forms with structured information
    for bug reports, feature requests, and quality issues.
    
    **Features:**
    - Structured feedback forms
    - Issue categorization and prioritization
    - Steps to reproduce for bug reports
    - Expected vs actual behavior tracking
    - Automatic routing to appropriate teams
    """
    try:
        # Extract session information
        session_id = request.session_id or getattr(http_request.state, "session_id", None)
        user_agent = http_request.headers.get("user-agent") if http_request else None
        
        # Submit detailed feedback
        feedback_result = await feedback_service.submit_detailed_feedback(
            tenant_id=tenant_id,
            user_id=current_user.id,
            feedback_data=request,
            session_id=session_id,
            user_agent=user_agent
        )
        
        logger.info(
            "detailed_feedback_submitted",
            feedback_id=feedback_result.feedback_id,
            feedback_type=request.feedback_type,
            category=request.feedback_category,
            priority=request.priority,
            user_id=current_user.id
        )
        
        return FeedbackSubmissionResponse(
            feedback_id=feedback_result.feedback_id,
            status=feedback_result.status,
            estimated_processing_time=feedback_result.estimated_processing_time,
            confirmation_message=feedback_result.confirmation_message
        )
        
    except Exception as e:
        logger.error("detailed_feedback_error", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while submitting detailed feedback"
        )


@router.get("/feedback", response_model=FeedbackListResponse)
async def get_user_feedback(
    current_user: User = Depends(get_current_user),
    tenant_id: uuid.UUID = Depends(get_effective_tenant_id),
    feedback_service: FeedbackService = Depends(get_feedback_service),
    pagination: PaginationParams = Depends(),
    sort: SortParams = Depends(),
    feedback_type: Optional[FeedbackTypeEnum] = Query(None, description="Filter by feedback type"),
    status: Optional[FeedbackStatusEnum] = Query(None, description="Filter by status"),
    category: Optional[FeedbackCategoryEnum] = Query(None, description="Filter by category")
):
    """
    Retrieve user's feedback history with filtering and pagination.
    
    This endpoint allows users to view their submitted feedback, track processing
    status, and see the impact of their contributions.
    
    **Features:**
    - Paginated feedback history
    - Filtering by type, status, and category
    - Sorting by date, status, or priority
    - Processing status tracking
    - Impact reporting
    """
    try:
        feedback_list = await feedback_service.get_user_feedback(
            tenant_id=tenant_id,
            user_id=current_user.id,
            pagination=pagination,
            sort=sort,
            filters={
                "feedback_type": feedback_type,
                "status": status,
                "category": category
            }
        )
        
        return FeedbackListResponse(
            items=feedback_list.items,
            total_count=feedback_list.total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=feedback_list.total_pages,
            has_next=feedback_list.has_next,
            has_previous=feedback_list.has_previous
        )
        
    except Exception as e:
        logger.error("get_feedback_error", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving feedback"
        )


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
    current_user: User = Depends(get_current_user),
    tenant_id: uuid.UUID = Depends(get_effective_tenant_id),
    feedback_service: FeedbackService = Depends(get_feedback_service)
):
    """
    Get feedback statistics and trends for the current tenant.
    
    This endpoint provides aggregated feedback statistics, trends, and insights
    for administrators and analysts to understand system performance and user satisfaction.
    
    **Features:**
    - Total submission counts
    - Processing status breakdown
    - Average ratings and satisfaction scores
    - Category distribution
    - Recent trends and patterns
    """
    try:
        stats = await feedback_service.get_feedback_stats(tenant_id=tenant_id)
        
        return FeedbackStatsResponse(
            total_submissions=stats.total_submissions,
            pending_count=stats.pending_count,
            processed_count=stats.processed_count,
            average_rating=stats.average_rating,
            category_breakdown=stats.category_breakdown,
            recent_trends=stats.recent_trends
        )
        
    except Exception as e:
        logger.error("get_feedback_stats_error", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving feedback statistics"
        )
