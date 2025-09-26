"""
Pydantic schemas for feedback system API.

This module contains request/response models for the feedback collection system
implementing Sprint 6, Story 6-01 requirements.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator

from .base import BaseResponse, PaginatedResponse


class FeedbackTypeEnum(str, Enum):
    """Feedback type enumeration for API."""
    
    SEARCH_RESULT = "search_result"
    LINK_QUALITY = "link_quality"
    ANSWER_QUALITY = "answer_quality"
    GENERAL = "general"
    SYSTEM_USABILITY = "system_usability"


class FeedbackCategoryEnum(str, Enum):
    """Feedback category enumeration for API."""
    
    # Search result categories
    NOT_RELEVANT = "not_relevant"
    MISSING_INFORMATION = "missing_information"
    OUTDATED_CONTENT = "outdated_content"
    WRONG_DOCUMENT_TYPE = "wrong_document_type"
    
    # Link quality categories
    INCORRECT_LINK = "incorrect_link"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_LINK = "missing_link"
    DUPLICATE_LINK = "duplicate_link"
    
    # Answer quality categories
    INACCURATE_INFORMATION = "inaccurate_information"
    INCOMPLETE_ANSWER = "incomplete_answer"
    POOR_FORMATTING = "poor_formatting"
    MISSING_CITATIONS = "missing_citations"
    
    # General categories
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    PERFORMANCE_ISSUE = "performance_issue"
    USABILITY_ISSUE = "usability_issue"


class FeedbackStatusEnum(str, Enum):
    """Feedback status enumeration for API."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    REVIEWED = "reviewed"
    IMPLEMENTED = "implemented"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class FeedbackPriorityEnum(str, Enum):
    """Feedback priority enumeration for API."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Request Models

class FeedbackSubmissionRequest(BaseModel):
    """Request model for submitting feedback."""
    
    feedback_type: FeedbackTypeEnum = Field(..., description="Type of feedback being submitted")
    feedback_category: Optional[FeedbackCategoryEnum] = Field(None, description="Specific category of feedback")
    
    # Target identification
    target_id: Optional[uuid.UUID] = Field(None, description="ID of the target being rated")
    target_type: Optional[str] = Field(None, description="Type of target (search_result, document_link, answer, etc.)")
    
    # Feedback content
    rating: Optional[int] = Field(None, ge=-1, le=5, description="Rating from -1 (thumbs down) to 5 (excellent)")
    feedback_text: Optional[str] = Field(None, max_length=5000, description="Detailed feedback text")
    
    # Context
    query: Optional[str] = Field(None, max_length=2000, description="Original query that led to this feedback")
    session_id: Optional[str] = Field(None, description="User session identifier")
    context_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    
    @validator('rating')
    def validate_rating(cls, v, values):
        """Validate rating based on feedback type."""
        if v is not None:
            feedback_type = values.get('feedback_type')
            if feedback_type in [FeedbackTypeEnum.SEARCH_RESULT, FeedbackTypeEnum.LINK_QUALITY]:
                # For search results and links, allow thumbs up/down (-1, 1) or 1-5 scale
                if v not in [-1, 1, 2, 3, 4, 5]:
                    raise ValueError("Rating for search results and links must be -1, 1, or 1-5")
            elif feedback_type == FeedbackTypeEnum.ANSWER_QUALITY:
                # For answers, use 1-5 scale
                if v < 1 or v > 5:
                    raise ValueError("Rating for answer quality must be 1-5")
        return v
    
    @root_validator
    def validate_feedback_content(cls, values):
        """Ensure at least rating or feedback_text is provided."""
        rating = values.get('rating')
        feedback_text = values.get('feedback_text')
        
        if rating is None and not feedback_text:
            raise ValueError("Either rating or feedback_text must be provided")
        
        return values

    class Config:
        schema_extra = {
            "example": {
                "feedback_type": "search_result",
                "feedback_category": "not_relevant",
                "target_id": "123e4567-e89b-12d3-a456-426614174000",
                "target_type": "search_result",
                "rating": -1,
                "feedback_text": "This result doesn't match my query about pricing information",
                "query": "What are the pricing details for the electrical components?",
                "session_id": "sess_abc123"
            }
        }


class ThumbsFeedbackRequest(BaseModel):
    """Simplified request model for thumbs up/down feedback."""
    
    target_id: uuid.UUID = Field(..., description="ID of the target being rated")
    target_type: str = Field(..., description="Type of target")
    thumbs_up: bool = Field(..., description="True for thumbs up, False for thumbs down")
    query: Optional[str] = Field(None, description="Original query context")
    session_id: Optional[str] = Field(None, description="User session identifier")

    class Config:
        schema_extra = {
            "example": {
                "target_id": "123e4567-e89b-12d3-a456-426614174000",
                "target_type": "search_result",
                "thumbs_up": True,
                "query": "pricing information for electrical components"
            }
        }


class DetailedFeedbackRequest(BaseModel):
    """Request model for detailed feedback forms."""
    
    feedback_type: FeedbackTypeEnum = Field(..., description="Type of feedback")
    feedback_category: FeedbackCategoryEnum = Field(..., description="Specific category")
    target_id: Optional[uuid.UUID] = Field(None, description="Target ID if applicable")
    target_type: Optional[str] = Field(None, description="Target type if applicable")
    
    # Detailed feedback content
    title: str = Field(..., max_length=200, description="Brief title for the feedback")
    description: str = Field(..., max_length=5000, description="Detailed description")
    steps_to_reproduce: Optional[str] = Field(None, max_length=2000, description="Steps to reproduce (for bugs)")
    expected_behavior: Optional[str] = Field(None, max_length=1000, description="Expected behavior")
    actual_behavior: Optional[str] = Field(None, max_length=1000, description="Actual behavior")
    
    # Priority and urgency
    priority: FeedbackPriorityEnum = Field(FeedbackPriorityEnum.MEDIUM, description="Feedback priority")
    
    # Context
    query: Optional[str] = Field(None, description="Original query context")
    session_id: Optional[str] = Field(None, description="User session identifier")
    context_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    class Config:
        schema_extra = {
            "example": {
                "feedback_type": "answer_quality",
                "feedback_category": "inaccurate_information",
                "title": "Incorrect pricing information in answer",
                "description": "The answer provided incorrect pricing for electrical components",
                "expected_behavior": "Accurate pricing information from the latest documents",
                "actual_behavior": "Outdated pricing from 2022 documents",
                "priority": "high"
            }
        }


# Response Models

class FeedbackSubmissionResponse(BaseResponse):
    """Response model for feedback submission."""
    
    feedback_id: uuid.UUID = Field(..., description="Unique identifier for the submitted feedback")
    status: FeedbackStatusEnum = Field(..., description="Current status of the feedback")
    estimated_processing_time: Optional[str] = Field(None, description="Estimated time for processing")
    confirmation_message: str = Field(..., description="Confirmation message for the user")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "feedback_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "pending",
                "estimated_processing_time": "24 hours",
                "confirmation_message": "Thank you for your feedback! We'll review it within 24 hours."
            }
        }


class FeedbackDetails(BaseModel):
    """Detailed feedback information."""
    
    id: uuid.UUID
    feedback_type: FeedbackTypeEnum
    feedback_category: Optional[FeedbackCategoryEnum]
    target_id: Optional[uuid.UUID]
    target_type: Optional[str]
    rating: Optional[int]
    feedback_text: Optional[str]
    query: Optional[str]
    status: FeedbackStatusEnum
    priority: FeedbackPriorityEnum
    created_at: datetime
    processed_at: Optional[datetime]
    processing_notes: Optional[str]

    class Config:
        orm_mode = True


class FeedbackListResponse(PaginatedResponse):
    """Response model for feedback list."""
    
    items: List[FeedbackDetails] = Field(..., description="List of feedback items")


class FeedbackAggregationDetails(BaseModel):
    """Aggregated feedback statistics."""
    
    target_id: uuid.UUID
    target_type: str
    feedback_type: FeedbackTypeEnum
    total_feedback_count: int
    positive_count: int
    negative_count: int
    average_rating: Optional[float]
    quality_score: Optional[float]
    confidence_score: Optional[float]
    category_breakdown: Optional[Dict[str, int]]
    last_updated: datetime

    class Config:
        orm_mode = True


class FeedbackStatsResponse(BaseResponse):
    """Response model for feedback statistics."""
    
    total_submissions: int = Field(..., description="Total number of feedback submissions")
    pending_count: int = Field(..., description="Number of pending feedback items")
    processed_count: int = Field(..., description="Number of processed feedback items")
    average_rating: Optional[float] = Field(None, description="Average rating across all feedback")
    category_breakdown: Dict[str, int] = Field(..., description="Breakdown by category")
    recent_trends: Dict[str, Any] = Field(..., description="Recent feedback trends")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_submissions": 1250,
                "pending_count": 45,
                "processed_count": 1205,
                "average_rating": 3.8,
                "category_breakdown": {
                    "not_relevant": 120,
                    "missing_information": 85,
                    "inaccurate_information": 65
                },
                "recent_trends": {
                    "daily_submissions": [12, 15, 8, 22, 18],
                    "satisfaction_trend": "improving"
                }
            }
        }
