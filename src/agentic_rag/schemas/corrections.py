"""
Pydantic schemas for content correction system API.

This module contains request/response models for the content correction and editing
system implementing Sprint 6, Story 6-02 requirements.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator

from .base import BaseResponse, PaginatedResponse


class CorrectionTypeEnum(str, Enum):
    """Correction type enumeration for API."""
    
    FACTUAL = "factual"
    FORMATTING = "formatting"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    GRAMMAR = "grammar"
    TERMINOLOGY = "terminology"


class CorrectionStatusEnum(str, Enum):
    """Correction status enumeration for API."""
    
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    REVERTED = "reverted"


class CorrectionPriorityEnum(str, Enum):
    """Correction priority enumeration for API."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewDecisionEnum(str, Enum):
    """Review decision enumeration for API."""
    
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"


# Request Models

class CorrectionSubmissionRequest(BaseModel):
    """Request model for submitting content corrections."""
    
    chunk_id: uuid.UUID = Field(..., description="ID of the chunk to correct")
    corrected_content: str = Field(..., min_length=1, max_length=50000, description="Corrected content")
    correction_reason: Optional[str] = Field(None, max_length=2000, description="Reason for the correction")
    correction_type: CorrectionTypeEnum = Field(..., description="Type of correction being made")
    priority: CorrectionPriorityEnum = Field(CorrectionPriorityEnum.MEDIUM, description="Priority of the correction")
    
    # Quality and confidence
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Submitter's confidence in correction")
    source_references: Optional[List[Dict[str, Any]]] = Field(None, description="Supporting references for correction")
    
    # Metadata
    correction_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional correction metadata")

    @validator('corrected_content')
    def validate_corrected_content(cls, v):
        """Validate corrected content is not empty and has meaningful changes."""
        if not v or v.strip() == "":
            raise ValueError("Corrected content cannot be empty")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
                "corrected_content": "The electrical components are priced at $150 per unit (updated from $120 as of December 2024).",
                "correction_reason": "Updated pricing information based on latest supplier quotes",
                "correction_type": "factual",
                "priority": "high",
                "confidence_score": 0.95,
                "source_references": [
                    {
                        "type": "supplier_quote",
                        "document": "Supplier_Quote_Dec2024.pdf",
                        "page": 3
                    }
                ]
            }
        }


class InlineEditRequest(BaseModel):
    """Request model for inline editing operations."""
    
    chunk_id: uuid.UUID = Field(..., description="ID of the chunk being edited")
    edit_operation: str = Field(..., description="Type of edit operation")
    start_position: int = Field(..., ge=0, description="Start position of edit")
    end_position: int = Field(..., ge=0, description="End position of edit")
    new_content: str = Field(..., description="New content for the edit")
    edit_metadata: Optional[Dict[str, Any]] = Field(None, description="Edit operation metadata")

    @validator('end_position')
    def validate_positions(cls, v, values):
        """Validate edit positions are logical."""
        start_pos = values.get('start_position', 0)
        if v < start_pos:
            raise ValueError("End position must be greater than or equal to start position")
        return v

    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
                "edit_operation": "replace",
                "start_position": 45,
                "end_position": 60,
                "new_content": "$150 per unit",
                "edit_metadata": {
                    "edit_type": "price_update",
                    "confidence": 0.9
                }
            }
        }


class ReviewSubmissionRequest(BaseModel):
    """Request model for expert review submission."""
    
    correction_id: uuid.UUID = Field(..., description="ID of the correction being reviewed")
    decision: ReviewDecisionEnum = Field(..., description="Review decision")
    review_notes: Optional[str] = Field(None, max_length=5000, description="Detailed review notes")
    
    # Quality assessment scores
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Accuracy assessment score")
    clarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Clarity assessment score")
    completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Completeness assessment score")
    
    # Review metadata
    quality_assessment: Optional[Dict[str, Any]] = Field(None, description="Structured quality assessment")
    review_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional review metadata")

    @root_validator
    def validate_review_decision(cls, values):
        """Validate review decision has appropriate supporting information."""
        decision = values.get('decision')
        review_notes = values.get('review_notes')
        
        if decision in [ReviewDecisionEnum.REJECT, ReviewDecisionEnum.REQUEST_CHANGES] and not review_notes:
            raise ValueError("Review notes are required for reject or request changes decisions")
        
        return values

    class Config:
        schema_extra = {
            "example": {
                "correction_id": "123e4567-e89b-12d3-a456-426614174000",
                "decision": "approve",
                "review_notes": "Correction is accurate and well-sourced. Pricing update is verified.",
                "accuracy_score": 0.95,
                "clarity_score": 0.9,
                "completeness_score": 0.85,
                "quality_assessment": {
                    "factual_accuracy": "excellent",
                    "source_quality": "high",
                    "impact_assessment": "medium"
                }
            }
        }


class VersionComparisonRequest(BaseModel):
    """Request model for version comparison."""
    
    chunk_id: uuid.UUID = Field(..., description="ID of the chunk to compare versions")
    version_1: int = Field(..., ge=1, description="First version number to compare")
    version_2: int = Field(..., ge=1, description="Second version number to compare")
    comparison_type: str = Field("side_by_side", description="Type of comparison view")

    @validator('version_2')
    def validate_different_versions(cls, v, values):
        """Ensure we're comparing different versions."""
        version_1 = values.get('version_1')
        if v == version_1:
            raise ValueError("Cannot compare a version with itself")
        return v

    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
                "version_1": 1,
                "version_2": 3,
                "comparison_type": "side_by_side"
            }
        }


# Response Models

class CorrectionSubmissionResponse(BaseResponse):
    """Response model for correction submission."""
    
    correction_id: uuid.UUID = Field(..., description="Unique identifier for the submitted correction")
    status: CorrectionStatusEnum = Field(..., description="Current status of the correction")
    workflow_id: uuid.UUID = Field(..., description="Workflow tracking ID")
    estimated_review_time: str = Field(..., description="Estimated time for review")
    next_steps: List[str] = Field(..., description="Next steps in the correction process")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "correction_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "pending",
                "workflow_id": "456e7890-e89b-12d3-a456-426614174001",
                "estimated_review_time": "2-3 business days",
                "next_steps": [
                    "Expert review assignment",
                    "Quality assessment",
                    "Approval decision"
                ]
            }
        }


class CorrectionDetails(BaseModel):
    """Detailed correction information."""
    
    id: uuid.UUID
    chunk_id: uuid.UUID
    original_content: str
    corrected_content: str
    correction_reason: Optional[str]
    correction_type: CorrectionTypeEnum
    status: CorrectionStatusEnum
    priority: CorrectionPriorityEnum
    
    # Submitter information
    submitter_id: uuid.UUID
    submitted_at: datetime
    
    # Review information
    reviewer_id: Optional[uuid.UUID]
    reviewed_at: Optional[datetime]
    review_decision: Optional[ReviewDecisionEnum]
    review_notes: Optional[str]
    
    # Quality scores
    confidence_score: Optional[float]
    impact_score: Optional[float]
    quality_score: Optional[float]
    
    # Metadata
    correction_metadata: Optional[Dict[str, Any]]
    source_references: Optional[List[Dict[str, Any]]]

    class Config:
        orm_mode = True


class VersionDetails(BaseModel):
    """Content version information."""
    
    id: uuid.UUID
    chunk_id: uuid.UUID
    version_number: int
    content: str
    content_hash: str
    change_summary: Optional[str]
    is_active: bool
    is_published: bool
    created_by: uuid.UUID
    created_at: datetime
    quality_score: Optional[float]
    readability_score: Optional[float]

    class Config:
        orm_mode = True


class VersionComparisonResponse(BaseResponse):
    """Response model for version comparison."""
    
    chunk_id: uuid.UUID
    version_1: VersionDetails
    version_2: VersionDetails
    differences: List[Dict[str, Any]] = Field(..., description="Structured differences between versions")
    similarity_score: float = Field(..., description="Similarity score between versions")
    change_summary: str = Field(..., description="Summary of changes between versions")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
                "differences": [
                    {
                        "type": "replacement",
                        "position": 45,
                        "old_text": "$120 per unit",
                        "new_text": "$150 per unit",
                        "change_type": "price_update"
                    }
                ],
                "similarity_score": 0.95,
                "change_summary": "Updated pricing information from $120 to $150 per unit"
            }
        }


class CorrectionListResponse(PaginatedResponse):
    """Response model for correction list."""
    
    items: List[CorrectionDetails] = Field(..., description="List of correction items")


class ReviewWorkflowResponse(BaseResponse):
    """Response model for review workflow information."""
    
    workflow_id: uuid.UUID
    correction_id: uuid.UUID
    current_step: str
    assigned_to: Optional[uuid.UUID]
    due_date: Optional[datetime]
    steps_completed: List[str]
    next_steps: List[str]
    workflow_data: Optional[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "workflow_id": "456e7890-e89b-12d3-a456-426614174001",
                "correction_id": "123e4567-e89b-12d3-a456-426614174000",
                "current_step": "expert_review",
                "assigned_to": "789e0123-e89b-12d3-a456-426614174002",
                "due_date": "2024-12-22T17:00:00Z",
                "steps_completed": ["submission", "validation"],
                "next_steps": ["review", "approval", "implementation"]
            }
        }


class CorrectionStatsResponse(BaseResponse):
    """Response model for correction statistics."""
    
    total_corrections: int = Field(..., description="Total number of corrections")
    pending_corrections: int = Field(..., description="Number of pending corrections")
    approved_corrections: int = Field(..., description="Number of approved corrections")
    rejected_corrections: int = Field(..., description="Number of rejected corrections")
    average_review_time_hours: float = Field(..., description="Average review time in hours")
    correction_type_breakdown: Dict[str, int] = Field(..., description="Breakdown by correction type")
    quality_improvement_metrics: Dict[str, float] = Field(..., description="Quality improvement metrics")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_corrections": 245,
                "pending_corrections": 12,
                "approved_corrections": 198,
                "rejected_corrections": 35,
                "average_review_time_hours": 18.5,
                "correction_type_breakdown": {
                    "factual": 120,
                    "formatting": 45,
                    "clarity": 35,
                    "completeness": 25,
                    "grammar": 15,
                    "terminology": 5
                },
                "quality_improvement_metrics": {
                    "average_accuracy_improvement": 0.15,
                    "user_satisfaction_increase": 0.12,
                    "search_quality_improvement": 0.08
                }
            }
        }
