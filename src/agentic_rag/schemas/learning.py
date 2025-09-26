"""
Learning Algorithm Schemas for Sprint 6 Story 6-03

This module contains Pydantic schemas for learning algorithm API requests and responses.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict

from agentic_rag.schemas.base import BaseResponse, PaginatedResponse
from agentic_rag.models.learning import (
    LearningAlgorithmType,
    LearningModelType,
    LearningStatus,
    FeedbackSignalType
)


# Request Schemas

class CreateLearningAlgorithmRequest(BaseModel):
    """Request schema for creating a learning algorithm."""
    
    algorithm_type: LearningAlgorithmType = Field(
        ...,
        description="Type of learning algorithm"
    )
    model_type: LearningModelType = Field(
        ...,
        description="Type of learning model"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the learning algorithm"
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Description of the learning algorithm"
    )
    learning_rate: float = Field(
        0.01,
        gt=0.0,
        le=1.0,
        description="Learning rate for the algorithm"
    )
    validation_threshold: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Validation threshold for performance"
    )
    decay_factor: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Decay factor for learning rate"
    )
    regularization_strength: Optional[float] = Field(
        None,
        ge=0.0,
        description="Regularization strength"
    )
    is_enabled: bool = Field(
        True,
        description="Whether the algorithm is enabled"
    )
    auto_update: bool = Field(
        True,
        description="Whether the algorithm should auto-update"
    )
    validation_frequency_hours: int = Field(
        24,
        gt=0,
        description="Validation frequency in hours"
    )
    algorithm_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional algorithm metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "algorithm_type": "link_confidence",
                "model_type": "exponential_moving_average",
                "name": "Link Confidence Learner",
                "description": "Learns to adjust link confidence based on user feedback",
                "learning_rate": 0.01,
                "validation_threshold": 0.05,
                "is_enabled": True,
                "auto_update": True,
                "validation_frequency_hours": 24
            }
        }
    )


class UpdateLearningAlgorithmRequest(BaseModel):
    """Request schema for updating a learning algorithm."""
    
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Name of the learning algorithm"
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Description of the learning algorithm"
    )
    learning_rate: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Learning rate for the algorithm"
    )
    validation_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Validation threshold for performance"
    )
    decay_factor: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Decay factor for learning rate"
    )
    regularization_strength: Optional[float] = Field(
        None,
        ge=0.0,
        description="Regularization strength"
    )
    status: Optional[LearningStatus] = Field(
        None,
        description="Algorithm status"
    )
    is_enabled: Optional[bool] = Field(
        None,
        description="Whether the algorithm is enabled"
    )
    auto_update: Optional[bool] = Field(
        None,
        description="Whether the algorithm should auto-update"
    )
    validation_frequency_hours: Optional[int] = Field(
        None,
        gt=0,
        description="Validation frequency in hours"
    )
    algorithm_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional algorithm metadata"
    )


class CreateFeedbackSignalRequest(BaseModel):
    """Request schema for creating a feedback signal."""
    
    signal_type: FeedbackSignalType = Field(
        ...,
        description="Type of feedback signal"
    )
    target_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Type of target (chunk, link, query, document)"
    )
    target_id: uuid.UUID = Field(
        ...,
        description="ID of the target"
    )
    signal_value: float = Field(
        ...,
        description="Value of the signal"
    )
    signal_strength: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Strength of the signal"
    )
    signal_confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the signal"
    )
    user_id: Optional[uuid.UUID] = Field(
        None,
        description="ID of the user providing feedback"
    )
    session_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Session ID"
    )
    query_context: Optional[str] = Field(
        None,
        description="Query context for the signal"
    )
    signal_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional signal metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "signal_type": "explicit_rating",
                "target_type": "chunk",
                "target_id": "123e4567-e89b-12d3-a456-426614174000",
                "signal_value": 4.0,
                "signal_strength": 1.0,
                "signal_confidence": 0.9,
                "query_context": "What is the pricing for this service?"
            }
        }
    )


class CreateABTestExperimentRequest(BaseModel):
    """Request schema for creating an A/B test experiment."""
    
    experiment_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the experiment"
    )
    description: Optional[str] = Field(
        None,
        description="Description of the experiment"
    )
    hypothesis: Optional[str] = Field(
        None,
        description="Hypothesis being tested"
    )
    control_algorithm_id: uuid.UUID = Field(
        ...,
        description="ID of the control algorithm"
    )
    treatment_algorithm_id: uuid.UUID = Field(
        ...,
        description="ID of the treatment algorithm"
    )
    traffic_split_percentage: float = Field(
        50.0,
        gt=0.0,
        le=100.0,
        description="Percentage of traffic for treatment"
    )
    primary_metric: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Primary metric to measure"
    )
    success_threshold: float = Field(
        ...,
        description="Success threshold for the metric"
    )
    minimum_sample_size: int = Field(
        1000,
        gt=0,
        description="Minimum sample size required"
    )
    confidence_level: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="Statistical confidence level"
    )
    experiment_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional experiment metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiment_name": "Link Confidence A/B Test",
                "description": "Testing new link confidence algorithm",
                "hypothesis": "New algorithm will improve click-through rates",
                "control_algorithm_id": "123e4567-e89b-12d3-a456-426614174000",
                "treatment_algorithm_id": "123e4567-e89b-12d3-a456-426614174001",
                "traffic_split_percentage": 50.0,
                "primary_metric": "click_through_rate",
                "success_threshold": 0.05,
                "minimum_sample_size": 1000,
                "confidence_level": 0.95
            }
        }
    )


# Response Schemas

class LearningAlgorithmResponse(BaseResponse):
    """Response schema for learning algorithm."""
    
    id: uuid.UUID
    algorithm_type: LearningAlgorithmType
    model_type: LearningModelType
    name: str
    description: Optional[str]
    learning_rate: float
    validation_threshold: float
    decay_factor: Optional[float]
    regularization_strength: Optional[float]
    status: LearningStatus
    current_version: str
    model_parameters: Optional[Dict[str, Any]]
    training_data_size: int
    accuracy_score: Optional[float]
    precision_score: Optional[float]
    recall_score: Optional[float]
    f1_score: Optional[float]
    is_enabled: bool
    auto_update: bool
    validation_frequency_hours: int
    algorithm_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    last_trained_at: Optional[datetime]
    last_validated_at: Optional[datetime]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "algorithm_type": "link_confidence",
                "model_type": "exponential_moving_average",
                "name": "Link Confidence Learner",
                "description": "Learns to adjust link confidence based on user feedback",
                "learning_rate": 0.01,
                "validation_threshold": 0.05,
                "status": "active",
                "current_version": "1.0.0",
                "training_data_size": 1500,
                "accuracy_score": 0.85,
                "is_enabled": True,
                "auto_update": True,
                "validation_frequency_hours": 24,
                "created_at": "2024-12-19T18:00:00Z",
                "updated_at": "2024-12-19T18:00:00Z"
            }
        }
    )


class FeedbackSignalResponse(BaseResponse):
    """Response schema for feedback signal."""
    
    id: uuid.UUID
    signal_type: FeedbackSignalType
    target_type: str
    target_id: uuid.UUID
    signal_value: float
    signal_strength: float
    signal_confidence: float
    user_id: Optional[uuid.UUID]
    session_id: Optional[str]
    query_context: Optional[str]
    is_processed: bool
    processed_at: Optional[datetime]
    processing_algorithm_id: Optional[uuid.UUID]
    signal_metadata: Optional[Dict[str, Any]]
    created_at: datetime


class LearningPerformanceMetricResponse(BaseResponse):
    """Response schema for learning performance metric."""
    
    id: uuid.UUID
    algorithm_id: uuid.UUID
    metric_name: str
    metric_value: float
    metric_type: str
    measurement_period_start: datetime
    measurement_period_end: datetime
    sample_size: int
    baseline_value: Optional[float]
    improvement_percentage: Optional[float]
    statistical_significance: Optional[float]
    metric_metadata: Optional[Dict[str, Any]]
    recorded_at: datetime


class ABTestExperimentResponse(BaseResponse):
    """Response schema for A/B test experiment."""
    
    id: uuid.UUID
    experiment_name: str
    description: Optional[str]
    hypothesis: Optional[str]
    control_algorithm_id: uuid.UUID
    treatment_algorithm_id: uuid.UUID
    traffic_split_percentage: float
    status: str
    is_active: bool
    primary_metric: str
    success_threshold: float
    minimum_sample_size: int
    confidence_level: float
    control_metric_value: Optional[float]
    treatment_metric_value: Optional[float]
    statistical_significance: Optional[float]
    effect_size: Optional[float]
    experiment_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]


class LearningValidationResultResponse(BaseResponse):
    """Response schema for learning validation result."""
    
    status: str
    score: float
    improvement_percentage: float
    statistical_significance: float
    confidence_interval: List[float]
    validation_metadata: Dict[str, Any]
    recommendations: List[str]


class ABTestResultResponse(BaseResponse):
    """Response schema for A/B test result."""
    
    experiment_id: uuid.UUID
    control_performance: float
    treatment_performance: float
    improvement_percentage: float
    statistical_significance: float
    confidence_level: float
    sample_size_control: int
    sample_size_treatment: int
    is_significant: bool
    recommendation: str


class LearningHealthCheckResponse(BaseResponse):
    """Response schema for learning health check."""
    
    algorithm_id: uuid.UUID
    algorithm_type: LearningAlgorithmType
    health_score: float
    status: str
    issues: List[str]
    recommendations: List[str]
    last_update: datetime
    performance_trend: str


class LearningIntegrationResultResponse(BaseResponse):
    """Response schema for learning integration result."""
    
    signals_processed: int
    algorithms_updated: int
    improvements_applied: int
    processing_time_seconds: float
    errors: List[str]


class LearningInsightsResponse(BaseResponse):
    """Response schema for learning insights."""
    
    time_period_hours: int
    signal_statistics: Dict[str, Dict[str, Union[int, float]]]
    algorithm_performance: Dict[str, Dict[str, Any]]
    total_signals_processed: int
    active_algorithms: int


# Paginated Response Schemas

class PaginatedLearningAlgorithmsResponse(PaginatedResponse):
    """Paginated response for learning algorithms."""
    
    items: List[LearningAlgorithmResponse]


class PaginatedFeedbackSignalsResponse(PaginatedResponse):
    """Paginated response for feedback signals."""
    
    items: List[FeedbackSignalResponse]


class PaginatedPerformanceMetricsResponse(PaginatedResponse):
    """Paginated response for performance metrics."""
    
    items: List[LearningPerformanceMetricResponse]


class PaginatedABTestExperimentsResponse(PaginatedResponse):
    """Paginated response for A/B test experiments."""
    
    items: List[ABTestExperimentResponse]
