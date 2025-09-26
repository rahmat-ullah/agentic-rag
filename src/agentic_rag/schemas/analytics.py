"""
Analytics Schemas for Sprint 6 Story 6-04: Feedback Analytics and Insights System

This module contains Pydantic schemas for analytics API requests and responses,
including metrics, recommendations, dashboard configurations, and aggregations.
"""

import uuid
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator

from .base import PaginatedResponse


class AnalyticsMetricTypeEnum(str, Enum):
    """Analytics metric type enumeration for API."""
    
    SEARCH_QUALITY = "search_quality"
    USER_SATISFACTION = "user_satisfaction"
    CONTENT_QUALITY = "content_quality"
    SYSTEM_PERFORMANCE = "system_performance"
    USER_ENGAGEMENT = "user_engagement"
    LEARNING_EFFECTIVENESS = "learning_effectiveness"


class RecommendationTypeEnum(str, Enum):
    """Recommendation type enumeration for API."""
    
    SEARCH_IMPROVEMENT = "search_improvement"
    CONTENT_OPTIMIZATION = "content_optimization"
    USER_EXPERIENCE = "user_experience"
    SYSTEM_OPTIMIZATION = "system_optimization"
    LEARNING_TUNING = "learning_tuning"
    QUALITY_ENHANCEMENT = "quality_enhancement"


class RecommendationStatusEnum(str, Enum):
    """Recommendation status enumeration for API."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class RecommendationPriorityEnum(str, Enum):
    """Recommendation priority enumeration for API."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImplementationEffortEnum(str, Enum):
    """Implementation effort enumeration for API."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DashboardComponentTypeEnum(str, Enum):
    """Dashboard component type enumeration for API."""
    
    METRIC_CARD = "metric_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"
    ALERT_PANEL = "alert_panel"


# Request Schemas

class CreateAnalyticsMetricRequest(BaseModel):
    """Request schema for creating analytics metrics."""
    
    metric_type: AnalyticsMetricTypeEnum = Field(..., description="Type of analytics metric")
    metric_name: str = Field(..., min_length=1, max_length=100, description="Name of the metric")
    metric_category: Optional[str] = Field(None, max_length=50, description="Metric category for grouping")
    metric_value: float = Field(..., description="Metric value")
    baseline_value: Optional[float] = Field(None, description="Baseline value for comparison")
    target_value: Optional[float] = Field(None, description="Target value for the metric")
    measurement_date: date = Field(..., description="Date of measurement")
    measurement_period_start: Optional[datetime] = Field(None, description="Start of measurement period")
    measurement_period_end: Optional[datetime] = Field(None, description="End of measurement period")
    dimension_values: Optional[Dict[str, Any]] = Field(None, description="Dimension values for segmentation")
    sample_size: Optional[int] = Field(None, ge=0, description="Sample size for the metric")
    confidence_level: Optional[float] = Field(None, ge=0, le=1, description="Confidence level of the metric")
    calculation_method: Optional[str] = Field(None, max_length=100, description="Method used to calculate metric")
    data_sources: Optional[List[str]] = Field(None, description="Data sources used for calculation")
    metric_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metric metadata")

    class Config:
        schema_extra = {
            "example": {
                "metric_type": "search_quality",
                "metric_name": "click_through_rate",
                "metric_category": "user_engagement",
                "metric_value": 0.65,
                "baseline_value": 0.60,
                "target_value": 0.70,
                "measurement_date": "2024-12-19",
                "dimension_values": {"user_segment": "power_users", "region": "US"},
                "sample_size": 1000,
                "confidence_level": 0.95,
                "calculation_method": "clicks / impressions",
                "data_sources": ["user_feedback", "search_logs"]
            }
        }


class CreatePerformanceRecommendationRequest(BaseModel):
    """Request schema for creating performance recommendations."""
    
    recommendation_type: RecommendationTypeEnum = Field(..., description="Type of recommendation")
    category: Optional[str] = Field(None, max_length=50, description="Recommendation category")
    title: str = Field(..., min_length=1, max_length=200, description="Recommendation title")
    description: str = Field(..., min_length=1, description="Detailed description")
    rationale: Optional[str] = Field(None, description="Rationale for the recommendation")
    priority: RecommendationPriorityEnum = Field(..., description="Priority level")
    estimated_impact: float = Field(..., ge=0, le=1, description="Estimated impact (0.0-1.0)")
    implementation_effort: ImplementationEffortEnum = Field(..., description="Implementation effort required")
    implementation_steps: Optional[List[str]] = Field(None, description="Implementation steps")
    required_resources: Optional[Dict[str, Any]] = Field(None, description="Required resources")
    estimated_duration_hours: Optional[int] = Field(None, ge=0, description="Estimated duration in hours")
    related_metrics: Optional[List[uuid.UUID]] = Field(None, description="Related metric IDs")
    recommendation_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "recommendation_type": "search_improvement",
                "category": "relevance",
                "title": "Improve search result ranking algorithm",
                "description": "Update the ranking algorithm to better weight user feedback signals",
                "rationale": "Current CTR is below target, user feedback indicates relevance issues",
                "priority": "high",
                "estimated_impact": 0.15,
                "implementation_effort": "medium",
                "implementation_steps": [
                    "Analyze current ranking factors",
                    "Implement feedback weighting",
                    "Test with A/B experiment",
                    "Deploy to production"
                ],
                "estimated_duration_hours": 40
            }
        }


class UpdateRecommendationStatusRequest(BaseModel):
    """Request schema for updating recommendation status."""
    
    status: RecommendationStatusEnum = Field(..., description="New status")
    assigned_to: Optional[uuid.UUID] = Field(None, description="User assigned to the recommendation")
    implementation_notes: Optional[str] = Field(None, description="Implementation notes")
    actual_metrics: Optional[Dict[str, float]] = Field(None, description="Actual metrics after implementation")

    class Config:
        schema_extra = {
            "example": {
                "status": "in_progress",
                "assigned_to": "123e4567-e89b-12d3-a456-426614174000",
                "implementation_notes": "Started implementation, expected completion in 2 weeks"
            }
        }


class CreateDashboardConfigurationRequest(BaseModel):
    """Request schema for creating dashboard configurations."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Dashboard name")
    description: Optional[str] = Field(None, description="Dashboard description")
    dashboard_type: str = Field(..., description="Dashboard type (executive, operational, analytical)")
    layout_config: Dict[str, Any] = Field(..., description="Grid layout configuration")
    components: List[Dict[str, Any]] = Field(..., description="Dashboard components")
    filters: Optional[Dict[str, Any]] = Field(None, description="Default filters")
    refresh_interval_minutes: int = Field(5, ge=1, le=60, description="Refresh interval in minutes")
    is_public: bool = Field(False, description="Whether dashboard is public")
    shared_with_users: Optional[List[uuid.UUID]] = Field(None, description="Users to share with")
    shared_with_roles: Optional[List[str]] = Field(None, description="Roles to share with")
    dashboard_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "name": "Executive Summary",
                "description": "High-level KPIs and trends for executives",
                "dashboard_type": "executive",
                "layout_config": {"columns": 12, "rows": 8},
                "components": [
                    {
                        "type": "metric_card",
                        "position": {"x": 0, "y": 0, "w": 3, "h": 2},
                        "config": {"metric": "search_quality_score", "title": "Search Quality"}
                    }
                ],
                "refresh_interval_minutes": 5,
                "is_public": False
            }
        }


class MetricQueryRequest(BaseModel):
    """Request schema for querying metrics."""
    
    metric_types: Optional[List[AnalyticsMetricTypeEnum]] = Field(None, description="Metric types to filter")
    metric_names: Optional[List[str]] = Field(None, description="Metric names to filter")
    metric_categories: Optional[List[str]] = Field(None, description="Metric categories to filter")
    start_date: Optional[date] = Field(None, description="Start date for filtering")
    end_date: Optional[date] = Field(None, description="End date for filtering")
    dimensions: Optional[Dict[str, Any]] = Field(None, description="Dimension filters")
    aggregation_level: Optional[str] = Field("daily", description="Aggregation level (daily, weekly, monthly)")
    include_trends: bool = Field(True, description="Whether to include trend analysis")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")

    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        if v and values.get('start_date') and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

    class Config:
        schema_extra = {
            "example": {
                "metric_types": ["search_quality", "user_satisfaction"],
                "start_date": "2024-12-01",
                "end_date": "2024-12-19",
                "dimensions": {"user_segment": "power_users"},
                "aggregation_level": "daily",
                "include_trends": True,
                "limit": 100
            }
        }


# Response Schemas

class AnalyticsMetricResponse(BaseModel):
    """Response schema for analytics metrics."""
    
    id: uuid.UUID
    metric_type: AnalyticsMetricTypeEnum
    metric_name: str
    metric_category: Optional[str]
    metric_value: float
    baseline_value: Optional[float]
    target_value: Optional[float]
    measurement_date: date
    measurement_period_start: Optional[datetime]
    measurement_period_end: Optional[datetime]
    dimension_values: Optional[Dict[str, Any]]
    sample_size: Optional[int]
    confidence_level: Optional[float]
    previous_value: Optional[float]
    change_percentage: Optional[float]
    trend_direction: Optional[str]
    data_quality_score: Optional[float]
    statistical_significance: Optional[float]
    calculation_method: Optional[str]
    data_sources: Optional[List[str]]
    metric_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class PerformanceRecommendationResponse(BaseModel):
    """Response schema for performance recommendations."""
    
    id: uuid.UUID
    recommendation_type: RecommendationTypeEnum
    category: Optional[str]
    title: str
    description: str
    rationale: Optional[str]
    priority: RecommendationPriorityEnum
    estimated_impact: float
    implementation_effort: ImplementationEffortEnum
    implementation_steps: Optional[List[str]]
    required_resources: Optional[Dict[str, Any]]
    estimated_duration_hours: Optional[int]
    status: RecommendationStatusEnum
    assigned_to: Optional[uuid.UUID]
    baseline_metrics: Optional[Dict[str, float]]
    target_metrics: Optional[Dict[str, float]]
    actual_metrics: Optional[Dict[str, float]]
    effectiveness_score: Optional[float]
    related_metrics: Optional[List[uuid.UUID]]
    related_recommendations: Optional[List[uuid.UUID]]
    recommendation_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    implemented_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        orm_mode = True


class DashboardConfigurationResponse(BaseModel):
    """Response schema for dashboard configurations."""

    id: uuid.UUID
    name: str
    description: Optional[str]
    dashboard_type: str
    layout_config: Dict[str, Any]
    components: List[Dict[str, Any]]
    filters: Optional[Dict[str, Any]]
    refresh_interval_minutes: int
    is_public: bool
    shared_with_users: Optional[List[uuid.UUID]]
    shared_with_roles: Optional[List[str]]
    is_active: bool
    is_default: bool
    dashboard_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    last_accessed_at: Optional[datetime]

    class Config:
        orm_mode = True


class MetricAggregationResponse(BaseModel):
    """Response schema for metric aggregations."""

    id: uuid.UUID
    aggregation_name: str
    metric_type: AnalyticsMetricTypeEnum
    aggregation_level: str
    dimensions: Optional[Dict[str, Any]]
    period_start: datetime
    period_end: datetime
    count: int
    sum_value: Optional[float]
    avg_value: Optional[float]
    min_value: Optional[float]
    max_value: Optional[float]
    median_value: Optional[float]
    std_dev: Optional[float]
    p25_value: Optional[float]
    p75_value: Optional[float]
    p90_value: Optional[float]
    p95_value: Optional[float]
    p99_value: Optional[float]
    data_completeness: Optional[float]
    outlier_count: Optional[int]
    calculation_timestamp: datetime
    source_record_count: Optional[int]
    aggregation_metadata: Optional[Dict[str, Any]]

    class Config:
        orm_mode = True


class TrendAnalysisResponse(BaseModel):
    """Response schema for trend analysis."""

    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0-1.0
    change_percentage: float
    statistical_significance: float
    confidence_interval: Dict[str, float]  # {"lower": 0.1, "upper": 0.3}
    trend_metadata: Optional[Dict[str, Any]]


class SearchQualityMetricsResponse(BaseModel):
    """Response schema for search quality metrics."""

    click_through_rate: float
    result_relevance_score: float
    user_satisfaction_rating: float
    search_success_rate: float
    average_results_per_query: float
    zero_results_rate: float
    query_refinement_rate: float
    session_abandonment_rate: float
    trend_analysis: List[TrendAnalysisResponse]
    quality_alerts: List[Dict[str, Any]]
    benchmark_comparison: Dict[str, float]

    class Config:
        schema_extra = {
            "example": {
                "click_through_rate": 0.65,
                "result_relevance_score": 0.82,
                "user_satisfaction_rating": 4.2,
                "search_success_rate": 0.89,
                "average_results_per_query": 8.5,
                "zero_results_rate": 0.05,
                "query_refinement_rate": 0.23,
                "session_abandonment_rate": 0.12,
                "trend_analysis": [],
                "quality_alerts": [],
                "benchmark_comparison": {"industry_average": 0.60}
            }
        }


class UserSatisfactionScoreResponse(BaseModel):
    """Response schema for user satisfaction scoring."""

    overall_satisfaction_score: float
    satisfaction_by_segment: Dict[str, float]
    satisfaction_trends: List[TrendAnalysisResponse]
    correlation_analysis: Dict[str, float]
    satisfaction_drivers: List[Dict[str, Any]]
    prediction_confidence: float
    predicted_satisfaction: float

    class Config:
        schema_extra = {
            "example": {
                "overall_satisfaction_score": 4.2,
                "satisfaction_by_segment": {
                    "power_users": 4.5,
                    "casual_users": 3.9,
                    "new_users": 4.0
                },
                "satisfaction_trends": [],
                "correlation_analysis": {
                    "search_quality": 0.78,
                    "response_time": -0.45,
                    "content_freshness": 0.62
                },
                "satisfaction_drivers": [
                    {"factor": "search_relevance", "impact": 0.35},
                    {"factor": "response_time", "impact": 0.28}
                ],
                "prediction_confidence": 0.85,
                "predicted_satisfaction": 4.3
            }
        }


class ContentQualityAssessmentResponse(BaseModel):
    """Response schema for content quality assessment."""

    overall_quality_score: float
    quality_by_category: Dict[str, float]
    quality_trends: List[TrendAnalysisResponse]
    quality_issues: List[Dict[str, Any]]
    improvement_opportunities: List[Dict[str, Any]]
    quality_alerts: List[Dict[str, Any]]
    content_freshness_score: float
    accuracy_score: float
    completeness_score: float

    class Config:
        schema_extra = {
            "example": {
                "overall_quality_score": 0.85,
                "quality_by_category": {
                    "technical_docs": 0.88,
                    "user_guides": 0.82,
                    "faqs": 0.87
                },
                "quality_trends": [],
                "quality_issues": [
                    {"type": "outdated_content", "count": 15, "severity": "medium"}
                ],
                "improvement_opportunities": [
                    {"area": "content_freshness", "potential_impact": 0.12}
                ],
                "quality_alerts": [],
                "content_freshness_score": 0.78,
                "accuracy_score": 0.92,
                "completeness_score": 0.85
            }
        }


class AnalyticsDashboardResponse(BaseModel):
    """Response schema for analytics dashboard data."""

    dashboard_id: uuid.UUID
    dashboard_name: str
    last_updated: datetime
    search_quality_metrics: SearchQualityMetricsResponse
    user_satisfaction_score: UserSatisfactionScoreResponse
    content_quality_assessment: ContentQualityAssessmentResponse
    system_performance_metrics: Dict[str, float]
    user_engagement_metrics: Dict[str, float]
    learning_effectiveness_metrics: Dict[str, float]
    active_recommendations: List[PerformanceRecommendationResponse]
    recent_alerts: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "dashboard_id": "123e4567-e89b-12d3-a456-426614174000",
                "dashboard_name": "Executive Summary",
                "last_updated": "2024-12-19T14:30:00Z",
                "system_performance_metrics": {
                    "avg_response_time": 0.25,
                    "system_uptime": 0.999,
                    "error_rate": 0.001
                },
                "user_engagement_metrics": {
                    "daily_active_users": 1250,
                    "session_duration": 12.5,
                    "return_user_rate": 0.72
                },
                "learning_effectiveness_metrics": {
                    "learning_accuracy": 0.88,
                    "improvement_rate": 0.15
                },
                "active_recommendations": [],
                "recent_alerts": []
            }
        }


# Paginated Response Models

class AnalyticsMetricListResponse(PaginatedResponse):
    """Paginated response for analytics metrics."""

    items: List[AnalyticsMetricResponse] = Field(..., description="List of analytics metrics")


class PerformanceRecommendationListResponse(PaginatedResponse):
    """Paginated response for performance recommendations."""

    items: List[PerformanceRecommendationResponse] = Field(..., description="List of performance recommendations")


class DashboardConfigurationListResponse(PaginatedResponse):
    """Paginated response for dashboard configurations."""

    items: List[DashboardConfigurationResponse] = Field(..., description="List of dashboard configurations")
