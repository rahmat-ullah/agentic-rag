"""
Analytics Models for Sprint 6 Story 6-04: Feedback Analytics and Insights System

This module contains database models for analytics metrics, performance recommendations,
dashboard configurations, and metric aggregations to support comprehensive analytics
and insights functionality.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey,
    Index, CheckConstraint, UniqueConstraint, Date
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as GUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SQLEnum

from .database import Base


class AnalyticsMetricType(str, Enum):
    """Analytics metric type enumeration."""
    
    SEARCH_QUALITY = "search_quality"
    USER_SATISFACTION = "user_satisfaction"
    CONTENT_QUALITY = "content_quality"
    SYSTEM_PERFORMANCE = "system_performance"
    USER_ENGAGEMENT = "user_engagement"
    LEARNING_EFFECTIVENESS = "learning_effectiveness"


class RecommendationType(str, Enum):
    """Performance recommendation type enumeration."""
    
    SEARCH_IMPROVEMENT = "search_improvement"
    CONTENT_OPTIMIZATION = "content_optimization"
    USER_EXPERIENCE = "user_experience"
    SYSTEM_OPTIMIZATION = "system_optimization"
    LEARNING_TUNING = "learning_tuning"
    QUALITY_ENHANCEMENT = "quality_enhancement"


class RecommendationStatus(str, Enum):
    """Recommendation status enumeration."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class RecommendationPriority(str, Enum):
    """Recommendation priority enumeration."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImplementationEffort(str, Enum):
    """Implementation effort enumeration."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DashboardComponentType(str, Enum):
    """Dashboard component type enumeration."""
    
    METRIC_CARD = "metric_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"
    ALERT_PANEL = "alert_panel"


class AnalyticsMetric(Base):
    """Analytics metrics model for tracking system performance and quality."""
    
    __tablename__ = "analytics_metrics"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    
    # Metric classification
    metric_type = Column(SQLEnum(AnalyticsMetricType), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_category = Column(String(50), nullable=True)  # Sub-category for grouping
    
    # Metric values
    metric_value = Column(Float, nullable=False)
    baseline_value = Column(Float, nullable=True)
    target_value = Column(Float, nullable=True)
    
    # Time information
    measurement_date = Column(Date, nullable=False)
    measurement_period_start = Column(DateTime(timezone=True), nullable=True)
    measurement_period_end = Column(DateTime(timezone=True), nullable=True)
    
    # Context information
    dimension_values = Column(JSONB, nullable=True)  # {"user_segment": "power_users", "region": "US"}
    sample_size = Column(Integer, nullable=True)
    confidence_level = Column(Float, nullable=True)
    
    # Comparison metrics
    previous_value = Column(Float, nullable=True)
    change_percentage = Column(Float, nullable=True)
    trend_direction = Column(String(20), nullable=True)  # "increasing", "decreasing", "stable"
    
    # Quality indicators
    data_quality_score = Column(Float, nullable=True)
    statistical_significance = Column(Float, nullable=True)
    
    # Metadata
    calculation_method = Column(String(100), nullable=True)
    data_sources = Column(JSONB, nullable=True)  # List of source tables/systems
    metric_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Constraints
    __table_args__ = (
        Index("idx_analytics_metrics_tenant_type_date", "tenant_id", "metric_type", "measurement_date"),
        Index("idx_analytics_metrics_name_date", "metric_name", "measurement_date"),
        Index("idx_analytics_metrics_category", "metric_category"),
        CheckConstraint("metric_value IS NOT NULL", name="ck_analytics_metrics_value_not_null"),
        CheckConstraint("confidence_level >= 0 AND confidence_level <= 1", name="ck_analytics_metrics_confidence_range"),
        CheckConstraint("data_quality_score >= 0 AND data_quality_score <= 1", name="ck_analytics_metrics_quality_range"),
    )


class PerformanceRecommendation(Base):
    """Performance improvement recommendations model."""
    
    __tablename__ = "performance_recommendations"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    
    # Recommendation classification
    recommendation_type = Column(SQLEnum(RecommendationType), nullable=False)
    category = Column(String(50), nullable=True)
    
    # Recommendation content
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    
    # Priority and impact
    priority = Column(SQLEnum(RecommendationPriority), nullable=False)
    estimated_impact = Column(Float, nullable=False)  # 0.0-1.0 scale
    implementation_effort = Column(SQLEnum(ImplementationEffort), nullable=False)
    
    # Implementation details
    implementation_steps = Column(JSONB, nullable=True)  # List of steps
    required_resources = Column(JSONB, nullable=True)  # Resource requirements
    estimated_duration_hours = Column(Integer, nullable=True)
    
    # Status tracking
    status = Column(SQLEnum(RecommendationStatus), default=RecommendationStatus.PENDING)
    assigned_to = Column(GUID, ForeignKey("app_user.id"), nullable=True)
    
    # Effectiveness tracking
    baseline_metrics = Column(JSONB, nullable=True)  # Metrics before implementation
    target_metrics = Column(JSONB, nullable=True)  # Expected metrics after implementation
    actual_metrics = Column(JSONB, nullable=True)  # Actual metrics after implementation
    effectiveness_score = Column(Float, nullable=True)  # 0.0-1.0 scale
    
    # Related data
    related_metrics = Column(JSONB, nullable=True)  # List of metric IDs that triggered this
    related_recommendations = Column(JSONB, nullable=True)  # List of related recommendation IDs
    
    # Metadata
    recommendation_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    implemented_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    assigned_user = relationship("User", foreign_keys=[assigned_to])
    
    # Constraints
    __table_args__ = (
        Index("idx_performance_recommendations_tenant_type", "tenant_id", "recommendation_type"),
        Index("idx_performance_recommendations_status", "status"),
        Index("idx_performance_recommendations_priority", "priority"),
        Index("idx_performance_recommendations_assigned", "assigned_to"),
        CheckConstraint("estimated_impact >= 0 AND estimated_impact <= 1", name="ck_perf_rec_impact_range"),
        CheckConstraint("effectiveness_score >= 0 AND effectiveness_score <= 1", name="ck_perf_rec_effectiveness_range"),
    )


class DashboardConfiguration(Base):
    """Dashboard configuration model for customizable analytics dashboards."""
    
    __tablename__ = "dashboard_configurations"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    user_id = Column(GUID, ForeignKey("app_user.id"), nullable=True)  # Null for tenant-wide dashboards
    
    # Dashboard information
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    dashboard_type = Column(String(50), nullable=False)  # "executive", "operational", "analytical"
    
    # Configuration
    layout_config = Column(JSONB, nullable=False)  # Grid layout configuration
    components = Column(JSONB, nullable=False)  # List of dashboard components
    filters = Column(JSONB, nullable=True)  # Default filters
    refresh_interval_minutes = Column(Integer, default=5)
    
    # Access control
    is_public = Column(Boolean, default=False)
    shared_with_users = Column(JSONB, nullable=True)  # List of user IDs
    shared_with_roles = Column(JSONB, nullable=True)  # List of roles
    
    # Status
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    
    # Metadata
    dashboard_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    owner = relationship("User", foreign_keys=[user_id])
    
    # Constraints
    __table_args__ = (
        Index("idx_dashboard_configurations_tenant_user", "tenant_id", "user_id"),
        Index("idx_dashboard_configurations_type", "dashboard_type"),
        Index("idx_dashboard_configurations_active", "is_active"),
        UniqueConstraint("tenant_id", "user_id", "name", name="uq_dashboard_configurations_tenant_user_name"),
    )


class MetricAggregation(Base):
    """Metric aggregation model for pre-calculated analytics summaries."""
    
    __tablename__ = "metric_aggregations"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    
    # Aggregation definition
    aggregation_name = Column(String(100), nullable=False)
    metric_type = Column(SQLEnum(AnalyticsMetricType), nullable=False)
    aggregation_level = Column(String(50), nullable=False)  # "daily", "weekly", "monthly"
    dimensions = Column(JSONB, nullable=True)  # Grouping dimensions
    
    # Time period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Aggregated values
    count = Column(Integer, nullable=False)
    sum_value = Column(Float, nullable=True)
    avg_value = Column(Float, nullable=True)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    median_value = Column(Float, nullable=True)
    std_dev = Column(Float, nullable=True)
    
    # Percentiles
    p25_value = Column(Float, nullable=True)
    p75_value = Column(Float, nullable=True)
    p90_value = Column(Float, nullable=True)
    p95_value = Column(Float, nullable=True)
    p99_value = Column(Float, nullable=True)
    
    # Quality metrics
    data_completeness = Column(Float, nullable=True)  # Percentage of expected data points
    outlier_count = Column(Integer, nullable=True)
    
    # Metadata
    calculation_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    source_record_count = Column(Integer, nullable=True)
    aggregation_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        Index("idx_metric_aggregations_tenant_type_level", "tenant_id", "metric_type", "aggregation_level"),
        Index("idx_metric_aggregations_period", "period_start", "period_end"),
        Index("idx_metric_aggregations_name", "aggregation_name"),
        UniqueConstraint("tenant_id", "aggregation_name", "period_start", "period_end", 
                        name="uq_metric_aggregations_tenant_name_period"),
        CheckConstraint("period_end > period_start", name="ck_metric_aggregations_period_order"),
        CheckConstraint("count >= 0", name="ck_metric_aggregations_count_positive"),
    )
