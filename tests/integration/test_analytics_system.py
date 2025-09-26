"""
Integration Tests for Sprint 6 Story 6-04: Feedback Analytics and Insights System

This module contains comprehensive integration tests for the analytics system,
including database operations, service functionality, API endpoints, and
end-to-end analytics workflows.
"""

import pytest
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, Any

from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from agentic_rag.models.tenant import Tenant
from agentic_rag.models.auth import User
from agentic_rag.models.analytics import (
    AnalyticsMetric,
    PerformanceRecommendation,
    DashboardConfiguration,
    MetricAggregation,
    AnalyticsMetricType,
    RecommendationType,
    RecommendationStatus,
    RecommendationPriority
)
from agentic_rag.models.feedback import UserFeedbackSubmission, FeedbackType
from agentic_rag.services.analytics_service import AnalyticsService
from agentic_rag.services.recommendation_service import RecommendationService
from agentic_rag.schemas.analytics import (
    CreateAnalyticsMetricRequest,
    CreatePerformanceRecommendationRequest
)


class TestAnalyticsDatabase:
    """Test analytics database operations."""
    
    def test_create_analytics_metric(self, db_session: Session, test_tenant: Tenant):
        """Test creating analytics metrics."""
        metric = AnalyticsMetric(
            tenant_id=test_tenant.id,
            metric_type=AnalyticsMetricType.SEARCH_QUALITY,
            metric_name="click_through_rate",
            metric_category="user_engagement",
            metric_value=0.65,
            baseline_value=0.60,
            target_value=0.70,
            measurement_date=date.today(),
            sample_size=1000,
            confidence_level=0.95,
            calculation_method="clicks / impressions",
            data_sources=["user_feedback"],
            metric_metadata={"test": True}
        )
        
        db_session.add(metric)
        db_session.commit()
        db_session.refresh(metric)
        
        assert metric.id is not None
        assert metric.metric_type == AnalyticsMetricType.SEARCH_QUALITY
        assert metric.metric_name == "click_through_rate"
        assert metric.metric_value == 0.65
        assert metric.tenant_id == test_tenant.id
    
    def test_create_performance_recommendation(self, db_session: Session, test_tenant: Tenant, test_user: User):
        """Test creating performance recommendations."""
        recommendation = PerformanceRecommendation(
            tenant_id=test_tenant.id,
            recommendation_type=RecommendationType.SEARCH_IMPROVEMENT,
            category="relevance",
            title="Improve search result ranking",
            description="Update ranking algorithm to better weight user feedback",
            rationale="Current CTR is below target",
            priority=RecommendationPriority.HIGH,
            estimated_impact=0.15,
            implementation_effort="medium",
            implementation_steps=["Analyze ranking factors", "Implement changes", "Test"],
            estimated_duration_hours=40,
            assigned_to=test_user.id
        )
        
        db_session.add(recommendation)
        db_session.commit()
        db_session.refresh(recommendation)
        
        assert recommendation.id is not None
        assert recommendation.recommendation_type == RecommendationType.SEARCH_IMPROVEMENT
        assert recommendation.title == "Improve search result ranking"
        assert recommendation.estimated_impact == 0.15
        assert recommendation.assigned_to == test_user.id
    
    def test_create_dashboard_configuration(self, db_session: Session, test_tenant: Tenant, test_user: User):
        """Test creating dashboard configurations."""
        dashboard = DashboardConfiguration(
            tenant_id=test_tenant.id,
            user_id=test_user.id,
            name="Test Dashboard",
            description="Test analytics dashboard",
            dashboard_type="executive",
            layout_config={"columns": 12, "rows": 8},
            components=[
                {
                    "type": "metric_card",
                    "position": {"x": 0, "y": 0, "w": 3, "h": 2},
                    "config": {"metric": "search_quality_score"}
                }
            ],
            refresh_interval_minutes=5,
            is_active=True
        )
        
        db_session.add(dashboard)
        db_session.commit()
        db_session.refresh(dashboard)
        
        assert dashboard.id is not None
        assert dashboard.name == "Test Dashboard"
        assert dashboard.dashboard_type == "executive"
        assert dashboard.is_active is True
    
    def test_create_metric_aggregation(self, db_session: Session, test_tenant: Tenant):
        """Test creating metric aggregations."""
        aggregation = MetricAggregation(
            tenant_id=test_tenant.id,
            aggregation_name="daily_search_quality",
            metric_type=AnalyticsMetricType.SEARCH_QUALITY,
            aggregation_level="daily",
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now(),
            count=100,
            sum_value=65.0,
            avg_value=0.65,
            min_value=0.45,
            max_value=0.85,
            median_value=0.67,
            std_dev=0.12,
            p95_value=0.82,
            data_completeness=0.98
        )
        
        db_session.add(aggregation)
        db_session.commit()
        db_session.refresh(aggregation)
        
        assert aggregation.id is not None
        assert aggregation.aggregation_name == "daily_search_quality"
        assert aggregation.count == 100
        assert aggregation.avg_value == 0.65


class TestAnalyticsService:
    """Test analytics service functionality."""
    
    @pytest.fixture
    def analytics_service(self, db_session: Session):
        """Create analytics service instance."""
        return AnalyticsService(db_session)
    
    @pytest.mark.asyncio
    async def test_create_analytics_metric(self, analytics_service: AnalyticsService, test_tenant: Tenant):
        """Test creating analytics metric via service."""
        metric_data = CreateAnalyticsMetricRequest(
            metric_type=AnalyticsMetricType.USER_SATISFACTION,
            metric_name="user_satisfaction_rating",
            metric_category="feedback",
            metric_value=4.2,
            baseline_value=4.0,
            target_value=4.5,
            measurement_date=date.today(),
            sample_size=500,
            confidence_level=0.95,
            calculation_method="average_rating",
            data_sources=["user_feedback"],
            metric_metadata={"test": True}
        )
        
        metric = await analytics_service.create_analytics_metric(test_tenant.id, metric_data)
        
        assert metric.id is not None
        assert metric.metric_type == AnalyticsMetricType.USER_SATISFACTION
        assert metric.metric_name == "user_satisfaction_rating"
        assert metric.metric_value == 4.2
        assert metric.tenant_id == test_tenant.id
    
    @pytest.mark.asyncio
    async def test_calculate_search_quality_metrics(self, analytics_service: AnalyticsService, test_tenant: Tenant, db_session: Session):
        """Test calculating search quality metrics."""
        # Create some test feedback data
        feedback = UserFeedbackSubmission(
            tenant_id=test_tenant.id,
            user_id=uuid.uuid4(),
            feedback_type=FeedbackType.SEARCH_RESULT,
            target_id=uuid.uuid4(),
            rating=4,
            feedback_text="Good search results",
            status="completed"
        )
        db_session.add(feedback)
        db_session.commit()
        
        # Calculate search quality metrics
        search_quality = await analytics_service.calculate_search_quality_metrics(
            test_tenant.id,
            start_date=date.today() - timedelta(days=7),
            end_date=date.today()
        )
        
        assert search_quality is not None
        assert hasattr(search_quality, 'click_through_rate')
        assert hasattr(search_quality, 'result_relevance_score')
        assert hasattr(search_quality, 'user_satisfaction_rating')
        assert hasattr(search_quality, 'search_success_rate')
    
    @pytest.mark.asyncio
    async def test_analyze_trends(self, analytics_service: AnalyticsService, test_tenant: Tenant, db_session: Session):
        """Test trend analysis functionality."""
        # Create historical metrics
        for i in range(10):
            metric = AnalyticsMetric(
                tenant_id=test_tenant.id,
                metric_type=AnalyticsMetricType.SEARCH_QUALITY,
                metric_name="test_metric",
                metric_value=0.5 + (i * 0.02),  # Increasing trend
                measurement_date=date.today() - timedelta(days=10-i),
                sample_size=100
            )
            db_session.add(metric)
        
        db_session.commit()
        
        # Analyze trends
        trend_result = await analytics_service.analyze_trends(
            test_tenant.id,
            "test_metric",
            date.today() - timedelta(days=10),
            date.today()
        )
        
        assert trend_result is not None
        assert trend_result.trend_direction in ["increasing", "decreasing", "stable"]
        assert 0.0 <= trend_result.trend_strength <= 1.0
        assert isinstance(trend_result.change_percentage, float)


class TestRecommendationService:
    """Test recommendation service functionality."""
    
    @pytest.fixture
    def recommendation_service(self, db_session: Session):
        """Create recommendation service instance."""
        return RecommendationService(db_session)
    
    @pytest.mark.asyncio
    async def test_create_recommendation(self, recommendation_service: RecommendationService, test_tenant: Tenant):
        """Test creating performance recommendation via service."""
        recommendation_data = CreatePerformanceRecommendationRequest(
            recommendation_type=RecommendationType.CONTENT_OPTIMIZATION,
            category="quality",
            title="Improve content freshness",
            description="Update outdated content to improve quality scores",
            rationale="Content freshness score is below target",
            priority=RecommendationPriority.MEDIUM,
            estimated_impact=0.12,
            implementation_effort="high",
            implementation_steps=["Audit content", "Update outdated items", "Validate changes"],
            estimated_duration_hours=60
        )
        
        recommendation = await recommendation_service.create_recommendation(test_tenant.id, recommendation_data)
        
        assert recommendation.id is not None
        assert recommendation.recommendation_type == RecommendationType.CONTENT_OPTIMIZATION
        assert recommendation.title == "Improve content freshness"
        assert recommendation.estimated_impact == 0.12
        assert recommendation.status == RecommendationStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_detect_improvement_opportunities(self, recommendation_service: RecommendationService, test_tenant: Tenant, db_session: Session):
        """Test detecting improvement opportunities."""
        # Create metrics that should trigger opportunities
        low_ctr_metric = AnalyticsMetric(
            tenant_id=test_tenant.id,
            metric_type=AnalyticsMetricType.SEARCH_QUALITY,
            metric_name="click_through_rate",
            metric_value=0.4,  # Below warning threshold
            measurement_date=date.today(),
            sample_size=1000
        )
        db_session.add(low_ctr_metric)
        db_session.commit()
        
        # Detect opportunities
        opportunities = await recommendation_service.detect_improvement_opportunities(
            test_tenant.id, analysis_period_days=30
        )
        
        assert len(opportunities) > 0
        assert any(opp.opportunity_type == "search_relevance" for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_track_recommendation_effectiveness(self, recommendation_service: RecommendationService, test_tenant: Tenant, db_session: Session):
        """Test tracking recommendation effectiveness."""
        # Create a recommendation
        recommendation = PerformanceRecommendation(
            tenant_id=test_tenant.id,
            recommendation_type=RecommendationType.SEARCH_IMPROVEMENT,
            title="Test Recommendation",
            description="Test description",
            priority=RecommendationPriority.MEDIUM,
            estimated_impact=0.1,
            implementation_effort="medium",
            baseline_metrics={"ctr": 0.5},
            target_metrics={"ctr": 0.6}
        )
        db_session.add(recommendation)
        db_session.commit()
        db_session.refresh(recommendation)
        
        # Track effectiveness
        actual_metrics = {"ctr": 0.65}
        effectiveness_score = await recommendation_service.track_recommendation_effectiveness(
            recommendation.id, actual_metrics
        )
        
        assert 0.0 <= effectiveness_score <= 2.0
        assert recommendation.actual_metrics == actual_metrics
        assert recommendation.effectiveness_score is not None
