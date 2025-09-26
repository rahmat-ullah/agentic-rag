"""
Analytics API Routes for Sprint 6 Story 6-04: Feedback Analytics and Insights System

This module provides FastAPI endpoints for analytics dashboard, metrics retrieval,
recommendation management, and real-time analytics data access.
"""

import uuid
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from agentic_rag.database.connection import get_database_session
from agentic_rag.api.dependencies.auth import get_current_user, require_role
from agentic_rag.models.auth import User
from agentic_rag.models.analytics import (
    AnalyticsMetric,
    PerformanceRecommendation,
    DashboardConfiguration,
    MetricAggregation,
    AnalyticsMetricType,
    RecommendationType,
    RecommendationStatus
)
from agentic_rag.schemas.analytics import (
    CreateAnalyticsMetricRequest,
    CreatePerformanceRecommendationRequest,
    UpdateRecommendationStatusRequest,
    CreateDashboardConfigurationRequest,
    MetricQueryRequest,
    AnalyticsMetricResponse,
    PerformanceRecommendationResponse,
    DashboardConfigurationResponse,
    AnalyticsDashboardResponse,
    SearchQualityMetricsResponse,
    UserSatisfactionScoreResponse,
    ContentQualityAssessmentResponse,
    AnalyticsMetricListResponse,
    PerformanceRecommendationListResponse,
    DashboardConfigurationListResponse
)
from agentic_rag.services.analytics_service import AnalyticsService
from agentic_rag.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/analytics", tags=["Analytics"])
logger = structlog.get_logger(__name__)


@router.post("/metrics", response_model=AnalyticsMetricResponse)
async def create_analytics_metric(
    metric_data: CreateAnalyticsMetricRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Create a new analytics metric."""
    try:
        # Require admin or analyst role
        require_role(current_user, ["admin", "analyst"])
        
        analytics_service = AnalyticsService(db)
        metric = await analytics_service.create_analytics_metric(
            current_user.tenant_id, metric_data
        )
        
        logger.info(
            "analytics_metric_created_via_api",
            metric_id=metric.id,
            metric_type=metric.metric_type,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        return AnalyticsMetricResponse.from_orm(metric)
        
    except Exception as e:
        logger.error("analytics_metric_creation_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create analytics metric: {str(e)}"
        )


@router.get("/metrics", response_model=AnalyticsMetricListResponse)
async def get_analytics_metrics(
    metric_types: Optional[List[AnalyticsMetricType]] = Query(None),
    metric_names: Optional[List[str]] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get analytics metrics with filtering and pagination."""
    try:
        # Build query
        query = db.query(AnalyticsMetric).filter(
            AnalyticsMetric.tenant_id == current_user.tenant_id
        )
        
        # Apply filters
        if metric_types:
            query = query.filter(AnalyticsMetric.metric_type.in_(metric_types))
        
        if metric_names:
            query = query.filter(AnalyticsMetric.metric_name.in_(metric_names))
        
        if start_date:
            query = query.filter(AnalyticsMetric.measurement_date >= start_date)
        
        if end_date:
            query = query.filter(AnalyticsMetric.measurement_date <= end_date)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        metrics = query.order_by(desc(AnalyticsMetric.measurement_date)).offset(offset).limit(page_size).all()
        
        # Convert to response models
        metric_responses = [AnalyticsMetricResponse.from_orm(metric) for metric in metrics]
        
        return AnalyticsMetricListResponse(
            items=metric_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )
        
    except Exception as e:
        logger.error("analytics_metrics_retrieval_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics metrics: {str(e)}"
        )


@router.get("/dashboard", response_model=AnalyticsDashboardResponse)
async def get_analytics_dashboard(
    dashboard_id: Optional[uuid.UUID] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get comprehensive analytics dashboard data."""
    try:
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()
        
        analytics_service = AnalyticsService(db)
        
        # Get dashboard configuration
        dashboard_config = None
        if dashboard_id:
            dashboard_config = db.query(DashboardConfiguration).filter(
                and_(
                    DashboardConfiguration.id == dashboard_id,
                    DashboardConfiguration.tenant_id == current_user.tenant_id
                )
            ).first()
        
        if not dashboard_config:
            # Use default dashboard
            dashboard_config = db.query(DashboardConfiguration).filter(
                and_(
                    DashboardConfiguration.tenant_id == current_user.tenant_id,
                    DashboardConfiguration.is_default == True
                )
            ).first()
        
        # Calculate metrics
        search_quality = await analytics_service.calculate_search_quality_metrics(
            current_user.tenant_id, start_date, end_date
        )
        
        user_satisfaction = await analytics_service.calculate_user_satisfaction_score(
            current_user.tenant_id, start_date, end_date
        )
        
        content_quality = await analytics_service.assess_content_quality(
            current_user.tenant_id, start_date, end_date
        )
        
        # Get active recommendations
        active_recommendations = db.query(PerformanceRecommendation).filter(
            and_(
                PerformanceRecommendation.tenant_id == current_user.tenant_id,
                PerformanceRecommendation.status.in_([
                    RecommendationStatus.PENDING,
                    RecommendationStatus.IN_PROGRESS
                ])
            )
        ).order_by(desc(PerformanceRecommendation.priority)).limit(10).all()
        
        recommendation_responses = [
            PerformanceRecommendationResponse.from_orm(rec) for rec in active_recommendations
        ]
        
        # Placeholder metrics for system performance, user engagement, and learning effectiveness
        system_performance_metrics = {
            "avg_response_time": 0.25,
            "system_uptime": 0.999,
            "error_rate": 0.001
        }
        
        user_engagement_metrics = {
            "daily_active_users": 1250,
            "session_duration": 12.5,
            "return_user_rate": 0.72
        }
        
        learning_effectiveness_metrics = {
            "learning_accuracy": 0.88,
            "improvement_rate": 0.15
        }
        
        return AnalyticsDashboardResponse(
            dashboard_id=dashboard_config.id if dashboard_config else uuid.uuid4(),
            dashboard_name=dashboard_config.name if dashboard_config else "Default Dashboard",
            last_updated=datetime.now(),
            search_quality_metrics=search_quality,
            user_satisfaction_score=user_satisfaction,
            content_quality_assessment=content_quality,
            system_performance_metrics=system_performance_metrics,
            user_engagement_metrics=user_engagement_metrics,
            learning_effectiveness_metrics=learning_effectiveness_metrics,
            active_recommendations=recommendation_responses,
            recent_alerts=[]
        )
        
    except Exception as e:
        logger.error("analytics_dashboard_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics dashboard: {str(e)}"
        )


@router.get("/search-quality", response_model=SearchQualityMetricsResponse)
async def get_search_quality_metrics(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get search quality metrics and analysis."""
    try:
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()
        
        analytics_service = AnalyticsService(db)
        search_quality = await analytics_service.calculate_search_quality_metrics(
            current_user.tenant_id, start_date, end_date
        )
        
        return search_quality
        
    except Exception as e:
        logger.error("search_quality_metrics_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get search quality metrics: {str(e)}"
        )


@router.get("/user-satisfaction", response_model=UserSatisfactionScoreResponse)
async def get_user_satisfaction_score(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get user satisfaction scoring and analysis."""
    try:
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()
        
        analytics_service = AnalyticsService(db)
        satisfaction = await analytics_service.calculate_user_satisfaction_score(
            current_user.tenant_id, start_date, end_date
        )
        
        return satisfaction
        
    except Exception as e:
        logger.error("user_satisfaction_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user satisfaction score: {str(e)}"
        )


@router.get("/content-quality", response_model=ContentQualityAssessmentResponse)
async def get_content_quality_assessment(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get content quality assessment and analysis."""
    try:
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()
        
        analytics_service = AnalyticsService(db)
        content_quality = await analytics_service.assess_content_quality(
            current_user.tenant_id, start_date, end_date
        )
        
        return content_quality
        
    except Exception as e:
        logger.error("content_quality_assessment_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get content quality assessment: {str(e)}"
        )


@router.post("/recommendations", response_model=PerformanceRecommendationResponse)
async def create_performance_recommendation(
    recommendation_data: CreatePerformanceRecommendationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Create a new performance recommendation."""
    try:
        # Require admin or analyst role
        require_role(current_user, ["admin", "analyst"])

        recommendation_service = RecommendationService(db)
        recommendation = await recommendation_service.create_recommendation(
            current_user.tenant_id, recommendation_data
        )

        logger.info(
            "performance_recommendation_created_via_api",
            recommendation_id=recommendation.id,
            recommendation_type=recommendation.recommendation_type,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )

        return PerformanceRecommendationResponse.from_orm(recommendation)

    except Exception as e:
        logger.error("performance_recommendation_creation_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create performance recommendation: {str(e)}"
        )


@router.get("/recommendations", response_model=PerformanceRecommendationListResponse)
async def get_performance_recommendations(
    recommendation_types: Optional[List[RecommendationType]] = Query(None),
    statuses: Optional[List[RecommendationStatus]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get performance recommendations with filtering and pagination."""
    try:
        # Build query
        query = db.query(PerformanceRecommendation).filter(
            PerformanceRecommendation.tenant_id == current_user.tenant_id
        )

        # Apply filters
        if recommendation_types:
            query = query.filter(PerformanceRecommendation.recommendation_type.in_(recommendation_types))

        if statuses:
            query = query.filter(PerformanceRecommendation.status.in_(statuses))

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * page_size
        recommendations = query.order_by(desc(PerformanceRecommendation.priority), desc(PerformanceRecommendation.created_at)).offset(offset).limit(page_size).all()

        # Convert to response models
        recommendation_responses = [PerformanceRecommendationResponse.from_orm(rec) for rec in recommendations]

        return PerformanceRecommendationListResponse(
            items=recommendation_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )

    except Exception as e:
        logger.error("performance_recommendations_retrieval_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance recommendations: {str(e)}"
        )


@router.put("/recommendations/{recommendation_id}/status", response_model=PerformanceRecommendationResponse)
async def update_recommendation_status(
    recommendation_id: uuid.UUID,
    status_update: UpdateRecommendationStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Update the status of a performance recommendation."""
    try:
        # Require admin or analyst role
        require_role(current_user, ["admin", "analyst"])

        # Get recommendation
        recommendation = db.query(PerformanceRecommendation).filter(
            and_(
                PerformanceRecommendation.id == recommendation_id,
                PerformanceRecommendation.tenant_id == current_user.tenant_id
            )
        ).first()

        if not recommendation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Recommendation not found"
            )

        # Update status
        recommendation.status = status_update.status
        if status_update.assigned_to:
            recommendation.assigned_to = status_update.assigned_to

        # Update timestamps based on status
        if status_update.status == RecommendationStatus.IN_PROGRESS and not recommendation.implemented_at:
            recommendation.implemented_at = datetime.now()
        elif status_update.status == RecommendationStatus.COMPLETED and not recommendation.completed_at:
            recommendation.completed_at = datetime.now()

        # Update actual metrics if provided
        if status_update.actual_metrics:
            recommendation.actual_metrics = status_update.actual_metrics

            # Track effectiveness if completed
            if status_update.status == RecommendationStatus.COMPLETED:
                recommendation_service = RecommendationService(db)
                await recommendation_service.track_recommendation_effectiveness(
                    recommendation_id, status_update.actual_metrics
                )

        db.commit()
        db.refresh(recommendation)

        logger.info(
            "recommendation_status_updated_via_api",
            recommendation_id=recommendation_id,
            new_status=status_update.status,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )

        return PerformanceRecommendationResponse.from_orm(recommendation)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("recommendation_status_update_api_failed", error=str(e), recommendation_id=recommendation_id, user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update recommendation status: {str(e)}"
        )


@router.post("/recommendations/generate", response_model=List[PerformanceRecommendationResponse])
async def generate_recommendations(
    analysis_period_days: int = Query(30, ge=7, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Generate performance recommendations based on current metrics."""
    try:
        # Require admin or analyst role
        require_role(current_user, ["admin", "analyst"])

        recommendation_service = RecommendationService(db)

        # Detect improvement opportunities
        opportunities = await recommendation_service.detect_improvement_opportunities(
            current_user.tenant_id, analysis_period_days
        )

        # Generate recommendations from opportunities
        recommendations = await recommendation_service.generate_recommendations_from_opportunities(
            current_user.tenant_id, opportunities
        )

        # Prioritize recommendations
        prioritized_recommendations = await recommendation_service.prioritize_recommendations(
            current_user.tenant_id, recommendations
        )

        logger.info(
            "recommendations_generated_via_api",
            recommendation_count=len(prioritized_recommendations),
            opportunity_count=len(opportunities),
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )

        return [PerformanceRecommendationResponse.from_orm(rec) for rec in prioritized_recommendations]

    except Exception as e:
        logger.error("recommendation_generation_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )
