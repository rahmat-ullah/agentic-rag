"""
Analytics Service for Sprint 6 Story 6-04: Feedback Analytics and Insights System

This service handles analytics metric calculation, trend analysis, data aggregation,
and performance optimization to support comprehensive analytics and insights functionality.
"""

import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
import math

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc, text
from sqlalchemy.sql import select

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
from agentic_rag.models.feedback import UserFeedbackSubmission, FeedbackAggregation
from agentic_rag.models.learning import LearningPerformanceMetric, LearningAlgorithm
from agentic_rag.schemas.analytics import (
    CreateAnalyticsMetricRequest,
    CreatePerformanceRecommendationRequest,
    MetricQueryRequest,
    TrendAnalysisResponse,
    SearchQualityMetricsResponse,
    UserSatisfactionScoreResponse,
    ContentQualityAssessmentResponse
)


@dataclass
class MetricCalculationResult:
    """Result of metric calculation."""
    
    metric_value: float
    sample_size: int
    confidence_level: float
    data_quality_score: float
    calculation_method: str
    data_sources: List[str]
    metadata: Dict[str, Any]


@dataclass
class TrendAnalysisResult:
    """Result of trend analysis."""
    
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0-1.0
    change_percentage: float
    statistical_significance: float
    confidence_interval: Dict[str, float]
    metadata: Dict[str, Any]


class AnalyticsService:
    """Core analytics service for metric calculation and analysis."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = structlog.get_logger(__name__)
        
        # Cache for frequently accessed data
        self._metric_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = {}
    
    async def create_analytics_metric(
        self,
        tenant_id: uuid.UUID,
        metric_data: CreateAnalyticsMetricRequest
    ) -> AnalyticsMetric:
        """Create a new analytics metric."""
        try:
            # Calculate derived values
            previous_value = await self._get_previous_metric_value(
                tenant_id, metric_data.metric_name, metric_data.measurement_date
            )
            
            change_percentage = None
            trend_direction = None
            
            if previous_value is not None:
                change_percentage = ((metric_data.metric_value - previous_value) / previous_value) * 100
                if abs(change_percentage) < 1.0:  # Less than 1% change
                    trend_direction = "stable"
                elif change_percentage > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
            
            # Create metric
            metric = AnalyticsMetric(
                tenant_id=tenant_id,
                metric_type=metric_data.metric_type,
                metric_name=metric_data.metric_name,
                metric_category=metric_data.metric_category,
                metric_value=metric_data.metric_value,
                baseline_value=metric_data.baseline_value,
                target_value=metric_data.target_value,
                measurement_date=metric_data.measurement_date,
                measurement_period_start=metric_data.measurement_period_start,
                measurement_period_end=metric_data.measurement_period_end,
                dimension_values=metric_data.dimension_values,
                sample_size=metric_data.sample_size,
                confidence_level=metric_data.confidence_level,
                previous_value=previous_value,
                change_percentage=change_percentage,
                trend_direction=trend_direction,
                calculation_method=metric_data.calculation_method,
                data_sources=metric_data.data_sources,
                metric_metadata=metric_data.metric_metadata
            )
            
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            
            self.logger.info(
                "analytics_metric_created",
                metric_id=metric.id,
                metric_type=metric.metric_type,
                metric_name=metric.metric_name,
                metric_value=metric.metric_value,
                tenant_id=tenant_id
            )
            
            return metric
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("analytics_metric_creation_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def calculate_search_quality_metrics(
        self,
        tenant_id: uuid.UUID,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> SearchQualityMetricsResponse:
        """Calculate comprehensive search quality metrics."""
        try:
            if not start_date:
                start_date = date.today() - timedelta(days=30)
            if not end_date:
                end_date = date.today()
            
            # Calculate individual metrics
            ctr = await self._calculate_click_through_rate(tenant_id, start_date, end_date, dimensions)
            relevance = await self._calculate_result_relevance_score(tenant_id, start_date, end_date, dimensions)
            satisfaction = await self._calculate_user_satisfaction_rating(tenant_id, start_date, end_date, dimensions)
            success_rate = await self._calculate_search_success_rate(tenant_id, start_date, end_date, dimensions)
            avg_results = await self._calculate_average_results_per_query(tenant_id, start_date, end_date, dimensions)
            zero_results = await self._calculate_zero_results_rate(tenant_id, start_date, end_date, dimensions)
            refinement = await self._calculate_query_refinement_rate(tenant_id, start_date, end_date, dimensions)
            abandonment = await self._calculate_session_abandonment_rate(tenant_id, start_date, end_date, dimensions)
            
            # Get trend analysis
            trends = await self._analyze_search_quality_trends(tenant_id, start_date, end_date)
            
            # Get quality alerts
            alerts = await self._get_search_quality_alerts(tenant_id)
            
            # Get benchmark comparison
            benchmarks = await self._get_search_quality_benchmarks(tenant_id)
            
            return SearchQualityMetricsResponse(
                click_through_rate=ctr.metric_value,
                result_relevance_score=relevance.metric_value,
                user_satisfaction_rating=satisfaction.metric_value,
                search_success_rate=success_rate.metric_value,
                average_results_per_query=avg_results.metric_value,
                zero_results_rate=zero_results.metric_value,
                query_refinement_rate=refinement.metric_value,
                session_abandonment_rate=abandonment.metric_value,
                trend_analysis=trends,
                quality_alerts=alerts,
                benchmark_comparison=benchmarks
            )
            
        except Exception as e:
            self.logger.error("search_quality_metrics_calculation_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def calculate_user_satisfaction_score(
        self,
        tenant_id: uuid.UUID,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> UserSatisfactionScoreResponse:
        """Calculate comprehensive user satisfaction scoring."""
        try:
            if not start_date:
                start_date = date.today() - timedelta(days=30)
            if not end_date:
                end_date = date.today()
            
            # Calculate overall satisfaction score
            overall_score = await self._calculate_overall_satisfaction_score(tenant_id, start_date, end_date)
            
            # Calculate satisfaction by user segments
            satisfaction_by_segment = await self._calculate_satisfaction_by_segment(tenant_id, start_date, end_date)
            
            # Get satisfaction trends
            trends = await self._analyze_satisfaction_trends(tenant_id, start_date, end_date)
            
            # Perform correlation analysis
            correlations = await self._analyze_satisfaction_correlations(tenant_id, start_date, end_date)
            
            # Identify satisfaction drivers
            drivers = await self._identify_satisfaction_drivers(tenant_id, start_date, end_date)
            
            # Predict future satisfaction
            prediction_result = await self._predict_satisfaction(tenant_id, start_date, end_date)
            
            return UserSatisfactionScoreResponse(
                overall_satisfaction_score=overall_score,
                satisfaction_by_segment=satisfaction_by_segment,
                satisfaction_trends=trends,
                correlation_analysis=correlations,
                satisfaction_drivers=drivers,
                prediction_confidence=prediction_result["confidence"],
                predicted_satisfaction=prediction_result["predicted_score"]
            )
            
        except Exception as e:
            self.logger.error("user_satisfaction_calculation_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def assess_content_quality(
        self,
        tenant_id: uuid.UUID,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> ContentQualityAssessmentResponse:
        """Assess comprehensive content quality."""
        try:
            if not start_date:
                start_date = date.today() - timedelta(days=30)
            if not end_date:
                end_date = date.today()
            
            # Calculate overall quality score
            overall_score = await self._calculate_overall_content_quality_score(tenant_id, start_date, end_date)
            
            # Calculate quality by category
            quality_by_category = await self._calculate_quality_by_category(tenant_id, start_date, end_date)
            
            # Get quality trends
            trends = await self._analyze_content_quality_trends(tenant_id, start_date, end_date)
            
            # Identify quality issues
            issues = await self._identify_content_quality_issues(tenant_id, start_date, end_date)
            
            # Find improvement opportunities
            opportunities = await self._find_content_improvement_opportunities(tenant_id, start_date, end_date)
            
            # Get quality alerts
            alerts = await self._get_content_quality_alerts(tenant_id)
            
            # Calculate component scores
            freshness_score = await self._calculate_content_freshness_score(tenant_id, start_date, end_date)
            accuracy_score = await self._calculate_content_accuracy_score(tenant_id, start_date, end_date)
            completeness_score = await self._calculate_content_completeness_score(tenant_id, start_date, end_date)
            
            return ContentQualityAssessmentResponse(
                overall_quality_score=overall_score,
                quality_by_category=quality_by_category,
                quality_trends=trends,
                quality_issues=issues,
                improvement_opportunities=opportunities,
                quality_alerts=alerts,
                content_freshness_score=freshness_score,
                accuracy_score=accuracy_score,
                completeness_score=completeness_score
            )
            
        except Exception as e:
            self.logger.error("content_quality_assessment_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def analyze_trends(
        self,
        tenant_id: uuid.UUID,
        metric_name: str,
        start_date: date,
        end_date: date,
        aggregation_level: str = "daily"
    ) -> TrendAnalysisResult:
        """Analyze trends for a specific metric."""
        try:
            # Get historical data
            metrics = self.db.query(AnalyticsMetric).filter(
                and_(
                    AnalyticsMetric.tenant_id == tenant_id,
                    AnalyticsMetric.metric_name == metric_name,
                    AnalyticsMetric.measurement_date >= start_date,
                    AnalyticsMetric.measurement_date <= end_date
                )
            ).order_by(AnalyticsMetric.measurement_date).all()
            
            if len(metrics) < 2:
                return TrendAnalysisResult(
                    trend_direction="stable",
                    trend_strength=0.0,
                    change_percentage=0.0,
                    statistical_significance=0.0,
                    confidence_interval={"lower": 0.0, "upper": 0.0},
                    metadata={"insufficient_data": True}
                )
            
            # Extract values and calculate trend
            values = [m.metric_value for m in metrics]
            dates = [(m.measurement_date - start_date).days for m in metrics]
            
            # Calculate linear regression
            n = len(values)
            sum_x = sum(dates)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(dates, values))
            sum_x2 = sum(x * x for x in dates)
            
            # Slope calculation
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend direction and strength
            if abs(slope) < 0.001:
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "increasing"
                trend_strength = min(abs(slope) * 100, 1.0)
            else:
                trend_direction = "decreasing"
                trend_strength = min(abs(slope) * 100, 1.0)
            
            # Calculate change percentage
            first_value = values[0]
            last_value = values[-1]
            change_percentage = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0.0
            
            # Calculate statistical significance (simplified)
            variance = statistics.variance(values) if len(values) > 1 else 0.0
            std_error = math.sqrt(variance / n) if variance > 0 else 0.0
            t_statistic = abs(slope) / std_error if std_error > 0 else 0.0
            statistical_significance = min(t_statistic / 2.0, 1.0)  # Simplified p-value approximation
            
            # Calculate confidence interval
            margin_of_error = 1.96 * std_error  # 95% confidence
            confidence_interval = {
                "lower": slope - margin_of_error,
                "upper": slope + margin_of_error
            }
            
            return TrendAnalysisResult(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                change_percentage=change_percentage,
                statistical_significance=statistical_significance,
                confidence_interval=confidence_interval,
                metadata={
                    "sample_size": n,
                    "slope": slope,
                    "variance": variance,
                    "std_error": std_error
                }
            )
            
        except Exception as e:
            self.logger.error("trend_analysis_failed", error=str(e), metric_name=metric_name, tenant_id=tenant_id)
            raise

    # Helper methods for metric calculations

    async def _get_previous_metric_value(
        self,
        tenant_id: uuid.UUID,
        metric_name: str,
        current_date: date
    ) -> Optional[float]:
        """Get the previous value for a metric."""
        try:
            previous_metric = self.db.query(AnalyticsMetric).filter(
                and_(
                    AnalyticsMetric.tenant_id == tenant_id,
                    AnalyticsMetric.metric_name == metric_name,
                    AnalyticsMetric.measurement_date < current_date
                )
            ).order_by(desc(AnalyticsMetric.measurement_date)).first()

            return previous_metric.metric_value if previous_metric else None

        except Exception as e:
            self.logger.error("previous_metric_value_retrieval_failed", error=str(e))
            return None

    async def _calculate_click_through_rate(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> MetricCalculationResult:
        """Calculate click-through rate from feedback data."""
        try:
            # Get search result feedback with clicks
            query = self.db.query(UserFeedbackSubmission).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.feedback_type == "search_result",
                    UserFeedbackSubmission.created_at >= datetime.combine(start_date, datetime.min.time()),
                    UserFeedbackSubmission.created_at <= datetime.combine(end_date, datetime.max.time())
                )
            )

            total_impressions = query.count()
            clicks = query.filter(UserFeedbackSubmission.rating > 0).count()

            ctr = (clicks / total_impressions) if total_impressions > 0 else 0.0

            return MetricCalculationResult(
                metric_value=ctr,
                sample_size=total_impressions,
                confidence_level=0.95,
                data_quality_score=1.0 if total_impressions >= 100 else total_impressions / 100,
                calculation_method="clicks / impressions",
                data_sources=["user_feedback"],
                metadata={"clicks": clicks, "impressions": total_impressions}
            )

        except Exception as e:
            self.logger.error("ctr_calculation_failed", error=str(e))
            return MetricCalculationResult(
                metric_value=0.0, sample_size=0, confidence_level=0.0,
                data_quality_score=0.0, calculation_method="error",
                data_sources=[], metadata={"error": str(e)}
            )

    async def _calculate_result_relevance_score(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> MetricCalculationResult:
        """Calculate result relevance score from user ratings."""
        try:
            # Get search result ratings
            ratings = self.db.query(UserFeedbackSubmission.rating).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.feedback_type == "search_result",
                    UserFeedbackSubmission.rating.isnot(None),
                    UserFeedbackSubmission.created_at >= datetime.combine(start_date, datetime.min.time()),
                    UserFeedbackSubmission.created_at <= datetime.combine(end_date, datetime.max.time())
                )
            ).all()

            if not ratings:
                return MetricCalculationResult(
                    metric_value=0.0, sample_size=0, confidence_level=0.0,
                    data_quality_score=0.0, calculation_method="no_data",
                    data_sources=["user_feedback"], metadata={}
                )

            rating_values = [r[0] for r in ratings if r[0] is not None]
            avg_rating = sum(rating_values) / len(rating_values)

            # Normalize to 0-1 scale (assuming ratings are 1-5)
            relevance_score = (avg_rating - 1) / 4 if avg_rating > 1 else 0.0

            return MetricCalculationResult(
                metric_value=relevance_score,
                sample_size=len(rating_values),
                confidence_level=0.95,
                data_quality_score=1.0 if len(rating_values) >= 50 else len(rating_values) / 50,
                calculation_method="average_rating_normalized",
                data_sources=["user_feedback"],
                metadata={"avg_rating": avg_rating, "rating_count": len(rating_values)}
            )

        except Exception as e:
            self.logger.error("relevance_score_calculation_failed", error=str(e))
            return MetricCalculationResult(
                metric_value=0.0, sample_size=0, confidence_level=0.0,
                data_quality_score=0.0, calculation_method="error",
                data_sources=[], metadata={"error": str(e)}
            )

    async def _calculate_user_satisfaction_rating(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> MetricCalculationResult:
        """Calculate overall user satisfaction rating."""
        try:
            # Get all feedback ratings
            ratings = self.db.query(UserFeedbackSubmission.rating).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.rating.isnot(None),
                    UserFeedbackSubmission.created_at >= datetime.combine(start_date, datetime.min.time()),
                    UserFeedbackSubmission.created_at <= datetime.combine(end_date, datetime.max.time())
                )
            ).all()

            if not ratings:
                return MetricCalculationResult(
                    metric_value=0.0, sample_size=0, confidence_level=0.0,
                    data_quality_score=0.0, calculation_method="no_data",
                    data_sources=["user_feedback"], metadata={}
                )

            rating_values = [r[0] for r in ratings if r[0] is not None]
            avg_satisfaction = sum(rating_values) / len(rating_values)

            return MetricCalculationResult(
                metric_value=avg_satisfaction,
                sample_size=len(rating_values),
                confidence_level=0.95,
                data_quality_score=1.0 if len(rating_values) >= 100 else len(rating_values) / 100,
                calculation_method="average_rating",
                data_sources=["user_feedback"],
                metadata={"rating_count": len(rating_values)}
            )

        except Exception as e:
            self.logger.error("satisfaction_rating_calculation_failed", error=str(e))
            return MetricCalculationResult(
                metric_value=0.0, sample_size=0, confidence_level=0.0,
                data_quality_score=0.0, calculation_method="error",
                data_sources=[], metadata={"error": str(e)}
            )

    async def _calculate_search_success_rate(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> MetricCalculationResult:
        """Calculate search success rate based on positive feedback."""
        try:
            # Get search feedback
            total_searches = self.db.query(UserFeedbackSubmission).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.feedback_type == "search_result",
                    UserFeedbackSubmission.created_at >= datetime.combine(start_date, datetime.min.time()),
                    UserFeedbackSubmission.created_at <= datetime.combine(end_date, datetime.max.time())
                )
            ).count()

            successful_searches = self.db.query(UserFeedbackSubmission).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.feedback_type == "search_result",
                    UserFeedbackSubmission.rating >= 3,  # Consider 3+ as successful
                    UserFeedbackSubmission.created_at >= datetime.combine(start_date, datetime.min.time()),
                    UserFeedbackSubmission.created_at <= datetime.combine(end_date, datetime.max.time())
                )
            ).count()

            success_rate = (successful_searches / total_searches) if total_searches > 0 else 0.0

            return MetricCalculationResult(
                metric_value=success_rate,
                sample_size=total_searches,
                confidence_level=0.95,
                data_quality_score=1.0 if total_searches >= 100 else total_searches / 100,
                calculation_method="successful_searches / total_searches",
                data_sources=["user_feedback"],
                metadata={"successful_searches": successful_searches, "total_searches": total_searches}
            )

        except Exception as e:
            self.logger.error("search_success_rate_calculation_failed", error=str(e))
            return MetricCalculationResult(
                metric_value=0.0, sample_size=0, confidence_level=0.0,
                data_quality_score=0.0, calculation_method="error",
                data_sources=[], metadata={"error": str(e)}
            )

    async def _calculate_average_results_per_query(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> MetricCalculationResult:
        """Calculate average number of results per query."""
        try:
            # This would typically come from search logs
            # For now, we'll use a placeholder calculation
            avg_results = 8.5  # Placeholder value

            return MetricCalculationResult(
                metric_value=avg_results,
                sample_size=1000,  # Placeholder
                confidence_level=0.95,
                data_quality_score=0.8,  # Placeholder
                calculation_method="search_log_analysis",
                data_sources=["search_logs"],
                metadata={"placeholder": True}
            )

        except Exception as e:
            self.logger.error("avg_results_calculation_failed", error=str(e))
            return MetricCalculationResult(
                metric_value=0.0, sample_size=0, confidence_level=0.0,
                data_quality_score=0.0, calculation_method="error",
                data_sources=[], metadata={"error": str(e)}
            )

    async def _calculate_zero_results_rate(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> MetricCalculationResult:
        """Calculate rate of queries returning zero results."""
        try:
            # Placeholder calculation - would come from search logs
            zero_results_rate = 0.05  # 5% placeholder

            return MetricCalculationResult(
                metric_value=zero_results_rate,
                sample_size=1000,  # Placeholder
                confidence_level=0.95,
                data_quality_score=0.8,
                calculation_method="search_log_analysis",
                data_sources=["search_logs"],
                metadata={"placeholder": True}
            )

        except Exception as e:
            self.logger.error("zero_results_rate_calculation_failed", error=str(e))
            return MetricCalculationResult(
                metric_value=0.0, sample_size=0, confidence_level=0.0,
                data_quality_score=0.0, calculation_method="error",
                data_sources=[], metadata={"error": str(e)}
            )

    async def _calculate_query_refinement_rate(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> MetricCalculationResult:
        """Calculate rate of query refinements."""
        try:
            # Placeholder calculation
            refinement_rate = 0.23  # 23% placeholder

            return MetricCalculationResult(
                metric_value=refinement_rate,
                sample_size=1000,
                confidence_level=0.95,
                data_quality_score=0.8,
                calculation_method="session_analysis",
                data_sources=["search_logs"],
                metadata={"placeholder": True}
            )

        except Exception as e:
            self.logger.error("query_refinement_rate_calculation_failed", error=str(e))
            return MetricCalculationResult(
                metric_value=0.0, sample_size=0, confidence_level=0.0,
                data_quality_score=0.0, calculation_method="error",
                data_sources=[], metadata={"error": str(e)}
            )

    async def _calculate_session_abandonment_rate(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> MetricCalculationResult:
        """Calculate session abandonment rate."""
        try:
            # Placeholder calculation
            abandonment_rate = 0.12  # 12% placeholder

            return MetricCalculationResult(
                metric_value=abandonment_rate,
                sample_size=1000,
                confidence_level=0.95,
                data_quality_score=0.8,
                calculation_method="session_analysis",
                data_sources=["user_sessions"],
                metadata={"placeholder": True}
            )

        except Exception as e:
            self.logger.error("session_abandonment_rate_calculation_failed", error=str(e))
            return MetricCalculationResult(
                metric_value=0.0, sample_size=0, confidence_level=0.0,
                data_quality_score=0.0, calculation_method="error",
                data_sources=[], metadata={"error": str(e)}
            )

    # Trend analysis helper methods

    async def _analyze_search_quality_trends(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date
    ) -> List[TrendAnalysisResponse]:
        """Analyze trends for search quality metrics."""
        try:
            trends = []

            # Analyze trends for key search quality metrics
            metric_names = [
                "click_through_rate",
                "result_relevance_score",
                "user_satisfaction_rating",
                "search_success_rate"
            ]

            for metric_name in metric_names:
                trend_result = await self.analyze_trends(
                    tenant_id, metric_name, start_date, end_date
                )

                trends.append(TrendAnalysisResponse(
                    metric_name=metric_name,
                    trend_direction=trend_result.trend_direction,
                    trend_strength=trend_result.trend_strength,
                    change_percentage=trend_result.change_percentage,
                    statistical_significance=trend_result.statistical_significance,
                    confidence_interval=trend_result.confidence_interval,
                    trend_metadata=trend_result.metadata
                ))

            return trends

        except Exception as e:
            self.logger.error("search_quality_trends_analysis_failed", error=str(e))
            return []

    async def _get_search_quality_alerts(self, tenant_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Get search quality alerts."""
        try:
            alerts = []

            # Check for quality degradation
            recent_ctr = await self._calculate_click_through_rate(
                tenant_id, date.today() - timedelta(days=7), date.today()
            )

            if recent_ctr.metric_value < 0.5:  # Below 50% CTR
                alerts.append({
                    "type": "quality_degradation",
                    "metric": "click_through_rate",
                    "severity": "medium",
                    "message": f"Click-through rate has dropped to {recent_ctr.metric_value:.1%}",
                    "threshold": 0.5,
                    "current_value": recent_ctr.metric_value
                })

            return alerts

        except Exception as e:
            self.logger.error("search_quality_alerts_failed", error=str(e))
            return []

    async def _get_search_quality_benchmarks(self, tenant_id: uuid.UUID) -> Dict[str, float]:
        """Get search quality benchmarks."""
        try:
            # Industry benchmarks (placeholder values)
            return {
                "industry_average_ctr": 0.60,
                "industry_average_satisfaction": 4.0,
                "industry_average_success_rate": 0.85
            }

        except Exception as e:
            self.logger.error("search_quality_benchmarks_failed", error=str(e))
            return {}
