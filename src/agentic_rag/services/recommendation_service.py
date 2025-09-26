"""
Recommendation Service for Sprint 6 Story 6-04: Feedback Analytics and Insights System

This service handles automated performance improvement recommendation generation,
opportunity detection, prioritization, and effectiveness tracking.
"""

import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from agentic_rag.models.analytics import (
    AnalyticsMetric,
    PerformanceRecommendation,
    AnalyticsMetricType,
    RecommendationType,
    RecommendationStatus,
    RecommendationPriority,
    ImplementationEffort
)
from agentic_rag.models.feedback import UserFeedbackSubmission, FeedbackAggregation
from agentic_rag.models.learning import LearningPerformanceMetric
from agentic_rag.schemas.analytics import CreatePerformanceRecommendationRequest


@dataclass
class RecommendationOpportunity:
    """Detected improvement opportunity."""
    
    opportunity_type: str
    title: str
    description: str
    rationale: str
    estimated_impact: float
    implementation_effort: str
    priority: str
    related_metrics: List[uuid.UUID]
    metadata: Dict[str, Any]


class RecommendationService:
    """Service for generating and managing performance improvement recommendations."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = structlog.get_logger(__name__)
        
        # Recommendation thresholds and rules
        self.quality_thresholds = {
            "click_through_rate": {"critical": 0.3, "warning": 0.5, "target": 0.7},
            "user_satisfaction": {"critical": 2.5, "warning": 3.5, "target": 4.5},
            "search_success_rate": {"critical": 0.6, "warning": 0.8, "target": 0.9},
            "content_quality": {"critical": 0.6, "warning": 0.75, "target": 0.9}
        }
    
    async def detect_improvement_opportunities(
        self,
        tenant_id: uuid.UUID,
        analysis_period_days: int = 30
    ) -> List[RecommendationOpportunity]:
        """Detect improvement opportunities based on metrics analysis."""
        try:
            opportunities = []
            end_date = date.today()
            start_date = end_date - timedelta(days=analysis_period_days)
            
            # Analyze search quality opportunities
            search_opportunities = await self._detect_search_quality_opportunities(
                tenant_id, start_date, end_date
            )
            opportunities.extend(search_opportunities)
            
            # Analyze user satisfaction opportunities
            satisfaction_opportunities = await self._detect_satisfaction_opportunities(
                tenant_id, start_date, end_date
            )
            opportunities.extend(satisfaction_opportunities)
            
            # Analyze content quality opportunities
            content_opportunities = await self._detect_content_quality_opportunities(
                tenant_id, start_date, end_date
            )
            opportunities.extend(content_opportunities)
            
            # Analyze learning effectiveness opportunities
            learning_opportunities = await self._detect_learning_opportunities(
                tenant_id, start_date, end_date
            )
            opportunities.extend(learning_opportunities)
            
            # Sort by estimated impact (descending)
            opportunities.sort(key=lambda x: x.estimated_impact, reverse=True)
            
            self.logger.info(
                "improvement_opportunities_detected",
                tenant_id=tenant_id,
                opportunity_count=len(opportunities),
                analysis_period_days=analysis_period_days
            )
            
            return opportunities
            
        except Exception as e:
            self.logger.error("opportunity_detection_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def create_recommendation(
        self,
        tenant_id: uuid.UUID,
        recommendation_data: CreatePerformanceRecommendationRequest
    ) -> PerformanceRecommendation:
        """Create a new performance recommendation."""
        try:
            recommendation = PerformanceRecommendation(
                tenant_id=tenant_id,
                recommendation_type=recommendation_data.recommendation_type,
                category=recommendation_data.category,
                title=recommendation_data.title,
                description=recommendation_data.description,
                rationale=recommendation_data.rationale,
                priority=recommendation_data.priority,
                estimated_impact=recommendation_data.estimated_impact,
                implementation_effort=recommendation_data.implementation_effort,
                implementation_steps=recommendation_data.implementation_steps,
                required_resources=recommendation_data.required_resources,
                estimated_duration_hours=recommendation_data.estimated_duration_hours,
                related_metrics=recommendation_data.related_metrics,
                recommendation_metadata=recommendation_data.recommendation_metadata
            )
            
            self.db.add(recommendation)
            self.db.commit()
            self.db.refresh(recommendation)
            
            self.logger.info(
                "performance_recommendation_created",
                recommendation_id=recommendation.id,
                recommendation_type=recommendation.recommendation_type,
                priority=recommendation.priority,
                estimated_impact=recommendation.estimated_impact,
                tenant_id=tenant_id
            )
            
            return recommendation
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("recommendation_creation_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def generate_recommendations_from_opportunities(
        self,
        tenant_id: uuid.UUID,
        opportunities: List[RecommendationOpportunity]
    ) -> List[PerformanceRecommendation]:
        """Generate recommendations from detected opportunities."""
        try:
            recommendations = []
            
            for opportunity in opportunities:
                # Check if similar recommendation already exists
                existing = await self._check_existing_recommendation(
                    tenant_id, opportunity.opportunity_type, opportunity.title
                )
                
                if existing:
                    self.logger.info(
                        "skipping_duplicate_recommendation",
                        opportunity_type=opportunity.opportunity_type,
                        title=opportunity.title
                    )
                    continue
                
                # Create recommendation request
                recommendation_data = CreatePerformanceRecommendationRequest(
                    recommendation_type=self._map_opportunity_to_recommendation_type(opportunity.opportunity_type),
                    category=opportunity.opportunity_type,
                    title=opportunity.title,
                    description=opportunity.description,
                    rationale=opportunity.rationale,
                    priority=RecommendationPriority(opportunity.priority),
                    estimated_impact=opportunity.estimated_impact,
                    implementation_effort=ImplementationEffort(opportunity.implementation_effort),
                    related_metrics=opportunity.related_metrics,
                    recommendation_metadata=opportunity.metadata
                )
                
                # Create recommendation
                recommendation = await self.create_recommendation(tenant_id, recommendation_data)
                recommendations.append(recommendation)
            
            self.logger.info(
                "recommendations_generated_from_opportunities",
                tenant_id=tenant_id,
                recommendation_count=len(recommendations),
                opportunity_count=len(opportunities)
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error("recommendation_generation_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def prioritize_recommendations(
        self,
        tenant_id: uuid.UUID,
        recommendations: List[PerformanceRecommendation]
    ) -> List[PerformanceRecommendation]:
        """Prioritize recommendations based on impact, effort, and urgency."""
        try:
            # Calculate priority scores
            scored_recommendations = []
            
            for rec in recommendations:
                # Priority score calculation
                impact_score = rec.estimated_impact * 100  # 0-100
                
                # Effort penalty (lower effort = higher score)
                effort_penalties = {
                    ImplementationEffort.LOW: 0,
                    ImplementationEffort.MEDIUM: 10,
                    ImplementationEffort.HIGH: 25,
                    ImplementationEffort.VERY_HIGH: 40
                }
                effort_penalty = effort_penalties.get(rec.implementation_effort, 20)
                
                # Priority bonus
                priority_bonuses = {
                    RecommendationPriority.CRITICAL: 50,
                    RecommendationPriority.HIGH: 30,
                    RecommendationPriority.MEDIUM: 10,
                    RecommendationPriority.LOW: 0
                }
                priority_bonus = priority_bonuses.get(rec.priority, 0)
                
                # Calculate final score
                priority_score = impact_score + priority_bonus - effort_penalty
                
                scored_recommendations.append((rec, priority_score))
            
            # Sort by priority score (descending)
            scored_recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # Return sorted recommendations
            prioritized = [rec for rec, score in scored_recommendations]
            
            self.logger.info(
                "recommendations_prioritized",
                tenant_id=tenant_id,
                recommendation_count=len(prioritized)
            )
            
            return prioritized
            
        except Exception as e:
            self.logger.error("recommendation_prioritization_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def track_recommendation_effectiveness(
        self,
        recommendation_id: uuid.UUID,
        actual_metrics: Dict[str, float]
    ) -> float:
        """Track the effectiveness of an implemented recommendation."""
        try:
            recommendation = self.db.query(PerformanceRecommendation).get(recommendation_id)
            if not recommendation:
                raise ValueError(f"Recommendation {recommendation_id} not found")
            
            # Update actual metrics
            recommendation.actual_metrics = actual_metrics
            
            # Calculate effectiveness score
            if recommendation.target_metrics:
                effectiveness_scores = []
                
                for metric_name, target_value in recommendation.target_metrics.items():
                    actual_value = actual_metrics.get(metric_name)
                    baseline_value = recommendation.baseline_metrics.get(metric_name) if recommendation.baseline_metrics else None
                    
                    if actual_value is not None and baseline_value is not None:
                        # Calculate improvement ratio
                        target_improvement = target_value - baseline_value
                        actual_improvement = actual_value - baseline_value
                        
                        if target_improvement != 0:
                            effectiveness = actual_improvement / target_improvement
                            effectiveness_scores.append(min(max(effectiveness, 0.0), 2.0))  # Cap at 200%
                
                if effectiveness_scores:
                    recommendation.effectiveness_score = sum(effectiveness_scores) / len(effectiveness_scores)
                else:
                    recommendation.effectiveness_score = 0.0
            else:
                recommendation.effectiveness_score = 0.5  # Default if no targets set
            
            self.db.commit()
            
            self.logger.info(
                "recommendation_effectiveness_tracked",
                recommendation_id=recommendation_id,
                effectiveness_score=recommendation.effectiveness_score
            )
            
            return recommendation.effectiveness_score
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("recommendation_effectiveness_tracking_failed", error=str(e), recommendation_id=recommendation_id)
            raise
    
    # Helper methods for opportunity detection
    
    async def _detect_search_quality_opportunities(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date
    ) -> List[RecommendationOpportunity]:
        """Detect search quality improvement opportunities."""
        opportunities = []
        
        try:
            # Check click-through rate
            ctr_metrics = self.db.query(AnalyticsMetric).filter(
                and_(
                    AnalyticsMetric.tenant_id == tenant_id,
                    AnalyticsMetric.metric_name == "click_through_rate",
                    AnalyticsMetric.measurement_date >= start_date,
                    AnalyticsMetric.measurement_date <= end_date
                )
            ).order_by(desc(AnalyticsMetric.measurement_date)).first()
            
            if ctr_metrics and ctr_metrics.metric_value < self.quality_thresholds["click_through_rate"]["warning"]:
                priority = "critical" if ctr_metrics.metric_value < self.quality_thresholds["click_through_rate"]["critical"] else "high"
                
                opportunities.append(RecommendationOpportunity(
                    opportunity_type="search_relevance",
                    title="Improve search result relevance",
                    description="Click-through rate is below target, indicating poor search result relevance",
                    rationale=f"Current CTR is {ctr_metrics.metric_value:.1%}, target is {self.quality_thresholds['click_through_rate']['target']:.1%}",
                    estimated_impact=0.15,
                    implementation_effort="medium",
                    priority=priority,
                    related_metrics=[ctr_metrics.id],
                    metadata={"current_ctr": ctr_metrics.metric_value, "target_ctr": self.quality_thresholds["click_through_rate"]["target"]}
                ))
            
            return opportunities
            
        except Exception as e:
            self.logger.error("search_quality_opportunity_detection_failed", error=str(e))
            return opportunities

    async def _detect_satisfaction_opportunities(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date
    ) -> List[RecommendationOpportunity]:
        """Detect user satisfaction improvement opportunities."""
        opportunities = []

        try:
            # Check user satisfaction ratings
            satisfaction_metrics = self.db.query(AnalyticsMetric).filter(
                and_(
                    AnalyticsMetric.tenant_id == tenant_id,
                    AnalyticsMetric.metric_name == "user_satisfaction_rating",
                    AnalyticsMetric.measurement_date >= start_date,
                    AnalyticsMetric.measurement_date <= end_date
                )
            ).order_by(desc(AnalyticsMetric.measurement_date)).first()

            if satisfaction_metrics and satisfaction_metrics.metric_value < self.quality_thresholds["user_satisfaction"]["warning"]:
                priority = "critical" if satisfaction_metrics.metric_value < self.quality_thresholds["user_satisfaction"]["critical"] else "high"

                opportunities.append(RecommendationOpportunity(
                    opportunity_type="user_experience",
                    title="Improve user experience and satisfaction",
                    description="User satisfaction ratings are below target levels",
                    rationale=f"Current satisfaction is {satisfaction_metrics.metric_value:.1f}/5.0, target is {self.quality_thresholds['user_satisfaction']['target']:.1f}/5.0",
                    estimated_impact=0.20,
                    implementation_effort="medium",
                    priority=priority,
                    related_metrics=[satisfaction_metrics.id],
                    metadata={"current_satisfaction": satisfaction_metrics.metric_value, "target_satisfaction": self.quality_thresholds["user_satisfaction"]["target"]}
                ))

            return opportunities

        except Exception as e:
            self.logger.error("satisfaction_opportunity_detection_failed", error=str(e))
            return opportunities

    async def _detect_content_quality_opportunities(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date
    ) -> List[RecommendationOpportunity]:
        """Detect content quality improvement opportunities."""
        opportunities = []

        try:
            # Check content quality metrics
            quality_metrics = self.db.query(AnalyticsMetric).filter(
                and_(
                    AnalyticsMetric.tenant_id == tenant_id,
                    AnalyticsMetric.metric_name == "overall_content_quality",
                    AnalyticsMetric.measurement_date >= start_date,
                    AnalyticsMetric.measurement_date <= end_date
                )
            ).order_by(desc(AnalyticsMetric.measurement_date)).first()

            if quality_metrics and quality_metrics.metric_value < self.quality_thresholds["content_quality"]["warning"]:
                priority = "critical" if quality_metrics.metric_value < self.quality_thresholds["content_quality"]["critical"] else "medium"

                opportunities.append(RecommendationOpportunity(
                    opportunity_type="content_optimization",
                    title="Improve content quality and accuracy",
                    description="Content quality scores are below target levels",
                    rationale=f"Current quality score is {quality_metrics.metric_value:.1%}, target is {self.quality_thresholds['content_quality']['target']:.1%}",
                    estimated_impact=0.12,
                    implementation_effort="high",
                    priority=priority,
                    related_metrics=[quality_metrics.id],
                    metadata={"current_quality": quality_metrics.metric_value, "target_quality": self.quality_thresholds["content_quality"]["target"]}
                ))

            return opportunities

        except Exception as e:
            self.logger.error("content_quality_opportunity_detection_failed", error=str(e))
            return opportunities

    async def _detect_learning_opportunities(
        self,
        tenant_id: uuid.UUID,
        start_date: date,
        end_date: date
    ) -> List[RecommendationOpportunity]:
        """Detect learning algorithm improvement opportunities."""
        opportunities = []

        try:
            # Check learning algorithm performance
            learning_metrics = self.db.query(LearningPerformanceMetric).filter(
                and_(
                    LearningPerformanceMetric.tenant_id == tenant_id,
                    LearningPerformanceMetric.recorded_at >= datetime.combine(start_date, datetime.min.time()),
                    LearningPerformanceMetric.recorded_at <= datetime.combine(end_date, datetime.max.time())
                )
            ).all()

            if learning_metrics:
                # Calculate average performance
                avg_performance = sum(m.metric_value for m in learning_metrics) / len(learning_metrics)

                if avg_performance < 0.8:  # Below 80% performance
                    opportunities.append(RecommendationOpportunity(
                        opportunity_type="learning_tuning",
                        title="Optimize learning algorithm parameters",
                        description="Learning algorithms are underperforming and need parameter tuning",
                        rationale=f"Average learning performance is {avg_performance:.1%}, target is 85%+",
                        estimated_impact=0.10,
                        implementation_effort="medium",
                        priority="medium",
                        related_metrics=[m.id for m in learning_metrics[:5]],  # Limit to first 5
                        metadata={"avg_performance": avg_performance, "metric_count": len(learning_metrics)}
                    ))

            return opportunities

        except Exception as e:
            self.logger.error("learning_opportunity_detection_failed", error=str(e))
            return opportunities

    async def _check_existing_recommendation(
        self,
        tenant_id: uuid.UUID,
        opportunity_type: str,
        title: str
    ) -> Optional[PerformanceRecommendation]:
        """Check if a similar recommendation already exists."""
        try:
            existing = self.db.query(PerformanceRecommendation).filter(
                and_(
                    PerformanceRecommendation.tenant_id == tenant_id,
                    PerformanceRecommendation.category == opportunity_type,
                    PerformanceRecommendation.title == title,
                    PerformanceRecommendation.status.in_([
                        RecommendationStatus.PENDING,
                        RecommendationStatus.IN_PROGRESS
                    ])
                )
            ).first()

            return existing

        except Exception as e:
            self.logger.error("existing_recommendation_check_failed", error=str(e))
            return None

    def _map_opportunity_to_recommendation_type(self, opportunity_type: str) -> RecommendationType:
        """Map opportunity type to recommendation type."""
        mapping = {
            "search_relevance": RecommendationType.SEARCH_IMPROVEMENT,
            "user_experience": RecommendationType.USER_EXPERIENCE,
            "content_optimization": RecommendationType.CONTENT_OPTIMIZATION,
            "learning_tuning": RecommendationType.LEARNING_TUNING,
            "system_performance": RecommendationType.SYSTEM_OPTIMIZATION
        }

        return mapping.get(opportunity_type, RecommendationType.SYSTEM_OPTIMIZATION)
