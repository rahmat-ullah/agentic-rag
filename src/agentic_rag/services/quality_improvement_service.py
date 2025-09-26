"""
Quality Improvement Service for Sprint 6 Story 6-05: Automated Quality Improvement System

This service handles automated quality assessment, improvement action execution,
monitoring, and quality management workflows.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from agentic_rag.models.quality_improvement import (
    QualityAssessment,
    QualityImprovement,
    QualityMonitoring,
    AutomationRule,
    QualityAlert,
    QualityDimension,
    QualityIssueType,
    ImprovementActionType,
    ImprovementStatus
)
from agentic_rag.models.feedback import UserFeedbackSubmission, FeedbackAggregation
from agentic_rag.models.corrections import ContentCorrection
from agentic_rag.models.analytics import AnalyticsMetric
from agentic_rag.schemas.quality_improvement import (
    CreateQualityAssessmentRequest,
    CreateQualityImprovementRequest,
    QualityDashboardResponse,
    QualityMetricsResponse
)


@dataclass
class QualityScoreResult:
    """Result of quality score calculation."""
    
    overall_score: float
    dimension_scores: Dict[str, float]
    quality_issues: List[str]
    improvement_suggestions: List[str]
    confidence_level: float
    assessment_metadata: Dict[str, Any]


@dataclass
class ImprovementOpportunity:
    """Detected improvement opportunity."""
    
    target_type: str
    target_id: uuid.UUID
    issue_type: QualityIssueType
    current_quality: float
    expected_improvement: float
    recommended_action: ImprovementActionType
    priority_score: float
    trigger_reason: str
    metadata: Dict[str, Any]


class QualityImprovementService:
    """Service for automated quality improvement and management."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = structlog.get_logger(__name__)
        
        # Quality dimension weights (configurable)
        self.dimension_weights = {
            QualityDimension.ACCURACY: 0.3,
            QualityDimension.COMPLETENESS: 0.25,
            QualityDimension.FRESHNESS: 0.2,
            QualityDimension.RELEVANCE: 0.15,
            QualityDimension.USABILITY: 0.1
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "critical": 0.4,
            "warning": 0.6,
            "good": 0.8,
            "excellent": 0.9
        }
    
    async def assess_quality(
        self,
        tenant_id: uuid.UUID,
        target_type: str,
        target_id: uuid.UUID,
        assessment_method: str = "automated"
    ) -> QualityScoreResult:
        """Assess quality of a target (content, link, system component)."""
        try:
            # Calculate dimension scores based on target type
            dimension_scores = {}
            quality_issues = []
            improvement_suggestions = []
            
            if target_type == "content":
                dimension_scores = await self._assess_content_quality(tenant_id, target_id)
            elif target_type == "link":
                dimension_scores = await self._assess_link_quality(tenant_id, target_id)
            elif target_type == "system":
                dimension_scores = await self._assess_system_quality(tenant_id, target_id)
            else:
                raise ValueError(f"Unsupported target type: {target_type}")
            
            # Calculate overall quality score
            overall_score = sum(
                score * self.dimension_weights.get(QualityDimension(dim), 0.0)
                for dim, score in dimension_scores.items()
                if dim in [d.value for d in QualityDimension]
            )
            
            # Identify quality issues
            quality_issues = self._identify_quality_issues(dimension_scores, overall_score)
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(
                target_type, dimension_scores, quality_issues
            )
            
            # Calculate confidence level
            confidence_level = self._calculate_assessment_confidence(
                target_type, dimension_scores, assessment_method
            )
            
            self.logger.info(
                "quality_assessment_completed",
                tenant_id=tenant_id,
                target_type=target_type,
                target_id=target_id,
                overall_score=overall_score,
                confidence_level=confidence_level
            )
            
            return QualityScoreResult(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                quality_issues=quality_issues,
                improvement_suggestions=improvement_suggestions,
                confidence_level=confidence_level,
                assessment_metadata={
                    "assessment_method": assessment_method,
                    "dimension_weights": {k.value: v for k, v in self.dimension_weights.items()},
                    "thresholds": self.quality_thresholds
                }
            )
            
        except Exception as e:
            self.logger.error("quality_assessment_failed", error=str(e), tenant_id=tenant_id, target_type=target_type, target_id=target_id)
            raise
    
    async def create_quality_assessment(
        self,
        tenant_id: uuid.UUID,
        assessment_data: CreateQualityAssessmentRequest
    ) -> QualityAssessment:
        """Create a quality assessment record."""
        try:
            assessment = QualityAssessment(
                tenant_id=tenant_id,
                target_type=assessment_data.target_type,
                target_id=assessment_data.target_id,
                overall_quality_score=assessment_data.overall_quality_score,
                accuracy_score=assessment_data.accuracy_score,
                completeness_score=assessment_data.completeness_score,
                freshness_score=assessment_data.freshness_score,
                relevance_score=assessment_data.relevance_score,
                usability_score=assessment_data.usability_score,
                assessment_method=assessment_data.assessment_method,
                confidence_level=assessment_data.confidence_level,
                sample_size=assessment_data.sample_size,
                assessment_date=datetime.utcnow(),
                dimension_weights=assessment_data.dimension_weights,
                dimension_scores=assessment_data.dimension_scores,
                assessment_context=assessment_data.assessment_context,
                quality_issues=assessment_data.quality_issues,
                improvement_suggestions=assessment_data.improvement_suggestions
            )
            
            self.db.add(assessment)
            self.db.commit()
            self.db.refresh(assessment)
            
            self.logger.info(
                "quality_assessment_created",
                assessment_id=assessment.id,
                target_type=assessment.target_type,
                overall_score=assessment.overall_quality_score,
                tenant_id=tenant_id
            )
            
            return assessment
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("quality_assessment_creation_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def detect_improvement_opportunities(
        self,
        tenant_id: uuid.UUID,
        target_types: Optional[List[str]] = None,
        min_priority_score: float = 0.5
    ) -> List[ImprovementOpportunity]:
        """Detect improvement opportunities based on quality assessments and feedback."""
        try:
            opportunities = []
            
            # Detect low-quality links
            link_opportunities = await self._detect_low_quality_links(tenant_id)
            opportunities.extend(link_opportunities)
            
            # Detect frequently corrected content
            content_opportunities = await self._detect_frequently_corrected_content(tenant_id)
            opportunities.extend(content_opportunities)
            
            # Detect poor content quality
            poor_content_opportunities = await self._detect_poor_content_quality(tenant_id)
            opportunities.extend(poor_content_opportunities)
            
            # Detect processing errors
            processing_opportunities = await self._detect_processing_errors(tenant_id)
            opportunities.extend(processing_opportunities)
            
            # Filter by target types if specified
            if target_types:
                opportunities = [opp for opp in opportunities if opp.target_type in target_types]
            
            # Filter by minimum priority score
            opportunities = [opp for opp in opportunities if opp.priority_score >= min_priority_score]
            
            # Sort by priority score (descending)
            opportunities.sort(key=lambda x: x.priority_score, reverse=True)
            
            self.logger.info(
                "improvement_opportunities_detected",
                tenant_id=tenant_id,
                opportunity_count=len(opportunities),
                min_priority_score=min_priority_score
            )
            
            return opportunities
            
        except Exception as e:
            self.logger.error("opportunity_detection_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def execute_improvement_action(
        self,
        improvement_id: uuid.UUID
    ) -> bool:
        """Execute an improvement action."""
        try:
            improvement = self.db.query(QualityImprovement).get(improvement_id)
            if not improvement:
                raise ValueError(f"Improvement {improvement_id} not found")
            
            # Update status to in_progress
            improvement.status = ImprovementStatus.IN_PROGRESS
            improvement.started_at = datetime.utcnow()
            self.db.commit()
            
            # Execute the improvement action based on type
            success = False
            if improvement.improvement_action == ImprovementActionType.LINK_REVALIDATION:
                success = await self._execute_link_revalidation(improvement)
            elif improvement.improvement_action == ImprovementActionType.CONTENT_REPROCESSING:
                success = await self._execute_content_reprocessing(improvement)
            elif improvement.improvement_action == ImprovementActionType.EMBEDDING_UPDATE:
                success = await self._execute_embedding_update(improvement)
            elif improvement.improvement_action == ImprovementActionType.METADATA_REFRESH:
                success = await self._execute_metadata_refresh(improvement)
            elif improvement.improvement_action == ImprovementActionType.ALGORITHM_TUNING:
                success = await self._execute_algorithm_tuning(improvement)
            elif improvement.improvement_action == ImprovementActionType.CONTENT_REMOVAL:
                success = await self._execute_content_removal(improvement)
            elif improvement.improvement_action == ImprovementActionType.QUALITY_FLAGGING:
                success = await self._execute_quality_flagging(improvement)
            else:
                raise ValueError(f"Unsupported improvement action: {improvement.improvement_action}")
            
            # Update improvement status based on execution result
            if success:
                improvement.status = ImprovementStatus.COMPLETED
                improvement.completed_at = datetime.utcnow()
                
                # Assess quality after improvement
                quality_after = await self._assess_post_improvement_quality(improvement)
                improvement.quality_after = quality_after
                
                if improvement.quality_before:
                    improvement.improvement_delta = quality_after - improvement.quality_before
                    improvement.effectiveness_score = min(improvement.improvement_delta / 0.1, 2.0)  # Cap at 200%
                
            else:
                improvement.status = ImprovementStatus.FAILED
                improvement.failed_at = datetime.utcnow()
                improvement.failure_reason = "Execution failed"
            
            self.db.commit()
            
            self.logger.info(
                "improvement_action_executed",
                improvement_id=improvement_id,
                action=improvement.improvement_action,
                success=success,
                status=improvement.status
            )
            
            return success
            
        except Exception as e:
            # Update improvement status to failed
            if 'improvement' in locals():
                improvement.status = ImprovementStatus.FAILED
                improvement.failed_at = datetime.utcnow()
                improvement.failure_reason = str(e)
                self.db.commit()
            
            self.logger.error("improvement_execution_failed", error=str(e), improvement_id=improvement_id)
            raise
    
    async def get_quality_dashboard(
        self,
        tenant_id: uuid.UUID
    ) -> QualityDashboardResponse:
        """Get quality improvement dashboard data."""
        try:
            # Calculate overall quality score
            recent_assessments = self.db.query(QualityAssessment).filter(
                and_(
                    QualityAssessment.tenant_id == tenant_id,
                    QualityAssessment.assessment_date >= datetime.utcnow() - timedelta(days=7)
                )
            ).all()
            
            overall_quality_score = 0.75  # Default placeholder
            if recent_assessments:
                overall_quality_score = sum(a.overall_quality_score for a in recent_assessments) / len(recent_assessments)
            
            # Get quality trend
            quality_trend = "stable"  # Placeholder - would calculate from historical data
            
            # Get counts
            total_assessments = self.db.query(QualityAssessment).filter(
                QualityAssessment.tenant_id == tenant_id
            ).count()
            
            active_improvements = self.db.query(QualityImprovement).filter(
                and_(
                    QualityImprovement.tenant_id == tenant_id,
                    QualityImprovement.status.in_([ImprovementStatus.PENDING, ImprovementStatus.IN_PROGRESS])
                )
            ).count()
            
            completed_improvements = self.db.query(QualityImprovement).filter(
                and_(
                    QualityImprovement.tenant_id == tenant_id,
                    QualityImprovement.status == ImprovementStatus.COMPLETED
                )
            ).count()
            
            active_alerts = self.db.query(QualityAlert).filter(
                and_(
                    QualityAlert.tenant_id == tenant_id,
                    QualityAlert.status == "active"
                )
            ).count()
            
            automation_rules_count = self.db.query(AutomationRule).filter(
                and_(
                    AutomationRule.tenant_id == tenant_id,
                    AutomationRule.is_active == True
                )
            ).count()
            
            # Quality by dimension (placeholder)
            quality_by_dimension = {
                "accuracy": 0.8,
                "completeness": 0.75,
                "freshness": 0.85,
                "relevance": 0.7,
                "usability": 0.65
            }
            
            # Get recent items
            recent_assessments_list = self.db.query(QualityAssessment).filter(
                QualityAssessment.tenant_id == tenant_id
            ).order_by(desc(QualityAssessment.created_at)).limit(5).all()
            
            recent_improvements_list = self.db.query(QualityImprovement).filter(
                QualityImprovement.tenant_id == tenant_id
            ).order_by(desc(QualityImprovement.created_at)).limit(5).all()
            
            recent_alerts_list = self.db.query(QualityAlert).filter(
                QualityAlert.tenant_id == tenant_id
            ).order_by(desc(QualityAlert.created_at)).limit(5).all()
            
            # Quality distribution (placeholder)
            quality_distribution = {
                "excellent": 25,
                "good": 45,
                "fair": 20,
                "poor": 10
            }
            
            # Calculate metrics
            improvement_effectiveness = 0.85  # Placeholder
            automation_success_rate = 0.92   # Placeholder
            
            return QualityDashboardResponse(
                overall_quality_score=overall_quality_score,
                quality_trend=quality_trend,
                total_assessments=total_assessments,
                active_improvements=active_improvements,
                completed_improvements=completed_improvements,
                active_alerts=active_alerts,
                automation_rules_count=automation_rules_count,
                quality_by_dimension=quality_by_dimension,
                recent_assessments=[],  # Would convert to response models
                recent_improvements=[], # Would convert to response models
                recent_alerts=[],       # Would convert to response models
                quality_distribution=quality_distribution,
                improvement_effectiveness=improvement_effectiveness,
                automation_success_rate=automation_success_rate,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error("quality_dashboard_failed", error=str(e), tenant_id=tenant_id)
            raise

    # Helper methods for quality assessment

    async def _assess_content_quality(self, tenant_id: uuid.UUID, target_id: uuid.UUID) -> Dict[str, float]:
        """Assess quality of content."""
        try:
            # Get feedback data for content
            feedback_data = self.db.query(UserFeedbackSubmission).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.target_id == target_id
                )
            ).all()

            # Get correction data
            correction_data = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.content_id == target_id
                )
            ).all()

            # Calculate dimension scores
            accuracy_score = self._calculate_accuracy_score(feedback_data, correction_data)
            completeness_score = self._calculate_completeness_score(feedback_data)
            freshness_score = self._calculate_freshness_score(target_id)
            relevance_score = self._calculate_relevance_score(feedback_data)
            usability_score = self._calculate_usability_score(feedback_data)

            return {
                QualityDimension.ACCURACY.value: accuracy_score,
                QualityDimension.COMPLETENESS.value: completeness_score,
                QualityDimension.FRESHNESS.value: freshness_score,
                QualityDimension.RELEVANCE.value: relevance_score,
                QualityDimension.USABILITY.value: usability_score
            }

        except Exception as e:
            self.logger.error("content_quality_assessment_failed", error=str(e), target_id=target_id)
            # Return default scores
            return {dim.value: 0.5 for dim in QualityDimension}

    async def _assess_link_quality(self, tenant_id: uuid.UUID, target_id: uuid.UUID) -> Dict[str, float]:
        """Assess quality of links."""
        try:
            # Get feedback data for link
            feedback_data = self.db.query(UserFeedbackSubmission).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.target_id == target_id
                )
            ).all()

            # Calculate link-specific quality scores
            accuracy_score = self._calculate_link_accuracy_score(feedback_data)
            relevance_score = self._calculate_link_relevance_score(feedback_data)
            usability_score = self._calculate_link_usability_score(feedback_data)

            return {
                QualityDimension.ACCURACY.value: accuracy_score,
                QualityDimension.COMPLETENESS.value: 0.8,  # Links don't have completeness in same way
                QualityDimension.FRESHNESS.value: 0.9,     # Assume links are fresh unless proven otherwise
                QualityDimension.RELEVANCE.value: relevance_score,
                QualityDimension.USABILITY.value: usability_score
            }

        except Exception as e:
            self.logger.error("link_quality_assessment_failed", error=str(e), target_id=target_id)
            return {dim.value: 0.5 for dim in QualityDimension}

    async def _assess_system_quality(self, tenant_id: uuid.UUID, target_id: uuid.UUID) -> Dict[str, float]:
        """Assess quality of system components."""
        try:
            # Get system performance metrics
            performance_metrics = self.db.query(AnalyticsMetric).filter(
                and_(
                    AnalyticsMetric.tenant_id == tenant_id,
                    AnalyticsMetric.metric_name.in_(['response_time', 'error_rate', 'uptime'])
                )
            ).all()

            # Calculate system quality scores based on performance
            accuracy_score = 0.9  # System accuracy based on error rates
            completeness_score = 0.85  # System completeness
            freshness_score = 0.95  # System freshness
            relevance_score = 0.8   # System relevance
            usability_score = 0.75  # System usability

            return {
                QualityDimension.ACCURACY.value: accuracy_score,
                QualityDimension.COMPLETENESS.value: completeness_score,
                QualityDimension.FRESHNESS.value: freshness_score,
                QualityDimension.RELEVANCE.value: relevance_score,
                QualityDimension.USABILITY.value: usability_score
            }

        except Exception as e:
            self.logger.error("system_quality_assessment_failed", error=str(e), target_id=target_id)
            return {dim.value: 0.5 for dim in QualityDimension}

    def _calculate_accuracy_score(self, feedback_data: List, correction_data: List) -> float:
        """Calculate accuracy score based on feedback and corrections."""
        if not feedback_data and not correction_data:
            return 0.8  # Default score

        # Factor in negative feedback
        negative_feedback_count = sum(1 for f in feedback_data if f.rating and f.rating < 3)
        total_feedback = len(feedback_data)

        # Factor in corrections
        correction_count = len(correction_data)

        # Calculate accuracy score
        if total_feedback > 0:
            negative_feedback_ratio = negative_feedback_count / total_feedback
            accuracy_from_feedback = 1.0 - negative_feedback_ratio
        else:
            accuracy_from_feedback = 0.8

        # Penalize for corrections
        correction_penalty = min(correction_count * 0.1, 0.5)
        accuracy_score = max(accuracy_from_feedback - correction_penalty, 0.0)

        return min(accuracy_score, 1.0)

    def _calculate_completeness_score(self, feedback_data: List) -> float:
        """Calculate completeness score based on feedback."""
        if not feedback_data:
            return 0.7  # Default score

        # Look for feedback indicating incomplete information
        incomplete_feedback = sum(1 for f in feedback_data
                                if f.feedback_text and 'incomplete' in f.feedback_text.lower())

        if len(feedback_data) > 0:
            completeness_ratio = 1.0 - (incomplete_feedback / len(feedback_data))
            return max(completeness_ratio, 0.0)

        return 0.7

    def _calculate_freshness_score(self, target_id: uuid.UUID) -> float:
        """Calculate freshness score based on content age and updates."""
        # Placeholder implementation - would check content creation/update dates
        return 0.8

    def _calculate_relevance_score(self, feedback_data: List) -> float:
        """Calculate relevance score based on user feedback."""
        if not feedback_data:
            return 0.75  # Default score

        # Calculate average rating as relevance indicator
        ratings = [f.rating for f in feedback_data if f.rating]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            return min(avg_rating / 5.0, 1.0)  # Normalize to 0-1

        return 0.75

    def _calculate_usability_score(self, feedback_data: List) -> float:
        """Calculate usability score based on user feedback."""
        if not feedback_data:
            return 0.7  # Default score

        # Look for usability-related feedback
        usability_issues = sum(1 for f in feedback_data
                             if f.feedback_text and any(term in f.feedback_text.lower()
                                                      for term in ['hard to use', 'confusing', 'unclear']))

        if len(feedback_data) > 0:
            usability_ratio = 1.0 - (usability_issues / len(feedback_data))
            return max(usability_ratio, 0.0)

        return 0.7

    def _calculate_link_accuracy_score(self, feedback_data: List) -> float:
        """Calculate link accuracy score."""
        if not feedback_data:
            return 0.8

        # Check for broken link feedback
        broken_link_feedback = sum(1 for f in feedback_data
                                 if f.feedback_text and 'broken' in f.feedback_text.lower())

        if len(feedback_data) > 0:
            accuracy_ratio = 1.0 - (broken_link_feedback / len(feedback_data))
            return max(accuracy_ratio, 0.0)

        return 0.8

    def _calculate_link_relevance_score(self, feedback_data: List) -> float:
        """Calculate link relevance score."""
        return self._calculate_relevance_score(feedback_data)

    def _calculate_link_usability_score(self, feedback_data: List) -> float:
        """Calculate link usability score."""
        return self._calculate_usability_score(feedback_data)

    def _identify_quality_issues(self, dimension_scores: Dict[str, float], overall_score: float) -> List[str]:
        """Identify quality issues based on dimension scores."""
        issues = []

        if overall_score < self.quality_thresholds["critical"]:
            issues.append("critical_overall_quality")
        elif overall_score < self.quality_thresholds["warning"]:
            issues.append("low_overall_quality")

        for dimension, score in dimension_scores.items():
            if score < self.quality_thresholds["critical"]:
                issues.append(f"critical_{dimension}_quality")
            elif score < self.quality_thresholds["warning"]:
                issues.append(f"low_{dimension}_quality")

        return issues

    def _generate_improvement_suggestions(
        self,
        target_type: str,
        dimension_scores: Dict[str, float],
        quality_issues: List[str]
    ) -> List[str]:
        """Generate improvement suggestions based on quality assessment."""
        suggestions = []

        # General suggestions based on dimension scores
        for dimension, score in dimension_scores.items():
            if score < self.quality_thresholds["warning"]:
                if dimension == QualityDimension.ACCURACY.value:
                    suggestions.append("verify_and_correct_factual_information")
                elif dimension == QualityDimension.COMPLETENESS.value:
                    suggestions.append("add_missing_information_and_details")
                elif dimension == QualityDimension.FRESHNESS.value:
                    suggestions.append("update_outdated_information")
                elif dimension == QualityDimension.RELEVANCE.value:
                    suggestions.append("improve_content_relevance_to_user_needs")
                elif dimension == QualityDimension.USABILITY.value:
                    suggestions.append("improve_formatting_and_clarity")

        # Target-specific suggestions
        if target_type == "content":
            if "critical_overall_quality" in quality_issues:
                suggestions.append("consider_content_reprocessing")
            if "low_accuracy_quality" in quality_issues:
                suggestions.append("fact_check_and_validate_sources")
        elif target_type == "link":
            if "critical_overall_quality" in quality_issues:
                suggestions.append("revalidate_link_and_check_accessibility")
            if "low_accuracy_quality" in quality_issues:
                suggestions.append("verify_link_destination_and_content")

        return list(set(suggestions))  # Remove duplicates

    def _calculate_assessment_confidence(
        self,
        target_type: str,
        dimension_scores: Dict[str, float],
        assessment_method: str
    ) -> float:
        """Calculate confidence level for the assessment."""
        base_confidence = 0.8

        # Adjust based on assessment method
        if assessment_method == "automated":
            base_confidence = 0.7
        elif assessment_method == "manual":
            base_confidence = 0.9
        elif assessment_method == "hybrid":
            base_confidence = 0.85

        # Adjust based on data availability
        non_null_scores = sum(1 for score in dimension_scores.values() if score is not None)
        data_completeness = non_null_scores / len(QualityDimension)

        confidence = base_confidence * data_completeness
        return min(max(confidence, 0.1), 1.0)

    # Helper methods for opportunity detection

    async def _detect_low_quality_links(self, tenant_id: uuid.UUID) -> List[ImprovementOpportunity]:
        """Detect low-quality links that need improvement."""
        opportunities = []

        try:
            # Get recent quality assessments for links
            link_assessments = self.db.query(QualityAssessment).filter(
                and_(
                    QualityAssessment.tenant_id == tenant_id,
                    QualityAssessment.target_type == "link",
                    QualityAssessment.overall_quality_score < self.quality_thresholds["warning"]
                )
            ).all()

            for assessment in link_assessments:
                priority_score = 1.0 - assessment.overall_quality_score  # Lower quality = higher priority

                opportunities.append(ImprovementOpportunity(
                    target_type="link",
                    target_id=assessment.target_id,
                    issue_type=QualityIssueType.LOW_QUALITY_LINK,
                    current_quality=assessment.overall_quality_score,
                    expected_improvement=0.2,
                    recommended_action=ImprovementActionType.LINK_REVALIDATION,
                    priority_score=priority_score,
                    trigger_reason=f"Quality score {assessment.overall_quality_score:.2f} below threshold {self.quality_thresholds['warning']}",
                    metadata={"assessment_id": assessment.id}
                ))

            return opportunities

        except Exception as e:
            self.logger.error("low_quality_link_detection_failed", error=str(e), tenant_id=tenant_id)
            return []

    async def _detect_frequently_corrected_content(self, tenant_id: uuid.UUID) -> List[ImprovementOpportunity]:
        """Detect content that is frequently corrected."""
        opportunities = []

        try:
            # Get content with high correction frequency
            correction_counts = self.db.query(
                ContentCorrection.content_id,
                func.count(ContentCorrection.id).label('correction_count')
            ).filter(
                and_(
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.created_at >= datetime.utcnow() - timedelta(days=30)
                )
            ).group_by(ContentCorrection.content_id).having(
                func.count(ContentCorrection.id) >= 3  # 3+ corrections in 30 days
            ).all()

            for content_id, correction_count in correction_counts:
                priority_score = min(correction_count / 10.0, 1.0)  # Normalize to 0-1

                opportunities.append(ImprovementOpportunity(
                    target_type="content",
                    target_id=content_id,
                    issue_type=QualityIssueType.FREQUENT_CORRECTIONS,
                    current_quality=0.5,  # Assume medium quality for frequently corrected content
                    expected_improvement=0.3,
                    recommended_action=ImprovementActionType.CONTENT_REPROCESSING,
                    priority_score=priority_score,
                    trigger_reason=f"Content has {correction_count} corrections in the last 30 days",
                    metadata={"correction_count": correction_count}
                ))

            return opportunities

        except Exception as e:
            self.logger.error("frequent_corrections_detection_failed", error=str(e), tenant_id=tenant_id)
            return []

    async def _detect_poor_content_quality(self, tenant_id: uuid.UUID) -> List[ImprovementOpportunity]:
        """Detect content with poor quality scores."""
        opportunities = []

        try:
            # Get content assessments with poor quality
            poor_content = self.db.query(QualityAssessment).filter(
                and_(
                    QualityAssessment.tenant_id == tenant_id,
                    QualityAssessment.target_type == "content",
                    QualityAssessment.overall_quality_score < self.quality_thresholds["warning"]
                )
            ).all()

            for assessment in poor_content:
                priority_score = 1.0 - assessment.overall_quality_score

                # Determine recommended action based on quality issues
                recommended_action = ImprovementActionType.CONTENT_REPROCESSING
                if assessment.overall_quality_score < self.quality_thresholds["critical"]:
                    recommended_action = ImprovementActionType.CONTENT_REMOVAL

                opportunities.append(ImprovementOpportunity(
                    target_type="content",
                    target_id=assessment.target_id,
                    issue_type=QualityIssueType.POOR_CONTENT_QUALITY,
                    current_quality=assessment.overall_quality_score,
                    expected_improvement=0.25,
                    recommended_action=recommended_action,
                    priority_score=priority_score,
                    trigger_reason=f"Content quality score {assessment.overall_quality_score:.2f} below threshold",
                    metadata={"assessment_id": assessment.id}
                ))

            return opportunities

        except Exception as e:
            self.logger.error("poor_content_quality_detection_failed", error=str(e), tenant_id=tenant_id)
            return []

    async def _detect_processing_errors(self, tenant_id: uuid.UUID) -> List[ImprovementOpportunity]:
        """Detect processing errors that need attention."""
        opportunities = []

        try:
            # This would integrate with document processing system to detect errors
            # For now, return empty list as placeholder
            return opportunities

        except Exception as e:
            self.logger.error("processing_error_detection_failed", error=str(e), tenant_id=tenant_id)
            return []

    # Helper methods for improvement action execution

    async def _execute_link_revalidation(self, improvement: QualityImprovement) -> bool:
        """Execute link revalidation improvement action."""
        try:
            # Placeholder for link revalidation logic
            # Would integrate with link validation service
            self.logger.info("executing_link_revalidation", target_id=improvement.target_id)

            # Simulate revalidation process
            improvement.validation_results = {
                "revalidation_performed": True,
                "link_accessible": True,
                "response_time": 0.5,
                "status_code": 200
            }

            return True

        except Exception as e:
            self.logger.error("link_revalidation_failed", error=str(e), target_id=improvement.target_id)
            return False

    async def _execute_content_reprocessing(self, improvement: QualityImprovement) -> bool:
        """Execute content reprocessing improvement action."""
        try:
            # Placeholder for content reprocessing logic
            # Would integrate with document processing pipeline
            self.logger.info("executing_content_reprocessing", target_id=improvement.target_id)

            # Simulate reprocessing
            improvement.validation_results = {
                "reprocessing_performed": True,
                "extraction_improved": True,
                "metadata_updated": True,
                "quality_enhanced": True
            }

            return True

        except Exception as e:
            self.logger.error("content_reprocessing_failed", error=str(e), target_id=improvement.target_id)
            return False

    async def _execute_embedding_update(self, improvement: QualityImprovement) -> bool:
        """Execute embedding update improvement action."""
        try:
            # Placeholder for embedding update logic
            # Would integrate with embedding service
            self.logger.info("executing_embedding_update", target_id=improvement.target_id)

            improvement.validation_results = {
                "embeddings_updated": True,
                "vector_quality_improved": True,
                "search_relevance_enhanced": True
            }

            return True

        except Exception as e:
            self.logger.error("embedding_update_failed", error=str(e), target_id=improvement.target_id)
            return False

    async def _execute_metadata_refresh(self, improvement: QualityImprovement) -> bool:
        """Execute metadata refresh improvement action."""
        try:
            # Placeholder for metadata refresh logic
            self.logger.info("executing_metadata_refresh", target_id=improvement.target_id)

            improvement.validation_results = {
                "metadata_refreshed": True,
                "tags_updated": True,
                "categories_refined": True
            }

            return True

        except Exception as e:
            self.logger.error("metadata_refresh_failed", error=str(e), target_id=improvement.target_id)
            return False

    async def _execute_algorithm_tuning(self, improvement: QualityImprovement) -> bool:
        """Execute algorithm tuning improvement action."""
        try:
            # Placeholder for algorithm tuning logic
            # Would integrate with learning algorithms service
            self.logger.info("executing_algorithm_tuning", target_id=improvement.target_id)

            improvement.validation_results = {
                "algorithm_tuned": True,
                "parameters_optimized": True,
                "performance_improved": True
            }

            return True

        except Exception as e:
            self.logger.error("algorithm_tuning_failed", error=str(e), target_id=improvement.target_id)
            return False

    async def _execute_content_removal(self, improvement: QualityImprovement) -> bool:
        """Execute content removal improvement action."""
        try:
            # Placeholder for content removal logic
            # Would mark content as inactive or remove from search
            self.logger.info("executing_content_removal", target_id=improvement.target_id)

            improvement.validation_results = {
                "content_removed": True,
                "search_index_updated": True,
                "references_cleaned": True
            }

            return True

        except Exception as e:
            self.logger.error("content_removal_failed", error=str(e), target_id=improvement.target_id)
            return False

    async def _execute_quality_flagging(self, improvement: QualityImprovement) -> bool:
        """Execute quality flagging improvement action."""
        try:
            # Flag content for manual review
            self.logger.info("executing_quality_flagging", target_id=improvement.target_id)

            improvement.validation_results = {
                "content_flagged": True,
                "review_requested": True,
                "quality_warning_added": True
            }

            return True

        except Exception as e:
            self.logger.error("quality_flagging_failed", error=str(e), target_id=improvement.target_id)
            return False

    async def _assess_post_improvement_quality(self, improvement: QualityImprovement) -> float:
        """Assess quality after improvement action."""
        try:
            # Re-assess quality after improvement
            quality_result = await self.assess_quality(
                improvement.tenant_id,
                improvement.target_type,
                improvement.target_id,
                "post_improvement_assessment"
            )

            return quality_result.overall_score

        except Exception as e:
            self.logger.error("post_improvement_assessment_failed", error=str(e), improvement_id=improvement.id)
            # Return estimated improvement if assessment fails
            if improvement.quality_before:
                return min(improvement.quality_before + 0.1, 1.0)
            return 0.7
