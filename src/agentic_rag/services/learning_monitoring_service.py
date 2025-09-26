"""
Learning Performance Monitoring Service for Sprint 6 Story 6-03

This service implements learning performance monitoring, validation framework,
and A/B testing capabilities for learning algorithms.
"""

import uuid
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from agentic_rag.models.learning import (
    LearningAlgorithm,
    LearningPerformanceMetric,
    LearningSession,
    ABTestExperiment,
    LearningStatus,
    LearningAlgorithmType
)
from agentic_rag.database.connection import get_database_session
from agentic_rag.api.exceptions import ValidationError, ServiceError

logger = structlog.get_logger(__name__)


class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class PerformanceValidationResult:
    """Result of performance validation."""
    status: ValidationStatus
    score: float
    improvement_percentage: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    validation_metadata: Dict[str, Any]
    recommendations: List[str]


@dataclass
class ABTestResult:
    """Result of A/B test analysis."""
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


@dataclass
class LearningHealthCheck:
    """Health check result for learning algorithms."""
    algorithm_id: uuid.UUID
    algorithm_type: LearningAlgorithmType
    health_score: float
    status: str
    issues: List[str]
    recommendations: List[str]
    last_update: datetime
    performance_trend: str


class LearningMonitoringService:
    """Service for monitoring learning algorithm performance."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logger.bind(service="learning_monitoring_service")
    
    async def validate_algorithm_performance(
        self,
        algorithm_id: uuid.UUID,
        validation_period_hours: int = 24,
        baseline_period_hours: int = 168  # 1 week
    ) -> PerformanceValidationResult:
        """Validate learning algorithm performance against baseline."""
        try:
            algorithm = self.db.query(LearningAlgorithm).get(algorithm_id)
            if not algorithm:
                raise ValidationError(f"Algorithm {algorithm_id} not found")
            
            current_time = datetime.now()
            validation_start = current_time - timedelta(hours=validation_period_hours)
            baseline_start = current_time - timedelta(hours=baseline_period_hours)
            baseline_end = current_time - timedelta(hours=validation_period_hours)
            
            # Get performance metrics for validation period
            validation_metrics = self.db.query(LearningPerformanceMetric).filter(
                and_(
                    LearningPerformanceMetric.algorithm_id == algorithm_id,
                    LearningPerformanceMetric.recorded_at >= validation_start
                )
            ).all()
            
            # Get baseline metrics
            baseline_metrics = self.db.query(LearningPerformanceMetric).filter(
                and_(
                    LearningPerformanceMetric.algorithm_id == algorithm_id,
                    LearningPerformanceMetric.recorded_at >= baseline_start,
                    LearningPerformanceMetric.recorded_at <= baseline_end
                )
            ).all()
            
            if not validation_metrics or not baseline_metrics:
                return PerformanceValidationResult(
                    status=ValidationStatus.PENDING,
                    score=0.0,
                    improvement_percentage=0.0,
                    statistical_significance=0.0,
                    confidence_interval=(0.0, 0.0),
                    validation_metadata={"insufficient_data": True},
                    recommendations=["Collect more performance data before validation"]
                )
            
            # Calculate performance scores
            validation_scores = [m.metric_value for m in validation_metrics]
            baseline_scores = [m.metric_value for m in baseline_metrics]
            
            validation_mean = statistics.mean(validation_scores)
            baseline_mean = statistics.mean(baseline_scores)
            
            # Calculate improvement percentage
            improvement_percentage = ((validation_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
            
            # Calculate statistical significance (simplified t-test)
            statistical_significance = self._calculate_statistical_significance(
                validation_scores, baseline_scores
            )
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(validation_scores)
            
            # Determine validation status
            status = self._determine_validation_status(
                improvement_percentage, statistical_significance, algorithm.validation_threshold
            )
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(
                algorithm, improvement_percentage, statistical_significance
            )
            
            result = PerformanceValidationResult(
                status=status,
                score=validation_mean,
                improvement_percentage=improvement_percentage,
                statistical_significance=statistical_significance,
                confidence_interval=confidence_interval,
                validation_metadata={
                    "validation_period_hours": validation_period_hours,
                    "baseline_period_hours": baseline_period_hours,
                    "validation_sample_size": len(validation_scores),
                    "baseline_sample_size": len(baseline_scores),
                    "validation_mean": validation_mean,
                    "baseline_mean": baseline_mean
                },
                recommendations=recommendations
            )
            
            # Update algorithm validation status
            algorithm.last_validated_at = current_time
            if status == ValidationStatus.PASSED:
                algorithm.accuracy_score = validation_mean
            
            self.db.commit()
            
            self.logger.info(
                "algorithm_performance_validated",
                algorithm_id=algorithm_id,
                status=status.value,
                improvement_percentage=improvement_percentage,
                statistical_significance=statistical_significance
            )
            
            return result
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("algorithm_validation_failed", error=str(e), algorithm_id=algorithm_id)
            raise ServiceError(f"Failed to validate algorithm performance: {str(e)}")
    
    async def create_ab_test_experiment(
        self,
        tenant_id: uuid.UUID,
        experiment_name: str,
        control_algorithm_id: uuid.UUID,
        treatment_algorithm_id: uuid.UUID,
        primary_metric: str,
        success_threshold: float,
        traffic_split_percentage: float = 50.0,
        minimum_sample_size: int = 1000,
        confidence_level: float = 0.95,
        description: Optional[str] = None,
        hypothesis: Optional[str] = None
    ) -> ABTestExperiment:
        """Create A/B test experiment for learning algorithms."""
        try:
            # Validate algorithms exist
            control_alg = self.db.query(LearningAlgorithm).get(control_algorithm_id)
            treatment_alg = self.db.query(LearningAlgorithm).get(treatment_algorithm_id)
            
            if not control_alg or not treatment_alg:
                raise ValidationError("Control or treatment algorithm not found")
            
            if control_alg.tenant_id != tenant_id or treatment_alg.tenant_id != tenant_id:
                raise ValidationError("Algorithms must belong to the same tenant")
            
            # Create experiment
            experiment = ABTestExperiment(
                tenant_id=tenant_id,
                experiment_name=experiment_name,
                description=description,
                hypothesis=hypothesis,
                control_algorithm_id=control_algorithm_id,
                treatment_algorithm_id=treatment_algorithm_id,
                traffic_split_percentage=traffic_split_percentage,
                primary_metric=primary_metric,
                success_threshold=success_threshold,
                minimum_sample_size=minimum_sample_size,
                confidence_level=confidence_level,
                status="draft"
            )
            
            self.db.add(experiment)
            self.db.commit()
            self.db.refresh(experiment)
            
            self.logger.info(
                "ab_test_experiment_created",
                experiment_id=experiment.id,
                experiment_name=experiment_name,
                control_algorithm=control_algorithm_id,
                treatment_algorithm=treatment_algorithm_id
            )
            
            return experiment
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("ab_test_creation_failed", error=str(e))
            raise ServiceError(f"Failed to create A/B test experiment: {str(e)}")
    
    async def start_ab_test_experiment(self, experiment_id: uuid.UUID) -> ABTestExperiment:
        """Start an A/B test experiment."""
        try:
            experiment = self.db.query(ABTestExperiment).get(experiment_id)
            if not experiment:
                raise ValidationError(f"Experiment {experiment_id} not found")
            
            if experiment.status != "draft":
                raise ValidationError(f"Experiment must be in draft status to start")
            
            # Validate algorithms are active
            control_alg = self.db.query(LearningAlgorithm).get(experiment.control_algorithm_id)
            treatment_alg = self.db.query(LearningAlgorithm).get(experiment.treatment_algorithm_id)
            
            if not control_alg.is_enabled or not treatment_alg.is_enabled:
                raise ValidationError("Both algorithms must be enabled to start experiment")
            
            # Start experiment
            experiment.status = "running"
            experiment.is_active = True
            experiment.started_at = datetime.now()
            
            self.db.commit()
            
            self.logger.info(
                "ab_test_experiment_started",
                experiment_id=experiment_id,
                experiment_name=experiment.experiment_name
            )
            
            return experiment
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("ab_test_start_failed", error=str(e), experiment_id=experiment_id)
            raise ServiceError(f"Failed to start A/B test experiment: {str(e)}")
    
    async def analyze_ab_test_results(self, experiment_id: uuid.UUID) -> ABTestResult:
        """Analyze A/B test experiment results."""
        try:
            experiment = self.db.query(ABTestExperiment).get(experiment_id)
            if not experiment:
                raise ValidationError(f"Experiment {experiment_id} not found")
            
            if not experiment.started_at:
                raise ValidationError("Experiment has not been started")
            
            # Get performance metrics for both algorithms since experiment start
            control_metrics = self.db.query(LearningPerformanceMetric).filter(
                and_(
                    LearningPerformanceMetric.algorithm_id == experiment.control_algorithm_id,
                    LearningPerformanceMetric.metric_name == experiment.primary_metric,
                    LearningPerformanceMetric.recorded_at >= experiment.started_at
                )
            ).all()
            
            treatment_metrics = self.db.query(LearningPerformanceMetric).filter(
                and_(
                    LearningPerformanceMetric.algorithm_id == experiment.treatment_algorithm_id,
                    LearningPerformanceMetric.metric_name == experiment.primary_metric,
                    LearningPerformanceMetric.recorded_at >= experiment.started_at
                )
            ).all()
            
            if not control_metrics or not treatment_metrics:
                raise ValidationError("Insufficient data for analysis")
            
            # Calculate performance
            control_values = [m.metric_value for m in control_metrics]
            treatment_values = [m.metric_value for m in treatment_metrics]
            
            control_performance = statistics.mean(control_values)
            treatment_performance = statistics.mean(treatment_values)
            
            # Calculate improvement
            improvement_percentage = ((treatment_performance - control_performance) / control_performance * 100) if control_performance != 0 else 0
            
            # Calculate statistical significance
            statistical_significance = self._calculate_statistical_significance(
                treatment_values, control_values
            )
            
            # Determine if result is significant
            is_significant = (
                statistical_significance >= experiment.confidence_level and
                abs(improvement_percentage) >= experiment.success_threshold and
                len(control_values) >= experiment.minimum_sample_size and
                len(treatment_values) >= experiment.minimum_sample_size
            )
            
            # Generate recommendation
            if is_significant and improvement_percentage > 0:
                recommendation = "Deploy treatment algorithm - significant improvement detected"
            elif is_significant and improvement_percentage < 0:
                recommendation = "Keep control algorithm - treatment performs worse"
            else:
                recommendation = "Continue experiment - results not yet significant"
            
            # Update experiment with results
            experiment.control_metric_value = control_performance
            experiment.treatment_metric_value = treatment_performance
            experiment.statistical_significance = statistical_significance
            experiment.effect_size = improvement_percentage
            
            if is_significant:
                experiment.status = "completed"
                experiment.is_active = False
                experiment.ended_at = datetime.now()
            
            self.db.commit()
            
            result = ABTestResult(
                experiment_id=experiment_id,
                control_performance=control_performance,
                treatment_performance=treatment_performance,
                improvement_percentage=improvement_percentage,
                statistical_significance=statistical_significance,
                confidence_level=experiment.confidence_level,
                sample_size_control=len(control_values),
                sample_size_treatment=len(treatment_values),
                is_significant=is_significant,
                recommendation=recommendation
            )
            
            self.logger.info(
                "ab_test_results_analyzed",
                experiment_id=experiment_id,
                improvement_percentage=improvement_percentage,
                is_significant=is_significant,
                recommendation=recommendation
            )
            
            return result
            
        except Exception as e:
            self.logger.error("ab_test_analysis_failed", error=str(e), experiment_id=experiment_id)
            raise ServiceError(f"Failed to analyze A/B test results: {str(e)}")
    
    async def perform_learning_health_check(
        self,
        tenant_id: uuid.UUID
    ) -> List[LearningHealthCheck]:
        """Perform health check on all learning algorithms for a tenant."""
        try:
            algorithms = self.db.query(LearningAlgorithm).filter(
                LearningAlgorithm.tenant_id == tenant_id
            ).all()
            
            health_checks = []
            
            for algorithm in algorithms:
                # Calculate health score based on multiple factors
                health_score = await self._calculate_algorithm_health_score(algorithm)
                
                # Determine status
                if health_score >= 0.8:
                    status = "healthy"
                elif health_score >= 0.6:
                    status = "warning"
                else:
                    status = "critical"
                
                # Identify issues
                issues = await self._identify_algorithm_issues(algorithm)
                
                # Generate recommendations
                recommendations = await self._generate_health_recommendations(algorithm, issues)
                
                # Determine performance trend
                trend = await self._calculate_performance_trend(algorithm.id)
                
                health_check = LearningHealthCheck(
                    algorithm_id=algorithm.id,
                    algorithm_type=algorithm.algorithm_type,
                    health_score=health_score,
                    status=status,
                    issues=issues,
                    recommendations=recommendations,
                    last_update=algorithm.updated_at,
                    performance_trend=trend
                )
                
                health_checks.append(health_check)
            
            self.logger.info(
                "learning_health_check_completed",
                tenant_id=tenant_id,
                algorithms_checked=len(health_checks),
                healthy_count=len([hc for hc in health_checks if hc.status == "healthy"]),
                warning_count=len([hc for hc in health_checks if hc.status == "warning"]),
                critical_count=len([hc for hc in health_checks if hc.status == "critical"])
            )
            
            return health_checks
            
        except Exception as e:
            self.logger.error("learning_health_check_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to perform learning health check: {str(e)}")
    
    # Private helper methods
    
    def _calculate_statistical_significance(
        self,
        sample1: List[float],
        sample2: List[float]
    ) -> float:
        """Calculate statistical significance using simplified t-test."""
        try:
            if len(sample1) < 2 or len(sample2) < 2:
                return 0.0
            
            mean1 = statistics.mean(sample1)
            mean2 = statistics.mean(sample2)
            
            if len(sample1) == 1 or len(sample2) == 1:
                return 0.0
            
            var1 = statistics.variance(sample1)
            var2 = statistics.variance(sample2)
            
            # Pooled standard error
            pooled_se = ((var1 / len(sample1)) + (var2 / len(sample2))) ** 0.5
            
            if pooled_se == 0:
                return 1.0 if mean1 == mean2 else 0.0
            
            # T-statistic
            t_stat = abs(mean1 - mean2) / pooled_se
            
            # Simplified p-value calculation (approximation)
            # For a more accurate calculation, use scipy.stats
            if t_stat > 2.576:  # 99% confidence
                return 0.99
            elif t_stat > 1.96:  # 95% confidence
                return 0.95
            elif t_stat > 1.645:  # 90% confidence
                return 0.90
            else:
                return max(0.0, 1.0 - (t_stat / 1.645) * 0.1)
            
        except Exception:
            return 0.0
    
    def _calculate_confidence_interval(
        self,
        sample: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for sample."""
        try:
            if len(sample) < 2:
                mean_val = sample[0] if sample else 0.0
                return (mean_val, mean_val)
            
            mean_val = statistics.mean(sample)
            std_dev = statistics.stdev(sample)
            
            # Simplified confidence interval (assuming normal distribution)
            margin_of_error = 1.96 * (std_dev / (len(sample) ** 0.5))  # 95% confidence
            
            return (mean_val - margin_of_error, mean_val + margin_of_error)
            
        except Exception:
            return (0.0, 0.0)
    
    def _determine_validation_status(
        self,
        improvement_percentage: float,
        statistical_significance: float,
        threshold: float
    ) -> ValidationStatus:
        """Determine validation status based on metrics."""
        if statistical_significance >= 0.95 and improvement_percentage >= threshold * 100:
            return ValidationStatus.PASSED
        elif statistical_significance >= 0.90 and improvement_percentage >= threshold * 50:
            return ValidationStatus.WARNING
        elif improvement_percentage < -threshold * 100:
            return ValidationStatus.FAILED
        else:
            return ValidationStatus.PENDING
    
    def _generate_performance_recommendations(
        self,
        algorithm: LearningAlgorithm,
        improvement_percentage: float,
        statistical_significance: float
    ) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if improvement_percentage < 0:
            recommendations.append("Consider reducing learning rate or adjusting algorithm parameters")
            recommendations.append("Review recent feedback signals for quality issues")
        
        if statistical_significance < 0.90:
            recommendations.append("Collect more performance data for reliable validation")
            recommendations.append("Consider extending validation period")
        
        if algorithm.training_data_size < 1000:
            recommendations.append("Increase training data size for better performance")
        
        if not algorithm.last_trained_at or (datetime.now() - algorithm.last_trained_at).days > 7:
            recommendations.append("Consider retraining algorithm with recent data")
        
        return recommendations
    
    async def _calculate_algorithm_health_score(self, algorithm: LearningAlgorithm) -> float:
        """Calculate health score for algorithm."""
        score = 1.0
        
        # Check if algorithm is enabled and active
        if not algorithm.is_enabled or algorithm.status != LearningStatus.ACTIVE:
            score *= 0.5
        
        # Check performance scores
        if algorithm.accuracy_score:
            score *= algorithm.accuracy_score
        
        # Check last training time
        if algorithm.last_trained_at:
            days_since_training = (datetime.now() - algorithm.last_trained_at).days
            if days_since_training > 30:
                score *= 0.8
            elif days_since_training > 7:
                score *= 0.9
        else:
            score *= 0.7
        
        # Check training data size
        if algorithm.training_data_size < 100:
            score *= 0.6
        elif algorithm.training_data_size < 1000:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    async def _identify_algorithm_issues(self, algorithm: LearningAlgorithm) -> List[str]:
        """Identify issues with algorithm."""
        issues = []
        
        if not algorithm.is_enabled:
            issues.append("Algorithm is disabled")
        
        if algorithm.status != LearningStatus.ACTIVE:
            issues.append(f"Algorithm status is {algorithm.status}")
        
        if not algorithm.last_trained_at:
            issues.append("Algorithm has never been trained")
        elif (datetime.now() - algorithm.last_trained_at).days > 30:
            issues.append("Algorithm training is outdated (>30 days)")
        
        if algorithm.training_data_size < 100:
            issues.append("Insufficient training data")
        
        if algorithm.accuracy_score and algorithm.accuracy_score < 0.5:
            issues.append("Low accuracy score")
        
        return issues
    
    async def _generate_health_recommendations(
        self,
        algorithm: LearningAlgorithm,
        issues: List[str]
    ) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        if "Algorithm is disabled" in issues:
            recommendations.append("Enable algorithm if it should be active")
        
        if "Algorithm training is outdated" in issues or "Algorithm has never been trained" in issues:
            recommendations.append("Retrain algorithm with recent data")
        
        if "Insufficient training data" in issues:
            recommendations.append("Collect more training data")
        
        if "Low accuracy score" in issues:
            recommendations.append("Review and tune algorithm parameters")
            recommendations.append("Validate training data quality")
        
        return recommendations
    
    async def _calculate_performance_trend(self, algorithm_id: uuid.UUID) -> str:
        """Calculate performance trend for algorithm."""
        try:
            # Get recent metrics
            recent_metrics = self.db.query(LearningPerformanceMetric).filter(
                and_(
                    LearningPerformanceMetric.algorithm_id == algorithm_id,
                    LearningPerformanceMetric.recorded_at >= datetime.now() - timedelta(days=7)
                )
            ).order_by(LearningPerformanceMetric.recorded_at).all()
            
            if len(recent_metrics) < 2:
                return "insufficient_data"
            
            # Calculate trend
            values = [m.metric_value for m in recent_metrics]
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            if second_avg > first_avg * 1.05:
                return "improving"
            elif second_avg < first_avg * 0.95:
                return "declining"
            else:
                return "stable"
                
        except Exception:
            return "unknown"


# Dependency injection
_learning_monitoring_service_instance = None


def get_learning_monitoring_service(db_session: Session = None) -> LearningMonitoringService:
    """Get learning monitoring service instance with dependency injection."""
    global _learning_monitoring_service_instance
    if _learning_monitoring_service_instance is None:
        if db_session is None:
            db_session = get_database_session()
        _learning_monitoring_service_instance = LearningMonitoringService(db_session)
    return _learning_monitoring_service_instance
