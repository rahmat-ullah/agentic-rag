"""
Quality Automation Service for Sprint 6 Story 6-05: Automated Quality Improvement System

This service handles automation rules, monitoring, alerting, and automated
execution of quality improvement actions.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from agentic_rag.models.quality_improvement import (
    QualityAssessment,
    QualityImprovement,
    QualityMonitoring,
    AutomationRule,
    QualityAlert,
    ImprovementStatus
)
from agentic_rag.schemas.quality_improvement import (
    CreateQualityMonitoringRequest,
    CreateAutomationRuleRequest
)
from agentic_rag.services.quality_improvement_service import QualityImprovementService


@dataclass
class MonitoringResult:
    """Result of quality monitoring check."""
    
    monitor_id: uuid.UUID
    current_value: float
    threshold_breached: bool
    trend_detected: bool
    alert_triggered: bool
    alert_severity: str
    metadata: Dict[str, Any]


@dataclass
class RuleExecutionResult:
    """Result of automation rule execution."""
    
    rule_id: uuid.UUID
    conditions_met: bool
    actions_executed: List[str]
    improvements_created: List[uuid.UUID]
    execution_success: bool
    failure_reason: Optional[str]
    metadata: Dict[str, Any]


class QualityAutomationService:
    """Service for quality automation, monitoring, and alerting."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = structlog.get_logger(__name__)
        self.quality_service = QualityImprovementService(db_session)
    
    async def create_quality_monitoring(
        self,
        tenant_id: uuid.UUID,
        monitoring_data: CreateQualityMonitoringRequest
    ) -> QualityMonitoring:
        """Create a quality monitoring configuration."""
        try:
            monitor = QualityMonitoring(
                tenant_id=tenant_id,
                monitor_name=monitoring_data.monitor_name,
                monitor_type=monitoring_data.monitor_type,
                target_type=monitoring_data.target_type,
                quality_threshold=monitoring_data.quality_threshold,
                trend_threshold=monitoring_data.trend_threshold,
                pattern_rules=monitoring_data.pattern_rules,
                alert_conditions=monitoring_data.alert_conditions,
                check_interval_minutes=monitoring_data.check_interval_minutes,
                alert_enabled=monitoring_data.alert_enabled,
                alert_recipients=monitoring_data.alert_recipients,
                alert_severity=monitoring_data.alert_severity,
                next_check=datetime.utcnow() + timedelta(minutes=monitoring_data.check_interval_minutes)
            )
            
            self.db.add(monitor)
            self.db.commit()
            self.db.refresh(monitor)
            
            self.logger.info(
                "quality_monitoring_created",
                monitor_id=monitor.id,
                monitor_name=monitor.monitor_name,
                tenant_id=tenant_id
            )
            
            return monitor
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("quality_monitoring_creation_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def create_automation_rule(
        self,
        tenant_id: uuid.UUID,
        rule_data: CreateAutomationRuleRequest
    ) -> AutomationRule:
        """Create an automation rule."""
        try:
            rule = AutomationRule(
                tenant_id=tenant_id,
                rule_name=rule_data.rule_name,
                rule_type=rule_data.rule_type,
                target_type=rule_data.target_type,
                trigger_conditions=rule_data.trigger_conditions,
                condition_logic=rule_data.condition_logic,
                improvement_actions=rule_data.improvement_actions,
                action_parameters=rule_data.action_parameters,
                dry_run_mode=rule_data.dry_run_mode,
                approval_required=rule_data.approval_required,
                max_executions_per_day=rule_data.max_executions_per_day,
                rule_description=rule_data.rule_description,
                rule_priority=rule_data.rule_priority
            )
            
            self.db.add(rule)
            self.db.commit()
            self.db.refresh(rule)
            
            self.logger.info(
                "automation_rule_created",
                rule_id=rule.id,
                rule_name=rule.rule_name,
                tenant_id=tenant_id
            )
            
            return rule
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("automation_rule_creation_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def execute_monitoring_checks(self, tenant_id: Optional[uuid.UUID] = None) -> List[MonitoringResult]:
        """Execute monitoring checks for active monitors."""
        try:
            # Get monitors that need checking
            query = self.db.query(QualityMonitoring).filter(
                and_(
                    QualityMonitoring.is_active == True,
                    QualityMonitoring.next_check <= datetime.utcnow()
                )
            )
            
            if tenant_id:
                query = query.filter(QualityMonitoring.tenant_id == tenant_id)
            
            monitors = query.all()
            results = []
            
            for monitor in monitors:
                result = await self._execute_monitor_check(monitor)
                results.append(result)
                
                # Update monitor
                monitor.last_check = datetime.utcnow()
                monitor.next_check = datetime.utcnow() + timedelta(minutes=monitor.check_interval_minutes)
                monitor.current_value = result.current_value
                
                if result.alert_triggered:
                    monitor.alert_count += 1
                    monitor.last_alert = datetime.utcnow()
            
            self.db.commit()
            
            self.logger.info(
                "monitoring_checks_executed",
                tenant_id=tenant_id,
                monitors_checked=len(monitors),
                alerts_triggered=sum(1 for r in results if r.alert_triggered)
            )
            
            return results
            
        except Exception as e:
            self.logger.error("monitoring_checks_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def execute_automation_rules(self, tenant_id: Optional[uuid.UUID] = None) -> List[RuleExecutionResult]:
        """Execute automation rules for quality improvement."""
        try:
            # Get active automation rules
            query = self.db.query(AutomationRule).filter(
                AutomationRule.is_active == True
            )
            
            if tenant_id:
                query = query.filter(AutomationRule.tenant_id == tenant_id)
            
            rules = query.order_by(desc(AutomationRule.rule_priority)).all()
            results = []
            
            for rule in rules:
                # Check daily execution limit
                today = datetime.utcnow().date()
                daily_executions = self.db.query(func.count(QualityImprovement.id)).filter(
                    and_(
                        QualityImprovement.tenant_id == rule.tenant_id,
                        QualityImprovement.trigger_metadata.contains({"rule_id": str(rule.id)}),
                        func.date(QualityImprovement.created_at) == today
                    )
                ).scalar() or 0
                
                if daily_executions >= rule.max_executions_per_day:
                    self.logger.warning(
                        "automation_rule_daily_limit_reached",
                        rule_id=rule.id,
                        daily_executions=daily_executions,
                        max_executions=rule.max_executions_per_day
                    )
                    continue
                
                result = await self._execute_automation_rule(rule)
                results.append(result)
                
                # Update rule execution stats
                rule.execution_count += 1
                rule.last_execution = datetime.utcnow()
                
                if result.execution_success:
                    rule.success_count += 1
                else:
                    rule.failure_count += 1
            
            self.db.commit()
            
            self.logger.info(
                "automation_rules_executed",
                tenant_id=tenant_id,
                rules_executed=len(results),
                successful_executions=sum(1 for r in results if r.execution_success)
            )
            
            return results
            
        except Exception as e:
            self.logger.error("automation_rules_execution_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def create_quality_alert(
        self,
        tenant_id: uuid.UUID,
        alert_type: str,
        alert_severity: str,
        alert_title: str,
        alert_message: str,
        monitor_id: Optional[uuid.UUID] = None,
        rule_id: Optional[uuid.UUID] = None,
        target_type: Optional[str] = None,
        target_id: Optional[uuid.UUID] = None,
        quality_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        alert_metadata: Optional[Dict[str, Any]] = None
    ) -> QualityAlert:
        """Create a quality alert."""
        try:
            alert = QualityAlert(
                tenant_id=tenant_id,
                monitor_id=monitor_id,
                rule_id=rule_id,
                alert_type=alert_type,
                alert_severity=alert_severity,
                alert_title=alert_title,
                alert_message=alert_message,
                target_type=target_type,
                target_id=target_id,
                quality_value=quality_value,
                threshold_value=threshold_value,
                alert_metadata=alert_metadata
            )
            
            self.db.add(alert)
            self.db.commit()
            self.db.refresh(alert)
            
            self.logger.info(
                "quality_alert_created",
                alert_id=alert.id,
                alert_type=alert_type,
                alert_severity=alert_severity,
                tenant_id=tenant_id
            )
            
            return alert
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("quality_alert_creation_failed", error=str(e), tenant_id=tenant_id)
            raise
    
    async def _execute_monitor_check(self, monitor: QualityMonitoring) -> MonitoringResult:
        """Execute a single monitor check."""
        try:
            current_value = 0.0
            threshold_breached = False
            trend_detected = False
            alert_triggered = False
            alert_severity = monitor.alert_severity
            
            # Get current quality value based on monitor type
            if monitor.monitor_type == "threshold":
                current_value = await self._get_current_quality_value(monitor)
                threshold_breached = (
                    monitor.quality_threshold is not None and 
                    current_value < monitor.quality_threshold
                )
            elif monitor.monitor_type == "trend":
                current_value, trend_detected = await self._check_quality_trend(monitor)
            elif monitor.monitor_type == "pattern":
                current_value, pattern_detected = await self._check_quality_pattern(monitor)
                threshold_breached = pattern_detected
            
            # Determine if alert should be triggered
            alert_triggered = threshold_breached or trend_detected
            
            # Create alert if triggered and alerts are enabled
            if alert_triggered and monitor.alert_enabled:
                await self.create_quality_alert(
                    tenant_id=monitor.tenant_id,
                    alert_type="threshold_breach" if threshold_breached else "trend_alert",
                    alert_severity=alert_severity,
                    alert_title=f"Quality Alert: {monitor.monitor_name}",
                    alert_message=f"Quality monitoring alert triggered for {monitor.target_type}",
                    monitor_id=monitor.id,
                    target_type=monitor.target_type,
                    quality_value=current_value,
                    threshold_value=monitor.quality_threshold,
                    alert_metadata={"monitor_type": monitor.monitor_type}
                )
            
            return MonitoringResult(
                monitor_id=monitor.id,
                current_value=current_value,
                threshold_breached=threshold_breached,
                trend_detected=trend_detected,
                alert_triggered=alert_triggered,
                alert_severity=alert_severity,
                metadata={"monitor_type": monitor.monitor_type}
            )
            
        except Exception as e:
            self.logger.error("monitor_check_failed", error=str(e), monitor_id=monitor.id)
            return MonitoringResult(
                monitor_id=monitor.id,
                current_value=0.0,
                threshold_breached=False,
                trend_detected=False,
                alert_triggered=False,
                alert_severity="low",
                metadata={"error": str(e)}
            )
    
    async def _get_current_quality_value(self, monitor: QualityMonitoring) -> float:
        """Get current quality value for threshold monitoring."""
        try:
            # Get recent quality assessments for the target type
            recent_assessments = self.db.query(QualityAssessment).filter(
                and_(
                    QualityAssessment.tenant_id == monitor.tenant_id,
                    QualityAssessment.target_type == monitor.target_type,
                    QualityAssessment.assessment_date >= datetime.utcnow() - timedelta(hours=24)
                )
            ).all()
            
            if recent_assessments:
                return sum(a.overall_quality_score for a in recent_assessments) / len(recent_assessments)
            
            return 0.75  # Default value if no recent assessments
            
        except Exception as e:
            self.logger.error("current_quality_value_failed", error=str(e), monitor_id=monitor.id)
            return 0.5
    
    async def _check_quality_trend(self, monitor: QualityMonitoring) -> Tuple[float, bool]:
        """Check quality trend for trend monitoring."""
        try:
            # Get quality assessments over time
            assessments = self.db.query(QualityAssessment).filter(
                and_(
                    QualityAssessment.tenant_id == monitor.tenant_id,
                    QualityAssessment.target_type == monitor.target_type,
                    QualityAssessment.assessment_date >= datetime.utcnow() - timedelta(days=7)
                )
            ).order_by(QualityAssessment.assessment_date).all()
            
            if len(assessments) < 2:
                return 0.75, False
            
            # Calculate trend
            recent_avg = sum(a.overall_quality_score for a in assessments[-3:]) / min(len(assessments), 3)
            older_avg = sum(a.overall_quality_score for a in assessments[:3]) / min(len(assessments), 3)
            
            trend_change = recent_avg - older_avg
            trend_detected = (
                monitor.trend_threshold is not None and 
                abs(trend_change) > monitor.trend_threshold
            )
            
            return recent_avg, trend_detected
            
        except Exception as e:
            self.logger.error("quality_trend_check_failed", error=str(e), monitor_id=monitor.id)
            return 0.5, False
    
    async def _check_quality_pattern(self, monitor: QualityMonitoring) -> Tuple[float, bool]:
        """Check quality pattern for pattern monitoring."""
        try:
            # Placeholder for pattern detection logic
            # Would implement specific pattern detection based on pattern_rules
            return 0.75, False
            
        except Exception as e:
            self.logger.error("quality_pattern_check_failed", error=str(e), monitor_id=monitor.id)
            return 0.5, False

    async def _execute_automation_rule(self, rule: AutomationRule) -> RuleExecutionResult:
        """Execute a single automation rule."""
        try:
            # Check if rule conditions are met
            conditions_met = await self._evaluate_rule_conditions(rule)

            if not conditions_met:
                return RuleExecutionResult(
                    rule_id=rule.id,
                    conditions_met=False,
                    actions_executed=[],
                    improvements_created=[],
                    execution_success=True,
                    failure_reason=None,
                    metadata={"conditions_evaluated": True}
                )

            # Execute improvement actions if conditions are met
            actions_executed = []
            improvements_created = []

            if rule.dry_run_mode:
                # In dry run mode, just log what would be done
                self.logger.info(
                    "automation_rule_dry_run",
                    rule_id=rule.id,
                    actions=rule.improvement_actions
                )
                actions_executed = rule.improvement_actions
            else:
                # Execute actual improvement actions
                for action in rule.improvement_actions:
                    try:
                        improvement_id = await self._execute_rule_action(rule, action)
                        if improvement_id:
                            improvements_created.append(improvement_id)
                            actions_executed.append(action)
                    except Exception as e:
                        self.logger.error(
                            "rule_action_execution_failed",
                            rule_id=rule.id,
                            action=action,
                            error=str(e)
                        )

            return RuleExecutionResult(
                rule_id=rule.id,
                conditions_met=True,
                actions_executed=actions_executed,
                improvements_created=improvements_created,
                execution_success=True,
                failure_reason=None,
                metadata={"dry_run": rule.dry_run_mode}
            )

        except Exception as e:
            self.logger.error("automation_rule_execution_failed", error=str(e), rule_id=rule.id)
            return RuleExecutionResult(
                rule_id=rule.id,
                conditions_met=False,
                actions_executed=[],
                improvements_created=[],
                execution_success=False,
                failure_reason=str(e),
                metadata={"error": True}
            )

    async def _evaluate_rule_conditions(self, rule: AutomationRule) -> bool:
        """Evaluate if rule conditions are met."""
        try:
            condition_results = []

            for condition_key, condition_config in rule.trigger_conditions.items():
                result = await self._evaluate_single_condition(
                    rule.tenant_id,
                    rule.target_type,
                    condition_key,
                    condition_config
                )
                condition_results.append(result)

            # Apply condition logic (AND/OR)
            if rule.condition_logic == "AND":
                return all(condition_results)
            elif rule.condition_logic == "OR":
                return any(condition_results)
            else:
                return all(condition_results)  # Default to AND

        except Exception as e:
            self.logger.error("rule_condition_evaluation_failed", error=str(e), rule_id=rule.id)
            return False

    async def _evaluate_single_condition(
        self,
        tenant_id: uuid.UUID,
        target_type: str,
        condition_key: str,
        condition_config: Dict[str, Any]
    ) -> bool:
        """Evaluate a single rule condition."""
        try:
            operator = condition_config.get("operator", "=")
            threshold_value = condition_config.get("value")

            # Get current value for the condition
            current_value = await self._get_condition_value(tenant_id, target_type, condition_key)

            # Evaluate condition based on operator
            if operator == "<":
                return current_value < threshold_value
            elif operator == "<=":
                return current_value <= threshold_value
            elif operator == ">":
                return current_value > threshold_value
            elif operator == ">=":
                return current_value >= threshold_value
            elif operator == "=":
                return current_value == threshold_value
            elif operator == "!=":
                return current_value != threshold_value
            else:
                self.logger.warning("unknown_condition_operator", operator=operator)
                return False

        except Exception as e:
            self.logger.error("single_condition_evaluation_failed", error=str(e), condition_key=condition_key)
            return False

    async def _get_condition_value(self, tenant_id: uuid.UUID, target_type: str, condition_key: str) -> float:
        """Get current value for a condition."""
        try:
            if condition_key == "confidence_score":
                # Get average confidence from recent assessments
                assessments = self.db.query(QualityAssessment).filter(
                    and_(
                        QualityAssessment.tenant_id == tenant_id,
                        QualityAssessment.target_type == target_type,
                        QualityAssessment.confidence_level.isnot(None)
                    )
                ).limit(10).all()

                if assessments:
                    return sum(a.confidence_level for a in assessments) / len(assessments)
                return 0.8

            elif condition_key == "quality_score":
                # Get average quality score from recent assessments
                assessments = self.db.query(QualityAssessment).filter(
                    and_(
                        QualityAssessment.tenant_id == tenant_id,
                        QualityAssessment.target_type == target_type
                    )
                ).limit(10).all()

                if assessments:
                    return sum(a.overall_quality_score for a in assessments) / len(assessments)
                return 0.75

            elif condition_key == "negative_feedback_rate":
                # Calculate negative feedback rate
                # This would integrate with feedback system
                return 0.2  # Placeholder

            elif condition_key == "correction_frequency":
                # Calculate correction frequency
                # This would integrate with correction system
                return 2.0  # Placeholder

            else:
                self.logger.warning("unknown_condition_key", condition_key=condition_key)
                return 0.0

        except Exception as e:
            self.logger.error("condition_value_retrieval_failed", error=str(e), condition_key=condition_key)
            return 0.0

    async def _execute_rule_action(self, rule: AutomationRule, action: str) -> Optional[uuid.UUID]:
        """Execute a single rule action."""
        try:
            # Create improvement based on action type
            from agentic_rag.schemas.quality_improvement import CreateQualityImprovementRequest
            from agentic_rag.models.quality_improvement import QualityIssueType, ImprovementActionType

            # Map action to improvement type and action
            action_mapping = {
                "link_revalidation": (QualityIssueType.LOW_QUALITY_LINK, ImprovementActionType.LINK_REVALIDATION),
                "content_reprocessing": (QualityIssueType.POOR_CONTENT_QUALITY, ImprovementActionType.CONTENT_REPROCESSING),
                "embedding_update": (QualityIssueType.POOR_CONTENT_QUALITY, ImprovementActionType.EMBEDDING_UPDATE),
                "metadata_refresh": (QualityIssueType.POOR_CONTENT_QUALITY, ImprovementActionType.METADATA_REFRESH),
                "quality_flagging": (QualityIssueType.LOW_QUALITY_LINK, ImprovementActionType.QUALITY_FLAGGING)
            }

            if action not in action_mapping:
                self.logger.warning("unknown_rule_action", action=action, rule_id=rule.id)
                return None

            issue_type, improvement_action = action_mapping[action]

            # Create improvement request
            improvement_request = CreateQualityImprovementRequest(
                improvement_type=issue_type,
                target_type=rule.target_type,
                target_id=uuid.uuid4(),  # Would be determined by rule logic
                trigger_reason=f"Automation rule: {rule.rule_name}",
                improvement_action=improvement_action,
                trigger_metadata={"rule_id": str(rule.id), "automated": True}
            )

            # Create improvement through quality service
            improvement = await self.quality_service.create_quality_improvement(
                rule.tenant_id, improvement_request
            )

            # Execute improvement if not requiring approval
            if not rule.approval_required:
                await self.quality_service.execute_improvement_action(improvement.id)

            return improvement.id

        except Exception as e:
            self.logger.error("rule_action_execution_failed", error=str(e), rule_id=rule.id, action=action)
            return None
