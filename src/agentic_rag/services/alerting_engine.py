"""
Alerting Engine

This module provides the alerting engine that evaluates alert rules,
manages alert lifecycle, and triggers notifications.
"""

import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID, uuid4

import structlog
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.adapters.database import get_database_adapter
from agentic_rag.models.monitoring import (
    Alert, Metric, HealthCheck, AlertSeverity, AlertStatus
)
from agentic_rag.schemas.monitoring import AlertCreateRequest
from agentic_rag.services.monitoring_service import get_monitoring_service
from agentic_rag.services.notification_service import get_notification_service

logger = structlog.get_logger(__name__)


class AlertRule:
    """Alert rule definition."""
    
    def __init__(
        self,
        name: str,
        description: str,
        condition: Callable,
        severity: AlertSeverity,
        threshold: float,
        duration: int = 300,  # seconds
        labels: Dict[str, str] = None,
        annotations: Dict[str, str] = None,
        runbook_url: str = None
    ):
        self.name = name
        self.description = description
        self.condition = condition
        self.severity = severity
        self.threshold = threshold
        self.duration = duration
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.runbook_url = runbook_url
        self._last_evaluation = None
        self._firing_since = None
    
    async def evaluate(self, tenant_id: UUID) -> Optional[Dict[str, Any]]:
        """Evaluate the alert rule."""
        try:
            # Call the condition function
            result = await self.condition(tenant_id, self.threshold)
            
            current_time = datetime.utcnow()
            
            if result["firing"]:
                # Rule is firing
                if self._firing_since is None:
                    self._firing_since = current_time
                
                # Check if firing for long enough
                firing_duration = (current_time - self._firing_since).total_seconds()
                if firing_duration >= self.duration:
                    return {
                        "firing": True,
                        "value": result["value"],
                        "firing_duration": firing_duration,
                        "labels": {**self.labels, **result.get("labels", {})},
                        "annotations": {**self.annotations, **result.get("annotations", {})}
                    }
            else:
                # Rule is not firing
                if self._firing_since is not None:
                    # Was firing, now resolved
                    self._firing_since = None
                    return {
                        "firing": False,
                        "resolved": True,
                        "value": result["value"]
                    }
                self._firing_since = None
            
            self._last_evaluation = current_time
            return None
            
        except Exception as e:
            logger.error(f"Failed to evaluate alert rule {self.name}", error=str(e))
            return None


class AlertingEngine:
    """Engine for evaluating alert rules and managing alert lifecycle."""
    
    def __init__(self):
        self.db = get_database_adapter()
        self._rules = []
        self._running = False
        self._evaluation_task = None
        self._evaluation_interval = 30.0  # seconds
        self._active_alerts = {}  # fingerprint -> alert_id
        
        # Load built-in rules
        self._load_builtin_rules()
        
        logger.info("Alerting engine initialized")
    
    async def start(self):
        """Start the alerting engine."""
        if self._running:
            return
        
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())
        
        logger.info("Alerting engine started")
    
    async def stop(self):
        """Stop the alerting engine."""
        if not self._running:
            return
        
        self._running = False
        
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alerting engine stopped")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self._rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        self._rules = [r for r in self._rules if r.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    async def _evaluation_loop(self):
        """Background task for evaluating alert rules."""
        while self._running:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(self._evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in alert evaluation loop", error=str(e))
                await asyncio.sleep(self._evaluation_interval)
    
    async def _evaluate_all_rules(self):
        """Evaluate all alert rules."""
        try:
            # Get all tenants (for now, use a default system tenant)
            system_tenant_id = UUID("00000000-0000-0000-0000-000000000000")
            
            for rule in self._rules:
                try:
                    result = await rule.evaluate(system_tenant_id)
                    if result:
                        await self._handle_rule_result(rule, result, system_tenant_id)
                except Exception as e:
                    logger.error(f"Failed to evaluate rule {rule.name}", error=str(e))
            
        except Exception as e:
            logger.error("Failed to evaluate alert rules", error=str(e))
    
    async def _handle_rule_result(
        self,
        rule: AlertRule,
        result: Dict[str, Any],
        tenant_id: UUID
    ):
        """Handle the result of a rule evaluation."""
        try:
            # Generate fingerprint for deduplication
            fingerprint_data = f"{rule.name}:{result.get('labels', {})}"
            fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()
            
            if result["firing"]:
                # Create or update alert
                await self._create_or_update_alert(rule, result, fingerprint, tenant_id)
            elif result.get("resolved"):
                # Resolve alert
                await self._resolve_alert(fingerprint, tenant_id)
            
        except Exception as e:
            logger.error(f"Failed to handle rule result for {rule.name}", error=str(e))
    
    async def _create_or_update_alert(
        self,
        rule: AlertRule,
        result: Dict[str, Any],
        fingerprint: str,
        tenant_id: UUID
    ):
        """Create or update an alert."""
        try:
            monitoring_service = await get_monitoring_service()
            
            # Check if alert already exists
            if fingerprint in self._active_alerts:
                # Alert already exists, just log
                logger.debug(f"Alert {rule.name} still firing")
                return
            
            # Create new alert
            alert_request = AlertCreateRequest(
                rule_name=rule.name,
                alert_name=rule.name,
                severity=rule.severity,
                labels=result["labels"],
                annotations=result["annotations"],
                description=rule.description,
                runbook_url=rule.runbook_url,
                source_service="alerting-engine"
            )
            
            alert = await monitoring_service.create_alert(tenant_id, alert_request)
            self._active_alerts[fingerprint] = alert.id
            
            # Send notification
            notification_service = await get_notification_service()
            await notification_service.send_alert_notification(alert)
            
            logger.info(
                "Alert created and notification sent",
                rule=rule.name,
                alert_id=str(alert.id),
                fingerprint=fingerprint
            )
            
        except Exception as e:
            logger.error(f"Failed to create alert for rule {rule.name}", error=str(e))
    
    async def _resolve_alert(self, fingerprint: str, tenant_id: UUID):
        """Resolve an alert."""
        try:
            alert_id = self._active_alerts.get(fingerprint)
            if not alert_id:
                return
            
            monitoring_service = await get_monitoring_service()
            
            # Update alert status to resolved
            await monitoring_service.update_alert_status(
                tenant_id,
                alert_id,
                AlertStatus.RESOLVED
            )
            
            # Remove from active alerts
            del self._active_alerts[fingerprint]
            
            logger.info(
                "Alert resolved",
                alert_id=str(alert_id),
                fingerprint=fingerprint
            )
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {fingerprint}", error=str(e))
    
    def _load_builtin_rules(self):
        """Load built-in alert rules."""
        # High error rate rule
        async def high_error_rate_condition(tenant_id: UUID, threshold: float):
            try:
                monitoring_service = await get_monitoring_service()
                
                # Get error metrics from last 5 minutes
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=5)
                
                # Query error count
                async with self.db.get_session() as session:
                    error_query = select(Metric).where(
                        and_(
                            Metric.tenant_id == tenant_id,
                            Metric.name == "api_errors_total",
                            Metric.timestamp >= start_time,
                            Metric.timestamp <= end_time
                        )
                    )
                    error_result = await session.execute(error_query)
                    error_metrics = error_result.scalars().all()
                    
                    # Query total requests
                    total_query = select(Metric).where(
                        and_(
                            Metric.tenant_id == tenant_id,
                            Metric.name == "api_requests_total",
                            Metric.timestamp >= start_time,
                            Metric.timestamp <= end_time
                        )
                    )
                    total_result = await session.execute(total_query)
                    total_metrics = total_result.scalars().all()
                
                # Calculate error rate
                error_count = sum(m.value for m in error_metrics)
                total_count = sum(m.value for m in total_metrics)
                
                if total_count == 0:
                    error_rate = 0
                else:
                    error_rate = (error_count / total_count) * 100
                
                return {
                    "firing": error_rate > threshold,
                    "value": error_rate,
                    "labels": {"metric": "error_rate"},
                    "annotations": {
                        "current_value": f"{error_rate:.2f}%",
                        "threshold": f"{threshold}%"
                    }
                }
                
            except Exception as e:
                logger.error("Failed to evaluate high error rate condition", error=str(e))
                return {"firing": False, "value": 0}
        
        self.add_rule(AlertRule(
            name="HighErrorRate",
            description="API error rate is above threshold",
            condition=high_error_rate_condition,
            severity=AlertSeverity.CRITICAL,
            threshold=5.0,  # 5%
            duration=120,  # 2 minutes
            labels={"service": "api", "team": "platform"},
            annotations={"summary": "High error rate detected"},
            runbook_url="https://wiki.company.com/runbooks/high-error-rate"
        ))
        
        # Service health rule
        async def service_health_condition(tenant_id: UUID, threshold: float):
            try:
                # Get recent health checks
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=5)
                
                async with self.db.get_session() as session:
                    health_query = select(HealthCheck).where(
                        and_(
                            HealthCheck.tenant_id == tenant_id,
                            HealthCheck.timestamp >= start_time,
                            HealthCheck.timestamp <= end_time
                        )
                    )
                    result = await session.execute(health_query)
                    health_checks = result.scalars().all()
                
                # Check for unhealthy services
                unhealthy_services = []
                for check in health_checks:
                    if check.status.value != "healthy":
                        unhealthy_services.append(check.service_name)
                
                unhealthy_count = len(set(unhealthy_services))
                
                return {
                    "firing": unhealthy_count > threshold,
                    "value": unhealthy_count,
                    "labels": {"metric": "unhealthy_services"},
                    "annotations": {
                        "unhealthy_services": ", ".join(set(unhealthy_services)),
                        "count": str(unhealthy_count)
                    }
                }
                
            except Exception as e:
                logger.error("Failed to evaluate service health condition", error=str(e))
                return {"firing": False, "value": 0}
        
        self.add_rule(AlertRule(
            name="UnhealthyServices",
            description="One or more services are unhealthy",
            condition=service_health_condition,
            severity=AlertSeverity.WARNING,
            threshold=0,  # Any unhealthy service
            duration=60,  # 1 minute
            labels={"service": "health", "team": "platform"},
            annotations={"summary": "Unhealthy services detected"},
            runbook_url="https://wiki.company.com/runbooks/service-health"
        ))
        
        logger.info(f"Loaded {len(self._rules)} built-in alert rules")


# Global alerting engine instance
_alerting_engine: Optional[AlertingEngine] = None


async def get_alerting_engine() -> AlertingEngine:
    """Get the global alerting engine instance."""
    global _alerting_engine
    
    if _alerting_engine is None:
        _alerting_engine = AlertingEngine()
        await _alerting_engine.start()
    
    return _alerting_engine
