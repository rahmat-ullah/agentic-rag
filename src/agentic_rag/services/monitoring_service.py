"""
Monitoring Service

This module provides comprehensive monitoring capabilities including Prometheus metrics collection,
health checks, and integration with the existing performance monitoring infrastructure.
"""

import asyncio
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from agentic_rag.adapters.database import get_database_adapter
from agentic_rag.models.monitoring import (
    Metric, Trace, Alert, HealthCheck, ServiceDiscovery, MonitoringConfiguration,
    MetricType, AlertSeverity, AlertStatus, HealthStatus, TraceStatus
)
from agentic_rag.services.notification_service import get_notification_service
from agentic_rag.schemas.monitoring import (
    MetricCreateRequest, TraceCreateRequest, AlertCreateRequest, HealthCheckCreateRequest,
    ServiceDiscoveryCreateRequest, MetricResponse, TraceResponse, AlertResponse,
    HealthCheckResponse, ServiceDiscoveryResponse
)

logger = structlog.get_logger(__name__)


class PrometheusMetrics:
    """Prometheus metrics registry and collectors."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # System metrics
        self.request_count = Counter(
            'agentic_rag_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'agentic_rag_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'agentic_rag_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # Application metrics
        self.search_queries = Counter(
            'agentic_rag_search_queries_total',
            'Total number of search queries',
            ['tenant_id', 'query_type'],
            registry=self.registry
        )
        
        self.document_uploads = Counter(
            'agentic_rag_document_uploads_total',
            'Total number of document uploads',
            ['tenant_id', 'document_type'],
            registry=self.registry
        )
        
        self.embedding_operations = Counter(
            'agentic_rag_embedding_operations_total',
            'Total number of embedding operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.vector_operations = Histogram(
            'agentic_rag_vector_operation_duration_seconds',
            'Vector operation duration in seconds',
            ['operation_type'],
            registry=self.registry
        )
        
        # Business metrics
        self.user_satisfaction = Gauge(
            'agentic_rag_user_satisfaction_score',
            'User satisfaction score',
            ['tenant_id'],
            registry=self.registry
        )
        
        self.system_utilization = Gauge(
            'agentic_rag_system_utilization_percent',
            'System utilization percentage',
            ['resource_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'agentic_rag_errors_total',
            'Total number of errors',
            ['error_type', 'service'],
            registry=self.registry
        )
        
        # Health metrics
        self.health_check_duration = Histogram(
            'agentic_rag_health_check_duration_seconds',
            'Health check duration in seconds',
            ['check_name', 'service'],
            registry=self.registry
        )
        
        self.service_up = Gauge(
            'agentic_rag_service_up',
            'Service availability (1 = up, 0 = down)',
            ['service', 'instance'],
            registry=self.registry
        )


class MonitoringService:
    """Core monitoring service for metrics collection and management."""
    
    def __init__(self):
        self.db = get_database_adapter()
        self.prometheus = PrometheusMetrics()
        self._running = False
        self._collection_task = None
        self._retention_task = None
        
        # Configuration
        self.collection_interval = 30.0  # seconds
        self.retention_days = 30
        self.max_metrics_per_batch = 1000
        
        logger.info("Monitoring service initialized")
    
    async def start(self):
        """Start the monitoring service."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._retention_task = asyncio.create_task(self._retention_loop())

        # Start alerting engine
        try:
            from agentic_rag.services.alerting_engine import get_alerting_engine
            alerting_engine = await get_alerting_engine()
            logger.info("Alerting engine started")
        except Exception as e:
            logger.warning("Failed to start alerting engine", error=str(e))

        logger.info("Monitoring service started")
    
    async def stop(self):
        """Stop the monitoring service."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        if self._retention_task:
            self._retention_task.cancel()
            try:
                await self._retention_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring service stopped")
    
    async def record_metric(
        self,
        tenant_id: UUID,
        request: MetricCreateRequest
    ) -> MetricResponse:
        """Record a new metric."""
        try:
            with self.db.get_session() as session:
                # Create metric record
                metric = Metric(
                    tenant_id=tenant_id,
                    name=request.name,
                    metric_type=request.metric_type.value,
                    value=request.value,
                    labels=request.labels,
                    timestamp=request.timestamp or datetime.utcnow(),
                    source_service=request.source_service,
                    source_instance=request.source_instance,
                    metadata=request.metadata or {}
                )
                
                session.add(metric)
                session.commit()
                session.refresh(metric)
                
                # Update Prometheus metrics
                await self._update_prometheus_metric(request)
                
                logger.info(
                    "Metric recorded",
                    metric_id=str(metric.id),
                    name=request.name,
                    value=request.value
                )
                
                return MetricResponse.from_orm(metric)
                
        except Exception as e:
            logger.error("Failed to record metric", error=str(e))
            raise
    
    async def record_trace(
        self,
        tenant_id: UUID,
        request: TraceCreateRequest
    ) -> TraceResponse:
        """Record a new trace."""
        try:
            with self.db.get_session() as session:
                # Calculate duration if end_time is provided
                duration_ms = None
                if request.end_time:
                    duration = request.end_time - request.start_time
                    duration_ms = duration.total_seconds() * 1000
                
                # Create trace record
                trace = Trace(
                    tenant_id=tenant_id,
                    trace_id=request.trace_id,
                    span_id=request.span_id,
                    parent_span_id=request.parent_span_id,
                    operation_name=request.operation_name,
                    service_name=request.service_name,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    duration_ms=duration_ms,
                    status=request.status.value,
                    tags=request.tags,
                    logs=request.logs or [],
                    error=request.error,
                    error_message=request.error_message,
                    error_stack=request.error_stack,
                    metadata=request.metadata or {}
                )
                
                session.add(trace)
                session.commit()
                session.refresh(trace)
                
                logger.info(
                    "Trace recorded",
                    trace_id=request.trace_id,
                    span_id=request.span_id,
                    operation=request.operation_name
                )
                
                return TraceResponse.from_orm(trace)
                
        except Exception as e:
            logger.error("Failed to record trace", error=str(e))
            raise
    
    async def create_alert(
        self,
        tenant_id: UUID,
        request: AlertCreateRequest
    ) -> AlertResponse:
        """Create a new alert."""
        try:
            with self.db.get_session() as session:
                # Generate fingerprint for alert deduplication
                fingerprint = self._generate_alert_fingerprint(request)
                
                # Check for existing active alert with same fingerprint
                existing_alert = session.query(Alert).filter(
                    and_(
                        Alert.tenant_id == tenant_id,
                        Alert.fingerprint == fingerprint,
                        Alert.status == AlertStatus.ACTIVE.value
                    )
                ).first()
                
                if existing_alert:
                    logger.info(
                        "Alert already exists",
                        fingerprint=fingerprint,
                        existing_id=str(existing_alert.id)
                    )
                    return AlertResponse.from_orm(existing_alert)
                
                # Create new alert
                alert = Alert(
                    tenant_id=tenant_id,
                    rule_name=request.rule_name,
                    alert_name=request.alert_name,
                    fingerprint=fingerprint,
                    severity=request.severity.value,
                    status=AlertStatus.ACTIVE.value,
                    labels=request.labels,
                    annotations=request.annotations,
                    starts_at=datetime.utcnow(),
                    description=request.description,
                    runbook_url=request.runbook_url,
                    dashboard_url=request.dashboard_url,
                    source_service=request.source_service,
                    metadata=request.metadata or {}
                )
                
                session.add(alert)
                session.commit()
                session.refresh(alert)
                
                logger.info(
                    "Alert created",
                    alert_id=str(alert.id),
                    rule_name=request.rule_name,
                    severity=request.severity.value
                )
                
                return AlertResponse.from_orm(alert)
                
        except Exception as e:
            logger.error("Failed to create alert", error=str(e))
            raise
    
    async def record_health_check(
        self,
        tenant_id: UUID,
        request: HealthCheckCreateRequest
    ) -> HealthCheckResponse:
        """Record a health check result."""
        try:
            with self.db.get_session() as session:
                # Create health check record
                health_check = HealthCheck(
                    tenant_id=tenant_id,
                    check_name=request.check_name,
                    service_name=request.service_name,
                    instance_id=request.instance_id,
                    status=request.status.value,
                    response_time_ms=request.response_time_ms,
                    check_timestamp=datetime.utcnow(),
                    details=request.details or {},
                    error_message=request.error_message,
                    metadata=request.metadata or {}
                )
                
                session.add(health_check)
                session.commit()
                session.refresh(health_check)
                
                # Update Prometheus service up metric
                service_up_value = 1 if request.status == HealthStatus.HEALTHY else 0
                self.prometheus.service_up.labels(
                    service=request.service_name,
                    instance=request.instance_id or "unknown"
                ).set(service_up_value)
                
                # Record health check duration
                if request.response_time_ms:
                    self.prometheus.health_check_duration.labels(
                        check_name=request.check_name,
                        service=request.service_name
                    ).observe(request.response_time_ms / 1000.0)
                
                logger.info(
                    "Health check recorded",
                    check_id=str(health_check.id),
                    service=request.service_name,
                    status=request.status.value
                )
                
                return HealthCheckResponse.from_orm(health_check)
                
        except Exception as e:
            logger.error("Failed to record health check", error=str(e))
            raise
    
    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.prometheus.registry).decode('utf-8')
    
    async def _update_prometheus_metric(self, request: MetricCreateRequest):
        """Update Prometheus metrics based on recorded metric."""
        try:
            labels = request.labels or {}
            
            if request.metric_type == MetricType.COUNTER:
                # Find appropriate counter metric
                if request.name.startswith('search_queries'):
                    self.prometheus.search_queries.labels(
                        tenant_id=labels.get('tenant_id', 'unknown'),
                        query_type=labels.get('query_type', 'unknown')
                    ).inc(request.value)
                elif request.name.startswith('document_uploads'):
                    self.prometheus.document_uploads.labels(
                        tenant_id=labels.get('tenant_id', 'unknown'),
                        document_type=labels.get('document_type', 'unknown')
                    ).inc(request.value)
                elif request.name.startswith('errors'):
                    self.prometheus.error_count.labels(
                        error_type=labels.get('error_type', 'unknown'),
                        service=labels.get('service', 'unknown')
                    ).inc(request.value)
            
            elif request.metric_type == MetricType.GAUGE:
                # Find appropriate gauge metric
                if request.name.startswith('user_satisfaction'):
                    self.prometheus.user_satisfaction.labels(
                        tenant_id=labels.get('tenant_id', 'unknown')
                    ).set(request.value)
                elif request.name.startswith('system_utilization'):
                    self.prometheus.system_utilization.labels(
                        resource_type=labels.get('resource_type', 'unknown')
                    ).set(request.value)
                elif request.name.startswith('active_connections'):
                    self.prometheus.active_connections.set(request.value)
            
            elif request.metric_type == MetricType.HISTOGRAM:
                # Find appropriate histogram metric
                if request.name.startswith('vector_operation_duration'):
                    self.prometheus.vector_operations.labels(
                        operation_type=labels.get('operation_type', 'unknown')
                    ).observe(request.value)
                elif request.name.startswith('request_duration'):
                    self.prometheus.request_duration.labels(
                        method=labels.get('method', 'unknown'),
                        endpoint=labels.get('endpoint', 'unknown')
                    ).observe(request.value)
                    
        except Exception as e:
            logger.warning("Failed to update Prometheus metric", error=str(e))
    
    def _generate_alert_fingerprint(self, request: AlertCreateRequest) -> str:
        """Generate a unique fingerprint for alert deduplication."""
        # Create a hash based on rule name and labels
        fingerprint_data = f"{request.rule_name}:{sorted(request.labels.items())}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()
    
    async def _collection_loop(self):
        """Background task for periodic metric collection."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in collection loop", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    async def _retention_loop(self):
        """Background task for data retention cleanup."""
        while self._running:
            try:
                await self._cleanup_old_data()
                # Run retention cleanup every hour
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in retention loop", error=str(e))
                await asyncio.sleep(3600)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        # This would integrate with system monitoring tools
        # For now, we'll just log that collection is happening
        logger.debug("Collecting system metrics")
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data based on retention policies."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            with self.db.get_session() as session:
                # Clean up old metrics
                deleted_metrics = session.query(Metric).filter(
                    Metric.created_at < cutoff_date
                ).delete()
                
                # Clean up old traces
                deleted_traces = session.query(Trace).filter(
                    Trace.created_at < cutoff_date
                ).delete()
                
                # Clean up old health checks
                deleted_health_checks = session.query(HealthCheck).filter(
                    HealthCheck.created_at < cutoff_date
                ).delete()
                
                session.commit()
                
                logger.info(
                    "Cleaned up old monitoring data",
                    deleted_metrics=deleted_metrics,
                    deleted_traces=deleted_traces,
                    deleted_health_checks=deleted_health_checks
                )
                
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))

    async def update_alert_status(
        self,
        tenant_id: UUID,
        alert_id: UUID,
        status: AlertStatus
    ) -> Alert:
        """Update alert status."""
        try:
            async with self.db.get_session() as session:
                # Get the alert
                query = select(Alert).where(
                    and_(
                        Alert.id == alert_id,
                        Alert.tenant_id == tenant_id
                    )
                )
                result = await session.execute(query)
                alert = result.scalar_one_or_none()

                if not alert:
                    raise ValueError(f"Alert {alert_id} not found")

                # Update status
                alert.status = status
                alert.updated_at = datetime.utcnow()

                if status == AlertStatus.RESOLVED:
                    alert.resolved_at = datetime.utcnow()
                elif status == AlertStatus.ACKNOWLEDGED:
                    alert.acknowledged_at = datetime.utcnow()

                await session.commit()
                await session.refresh(alert)

                logger.info(f"Alert status updated to {status.value}", alert_id=str(alert_id))
                return alert

        except Exception as e:
            logger.error("Failed to update alert status", error=str(e))
            raise


# Global monitoring service instance
_monitoring_service: Optional[MonitoringService] = None


async def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance."""
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
        await _monitoring_service.start()
    
    return _monitoring_service
