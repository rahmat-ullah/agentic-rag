"""
Monitoring and Observability Database Models

This module defines the database models for the monitoring and observability system,
including metrics, traces, alerts, and health checks.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from agentic_rag.models.database import Base


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class TraceStatus(str, Enum):
    """Trace status values."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class Metric(Base):
    """Model for storing application metrics."""
    
    __tablename__ = "metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Metric identification
    name = Column(String(255), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # MetricType enum
    labels = Column(JSONB, nullable=False, default=dict)
    
    # Metric data
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Additional metadata
    source_service = Column(String(100), nullable=True, index=True)
    source_instance = Column(String(100), nullable=True)
    metadata = Column(JSONB, nullable=True, default=dict)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_metrics_tenant_name_timestamp', 'tenant_id', 'name', 'timestamp'),
        Index('idx_metrics_service_timestamp', 'source_service', 'timestamp'),
        Index('idx_metrics_type_timestamp', 'metric_type', 'timestamp'),
    )


class Trace(Base):
    """Model for storing distributed traces."""
    
    __tablename__ = "traces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Trace identification
    trace_id = Column(String(32), nullable=False, index=True)  # OpenTelemetry trace ID
    span_id = Column(String(16), nullable=False, index=True)   # OpenTelemetry span ID
    parent_span_id = Column(String(16), nullable=True, index=True)
    
    # Trace metadata
    operation_name = Column(String(255), nullable=False, index=True)
    service_name = Column(String(100), nullable=False, index=True)
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)
    
    # Trace status and data
    status = Column(String(50), nullable=False, default=TraceStatus.OK)  # TraceStatus enum
    tags = Column(JSONB, nullable=False, default=dict)
    logs = Column(JSONB, nullable=True, default=list)
    
    # Error information
    error = Column(Boolean, nullable=False, default=False)
    error_message = Column(Text, nullable=True)
    error_stack = Column(Text, nullable=True)
    
    # Additional metadata
    metadata = Column(JSONB, nullable=True, default=dict)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_traces_tenant_trace_id', 'tenant_id', 'trace_id'),
        Index('idx_traces_service_start_time', 'service_name', 'start_time'),
        Index('idx_traces_operation_start_time', 'operation_name', 'start_time'),
        Index('idx_traces_error_start_time', 'error', 'start_time'),
    )


class Alert(Base):
    """Model for storing system alerts."""
    
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Alert identification
    rule_name = Column(String(255), nullable=False, index=True)
    alert_name = Column(String(255), nullable=False)
    fingerprint = Column(String(64), nullable=False, index=True)  # Hash of alert labels
    
    # Alert metadata
    severity = Column(String(50), nullable=False, index=True)  # AlertSeverity enum
    status = Column(String(50), nullable=False, default=AlertStatus.ACTIVE)  # AlertStatus enum
    labels = Column(JSONB, nullable=False, default=dict)
    annotations = Column(JSONB, nullable=False, default=dict)
    
    # Alert timing
    starts_at = Column(DateTime(timezone=True), nullable=False, index=True)
    ends_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Alert details
    description = Column(Text, nullable=True)
    runbook_url = Column(String(500), nullable=True)
    dashboard_url = Column(String(500), nullable=True)
    
    # Escalation and notification
    escalation_level = Column(Integer, nullable=False, default=0)
    notification_sent = Column(Boolean, nullable=False, default=False)
    notification_channels = Column(JSONB, nullable=True, default=list)
    
    # Additional metadata
    source_service = Column(String(100), nullable=True, index=True)
    metadata = Column(JSONB, nullable=True, default=dict)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_alerts_tenant_status_severity', 'tenant_id', 'status', 'severity'),
        Index('idx_alerts_rule_starts_at', 'rule_name', 'starts_at'),
        Index('idx_alerts_fingerprint_starts_at', 'fingerprint', 'starts_at'),
    )


class HealthCheck(Base):
    """Model for storing health check results."""
    
    __tablename__ = "health_checks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Health check identification
    check_name = Column(String(255), nullable=False, index=True)
    service_name = Column(String(100), nullable=False, index=True)
    instance_id = Column(String(100), nullable=True)
    
    # Health check results
    status = Column(String(50), nullable=False, index=True)  # HealthStatus enum
    response_time_ms = Column(Float, nullable=True)
    check_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Health check details
    details = Column(JSONB, nullable=True, default=dict)
    error_message = Column(Text, nullable=True)
    
    # Additional metadata
    metadata = Column(JSONB, nullable=True, default=dict)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_health_checks_tenant_service_timestamp', 'tenant_id', 'service_name', 'check_timestamp'),
        Index('idx_health_checks_status_timestamp', 'status', 'check_timestamp'),
        Index('idx_health_checks_check_name_timestamp', 'check_name', 'check_timestamp'),
    )


class MonitoringConfiguration(Base):
    """Model for storing monitoring system configuration."""

    __tablename__ = "monitoring_configurations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Configuration identification
    config_name = Column(String(255), nullable=False, index=True)
    config_type = Column(String(100), nullable=False, index=True)  # prometheus, grafana, alerting, etc.

    # Configuration data
    config_data = Column(JSONB, nullable=False, default=dict)
    is_active = Column(Boolean, nullable=False, default=True)

    # Version control
    version = Column(Integer, nullable=False, default=1)
    previous_version_id = Column(UUID(as_uuid=True), nullable=True)

    # Additional metadata
    description = Column(Text, nullable=True)
    metadata = Column(JSONB, nullable=True, default=dict)

    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    created_by = Column(String(255), nullable=True)

    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_monitoring_configs_tenant_type_active', 'tenant_id', 'config_type', 'is_active'),
        Index('idx_monitoring_configs_name_version', 'config_name', 'version'),
    )


class ServiceDiscovery(Base):
    """Model for storing service discovery information."""

    __tablename__ = "service_discovery"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Service identification
    service_name = Column(String(100), nullable=False, index=True)
    instance_id = Column(String(100), nullable=False, index=True)

    # Service endpoint information
    host = Column(String(255), nullable=False)
    port = Column(Integer, nullable=False)
    protocol = Column(String(10), nullable=False, default="http")
    metrics_path = Column(String(255), nullable=False, default="/metrics")
    health_path = Column(String(255), nullable=False, default="/health")

    # Service metadata
    labels = Column(JSONB, nullable=False, default=dict)
    tags = Column(JSONB, nullable=False, default=dict)

    # Service status
    is_active = Column(Boolean, nullable=False, default=True)
    last_seen = Column(DateTime(timezone=True), nullable=False, default=func.now())

    # Additional metadata
    metadata = Column(JSONB, nullable=True, default=dict)

    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_service_discovery_tenant_service', 'tenant_id', 'service_name'),
        Index('idx_service_discovery_active_last_seen', 'is_active', 'last_seen'),
        Index('idx_service_discovery_instance', 'instance_id'),
    )
