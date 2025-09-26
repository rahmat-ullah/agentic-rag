"""
Monitoring and Observability Pydantic Schemas

This module defines the Pydantic schemas for the monitoring and observability system,
including request/response models for metrics, traces, alerts, and health checks.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

from agentic_rag.schemas.base import BaseResponse
from agentic_rag.models.monitoring import (
    MetricType, AlertSeverity, AlertStatus, HealthStatus, TraceStatus
)


# Base schemas
class MetricLabels(BaseModel):
    """Schema for metric labels."""
    
    class Config:
        extra = "allow"  # Allow additional labels


class TraceTag(BaseModel):
    """Schema for trace tags."""
    
    key: str = Field(..., description="Tag key")
    value: Union[str, int, float, bool] = Field(..., description="Tag value")


# Request schemas
class MetricCreateRequest(BaseModel):
    """Schema for creating a new metric."""
    
    name: str = Field(..., description="Metric name", max_length=255)
    metric_type: MetricType = Field(..., description="Type of metric")
    value: float = Field(..., description="Metric value")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
    timestamp: Optional[datetime] = Field(None, description="Metric timestamp")
    source_service: Optional[str] = Field(None, description="Source service name", max_length=100)
    source_instance: Optional[str] = Field(None, description="Source instance ID", max_length=100)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class TraceCreateRequest(BaseModel):
    """Schema for creating a new trace."""
    
    trace_id: str = Field(..., description="OpenTelemetry trace ID", max_length=32)
    span_id: str = Field(..., description="OpenTelemetry span ID", max_length=16)
    parent_span_id: Optional[str] = Field(None, description="Parent span ID", max_length=16)
    operation_name: str = Field(..., description="Operation name", max_length=255)
    service_name: str = Field(..., description="Service name", max_length=100)
    start_time: datetime = Field(..., description="Trace start time")
    end_time: Optional[datetime] = Field(None, description="Trace end time")
    status: TraceStatus = Field(default=TraceStatus.OK, description="Trace status")
    tags: Dict[str, Any] = Field(default_factory=dict, description="Trace tags")
    logs: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Trace logs")
    error: bool = Field(default=False, description="Whether trace has error")
    error_message: Optional[str] = Field(None, description="Error message")
    error_stack: Optional[str] = Field(None, description="Error stack trace")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if v and 'start_time' in values and v < values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v


class AlertCreateRequest(BaseModel):
    """Schema for creating a new alert."""
    
    rule_name: str = Field(..., description="Alert rule name", max_length=255)
    alert_name: str = Field(..., description="Alert name", max_length=255)
    severity: AlertSeverity = Field(..., description="Alert severity")
    labels: Dict[str, str] = Field(default_factory=dict, description="Alert labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")
    description: Optional[str] = Field(None, description="Alert description")
    runbook_url: Optional[str] = Field(None, description="Runbook URL", max_length=500)
    dashboard_url: Optional[str] = Field(None, description="Dashboard URL", max_length=500)
    source_service: Optional[str] = Field(None, description="Source service", max_length=100)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class HealthCheckCreateRequest(BaseModel):
    """Schema for creating a health check result."""
    
    check_name: str = Field(..., description="Health check name", max_length=255)
    service_name: str = Field(..., description="Service name", max_length=100)
    instance_id: Optional[str] = Field(None, description="Instance ID", max_length=100)
    status: HealthStatus = Field(..., description="Health status")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Health check details")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ServiceDiscoveryCreateRequest(BaseModel):
    """Schema for registering a service for discovery."""
    
    service_name: str = Field(..., description="Service name", max_length=100)
    instance_id: str = Field(..., description="Instance ID", max_length=100)
    host: str = Field(..., description="Service host", max_length=255)
    port: int = Field(..., description="Service port", ge=1, le=65535)
    protocol: str = Field(default="http", description="Service protocol")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path", max_length=255)
    health_path: str = Field(default="/health", description="Health endpoint path", max_length=255)
    labels: Dict[str, str] = Field(default_factory=dict, description="Service labels")
    tags: Dict[str, str] = Field(default_factory=dict, description="Service tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


# Response schemas
class MetricResponse(BaseModel):
    """Schema for metric response."""
    
    id: UUID = Field(..., description="Metric ID")
    name: str = Field(..., description="Metric name")
    metric_type: MetricType = Field(..., description="Type of metric")
    value: float = Field(..., description="Metric value")
    labels: Dict[str, str] = Field(..., description="Metric labels")
    timestamp: datetime = Field(..., description="Metric timestamp")
    source_service: Optional[str] = Field(None, description="Source service name")
    source_instance: Optional[str] = Field(None, description="Source instance ID")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        from_attributes = True


class TraceResponse(BaseModel):
    """Schema for trace response."""
    
    id: UUID = Field(..., description="Trace ID")
    trace_id: str = Field(..., description="OpenTelemetry trace ID")
    span_id: str = Field(..., description="OpenTelemetry span ID")
    parent_span_id: Optional[str] = Field(None, description="Parent span ID")
    operation_name: str = Field(..., description="Operation name")
    service_name: str = Field(..., description="Service name")
    start_time: datetime = Field(..., description="Trace start time")
    end_time: Optional[datetime] = Field(None, description="Trace end time")
    duration_ms: Optional[float] = Field(None, description="Trace duration in milliseconds")
    status: TraceStatus = Field(..., description="Trace status")
    tags: Dict[str, Any] = Field(..., description="Trace tags")
    logs: List[Dict[str, Any]] = Field(..., description="Trace logs")
    error: bool = Field(..., description="Whether trace has error")
    error_message: Optional[str] = Field(None, description="Error message")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    """Schema for alert response."""
    
    id: UUID = Field(..., description="Alert ID")
    rule_name: str = Field(..., description="Alert rule name")
    alert_name: str = Field(..., description="Alert name")
    fingerprint: str = Field(..., description="Alert fingerprint")
    severity: AlertSeverity = Field(..., description="Alert severity")
    status: AlertStatus = Field(..., description="Alert status")
    labels: Dict[str, str] = Field(..., description="Alert labels")
    annotations: Dict[str, str] = Field(..., description="Alert annotations")
    starts_at: datetime = Field(..., description="Alert start time")
    ends_at: Optional[datetime] = Field(None, description="Alert end time")
    acknowledged_at: Optional[datetime] = Field(None, description="Alert acknowledgment time")
    resolved_at: Optional[datetime] = Field(None, description="Alert resolution time")
    description: Optional[str] = Field(None, description="Alert description")
    runbook_url: Optional[str] = Field(None, description="Runbook URL")
    dashboard_url: Optional[str] = Field(None, description="Dashboard URL")
    escalation_level: int = Field(..., description="Escalation level")
    notification_sent: bool = Field(..., description="Whether notification was sent")
    source_service: Optional[str] = Field(None, description="Source service")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    
    id: UUID = Field(..., description="Health check ID")
    check_name: str = Field(..., description="Health check name")
    service_name: str = Field(..., description="Service name")
    instance_id: Optional[str] = Field(None, description="Instance ID")
    status: HealthStatus = Field(..., description="Health status")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    check_timestamp: datetime = Field(..., description="Check timestamp")
    details: Dict[str, Any] = Field(..., description="Health check details")
    error_message: Optional[str] = Field(None, description="Error message")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        from_attributes = True


class ServiceDiscoveryResponse(BaseModel):
    """Schema for service discovery response."""
    
    id: UUID = Field(..., description="Service discovery ID")
    service_name: str = Field(..., description="Service name")
    instance_id: str = Field(..., description="Instance ID")
    host: str = Field(..., description="Service host")
    port: int = Field(..., description="Service port")
    protocol: str = Field(..., description="Service protocol")
    metrics_path: str = Field(..., description="Metrics endpoint path")
    health_path: str = Field(..., description="Health endpoint path")
    labels: Dict[str, str] = Field(..., description="Service labels")
    tags: Dict[str, str] = Field(..., description="Service tags")
    is_active: bool = Field(..., description="Whether service is active")
    last_seen: datetime = Field(..., description="Last seen timestamp")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True


# List response schemas
class MetricsListResponse(BaseResponse):
    """Schema for metrics list response."""
    
    data: List[MetricResponse] = Field(..., description="List of metrics")


class TracesListResponse(BaseResponse):
    """Schema for traces list response."""
    
    data: List[TraceResponse] = Field(..., description="List of traces")


class AlertsListResponse(BaseResponse):
    """Schema for alerts list response."""
    
    data: List[AlertResponse] = Field(..., description="List of alerts")


class HealthChecksListResponse(BaseResponse):
    """Schema for health checks list response."""
    
    data: List[HealthCheckResponse] = Field(..., description="List of health checks")


class ServicesListResponse(BaseResponse):
    """Schema for services list response."""
    
    data: List[ServiceDiscoveryResponse] = Field(..., description="List of services")


# Update request schemas
class AlertUpdateRequest(BaseModel):
    """Schema for updating an alert."""
    
    status: Optional[AlertStatus] = Field(None, description="Alert status")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    escalation_level: Optional[int] = Field(None, description="Escalation level")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# Query parameter schemas
class MetricsQueryParams(BaseModel):
    """Schema for metrics query parameters."""
    
    name: Optional[str] = Field(None, description="Filter by metric name")
    metric_type: Optional[MetricType] = Field(None, description="Filter by metric type")
    source_service: Optional[str] = Field(None, description="Filter by source service")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    labels: Optional[Dict[str, str]] = Field(None, description="Filter by labels")
    limit: int = Field(default=100, description="Maximum number of results", ge=1, le=1000)
    offset: int = Field(default=0, description="Number of results to skip", ge=0)


class TracesQueryParams(BaseModel):
    """Schema for traces query parameters."""
    
    trace_id: Optional[str] = Field(None, description="Filter by trace ID")
    service_name: Optional[str] = Field(None, description="Filter by service name")
    operation_name: Optional[str] = Field(None, description="Filter by operation name")
    status: Optional[TraceStatus] = Field(None, description="Filter by trace status")
    error: Optional[bool] = Field(None, description="Filter by error status")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    min_duration_ms: Optional[float] = Field(None, description="Minimum duration filter")
    max_duration_ms: Optional[float] = Field(None, description="Maximum duration filter")
    limit: int = Field(default=100, description="Maximum number of results", ge=1, le=1000)
    offset: int = Field(default=0, description="Number of results to skip", ge=0)


class AlertsQueryParams(BaseModel):
    """Schema for alerts query parameters."""
    
    rule_name: Optional[str] = Field(None, description="Filter by rule name")
    severity: Optional[AlertSeverity] = Field(None, description="Filter by severity")
    status: Optional[AlertStatus] = Field(None, description="Filter by status")
    source_service: Optional[str] = Field(None, description="Filter by source service")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    limit: int = Field(default=100, description="Maximum number of results", ge=1, le=1000)
    offset: int = Field(default=0, description="Number of results to skip", ge=0)
