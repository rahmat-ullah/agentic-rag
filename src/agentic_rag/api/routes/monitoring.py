"""
Monitoring and Observability API Routes

This module provides REST API endpoints for the monitoring and observability system,
including metrics collection, traces, alerts, and health checks.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse
import structlog

from agentic_rag.api.dependencies.auth import get_current_user, require_admin
from agentic_rag.api.dependencies.tenant import get_tenant_id
from agentic_rag.schemas.monitoring import (
    MetricCreateRequest, MetricResponse, MetricsListResponse,
    TraceCreateRequest, TraceResponse, TracesListResponse,
    AlertCreateRequest, AlertResponse, AlertsListResponse, AlertUpdateRequest,
    HealthCheckCreateRequest, HealthCheckResponse, HealthChecksListResponse,
    ServiceDiscoveryCreateRequest, ServiceDiscoveryResponse, ServicesListResponse,
    MetricsQueryParams, TracesQueryParams, AlertsQueryParams
)
from agentic_rag.schemas.base import BaseResponse
from agentic_rag.services.monitoring_service import get_monitoring_service
from agentic_rag.services.log_aggregation_service import get_log_aggregation_service, LogEntry
from agentic_rag.models.monitoring import AlertStatus, HealthStatus

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/metrics", response_model=MetricResponse)
async def create_metric(
    request: MetricCreateRequest,
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """Create a new metric record."""
    try:
        result = await monitoring_service.record_metric(tenant_id, request)
        
        logger.info(
            "Metric created via API",
            metric_id=str(result.id),
            name=request.name,
            user_id=str(current_user.id)
        )
        
        return result
        
    except Exception as e:
        logger.error("Failed to create metric", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create metric")


@router.get("/metrics", response_model=MetricsListResponse)
async def list_metrics(
    name: Optional[str] = Query(None, description="Filter by metric name"),
    metric_type: Optional[str] = Query(None, description="Filter by metric type"),
    source_service: Optional[str] = Query(None, description="Filter by source service"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """List metrics with optional filtering."""
    try:
        # This would be implemented in the monitoring service
        # For now, return empty list
        return MetricsListResponse(
            message="Metrics retrieved successfully",
            data=[],
            total=0,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error("Failed to list metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list metrics")


@router.post("/traces", response_model=TraceResponse)
async def create_trace(
    request: TraceCreateRequest,
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """Create a new trace record."""
    try:
        result = await monitoring_service.record_trace(tenant_id, request)
        
        logger.info(
            "Trace created via API",
            trace_id=request.trace_id,
            span_id=request.span_id,
            user_id=str(current_user.id)
        )
        
        return result
        
    except Exception as e:
        logger.error("Failed to create trace", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create trace")


@router.get("/traces", response_model=TracesListResponse)
async def list_traces(
    trace_id: Optional[str] = Query(None, description="Filter by trace ID"),
    service_name: Optional[str] = Query(None, description="Filter by service name"),
    operation_name: Optional[str] = Query(None, description="Filter by operation name"),
    status: Optional[str] = Query(None, description="Filter by trace status"),
    error: Optional[bool] = Query(None, description="Filter by error status"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    min_duration_ms: Optional[float] = Query(None, description="Minimum duration filter"),
    max_duration_ms: Optional[float] = Query(None, description="Maximum duration filter"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """List traces with optional filtering."""
    try:
        # This would be implemented in the monitoring service
        # For now, return empty list
        return TracesListResponse(
            message="Traces retrieved successfully",
            data=[],
            total=0,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error("Failed to list traces", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list traces")


@router.post("/alerts", response_model=AlertResponse)
async def create_alert(
    request: AlertCreateRequest,
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """Create a new alert."""
    try:
        result = await monitoring_service.create_alert(tenant_id, request)
        
        logger.info(
            "Alert created via API",
            alert_id=str(result.id),
            rule_name=request.rule_name,
            severity=request.severity.value,
            user_id=str(current_user.id)
        )
        
        return result
        
    except Exception as e:
        logger.error("Failed to create alert", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create alert")


@router.get("/alerts", response_model=AlertsListResponse)
async def list_alerts(
    rule_name: Optional[str] = Query(None, description="Filter by rule name"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status: Optional[str] = Query(None, description="Filter by status"),
    source_service: Optional[str] = Query(None, description="Filter by source service"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """List alerts with optional filtering."""
    try:
        # This would be implemented in the monitoring service
        # For now, return empty list
        return AlertsListResponse(
            message="Alerts retrieved successfully",
            data=[],
            total=0,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error("Failed to list alerts", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list alerts")


@router.patch("/alerts/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: UUID,
    request: AlertUpdateRequest,
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """Update an alert (acknowledge, resolve, etc.)."""
    try:
        # This would be implemented in the monitoring service
        # For now, raise not implemented
        raise HTTPException(status_code=501, detail="Alert update not implemented yet")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update alert", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update alert")


@router.post("/health-checks", response_model=HealthCheckResponse)
async def create_health_check(
    request: HealthCheckCreateRequest,
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """Record a health check result."""
    try:
        result = await monitoring_service.record_health_check(tenant_id, request)
        
        logger.info(
            "Health check recorded via API",
            check_id=str(result.id),
            service=request.service_name,
            status=request.status.value,
            user_id=str(current_user.id)
        )
        
        return result
        
    except Exception as e:
        logger.error("Failed to record health check", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to record health check")


@router.get("/health-checks", response_model=HealthChecksListResponse)
async def list_health_checks(
    check_name: Optional[str] = Query(None, description="Filter by check name"),
    service_name: Optional[str] = Query(None, description="Filter by service name"),
    status: Optional[HealthStatus] = Query(None, description="Filter by status"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """List health check results with optional filtering."""
    try:
        # This would be implemented in the monitoring service
        # For now, return empty list
        return HealthChecksListResponse(
            message="Health checks retrieved successfully",
            data=[],
            total=0,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error("Failed to list health checks", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list health checks")


@router.get("/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics(
    monitoring_service = Depends(get_monitoring_service)
):
    """Get Prometheus metrics in text format."""
    try:
        metrics = await monitoring_service.get_prometheus_metrics()
        return Response(content=metrics, media_type="text/plain")
        
    except Exception as e:
        logger.error("Failed to get Prometheus metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get Prometheus metrics")


@router.get("/dashboard/overview")
async def get_dashboard_overview(
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """Get overview dashboard data."""
    try:
        # This would aggregate key metrics for dashboard display
        overview_data = {
            "system_health": "healthy",
            "active_alerts": 0,
            "total_requests_24h": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "user_satisfaction": 0.0,
            "system_utilization": {
                "cpu": 0.0,
                "memory": 0.0,
                "disk": 0.0
            },
            "service_status": {
                "api": "healthy",
                "database": "healthy",
                "vector_store": "healthy",
                "search": "healthy"
            }
        }
        
        return BaseResponse(
            message="Dashboard overview retrieved successfully",
            data=overview_data
        )
        
    except Exception as e:
        logger.error("Failed to get dashboard overview", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get dashboard overview")


@router.get("/service-discovery", response_model=ServicesListResponse)
async def list_services(
    service_name: Optional[str] = Query(None, description="Filter by service name"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """List discovered services."""
    try:
        # This would be implemented in the monitoring service
        # For now, return empty list
        return ServicesListResponse(
            message="Services retrieved successfully",
            data=[],
            total=0,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error("Failed to list services", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list services")


@router.post("/service-discovery", response_model=ServiceDiscoveryResponse)
async def register_service(
    request: ServiceDiscoveryCreateRequest,
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    monitoring_service = Depends(get_monitoring_service)
):
    """Register a service for discovery."""
    try:
        # This would be implemented in the monitoring service
        # For now, raise not implemented
        raise HTTPException(status_code=501, detail="Service registration not implemented yet")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to register service", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to register service")


# Log Aggregation Endpoints

@router.post("/logs", response_model=BaseResponse)
async def ship_log(
    level: str,
    message: str,
    service: str,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    log_service = Depends(get_log_aggregation_service)
):
    """Ship a log entry to the aggregation system."""
    try:
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level.upper(),
            message=message,
            service=service,
            tenant_id=tenant_id,
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id
        )

        await log_service.ship_log(log_entry)

        return BaseResponse(
            success=True,
            message="Log entry shipped successfully"
        )

    except Exception as e:
        logger.error("Failed to ship log", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to ship log")


@router.get("/logs/search")
async def search_logs(
    query: str = Query("*", description="Search query"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    service: Optional[str] = Query(None, description="Service filter"),
    level: Optional[str] = Query(None, description="Log level filter"),
    size: int = Query(100, ge=1, le=1000, description="Number of results"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    log_service = Depends(get_log_aggregation_service)
):
    """Search logs with filters."""
    try:
        results = await log_service.search_logs(
            query=query,
            start_time=start_time,
            end_time=end_time,
            service=service,
            level=level,
            tenant_id=tenant_id,
            size=size,
            sort_order=sort_order
        )

        return results

    except Exception as e:
        logger.error("Failed to search logs", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to search logs")


@router.get("/logs/statistics")
async def get_log_statistics(
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    tenant_id: UUID = Depends(get_tenant_id),
    current_user = Depends(get_current_user),
    log_service = Depends(get_log_aggregation_service)
):
    """Get log statistics and aggregations."""
    try:
        stats = await log_service.get_log_statistics(
            start_time=start_time,
            end_time=end_time,
            tenant_id=tenant_id
        )

        return stats

    except Exception as e:
        logger.error("Failed to get log statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get log statistics")
