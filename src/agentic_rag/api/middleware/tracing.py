"""
Tracing Middleware

This module provides middleware for automatic request tracing and monitoring
integration with OpenTelemetry and the monitoring service.
"""

import time
import uuid
from datetime import datetime
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from agentic_rag.services.tracing_service import get_tracing_service, start_span
from agentic_rag.services.monitoring_service import get_monitoring_service
from agentic_rag.schemas.monitoring import MetricCreateRequest, TraceCreateRequest
from agentic_rag.models.monitoring import MetricType, TraceStatus

logger = structlog.get_logger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic request tracing and metrics collection."""
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health", "/metrics", "/docs", "/openapi.json", "/favicon.ico"
        ]
        self.tracing_service = get_tracing_service()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing and metrics collection."""
        # Skip tracing for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Extract tenant ID if available
        tenant_id = getattr(request.state, 'tenant_id', None)
        
        # Create span attributes
        span_attributes = {
            "http.method": request.method,
            "http.url": str(request.url),
            "http.scheme": request.url.scheme,
            "http.host": request.url.hostname,
            "http.target": request.url.path,
            "http.user_agent": request.headers.get("user-agent", ""),
            "request.id": request_id,
        }
        
        if tenant_id:
            span_attributes["tenant.id"] = str(tenant_id)
        
        # Start span for the request
        with start_span(
            f"{request.method} {request.url.path}",
            attributes=span_attributes,
            tenant_id=tenant_id
        ) as span:
            
            response = None
            error = None
            
            try:
                # Process the request
                response = await call_next(request)
                
                # Add response attributes to span
                if span:
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("http.response.size", 
                                     response.headers.get("content-length", "0"))
                
            except Exception as e:
                error = e
                logger.error(
                    "Request processing failed",
                    request_id=request_id,
                    method=request.method,
                    path=request.url.path,
                    error=str(e)
                )
                
                # Record exception in span
                if span:
                    span.record_exception(e)
                    span.set_attribute("http.status_code", 500)
                
                # Re-raise the exception
                raise
            
            finally:
                # Calculate request duration
                end_time = time.time()
                duration_seconds = end_time - start_time
                duration_ms = duration_seconds * 1000
                
                # Record metrics and traces asynchronously
                try:
                    await self._record_request_metrics(
                        request, response, duration_seconds, error, tenant_id
                    )
                    
                    await self._record_request_trace(
                        request, response, start_time, end_time, duration_ms, error, tenant_id
                    )
                    
                except Exception as e:
                    logger.warning("Failed to record request metrics/traces", error=str(e))
        
        return response
    
    async def _record_request_metrics(
        self, 
        request: Request, 
        response: Response, 
        duration_seconds: float, 
        error: Exception, 
        tenant_id: str
    ):
        """Record request metrics."""
        try:
            monitoring_service = await get_monitoring_service()
            
            # Determine status for metrics
            if error:
                status = "error"
                status_code = 500
            elif response:
                status = "success" if response.status_code < 400 else "error"
                status_code = response.status_code
            else:
                status = "unknown"
                status_code = 0
            
            # Record request count metric
            count_request = MetricCreateRequest(
                name="api_requests_total",
                metric_type=MetricType.COUNTER,
                value=1.0,
                labels={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status": str(status_code),
                    "status_class": f"{status_code // 100}xx" if status_code > 0 else "unknown"
                },
                source_service="api-service"
            )
            
            await monitoring_service.record_metric(tenant_id, count_request)
            
            # Record request duration metric
            duration_request = MetricCreateRequest(
                name="api_request_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                value=duration_seconds,
                labels={
                    "method": request.method,
                    "endpoint": request.url.path
                },
                source_service="api-service"
            )
            
            await monitoring_service.record_metric(tenant_id, duration_request)
            
            # Record error metric if there was an error
            if error or (response and response.status_code >= 400):
                error_request = MetricCreateRequest(
                    name="api_errors_total",
                    metric_type=MetricType.COUNTER,
                    value=1.0,
                    labels={
                        "method": request.method,
                        "endpoint": request.url.path,
                        "status": str(status_code),
                        "error_type": type(error).__name__ if error else "http_error"
                    },
                    source_service="api-service"
                )
                
                await monitoring_service.record_metric(tenant_id, error_request)
            
        except Exception as e:
            logger.warning("Failed to record request metrics", error=str(e))
    
    async def _record_request_trace(
        self, 
        request: Request, 
        response: Response, 
        start_time: float, 
        end_time: float, 
        duration_ms: float, 
        error: Exception, 
        tenant_id: str
    ):
        """Record request trace."""
        try:
            monitoring_service = await get_monitoring_service()
            
            # Get trace context from OpenTelemetry
            trace_context = self.tracing_service.get_trace_context()
            
            if not trace_context:
                # Generate trace IDs if not available
                trace_id = f"{uuid.uuid4().hex}"
                span_id = f"{uuid.uuid4().hex[:16]}"
            else:
                trace_id = trace_context["trace_id"]
                span_id = trace_context["span_id"]
            
            # Determine trace status
            if error:
                trace_status = TraceStatus.ERROR
            elif response and response.status_code >= 400:
                trace_status = TraceStatus.ERROR
            else:
                trace_status = TraceStatus.OK
            
            # Create trace tags
            tags = {
                "http.method": request.method,
                "http.url": str(request.url),
                "http.status_code": response.status_code if response else 500,
                "request.id": getattr(request.state, 'request_id', ''),
                "user_agent": request.headers.get("user-agent", ""),
            }
            
            if tenant_id:
                tags["tenant.id"] = str(tenant_id)
            
            # Create trace request
            trace_request = TraceCreateRequest(
                trace_id=trace_id,
                span_id=span_id,
                operation_name=f"{request.method} {request.url.path}",
                service_name="api-service",
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.fromtimestamp(end_time),
                status=trace_status,
                tags=tags,
                error=error is not None,
                error_message=str(error) if error else None
            )
            
            await monitoring_service.record_trace(tenant_id, trace_request)
            
        except Exception as e:
            logger.warning("Failed to record request trace", error=str(e))


class MetricsMiddleware(BaseHTTPMiddleware):
    """Lightweight middleware for basic metrics collection without tracing."""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with basic metrics collection."""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record basic metrics
            await self._record_basic_metrics(request, response, start_time)
            
            return response
            
        except Exception as e:
            # Record error metrics
            await self._record_error_metrics(request, e, start_time)
            raise
    
    async def _record_basic_metrics(self, request: Request, response: Response, start_time: float):
        """Record basic request metrics."""
        try:
            duration = time.time() - start_time
            
            # Update Prometheus metrics directly
            monitoring_service = await get_monitoring_service()
            prometheus = monitoring_service.prometheus
            
            # Request count
            prometheus.request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=str(response.status_code)
            ).inc()
            
            # Request duration
            prometheus.request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Error count if error status
            if response.status_code >= 400:
                prometheus.error_count.labels(
                    error_type="http_error",
                    service="api"
                ).inc()
            
        except Exception as e:
            logger.warning("Failed to record basic metrics", error=str(e))
    
    async def _record_error_metrics(self, request: Request, error: Exception, start_time: float):
        """Record error metrics."""
        try:
            monitoring_service = await get_monitoring_service()
            prometheus = monitoring_service.prometheus
            
            # Error count
            prometheus.error_count.labels(
                error_type=type(error).__name__,
                service="api"
            ).inc()
            
            # Request count with error status
            prometheus.request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status="500"
            ).inc()
            
        except Exception as e:
            logger.warning("Failed to record error metrics", error=str(e))
