"""
Integration tests for the monitoring and observability system.

This module tests the complete monitoring workflow including metrics collection,
tracing, alerting, and health checks.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from agentic_rag.models.monitoring import MetricType, AlertSeverity, HealthStatus, TraceStatus
from agentic_rag.schemas.monitoring import (
    MetricCreateRequest, TraceCreateRequest, AlertCreateRequest, HealthCheckCreateRequest
)
from agentic_rag.services.monitoring_service import get_monitoring_service
from agentic_rag.services.tracing_service import get_tracing_service, initialize_tracing


@pytest.fixture
async def monitoring_service():
    """Get monitoring service instance."""
    service = await get_monitoring_service()
    yield service
    await service.stop()


@pytest.fixture
def tracing_service():
    """Get tracing service instance."""
    service = get_tracing_service()
    service.initialize()
    yield service
    service.shutdown()


@pytest.fixture
def tenant_id():
    """Generate a test tenant ID."""
    return uuid4()


class TestMonitoringService:
    """Test monitoring service functionality."""
    
    async def test_record_metric(self, monitoring_service, tenant_id):
        """Test recording a metric."""
        request = MetricCreateRequest(
            name="test_counter",
            metric_type=MetricType.COUNTER,
            value=1.0,
            labels={"service": "test", "environment": "test"},
            source_service="test-service",
            metadata={"test": True}
        )
        
        result = await monitoring_service.record_metric(tenant_id, request)
        
        assert result.name == "test_counter"
        assert result.metric_type == MetricType.COUNTER
        assert result.value == 1.0
        assert result.labels["service"] == "test"
        assert result.source_service == "test-service"
        assert result.metadata["test"] is True
    
    async def test_record_trace(self, monitoring_service, tenant_id):
        """Test recording a trace."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(milliseconds=100)
        
        request = TraceCreateRequest(
            trace_id="1234567890abcdef1234567890abcdef",
            span_id="1234567890abcdef",
            operation_name="test_operation",
            service_name="test-service",
            start_time=start_time,
            end_time=end_time,
            status=TraceStatus.OK,
            tags={"http.method": "GET", "http.status_code": 200},
            metadata={"test": True}
        )
        
        result = await monitoring_service.record_trace(tenant_id, request)
        
        assert result.trace_id == "1234567890abcdef1234567890abcdef"
        assert result.span_id == "1234567890abcdef"
        assert result.operation_name == "test_operation"
        assert result.service_name == "test-service"
        assert result.status == TraceStatus.OK
        assert result.tags["http.method"] == "GET"
        assert result.duration_ms == 100.0
    
    async def test_create_alert(self, monitoring_service, tenant_id):
        """Test creating an alert."""
        request = AlertCreateRequest(
            rule_name="high_error_rate",
            alert_name="High Error Rate Detected",
            severity=AlertSeverity.CRITICAL,
            labels={"service": "test-service", "environment": "test"},
            annotations={"summary": "Error rate is above threshold"},
            description="The error rate has exceeded the critical threshold",
            source_service="test-service"
        )
        
        result = await monitoring_service.create_alert(tenant_id, request)
        
        assert result.rule_name == "high_error_rate"
        assert result.alert_name == "High Error Rate Detected"
        assert result.severity == AlertSeverity.CRITICAL
        assert result.labels["service"] == "test-service"
        assert result.description == "The error rate has exceeded the critical threshold"
        assert result.fingerprint is not None
    
    async def test_alert_deduplication(self, monitoring_service, tenant_id):
        """Test that duplicate alerts are not created."""
        request = AlertCreateRequest(
            rule_name="duplicate_test",
            alert_name="Duplicate Alert Test",
            severity=AlertSeverity.WARNING,
            labels={"service": "test", "type": "duplicate"},
            annotations={"summary": "This is a duplicate test"}
        )
        
        # Create first alert
        result1 = await monitoring_service.create_alert(tenant_id, request)
        
        # Create second alert with same rule and labels
        result2 = await monitoring_service.create_alert(tenant_id, request)
        
        # Should return the same alert (deduplication)
        assert result1.id == result2.id
        assert result1.fingerprint == result2.fingerprint
    
    async def test_record_health_check(self, monitoring_service, tenant_id):
        """Test recording a health check."""
        request = HealthCheckCreateRequest(
            check_name="database_connectivity",
            service_name="postgres",
            instance_id="postgres-1",
            status=HealthStatus.HEALTHY,
            response_time_ms=25.5,
            details={"connection_pool": "active", "query_test": "passed"},
            metadata={"version": "13.4"}
        )
        
        result = await monitoring_service.record_health_check(tenant_id, request)
        
        assert result.check_name == "database_connectivity"
        assert result.service_name == "postgres"
        assert result.instance_id == "postgres-1"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms == 25.5
        assert result.details["connection_pool"] == "active"
    
    async def test_prometheus_metrics_export(self, monitoring_service):
        """Test Prometheus metrics export."""
        metrics_text = await monitoring_service.get_prometheus_metrics()
        
        assert isinstance(metrics_text, str)
        assert len(metrics_text) > 0
        # Should contain some basic Prometheus metric format
        assert "# HELP" in metrics_text or "# TYPE" in metrics_text


class TestTracingService:
    """Test tracing service functionality."""
    
    def test_start_span(self, tracing_service):
        """Test starting a span."""
        with tracing_service.start_span("test_span", {"test": "value"}) as span:
            assert span is not None
            tracing_service.add_span_event("test_event", {"event": "data"})
            tracing_service.set_span_attribute("custom.attribute", "test_value")
    
    def test_span_exception_handling(self, tracing_service):
        """Test exception handling in spans."""
        try:
            with tracing_service.start_span("error_span") as span:
                assert span is not None
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
    
    def test_get_trace_context(self, tracing_service):
        """Test getting trace context."""
        with tracing_service.start_span("context_test") as span:
            if span:  # Only test if tracing is actually enabled
                context = tracing_service.get_trace_context()
                assert context is not None
                assert "trace_id" in context
                assert "span_id" in context
    
    async def test_custom_span_creation(self, tracing_service, tenant_id):
        """Test creating custom spans."""
        await tracing_service.create_custom_span(
            name="custom_test_span",
            operation_name="test_operation",
            service_name="test-service",
            attributes={"custom": "attribute"},
            tenant_id=tenant_id,
            duration_ms=50.0
        )
        # If no exception is raised, the test passes


class TestMonitoringIntegration:
    """Test integration between monitoring components."""
    
    async def test_metrics_and_tracing_integration(self, monitoring_service, tracing_service, tenant_id):
        """Test integration between metrics and tracing."""
        # Record a metric
        metric_request = MetricCreateRequest(
            name="integration_test_counter",
            metric_type=MetricType.COUNTER,
            value=1.0,
            labels={"integration": "test"},
            source_service="integration-service"
        )
        
        metric_result = await monitoring_service.record_metric(tenant_id, metric_request)
        
        # Create a trace with the same service
        with tracing_service.start_span("integration_test", {"metric_id": str(metric_result.id)}):
            trace_request = TraceCreateRequest(
                trace_id="abcdef1234567890abcdef1234567890",
                span_id="abcdef1234567890",
                operation_name="integration_test",
                service_name="integration-service",
                start_time=datetime.utcnow(),
                tags={"metric.recorded": "true"}
            )
            
            trace_result = await monitoring_service.record_trace(tenant_id, trace_request)
        
        # Verify both were recorded
        assert metric_result.source_service == trace_result.service_name
        assert trace_result.tags["metric.recorded"] == "true"
    
    async def test_health_check_to_alert_flow(self, monitoring_service, tenant_id):
        """Test flow from health check failure to alert creation."""
        # Record a failing health check
        health_request = HealthCheckCreateRequest(
            check_name="service_health",
            service_name="failing-service",
            status=HealthStatus.UNHEALTHY,
            error_message="Service is not responding",
            response_time_ms=5000.0
        )
        
        health_result = await monitoring_service.record_health_check(tenant_id, health_request)
        
        # Create an alert based on the health check failure
        alert_request = AlertCreateRequest(
            rule_name="service_down",
            alert_name="Service Down",
            severity=AlertSeverity.CRITICAL,
            labels={"service": "failing-service", "check": "service_health"},
            annotations={
                "summary": "Service health check failed",
                "description": health_result.error_message
            },
            source_service="failing-service"
        )
        
        alert_result = await monitoring_service.create_alert(tenant_id, alert_request)
        
        # Verify the alert was created with correct information
        assert alert_result.severity == AlertSeverity.CRITICAL
        assert alert_result.labels["service"] == "failing-service"
        assert alert_result.annotations["description"] == "Service is not responding"
    
    async def test_monitoring_service_lifecycle(self, tenant_id):
        """Test the complete monitoring service lifecycle."""
        # Create a new monitoring service
        service = await get_monitoring_service()
        
        try:
            # Test that service is running
            assert service._running
            
            # Record some data
            metric_request = MetricCreateRequest(
                name="lifecycle_test",
                metric_type=MetricType.GAUGE,
                value=42.0
            )
            
            result = await service.record_metric(tenant_id, metric_request)
            assert result.value == 42.0
            
        finally:
            # Stop the service
            await service.stop()
            assert not service._running


class TestMonitoringPerformance:
    """Test monitoring system performance."""
    
    async def test_bulk_metric_recording(self, monitoring_service, tenant_id):
        """Test recording multiple metrics efficiently."""
        start_time = datetime.utcnow()
        
        # Record 100 metrics
        tasks = []
        for i in range(100):
            request = MetricCreateRequest(
                name=f"bulk_test_{i}",
                metric_type=MetricType.COUNTER,
                value=float(i),
                labels={"batch": "test", "index": str(i)}
            )
            tasks.append(monitoring_service.record_metric(tenant_id, request))
        
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Verify all metrics were recorded
        assert len(results) == 100
        for i, result in enumerate(results):
            assert result.name == f"bulk_test_{i}"
            assert result.value == float(i)
        
        # Performance check: should complete within reasonable time
        assert duration < 10.0  # 10 seconds for 100 metrics
        
        print(f"Recorded 100 metrics in {duration:.2f} seconds")
    
    async def test_concurrent_operations(self, monitoring_service, tenant_id):
        """Test concurrent monitoring operations."""
        async def record_metric():
            request = MetricCreateRequest(
                name="concurrent_test",
                metric_type=MetricType.COUNTER,
                value=1.0
            )
            return await monitoring_service.record_metric(tenant_id, request)
        
        async def record_trace():
            request = TraceCreateRequest(
                trace_id=f"{uuid4().hex}",
                span_id=f"{uuid4().hex[:16]}",
                operation_name="concurrent_test",
                service_name="test-service",
                start_time=datetime.utcnow()
            )
            return await monitoring_service.record_trace(tenant_id, request)
        
        async def record_health_check():
            request = HealthCheckCreateRequest(
                check_name="concurrent_test",
                service_name="test-service",
                status=HealthStatus.HEALTHY
            )
            return await monitoring_service.record_health_check(tenant_id, request)
        
        # Run operations concurrently
        tasks = []
        for _ in range(10):
            tasks.extend([record_metric(), record_trace(), record_health_check()])
        
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed successfully
        assert len(results) == 30  # 10 * 3 operations
        assert all(result is not None for result in results)
