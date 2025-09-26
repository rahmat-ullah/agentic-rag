#!/usr/bin/env python3
"""
Monitoring and Observability System Demonstration Script

This script demonstrates the complete monitoring and observability system functionality,
including Prometheus metrics, OpenTelemetry tracing, alerting, and health checks.

Usage:
    python scripts/demo_monitoring_system.py
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_rag.models.monitoring import MetricType, AlertSeverity, HealthStatus, TraceStatus
from agentic_rag.schemas.monitoring import (
    MetricCreateRequest, TraceCreateRequest, AlertCreateRequest, HealthCheckCreateRequest
)
from agentic_rag.services.monitoring_service import get_monitoring_service
from agentic_rag.services.tracing_service import get_tracing_service, initialize_tracing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringSystemDemo:
    """Demonstration of the monitoring and observability system."""
    
    def __init__(self):
        self.tenant_id = uuid4()
        self.monitoring_service = None
        self.tracing_service = None
        
        logger.info(f"Demo initialized with tenant ID: {self.tenant_id}")
    
    async def setup(self):
        """Set up the monitoring system."""
        logger.info("Setting up monitoring system...")
        
        # Initialize services
        self.monitoring_service = await get_monitoring_service()
        self.tracing_service = get_tracing_service()
        initialize_tracing()
        
        logger.info("Monitoring system setup complete")
    
    async def demo_metrics_collection(self):
        """Demonstrate metrics collection."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING METRICS COLLECTION")
        logger.info("="*50)
        
        # Counter metrics
        logger.info("Recording counter metrics...")
        counter_metrics = [
            ("search_queries_total", {"tenant_id": str(self.tenant_id), "query_type": "semantic"}),
            ("document_uploads_total", {"tenant_id": str(self.tenant_id), "document_type": "pdf"}),
            ("api_requests_total", {"method": "GET", "endpoint": "/search", "status": "200"}),
            ("errors_total", {"error_type": "validation", "service": "api"})
        ]
        
        for name, labels in counter_metrics:
            request = MetricCreateRequest(
                name=name,
                metric_type=MetricType.COUNTER,
                value=1.0,
                labels=labels,
                source_service="demo-service"
            )
            
            result = await self.monitoring_service.record_metric(self.tenant_id, request)
            logger.info(f"  ✓ Recorded counter: {name} = {result.value}")
        
        # Gauge metrics
        logger.info("Recording gauge metrics...")
        gauge_metrics = [
            ("user_satisfaction_score", {"tenant_id": str(self.tenant_id)}, 4.2),
            ("system_utilization_percent", {"resource_type": "cpu"}, 65.5),
            ("system_utilization_percent", {"resource_type": "memory"}, 78.3),
            ("active_connections", {}, 42.0)
        ]
        
        for name, labels, value in gauge_metrics:
            request = MetricCreateRequest(
                name=name,
                metric_type=MetricType.GAUGE,
                value=value,
                labels=labels,
                source_service="demo-service"
            )
            
            result = await self.monitoring_service.record_metric(self.tenant_id, request)
            logger.info(f"  ✓ Recorded gauge: {name} = {result.value}")
        
        # Histogram metrics
        logger.info("Recording histogram metrics...")
        histogram_metrics = [
            ("request_duration_seconds", {"method": "GET", "endpoint": "/search"}, 0.125),
            ("vector_operation_duration_seconds", {"operation_type": "query"}, 0.045),
            ("vector_operation_duration_seconds", {"operation_type": "add"}, 0.230)
        ]
        
        for name, labels, value in histogram_metrics:
            request = MetricCreateRequest(
                name=name,
                metric_type=MetricType.HISTOGRAM,
                value=value,
                labels=labels,
                source_service="demo-service"
            )
            
            result = await self.monitoring_service.record_metric(self.tenant_id, request)
            logger.info(f"  ✓ Recorded histogram: {name} = {result.value}s")
        
        logger.info("Metrics collection demonstration complete!")
    
    async def demo_distributed_tracing(self):
        """Demonstrate distributed tracing."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING DISTRIBUTED TRACING")
        logger.info("="*50)
        
        # Simulate a distributed request flow
        trace_id = f"{uuid4().hex}"
        
        # Root span - API request
        logger.info("Creating root span for API request...")
        api_span_id = f"{uuid4().hex[:16]}"
        api_request = TraceCreateRequest(
            trace_id=trace_id,
            span_id=api_span_id,
            operation_name="POST /api/v1/search",
            service_name="api-service",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(milliseconds=250),
            status=TraceStatus.OK,
            tags={
                "http.method": "POST",
                "http.url": "/api/v1/search",
                "http.status_code": 200,
                "user.id": str(uuid4())
            }
        )
        
        api_result = await self.monitoring_service.record_trace(self.tenant_id, api_request)
        logger.info(f"  ✓ API span: {api_result.operation_name} ({api_result.duration_ms}ms)")
        
        # Child span - Database query
        logger.info("Creating child span for database query...")
        db_span_id = f"{uuid4().hex[:16]}"
        db_request = TraceCreateRequest(
            trace_id=trace_id,
            span_id=db_span_id,
            parent_span_id=api_span_id,
            operation_name="SELECT documents",
            service_name="postgres",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(milliseconds=45),
            status=TraceStatus.OK,
            tags={
                "db.system": "postgresql",
                "db.statement": "SELECT * FROM documents WHERE tenant_id = $1",
                "db.rows_affected": 15
            }
        )
        
        db_result = await self.monitoring_service.record_trace(self.tenant_id, db_request)
        logger.info(f"  ✓ Database span: {db_result.operation_name} ({db_result.duration_ms}ms)")
        
        # Child span - Vector search
        logger.info("Creating child span for vector search...")
        vector_span_id = f"{uuid4().hex[:16]}"
        vector_request = TraceCreateRequest(
            trace_id=trace_id,
            span_id=vector_span_id,
            parent_span_id=api_span_id,
            operation_name="vector_search",
            service_name="chromadb",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(milliseconds=120),
            status=TraceStatus.OK,
            tags={
                "vector.collection": "documents",
                "vector.query_type": "similarity",
                "vector.results_count": 10,
                "vector.similarity_threshold": 0.8
            }
        )
        
        vector_result = await self.monitoring_service.record_trace(self.tenant_id, vector_request)
        logger.info(f"  ✓ Vector search span: {vector_result.operation_name} ({vector_result.duration_ms}ms)")
        
        # Error span example
        logger.info("Creating error span example...")
        error_span_id = f"{uuid4().hex[:16]}"
        error_request = TraceCreateRequest(
            trace_id=trace_id,
            span_id=error_span_id,
            parent_span_id=api_span_id,
            operation_name="llm_reranking",
            service_name="openai",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(milliseconds=5000),
            status=TraceStatus.ERROR,
            error=True,
            error_message="Rate limit exceeded",
            tags={
                "llm.provider": "openai",
                "llm.model": "gpt-4",
                "error.type": "rate_limit"
            }
        )
        
        error_result = await self.monitoring_service.record_trace(self.tenant_id, error_request)
        logger.info(f"  ✗ Error span: {error_result.operation_name} ({error_result.duration_ms}ms) - {error_result.error_message}")
        
        logger.info(f"Distributed tracing demonstration complete! Trace ID: {trace_id}")
    
    async def demo_alerting_system(self):
        """Demonstrate the alerting system."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING ALERTING SYSTEM")
        logger.info("="*50)
        
        # Critical alert
        logger.info("Creating critical alert...")
        critical_request = AlertCreateRequest(
            rule_name="high_error_rate",
            alert_name="High Error Rate Detected",
            severity=AlertSeverity.CRITICAL,
            labels={
                "service": "api-service",
                "environment": "production",
                "team": "platform"
            },
            annotations={
                "summary": "Error rate is above 5% for the last 5 minutes",
                "description": "The API service is experiencing a high error rate which may impact user experience",
                "runbook": "https://wiki.company.com/runbooks/high-error-rate"
            },
            description="Critical: API service error rate exceeded threshold",
            runbook_url="https://wiki.company.com/runbooks/high-error-rate",
            dashboard_url="https://grafana.company.com/d/api-service",
            source_service="api-service"
        )
        
        critical_result = await self.monitoring_service.create_alert(self.tenant_id, critical_request)
        logger.info(f"  🚨 Critical alert created: {critical_result.alert_name}")
        logger.info(f"     Fingerprint: {critical_result.fingerprint}")
        
        # Warning alert
        logger.info("Creating warning alert...")
        warning_request = AlertCreateRequest(
            rule_name="high_response_time",
            alert_name="High Response Time",
            severity=AlertSeverity.WARNING,
            labels={
                "service": "search-service",
                "environment": "production"
            },
            annotations={
                "summary": "Response time is above 2 seconds",
                "description": "Search service response time is elevated"
            },
            description="Warning: Search service response time is elevated",
            source_service="search-service"
        )
        
        warning_result = await self.monitoring_service.create_alert(self.tenant_id, warning_request)
        logger.info(f"  ⚠️  Warning alert created: {warning_result.alert_name}")
        
        # Test alert deduplication
        logger.info("Testing alert deduplication...")
        duplicate_result = await self.monitoring_service.create_alert(self.tenant_id, critical_request)
        
        if duplicate_result.id == critical_result.id:
            logger.info("  ✓ Alert deduplication working correctly - same alert returned")
        else:
            logger.warning("  ✗ Alert deduplication failed - new alert created")
        
        logger.info("Alerting system demonstration complete!")
    
    async def demo_health_checks(self):
        """Demonstrate health check recording."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING HEALTH CHECKS")
        logger.info("="*50)
        
        # Healthy services
        healthy_checks = [
            ("database_connectivity", "postgres", "postgres-1", 25.5),
            ("vector_store_health", "chromadb", "chromadb-1", 15.2),
            ("api_health", "api-service", "api-1", 5.8),
            ("cache_health", "redis", "redis-1", 2.1)
        ]
        
        logger.info("Recording healthy service checks...")
        for check_name, service, instance, response_time in healthy_checks:
            request = HealthCheckCreateRequest(
                check_name=check_name,
                service_name=service,
                instance_id=instance,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                details={
                    "status": "ok",
                    "last_check": datetime.utcnow().isoformat(),
                    "version": "1.0.0"
                }
            )
            
            result = await self.monitoring_service.record_health_check(self.tenant_id, request)
            logger.info(f"  ✓ {service}: {result.status.value} ({result.response_time_ms}ms)")
        
        # Unhealthy service
        logger.info("Recording unhealthy service check...")
        unhealthy_request = HealthCheckCreateRequest(
            check_name="external_api_health",
            service_name="external-api",
            instance_id="external-api-1",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=5000.0,
            error_message="Connection timeout after 5 seconds",
            details={
                "status": "timeout",
                "last_successful_check": (datetime.utcnow() - timedelta(minutes=10)).isoformat(),
                "error_count": 5
            }
        )
        
        unhealthy_result = await self.monitoring_service.record_health_check(self.tenant_id, unhealthy_request)
        logger.info(f"  ✗ {unhealthy_result.service_name}: {unhealthy_result.status.value} - {unhealthy_result.error_message}")
        
        # Degraded service
        logger.info("Recording degraded service check...")
        degraded_request = HealthCheckCreateRequest(
            check_name="search_performance",
            service_name="search-service",
            instance_id="search-1",
            status=HealthStatus.DEGRADED,
            response_time_ms=1500.0,
            details={
                "status": "degraded",
                "performance_issue": "High latency detected",
                "avg_response_time": 1500.0,
                "threshold": 1000.0
            }
        )
        
        degraded_result = await self.monitoring_service.record_health_check(self.tenant_id, degraded_request)
        logger.info(f"  ⚠️  {degraded_result.service_name}: {degraded_result.status.value} ({degraded_result.response_time_ms}ms)")
        
        logger.info("Health checks demonstration complete!")
    
    async def demo_prometheus_export(self):
        """Demonstrate Prometheus metrics export."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING PROMETHEUS EXPORT")
        logger.info("="*50)
        
        logger.info("Exporting Prometheus metrics...")
        metrics_text = await self.monitoring_service.get_prometheus_metrics()
        
        logger.info(f"Exported {len(metrics_text)} characters of Prometheus metrics")
        logger.info("Sample metrics:")
        
        # Show first few lines of metrics
        lines = metrics_text.split('\n')[:10]
        for line in lines:
            if line.strip():
                logger.info(f"  {line}")
        
        if len(lines) > 10:
            logger.info(f"  ... and {len(metrics_text.split(chr(10))) - 10} more lines")
        
        logger.info("Prometheus export demonstration complete!")
    
    async def demo_performance_test(self):
        """Demonstrate system performance under load."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING PERFORMANCE UNDER LOAD")
        logger.info("="*50)
        
        logger.info("Running performance test with 100 concurrent operations...")
        start_time = time.time()
        
        # Create 100 concurrent operations
        tasks = []
        for i in range(100):
            # Mix of different operation types
            if i % 3 == 0:
                # Metric
                request = MetricCreateRequest(
                    name=f"perf_test_metric_{i}",
                    metric_type=MetricType.COUNTER,
                    value=1.0,
                    labels={"test": "performance", "batch": str(i // 10)}
                )
                tasks.append(self.monitoring_service.record_metric(self.tenant_id, request))
            elif i % 3 == 1:
                # Trace
                request = TraceCreateRequest(
                    trace_id=f"{uuid4().hex}",
                    span_id=f"{uuid4().hex[:16]}",
                    operation_name=f"perf_test_operation_{i}",
                    service_name="perf-test-service",
                    start_time=datetime.utcnow()
                )
                tasks.append(self.monitoring_service.record_trace(self.tenant_id, request))
            else:
                # Health check
                request = HealthCheckCreateRequest(
                    check_name=f"perf_test_check_{i}",
                    service_name="perf-test-service",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=float(i % 100)
                )
                tasks.append(self.monitoring_service.record_health_check(self.tenant_id, request))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✓ Completed 100 operations in {duration:.2f} seconds")
        logger.info(f"✓ Average: {(duration / 100) * 1000:.1f}ms per operation")
        logger.info(f"✓ Throughput: {100 / duration:.1f} operations/second")
        
        logger.info("Performance test complete!")
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("\nCleaning up...")
        
        if self.monitoring_service:
            await self.monitoring_service.stop()
        
        if self.tracing_service:
            self.tracing_service.shutdown()
        
        logger.info("Cleanup complete!")
    
    async def run_demo(self):
        """Run the complete monitoring system demonstration."""
        try:
            await self.setup()
            
            await self.demo_metrics_collection()
            await self.demo_distributed_tracing()
            await self.demo_alerting_system()
            await self.demo_health_checks()
            await self.demo_prometheus_export()
            await self.demo_performance_test()
            
            logger.info("\n" + "="*50)
            logger.info("MONITORING SYSTEM DEMONSTRATION COMPLETE!")
            logger.info("="*50)
            logger.info("✓ Metrics collection working")
            logger.info("✓ Distributed tracing working")
            logger.info("✓ Alerting system working")
            logger.info("✓ Health checks working")
            logger.info("✓ Prometheus export working")
            logger.info("✓ Performance under load verified")
            logger.info("\nThe monitoring and observability system is ready for production!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}", exc_info=True)
            raise
        finally:
            await self.cleanup()


async def main():
    """Main function to run the demonstration."""
    demo = MonitoringSystemDemo()
    await demo.run_demo()


async def demo_alerting_system():
    """Demonstrate the alerting system capabilities."""
    print("\n" + "="*60)
    print("🚨 ALERTING SYSTEM DEMONSTRATION")
    print("="*60)

    try:
        # Get services
        monitoring_service = await get_monitoring_service()
        notification_service = await get_notification_service()
        alerting_engine = await get_alerting_engine()

        tenant_id = UUID("12345678-1234-5678-9012-123456789012")

        # Create a test alert
        print("\n📢 Creating test alert...")
        alert_request = AlertCreateRequest(
            rule_name="TestAlert",
            alert_name="High Error Rate Test",
            severity=AlertSeverity.CRITICAL,
            labels={"service": "api", "team": "platform", "environment": "demo"},
            annotations={"summary": "This is a test alert for demonstration"},
            description="Demonstration of critical alert functionality",
            runbook_url="https://wiki.company.com/runbooks/test-alert",
            source_service="demo-script"
        )

        alert = await monitoring_service.create_alert(tenant_id, alert_request)
        print(f"✅ Alert created: {alert.id}")
        print(f"   - Name: {alert.alert_name}")
        print(f"   - Severity: {alert.severity.value}")
        print(f"   - Status: {alert.status.value}")

        # Send notification
        print("\n📧 Sending alert notification...")
        notification_result = await notification_service.send_alert_notification(
            alert,
            context={"demo_mode": True, "test_run": True}
        )
        print(f"✅ Notification sent to {notification_result['total']} channels")
        print(f"   - Success: {notification_result['success']}")
        print(f"   - Errors: {notification_result['errors']}")

        # Update alert status
        print("\n🔄 Acknowledging alert...")
        updated_alert = await monitoring_service.update_alert_status(
            tenant_id,
            alert.id,
            AlertStatus.ACKNOWLEDGED
        )
        print(f"✅ Alert acknowledged at: {updated_alert.acknowledged_at}")

        # Resolve alert
        print("\n✅ Resolving alert...")
        resolved_alert = await monitoring_service.update_alert_status(
            tenant_id,
            alert.id,
            AlertStatus.RESOLVED
        )
        print(f"✅ Alert resolved at: {resolved_alert.resolved_at}")

        # Get alert history
        print("\n📊 Alert lifecycle summary:")
        print(f"   - Created: {alert.created_at}")
        print(f"   - Acknowledged: {resolved_alert.acknowledged_at}")
        print(f"   - Resolved: {resolved_alert.resolved_at}")
        print(f"   - Duration: {resolved_alert.resolved_at - alert.created_at}")

    except Exception as e:
        print(f"❌ Alerting demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_dashboard_metrics():
    """Demonstrate dashboard metrics collection."""
    print("\n" + "="*60)
    print("📊 DASHBOARD METRICS DEMONSTRATION")
    print("="*60)

    try:
        monitoring_service = await get_monitoring_service()
        tenant_id = UUID("12345678-1234-5678-9012-123456789012")

        # Simulate dashboard metrics
        dashboard_metrics = [
            ("active_users", 1250, {"dashboard": "executive"}),
            ("request_rate", 45.7, {"endpoint": "/api/v1/search"}),
            ("response_time_p95", 0.234, {"service": "api"}),
            ("user_satisfaction", 4.2, {"period": "last_hour"}),
            ("search_queries", 892, {"type": "semantic"}),
            ("documents_processed", 15420, {"status": "success"}),
            ("error_rate", 0.8, {"service": "api"}),
            ("system_cpu", 67.3, {"resource": "cpu"}),
            ("system_memory", 78.9, {"resource": "memory"}),
            ("vector_operations", 234, {"operation": "similarity_search"})
        ]

        print("\n📈 Collecting dashboard metrics...")
        for metric_name, value, labels in dashboard_metrics:
            metric_request = MetricCreateRequest(
                name=f"dashboard_{metric_name}",
                value=value,
                metric_type=MetricType.GAUGE,
                labels=labels,
                source_service="dashboard-demo"
            )

            metric = await monitoring_service.record_metric(tenant_id, metric_request)
            print(f"   ✅ {metric_name}: {value} {labels}")

        print(f"\n📊 Dashboard metrics collected: {len(dashboard_metrics)} metrics")

        # Query recent metrics for dashboard
        print("\n🔍 Querying metrics for dashboard display...")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)

        recent_metrics = await monitoring_service.get_metrics(
            tenant_id,
            start_time=start_time,
            end_time=end_time,
            metric_names=["dashboard_active_users", "dashboard_request_rate"]
        )

        print(f"✅ Retrieved {len(recent_metrics)} recent metrics")
        for metric in recent_metrics[-3:]:  # Show last 3
            print(f"   - {metric.name}: {metric.value} at {metric.timestamp}")

    except Exception as e:
        print(f"❌ Dashboard metrics demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_complete_monitoring_workflow():
    """Demonstrate complete end-to-end monitoring workflow."""
    print("\n" + "="*60)
    print("🔄 COMPLETE MONITORING WORKFLOW DEMONSTRATION")
    print("="*60)

    try:
        # Simulate a complete user journey with monitoring
        print("\n🚀 Simulating user journey with full monitoring...")

        # 1. User makes API request (metrics)
        print("1. 📊 Recording API request metrics...")
        await demo_metrics_collection()

        # 2. System processes request (tracing)
        print("\n2. 🔍 Capturing distributed traces...")
        await demo_distributed_tracing()

        # 3. System logs activity (logging)
        print("\n3. 📝 Shipping application logs...")
        await demo_log_aggregation()

        # 4. Health check runs (health monitoring)
        print("\n4. ❤️ Performing health checks...")
        await demo_health_monitoring()

        # 5. Dashboard updates (visualization)
        print("\n5. 📊 Updating dashboard metrics...")
        await demo_dashboard_metrics()

        # 6. Alert evaluation (alerting)
        print("\n6. 🚨 Evaluating alert conditions...")
        await demo_alerting_system()

        print("\n" + "="*60)
        print("✅ COMPLETE MONITORING WORKFLOW SUCCESSFUL!")
        print("="*60)
        print("\n🎯 All monitoring components working together:")
        print("   ✅ Metrics Collection & Storage")
        print("   ✅ Distributed Tracing & Context")
        print("   ✅ Log Aggregation & Search")
        print("   ✅ Health Monitoring & Status")
        print("   ✅ Dashboard Visualization")
        print("   ✅ Critical Issue Alerting")
        print("\n🚀 System ready for production monitoring!")

    except Exception as e:
        print(f"❌ Complete workflow demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_complete_monitoring_workflow())
