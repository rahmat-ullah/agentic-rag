#!/usr/bin/env python3
"""
Comprehensive Monitoring Stack Testing Script

This script validates all monitoring components and their integration
with the main Agentic RAG system.
"""

import asyncio
import json
import time
import requests
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

def check_service_health(url: str, service_name: str, timeout: int = 10) -> bool:
    """Check if a service is healthy."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print_success(f"{service_name} is healthy")
            return True
        else:
            print_error(f"{service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"{service_name} is not accessible: {e}")
        return False

def check_docker_services() -> Dict[str, bool]:
    """Check Docker services status."""
    print_header("DOCKER SERVICES STATUS CHECK")
    
    services = {
        "agentic-rag-api": "http://localhost:8000/health",
        "agentic-rag-prometheus": "http://localhost:9090/-/healthy",
        "agentic-rag-grafana": "http://localhost:3000/api/health",
        "agentic-rag-elasticsearch": "http://localhost:9200/_cluster/health",
        "agentic-rag-kibana": "http://localhost:5601/api/status",
        "agentic-rag-alertmanager": "http://localhost:9093/-/healthy",
        "agentic-rag-jaeger": "http://localhost:16686",
        "agentic-rag-postgres": None,  # No HTTP endpoint
        "agentic-rag-chromadb": "http://localhost:8001/api/v1/heartbeat",
        "agentic-rag-redis": None,  # No HTTP endpoint
        "agentic-rag-minio": "http://localhost:9000/minio/health/live"
    }
    
    results = {}
    
    for service_name, health_url in services.items():
        if health_url:
            results[service_name] = check_service_health(health_url, service_name)
        else:
            # Check if container is running
            try:
                result = subprocess.run(
                    ["docker", "ps", "--filter", f"name={service_name}", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if service_name in result.stdout:
                    print_success(f"{service_name} container is running")
                    results[service_name] = True
                else:
                    print_error(f"{service_name} container is not running")
                    results[service_name] = False
            except subprocess.CalledProcessError:
                print_error(f"Failed to check {service_name} container status")
                results[service_name] = False
    
    return results

def test_prometheus_metrics() -> bool:
    """Test Prometheus metrics collection."""
    print_header("PROMETHEUS METRICS TESTING")
    
    try:
        # Check Prometheus targets
        response = requests.get("http://localhost:9090/api/v1/targets")
        if response.status_code == 200:
            targets = response.json()
            active_targets = [t for t in targets['data']['activeTargets'] if t['health'] == 'up']
            print_success(f"Prometheus has {len(active_targets)} active targets")
            
            for target in active_targets:
                print_info(f"  - {target['labels']['job']}: {target['scrapeUrl']}")
        
        # Check for custom metrics
        custom_metrics = [
            "agentic_rag_requests_total",
            "agentic_rag_request_duration_seconds",
            "agentic_rag_active_connections",
            "agentic_rag_user_satisfaction_score"
        ]
        
        for metric in custom_metrics:
            response = requests.get(f"http://localhost:9090/api/v1/query?query={metric}")
            if response.status_code == 200:
                data = response.json()
                if data['data']['result']:
                    print_success(f"Metric {metric} is available")
                else:
                    print_warning(f"Metric {metric} has no data yet")
            else:
                print_error(f"Failed to query metric {metric}")
        
        return True
        
    except Exception as e:
        print_error(f"Prometheus testing failed: {e}")
        return False

def test_grafana_dashboards() -> bool:
    """Test Grafana dashboards."""
    print_header("GRAFANA DASHBOARDS TESTING")
    
    try:
        # Login to Grafana
        login_data = {
            "user": "admin",
            "password": "agentic-rag-admin"
        }
        
        session = requests.Session()
        response = session.post("http://localhost:3000/login", json=login_data)
        
        if response.status_code == 200:
            print_success("Successfully logged into Grafana")
        else:
            print_error("Failed to login to Grafana")
            return False
        
        # Check dashboards
        response = session.get("http://localhost:3000/api/search?type=dash-db")
        if response.status_code == 200:
            dashboards = response.json()
            print_success(f"Found {len(dashboards)} dashboards")
            
            for dashboard in dashboards:
                print_info(f"  - {dashboard['title']}: {dashboard['uid']}")
        
        # Check data sources
        response = session.get("http://localhost:3000/api/datasources")
        if response.status_code == 200:
            datasources = response.json()
            print_success(f"Found {len(datasources)} data sources")
            
            for ds in datasources:
                print_info(f"  - {ds['name']}: {ds['type']}")
        
        return True
        
    except Exception as e:
        print_error(f"Grafana testing failed: {e}")
        return False

def test_elasticsearch_logs() -> bool:
    """Test Elasticsearch log storage."""
    print_header("ELASTICSEARCH LOGS TESTING")
    
    try:
        # Check cluster health
        response = requests.get("http://localhost:9200/_cluster/health")
        if response.status_code == 200:
            health = response.json()
            print_success(f"Elasticsearch cluster status: {health['status']}")
            print_info(f"  - Nodes: {health['number_of_nodes']}")
            print_info(f"  - Data nodes: {health['number_of_data_nodes']}")
        
        # Check indices
        response = requests.get("http://localhost:9200/_cat/indices?format=json")
        if response.status_code == 200:
            indices = response.json()
            print_success(f"Found {len(indices)} indices")
            
            for index in indices:
                print_info(f"  - {index['index']}: {index['docs.count']} docs")
        
        # Test log search
        search_query = {
            "query": {
                "match_all": {}
            },
            "size": 1
        }
        
        response = requests.post(
            "http://localhost:9200/_search",
            json=search_query,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            results = response.json()
            print_success(f"Log search successful: {results['hits']['total']['value']} total logs")
        
        return True
        
    except Exception as e:
        print_error(f"Elasticsearch testing failed: {e}")
        return False

def test_kibana_visualization() -> bool:
    """Test Kibana log visualization."""
    print_header("KIBANA VISUALIZATION TESTING")
    
    try:
        # Check Kibana status
        response = requests.get("http://localhost:5601/api/status")
        if response.status_code == 200:
            status = response.json()
            print_success(f"Kibana status: {status['status']['overall']['level']}")
        
        # Check saved objects (dashboards, visualizations)
        response = requests.get("http://localhost:5601/api/saved_objects/_find?type=dashboard")
        if response.status_code == 200:
            dashboards = response.json()
            print_success(f"Found {dashboards['total']} Kibana dashboards")
        
        return True
        
    except Exception as e:
        print_error(f"Kibana testing failed: {e}")
        return False

def test_jaeger_tracing() -> bool:
    """Test Jaeger distributed tracing."""
    print_header("JAEGER TRACING TESTING")
    
    try:
        # Check Jaeger UI
        response = requests.get("http://localhost:16686")
        if response.status_code == 200:
            print_success("Jaeger UI is accessible")
        
        # Check for services
        response = requests.get("http://localhost:16686/api/services")
        if response.status_code == 200:
            services = response.json()
            print_success(f"Found {len(services['data'])} traced services")
            
            for service in services['data']:
                print_info(f"  - {service}")
        
        return True
        
    except Exception as e:
        print_error(f"Jaeger testing failed: {e}")
        return False

def test_alertmanager() -> bool:
    """Test Alertmanager configuration."""
    print_header("ALERTMANAGER TESTING")
    
    try:
        # Check Alertmanager status
        response = requests.get("http://localhost:9093/api/v1/status")
        if response.status_code == 200:
            status = response.json()
            print_success("Alertmanager is running")
            print_info(f"  - Version: {status['data']['versionInfo']['version']}")
        
        # Check configuration
        response = requests.get("http://localhost:9093/api/v1/status/config")
        if response.status_code == 200:
            print_success("Alertmanager configuration is valid")
        
        # Check alerts
        response = requests.get("http://localhost:9093/api/v1/alerts")
        if response.status_code == 200:
            alerts = response.json()
            print_success(f"Found {len(alerts['data'])} active alerts")
        
        return True
        
    except Exception as e:
        print_error(f"Alertmanager testing failed: {e}")
        return False

def test_api_monitoring_endpoints() -> bool:
    """Test API monitoring endpoints."""
    print_header("API MONITORING ENDPOINTS TESTING")
    
    try:
        # Test metrics endpoint
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            print_success("Prometheus metrics endpoint is working")
            metrics_count = len([line for line in response.text.split('\n') if line and not line.startswith('#')])
            print_info(f"  - Exported {metrics_count} metric lines")
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            health = response.json()
            print_success(f"Health endpoint status: {health.get('status', 'unknown')}")
        
        # Test monitoring API endpoints
        monitoring_endpoints = [
            "/api/v1/monitoring/metrics",
            "/api/v1/monitoring/traces",
            "/api/v1/monitoring/logs",
            "/api/v1/monitoring/alerts",
            "/api/v1/monitoring/health"
        ]
        
        for endpoint in monitoring_endpoints:
            response = requests.get(f"http://localhost:8000{endpoint}")
            if response.status_code in [200, 401]:  # 401 is OK (auth required)
                print_success(f"Monitoring endpoint {endpoint} is available")
            else:
                print_warning(f"Monitoring endpoint {endpoint} returned {response.status_code}")
        
        return True
        
    except Exception as e:
        print_error(f"API monitoring testing failed: {e}")
        return False

def generate_test_data() -> bool:
    """Generate test data for monitoring."""
    print_header("GENERATING TEST DATA")
    
    try:
        # Make some API requests to generate metrics
        test_requests = [
            "http://localhost:8000/health",
            "http://localhost:8000/api/v1/search?q=test",
            "http://localhost:8000/api/v1/documents",
            "http://localhost:8000/metrics"
        ]
        
        for _ in range(10):
            for url in test_requests:
                try:
                    requests.get(url, timeout=5)
                except:
                    pass  # Ignore errors, we just want to generate traffic
            time.sleep(1)
        
        print_success("Generated test traffic for monitoring")
        return True
        
    except Exception as e:
        print_error(f"Test data generation failed: {e}")
        return False

def run_comprehensive_test() -> Dict[str, bool]:
    """Run comprehensive monitoring stack test."""
    print_header("COMPREHENSIVE MONITORING STACK TEST")
    
    test_results = {}
    
    # Test each component
    test_functions = [
        ("Docker Services", check_docker_services),
        ("Prometheus Metrics", test_prometheus_metrics),
        ("Grafana Dashboards", test_grafana_dashboards),
        ("Elasticsearch Logs", test_elasticsearch_logs),
        ("Kibana Visualization", test_kibana_visualization),
        ("Jaeger Tracing", test_jaeger_tracing),
        ("Alertmanager", test_alertmanager),
        ("API Monitoring", test_api_monitoring_endpoints),
        ("Test Data Generation", generate_test_data)
    ]
    
    for test_name, test_func in test_functions:
        try:
            if test_name == "Docker Services":
                result = test_func()
                test_results[test_name] = all(result.values())
            else:
                test_results[test_name] = test_func()
        except Exception as e:
            print_error(f"{test_name} test failed with exception: {e}")
            test_results[test_name] = False
    
    return test_results

def print_test_summary(results: Dict[str, bool]):
    """Print test results summary."""
    print_header("TEST RESULTS SUMMARY")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n{Colors.BOLD}Overall Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print_success("🎉 All monitoring components are working correctly!")
        return True
    else:
        print_error(f"❌ {total - passed} tests failed. Please check the logs above.")
        return False

def main():
    """Main test function."""
    print_header("AGENTIC RAG MONITORING STACK VALIDATION")
    print_info("This script will test all monitoring components")
    print_info("Make sure all Docker services are running before proceeding")
    
    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
    
    # Run comprehensive test
    results = run_comprehensive_test()
    
    # Print summary
    success = print_test_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
