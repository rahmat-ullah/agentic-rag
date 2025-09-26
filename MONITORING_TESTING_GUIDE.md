# 📊 Sprint 7 Monitoring & Observability - Local Testing Guide

This comprehensive guide provides step-by-step instructions for setting up and testing the complete monitoring and observability system implemented in Sprint 7, Story 7-01.

## 🎯 **Overview**

The monitoring stack includes:

- **Prometheus** - Metrics collection and storage
- **Grafana** - Dashboards and visualization
- **ELK Stack** - Log aggregation and analysis (Elasticsearch, Logstash, Kibana)
- **Jaeger** - Distributed tracing
- **Alertmanager** - Alert routing and notifications
- **Node Exporter & cAdvisor** - System and container metrics

## 📋 **Prerequisites**

### System Requirements

- **OS**: Windows 10/11 with Docker Desktop
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk**: 10GB free space for Docker volumes
- **CPU**: 4+ cores recommended

### Software Requirements

- **Docker Desktop** 4.20+ with Docker Compose V2
- **Python** 3.10+ (for testing scripts)
- **Git** (for repository access)
- **Web Browser** (Chrome/Firefox recommended)

### Required Ports

Ensure these ports are available:

```
3000  - Grafana Dashboard
5601  - Kibana Dashboard
8000  - Agentic RAG API
8080  - cAdvisor Metrics
9000  - MinIO Console
9090  - Prometheus UI
9093  - Alertmanager UI
9100  - Node Exporter
9200  - Elasticsearch HTTP
16686 - Jaeger UI
```

## 🚀 **Quick Start Setup**

### 1. Environment Preparation

```bash
# Navigate to project directory
cd f:\Projects\agentic-contextual-rag

# Copy monitoring environment configuration
copy .env.monitoring .env.local

# Create data directories
mkdir -p data\monitoring\prometheus
mkdir -p data\monitoring\grafana
mkdir -p data\monitoring\elasticsearch
mkdir -p data\monitoring\alertmanager
mkdir -p data\monitoring\filebeat

# Set proper permissions (if needed)
# On Windows, ensure Docker Desktop has access to the project directory
```

### 2. Start Core Services

```bash
# Start the main application stack
docker-compose up -d

# Wait for services to be healthy (2-3 minutes)
docker-compose ps

# Verify core services are running
docker-compose logs api
```

### 3. Start Monitoring Stack

```bash
# Start the complete monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Check monitoring services status
docker-compose -f docker-compose.monitoring.yml ps

# View logs for troubleshooting
docker-compose -f docker-compose.monitoring.yml logs
```

### 4. Verify Installation

```bash
# Run the comprehensive test script
python scripts/test_monitoring_stack.py

# Or manually check key services
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health # Grafana
curl http://localhost:9200/_cluster/health # Elasticsearch
```

## 🔍 **Component Testing**

### Prometheus Metrics Testing

1. **Access Prometheus UI**

   ```
   URL: http://localhost:9090
   ```

2. **Verify Targets**

   - Go to Status → Targets
   - Ensure all targets are "UP"
   - Check scrape intervals and last scrape times

3. **Test Custom Metrics**

   ```promql
   # API request rate
   rate(agentic_rag_requests_total[5m])

   # Response time percentiles
   histogram_quantile(0.95, rate(agentic_rag_request_duration_seconds_bucket[5m]))

   # Active connections
   agentic_rag_active_connections

   # User satisfaction
   agentic_rag_user_satisfaction_score
   ```

4. **Verify Alert Rules**
   - Go to Status → Rules
   - Check that alert rules are loaded
   - Verify rule evaluation times

### Grafana Dashboard Testing

1. **Access Grafana**

   ```
   URL: http://localhost:3000
   Username: admin
   Password: agentic-rag-admin
   ```

2. **Verify Data Sources**

   - Go to Configuration → Data Sources
   - Check Prometheus connection (should be green)
   - Test Elasticsearch connection

3. **Test Dashboards**

   - **Executive Dashboard**: Business metrics overview
   - **API Service Dashboard**: Technical performance metrics
   - Verify all panels load data
   - Check time range selectors work

4. **Create Test Dashboard**
   ```json
   {
     "dashboard": {
       "title": "Test Dashboard",
       "panels": [
         {
           "title": "API Requests",
           "type": "graph",
           "targets": [
             {
               "expr": "rate(agentic_rag_requests_total[5m])"
             }
           ]
         }
       ]
     }
   }
   ```

### ELK Stack Testing

1. **Elasticsearch Health Check**

   ```bash
   # Cluster health
   curl http://localhost:9200/_cluster/health

   # List indices
   curl http://localhost:9200/_cat/indices?v

   # Search logs
   curl -X POST "http://localhost:9200/_search" -H "Content-Type: application/json" -d '
   {
     "query": {
       "match_all": {}
     },
     "size": 10
   }'
   ```

2. **Kibana Dashboard Access**

   ```
   URL: http://localhost:5601
   ```

3. **Create Index Patterns**

   - Go to Stack Management → Index Patterns
   - Create pattern for `agentic-rag-*`
   - Set `@timestamp` as time field

4. **Test Log Search**
   - Go to Discover
   - Search for application logs
   - Filter by log level, service, etc.

### Jaeger Tracing Testing

1. **Access Jaeger UI**

   ```
   URL: http://localhost:16686
   ```

2. **Verify Services**

   - Check that `agentic-rag-api` appears in services list
   - Look for traces from recent API calls

3. **Trace Analysis**
   - Search for traces by service
   - Examine trace details and spans
   - Check for error traces

### Alertmanager Testing

1. **Access Alertmanager UI**

   ```
   URL: http://localhost:9093
   ```

2. **Verify Configuration**

   - Check Status page for config validity
   - Verify routing rules are loaded

3. **Test Alert Generation**

   ```bash
   # Generate high load to trigger alerts
   for i in {1..100}; do
     curl http://localhost:8000/api/v1/search?q=test &
   done
   ```

4. **Check Alert Notifications**
   - Monitor Alertmanager UI for firing alerts
   - Check configured notification channels

## 🧪 **End-to-End Testing Scenarios**

### Scenario 1: Complete User Journey Monitoring

1. **Generate User Activity**

   ```bash
   # Upload a document
   curl -X POST "http://localhost:8000/api/v1/upload" \
     -F "file=@test-document.pdf"

   # Perform searches
   curl "http://localhost:8000/api/v1/search?q=procurement"

   # Check document status
   curl "http://localhost:8000/api/v1/documents"
   ```

2. **Verify Monitoring Data**
   - **Metrics**: Check request counts, response times
   - **Logs**: Verify log entries in Kibana
   - **Traces**: Examine request traces in Jaeger
   - **Dashboards**: See real-time updates in Grafana

### Scenario 2: Error Handling and Alerting

1. **Generate Errors**

   ```bash
   # Invalid API calls
   curl "http://localhost:8000/api/v1/invalid-endpoint"
   curl -X POST "http://localhost:8000/api/v1/search" -d "invalid-json"
   ```

2. **Monitor Error Tracking**
   - Check error rate metrics in Prometheus
   - Verify error logs in Kibana
   - Look for error traces in Jaeger
   - Confirm alerts fire in Alertmanager

### Scenario 3: Performance Load Testing

1. **Generate Load**

   ```bash
   # Run the demo script with load testing
   python scripts/demo_monitoring_system.py
   ```

2. **Monitor Performance**
   - Watch response time percentiles
   - Check resource utilization
   - Monitor queue depths and connection pools
   - Verify auto-scaling triggers (if configured)

## 🔧 **Troubleshooting Guide**

### Common Issues and Solutions

#### Port Conflicts

```bash
# Check port usage
netstat -an | findstr :9090

# Stop conflicting services
# Update port mappings in docker-compose files
```

#### Memory Issues

```bash
# Check Docker memory usage
docker stats

# Increase Docker Desktop memory allocation
# Reduce service memory limits in compose files
```

#### Volume Mount Issues

```bash
# Check volume permissions
docker volume ls
docker volume inspect agentic-rag-prometheus-data

# Recreate volumes if needed
docker-compose down -v
docker-compose up -d
```

#### Service Connectivity

```bash
# Check network connectivity
docker network ls
docker network inspect agentic-rag-network

# Test inter-service communication
docker exec agentic-rag-api curl http://prometheus:9090/-/healthy
```

### Health Check Commands

```bash
# Quick health check for all services
docker-compose ps
docker-compose -f docker-compose.monitoring.yml ps

# Individual service health
curl http://localhost:8000/health      # API
curl http://localhost:9090/-/healthy   # Prometheus
curl http://localhost:3000/api/health  # Grafana
curl http://localhost:9200/_cluster/health # Elasticsearch
curl http://localhost:5601/api/status  # Kibana
curl http://localhost:9093/-/healthy   # Alertmanager

# Container logs
docker-compose logs api
docker-compose logs prometheus
docker-compose logs grafana
```

### Performance Optimization

1. **Resource Allocation**

   ```yaml
   # Adjust in docker-compose files
   deploy:
     resources:
       limits:
         memory: 2G
         cpus: "1.0"
   ```

2. **Data Retention**

   ```yaml
   # Prometheus retention
   command:
     - "--storage.tsdb.retention.time=7d"

   # Elasticsearch retention
   environment:
     - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
   ```

## 📊 **Validation Checklist**

### ✅ **Core Functionality**

- [ ] All Docker services start successfully
- [ ] API endpoints respond correctly
- [ ] Database connections are healthy
- [ ] File uploads work properly

### ✅ **Metrics Collection**

- [ ] Prometheus scrapes all targets
- [ ] Custom metrics are exported
- [ ] Metric retention works correctly
- [ ] Alert rules evaluate properly

### ✅ **Visualization**

- [ ] Grafana dashboards load data
- [ ] All panels display correctly
- [ ] Time range selection works
- [ ] Data source connections are stable

### ✅ **Log Aggregation**

- [ ] Logs are shipped to Elasticsearch
- [ ] Kibana can search and filter logs
- [ ] Log parsing works correctly
- [ ] Log retention policies are active

### ✅ **Distributed Tracing**

- [ ] Traces are collected in Jaeger
- [ ] Service dependencies are visible
- [ ] Trace sampling works correctly
- [ ] Error traces are captured

### ✅ **Alerting**

- [ ] Alert rules fire correctly
- [ ] Notifications are sent
- [ ] Alert routing works
- [ ] Alert resolution is tracked

## 🎯 **Next Steps**

After successful local testing:

1. **Production Deployment**

   - Configure production environment variables
   - Set up external monitoring endpoints
   - Configure production alert channels

2. **Team Training**

   - Train operations team on dashboard usage
   - Document alert response procedures
   - Create runbooks for common issues

3. **Continuous Improvement**
   - Monitor system performance
   - Adjust alert thresholds based on baseline
   - Add custom metrics for business KPIs

---

## 📞 **Support**

If you encounter issues:

1. Check the troubleshooting section above
2. Review Docker logs for error messages
3. Verify all prerequisites are met
4. Ensure sufficient system resources

**🎉 Congratulations! You now have a fully functional monitoring and observability system for the Agentic RAG platform.**

## 📁 **Quick Reference - Service URLs**

| Service               | URL                           | Purpose                        |
| --------------------- | ----------------------------- | ------------------------------ |
| **Agentic RAG API**   | http://localhost:8000         | Main application API           |
| **API Documentation** | http://localhost:8000/docs    | Interactive API docs           |
| **Prometheus**        | http://localhost:9090         | Metrics collection and queries |
| **Grafana**           | http://localhost:3000         | Dashboards and visualization   |
| **Kibana**            | http://localhost:5601         | Log search and analysis        |
| **Jaeger**            | http://localhost:16686        | Distributed tracing            |
| **Alertmanager**      | http://localhost:9093         | Alert management               |
| **MinIO Console**     | http://localhost:9001         | Object storage management      |
| **Node Exporter**     | http://localhost:9100/metrics | System metrics                 |
| **cAdvisor**          | http://localhost:8080         | Container metrics              |

## 🔑 **Default Credentials**

| Service     | Username   | Password          |
| ----------- | ---------- | ----------------- |
| **Grafana** | admin      | agentic-rag-admin |
| **MinIO**   | minioadmin | minioadmin        |

## 🚀 **One-Command Startup**

For convenience, you can use these commands to start everything:

```bash
# Start everything with monitoring
docker-compose up -d && docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
timeout 120 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'

# Run validation tests
python scripts/test_monitoring_stack.py
```
