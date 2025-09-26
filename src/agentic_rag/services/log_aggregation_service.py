"""
Log Aggregation Service

This module provides log aggregation and analysis capabilities for the ELK stack,
including log shipping, parsing, enrichment, and alerting.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

import structlog
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

from agentic_rag.config import get_settings
from agentic_rag.schemas.monitoring import AlertCreateRequest
from agentic_rag.models.monitoring import AlertSeverity
from agentic_rag.services.monitoring_service import get_monitoring_service

logger = structlog.get_logger(__name__)


class LogEntry:
    """Structured log entry for aggregation."""
    
    def __init__(
        self,
        timestamp: datetime,
        level: str,
        message: str,
        service: str,
        tenant_id: Optional[UUID] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **kwargs
    ):
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.service = service
        self.tenant_id = tenant_id
        self.request_id = request_id
        self.trace_id = trace_id
        self.span_id = span_id
        self.extra_fields = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for Elasticsearch."""
        doc = {
            "@timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "service": self.service,
        }
        
        if self.tenant_id:
            doc["tenant_id"] = str(self.tenant_id)
        
        if self.request_id:
            doc["request_id"] = self.request_id
        
        if self.trace_id:
            doc["trace_id"] = self.trace_id
        
        if self.span_id:
            doc["span_id"] = self.span_id
        
        # Add extra fields
        doc.update(self.extra_fields)
        
        return doc


class LogAggregationService:
    """Service for log aggregation and analysis using Elasticsearch."""
    
    def __init__(self):
        self.settings = get_settings()
        self.es_client = None
        self._running = False
        self._log_buffer = []
        self._buffer_size = 100
        self._flush_interval = 30.0  # seconds
        self._flush_task = None
        self._alert_rules = []
        
        # Index configuration
        self.index_prefix = "agentic-rag-logs"
        self.index_pattern = f"{self.index_prefix}-*"
        
        logger.info("Log aggregation service initialized")
    
    async def start(self):
        """Start the log aggregation service."""
        if self._running:
            return
        
        try:
            # Initialize Elasticsearch client
            await self._initialize_elasticsearch()
            
            # Create index templates
            await self._create_index_templates()
            
            # Load alert rules
            await self._load_alert_rules()
            
            # Start background tasks
            self._running = True
            self._flush_task = asyncio.create_task(self._flush_loop())
            
            logger.info("Log aggregation service started")
            
        except Exception as e:
            logger.error("Failed to start log aggregation service", error=str(e))
            raise
    
    async def stop(self):
        """Stop the log aggregation service."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining logs
        await self._flush_logs()
        
        # Close Elasticsearch client
        if self.es_client:
            await self.es_client.close()
        
        logger.info("Log aggregation service stopped")
    
    async def ship_log(self, log_entry: LogEntry):
        """Ship a log entry to the aggregation system."""
        try:
            # Add to buffer
            self._log_buffer.append(log_entry)
            
            # Flush if buffer is full
            if len(self._log_buffer) >= self._buffer_size:
                await self._flush_logs()
                
        except Exception as e:
            logger.warning("Failed to ship log entry", error=str(e))
    
    async def ship_logs_batch(self, log_entries: List[LogEntry]):
        """Ship multiple log entries in batch."""
        try:
            self._log_buffer.extend(log_entries)
            
            # Flush if buffer is full
            if len(self._log_buffer) >= self._buffer_size:
                await self._flush_logs()
                
        except Exception as e:
            logger.warning("Failed to ship log batch", error=str(e))
    
    async def search_logs(
        self,
        query: str = "*",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        service: Optional[str] = None,
        level: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
        size: int = 100,
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """Search logs with filters."""
        try:
            if not self.es_client:
                raise RuntimeError("Elasticsearch client not initialized")
            
            # Build query
            es_query = {
                "query": {
                    "bool": {
                        "must": [],
                        "filter": []
                    }
                },
                "sort": [
                    {"@timestamp": {"order": sort_order}}
                ],
                "size": size
            }
            
            # Add text query
            if query and query != "*":
                es_query["query"]["bool"]["must"].append({
                    "multi_match": {
                        "query": query,
                        "fields": ["message", "service", "level"]
                    }
                })
            
            # Add time range filter
            if start_time or end_time:
                time_filter = {"range": {"@timestamp": {}}}
                if start_time:
                    time_filter["range"]["@timestamp"]["gte"] = start_time.isoformat()
                if end_time:
                    time_filter["range"]["@timestamp"]["lte"] = end_time.isoformat()
                es_query["query"]["bool"]["filter"].append(time_filter)
            
            # Add service filter
            if service:
                es_query["query"]["bool"]["filter"].append({
                    "term": {"service": service}
                })
            
            # Add level filter
            if level:
                es_query["query"]["bool"]["filter"].append({
                    "term": {"level": level}
                })
            
            # Add tenant filter
            if tenant_id:
                es_query["query"]["bool"]["filter"].append({
                    "term": {"tenant_id": str(tenant_id)}
                })
            
            # Execute search
            response = await self.es_client.search(
                index=self.index_pattern,
                body=es_query
            )
            
            return {
                "total": response["hits"]["total"]["value"],
                "logs": [hit["_source"] for hit in response["hits"]["hits"]],
                "took": response["took"]
            }
            
        except Exception as e:
            logger.error("Failed to search logs", error=str(e))
            raise
    
    async def get_log_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tenant_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Get log statistics and aggregations."""
        try:
            if not self.es_client:
                raise RuntimeError("Elasticsearch client not initialized")
            
            # Build aggregation query
            agg_query = {
                "size": 0,
                "query": {
                    "bool": {
                        "filter": []
                    }
                },
                "aggs": {
                    "levels": {
                        "terms": {"field": "level"}
                    },
                    "services": {
                        "terms": {"field": "service"}
                    },
                    "timeline": {
                        "date_histogram": {
                            "field": "@timestamp",
                            "interval": "1h"
                        }
                    }
                }
            }
            
            # Add time range filter
            if start_time or end_time:
                time_filter = {"range": {"@timestamp": {}}}
                if start_time:
                    time_filter["range"]["@timestamp"]["gte"] = start_time.isoformat()
                if end_time:
                    time_filter["range"]["@timestamp"]["lte"] = end_time.isoformat()
                agg_query["query"]["bool"]["filter"].append(time_filter)
            
            # Add tenant filter
            if tenant_id:
                agg_query["query"]["bool"]["filter"].append({
                    "term": {"tenant_id": str(tenant_id)}
                })
            
            # Execute aggregation
            response = await self.es_client.search(
                index=self.index_pattern,
                body=agg_query
            )
            
            return {
                "total_logs": response["hits"]["total"]["value"],
                "levels": {
                    bucket["key"]: bucket["doc_count"]
                    for bucket in response["aggregations"]["levels"]["buckets"]
                },
                "services": {
                    bucket["key"]: bucket["doc_count"]
                    for bucket in response["aggregations"]["services"]["buckets"]
                },
                "timeline": [
                    {
                        "timestamp": bucket["key_as_string"],
                        "count": bucket["doc_count"]
                    }
                    for bucket in response["aggregations"]["timeline"]["buckets"]
                ]
            }
            
        except Exception as e:
            logger.error("Failed to get log statistics", error=str(e))
            raise
    
    async def _initialize_elasticsearch(self):
        """Initialize Elasticsearch client."""
        try:
            # Configure Elasticsearch client
            es_config = {
                "hosts": [self.settings.elasticsearch_url or "http://localhost:9200"],
                "timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True
            }
            
            # Add authentication if configured
            if self.settings.elasticsearch_username and self.settings.elasticsearch_password:
                es_config["http_auth"] = (
                    self.settings.elasticsearch_username,
                    self.settings.elasticsearch_password
                )
            
            self.es_client = AsyncElasticsearch(**es_config)
            
            # Test connection
            await self.es_client.ping()
            logger.info("Elasticsearch connection established")
            
        except Exception as e:
            logger.warning("Elasticsearch connection failed - log aggregation disabled", error=str(e))
            self.es_client = None
    
    async def _create_index_templates(self):
        """Create Elasticsearch index templates."""
        if not self.es_client:
            return
        
        try:
            template = {
                "index_patterns": [f"{self.index_prefix}-*"],
                "template": {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "index.lifecycle.name": "agentic-rag-logs-policy",
                        "index.lifecycle.rollover_alias": f"{self.index_prefix}-alias"
                    },
                    "mappings": {
                        "properties": {
                            "@timestamp": {"type": "date"},
                            "level": {"type": "keyword"},
                            "message": {"type": "text"},
                            "service": {"type": "keyword"},
                            "tenant_id": {"type": "keyword"},
                            "request_id": {"type": "keyword"},
                            "trace_id": {"type": "keyword"},
                            "span_id": {"type": "keyword"}
                        }
                    }
                }
            }
            
            await self.es_client.indices.put_index_template(
                name=f"{self.index_prefix}-template",
                body=template
            )
            
            logger.info("Elasticsearch index template created")
            
        except Exception as e:
            logger.warning("Failed to create index template", error=str(e))
    
    async def _load_alert_rules(self):
        """Load log-based alert rules."""
        # Define default alert rules
        self._alert_rules = [
            {
                "name": "high_error_rate",
                "query": "level:ERROR",
                "threshold": 10,
                "window": 300,  # 5 minutes
                "severity": AlertSeverity.CRITICAL,
                "description": "High error rate detected in logs"
            },
            {
                "name": "service_unavailable",
                "query": "message:\"Connection refused\" OR message:\"Service unavailable\"",
                "threshold": 5,
                "window": 60,  # 1 minute
                "severity": AlertSeverity.CRITICAL,
                "description": "Service unavailability detected"
            },
            {
                "name": "authentication_failures",
                "query": "message:\"Authentication failed\" OR message:\"Unauthorized\"",
                "threshold": 20,
                "window": 300,  # 5 minutes
                "severity": AlertSeverity.WARNING,
                "description": "High number of authentication failures"
            }
        ]
        
        logger.info(f"Loaded {len(self._alert_rules)} log-based alert rules")
    
    async def _flush_logs(self):
        """Flush buffered logs to Elasticsearch."""
        if not self._log_buffer or not self.es_client:
            return
        
        try:
            # Prepare bulk operations
            operations = []
            current_date = datetime.utcnow().strftime("%Y.%m.%d")
            index_name = f"{self.index_prefix}-{current_date}"
            
            for log_entry in self._log_buffer:
                operations.append({
                    "index": {"_index": index_name}
                })
                operations.append(log_entry.to_dict())
            
            # Execute bulk operation
            if operations:
                await self.es_client.bulk(body=operations)
                logger.debug(f"Flushed {len(self._log_buffer)} log entries to Elasticsearch")
            
            # Clear buffer
            self._log_buffer.clear()
            
            # Check alert rules
            await self._check_alert_rules()
            
        except Exception as e:
            logger.error("Failed to flush logs to Elasticsearch", error=str(e))
            # Keep logs in buffer for retry
    
    async def _flush_loop(self):
        """Background task for periodic log flushing."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in flush loop", error=str(e))
    
    async def _check_alert_rules(self):
        """Check log-based alert rules."""
        if not self.es_client:
            return
        
        try:
            for rule in self._alert_rules:
                await self._evaluate_alert_rule(rule)
        except Exception as e:
            logger.error("Failed to check alert rules", error=str(e))
    
    async def _evaluate_alert_rule(self, rule: Dict[str, Any]):
        """Evaluate a single alert rule."""
        try:
            # Query logs for the rule
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=rule["window"])
            
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"query_string": {"query": rule["query"]}},
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": start_time.isoformat(),
                                        "lte": end_time.isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": 0
            }
            
            response = await self.es_client.search(
                index=self.index_pattern,
                body=query
            )
            
            count = response["hits"]["total"]["value"]
            
            # Check if threshold is exceeded
            if count >= rule["threshold"]:
                await self._create_log_alert(rule, count)
                
        except Exception as e:
            logger.warning(f"Failed to evaluate alert rule {rule['name']}", error=str(e))
    
    async def _create_log_alert(self, rule: Dict[str, Any], count: int):
        """Create an alert based on log rule violation."""
        try:
            monitoring_service = await get_monitoring_service()
            
            alert_request = AlertCreateRequest(
                rule_name=rule["name"],
                alert_name=f"Log Alert: {rule['name']}",
                severity=rule["severity"],
                labels={
                    "source": "logs",
                    "rule": rule["name"],
                    "query": rule["query"]
                },
                annotations={
                    "summary": f"{count} log entries matched rule in {rule['window']} seconds",
                    "description": rule["description"],
                    "threshold": str(rule["threshold"]),
                    "actual_count": str(count)
                },
                description=f"Log-based alert: {rule['description']}",
                source_service="log-aggregation"
            )
            
            # Use a default tenant ID for system alerts
            system_tenant_id = UUID("00000000-0000-0000-0000-000000000000")
            await monitoring_service.create_alert(system_tenant_id, alert_request)
            
            logger.info(
                "Log-based alert created",
                rule=rule["name"],
                count=count,
                threshold=rule["threshold"]
            )
            
        except Exception as e:
            logger.error("Failed to create log alert", error=str(e))


# Global log aggregation service instance
_log_aggregation_service: Optional[LogAggregationService] = None


async def get_log_aggregation_service() -> LogAggregationService:
    """Get the global log aggregation service instance."""
    global _log_aggregation_service
    
    if _log_aggregation_service is None:
        _log_aggregation_service = LogAggregationService()
        await _log_aggregation_service.start()
    
    return _log_aggregation_service


# Convenience function for shipping logs
async def ship_log(
    level: str,
    message: str,
    service: str,
    tenant_id: Optional[UUID] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **kwargs
):
    """Ship a log entry to the aggregation system."""
    try:
        log_service = await get_log_aggregation_service()
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            service=service,
            tenant_id=tenant_id,
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id,
            **kwargs
        )
        await log_service.ship_log(log_entry)
    except Exception as e:
        # Don't let log shipping failures break the application
        logger.warning("Failed to ship log", error=str(e))
