"""
OpenTelemetry Tracing Service

This module provides distributed tracing capabilities using OpenTelemetry,
including automatic instrumentation and custom span creation.
"""

import asyncio
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

from agentic_rag.config import get_settings
from agentic_rag.schemas.monitoring import TraceCreateRequest
from agentic_rag.services.monitoring_service import get_monitoring_service

logger = structlog.get_logger(__name__)


class TracingService:
    """Service for managing OpenTelemetry distributed tracing."""
    
    def __init__(self):
        self.settings = get_settings()
        self.tracer_provider = None
        self.tracer = None
        self._initialized = False
        
        logger.info("Tracing service initialized")
    
    def initialize(self, app=None):
        """Initialize OpenTelemetry tracing."""
        if self._initialized:
            return
        
        try:
            # Create resource with service information
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.settings.otel_service_name,
                ResourceAttributes.SERVICE_VERSION: "1.0.0",
                ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("HOSTNAME", "unknown"),
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.settings.environment,
            })
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self.tracer_provider)
            
            # Configure span exporters
            self._configure_exporters()
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Instrument FastAPI if app is provided
            if app:
                self._instrument_fastapi(app)
            
            # Instrument other libraries
            self._instrument_libraries()
            
            self._initialized = True
            logger.info("OpenTelemetry tracing initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize tracing", error=str(e))
            raise
    
    def _configure_exporters(self):
        """Configure span exporters based on settings."""
        exporters = []
        
        # OTLP exporter (for Jaeger, Zipkin, etc.)
        if self.settings.otel_enabled and self.settings.otel_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.settings.otel_endpoint,
                    insecure=True  # Use secure=False for HTTPS in production
                )
                exporters.append(otlp_exporter)
                logger.info("OTLP span exporter configured", endpoint=self.settings.otel_endpoint)
            except Exception as e:
                logger.warning("Failed to configure OTLP exporter", error=str(e))
        
        # Console exporter for development
        if self.settings.environment == "development":
            console_exporter = ConsoleSpanExporter()
            exporters.append(console_exporter)
            logger.info("Console span exporter configured")
        
        # Add exporters to tracer provider
        for exporter in exporters:
            span_processor = BatchSpanProcessor(exporter)
            self.tracer_provider.add_span_processor(span_processor)
    
    def _instrument_fastapi(self, app):
        """Instrument FastAPI application."""
        try:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=self.tracer_provider,
                excluded_urls="health,metrics,docs,openapi.json"
            )
            logger.info("FastAPI instrumentation configured")
        except Exception as e:
            logger.warning("Failed to instrument FastAPI", error=str(e))
    
    def _instrument_libraries(self):
        """Instrument common libraries."""
        try:
            # Instrument SQLAlchemy
            SQLAlchemyInstrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
            logger.info("SQLAlchemy instrumentation configured")
            
            # Instrument HTTPX
            HTTPXClientInstrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
            logger.info("HTTPX instrumentation configured")
            
        except Exception as e:
            logger.warning("Failed to instrument libraries", error=str(e))
    
    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[UUID] = None
    ):
        """Start a new span with optional attributes."""
        if not self._initialized or not self.tracer:
            # If tracing is not initialized, provide a no-op context
            yield None
            return
        
        span = self.tracer.start_span(name)
        
        try:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            # Add tenant ID if provided
            if tenant_id:
                span.set_attribute("tenant.id", str(tenant_id))
            
            # Set span in context
            with trace.use_span(span, end_on_exit=True):
                yield span
                
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
    
    async def create_custom_span(
        self,
        name: str,
        operation_name: str,
        service_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[UUID] = None,
        duration_ms: Optional[float] = None,
        error: bool = False,
        error_message: Optional[str] = None
    ):
        """Create a custom span and record it in the monitoring system."""
        try:
            # Get current span context
            current_span = trace.get_current_span()
            trace_id = None
            span_id = None
            parent_span_id = None
            
            if current_span and current_span.is_recording():
                span_context = current_span.get_span_context()
                trace_id = format(span_context.trace_id, '032x')
                span_id = format(span_context.span_id, '016x')
                
                # Get parent span ID if available
                parent_context = trace.get_current_span().parent
                if parent_context:
                    parent_span_id = format(parent_context.span_id, '016x')
            
            # If no active span, create trace IDs
            if not trace_id:
                import random
                trace_id = format(random.getrandbits(128), '032x')
                span_id = format(random.getrandbits(64), '016x')
            
            # Create trace request
            trace_request = TraceCreateRequest(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                service_name=service_name,
                start_time=datetime.utcnow(),
                tags=attributes or {},
                error=error,
                error_message=error_message
            )
            
            # If duration is provided, calculate end time
            if duration_ms:
                from datetime import timedelta
                trace_request.end_time = trace_request.start_time + timedelta(milliseconds=duration_ms)
            
            # Record in monitoring system
            if tenant_id:
                monitoring_service = await get_monitoring_service()
                await monitoring_service.record_trace(tenant_id, trace_request)
            
            logger.debug(
                "Custom span created",
                trace_id=trace_id,
                span_id=span_id,
                operation=operation_name
            )
            
        except Exception as e:
            logger.warning("Failed to create custom span", error=str(e))
    
    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the current span."""
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.add_event(name, attributes or {})
        except Exception as e:
            logger.warning("Failed to add span event", error=str(e))
    
    def set_span_attribute(self, key: str, value: Any):
        """Set an attribute on the current span."""
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute(key, str(value))
        except Exception as e:
            logger.warning("Failed to set span attribute", error=str(e))
    
    def record_exception(self, exception: Exception):
        """Record an exception in the current span."""
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.record_exception(exception)
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
        except Exception as e:
            logger.warning("Failed to record exception", error=str(e))
    
    def get_trace_context(self) -> Optional[Dict[str, str]]:
        """Get current trace context for propagation."""
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                span_context = current_span.get_span_context()
                return {
                    "trace_id": format(span_context.trace_id, '032x'),
                    "span_id": format(span_context.span_id, '016x'),
                    "trace_flags": format(span_context.trace_flags, '02x')
                }
        except Exception as e:
            logger.warning("Failed to get trace context", error=str(e))
        
        return None
    
    def shutdown(self):
        """Shutdown tracing service."""
        try:
            if self.tracer_provider:
                self.tracer_provider.shutdown()
            logger.info("Tracing service shutdown completed")
        except Exception as e:
            logger.error("Failed to shutdown tracing service", error=str(e))


# Global tracing service instance
_tracing_service: Optional[TracingService] = None


def get_tracing_service() -> TracingService:
    """Get the global tracing service instance."""
    global _tracing_service
    
    if _tracing_service is None:
        _tracing_service = TracingService()
    
    return _tracing_service


def initialize_tracing(app=None):
    """Initialize tracing for the application."""
    tracing_service = get_tracing_service()
    tracing_service.initialize(app)


# Convenience functions for common tracing operations
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None, tenant_id: Optional[UUID] = None):
    """Start a new span."""
    tracing_service = get_tracing_service()
    return tracing_service.start_span(name, attributes, tenant_id)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Add an event to the current span."""
    tracing_service = get_tracing_service()
    tracing_service.add_span_event(name, attributes)


def set_span_attribute(key: str, value: Any):
    """Set an attribute on the current span."""
    tracing_service = get_tracing_service()
    tracing_service.set_span_attribute(key, value)


def record_exception(exception: Exception):
    """Record an exception in the current span."""
    tracing_service = get_tracing_service()
    tracing_service.record_exception(exception)
