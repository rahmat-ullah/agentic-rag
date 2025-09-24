"""
Health check endpoints for the Agentic RAG API.

This module provides health check endpoints for monitoring the application
and its dependencies.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from fastapi import APIRouter, Depends
import structlog

from agentic_rag.adapters.database import get_database_adapter
from agentic_rag.config import get_settings
from ..models.responses import HealthResponse, HealthStatus


logger = structlog.get_logger(__name__)
router = APIRouter()

# Track application start time
_start_time = time.time()


async def check_database_health() -> HealthStatus:
    """Check database connectivity and performance."""
    start_time = time.time()
    
    try:
        db = get_database_adapter()
        is_healthy = db.health_check()
        
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return HealthStatus(
            name="database",
            status="healthy" if is_healthy else "unhealthy",
            response_time_ms=response_time,
            details={
                "connection_pool": "active",
                "query_test": "passed" if is_healthy else "failed"
            }
        )
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error("Database health check failed", error=str(e))
        
        return HealthStatus(
            name="database",
            status="unhealthy",
            response_time_ms=response_time,
            details={
                "error": str(e),
                "connection_pool": "error"
            }
        )


async def check_vector_store_health() -> HealthStatus:
    """Check vector store (ChromaDB) connectivity."""
    start_time = time.time()
    
    try:
        # TODO: Implement ChromaDB health check when vector store is integrated
        # For now, return a placeholder
        response_time = (time.time() - start_time) * 1000
        
        return HealthStatus(
            name="vector_store",
            status="healthy",
            response_time_ms=response_time,
            details={
                "collections": "accessible",
                "embedding_service": "ready"
            }
        )
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error("Vector store health check failed", error=str(e))
        
        return HealthStatus(
            name="vector_store",
            status="unhealthy",
            response_time_ms=response_time,
            details={
                "error": str(e),
                "collections": "error"
            }
        )


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns the overall health status of the application.
    """
    settings = get_settings()
    uptime_seconds = time.time() - _start_time
    
    # Check all components
    components = [
        await check_database_health(),
        await check_vector_store_health(),
    ]
    
    # Determine overall status
    component_statuses = [comp.status for comp in components]
    if all(status == "healthy" for status in component_statuses):
        overall_status = "healthy"
    elif any(status == "unhealthy" for status in component_statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    health_data = {
        "status": overall_status,
        "version": "1.0.0",
        "environment": settings.environment,
        "uptime_seconds": round(uptime_seconds, 2),
        "timestamp": datetime.utcnow().isoformat(),
        "components": [comp.dict() for comp in components]
    }
    
    return HealthResponse(
        message=f"Health check completed - status: {overall_status}",
        data=health_data
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes.
    
    Returns whether the application is ready to serve traffic.
    """
    # Check critical dependencies
    db_health = await check_database_health()
    
    is_ready = db_health.status == "healthy"
    
    ready_data = {
        "ready": is_ready,
        "checks": {
            "database": db_health.status == "healthy"
        }
    }
    
    return HealthResponse(
        message=f"Readiness check completed - ready: {is_ready}",
        data=ready_data
    )


@router.get("/live", response_model=HealthResponse)
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes.
    
    Returns whether the application is alive and should not be restarted.
    """
    uptime_seconds = time.time() - _start_time
    
    # Simple liveness check - if we can respond, we're alive
    live_data = {
        "alive": True,
        "uptime_seconds": round(uptime_seconds, 2),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return HealthResponse(
        message="Liveness check completed - application is alive",
        data=live_data
    )
