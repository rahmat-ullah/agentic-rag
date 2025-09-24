"""
Three-Hop Search API routes.

This module provides API endpoints for the three-hop retrieval pipeline
that follows document relationships to find the most relevant information.
"""

import asyncio
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from agentic_rag.api.dependencies.auth import get_current_user, get_effective_tenant_id
from agentic_rag.api.dependencies.database import get_db_session
from agentic_rag.api.models.responses import SuccessResponse
from agentic_rag.api.models.three_hop_search import (
    ThreeHopSearchRequest, ThreeHopSearchResponse, ThreeHopConfig
)
from agentic_rag.models.database import User
from agentic_rag.services.three_hop_search import get_three_hop_search_service
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/three-hop", tags=["Three-Hop Search"])


@router.post("/search", response_model=SuccessResponse[ThreeHopSearchResponse])
async def three_hop_search(
    request: ThreeHopSearchRequest,
    current_user: User = Depends(get_current_user),
    tenant_id: UUID = Depends(get_effective_tenant_id),
    db_session: Session = Depends(get_db_session),
    http_request: Request = None
):
    """
    Execute a three-hop search query.
    
    This endpoint performs the sophisticated three-hop search pattern:
    - **H1**: RFQ anchor search to find relevant RFQ documents
    - **H2**: Linked offer discovery via document relationships
    - **H3**: Targeted chunk retrieval from linked offers
    
    **Features:**
    - Configurable parameters for each hop
    - Parallel processing for improved performance
    - Intelligent fallback for unlinked documents
    - Comprehensive result explanation
    - Performance monitoring and caching
    
    **Performance:**
    - Target: <10 seconds for complete search (95th percentile)
    - Supports 50+ concurrent queries
    - Handles 100+ linked documents efficiently
    
    **Use Cases:**
    - Finding technical specifications in linked offers
    - Discovering pricing information across document relationships
    - Locating compliance details in related documents
    - Cross-referencing requirements with solutions
    """
    try:
        # Log the request
        logger.info(
            f"Three-hop search request",
            query=request.query[:100],
            tenant_id=str(tenant_id),
            user_id=str(current_user.id),
            config_provided=request.config is not None,
            explain_results=request.explain_results
        )
        
        # Execute the three-hop search
        three_hop_service = get_three_hop_search_service()
        
        # Set timeout based on config or default
        timeout_seconds = 10
        if request.config and request.config.timeout_seconds:
            timeout_seconds = request.config.timeout_seconds
        
        # Execute with timeout
        search_result = await asyncio.wait_for(
            three_hop_service.three_hop_search(
                db_session=db_session,
                request=request,
                tenant_id=tenant_id,
                user_id=current_user.id
            ),
            timeout=timeout_seconds
        )
        
        # Log successful completion
        logger.info(
            f"Three-hop search completed successfully",
            query=request.query[:50],
            total_time_ms=search_result.statistics.timings.total_time_ms,
            h1_results=len(search_result.three_hop_results.h1_anchors),
            h2_results=len(search_result.three_hop_results.h2_offers),
            h3_results=len(search_result.three_hop_results.h3_chunks),
            cache_hits=search_result.statistics.cache_hits
        )
        
        return SuccessResponse(
            data=search_result,
            message=f"Three-hop search completed in {search_result.statistics.timings.total_time_ms}ms"
        )
        
    except asyncio.TimeoutError:
        logger.error(
            f"Three-hop search timeout",
            query=request.query[:50],
            timeout_seconds=timeout_seconds
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Three-hop search timed out after {timeout_seconds} seconds"
        )
    except ValueError as e:
        logger.error(f"Three-hop search validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Three-hop search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Three-hop search failed"
        )


@router.get("/config/default", response_model=SuccessResponse[ThreeHopConfig])
async def get_default_config():
    """
    Get the default three-hop search configuration.

    This endpoint returns the default configuration parameters for three-hop search,
    which can be used as a starting point for customization.

    **Configuration Categories:**
    - **H1 (RFQ Search)**: Parameters for finding anchor RFQ documents
    - **H2 (Link Discovery)**: Parameters for discovering linked offers
    - **H3 (Chunk Retrieval)**: Parameters for retrieving relevant chunks
    - **Performance**: Timeout, caching, and parallel processing settings
    """
    try:
        default_config = ThreeHopConfig()

        return SuccessResponse(
            data=default_config,
            message="Default three-hop search configuration retrieved"
        )

    except Exception as e:
        logger.error(f"Failed to get default config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve default configuration"
        )


@router.get("/config/presets", response_model=SuccessResponse[List[str]])
async def list_config_presets():
    """
    List available configuration presets.

    This endpoint returns a list of predefined configuration presets that can be used
    for different search scenarios and performance requirements.

    **Available Presets:**
    - **high_precision**: Optimized for accuracy with stricter thresholds
    - **high_recall**: Optimized for completeness with relaxed thresholds
    - **balanced**: Default balanced configuration
    - **fast**: Optimized for speed with reduced result counts
    - **comprehensive**: Optimized for thorough search with maximum coverage
    """
    try:
        presets = ThreeHopConfig.list_presets()

        return SuccessResponse(
            data=presets,
            message=f"Found {len(presets)} available configuration presets"
        )

    except Exception as e:
        logger.error(f"Failed to list presets: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration presets"
        )


@router.get("/config/presets/{preset_name}", response_model=SuccessResponse[ThreeHopConfig])
async def get_config_preset(preset_name: str):
    """
    Get a specific configuration preset.

    This endpoint returns the configuration for a specific preset, allowing users
    to see the exact parameters used for different search strategies.

    **Preset Descriptions:**
    - **high_precision**: Stricter thresholds, fewer results, higher accuracy
    - **high_recall**: Relaxed thresholds, more results, broader coverage
    - **balanced**: Default configuration with good balance of speed and quality
    - **fast**: Optimized for speed with minimal processing time
    - **comprehensive**: Maximum coverage with extensive search parameters
    """
    try:
        config = ThreeHopConfig.get_preset(preset_name)

        return SuccessResponse(
            data=config,
            message=f"Configuration preset '{preset_name}' retrieved"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get preset {preset_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve preset '{preset_name}'"
        )


@router.post("/config/validate", response_model=SuccessResponse[dict])
async def validate_config(config: ThreeHopConfig):
    """
    Validate a three-hop search configuration.

    This endpoint validates a configuration and returns warnings or recommendations
    for optimization. It helps users understand the implications of their configuration
    choices and suggests improvements.

    **Validation Checks:**
    - Performance impact assessment
    - Quality threshold recommendations
    - Timeout and resource usage warnings
    - Efficiency optimization suggestions
    """
    try:
        warnings = config.validate_configuration()

        validation_result = {
            "valid": True,
            "warnings": warnings,
            "warning_count": len(warnings),
            "recommendations": []
        }

        # Add specific recommendations based on configuration
        if config.timeout_seconds > 20:
            validation_result["recommendations"].append(
                "Consider enabling parallel processing for better performance with long timeouts"
            )

        if config.h1_max_results > 15 and not config.enable_caching:
            validation_result["recommendations"].append(
                "Enable caching to improve performance with high result counts"
            )

        if config.h3_enable_reranking and config.timeout_seconds < 10:
            validation_result["recommendations"].append(
                "Reranking adds processing time - consider increasing timeout or disabling reranking"
            )

        return SuccessResponse(
            data=validation_result,
            message=f"Configuration validated with {len(warnings)} warnings"
        )

    except Exception as e:
        logger.error(f"Failed to validate config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate configuration"
        )


@router.get("/health", response_model=SuccessResponse[dict])
async def three_hop_health_check():
    """
    Check the health of the three-hop search service.
    
    This endpoint provides health status for the three-hop search service including
    all dependent services and performance metrics.
    """
    try:
        three_hop_service = get_three_hop_search_service()
        
        # Initialize service to check dependencies
        await three_hop_service.initialize()
        
        # Get service statistics
        stats = three_hop_service.get_stats()
        
        health_status = {
            "status": "healthy",
            "service": "three_hop_search",
            "dependencies": {
                "search_service": "healthy",
                "linking_service": "healthy", 
                "vector_search": "healthy",
                "query_processor": "healthy"
            },
            "statistics": stats,
            "cache_size": len(three_hop_service._cache),
            "version": "1.0.0"
        }
        
        return SuccessResponse(
            data=health_status,
            message="Three-hop search service is healthy"
        )
        
    except Exception as e:
        logger.error(f"Three-hop health check failed: {e}", exc_info=True)
        
        health_status = {
            "status": "unhealthy",
            "service": "three_hop_search",
            "error": str(e),
            "dependencies": {
                "search_service": "unknown",
                "linking_service": "unknown",
                "vector_search": "unknown", 
                "query_processor": "unknown"
            }
        }
        
        return SuccessResponse(
            data=health_status,
            message="Three-hop search service health check completed with issues"
        )


@router.get("/stats", response_model=SuccessResponse[dict])
async def get_three_hop_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get three-hop search service statistics.
    
    This endpoint provides detailed performance and usage statistics
    for the three-hop search service.
    
    **Statistics Include:**
    - Total searches performed
    - Cache hit/miss ratios
    - Average search times
    - Hop-specific metrics
    - Performance trends
    """
    try:
        three_hop_service = get_three_hop_search_service()
        stats = three_hop_service.get_stats()
        
        # Add additional computed metrics
        total_searches = stats.get("total_searches", 0)
        cache_hits = stats.get("cache_hits", 0)
        cache_misses = stats.get("cache_misses", 0)
        
        enhanced_stats = {
            **stats,
            "cache_hit_rate": cache_hits / max(total_searches, 1),
            "cache_miss_rate": cache_misses / max(total_searches, 1),
            "total_cache_requests": cache_hits + cache_misses
        }
        
        return SuccessResponse(
            data=enhanced_stats,
            message="Three-hop search statistics retrieved"
        )
        
    except Exception as e:
        logger.error(f"Failed to get three-hop stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve three-hop search statistics"
        )


@router.get("/performance", response_model=SuccessResponse[dict])
async def get_performance_summary(
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive performance summary for three-hop search.

    This endpoint provides detailed performance analytics including:
    - Response time percentiles (P50, P95, P99)
    - Success rates and error patterns
    - Hop-specific performance metrics
    - Configuration usage patterns
    - Cache performance statistics
    - Execution pattern analysis

    **Performance Metrics:**
    - **Response Times**: Detailed timing analysis across all hops
    - **Success Rates**: Percentage of searches returning results
    - **Cache Performance**: Hit rates and efficiency metrics
    - **Configuration Analysis**: Most effective configuration patterns
    - **Execution Patterns**: Parallel vs sequential performance
    """
    try:
        three_hop_service = get_three_hop_search_service()
        performance_summary = three_hop_service.get_performance_summary()

        return SuccessResponse(
            data=performance_summary,
            message="Three-hop search performance summary retrieved"
        )

    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance summary"
        )
