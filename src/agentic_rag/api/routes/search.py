"""
Search API endpoints for semantic search and retrieval.

This module provides endpoints for searching documents using natural language
queries with vector similarity search, filtering, and ranking capabilities.
"""

import time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer
import structlog

from agentic_rag.models.database import User
from agentic_rag.api.models.search import (
    SearchRequest, SearchResponse, SearchSuggestionsRequest, SearchSuggestionsResponse
)
from agentic_rag.api.models.responses import SuccessResponse, ErrorResponse
from agentic_rag.api.dependencies.auth import get_current_user, get_effective_tenant_id
from agentic_rag.api.exceptions import ValidationError, ServiceError
from agentic_rag.services.search_service import get_search_service, SearchService

logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/search", response_model=SuccessResponse[SearchResponse])
async def search_documents(
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    tenant_id: UUID = Depends(get_effective_tenant_id),
    search_service: SearchService = Depends(get_search_service),
    http_request: Request = None
):
    """
    Search documents using natural language queries.
    
    This endpoint performs semantic search across indexed document chunks
    using vector similarity search with optional filtering and ranking.
    
    **Features:**
    - Natural language query processing
    - Vector similarity search
    - Document type and metadata filtering
    - Relevance scoring and ranking
    - Pagination support
    - Multi-tenant isolation
    
    **Performance:**
    - Response time: < 2 seconds (95th percentile)
    - Concurrent searches: Up to 100
    - Query length: Up to 1000 characters
    - Results per query: Up to 100
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', None) if http_request else None
    
    logger.info(
        "Search request received",
        query=request.query[:100] + "..." if len(request.query) > 100 else request.query,
        user_id=str(current_user.id),
        tenant_id=str(tenant_id),
        page=request.page,
        page_size=request.page_size,
        request_id=request_id
    )
    
    try:
        # Validate request
        if not request.query.strip():
            raise ValidationError("Search query cannot be empty")
        
        # Perform search
        search_result = await search_service.search_documents(
            query=request.query,
            tenant_id=str(tenant_id),
            user_id=str(current_user.id),
            filters=request.filters,
            options=request.options,
            page=request.page,
            page_size=request.page_size
        )
        
        # Calculate total time
        total_time_ms = int((time.time() - start_time) * 1000)
        
        # Update statistics with actual timing
        search_result.statistics.search_time_ms = total_time_ms
        
        logger.info(
            "Search completed successfully",
            query=request.query[:50] + "..." if len(request.query) > 50 else request.query,
            results_count=len(search_result.results),
            total_results=search_result.pagination.total_results,
            search_time_ms=total_time_ms,
            cache_hit=search_result.statistics.cache_hit,
            user_id=str(current_user.id),
            tenant_id=str(tenant_id),
            request_id=request_id
        )
        
        return SuccessResponse(
            message=f"Search completed successfully. Found {search_result.pagination.total_results} results.",
            data=search_result
        )
        
    except ValidationError as e:
        logger.warning(
            "Search validation error",
            error=str(e),
            query=request.query[:100],
            user_id=str(current_user.id),
            request_id=request_id
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        
    except ServiceError as e:
        logger.error(
            "Search service error",
            error=str(e),
            query=request.query[:100],
            user_id=str(current_user.id),
            request_id=request_id
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service temporarily unavailable. Please try again."
        )
        
    except Exception as e:
        logger.error(
            "Unexpected search error",
            error=str(e),
            query=request.query[:100],
            user_id=str(current_user.id),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during search"
        )


@router.post("/search/suggestions", response_model=SuccessResponse[SearchSuggestionsResponse])
async def get_search_suggestions(
    request: SearchSuggestionsRequest,
    current_user: User = Depends(get_current_user),
    tenant_id: UUID = Depends(get_effective_tenant_id),
    search_service: SearchService = Depends(get_search_service),
    http_request: Request = None
):
    """
    Get search query suggestions based on partial input.
    
    This endpoint provides autocomplete suggestions for search queries
    based on previous searches and document content analysis.
    
    **Features:**
    - Real-time query suggestions
    - Based on document content and search history
    - Tenant-specific suggestions
    - Configurable suggestion limit
    """
    request_id = getattr(http_request.state, 'request_id', None) if http_request else None
    
    logger.info(
        "Search suggestions request",
        partial_query=request.partial_query,
        limit=request.limit,
        user_id=str(current_user.id),
        tenant_id=str(tenant_id),
        request_id=request_id
    )
    
    try:
        # Get suggestions
        suggestions = await search_service.get_search_suggestions(
            partial_query=request.partial_query,
            tenant_id=str(tenant_id),
            limit=request.limit
        )
        
        logger.info(
            "Search suggestions generated",
            partial_query=request.partial_query,
            suggestions_count=len(suggestions),
            user_id=str(current_user.id),
            request_id=request_id
        )
        
        return SuccessResponse(
            message=f"Generated {len(suggestions)} search suggestions",
            data=SearchSuggestionsResponse(suggestions=suggestions)
        )
        
    except Exception as e:
        logger.error(
            "Search suggestions error",
            error=str(e),
            partial_query=request.partial_query,
            user_id=str(current_user.id),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate search suggestions"
        )


@router.get("/search/health", response_model=SuccessResponse[dict])
async def search_health_check(
    search_service: SearchService = Depends(get_search_service)
):
    """
    Check the health of the search service.
    
    This endpoint provides health status for the search service including
    vector store connectivity, embedding service status, and performance metrics.
    """
    try:
        health_status = await search_service.health_check()
        
        return SuccessResponse(
            message="Search service health check completed",
            data=health_status
        )
        
    except Exception as e:
        logger.error("Search health check failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service health check failed"
        )


@router.get("/search/stats", response_model=SuccessResponse[dict])
async def get_search_statistics(
    current_user: User = Depends(get_current_user),
    tenant_id: UUID = Depends(get_effective_tenant_id),
    search_service: SearchService = Depends(get_search_service)
):
    """
    Get search service statistics and metrics.
    
    This endpoint provides performance statistics and usage metrics
    for the search service, including query patterns and performance data.
    """
    try:
        stats = await search_service.get_statistics(
            tenant_id=str(tenant_id)
        )
        
        return SuccessResponse(
            message="Search statistics retrieved successfully",
            data=stats
        )
        
    except Exception as e:
        logger.error(
            "Failed to get search statistics",
            error=str(e),
            user_id=str(current_user.id),
            tenant_id=str(tenant_id),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve search statistics"
        )
