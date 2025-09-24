"""
Search API models for request and response handling.

This module defines Pydantic models for search requests, responses,
and related data structures for the search and retrieval API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

from agentic_rag.models.database import DocumentKind


class SearchFilters(BaseModel):
    """Filters for search requests."""
    
    document_types: Optional[List[DocumentKind]] = Field(
        None, 
        description="Filter by document types (RFQ, OFFER, etc.)"
    )
    date_from: Optional[datetime] = Field(
        None, 
        description="Filter documents created after this date"
    )
    date_to: Optional[datetime] = Field(
        None, 
        description="Filter documents created before this date"
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional metadata-based filters"
    )
    section_path: Optional[List[str]] = Field(
        None, 
        description="Filter by document section path"
    )
    page_range: Optional[Dict[str, int]] = Field(
        None, 
        description="Filter by page range (page_from, page_to)"
    )
    is_table: Optional[bool] = Field(
        None, 
        description="Filter by table content"
    )
    min_token_count: Optional[int] = Field(
        None, 
        ge=1,
        description="Minimum token count for chunks"
    )
    max_token_count: Optional[int] = Field(
        None, 
        ge=1,
        description="Maximum token count for chunks"
    )
    
    @validator('page_range')
    def validate_page_range(cls, v):
        if v is not None:
            if 'page_from' in v and 'page_to' in v:
                if v['page_from'] > v['page_to']:
                    raise ValueError("page_from must be less than or equal to page_to")
        return v
    
    @validator('date_to')
    def validate_date_range(cls, v, values):
        if v is not None and 'date_from' in values and values['date_from'] is not None:
            if v < values['date_from']:
                raise ValueError("date_to must be after date_from")
        return v


class SearchOptions(BaseModel):
    """Options for search behavior."""
    
    include_metadata: bool = Field(
        True, 
        description="Include chunk metadata in results"
    )
    include_content: bool = Field(
        True, 
        description="Include chunk content in results"
    )
    include_document_info: bool = Field(
        True, 
        description="Include document information in results"
    )
    highlight_matches: bool = Field(
        False, 
        description="Highlight matching terms in content"
    )
    deduplicate_results: bool = Field(
        True, 
        description="Remove duplicate results"
    )
    score_threshold: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Minimum similarity score threshold"
    )


class SearchRequest(BaseModel):
    """Search request model."""
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Natural language search query"
    )
    filters: Optional[SearchFilters] = Field(
        None, 
        description="Search filters"
    )
    options: Optional[SearchOptions] = Field(
        None, 
        description="Search options"
    )
    page: int = Field(
        1, 
        ge=1,
        description="Page number for pagination"
    )
    page_size: int = Field(
        20, 
        ge=1, 
        le=100,
        description="Number of results per page"
    )
    
    @validator('query')
    def validate_query(cls, v):
        # Remove excessive whitespace
        v = ' '.join(v.split())
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "requirements for data processing",
                "filters": {
                    "document_types": ["RFQ"],
                    "date_from": "2024-01-01T00:00:00Z",
                    "metadata_filters": {
                        "section": "Requirements"
                    }
                },
                "options": {
                    "include_metadata": True,
                    "score_threshold": 0.7
                },
                "page": 1,
                "page_size": 20
            }
        }


class SearchResultItem(BaseModel):
    """Individual search result item."""
    
    document_id: UUID = Field(..., description="Document identifier")
    chunk_id: UUID = Field(..., description="Chunk identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Chunk metadata")
    document_info: Optional[Dict[str, Any]] = Field(None, description="Document information")
    highlighted_content: Optional[str] = Field(None, description="Content with highlighted matches")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "chunk_id": "123e4567-e89b-12d3-a456-426614174001",
                "title": "Data Processing Requirements",
                "content": "The system must process data in real-time with high accuracy...",
                "score": 0.95,
                "metadata": {
                    "document_type": "RFQ",
                    "section_path": ["Requirements", "Data Processing"],
                    "page_from": 5,
                    "page_to": 5,
                    "token_count": 150,
                    "is_table": False
                },
                "document_info": {
                    "kind": "RFQ",
                    "status": "PROCESSED",
                    "created_at": "2024-01-15T10:30:00Z"
                }
            }
        }


class SearchPagination(BaseModel):
    """Pagination metadata for search results."""
    
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Results per page")
    total_results: int = Field(..., ge=0, description="Total number of results")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "page": 1,
                "page_size": 20,
                "total_results": 150,
                "total_pages": 8,
                "has_next": True,
                "has_previous": False
            }
        }


class SearchStatistics(BaseModel):
    """Search performance and statistics."""
    
    search_time_ms: int = Field(..., ge=0, description="Total search time in milliseconds")
    query_processing_time_ms: int = Field(..., ge=0, description="Query processing time")
    vector_search_time_ms: int = Field(..., ge=0, description="Vector search time")
    ranking_time_ms: int = Field(..., ge=0, description="Result ranking time")
    total_chunks_searched: int = Field(..., ge=0, description="Total chunks searched")
    cache_hit: bool = Field(..., description="Whether query was served from cache")
    
    class Config:
        json_schema_extra = {
            "example": {
                "search_time_ms": 245,
                "query_processing_time_ms": 50,
                "vector_search_time_ms": 150,
                "ranking_time_ms": 45,
                "total_chunks_searched": 1500,
                "cache_hit": False
            }
        }


class SearchResponse(BaseModel):
    """Search response model."""
    
    query: str = Field(..., description="Original search query")
    results: List[SearchResultItem] = Field(..., description="Search results")
    pagination: SearchPagination = Field(..., description="Pagination information")
    statistics: SearchStatistics = Field(..., description="Search statistics")
    suggestions: Optional[List[str]] = Field(None, description="Query suggestions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "requirements for data processing",
                "results": [
                    {
                        "document_id": "123e4567-e89b-12d3-a456-426614174000",
                        "chunk_id": "123e4567-e89b-12d3-a456-426614174001",
                        "title": "Data Processing Requirements",
                        "content": "The system must process data in real-time...",
                        "score": 0.95,
                        "metadata": {
                            "document_type": "RFQ",
                            "section_path": ["Requirements"],
                            "page_from": 5
                        }
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 20,
                    "total_results": 150,
                    "total_pages": 8,
                    "has_next": True,
                    "has_previous": False
                },
                "statistics": {
                    "search_time_ms": 245,
                    "query_processing_time_ms": 50,
                    "vector_search_time_ms": 150,
                    "ranking_time_ms": 45,
                    "total_chunks_searched": 1500,
                    "cache_hit": False
                }
            }
        }


class SearchSuggestionsRequest(BaseModel):
    """Request model for search suggestions."""
    
    partial_query: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="Partial query for suggestions"
    )
    limit: int = Field(
        5, 
        ge=1, 
        le=20,
        description="Maximum number of suggestions"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "partial_query": "data proc",
                "limit": 5
            }
        }


class SearchSuggestionsResponse(BaseModel):
    """Response model for search suggestions."""
    
    suggestions: List[str] = Field(..., description="Query suggestions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "suggestions": [
                    "data processing",
                    "data processing requirements",
                    "data processing pipeline",
                    "data processing architecture",
                    "data processing performance"
                ]
            }
        }
