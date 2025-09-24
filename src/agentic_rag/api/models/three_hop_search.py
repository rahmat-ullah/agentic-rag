"""
Three-Hop Search API models for request and response handling.

This module defines Pydantic models for three-hop search requests, responses,
and related data structures for the three-hop retrieval pipeline.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

from agentic_rag.models.database import DocumentKind


class ThreeHopConfig(BaseModel):
    """Configuration for three-hop search parameters."""
    
    # H1: RFQ Anchor Search Configuration
    h1_max_results: int = Field(
        10, 
        ge=1, 
        le=50,
        description="Maximum RFQ anchor documents to find"
    )
    h1_similarity_threshold: float = Field(
        0.7, 
        ge=0.0, 
        le=1.0,
        description="Minimum similarity threshold for RFQ search"
    )
    h1_boost_recent: bool = Field(
        True,
        description="Boost recently created RFQ documents"
    )
    h1_boost_factor: float = Field(
        1.2,
        ge=1.0,
        le=2.0,
        description="Boost factor for recent documents"
    )
    
    # H2: Link Discovery Configuration
    h2_min_confidence: float = Field(
        0.6, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence for document links"
    )
    h2_max_offers_per_rfq: int = Field(
        5, 
        ge=1, 
        le=20,
        description="Maximum offers to discover per RFQ"
    )
    h2_include_suggestions: bool = Field(
        False,
        description="Include suggested links in discovery"
    )
    h2_fallback_search: bool = Field(
        True,
        description="Enable fallback search for unlinked RFQs"
    )
    
    # H3: Chunk Retrieval Configuration
    h3_max_chunks_per_offer: int = Field(
        3, 
        ge=1, 
        le=10,
        description="Maximum chunks to retrieve per offer"
    )
    h3_chunk_similarity_threshold: float = Field(
        0.8, 
        ge=0.0, 
        le=1.0,
        description="Minimum similarity threshold for chunk retrieval"
    )
    h3_enable_reranking: bool = Field(
        True,
        description="Enable final reranking of chunks"
    )
    h3_context_window: int = Field(
        2,
        ge=0,
        le=5,
        description="Number of surrounding chunks to include for context"
    )
    
    # Performance Configuration
    max_total_results: int = Field(
        50,
        ge=1,
        le=200,
        description="Maximum total results to return"
    )
    timeout_seconds: int = Field(
        10,
        ge=1,
        le=30,
        description="Maximum search timeout in seconds"
    )
    enable_parallel_processing: bool = Field(
        True,
        description="Enable parallel processing for hops"
    )
    enable_caching: bool = Field(
        True,
        description="Enable caching for intermediate results"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "h1_max_results": 10,
                "h1_similarity_threshold": 0.7,
                "h1_boost_recent": True,
                "h2_min_confidence": 0.6,
                "h2_max_offers_per_rfq": 5,
                "h2_include_suggestions": False,
                "h3_max_chunks_per_offer": 3,
                "h3_chunk_similarity_threshold": 0.8,
                "h3_enable_reranking": True,
                "max_total_results": 50,
                "timeout_seconds": 10,
                "enable_parallel_processing": True,
                "enable_caching": True
            }
        }

    @classmethod
    def get_preset(cls, preset_name: str) -> "ThreeHopConfig":
        """Get a predefined configuration preset."""
        presets = {
            "high_precision": cls(
                h1_similarity_threshold=0.8,
                h2_min_confidence=0.8,
                h3_chunk_similarity_threshold=0.85,
                h1_max_results=5,
                h2_max_offers_per_rfq=3,
                h3_max_chunks_per_offer=2,
                max_total_results=20,
                timeout_seconds=15
            ),
            "high_recall": cls(
                h1_similarity_threshold=0.6,
                h2_min_confidence=0.5,
                h3_chunk_similarity_threshold=0.7,
                h1_max_results=15,
                h2_max_offers_per_rfq=8,
                h3_max_chunks_per_offer=5,
                max_total_results=100,
                timeout_seconds=20,
                h2_include_suggestions=True,
                h2_fallback_search=True
            ),
            "balanced": cls(),  # Default values
            "fast": cls(
                h1_max_results=5,
                h2_max_offers_per_rfq=3,
                h3_max_chunks_per_offer=2,
                max_total_results=20,
                timeout_seconds=5,
                enable_parallel_processing=True,
                h3_enable_reranking=False
            ),
            "comprehensive": cls(
                h1_max_results=20,
                h2_max_offers_per_rfq=10,
                h3_max_chunks_per_offer=8,
                max_total_results=150,
                timeout_seconds=25,
                h2_include_suggestions=True,
                h2_fallback_search=True,
                h3_context_window=3
            )
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")

        return presets[preset_name]

    @classmethod
    def list_presets(cls) -> List[str]:
        """List available configuration presets."""
        return ["high_precision", "high_recall", "balanced", "fast", "comprehensive"]

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of warnings/recommendations."""
        warnings = []

        # Performance warnings
        if self.timeout_seconds < 5:
            warnings.append("Timeout less than 5 seconds may cause incomplete results")

        if self.h1_max_results * self.h2_max_offers_per_rfq * self.h3_max_chunks_per_offer > 1000:
            warnings.append("Configuration may generate excessive intermediate results")

        # Quality warnings
        if self.h1_similarity_threshold < 0.5:
            warnings.append("Very low H1 similarity threshold may return irrelevant RFQs")

        if self.h2_min_confidence < 0.4:
            warnings.append("Very low H2 confidence threshold may include poor quality links")

        if self.h3_chunk_similarity_threshold < 0.6:
            warnings.append("Very low H3 similarity threshold may return irrelevant chunks")

        # Efficiency warnings
        if not self.enable_parallel_processing and self.timeout_seconds < 15:
            warnings.append("Sequential processing with short timeout may cause timeouts")

        if not self.enable_caching and self.h1_max_results > 10:
            warnings.append("Disabled caching with high result counts may impact performance")

        return warnings


class ThreeHopSearchRequest(BaseModel):
    """Three-hop search request model."""
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Natural language search query"
    )
    config: Optional[ThreeHopConfig] = Field(
        None,
        description="Three-hop search configuration"
    )
    explain_results: bool = Field(
        False,
        description="Include detailed explanation of search path"
    )
    include_intermediate_results: bool = Field(
        False,
        description="Include intermediate hop results in response"
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
                "query": "data processing requirements for real-time analytics",
                "config": {
                    "h1_max_results": 8,
                    "h2_min_confidence": 0.7,
                    "h3_max_chunks_per_offer": 4
                },
                "explain_results": True,
                "include_intermediate_results": False
            }
        }


class H1AnchorResult(BaseModel):
    """H1 RFQ anchor search result."""
    
    document_id: UUID = Field(..., description="RFQ document identifier")
    title: str = Field(..., description="RFQ document title")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    linked_offers_count: int = Field(..., ge=0, description="Number of linked offers")
    created_at: datetime = Field(..., description="Document creation date")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Data Analytics Platform RFQ",
                "score": 0.95,
                "linked_offers_count": 5,
                "created_at": "2024-01-15T10:30:00Z",
                "metadata": {
                    "document_type": "RFQ",
                    "status": "PROCESSED",
                    "chunk_count": 25
                }
            }
        }


class H2OfferResult(BaseModel):
    """H2 linked offer discovery result."""
    
    document_id: UUID = Field(..., description="Offer document identifier")
    title: str = Field(..., description="Offer document title")
    link_confidence: float = Field(..., ge=0.0, le=1.0, description="Link confidence score")
    chunks_found: int = Field(..., ge=0, description="Number of relevant chunks found")
    source_rfq_id: UUID = Field(..., description="Source RFQ document ID")
    link_type: str = Field(..., description="Type of link (manual, automatic, suggested)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Offer metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "456e7890-e89b-12d3-a456-426614174001",
                "title": "Analytics Platform Technical Offer",
                "link_confidence": 0.85,
                "chunks_found": 3,
                "source_rfq_id": "123e4567-e89b-12d3-a456-426614174000",
                "link_type": "automatic",
                "metadata": {
                    "document_type": "OFFER",
                    "offer_type": "technical",
                    "status": "PROCESSED"
                }
            }
        }


class H3ChunkResult(BaseModel):
    """H3 targeted chunk retrieval result."""
    
    chunk_id: UUID = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    source_path: str = Field(..., description="Path through the three hops")
    source_document_id: UUID = Field(..., description="Source document ID")
    source_document_title: str = Field(..., description="Source document title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Chunk metadata")
    context_chunks: Optional[List[str]] = Field(None, description="Surrounding context chunks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "789e0123-e89b-12d3-a456-426614174002",
                "content": "Our real-time analytics platform processes data streams with sub-second latency...",
                "score": 0.92,
                "source_path": "Data Analytics Platform RFQ → Technical Offer → Analytics Architecture",
                "source_document_id": "456e7890-e89b-12d3-a456-426614174001",
                "source_document_title": "Analytics Platform Technical Offer",
                "metadata": {
                    "section_path": ["Technical Architecture", "Analytics"],
                    "page_from": 12,
                    "token_count": 180,
                    "is_table": False
                }
            }
        }


class ThreeHopTimings(BaseModel):
    """Timing information for each hop."""
    
    h1_time_ms: int = Field(..., ge=0, description="H1 RFQ search time in milliseconds")
    h2_time_ms: int = Field(..., ge=0, description="H2 link discovery time in milliseconds")
    h3_time_ms: int = Field(..., ge=0, description="H3 chunk retrieval time in milliseconds")
    total_time_ms: int = Field(..., ge=0, description="Total search time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "h1_time_ms": 450,
                "h2_time_ms": 200,
                "h3_time_ms": 1800,
                "total_time_ms": 2450
            }
        }


class ThreeHopStatistics(BaseModel):
    """Statistics for three-hop search execution."""
    
    timings: ThreeHopTimings = Field(..., description="Timing information")
    rfqs_searched: int = Field(..., ge=0, description="Number of RFQs searched")
    offers_discovered: int = Field(..., ge=0, description="Number of offers discovered")
    chunks_evaluated: int = Field(..., ge=0, description="Number of chunks evaluated")
    cache_hits: int = Field(..., ge=0, description="Number of cache hits")
    parallel_execution: bool = Field(..., description="Whether parallel execution was used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timings": {
                    "h1_time_ms": 450,
                    "h2_time_ms": 200,
                    "h3_time_ms": 1800,
                    "total_time_ms": 2450
                },
                "rfqs_searched": 1500,
                "offers_discovered": 25,
                "chunks_evaluated": 75,
                "cache_hits": 2,
                "parallel_execution": True
            }
        }


class ThreeHopResults(BaseModel):
    """Container for three-hop search results."""

    h1_anchors: List[H1AnchorResult] = Field(..., description="H1 RFQ anchor results")
    h2_offers: List[H2OfferResult] = Field(..., description="H2 linked offer results")
    h3_chunks: List[H3ChunkResult] = Field(..., description="H3 chunk retrieval results")

    class Config:
        json_schema_extra = {
            "example": {
                "h1_anchors": [
                    {
                        "document_id": "123e4567-e89b-12d3-a456-426614174000",
                        "title": "Data Analytics Platform RFQ",
                        "score": 0.95,
                        "linked_offers_count": 5,
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                ],
                "h2_offers": [
                    {
                        "document_id": "456e7890-e89b-12d3-a456-426614174001",
                        "title": "Analytics Platform Technical Offer",
                        "link_confidence": 0.85,
                        "chunks_found": 3,
                        "source_rfq_id": "123e4567-e89b-12d3-a456-426614174000",
                        "link_type": "automatic"
                    }
                ],
                "h3_chunks": [
                    {
                        "chunk_id": "789e0123-e89b-12d3-a456-426614174002",
                        "content": "Our real-time analytics platform processes data streams...",
                        "score": 0.92,
                        "source_path": "Data Analytics Platform RFQ → Technical Offer → Analytics Architecture",
                        "source_document_id": "456e7890-e89b-12d3-a456-426614174001",
                        "source_document_title": "Analytics Platform Technical Offer"
                    }
                ]
            }
        }


class ThreeHopSearchResponse(BaseModel):
    """Three-hop search response model."""

    query: str = Field(..., description="Original search query")
    three_hop_results: ThreeHopResults = Field(..., description="Three-hop search results")
    statistics: ThreeHopStatistics = Field(..., description="Search execution statistics")
    explanation: Optional[str] = Field(None, description="Detailed explanation of search path")
    config_used: ThreeHopConfig = Field(..., description="Configuration used for search")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "data processing requirements for real-time analytics",
                "three_hop_results": {
                    "h1_anchors": [
                        {
                            "document_id": "123e4567-e89b-12d3-a456-426614174000",
                            "title": "Data Analytics Platform RFQ",
                            "score": 0.95,
                            "linked_offers_count": 5,
                            "created_at": "2024-01-15T10:30:00Z"
                        }
                    ],
                    "h2_offers": [
                        {
                            "document_id": "456e7890-e89b-12d3-a456-426614174001",
                            "title": "Analytics Platform Technical Offer",
                            "link_confidence": 0.85,
                            "chunks_found": 3,
                            "source_rfq_id": "123e4567-e89b-12d3-a456-426614174000",
                            "link_type": "automatic"
                        }
                    ],
                    "h3_chunks": [
                        {
                            "chunk_id": "789e0123-e89b-12d3-a456-426614174002",
                            "content": "Our real-time analytics platform processes data streams with sub-second latency...",
                            "score": 0.92,
                            "source_path": "Data Analytics Platform RFQ → Technical Offer → Analytics Architecture",
                            "source_document_id": "456e7890-e89b-12d3-a456-426614174001",
                            "source_document_title": "Analytics Platform Technical Offer"
                        }
                    ]
                },
                "statistics": {
                    "timings": {
                        "h1_time_ms": 450,
                        "h2_time_ms": 200,
                        "h3_time_ms": 1800,
                        "total_time_ms": 2450
                    },
                    "rfqs_searched": 1500,
                    "offers_discovered": 25,
                    "chunks_evaluated": 75,
                    "cache_hits": 2,
                    "parallel_execution": True
                },
                "explanation": "Found 1 relevant RFQ, discovered 1 linked offer, retrieved 1 highly relevant chunk",
                "config_used": {
                    "h1_max_results": 10,
                    "h2_min_confidence": 0.6,
                    "h3_max_chunks_per_offer": 3
                }
            }
        }
