"""
Shared Models for LLM Reranking

This module contains shared models and enums used across reranking services
to avoid circular imports.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, Field

from agentic_rag.services.llm_client import LLMModel


class RerankingStrategy(str, Enum):
    """Reranking strategies."""
    LLM_ONLY = "llm_only"
    LLM_WITH_FALLBACK = "llm_with_fallback"
    HYBRID = "hybrid"


class ScoringCriteria(str, Enum):
    """Scoring criteria for reranking."""
    RELEVANCE = "relevance"
    SPECIFICITY = "specificity"
    COMPLETENESS = "completeness"
    AUTHORITY = "authority"


@dataclass
class ScoringWeights:
    """Weights for different scoring criteria."""
    relevance: float = 0.40
    specificity: float = 0.25
    completeness: float = 0.20
    authority: float = 0.15


class RerankingConfig(BaseModel):
    """Configuration for LLM reranking."""
    
    model: LLMModel = Field(LLMModel.GPT_4_TURBO, description="LLM model to use")
    strategy: RerankingStrategy = Field(RerankingStrategy.LLM_WITH_FALLBACK, description="Reranking strategy")
    max_results_to_rerank: int = Field(20, ge=1, le=50, description="Maximum results to rerank")
    batch_size: int = Field(10, ge=1, le=20, description="Batch size for processing")
    timeout_seconds: int = Field(30, ge=5, le=60, description="Timeout for reranking")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="LLM temperature")
    enable_caching: bool = Field(True, description="Enable result caching")
    cache_ttl_hours: int = Field(24, ge=1, le=168, description="Cache TTL in hours")
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights, description="Scoring weights")
    enable_explanations: bool = Field(True, description="Include explanations in results")
    enable_parallel_processing: bool = Field(True, description="Enable parallel batch processing")


class LLMScore(BaseModel):
    """Individual LLM score for a result."""
    
    relevance: float = Field(..., ge=0.0, le=10.0, description="Relevance score (0-10)")
    specificity: float = Field(..., ge=0.0, le=10.0, description="Specificity score (0-10)")
    completeness: float = Field(..., ge=0.0, le=10.0, description="Completeness score (0-10)")
    authority: float = Field(..., ge=0.0, le=10.0, description="Authority score (0-10)")
    composite_score: float = Field(..., ge=0.0, le=10.0, description="Composite score")
    explanation: str = Field(..., description="Explanation of the score")


class RerankedResult(BaseModel):
    """A reranked search result."""
    
    document_id: UUID = Field(..., description="Document identifier")
    chunk_id: UUID = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(..., description="Result metadata")
    original_score: float = Field(..., description="Original vector similarity score")
    llm_score: Optional[LLMScore] = Field(None, description="LLM-generated score")
    final_score: float = Field(..., description="Final reranked score")
    rank_position: int = Field(..., description="Final rank position")
    reranking_method: str = Field(..., description="Method used for reranking")


class RerankingResponse(BaseModel):
    """Response from reranking operation."""
    
    results: List[RerankedResult] = Field(..., description="Reranked results")
    total_results: int = Field(..., description="Total number of results processed")
    reranking_time_ms: int = Field(..., description="Time taken for reranking")
    method_used: str = Field(..., description="Reranking method used")
    llm_calls_made: int = Field(..., description="Number of LLM API calls made")
    cache_hits: int = Field(..., description="Number of cache hits")
    fallback_used: bool = Field(..., description="Whether fallback was used")
