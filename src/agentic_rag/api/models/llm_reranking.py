"""
API Models for LLM Reranking

This module defines the Pydantic models for LLM-based reranking API endpoints.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field

from agentic_rag.services.llm_reranking import (
    RerankingConfig, RerankingStrategy, ScoringWeights,
    RerankedResult, RerankingResponse, LLMScore
)
from agentic_rag.services.llm_client import LLMModel


class RerankingRequest(BaseModel):
    """Request model for LLM reranking."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Original search query")
    result_ids: List[UUID] = Field(..., min_items=1, max_items=50, description="Result IDs to rerank")
    config: Optional[RerankingConfig] = Field(None, description="Reranking configuration")
    explain_results: bool = Field(True, description="Include explanations in results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "requirements for data processing in procurement",
                "result_ids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "123e4567-e89b-12d3-a456-426614174001"
                ],
                "config": {
                    "model": "gpt-4-turbo-preview",
                    "strategy": "llm_with_fallback",
                    "max_results_to_rerank": 20,
                    "batch_size": 10,
                    "timeout_seconds": 30,
                    "enable_caching": True
                },
                "explain_results": True
            }
        }


class RerankingConfigRequest(BaseModel):
    """Request model for reranking configuration."""
    
    model: LLMModel = Field(LLMModel.GPT_4_TURBO, description="LLM model to use")
    strategy: RerankingStrategy = Field(RerankingStrategy.LLM_WITH_FALLBACK, description="Reranking strategy")
    max_results_to_rerank: int = Field(20, ge=1, le=50, description="Maximum results to rerank")
    batch_size: int = Field(10, ge=1, le=20, description="Batch size for processing")
    timeout_seconds: int = Field(30, ge=5, le=60, description="Timeout for reranking")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="LLM temperature")
    enable_caching: bool = Field(True, description="Enable result caching")
    cache_ttl_hours: int = Field(24, ge=1, le=168, description="Cache TTL in hours")
    scoring_weights: Optional[ScoringWeights] = Field(None, description="Custom scoring weights")
    enable_explanations: bool = Field(True, description="Include explanations in results")


class RerankingPreset(str, Enum):
    """Predefined reranking presets."""
    HIGH_PRECISION = "high_precision"
    BALANCED = "balanced"
    FAST = "fast"
    COMPREHENSIVE = "comprehensive"


class RerankingHealthStatus(BaseModel):
    """Health status for reranking service."""
    
    status: str = Field(..., description="Overall health status")
    llm_service_status: str = Field(..., description="LLM service status")
    circuit_breaker_state: str = Field(..., description="Circuit breaker state")
    cache_size: int = Field(..., description="Current cache size")
    recent_error_rate: float = Field(..., description="Recent error rate percentage")
    average_response_time_ms: float = Field(..., description="Average response time")


class RerankingStatistics(BaseModel):
    """Statistics for reranking service."""
    
    total_requests: int = Field(..., description="Total reranking requests")
    successful_requests: int = Field(..., description="Successful reranking requests")
    failed_requests: int = Field(..., description="Failed reranking requests")
    fallback_used: int = Field(..., description="Times fallback was used")
    cache_hits: int = Field(..., description="Cache hits")
    cache_misses: int = Field(..., description="Cache misses")
    average_reranking_time_ms: float = Field(..., description="Average reranking time")
    llm_calls_made: int = Field(..., description="Total LLM API calls made")
    total_results_reranked: int = Field(..., description="Total results reranked")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    success_rate: float = Field(..., description="Success rate percentage")


class RerankingCostAnalysis(BaseModel):
    """Cost analysis for reranking operations."""
    
    total_cost_usd: float = Field(..., description="Total cost in USD")
    cost_per_request: float = Field(..., description="Average cost per request")
    cost_by_model: Dict[str, float] = Field(..., description="Cost breakdown by model")
    tokens_used: int = Field(..., description="Total tokens used")
    estimated_monthly_cost: float = Field(..., description="Estimated monthly cost")
    cost_savings_from_cache: float = Field(..., description="Cost savings from caching")


class RerankingBenchmark(BaseModel):
    """Benchmark results for reranking quality."""
    
    query: str = Field(..., description="Test query")
    original_ranking: List[int] = Field(..., description="Original result rankings")
    reranked_ranking: List[int] = Field(..., description="Reranked result rankings")
    improvement_score: float = Field(..., description="Ranking improvement score")
    relevance_improvement: float = Field(..., description="Relevance improvement percentage")
    user_satisfaction_score: float = Field(..., description="User satisfaction score")


class RerankingExplanation(BaseModel):
    """Detailed explanation of reranking decisions."""
    
    query_analysis: str = Field(..., description="Analysis of the query")
    ranking_rationale: str = Field(..., description="Overall ranking rationale")
    top_factors: List[str] = Field(..., description="Top factors influencing ranking")
    result_explanations: List[Dict[str, Any]] = Field(..., description="Per-result explanations")
    confidence_level: float = Field(..., description="Confidence in reranking quality")


class RerankingConfigPreset(BaseModel):
    """Predefined reranking configuration preset."""
    
    name: str = Field(..., description="Preset name")
    description: str = Field(..., description="Preset description")
    config: RerankingConfig = Field(..., description="Configuration settings")
    use_cases: List[str] = Field(..., description="Recommended use cases")
    performance_characteristics: Dict[str, str] = Field(..., description="Performance characteristics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "high_precision",
                "description": "Optimized for maximum accuracy with detailed analysis",
                "config": {
                    "model": "gpt-4-turbo-preview",
                    "strategy": "llm_only",
                    "max_results_to_rerank": 10,
                    "batch_size": 5,
                    "timeout_seconds": 45,
                    "temperature": 0.05,
                    "enable_caching": True,
                    "cache_ttl_hours": 48,
                    "enable_explanations": True
                },
                "use_cases": [
                    "Critical procurement decisions",
                    "High-value contract analysis",
                    "Compliance-sensitive searches"
                ],
                "performance_characteristics": {
                    "accuracy": "Highest",
                    "speed": "Slower",
                    "cost": "Higher",
                    "reliability": "Maximum"
                }
            }
        }


class RerankingValidationResult(BaseModel):
    """Result of reranking configuration validation."""
    
    valid: bool = Field(..., description="Whether configuration is valid")
    warnings: List[str] = Field(..., description="Configuration warnings")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    estimated_cost_per_request: float = Field(..., description="Estimated cost per request")
    estimated_response_time_ms: int = Field(..., description="Estimated response time")
    risk_factors: List[str] = Field(..., description="Potential risk factors")


class RerankingPerformanceMetrics(BaseModel):
    """Performance metrics for reranking operations."""
    
    response_time_percentiles: Dict[str, float] = Field(..., description="Response time percentiles")
    throughput_requests_per_minute: float = Field(..., description="Throughput in requests per minute")
    error_rates: Dict[str, float] = Field(..., description="Error rates by type")
    resource_utilization: Dict[str, float] = Field(..., description="Resource utilization metrics")
    quality_metrics: Dict[str, float] = Field(..., description="Reranking quality metrics")
    cost_efficiency: Dict[str, float] = Field(..., description="Cost efficiency metrics")


class RerankingABTestResult(BaseModel):
    """A/B test results for reranking configurations."""
    
    test_name: str = Field(..., description="Test name")
    config_a: RerankingConfig = Field(..., description="Configuration A")
    config_b: RerankingConfig = Field(..., description="Configuration B")
    sample_size: int = Field(..., description="Sample size for test")
    metrics_a: Dict[str, float] = Field(..., description="Metrics for configuration A")
    metrics_b: Dict[str, float] = Field(..., description="Metrics for configuration B")
    statistical_significance: float = Field(..., description="Statistical significance")
    winner: str = Field(..., description="Winning configuration")
    confidence_interval: Dict[str, float] = Field(..., description="Confidence intervals")


class RerankingFeedback(BaseModel):
    """User feedback on reranking results."""
    
    query: str = Field(..., description="Original query")
    result_id: UUID = Field(..., description="Result identifier")
    feedback_type: str = Field(..., description="Type of feedback")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    comments: Optional[str] = Field(None, description="Additional comments")
    user_id: UUID = Field(..., description="User identifier")
    timestamp: float = Field(..., description="Feedback timestamp")


class RerankingOptimizationSuggestion(BaseModel):
    """Optimization suggestions for reranking configuration."""
    
    current_config: RerankingConfig = Field(..., description="Current configuration")
    suggested_config: RerankingConfig = Field(..., description="Suggested configuration")
    expected_improvements: Dict[str, str] = Field(..., description="Expected improvements")
    trade_offs: Dict[str, str] = Field(..., description="Trade-offs to consider")
    confidence_score: float = Field(..., description="Confidence in suggestion")
    implementation_effort: str = Field(..., description="Implementation effort level")
