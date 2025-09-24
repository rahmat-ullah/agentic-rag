"""
LLM Reranking Service

This module provides LLM-based reranking of search results using OpenAI models.
Includes procurement-optimized prompts, multi-criteria scoring, batch processing,
and fallback mechanisms.
"""

import asyncio
import time
import json
import math
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.llm_client import (
    get_llm_client_service, LLMRequest, LLMModel, LLMRequestType
)
from agentic_rag.services.vector_search import VectorSearchResult
from agentic_rag.services.result_ranker import RankedResult
from agentic_rag.services.reranking_models import (
    RerankingStrategy, ScoringCriteria, ScoringWeights, RerankingConfig,
    LLMScore, RerankedResult, RerankingResponse
)
from agentic_rag.services.procurement_prompts import (
    get_procurement_prompt_generator, QueryType
)

logger = structlog.get_logger(__name__)





class LLMRerankingService:
    """LLM-based reranking service."""
    
    def __init__(self):
        self._llm_client = get_llm_client_service()
        self._prompt_generator = get_procurement_prompt_generator()
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Circuit breaker state
        self._circuit_breaker = {
            "state": "closed",  # closed, open, half_open
            "failure_count": 0,
            "last_failure_time": 0,
            "failure_threshold": 5,
            "recovery_timeout": 300  # 5 minutes
        }
        
        # Statistics
        self._stats = {
            "total_reranking_requests": 0,
            "successful_reranking": 0,
            "failed_reranking": 0,
            "fallback_used": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_reranking_time_ms": 0.0,
            "llm_calls_made": 0,
            "total_results_reranked": 0
        }
        
        logger.info("LLM reranking service initialized")
    
    async def rerank_results(
        self,
        results: List[VectorSearchResult],
        query: str,
        config: RerankingConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> RerankingResponse:
        """
        Rerank search results using LLM.
        
        Args:
            results: Original search results to rerank
            query: Original search query
            config: Reranking configuration
            tenant_id: Tenant identifier
            user_id: User identifier
            
        Returns:
            Reranked results with scores and explanations
        """
        start_time = time.time()
        
        if not results:
            return RerankingResponse(
                results=[],
                total_results=0,
                reranking_time_ms=0,
                method_used="none",
                llm_calls_made=0,
                cache_hits=0,
                fallback_used=False
            )
        
        self._stats["total_reranking_requests"] += 1
        
        logger.info(
            f"Starting LLM reranking",
            query=query[:50],
            result_count=len(results),
            strategy=config.strategy,
            tenant_id=str(tenant_id)
        )
        
        try:
            # Limit results to rerank
            results_to_rerank = results[:config.max_results_to_rerank]
            
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                logger.warning("Circuit breaker is open, using fallback")
                return await self._fallback_reranking(results_to_rerank, query, config, start_time)
            
            # Try LLM reranking
            try:
                reranked_results = await self._llm_rerank(
                    results_to_rerank, query, config, tenant_id, user_id
                )
                
                # Reset circuit breaker on success
                self._reset_circuit_breaker()
                
                reranking_time_ms = int((time.time() - start_time) * 1000)
                self._stats["successful_reranking"] += 1
                self._update_average_reranking_time(reranking_time_ms)
                
                return reranked_results
                
            except Exception as e:
                logger.error(f"LLM reranking failed: {e}")
                self._record_circuit_breaker_failure()
                
                if config.strategy == RerankingStrategy.LLM_WITH_FALLBACK:
                    logger.info("Falling back to vector-based ranking")
                    return await self._fallback_reranking(results_to_rerank, query, config, start_time)
                else:
                    raise
                    
        except Exception as e:
            self._stats["failed_reranking"] += 1
            logger.error(f"Reranking failed completely: {e}")
            raise
    
    async def _llm_rerank(
        self,
        results: List[VectorSearchResult],
        query: str,
        config: RerankingConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> RerankingResponse:
        """Perform LLM-based reranking."""
        
        # Check cache first
        cache_key = self._generate_cache_key(query, results, config)
        cached_result = self._get_cached_result(cache_key, config)
        
        if cached_result:
            self._stats["cache_hits"] += 1
            logger.info("Using cached reranking result")
            return cached_result
        
        self._stats["cache_misses"] += 1
        
        # Process in batches with parallel processing if enabled
        batches = self._create_optimal_batches(results, config)
        all_reranked_results = []
        total_llm_calls = 0

        if config.enable_parallel_processing and len(batches) > 1:
            # Process batches in parallel
            logger.info(f"Processing {len(batches)} batches in parallel")

            batch_tasks = []
            for batch_idx, batch in enumerate(batches):
                task = self._process_batch_async(
                    batch, batch_idx, query, config, tenant_id, user_id
                )
                batch_tasks.append(task)

            # Wait for all batches to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Process results and handle exceptions
            for batch_idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch {batch_idx + 1} failed: {result}")
                    # Use fallback for failed batch
                    failed_batch = batches[batch_idx]
                    fallback_results = self._create_fallback_results(failed_batch)
                    all_reranked_results.extend(fallback_results)
                else:
                    batch_reranked, batch_llm_calls = result
                    all_reranked_results.extend(batch_reranked)
                    total_llm_calls += batch_llm_calls

        else:
            # Process batches sequentially
            logger.info(f"Processing {len(batches)} batches sequentially")

            for batch_idx, batch in enumerate(batches):
                try:
                    batch_reranked, batch_llm_calls = await self._process_batch_async(
                        batch, batch_idx, query, config, tenant_id, user_id
                    )
                    all_reranked_results.extend(batch_reranked)
                    total_llm_calls += batch_llm_calls

                except Exception as e:
                    logger.error(f"Batch {batch_idx + 1} failed: {e}")
                    # Use fallback for failed batch
                    fallback_results = self._create_fallback_results(batch)
                    all_reranked_results.extend(fallback_results)
        
        # Final ranking across all batches
        final_results = self._final_ranking(all_reranked_results, config)
        
        reranking_time_ms = int((time.time() - time.time()) * 1000)  # Will be corrected by caller
        
        response = RerankingResponse(
            results=final_results,
            total_results=len(results),
            reranking_time_ms=reranking_time_ms,
            method_used="llm",
            llm_calls_made=total_llm_calls,
            cache_hits=0,
            fallback_used=False
        )
        
        # Cache the result
        if config.enable_caching:
            self._cache_result(cache_key, response, config)
        
        return response
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type for prompt optimization."""
        return self._prompt_generator._detect_query_type(query)
    
    def _parse_llm_response(
        self,
        response_content: str,
        original_results: List[VectorSearchResult],
        config: RerankingConfig
    ) -> List[RerankedResult]:
        """Parse LLM response and create reranked results."""
        
        try:
            # Extract JSON from response
            response_data = json.loads(response_content)
            rankings = response_data.get("rankings", [])
            
            reranked_results = []
            
            for ranking in rankings:
                result_id = ranking.get("result_id", 1) - 1  # Convert to 0-based index
                
                if 0 <= result_id < len(original_results):
                    original_result = original_results[result_id]
                    
                    # Create LLM score
                    llm_score = LLMScore(
                        relevance=ranking.get("relevance", 5.0),
                        specificity=ranking.get("specificity", 5.0),
                        completeness=ranking.get("completeness", 5.0),
                        authority=ranking.get("authority", 5.0),
                        composite_score=self._calculate_composite_score(ranking, config),
                        explanation=ranking.get("explanation", "No explanation provided")
                    )
                    
                    # Create reranked result
                    reranked_result = RerankedResult(
                        document_id=original_result.document_id,
                        chunk_id=original_result.chunk_id,
                        content=original_result.content,
                        metadata=original_result.metadata,
                        original_score=original_result.similarity_score,
                        llm_score=llm_score,
                        final_score=llm_score.composite_score,
                        rank_position=0,  # Will be set later
                        reranking_method="llm"
                    )
                    
                    reranked_results.append(reranked_result)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback to original ranking
            return self._create_fallback_results(original_results)
    
    def _calculate_composite_score(
        self,
        scores: Dict[str, float],
        config: RerankingConfig
    ) -> float:
        """Calculate composite score from individual criteria scores."""
        
        relevance = scores.get("relevance", 5.0)
        specificity = scores.get("specificity", 5.0)
        completeness = scores.get("completeness", 5.0)
        authority = scores.get("authority", 5.0)
        
        composite = (
            relevance * config.scoring_weights.relevance +
            specificity * config.scoring_weights.specificity +
            completeness * config.scoring_weights.completeness +
            authority * config.scoring_weights.authority
        )
        
        return min(max(composite, 0.0), 10.0)
    
    def _final_ranking(
        self,
        results: List[RerankedResult],
        config: RerankingConfig
    ) -> List[RerankedResult]:
        """Apply final ranking to all results."""
        
        # Sort by final score (descending)
        sorted_results = sorted(results, key=lambda x: x.final_score, reverse=True)
        
        # Assign rank positions
        for idx, result in enumerate(sorted_results, 1):
            result.rank_position = idx
        
        return sorted_results
    
    async def _fallback_reranking(
        self,
        results: List[VectorSearchResult],
        query: str,
        config: RerankingConfig,
        start_time: float
    ) -> RerankingResponse:
        """Enhanced fallback to vector-based ranking with heuristic improvements."""

        self._stats["fallback_used"] += 1

        logger.info("Using enhanced fallback reranking with heuristic scoring")

        # Use multi-criteria scoring service for enhanced fallback
        try:
            from agentic_rag.services.multi_criteria_scoring import (
                get_multi_criteria_scoring_service, ScoringContext, ScoringMethod
            )

            scoring_service = get_multi_criteria_scoring_service()

            # Create scoring context
            context = ScoringContext(
                query=query,
                query_type="general",
                document_types=[r.metadata.get("document_type", "unknown") for r in results]
            )

            # Calculate enhanced scores for each result
            enhanced_results = []
            for result in results:
                try:
                    detailed_score = scoring_service.calculate_detailed_score(
                        result=result,
                        context=context,
                        weights=config.scoring_weights,
                        method=ScoringMethod.HEURISTIC
                    )

                    # Create LLM score from detailed score
                    llm_score = LLMScore(
                        relevance=detailed_score.relevance,
                        specificity=detailed_score.specificity,
                        completeness=detailed_score.completeness,
                        authority=detailed_score.authority,
                        composite_score=detailed_score.composite_score,
                        explanation=detailed_score.explanation
                    )

                    reranked_result = RerankedResult(
                        document_id=result.document_id,
                        chunk_id=result.chunk_id,
                        content=result.content,
                        metadata=result.metadata,
                        original_score=result.similarity_score,
                        llm_score=llm_score,
                        final_score=detailed_score.composite_score,
                        rank_position=0,  # Will be set after sorting
                        reranking_method="enhanced_heuristic_fallback"
                    )

                    enhanced_results.append(reranked_result)

                except Exception as e:
                    logger.warning(f"Heuristic scoring failed for result, using vector score: {e}")

                    # Basic fallback to vector score
                    reranked_result = RerankedResult(
                        document_id=result.document_id,
                        chunk_id=result.chunk_id,
                        content=result.content,
                        metadata=result.metadata,
                        original_score=result.similarity_score,
                        llm_score=None,
                        final_score=result.similarity_score,
                        rank_position=0,
                        reranking_method="vector_fallback"
                    )

                    enhanced_results.append(reranked_result)

            # Sort by final score and assign ranks
            enhanced_results.sort(key=lambda x: x.final_score, reverse=True)
            for idx, result in enumerate(enhanced_results, 1):
                result.rank_position = idx

            method_used = "enhanced_heuristic_fallback"

        except Exception as e:
            logger.error(f"Enhanced fallback failed, using basic vector fallback: {e}")

            # Basic vector fallback
            enhanced_results = []
            for idx, result in enumerate(results, 1):
                reranked_result = RerankedResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    content=result.content,
                    metadata=result.metadata,
                    original_score=result.similarity_score,
                    llm_score=None,
                    final_score=result.similarity_score,
                    rank_position=idx,
                    reranking_method="vector_fallback"
                )
                enhanced_results.append(reranked_result)

            method_used = "vector_fallback"

        reranking_time_ms = int((time.time() - start_time) * 1000)

        return RerankingResponse(
            results=enhanced_results,
            total_results=len(results),
            reranking_time_ms=reranking_time_ms,
            method_used=method_used,
            llm_calls_made=0,
            cache_hits=0,
            fallback_used=True
        )
    
    def _create_fallback_results(
        self,
        results: List[VectorSearchResult]
    ) -> List[RerankedResult]:
        """Create fallback results when LLM parsing fails."""
        
        fallback_results = []
        for result in results:
            reranked_result = RerankedResult(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                content=result.content,
                metadata=result.metadata,
                original_score=result.similarity_score,
                llm_score=None,
                final_score=result.similarity_score,
                rank_position=0,
                reranking_method="vector_fallback"
            )
            fallback_results.append(reranked_result)
        
        return fallback_results

    def _create_optimal_batches(
        self,
        results: List[VectorSearchResult],
        config: RerankingConfig
    ) -> List[List[VectorSearchResult]]:
        """Create optimally sized batches for processing."""

        # Determine optimal batch size based on configuration and result count
        optimal_batch_size = self._calculate_optimal_batch_size(len(results), config)

        batches = []
        for i in range(0, len(results), optimal_batch_size):
            batch = results[i:i + optimal_batch_size]
            batches.append(batch)

        logger.info(f"Created {len(batches)} batches with size {optimal_batch_size}")
        return batches

    def _calculate_optimal_batch_size(self, total_results: int, config: RerankingConfig) -> int:
        """Calculate optimal batch size based on results and configuration."""

        base_batch_size = config.batch_size

        # Adjust batch size based on total results
        if total_results <= 5:
            return total_results  # Single batch for small result sets
        elif total_results <= 10:
            return min(base_batch_size, 5)  # Smaller batches for medium sets
        else:
            # For larger sets, use configured batch size but ensure reasonable distribution
            num_batches = math.ceil(total_results / base_batch_size)
            optimal_size = math.ceil(total_results / num_batches)
            return min(optimal_size, base_batch_size)

    async def _process_batch_async(
        self,
        batch: List[VectorSearchResult],
        batch_idx: int,
        query: str,
        config: RerankingConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> Tuple[List[RerankedResult], int]:
        """Process a single batch asynchronously."""

        logger.info(f"Processing batch {batch_idx + 1} with {len(batch)} results")

        try:
            # Generate procurement-optimized reranking prompt
            prompt = self._prompt_generator.generate_reranking_prompt(
                query=query,
                results=batch,
                config=config,
                include_few_shot=True
            )

            # Make LLM request with timeout
            llm_request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                model=config.model,
                temperature=config.temperature,
                max_tokens=2000,
                request_type=LLMRequestType.RERANKING,
                tenant_id=str(tenant_id),
                user_id=str(user_id)
            )

            # Add timeout handling
            try:
                llm_response = await asyncio.wait_for(
                    self._llm_client.generate_completion(llm_request),
                    timeout=config.timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"Batch {batch_idx + 1} timed out")
                raise Exception(f"Batch processing timed out after {config.timeout_seconds} seconds")

            self._stats["llm_calls_made"] += 1

            # Parse LLM response
            batch_reranked = self._parse_llm_response(
                llm_response.content, batch, config
            )

            logger.info(f"Batch {batch_idx + 1} processed successfully")
            return batch_reranked, 1

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} processing failed: {e}")
            raise

    def _generate_cache_key(
        self,
        query: str,
        results: List[VectorSearchResult],
        config: RerankingConfig
    ) -> str:
        """Generate cache key for reranking results."""
        import hashlib

        # Create a hash of query, result IDs, and config
        content = f"{query}_{len(results)}_{config.model}_{config.scoring_weights.relevance}"
        for result in results[:5]:  # Use first 5 results for key
            content += f"_{result.chunk_id}"

        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(
        self,
        cache_key: str,
        config: RerankingConfig
    ) -> Optional[RerankingResponse]:
        """Get cached reranking result if valid."""
        if not config.enable_caching or cache_key not in self._cache:
            return None

        # Check if cache entry is still valid
        cache_time = self._cache_timestamps.get(cache_key, 0)
        cache_ttl_seconds = config.cache_ttl_hours * 3600

        if time.time() - cache_time > cache_ttl_seconds:
            # Cache expired
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._cache[cache_key]

    def _cache_result(
        self,
        cache_key: str,
        result: RerankingResponse,
        config: RerankingConfig
    ):
        """Cache reranking result."""
        if config.enable_caching:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()

            # Limit cache size
            if len(self._cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._cache_timestamps.keys(),
                    key=lambda k: self._cache_timestamps[k]
                )[:100]

                for key in oldest_keys:
                    del self._cache[key]
                    del self._cache_timestamps[key]

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker["state"] == "open":
            # Check if recovery timeout has passed
            if (time.time() - self._circuit_breaker["last_failure_time"] >
                self._circuit_breaker["recovery_timeout"]):
                self._circuit_breaker["state"] = "half_open"
                logger.info("Circuit breaker moved to half-open state")
                return False
            return True

        return False

    def _record_circuit_breaker_failure(self):
        """Record a failure for circuit breaker."""
        self._circuit_breaker["failure_count"] += 1
        self._circuit_breaker["last_failure_time"] = time.time()

        if (self._circuit_breaker["failure_count"] >=
            self._circuit_breaker["failure_threshold"]):
            self._circuit_breaker["state"] = "open"
            logger.warning("Circuit breaker opened due to failures")

    def _reset_circuit_breaker(self):
        """Reset circuit breaker on successful operation."""
        if self._circuit_breaker["state"] in ["half_open", "open"]:
            self._circuit_breaker["state"] = "closed"
            self._circuit_breaker["failure_count"] = 0
            logger.info("Circuit breaker reset to closed state")

    def _update_average_reranking_time(self, reranking_time_ms: int):
        """Update average reranking time."""
        current_avg = self._stats["average_reranking_time_ms"]
        successful_requests = self._stats["successful_reranking"]

        if successful_requests > 1:
            self._stats["average_reranking_time_ms"] = (
                (current_avg * (successful_requests - 1) + reranking_time_ms) / successful_requests
            )
        else:
            self._stats["average_reranking_time_ms"] = reranking_time_ms

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for reranking service."""

        try:
            llm_health = await self._llm_client.health_check()

            # Calculate service health metrics
            total_requests = self._stats["total_reranking_requests"]
            successful_requests = self._stats["successful_reranking"]

            success_rate = (successful_requests / max(total_requests, 1)) * 100
            fallback_rate = (self._stats["fallback_used"] / max(total_requests, 1)) * 100
            cache_hit_rate = (self._stats["cache_hits"] / max(self._stats["cache_hits"] + self._stats["cache_misses"], 1)) * 100

            # Determine overall health status
            if llm_health.status == "healthy" and success_rate >= 95 and fallback_rate <= 10:
                overall_status = "healthy"
            elif llm_health.status in ["healthy", "degraded"] and success_rate >= 80:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"

            # Performance metrics
            performance_metrics = {
                "success_rate": success_rate,
                "fallback_rate": fallback_rate,
                "cache_hit_rate": cache_hit_rate,
                "average_response_time_ms": self._stats["average_reranking_time_ms"],
                "circuit_breaker_state": self._circuit_breaker["state"],
                "cache_efficiency": len(self._cache) / max(len(self._cache) + 100, 1)  # Rough efficiency metric
            }

            # Service capabilities
            capabilities = {
                "llm_reranking": llm_health.status == "healthy",
                "heuristic_fallback": True,
                "batch_processing": True,
                "parallel_processing": True,
                "caching": True,
                "circuit_breaker": True
            }

            return {
                "status": overall_status,
                "llm_service": llm_health.dict(),
                "circuit_breaker": self._circuit_breaker.copy(),
                "cache_size": len(self._cache),
                "statistics": self._stats.copy(),
                "performance_metrics": performance_metrics,
                "capabilities": capabilities,
                "recommendations": self._generate_health_recommendations(
                    success_rate, fallback_rate, cache_hit_rate, llm_health.status
                )
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "llm_service": {"status": "unknown"},
                "circuit_breaker": self._circuit_breaker.copy(),
                "cache_size": len(self._cache),
                "statistics": self._stats.copy()
            }

    def _generate_health_recommendations(
        self,
        success_rate: float,
        fallback_rate: float,
        cache_hit_rate: float,
        llm_status: str
    ) -> List[str]:
        """Generate health recommendations based on metrics."""

        recommendations = []

        if success_rate < 90:
            recommendations.append("Consider investigating LLM service issues or adjusting timeout settings")

        if fallback_rate > 20:
            recommendations.append("High fallback rate detected - check LLM service stability")

        if cache_hit_rate < 30:
            recommendations.append("Low cache hit rate - consider adjusting cache TTL or query patterns")

        if llm_status != "healthy":
            recommendations.append("LLM service is not healthy - monitor for service restoration")

        if self._circuit_breaker["state"] != "closed":
            recommendations.append("Circuit breaker is open - LLM service may be experiencing issues")

        if len(self._cache) > 1000:
            recommendations.append("Cache size is large - consider implementing cache cleanup")

        if not recommendations:
            recommendations.append("Service is operating optimally")

        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """Get reranking service statistics."""
        return self._stats.copy()


# Singleton instance
_llm_reranking_service: Optional[LLMRerankingService] = None


def get_llm_reranking_service() -> LLMRerankingService:
    """Get the singleton LLM reranking service instance."""
    global _llm_reranking_service
    if _llm_reranking_service is None:
        _llm_reranking_service = LLMRerankingService()
    return _llm_reranking_service
