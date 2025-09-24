"""
Result Ranking and Filtering Service

This module provides advanced result ranking and filtering capabilities
for search results including relevance scoring, metadata-based filtering,
and result deduplication.
"""

import math
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.vector_store import VectorSearchResult
from agentic_rag.models.database import DocumentKind

logger = structlog.get_logger(__name__)


class RankingStrategy(str, Enum):
    """Ranking strategies for search results."""
    SIMILARITY = "similarity"
    HYBRID = "hybrid"
    RECENCY = "recency"
    POPULARITY = "popularity"
    CUSTOM = "custom"


class FilterType(str, Enum):
    """Types of filters that can be applied."""
    DOCUMENT_TYPE = "document_type"
    DATE_RANGE = "date_range"
    METADATA = "metadata"
    SCORE_THRESHOLD = "score_threshold"
    CONTENT_LENGTH = "content_length"
    SECTION_PATH = "section_path"


@dataclass
class RankingWeights:
    """Weights for different ranking factors."""
    similarity_weight: float = 0.7
    recency_weight: float = 0.1
    popularity_weight: float = 0.1
    metadata_weight: float = 0.1


class RankingConfiguration(BaseModel):
    """Configuration for result ranking."""
    
    strategy: RankingStrategy = Field(default=RankingStrategy.HYBRID)
    weights: RankingWeights = Field(default_factory=RankingWeights)
    boost_exact_matches: bool = Field(default=True)
    boost_title_matches: bool = Field(default=True)
    boost_recent_documents: bool = Field(default=True)
    recency_decay_days: int = Field(default=30, ge=1)
    diversity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_results_per_document: int = Field(default=3, ge=1)


class FilterConfiguration(BaseModel):
    """Configuration for result filtering."""
    
    document_types: Optional[List[DocumentKind]] = Field(None)
    date_from: Optional[datetime] = Field(None)
    date_to: Optional[datetime] = Field(None)
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_content_length: Optional[int] = Field(None, ge=1)
    max_content_length: Optional[int] = Field(None, ge=1)
    section_paths: Optional[List[str]] = Field(None)
    metadata_filters: Optional[Dict[str, Any]] = Field(None)
    exclude_tables: bool = Field(default=False)
    exclude_short_chunks: bool = Field(default=True)
    min_token_count: int = Field(default=10, ge=1)


class RankedResult(BaseModel):
    """A search result with ranking information."""
    
    result: VectorSearchResult = Field(..., description="Original search result")
    final_score: float = Field(..., description="Final ranking score")
    similarity_score: float = Field(..., description="Vector similarity score")
    recency_score: float = Field(..., description="Recency boost score")
    popularity_score: float = Field(..., description="Popularity boost score")
    metadata_score: float = Field(..., description="Metadata relevance score")
    boost_factors: Dict[str, float] = Field(default_factory=dict, description="Applied boost factors")
    rank_position: int = Field(..., description="Final rank position")


class ResultRanker:
    """Service for ranking and filtering search results."""
    
    def __init__(self):
        self._default_ranking_config = RankingConfiguration()
        self._default_filter_config = FilterConfiguration()
        
        # Popularity tracking (in production, this would be in a database)
        self._document_popularity: Dict[str, int] = {}
        self._chunk_popularity: Dict[str, int] = {}
        
        # Statistics
        self._stats = {
            "total_rankings": 0,
            "total_filtered": 0,
            "average_ranking_time_ms": 0.0,
            "boost_applications": {
                "exact_match": 0,
                "title_match": 0,
                "recency": 0,
                "popularity": 0
            }
        }
        
        logger.info("Result ranker initialized")
    
    async def rank_and_filter_results(
        self,
        results: List[VectorSearchResult],
        query: str,
        ranking_config: Optional[RankingConfiguration] = None,
        filter_config: Optional[FilterConfiguration] = None
    ) -> List[RankedResult]:
        """
        Rank and filter search results.
        
        Args:
            results: Raw search results
            query: Original search query
            ranking_config: Ranking configuration
            filter_config: Filter configuration
            
        Returns:
            List of ranked and filtered results
        """
        start_time = time.time()
        
        if not results:
            return []
        
        # Use provided configs or defaults
        ranking_config = ranking_config or self._default_ranking_config
        filter_config = filter_config or self._default_filter_config
        
        logger.info(
            f"Ranking and filtering {len(results)} results",
            strategy=ranking_config.strategy.value,
            query_length=len(query)
        )
        
        try:
            self._stats["total_rankings"] += 1
            
            # Step 1: Apply filters
            filtered_results = self._apply_filters(results, filter_config)
            self._stats["total_filtered"] += len(results) - len(filtered_results)
            
            if not filtered_results:
                return []
            
            # Step 2: Calculate ranking scores
            ranked_results = self._calculate_ranking_scores(
                filtered_results, 
                query, 
                ranking_config
            )
            
            # Step 3: Apply deduplication
            deduplicated_results = self._apply_deduplication(
                ranked_results, 
                ranking_config
            )
            
            # Step 4: Final sorting and ranking
            final_results = self._finalize_ranking(deduplicated_results)
            
            ranking_time_ms = int((time.time() - start_time) * 1000)
            self._update_average_ranking_time(ranking_time_ms)
            
            logger.info(
                f"Ranking completed",
                original_count=len(results),
                filtered_count=len(filtered_results),
                final_count=len(final_results),
                ranking_time_ms=ranking_time_ms
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ranking and filtering failed: {e}", exc_info=True)
            raise
    
    def _apply_filters(
        self, 
        results: List[VectorSearchResult], 
        config: FilterConfiguration
    ) -> List[VectorSearchResult]:
        """Apply filtering to search results."""
        filtered = results
        
        # Filter by document types
        if config.document_types:
            doc_type_values = [dt.value for dt in config.document_types]
            filtered = [
                r for r in filtered 
                if r.metadata and r.metadata.document_kind in doc_type_values
            ]
        
        # Filter by date range
        if config.date_from or config.date_to:
            filtered = self._filter_by_date_range(filtered, config.date_from, config.date_to)
        
        # Filter by score threshold
        if config.min_score is not None:
            # Convert distance to similarity score (1 - distance)
            filtered = [
                r for r in filtered 
                if (1.0 - r.distance) >= config.min_score
            ]
        
        if config.max_score is not None:
            filtered = [
                r for r in filtered 
                if (1.0 - r.distance) <= config.max_score
            ]
        
        # Filter by content length
        if config.min_content_length is not None:
            filtered = [
                r for r in filtered 
                if len(r.document) >= config.min_content_length
            ]
        
        if config.max_content_length is not None:
            filtered = [
                r for r in filtered 
                if len(r.document) <= config.max_content_length
            ]
        
        # Filter by section paths
        if config.section_paths:
            filtered = [
                r for r in filtered 
                if r.metadata and any(
                    path in (r.metadata.section_path or []) 
                    for path in config.section_paths
                )
            ]
        
        # Exclude tables if requested
        if config.exclude_tables:
            filtered = [
                r for r in filtered 
                if not (r.metadata and r.metadata.is_table)
            ]
        
        # Exclude short chunks
        if config.exclude_short_chunks:
            filtered = [
                r for r in filtered 
                if not r.metadata or (r.metadata.token_count or 0) >= config.min_token_count
            ]
        
        # Apply metadata filters
        if config.metadata_filters:
            filtered = self._apply_metadata_filters(filtered, config.metadata_filters)
        
        return filtered
    
    def _filter_by_date_range(
        self, 
        results: List[VectorSearchResult], 
        date_from: Optional[datetime], 
        date_to: Optional[datetime]
    ) -> List[VectorSearchResult]:
        """Filter results by date range."""
        if not date_from and not date_to:
            return results
        
        filtered = []
        for result in results:
            if not result.metadata or not result.metadata.created_at:
                continue
            
            created_at = result.metadata.created_at
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    continue
            
            if date_from and created_at < date_from:
                continue
            
            if date_to and created_at > date_to:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def _apply_metadata_filters(
        self, 
        results: List[VectorSearchResult], 
        metadata_filters: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Apply custom metadata filters."""
        filtered = []
        
        for result in results:
            if not result.metadata:
                continue
            
            matches_all_filters = True
            
            for key, expected_value in metadata_filters.items():
                # Get actual value from metadata
                actual_value = getattr(result.metadata, key, None)
                
                # Handle different comparison types
                if isinstance(expected_value, list):
                    if actual_value not in expected_value:
                        matches_all_filters = False
                        break
                elif isinstance(expected_value, dict):
                    # Range comparison
                    if 'min' in expected_value and actual_value < expected_value['min']:
                        matches_all_filters = False
                        break
                    if 'max' in expected_value and actual_value > expected_value['max']:
                        matches_all_filters = False
                        break
                else:
                    if actual_value != expected_value:
                        matches_all_filters = False
                        break
            
            if matches_all_filters:
                filtered.append(result)
        
        return filtered

    def _calculate_ranking_scores(
        self,
        results: List[VectorSearchResult],
        query: str,
        config: RankingConfiguration
    ) -> List[RankedResult]:
        """Calculate ranking scores for all results."""
        ranked_results = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for result in results:
            # Base similarity score (convert distance to similarity)
            similarity_score = 1.0 - result.distance

            # Calculate component scores
            recency_score = self._calculate_recency_score(result, config)
            popularity_score = self._calculate_popularity_score(result)
            metadata_score = self._calculate_metadata_score(result, query_terms)

            # Calculate boost factors
            boost_factors = self._calculate_boost_factors(result, query_lower, config)

            # Calculate final score based on strategy
            final_score = self._calculate_final_score(
                similarity_score, recency_score, popularity_score, metadata_score,
                boost_factors, config
            )

            ranked_result = RankedResult(
                result=result,
                final_score=final_score,
                similarity_score=similarity_score,
                recency_score=recency_score,
                popularity_score=popularity_score,
                metadata_score=metadata_score,
                boost_factors=boost_factors,
                rank_position=0  # Will be set later
            )

            ranked_results.append(ranked_result)

        return ranked_results

    def _calculate_recency_score(
        self,
        result: VectorSearchResult,
        config: RankingConfiguration
    ) -> float:
        """Calculate recency score based on document age."""
        if not result.metadata or not result.metadata.created_at:
            return 0.0

        try:
            created_at = result.metadata.created_at
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

            # Calculate age in days
            age_days = (datetime.now() - created_at).days

            # Apply exponential decay
            decay_factor = math.exp(-age_days / config.recency_decay_days)

            return decay_factor

        except Exception:
            return 0.0

    def _calculate_popularity_score(self, result: VectorSearchResult) -> float:
        """Calculate popularity score based on access patterns."""
        # Simple popularity based on document and chunk access
        doc_popularity = self._document_popularity.get(result.metadata.document_id, 0)
        chunk_popularity = self._chunk_popularity.get(result.id, 0)

        # Normalize popularity scores (simple approach)
        max_doc_popularity = max(self._document_popularity.values()) if self._document_popularity else 1
        max_chunk_popularity = max(self._chunk_popularity.values()) if self._chunk_popularity else 1

        doc_score = doc_popularity / max_doc_popularity if max_doc_popularity > 0 else 0
        chunk_score = chunk_popularity / max_chunk_popularity if max_chunk_popularity > 0 else 0

        return (doc_score + chunk_score) / 2

    def _calculate_metadata_score(
        self,
        result: VectorSearchResult,
        query_terms: Set[str]
    ) -> float:
        """Calculate metadata relevance score."""
        if not result.metadata:
            return 0.0

        score = 0.0

        # Check section path relevance
        if result.metadata.section_path:
            section_text = ' '.join(result.metadata.section_path).lower()
            section_terms = set(section_text.split())
            overlap = len(query_terms.intersection(section_terms))
            if overlap > 0:
                score += 0.3 * (overlap / len(query_terms))

        # Boost for certain document types based on query
        if result.metadata.document_kind:
            if 'requirement' in query_terms and result.metadata.document_kind == 'RFQ':
                score += 0.2
            elif 'offer' in query_terms and result.metadata.document_kind == 'OFFER':
                score += 0.2

        return min(score, 1.0)

    def _calculate_boost_factors(
        self,
        result: VectorSearchResult,
        query_lower: str,
        config: RankingConfiguration
    ) -> Dict[str, float]:
        """Calculate various boost factors."""
        boost_factors = {}

        content_lower = result.document.lower()

        # Exact match boost
        if config.boost_exact_matches and query_lower in content_lower:
            boost_factors['exact_match'] = 1.2
            self._stats["boost_applications"]["exact_match"] += 1

        # Title/heading boost (simple heuristic)
        if config.boost_title_matches:
            # Check if content appears to be a title or heading
            if len(result.document) < 100 and result.document.isupper():
                boost_factors['title_match'] = 1.1
                self._stats["boost_applications"]["title_match"] += 1

        # Recency boost
        if config.boost_recent_documents:
            if result.metadata and result.metadata.created_at:
                try:
                    created_at = result.metadata.created_at
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

                    # Boost documents from last 7 days
                    if (datetime.now() - created_at).days <= 7:
                        boost_factors['recency'] = 1.1
                        self._stats["boost_applications"]["recency"] += 1
                except:
                    pass

        return boost_factors

    def _calculate_final_score(
        self,
        similarity_score: float,
        recency_score: float,
        popularity_score: float,
        metadata_score: float,
        boost_factors: Dict[str, float],
        config: RankingConfiguration
    ) -> float:
        """Calculate the final ranking score."""
        if config.strategy == RankingStrategy.SIMILARITY:
            base_score = similarity_score
        elif config.strategy == RankingStrategy.RECENCY:
            base_score = recency_score
        elif config.strategy == RankingStrategy.POPULARITY:
            base_score = popularity_score
        elif config.strategy == RankingStrategy.HYBRID:
            # Weighted combination
            base_score = (
                similarity_score * config.weights.similarity_weight +
                recency_score * config.weights.recency_weight +
                popularity_score * config.weights.popularity_weight +
                metadata_score * config.weights.metadata_weight
            )
        else:  # CUSTOM
            base_score = similarity_score

        # Apply boost factors
        final_score = base_score
        for boost_name, boost_value in boost_factors.items():
            final_score *= boost_value

        return min(final_score, 1.0)

    def _apply_deduplication(
        self,
        results: List[RankedResult],
        config: RankingConfiguration
    ) -> List[RankedResult]:
        """Apply deduplication to remove similar results."""
        if config.diversity_threshold >= 1.0:
            return results

        deduplicated = []
        seen_documents = {}

        # Sort by score first
        sorted_results = sorted(results, key=lambda x: x.final_score, reverse=True)

        for result in sorted_results:
            doc_id = result.result.metadata.document_id if result.result.metadata else "unknown"

            # Limit results per document
            if doc_id in seen_documents:
                if seen_documents[doc_id] >= config.max_results_per_document:
                    continue
                seen_documents[doc_id] += 1
            else:
                seen_documents[doc_id] = 1

            # Check content similarity with existing results
            is_diverse = True
            for existing in deduplicated:
                similarity = self._calculate_content_similarity(
                    result.result.document,
                    existing.result.document
                )
                if similarity > config.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                deduplicated.append(result)

        return deduplicated

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity between two texts."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 and not words2:
            return 1.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _finalize_ranking(self, results: List[RankedResult]) -> List[RankedResult]:
        """Finalize ranking by sorting and assigning positions."""
        # Sort by final score (descending)
        sorted_results = sorted(results, key=lambda x: x.final_score, reverse=True)

        # Assign rank positions
        for i, result in enumerate(sorted_results):
            result.rank_position = i + 1

        return sorted_results

    def _update_average_ranking_time(self, ranking_time_ms: int) -> None:
        """Update the running average ranking time."""
        total_rankings = self._stats["total_rankings"]
        current_avg = self._stats["average_ranking_time_ms"]

        # Calculate new average
        new_avg = ((current_avg * (total_rankings - 1)) + ranking_time_ms) / total_rankings
        self._stats["average_ranking_time_ms"] = new_avg

    def track_result_access(self, document_id: str, chunk_id: str) -> None:
        """Track access to a result for popularity scoring."""
        self._document_popularity[document_id] = self._document_popularity.get(document_id, 0) + 1
        self._chunk_popularity[chunk_id] = self._chunk_popularity.get(chunk_id, 0) + 1

    async def get_ranking_statistics(self) -> Dict[str, Any]:
        """Get ranking service statistics."""
        return dict(self._stats)


# Global instance
_result_ranker: Optional[ResultRanker] = None


async def get_result_ranker() -> ResultRanker:
    """Get the global result ranker instance."""
    global _result_ranker

    if _result_ranker is None:
        _result_ranker = ResultRanker()

    return _result_ranker
