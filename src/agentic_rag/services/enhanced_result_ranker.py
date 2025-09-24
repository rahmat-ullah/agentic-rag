"""
Enhanced Result Ranking Service with Multi-Factor Scoring.

This module provides advanced result ranking capabilities with composite scoring,
improved recency-based scoring, document type preferences, and user interaction tracking.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.result_ranker import (
    RankedResult, VectorSearchResult, RankingConfiguration, RankingStrategy
)

logger = structlog.get_logger(__name__)


class ScoringComponent(str, Enum):
    """Individual scoring components."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    RECENCY = "recency"
    DOCUMENT_TYPE = "document_type"
    SECTION_RELEVANCE = "section_relevance"
    USER_INTERACTIONS = "user_interactions"
    CONTENT_QUALITY = "content_quality"
    AUTHORITY = "authority"


@dataclass
class UserInteractionData:
    """User interaction data for ranking."""
    click_count: int = 0
    view_duration_seconds: float = 0.0
    bookmark_count: int = 0
    share_count: int = 0
    rating: Optional[float] = None
    last_accessed: Optional[datetime] = None
    access_frequency: float = 0.0  # accesses per day


@dataclass
class DocumentMetrics:
    """Document-level metrics for ranking."""
    creation_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    version: int = 1
    author_authority: float = 0.5
    content_length: int = 0
    section_depth: int = 0
    is_title_section: bool = False
    is_summary_section: bool = False
    keyword_density: float = 0.0


@dataclass
class CompositeScore:
    """Detailed composite score breakdown."""
    final_score: float
    component_scores: Dict[ScoringComponent, float] = field(default_factory=dict)
    boost_factors: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class EnhancedResultRanker:
    """Enhanced result ranker with multi-factor scoring."""
    
    def __init__(self):
        """Initialize the enhanced result ranker."""
        self._user_interactions: Dict[str, UserInteractionData] = {}
        self._document_metrics: Dict[str, DocumentMetrics] = {}
        self._stats = {
            "total_rankings": 0,
            "average_ranking_time_ms": 0.0,
            "score_distributions": {},
            "component_weights_used": {}
        }
        
        logger.info("Enhanced result ranker initialized")
    
    async def rank_results_enhanced(
        self,
        results: List[VectorSearchResult],
        query: str,
        user_id: str,
        config: RankingConfiguration,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> List[RankedResult]:
        """
        Rank results using enhanced multi-factor scoring.
        
        Args:
            results: Vector search results to rank
            query: Original search query
            user_id: User identifier for personalization
            config: Ranking configuration
            user_preferences: User-specific preferences
            
        Returns:
            List of ranked results with detailed scoring
        """
        start_time = time.time()
        
        if not results:
            return []
        
        logger.info(f"Ranking {len(results)} results with enhanced scoring",
                   user_id=user_id, query_length=len(query))
        
        try:
            self._stats["total_rankings"] += 1
            
            # Calculate composite scores for each result
            scored_results = []
            for result in results:
                composite_score = await self._calculate_composite_score(
                    result, query, user_id, config, user_preferences
                )
                
                ranked_result = RankedResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    content=result.content,
                    metadata=result.metadata,
                    similarity_score=result.similarity_score,
                    final_score=composite_score.final_score,
                    rank=0,  # Will be set after sorting
                    explanation=composite_score.explanation
                )
                
                # Add detailed scoring information
                ranked_result.scoring_details = {
                    "component_scores": composite_score.component_scores,
                    "boost_factors": composite_score.boost_factors,
                    "penalties": composite_score.penalties
                }
                
                scored_results.append(ranked_result)
            
            # Sort by final score (descending)
            scored_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Assign ranks
            for i, result in enumerate(scored_results):
                result.rank = i + 1
            
            # Update statistics
            ranking_time_ms = int((time.time() - start_time) * 1000)
            self._update_ranking_stats(ranking_time_ms, scored_results)
            
            logger.info(f"Enhanced ranking completed",
                       results_count=len(scored_results),
                       ranking_time_ms=ranking_time_ms)
            
            return scored_results
            
        except Exception as e:
            logger.error(f"Enhanced ranking failed: {e}", exc_info=True)
            raise
    
    async def _calculate_composite_score(
        self,
        result: VectorSearchResult,
        query: str,
        user_id: str,
        config: RankingConfiguration,
        user_preferences: Optional[Dict[str, Any]]
    ) -> CompositeScore:
        """Calculate composite score with detailed breakdown."""
        component_scores = {}
        boost_factors = {}
        penalties = {}
        
        # 1. Semantic Similarity Score (base score)
        similarity_score = self._calculate_similarity_score(result)
        component_scores[ScoringComponent.SEMANTIC_SIMILARITY] = similarity_score
        
        # 2. Recency Score
        recency_score = self._calculate_enhanced_recency_score(result)
        component_scores[ScoringComponent.RECENCY] = recency_score
        
        # 3. Document Type Preference Score
        doc_type_score = self._calculate_document_type_score(result, user_preferences)
        component_scores[ScoringComponent.DOCUMENT_TYPE] = doc_type_score
        
        # 4. Section Relevance Score
        section_score = self._calculate_section_relevance_score(result, query)
        component_scores[ScoringComponent.SECTION_RELEVANCE] = section_score
        
        # 5. User Interaction Score
        interaction_score = self._calculate_user_interaction_score(result, user_id)
        component_scores[ScoringComponent.USER_INTERACTIONS] = interaction_score
        
        # 6. Content Quality Score
        quality_score = self._calculate_content_quality_score(result)
        component_scores[ScoringComponent.CONTENT_QUALITY] = quality_score
        
        # 7. Authority Score
        authority_score = self._calculate_authority_score(result)
        component_scores[ScoringComponent.AUTHORITY] = authority_score
        
        # Calculate boost factors
        boost_factors.update(self._calculate_boost_factors(result, query))
        
        # Calculate penalties
        penalties.update(self._calculate_penalties(result))
        
        # Combine scores using weighted average
        final_score = self._combine_scores(
            component_scores, boost_factors, penalties, config
        )
        
        # Generate explanation
        explanation = self._generate_score_explanation(
            component_scores, boost_factors, penalties, final_score
        )
        
        return CompositeScore(
            final_score=final_score,
            component_scores=component_scores,
            boost_factors=boost_factors,
            penalties=penalties,
            explanation=explanation
        )
    
    def _calculate_similarity_score(self, result: VectorSearchResult) -> float:
        """Calculate semantic similarity score."""
        # Convert distance to similarity (assuming cosine distance)
        return max(0.0, 1.0 - result.distance)
    
    def _calculate_enhanced_recency_score(self, result: VectorSearchResult) -> float:
        """Calculate enhanced recency score with decay functions."""
        doc_metrics = self._get_document_metrics(result.document_id)
        
        if not doc_metrics.last_modified and not doc_metrics.creation_date:
            return 0.5  # Neutral score for unknown dates
        
        # Use last_modified if available, otherwise creation_date
        reference_date = doc_metrics.last_modified or doc_metrics.creation_date
        if not reference_date:
            return 0.5
        
        # Calculate age in days
        age_days = (datetime.now() - reference_date).days
        
        # Apply exponential decay with configurable half-life
        half_life_days = 90  # Documents lose half relevance after 90 days
        decay_factor = math.exp(-math.log(2) * age_days / half_life_days)
        
        # Boost for very recent documents (last 7 days)
        if age_days <= 7:
            decay_factor *= 1.2
        elif age_days <= 30:
            decay_factor *= 1.1
        
        return min(1.0, decay_factor)
    
    def _calculate_document_type_score(
        self, 
        result: VectorSearchResult, 
        user_preferences: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate document type preference score."""
        if not user_preferences or 'document_type_preferences' not in user_preferences:
            return 0.5  # Neutral score
        
        doc_type = result.metadata.get('document_type', 'unknown')
        preferences = user_preferences['document_type_preferences']
        
        return preferences.get(doc_type, 0.5)
    
    def _calculate_section_relevance_score(
        self, 
        result: VectorSearchResult, 
        query: str
    ) -> float:
        """Calculate section relevance score."""
        doc_metrics = self._get_document_metrics(result.document_id)
        base_score = 0.5
        
        # Boost for title sections
        if doc_metrics.is_title_section:
            base_score += 0.3
        
        # Boost for summary sections
        if doc_metrics.is_summary_section:
            base_score += 0.2
        
        # Boost based on section depth (higher level = more important)
        if doc_metrics.section_depth > 0:
            depth_boost = max(0, 0.2 - (doc_metrics.section_depth - 1) * 0.05)
            base_score += depth_boost
        
        # Keyword density boost
        base_score += min(0.2, doc_metrics.keyword_density * 0.5)
        
        return min(1.0, base_score)
    
    def _calculate_user_interaction_score(
        self, 
        result: VectorSearchResult, 
        user_id: str
    ) -> float:
        """Calculate user interaction score."""
        interaction_key = f"{user_id}:{result.document_id}"
        interactions = self._user_interactions.get(interaction_key, UserInteractionData())
        
        score = 0.0
        
        # Click-through rate contribution
        if interactions.click_count > 0:
            score += min(0.3, interactions.click_count * 0.1)
        
        # View duration contribution
        if interactions.view_duration_seconds > 0:
            # Normalize by expected reading time (assume 200 words/minute)
            content_length = len(result.content.split())
            expected_duration = content_length / 200 * 60  # seconds
            duration_ratio = interactions.view_duration_seconds / max(expected_duration, 30)
            score += min(0.2, duration_ratio * 0.2)
        
        # Bookmark/share contribution
        score += min(0.2, (interactions.bookmark_count + interactions.share_count) * 0.1)
        
        # Rating contribution
        if interactions.rating is not None:
            score += (interactions.rating / 5.0) * 0.2
        
        # Access frequency contribution
        score += min(0.1, interactions.access_frequency * 0.05)
        
        return min(1.0, score)
    
    def _calculate_content_quality_score(self, result: VectorSearchResult) -> float:
        """Calculate content quality score."""
        doc_metrics = self._get_document_metrics(result.document_id)
        
        score = 0.5  # Base score
        
        # Content length score (prefer moderate length)
        if doc_metrics.content_length > 0:
            # Optimal length around 500-2000 characters
            if 500 <= doc_metrics.content_length <= 2000:
                score += 0.2
            elif 200 <= doc_metrics.content_length <= 3000:
                score += 0.1
            elif doc_metrics.content_length < 100:
                score -= 0.2  # Too short
        
        # Version score (newer versions are better)
        if doc_metrics.version > 1:
            score += min(0.1, (doc_metrics.version - 1) * 0.02)
        
        return min(1.0, max(0.0, score))
    
    def _calculate_authority_score(self, result: VectorSearchResult) -> float:
        """Calculate authority score."""
        doc_metrics = self._get_document_metrics(result.document_id)
        return doc_metrics.author_authority
    
    def _calculate_boost_factors(
        self, 
        result: VectorSearchResult, 
        query: str
    ) -> Dict[str, float]:
        """Calculate boost factors."""
        boosts = {}
        query_lower = query.lower()
        content_lower = result.content.lower()
        
        # Exact phrase match boost
        if query_lower in content_lower:
            boosts["exact_phrase_match"] = 1.5
        
        # Title match boost
        title = result.metadata.get('title', '').lower()
        if any(word in title for word in query_lower.split()):
            boosts["title_match"] = 1.3
        
        # Multiple term match boost
        query_terms = set(query_lower.split())
        content_terms = set(content_lower.split())
        match_ratio = len(query_terms.intersection(content_terms)) / len(query_terms)
        if match_ratio > 0.7:
            boosts["high_term_coverage"] = 1.2
        
        return boosts
    
    def _calculate_penalties(self, result: VectorSearchResult) -> Dict[str, float]:
        """Calculate penalty factors."""
        penalties = {}
        
        # Very short content penalty
        if len(result.content) < 50:
            penalties["short_content"] = 0.8
        
        # Missing metadata penalty
        if not result.metadata.get('title'):
            penalties["missing_title"] = 0.9
        
        return penalties

    def _combine_scores(
        self,
        component_scores: Dict[ScoringComponent, float],
        boost_factors: Dict[str, float],
        penalties: Dict[str, float],
        config: RankingConfiguration
    ) -> float:
        """Combine component scores into final score."""
        # Get weights from configuration
        weights = {
            ScoringComponent.SEMANTIC_SIMILARITY: config.weights.similarity_weight,
            ScoringComponent.RECENCY: config.weights.recency_weight,
            ScoringComponent.DOCUMENT_TYPE: 0.15,  # Default weight
            ScoringComponent.SECTION_RELEVANCE: config.weights.metadata_weight,
            ScoringComponent.USER_INTERACTIONS: config.weights.popularity_weight,
            ScoringComponent.CONTENT_QUALITY: 0.1,  # Default weight
            ScoringComponent.AUTHORITY: 0.05  # Default weight
        }

        # Calculate weighted sum
        weighted_score = 0.0
        total_weight = 0.0

        for component, score in component_scores.items():
            weight = weights.get(component, 0.0)
            weighted_score += score * weight
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            base_score = weighted_score / total_weight
        else:
            base_score = component_scores.get(ScoringComponent.SEMANTIC_SIMILARITY, 0.0)

        # Apply boost factors
        final_score = base_score
        for boost_name, boost_value in boost_factors.items():
            final_score *= boost_value

        # Apply penalties
        for penalty_name, penalty_value in penalties.items():
            final_score *= penalty_value

        # Ensure score is in valid range
        return max(0.0, min(1.0, final_score))

    def _generate_score_explanation(
        self,
        component_scores: Dict[ScoringComponent, float],
        boost_factors: Dict[str, float],
        penalties: Dict[str, float],
        final_score: float
    ) -> str:
        """Generate human-readable score explanation."""
        explanation_parts = []

        # Component scores
        for component, score in component_scores.items():
            component_name = component.value.replace('_', ' ').title()
            explanation_parts.append(f"{component_name}: {score:.2f}")

        # Boost factors
        if boost_factors:
            boosts = [f"{name.replace('_', ' ').title()}: {value:.2f}x"
                     for name, value in boost_factors.items()]
            explanation_parts.append(f"Boosts: {', '.join(boosts)}")

        # Penalties
        if penalties:
            penalty_list = [f"{name.replace('_', ' ').title()}: {value:.2f}x"
                           for name, value in penalties.items()]
            explanation_parts.append(f"Penalties: {', '.join(penalty_list)}")

        explanation_parts.append(f"Final Score: {final_score:.3f}")

        return " | ".join(explanation_parts)

    def _get_document_metrics(self, document_id: str) -> DocumentMetrics:
        """Get or create document metrics."""
        if document_id not in self._document_metrics:
            self._document_metrics[document_id] = DocumentMetrics()
        return self._document_metrics[document_id]

    def _update_ranking_stats(
        self,
        ranking_time_ms: int,
        results: List[RankedResult]
    ) -> None:
        """Update ranking statistics."""
        # Update average ranking time
        current_avg = self._stats["average_ranking_time_ms"]
        total_rankings = self._stats["total_rankings"]

        new_avg = ((current_avg * (total_rankings - 1)) + ranking_time_ms) / total_rankings
        self._stats["average_ranking_time_ms"] = new_avg

        # Update score distributions
        if results:
            scores = [result.final_score for result in results]
            self._stats["score_distributions"] = {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores),
                "count": len(scores)
            }

    async def update_user_interaction(
        self,
        user_id: str,
        document_id: str,
        interaction_type: str,
        value: Optional[float] = None
    ) -> None:
        """Update user interaction data."""
        interaction_key = f"{user_id}:{document_id}"

        if interaction_key not in self._user_interactions:
            self._user_interactions[interaction_key] = UserInteractionData()

        interaction = self._user_interactions[interaction_key]

        if interaction_type == "click":
            interaction.click_count += 1
        elif interaction_type == "view_duration" and value is not None:
            interaction.view_duration_seconds += value
        elif interaction_type == "bookmark":
            interaction.bookmark_count += 1
        elif interaction_type == "share":
            interaction.share_count += 1
        elif interaction_type == "rating" and value is not None:
            interaction.rating = value

        interaction.last_accessed = datetime.now()

        # Update access frequency (simple daily average)
        if interaction.click_count > 0:
            days_since_first_access = max(1, (datetime.now() - interaction.last_accessed).days)
            interaction.access_frequency = interaction.click_count / days_since_first_access

    async def update_document_metrics(
        self,
        document_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Update document metrics."""
        doc_metrics = self._get_document_metrics(document_id)

        if 'creation_date' in metrics:
            doc_metrics.creation_date = metrics['creation_date']
        if 'last_modified' in metrics:
            doc_metrics.last_modified = metrics['last_modified']
        if 'version' in metrics:
            doc_metrics.version = metrics['version']
        if 'author_authority' in metrics:
            doc_metrics.author_authority = metrics['author_authority']
        if 'content_length' in metrics:
            doc_metrics.content_length = metrics['content_length']
        if 'section_depth' in metrics:
            doc_metrics.section_depth = metrics['section_depth']
        if 'is_title_section' in metrics:
            doc_metrics.is_title_section = metrics['is_title_section']
        if 'is_summary_section' in metrics:
            doc_metrics.is_summary_section = metrics['is_summary_section']
        if 'keyword_density' in metrics:
            doc_metrics.keyword_density = metrics['keyword_density']

    async def get_ranking_statistics(self) -> Dict[str, Any]:
        """Get ranking statistics."""
        return {
            "total_rankings": self._stats["total_rankings"],
            "average_ranking_time_ms": self._stats["average_ranking_time_ms"],
            "score_distributions": self._stats.get("score_distributions", {}),
            "user_interactions_count": len(self._user_interactions),
            "document_metrics_count": len(self._document_metrics)
        }


# Singleton instance
_enhanced_ranker_instance: Optional[EnhancedResultRanker] = None


async def get_enhanced_result_ranker() -> EnhancedResultRanker:
    """Get the enhanced result ranker instance."""
    global _enhanced_ranker_instance

    if _enhanced_ranker_instance is None:
        _enhanced_ranker_instance = EnhancedResultRanker()

    return _enhanced_ranker_instance


def reset_enhanced_result_ranker():
    """Reset the enhanced ranker instance (for testing)."""
    global _enhanced_ranker_instance
    _enhanced_ranker_instance = None
