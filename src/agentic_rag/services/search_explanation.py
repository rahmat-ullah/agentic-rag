"""
Search Result Explanation Service.

This module provides comprehensive search result explanation capabilities including
scoring breakdown, match highlighting, relevance explanation text, confidence indicators,
and debug mode for detailed scoring information.
"""

import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ExplanationLevel(str, Enum):
    """Explanation detail levels."""
    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"


class ConfidenceLevel(str, Enum):
    """Confidence level indicators."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class MatchHighlight:
    """Highlighted match in content."""
    text: str
    start_pos: int
    end_pos: int
    match_type: str  # exact, partial, semantic
    relevance_score: float


@dataclass
class ScoringBreakdown:
    """Detailed scoring breakdown."""
    component_name: str
    score: float
    weight: float
    contribution: float
    explanation: str


@dataclass
class SearchExplanation:
    """Comprehensive search result explanation."""
    result_id: str
    final_score: float
    confidence_level: ConfidenceLevel
    confidence_score: float
    
    # Scoring details
    scoring_breakdown: List[ScoringBreakdown] = field(default_factory=list)
    boost_factors: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    
    # Content analysis
    highlighted_matches: List[MatchHighlight] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    relevance_summary: str = ""
    
    # Debug information
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    # Human-readable explanation
    explanation_text: str = ""


class SearchExplanationService:
    """Service for generating search result explanations."""
    
    def __init__(self):
        """Initialize the search explanation service."""
        self._stats = {
            "explanations_generated": 0,
            "average_generation_time_ms": 0.0,
            "explanation_levels_used": {}
        }
        
        logger.info("Search explanation service initialized")
    
    async def generate_explanation(
        self,
        result: Any,  # RankedResult or similar
        query: str,
        level: ExplanationLevel = ExplanationLevel.BASIC,
        include_debug: bool = False
    ) -> SearchExplanation:
        """
        Generate comprehensive explanation for a search result.
        
        Args:
            result: Search result to explain
            query: Original search query
            level: Level of detail for explanation
            include_debug: Whether to include debug information
            
        Returns:
            SearchExplanation with detailed breakdown
        """
        start_time = time.time()
        
        logger.info(f"Generating explanation for result",
                   result_id=getattr(result, 'document_id', 'unknown'),
                   level=level.value)
        
        try:
            self._stats["explanations_generated"] += 1
            self._stats["explanation_levels_used"][level.value] = \
                self._stats["explanation_levels_used"].get(level.value, 0) + 1
            
            # Create base explanation
            explanation = SearchExplanation(
                result_id=getattr(result, 'document_id', 'unknown'),
                final_score=getattr(result, 'final_score', 0.0),
                confidence_level=self._calculate_confidence_level(result),
                confidence_score=self._calculate_confidence_score(result)
            )
            
            # Generate scoring breakdown
            explanation.scoring_breakdown = self._generate_scoring_breakdown(result, level)
            
            # Extract boost factors and penalties
            if hasattr(result, 'scoring_details'):
                explanation.boost_factors = result.scoring_details.get('boost_factors', {})
                explanation.penalties = result.scoring_details.get('penalties', {})
            
            # Generate content analysis
            explanation.highlighted_matches = self._highlight_matches(result, query)
            explanation.key_phrases = self._extract_key_phrases(result, query)
            explanation.relevance_summary = self._generate_relevance_summary(result, query)
            
            # Generate human-readable explanation
            explanation.explanation_text = self._generate_explanation_text(
                explanation, query, level
            )
            
            # Add debug information if requested
            if include_debug or level == ExplanationLevel.DEBUG:
                explanation.debug_info = self._generate_debug_info(result, query)
            
            # Update statistics
            generation_time_ms = int((time.time() - start_time) * 1000)
            self._update_stats(generation_time_ms)
            
            logger.info(f"Explanation generated",
                       generation_time_ms=generation_time_ms,
                       confidence_level=explanation.confidence_level.value)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}", exc_info=True)
            raise
    
    def _calculate_confidence_level(self, result: Any) -> ConfidenceLevel:
        """Calculate confidence level based on result score and factors."""
        score = getattr(result, 'final_score', 0.0)
        
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_confidence_score(self, result: Any) -> float:
        """Calculate numerical confidence score."""
        base_score = getattr(result, 'final_score', 0.0)
        
        # Adjust based on various factors
        confidence = base_score
        
        # Boost confidence if multiple scoring factors agree
        if hasattr(result, 'scoring_details'):
            component_scores = result.scoring_details.get('component_scores', {})
            if len(component_scores) > 3:
                # Check if scores are consistent
                scores = list(component_scores.values())
                if scores:
                    score_variance = sum((s - base_score) ** 2 for s in scores) / len(scores)
                    if score_variance < 0.1:  # Low variance = high confidence
                        confidence *= 1.1
        
        return min(1.0, confidence)
    
    def _generate_scoring_breakdown(
        self, 
        result: Any, 
        level: ExplanationLevel
    ) -> List[ScoringBreakdown]:
        """Generate detailed scoring breakdown."""
        breakdown = []
        
        if not hasattr(result, 'scoring_details'):
            # Fallback for basic results
            breakdown.append(ScoringBreakdown(
                component_name="Similarity",
                score=getattr(result, 'similarity_score', 0.0),
                weight=1.0,
                contribution=getattr(result, 'final_score', 0.0),
                explanation="Semantic similarity to query"
            ))
            return breakdown
        
        component_scores = result.scoring_details.get('component_scores', {})
        
        for component, score in component_scores.items():
            # Determine weight and contribution
            weight = self._get_component_weight(component)
            contribution = score * weight
            
            # Generate explanation based on component
            explanation = self._get_component_explanation(component, score, level)
            
            breakdown.append(ScoringBreakdown(
                component_name=component.replace('_', ' ').title(),
                score=score,
                weight=weight,
                contribution=contribution,
                explanation=explanation
            ))
        
        # Sort by contribution (descending)
        breakdown.sort(key=lambda x: x.contribution, reverse=True)
        
        return breakdown
    
    def _get_component_weight(self, component: str) -> float:
        """Get weight for scoring component."""
        weights = {
            'semantic_similarity': 0.4,
            'recency': 0.2,
            'document_type': 0.15,
            'section_relevance': 0.15,
            'user_interactions': 0.1,
            'content_quality': 0.05,
            'authority': 0.05
        }
        return weights.get(component, 0.1)
    
    def _get_component_explanation(
        self, 
        component: str, 
        score: float, 
        level: ExplanationLevel
    ) -> str:
        """Get explanation for scoring component."""
        explanations = {
            'semantic_similarity': self._explain_similarity(score, level),
            'recency': self._explain_recency(score, level),
            'document_type': self._explain_document_type(score, level),
            'section_relevance': self._explain_section_relevance(score, level),
            'user_interactions': self._explain_user_interactions(score, level),
            'content_quality': self._explain_content_quality(score, level),
            'authority': self._explain_authority(score, level)
        }
        
        return explanations.get(component, f"Score: {score:.2f}")
    
    def _explain_similarity(self, score: float, level: ExplanationLevel) -> str:
        """Explain similarity score."""
        if level == ExplanationLevel.BASIC:
            if score >= 0.8:
                return "Very similar to your query"
            elif score >= 0.6:
                return "Moderately similar to your query"
            else:
                return "Somewhat related to your query"
        else:
            return f"Semantic similarity score of {score:.2f} based on vector embeddings"
    
    def _explain_recency(self, score: float, level: ExplanationLevel) -> str:
        """Explain recency score."""
        if level == ExplanationLevel.BASIC:
            if score >= 0.8:
                return "Very recent document"
            elif score >= 0.6:
                return "Moderately recent document"
            else:
                return "Older document"
        else:
            return f"Recency score of {score:.2f} based on document age and updates"
    
    def _explain_document_type(self, score: float, level: ExplanationLevel) -> str:
        """Explain document type score."""
        if level == ExplanationLevel.BASIC:
            if score >= 0.7:
                return "Preferred document type"
            else:
                return "Standard document type"
        else:
            return f"Document type preference score of {score:.2f}"
    
    def _explain_section_relevance(self, score: float, level: ExplanationLevel) -> str:
        """Explain section relevance score."""
        if level == ExplanationLevel.BASIC:
            if score >= 0.8:
                return "From important section (title/summary)"
            elif score >= 0.6:
                return "From relevant section"
            else:
                return "From general content"
        else:
            return f"Section relevance score of {score:.2f} based on section importance"
    
    def _explain_user_interactions(self, score: float, level: ExplanationLevel) -> str:
        """Explain user interactions score."""
        if level == ExplanationLevel.BASIC:
            if score >= 0.6:
                return "Popular with users"
            elif score >= 0.3:
                return "Some user engagement"
            else:
                return "Limited user engagement"
        else:
            return f"User interaction score of {score:.2f} based on clicks, views, and ratings"
    
    def _explain_content_quality(self, score: float, level: ExplanationLevel) -> str:
        """Explain content quality score."""
        if level == ExplanationLevel.BASIC:
            if score >= 0.7:
                return "High quality content"
            else:
                return "Standard quality content"
        else:
            return f"Content quality score of {score:.2f} based on length, structure, and version"
    
    def _explain_authority(self, score: float, level: ExplanationLevel) -> str:
        """Explain authority score."""
        if level == ExplanationLevel.BASIC:
            if score >= 0.7:
                return "From authoritative source"
            else:
                return "From standard source"
        else:
            return f"Authority score of {score:.2f} based on author and source credibility"

    def _highlight_matches(self, result: Any, query: str) -> List[MatchHighlight]:
        """Highlight query matches in result content."""
        content = getattr(result, 'content', '')
        if not content or not query:
            return []

        highlights = []
        query_terms = query.lower().split()
        content_lower = content.lower()

        # Find exact phrase matches
        if query.lower() in content_lower:
            start_pos = content_lower.find(query.lower())
            highlights.append(MatchHighlight(
                text=content[start_pos:start_pos + len(query)],
                start_pos=start_pos,
                end_pos=start_pos + len(query),
                match_type="exact",
                relevance_score=1.0
            ))

        # Find individual term matches
        for term in query_terms:
            if len(term) < 3:  # Skip very short terms
                continue

            for match in re.finditer(re.escape(term), content_lower):
                start_pos = match.start()
                end_pos = match.end()

                # Avoid duplicate highlights
                if not any(h.start_pos <= start_pos < h.end_pos for h in highlights):
                    highlights.append(MatchHighlight(
                        text=content[start_pos:end_pos],
                        start_pos=start_pos,
                        end_pos=end_pos,
                        match_type="partial",
                        relevance_score=0.7
                    ))

        # Sort by position
        highlights.sort(key=lambda x: x.start_pos)

        # Limit to top 10 highlights
        return highlights[:10]

    def _extract_key_phrases(self, result: Any, query: str) -> List[str]:
        """Extract key phrases from result content."""
        content = getattr(result, 'content', '')
        if not content:
            return []

        # Simple key phrase extraction
        phrases = []

        # Extract sentences containing query terms
        query_terms = set(query.lower().split())
        sentences = re.split(r'[.!?]+', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue

            sentence_words = set(sentence.lower().split())
            if query_terms.intersection(sentence_words):
                # Truncate long sentences
                if len(sentence) > 150:
                    sentence = sentence[:147] + "..."
                phrases.append(sentence)

        # Limit to top 3 phrases
        return phrases[:3]

    def _generate_relevance_summary(self, result: Any, query: str) -> str:
        """Generate relevance summary."""
        score = getattr(result, 'final_score', 0.0)

        if score >= 0.9:
            return f"Highly relevant to '{query}' with strong semantic similarity and additional relevance factors."
        elif score >= 0.75:
            return f"Very relevant to '{query}' with good semantic match and supporting factors."
        elif score >= 0.6:
            return f"Moderately relevant to '{query}' with decent semantic similarity."
        elif score >= 0.4:
            return f"Somewhat relevant to '{query}' with partial semantic match."
        else:
            return f"Limited relevance to '{query}' but may contain related information."

    def _generate_explanation_text(
        self,
        explanation: SearchExplanation,
        query: str,
        level: ExplanationLevel
    ) -> str:
        """Generate human-readable explanation text."""
        if level == ExplanationLevel.BASIC:
            return self._generate_basic_explanation(explanation, query)
        elif level == ExplanationLevel.DETAILED:
            return self._generate_detailed_explanation(explanation, query)
        else:  # DEBUG
            return self._generate_debug_explanation(explanation, query)

    def _generate_basic_explanation(self, explanation: SearchExplanation, query: str) -> str:
        """Generate basic explanation text."""
        confidence_text = {
            ConfidenceLevel.VERY_HIGH: "very confident",
            ConfidenceLevel.HIGH: "confident",
            ConfidenceLevel.MEDIUM: "moderately confident",
            ConfidenceLevel.LOW: "somewhat confident",
            ConfidenceLevel.VERY_LOW: "not very confident"
        }

        confidence = confidence_text.get(explanation.confidence_level, "uncertain")

        text = f"This result scored {explanation.final_score:.2f} for your query '{query}'. "
        text += f"We are {confidence} this is relevant. "

        if explanation.highlighted_matches:
            match_count = len(explanation.highlighted_matches)
            text += f"Found {match_count} direct matches in the content. "

        if explanation.boost_factors:
            text += "This result received additional relevance boosts. "

        return text.strip()

    def _generate_detailed_explanation(self, explanation: SearchExplanation, query: str) -> str:
        """Generate detailed explanation text."""
        text = f"Relevance Analysis for '{query}':\n\n"

        text += f"Final Score: {explanation.final_score:.3f} "
        text += f"(Confidence: {explanation.confidence_level.value.replace('_', ' ').title()})\n\n"

        text += "Scoring Breakdown:\n"
        for breakdown in explanation.scoring_breakdown:
            contribution_pct = (breakdown.contribution / explanation.final_score) * 100 if explanation.final_score > 0 else 0
            text += f"• {breakdown.component_name}: {breakdown.score:.2f} "
            text += f"(weight: {breakdown.weight:.1f}, {contribution_pct:.1f}% of final score)\n"
            text += f"  {breakdown.explanation}\n"

        if explanation.boost_factors:
            text += f"\nBoost Factors:\n"
            for factor, value in explanation.boost_factors.items():
                text += f"• {factor.replace('_', ' ').title()}: {value:.2f}x\n"

        if explanation.penalties:
            text += f"\nPenalties Applied:\n"
            for penalty, value in explanation.penalties.items():
                text += f"• {penalty.replace('_', ' ').title()}: {value:.2f}x\n"

        if explanation.key_phrases:
            text += f"\nRelevant Content:\n"
            for phrase in explanation.key_phrases:
                text += f"• \"{phrase}\"\n"

        return text.strip()

    def _generate_debug_explanation(self, explanation: SearchExplanation, query: str) -> str:
        """Generate debug explanation text."""
        text = self._generate_detailed_explanation(explanation, query)

        if explanation.debug_info:
            text += f"\n\nDebug Information:\n"
            for key, value in explanation.debug_info.items():
                text += f"• {key}: {value}\n"

        if explanation.highlighted_matches:
            text += f"\nMatch Details:\n"
            for match in explanation.highlighted_matches:
                text += f"• {match.match_type.title()} match: \"{match.text}\" "
                text += f"(pos: {match.start_pos}-{match.end_pos}, score: {match.relevance_score:.2f})\n"

        return text.strip()

    def _generate_debug_info(self, result: Any, query: str) -> Dict[str, Any]:
        """Generate debug information."""
        debug_info = {}

        # Basic result information
        debug_info["result_type"] = type(result).__name__
        debug_info["has_scoring_details"] = hasattr(result, 'scoring_details')
        debug_info["content_length"] = len(getattr(result, 'content', ''))
        debug_info["metadata_keys"] = list(getattr(result, 'metadata', {}).keys())

        # Query analysis
        debug_info["query_length"] = len(query)
        debug_info["query_terms"] = len(query.split())

        # Scoring details
        if hasattr(result, 'scoring_details'):
            scoring_details = result.scoring_details
            debug_info["component_count"] = len(scoring_details.get('component_scores', {}))
            debug_info["boost_count"] = len(scoring_details.get('boost_factors', {}))
            debug_info["penalty_count"] = len(scoring_details.get('penalties', {}))

        return debug_info

    def _update_stats(self, generation_time_ms: int) -> None:
        """Update service statistics."""
        current_avg = self._stats["average_generation_time_ms"]
        total_explanations = self._stats["explanations_generated"]

        new_avg = ((current_avg * (total_explanations - 1)) + generation_time_ms) / total_explanations
        self._stats["average_generation_time_ms"] = new_avg

    async def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get explanation service statistics."""
        return {
            "explanations_generated": self._stats["explanations_generated"],
            "average_generation_time_ms": self._stats["average_generation_time_ms"],
            "explanation_levels_used": self._stats["explanation_levels_used"]
        }


# Singleton instance
_explanation_service_instance: Optional[SearchExplanationService] = None


async def get_search_explanation_service() -> SearchExplanationService:
    """Get the search explanation service instance."""
    global _explanation_service_instance

    if _explanation_service_instance is None:
        _explanation_service_instance = SearchExplanationService()

    return _explanation_service_instance


def reset_search_explanation_service():
    """Reset the explanation service instance (for testing)."""
    global _explanation_service_instance
    _explanation_service_instance = None
