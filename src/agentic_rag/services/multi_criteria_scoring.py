"""
Multi-Criteria Scoring System for LLM Reranking

This module provides comprehensive scoring algorithms for evaluating search results
across multiple criteria including relevance, specificity, completeness, and authority.
"""

import re
import math
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.vector_search import VectorSearchResult
from agentic_rag.services.reranking_models import ScoringWeights, LLMScore

logger = structlog.get_logger(__name__)


class ScoringMethod(str, Enum):
    """Methods for calculating scores."""
    LLM_BASED = "llm_based"
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"


class ContentType(str, Enum):
    """Types of content for scoring."""
    TECHNICAL = "technical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    PROCEDURAL = "procedural"
    GENERAL = "general"


@dataclass
class ScoringContext:
    """Context information for scoring."""
    query: str
    query_type: str
    document_types: List[str]
    user_preferences: Optional[Dict[str, Any]] = None
    business_context: Optional[Dict[str, Any]] = None


class DetailedScore(BaseModel):
    """Detailed scoring breakdown for a result."""
    
    relevance: float = Field(..., ge=0.0, le=10.0, description="Relevance score")
    specificity: float = Field(..., ge=0.0, le=10.0, description="Specificity score")
    completeness: float = Field(..., ge=0.0, le=10.0, description="Completeness score")
    authority: float = Field(..., ge=0.0, le=10.0, description="Authority score")
    composite_score: float = Field(..., ge=0.0, le=10.0, description="Weighted composite score")
    
    # Detailed breakdowns
    relevance_factors: Dict[str, float] = Field(default_factory=dict, description="Relevance scoring factors")
    specificity_factors: Dict[str, float] = Field(default_factory=dict, description="Specificity scoring factors")
    completeness_factors: Dict[str, float] = Field(default_factory=dict, description="Completeness scoring factors")
    authority_factors: Dict[str, float] = Field(default_factory=dict, description="Authority scoring factors")
    
    explanation: str = Field(..., description="Detailed explanation of the score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the scoring")


class MultiCriteriaScoringService:
    """Service for multi-criteria scoring of search results."""
    
    def __init__(self):
        # Authority indicators
        self._authority_indicators = {
            "official_documents": ["specification", "standard", "regulation", "policy"],
            "technical_sources": ["manual", "guide", "documentation", "reference"],
            "verified_content": ["certified", "approved", "validated", "verified"],
            "recent_content": 365  # Days for considering content recent
        }
        
        # Specificity indicators
        self._specificity_indicators = {
            "numbers": r'\b\d+(?:\.\d+)?\s*(?:%|percent|dollars?|\$|EUR|GBP|units?|pieces?|hours?|days?|months?|years?)\b',
            "technical_terms": ["algorithm", "protocol", "specification", "parameter", "configuration"],
            "measurements": r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|kg|g|lb|oz|ft|in|yards?|miles?)\b',
            "dates": r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
        }
        
        # Completeness indicators
        self._completeness_indicators = {
            "sections": ["introduction", "overview", "details", "conclusion", "summary"],
            "coverage_keywords": ["including", "such as", "for example", "specifically", "comprehensive"],
            "structure_indicators": ["first", "second", "finally", "additionally", "furthermore"]
        }
        
        logger.info("Multi-criteria scoring service initialized")
    
    def calculate_detailed_score(
        self,
        result: VectorSearchResult,
        context: ScoringContext,
        weights: ScoringWeights,
        method: ScoringMethod = ScoringMethod.HEURISTIC
    ) -> DetailedScore:
        """
        Calculate detailed multi-criteria score for a search result.
        
        Args:
            result: Search result to score
            context: Scoring context with query and metadata
            weights: Scoring weights for different criteria
            method: Scoring method to use
            
        Returns:
            Detailed score breakdown
        """
        
        if method == ScoringMethod.LLM_BASED:
            # This would be handled by the LLM reranking service
            raise NotImplementedError("LLM-based scoring handled by LLM reranking service")
        
        # Calculate individual criterion scores
        relevance_score, relevance_factors = self._calculate_relevance_score(result, context)
        specificity_score, specificity_factors = self._calculate_specificity_score(result, context)
        completeness_score, completeness_factors = self._calculate_completeness_score(result, context)
        authority_score, authority_factors = self._calculate_authority_score(result, context)
        
        # Calculate composite score
        composite_score = (
            relevance_score * weights.relevance +
            specificity_score * weights.specificity +
            completeness_score * weights.completeness +
            authority_score * weights.authority
        )
        
        # Generate explanation
        explanation = self._generate_score_explanation(
            relevance_score, specificity_score, completeness_score, authority_score,
            relevance_factors, specificity_factors, completeness_factors, authority_factors
        )
        
        # Calculate confidence based on scoring factors
        confidence = self._calculate_confidence(
            relevance_factors, specificity_factors, completeness_factors, authority_factors
        )
        
        return DetailedScore(
            relevance=relevance_score,
            specificity=specificity_score,
            completeness=completeness_score,
            authority=authority_score,
            composite_score=composite_score,
            relevance_factors=relevance_factors,
            specificity_factors=specificity_factors,
            completeness_factors=completeness_factors,
            authority_factors=authority_factors,
            explanation=explanation,
            confidence=confidence
        )
    
    def _calculate_relevance_score(
        self,
        result: VectorSearchResult,
        context: ScoringContext
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate relevance score based on query-content matching."""
        
        factors = {}
        
        # 1. Vector similarity (baseline)
        vector_similarity = result.similarity_score
        factors["vector_similarity"] = vector_similarity * 10  # Convert to 1-10 scale
        
        # 2. Query term coverage
        query_terms = set(context.query.lower().split())
        content_terms = set(result.content.lower().split())
        term_coverage = len(query_terms.intersection(content_terms)) / max(len(query_terms), 1)
        factors["term_coverage"] = term_coverage * 10
        
        # 3. Document type relevance
        doc_type = result.metadata.get("document_type", "").lower()
        query_lower = context.query.lower()
        
        type_relevance = 5.0  # Default
        if "requirement" in query_lower and doc_type in ["rfq", "specification"]:
            type_relevance = 8.0
        elif "offer" in query_lower and doc_type == "offer":
            type_relevance = 8.0
        elif "contract" in query_lower and doc_type == "contract":
            type_relevance = 8.0
        
        factors["document_type_relevance"] = type_relevance
        
        # 4. Section relevance
        section_path = result.metadata.get("section_path", [])
        section_relevance = 5.0
        
        if section_path:
            section_text = " ".join(section_path).lower()
            if any(term in section_text for term in query_terms):
                section_relevance = 7.5
        
        factors["section_relevance"] = section_relevance
        
        # Calculate weighted average
        relevance_score = (
            factors["vector_similarity"] * 0.4 +
            factors["term_coverage"] * 0.3 +
            factors["document_type_relevance"] * 0.2 +
            factors["section_relevance"] * 0.1
        )
        
        return min(max(relevance_score, 0.0), 10.0), factors
    
    def _calculate_specificity_score(
        self,
        result: VectorSearchResult,
        context: ScoringContext
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate specificity score based on detail level and precision."""
        
        factors = {}
        content = result.content
        
        # 1. Numerical specificity
        number_matches = len(re.findall(self._specificity_indicators["numbers"], content, re.IGNORECASE))
        factors["numerical_specificity"] = min(number_matches * 2, 10)
        
        # 2. Technical term density
        technical_terms = sum(1 for term in self._specificity_indicators["technical_terms"] 
                             if term.lower() in content.lower())
        factors["technical_density"] = min(technical_terms * 1.5, 10)
        
        # 3. Measurement specificity
        measurement_matches = len(re.findall(self._specificity_indicators["measurements"], content, re.IGNORECASE))
        factors["measurement_specificity"] = min(measurement_matches * 2.5, 10)
        
        # 4. Date specificity
        date_matches = len(re.findall(self._specificity_indicators["dates"], content, re.IGNORECASE))
        factors["date_specificity"] = min(date_matches * 3, 10)
        
        # 5. Content length (longer content often more specific)
        length_score = min(len(content) / 200, 10)  # Normalize by 200 chars
        factors["content_length"] = length_score
        
        # Calculate weighted average
        specificity_score = (
            factors["numerical_specificity"] * 0.25 +
            factors["technical_density"] * 0.25 +
            factors["measurement_specificity"] * 0.2 +
            factors["date_specificity"] * 0.15 +
            factors["content_length"] * 0.15
        )
        
        return min(max(specificity_score, 0.0), 10.0), factors
    
    def _calculate_completeness_score(
        self,
        result: VectorSearchResult,
        context: ScoringContext
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate completeness score based on coverage and comprehensiveness."""
        
        factors = {}
        content = result.content.lower()
        
        # 1. Section coverage
        section_indicators = sum(1 for section in self._completeness_indicators["sections"]
                               if section in content)
        factors["section_coverage"] = min(section_indicators * 2, 10)
        
        # 2. Coverage keywords
        coverage_keywords = sum(1 for keyword in self._completeness_indicators["coverage_keywords"]
                              if keyword in content)
        factors["coverage_keywords"] = min(coverage_keywords * 1.5, 10)
        
        # 3. Structure indicators
        structure_indicators = sum(1 for indicator in self._completeness_indicators["structure_indicators"]
                                 if indicator in content)
        factors["structure_indicators"] = min(structure_indicators * 1.5, 10)
        
        # 4. Query aspect coverage
        query_aspects = self._extract_query_aspects(context.query)
        covered_aspects = sum(1 for aspect in query_aspects if aspect.lower() in content)
        aspect_coverage = (covered_aspects / max(len(query_aspects), 1)) * 10
        factors["query_aspect_coverage"] = aspect_coverage
        
        # 5. Content depth (based on content length and structure)
        depth_score = min(math.log(len(result.content) + 1) / math.log(1000), 1) * 10
        factors["content_depth"] = depth_score
        
        # Calculate weighted average
        completeness_score = (
            factors["section_coverage"] * 0.2 +
            factors["coverage_keywords"] * 0.2 +
            factors["structure_indicators"] * 0.15 +
            factors["query_aspect_coverage"] * 0.3 +
            factors["content_depth"] * 0.15
        )
        
        return min(max(completeness_score, 0.0), 10.0), factors
    
    def _calculate_authority_score(
        self,
        result: VectorSearchResult,
        context: ScoringContext
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate authority score based on source credibility and reliability."""
        
        factors = {}
        content = result.content.lower()
        metadata = result.metadata
        
        # 1. Document type authority
        doc_type = metadata.get("document_type", "").lower()
        type_authority = {
            "specification": 9.0,
            "standard": 9.0,
            "regulation": 10.0,
            "contract": 8.5,
            "rfq": 8.0,
            "offer": 7.0,
            "report": 6.5,
            "other": 5.0
        }
        factors["document_type_authority"] = type_authority.get(doc_type, 5.0)
        
        # 2. Official content indicators
        official_indicators = sum(1 for indicator in self._authority_indicators["official_documents"]
                                if indicator in content)
        factors["official_indicators"] = min(official_indicators * 2, 10)
        
        # 3. Technical source indicators
        technical_indicators = sum(1 for indicator in self._authority_indicators["technical_sources"]
                                 if indicator in content)
        factors["technical_indicators"] = min(technical_indicators * 1.5, 10)
        
        # 4. Verification indicators
        verification_indicators = sum(1 for indicator in self._authority_indicators["verified_content"]
                                    if indicator in content)
        factors["verification_indicators"] = min(verification_indicators * 2.5, 10)
        
        # 5. Recency (if date available)
        recency_score = 7.0  # Default for unknown dates
        created_date = metadata.get("created_date")
        if created_date:
            try:
                if isinstance(created_date, str):
                    created_date = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                
                days_old = (datetime.now() - created_date.replace(tzinfo=None)).days
                if days_old <= self._authority_indicators["recent_content"]:
                    recency_score = 10.0 - (days_old / self._authority_indicators["recent_content"]) * 3
                else:
                    recency_score = 5.0
            except:
                pass
        
        factors["recency"] = max(recency_score, 0.0)
        
        # Calculate weighted average
        authority_score = (
            factors["document_type_authority"] * 0.3 +
            factors["official_indicators"] * 0.2 +
            factors["technical_indicators"] * 0.15 +
            factors["verification_indicators"] * 0.2 +
            factors["recency"] * 0.15
        )
        
        return min(max(authority_score, 0.0), 10.0), factors
    
    def _extract_query_aspects(self, query: str) -> List[str]:
        """Extract different aspects from the query for completeness assessment."""
        
        # Simple aspect extraction based on common patterns
        aspects = []
        
        # Split by common conjunctions
        for conjunction in [" and ", " or ", ", "]:
            if conjunction in query:
                aspects.extend(query.split(conjunction))
        
        # If no conjunctions found, treat whole query as one aspect
        if not aspects:
            aspects = [query]
        
        # Clean and filter aspects
        aspects = [aspect.strip() for aspect in aspects if len(aspect.strip()) > 3]
        
        return aspects
    
    def _generate_score_explanation(
        self,
        relevance: float,
        specificity: float,
        completeness: float,
        authority: float,
        relevance_factors: Dict[str, float],
        specificity_factors: Dict[str, float],
        completeness_factors: Dict[str, float],
        authority_factors: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation of the scoring."""
        
        explanation_parts = []
        
        # Relevance explanation
        if relevance >= 8.0:
            explanation_parts.append("Highly relevant to the query with strong content matching")
        elif relevance >= 6.0:
            explanation_parts.append("Good relevance with adequate content matching")
        else:
            explanation_parts.append("Limited relevance to the query")
        
        # Specificity explanation
        if specificity >= 8.0:
            explanation_parts.append("Contains specific details, numbers, and technical information")
        elif specificity >= 6.0:
            explanation_parts.append("Provides moderate level of detail and specificity")
        else:
            explanation_parts.append("General information with limited specific details")
        
        # Completeness explanation
        if completeness >= 8.0:
            explanation_parts.append("Comprehensive coverage of query aspects")
        elif completeness >= 6.0:
            explanation_parts.append("Covers most relevant aspects of the query")
        else:
            explanation_parts.append("Partial coverage of query aspects")
        
        # Authority explanation
        if authority >= 8.0:
            explanation_parts.append("High authority source with strong credibility")
        elif authority >= 6.0:
            explanation_parts.append("Reliable source with good credibility")
        else:
            explanation_parts.append("Standard source with basic credibility")
        
        return ". ".join(explanation_parts) + "."
    
    def _calculate_confidence(
        self,
        relevance_factors: Dict[str, float],
        specificity_factors: Dict[str, float],
        completeness_factors: Dict[str, float],
        authority_factors: Dict[str, float]
    ) -> float:
        """Calculate confidence in the scoring based on available factors."""
        
        # Count available factors
        total_factors = (
            len(relevance_factors) + len(specificity_factors) +
            len(completeness_factors) + len(authority_factors)
        )
        
        # Base confidence on factor availability and score consistency
        base_confidence = min(total_factors / 20, 1.0)  # Normalize by expected factor count
        
        # Adjust based on score variance (more consistent scores = higher confidence)
        all_scores = (
            list(relevance_factors.values()) + list(specificity_factors.values()) +
            list(completeness_factors.values()) + list(authority_factors.values())
        )
        
        if all_scores:
            score_variance = sum((score - sum(all_scores)/len(all_scores))**2 for score in all_scores) / len(all_scores)
            variance_penalty = min(score_variance / 25, 0.3)  # Max 30% penalty for high variance
            base_confidence -= variance_penalty
        
        return max(min(base_confidence, 1.0), 0.1)  # Ensure confidence is between 0.1 and 1.0


# Singleton instance
_multi_criteria_scoring_service: Optional[MultiCriteriaScoringService] = None


def get_multi_criteria_scoring_service() -> MultiCriteriaScoringService:
    """Get the singleton multi-criteria scoring service instance."""
    global _multi_criteria_scoring_service
    if _multi_criteria_scoring_service is None:
        _multi_criteria_scoring_service = MultiCriteriaScoringService()
    return _multi_criteria_scoring_service
