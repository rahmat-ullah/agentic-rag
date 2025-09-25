"""
Answer Quality Assessment System

This module implements comprehensive quality assessment algorithms for answer synthesis,
including completeness, accuracy, relevance, clarity, and overall quality scoring.
"""

import re
import json
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)


class QualityDimension(str, Enum):
    """Quality assessment dimensions."""
    
    COMPLETENESS = "completeness"          # How completely the answer addresses the query
    ACCURACY = "accuracy"                  # Factual correctness and reliability
    RELEVANCE = "relevance"                # Direct relevance to the query
    CLARITY = "clarity"                    # Readability and comprehensibility
    COHERENCE = "coherence"                # Logical flow and consistency
    CITATION_QUALITY = "citation_quality"  # Quality of source citations
    OBJECTIVITY = "objectivity"            # Balanced and unbiased presentation
    SPECIFICITY = "specificity"            # Level of detail and precision
    AUTHORITY = "authority"                # Credibility of sources and information
    TIMELINESS = "timeliness"              # Recency and currency of information


class AssessmentMethod(str, Enum):
    """Quality assessment methods."""
    
    HEURISTIC = "heuristic"                # Rule-based heuristic assessment
    STATISTICAL = "statistical"           # Statistical analysis methods
    LLM_BASED = "llm_based"               # LLM-powered assessment
    HYBRID = "hybrid"                     # Combination of methods
    MANUAL = "manual"                     # Human assessment


class QualityLevel(str, Enum):
    """Quality level classifications."""
    
    EXCELLENT = "excellent"               # 0.9 - 1.0
    GOOD = "good"                        # 0.7 - 0.89
    SATISFACTORY = "satisfactory"        # 0.5 - 0.69
    POOR = "poor"                        # 0.3 - 0.49
    UNACCEPTABLE = "unacceptable"        # 0.0 - 0.29


@dataclass
class QualityMetrics:
    """Detailed quality metrics for analysis."""
    
    # Text statistics
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_sentence_length: float = 0.0
    
    # Readability metrics
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    
    # Content metrics
    unique_words: int = 0
    vocabulary_richness: float = 0.0
    
    # Citation metrics
    citation_count: int = 0
    citation_density: float = 0.0
    unique_sources: int = 0
    
    # Structure metrics
    has_introduction: bool = False
    has_conclusion: bool = False
    has_headings: bool = False
    
    # Query alignment metrics
    query_term_coverage: float = 0.0
    semantic_similarity: float = 0.0


class QualityScore(BaseModel):
    """Individual quality dimension score."""
    
    dimension: QualityDimension = Field(..., description="Quality dimension")
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score (0-1)")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in score")
    method: AssessmentMethod = Field(default=AssessmentMethod.HEURISTIC, description="Assessment method")
    
    # Detailed breakdown
    sub_scores: Dict[str, float] = Field(default_factory=dict, description="Sub-component scores")
    explanation: str = Field(default="", description="Explanation of the score")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    
    # Improvement suggestions
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Identified weaknesses")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    
    @property
    def quality_level(self) -> QualityLevel:
        """Get quality level based on score."""
        if self.score >= 0.9:
            return QualityLevel.EXCELLENT
        elif self.score >= 0.7:
            return QualityLevel.GOOD
        elif self.score >= 0.5:
            return QualityLevel.SATISFACTORY
        elif self.score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE


class QualityAssessment(BaseModel):
    """Comprehensive quality assessment result."""
    
    assessment_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Overall assessment
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    overall_confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Overall confidence")
    quality_level: QualityLevel = Field(..., description="Overall quality level")
    
    # Dimension scores
    dimension_scores: List[QualityScore] = Field(..., description="Individual dimension scores")
    
    # Detailed metrics
    metrics: QualityMetrics = Field(default_factory=QualityMetrics, description="Detailed metrics")
    
    # Assessment metadata
    assessment_method: AssessmentMethod = Field(default=AssessmentMethod.HYBRID)
    assessment_duration_ms: int = Field(default=0, description="Assessment duration in milliseconds")
    
    # Summary
    summary: str = Field(default="", description="Quality assessment summary")
    key_findings: List[str] = Field(default_factory=list, description="Key findings")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    # Comparative analysis
    benchmark_comparison: Optional[Dict[str, float]] = Field(None, description="Comparison to benchmarks")
    
    def __init__(self, **data):
        # Set quality_level based on overall_score if not provided
        if 'quality_level' not in data and 'overall_score' in data:
            score = data['overall_score']
            if score >= 0.9:
                data['quality_level'] = QualityLevel.EXCELLENT
            elif score >= 0.7:
                data['quality_level'] = QualityLevel.GOOD
            elif score >= 0.5:
                data['quality_level'] = QualityLevel.SATISFACTORY
            elif score >= 0.3:
                data['quality_level'] = QualityLevel.POOR
            else:
                data['quality_level'] = QualityLevel.UNACCEPTABLE

        super().__init__(**data)


class QualityConfig(BaseModel):
    """Configuration for quality assessment."""
    
    # Assessment settings
    enabled_dimensions: List[QualityDimension] = Field(
        default_factory=lambda: list(QualityDimension),
        description="Enabled quality dimensions"
    )
    assessment_method: AssessmentMethod = Field(default=AssessmentMethod.HYBRID)
    
    # Scoring weights
    dimension_weights: Dict[QualityDimension, float] = Field(
        default_factory=lambda: {
            QualityDimension.COMPLETENESS: 0.25,
            QualityDimension.ACCURACY: 0.20,
            QualityDimension.RELEVANCE: 0.20,
            QualityDimension.CLARITY: 0.15,
            QualityDimension.CITATION_QUALITY: 0.10,
            QualityDimension.COHERENCE: 0.10
        }
    )
    
    # Thresholds
    min_acceptable_score: float = Field(default=0.5, ge=0.0, le=1.0)
    high_quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Assessment parameters
    enable_detailed_metrics: bool = Field(default=True)
    enable_improvement_suggestions: bool = Field(default=True)
    enable_benchmark_comparison: bool = Field(default=False)
    
    # Performance settings
    max_assessment_time_ms: int = Field(default=5000, gt=0)
    enable_caching: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=24, gt=0)
    
    @validator('dimension_weights')
    def validate_weights(cls, v):
        """Validate that weights sum to approximately 1.0."""
        total_weight = sum(v.values())
        if not (0.95 <= total_weight <= 1.05):
            raise ValueError(f"Dimension weights must sum to ~1.0, got {total_weight}")
        return v


class QualityAssessmentService:
    """Service for comprehensive answer quality assessment."""
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        
        # Assessment cache
        self._cache: Dict[str, Tuple[QualityAssessment, datetime]] = {}
        self._max_cache_size = 1000
        
        # Benchmarks and baselines
        self._benchmarks: Dict[str, Dict[QualityDimension, float]] = {}
        
        # Statistics
        self._stats = {
            "assessments_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_assessment_time_ms": 0,
            "quality_distribution": {level.value: 0 for level in QualityLevel}
        }
        
        logger.info(f"Quality assessment service initialized with {len(self.config.enabled_dimensions)} dimensions")
    
    async def assess_quality(
        self,
        answer: str,
        query: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QualityAssessment:
        """Perform comprehensive quality assessment of an answer."""
        
        start_time = datetime.now()
        
        # Check cache
        cache_key = self._generate_cache_key(answer, query)
        if self.config.enable_caching:
            cached_assessment = self._get_cached_assessment(cache_key)
            if cached_assessment:
                self._stats["cache_hits"] += 1
                return cached_assessment
        
        self._stats["cache_misses"] += 1
        
        # Calculate detailed metrics
        metrics = self._calculate_metrics(answer, query, sources)
        
        # Assess each dimension
        dimension_scores = []
        for dimension in self.config.enabled_dimensions:
            score = await self._assess_dimension(dimension, answer, query, sources, metrics, context)
            dimension_scores.append(score)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        overall_confidence = self._calculate_overall_confidence(dimension_scores)
        
        # Generate summary and recommendations
        summary = self._generate_summary(dimension_scores, metrics)
        key_findings = self._extract_key_findings(dimension_scores)
        recommendations = self._generate_recommendations(dimension_scores)
        
        # Create assessment
        assessment_duration = int((datetime.now() - start_time).total_seconds() * 1000)
        
        assessment = QualityAssessment(
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            dimension_scores=dimension_scores,
            metrics=metrics,
            assessment_method=self.config.assessment_method,
            assessment_duration_ms=assessment_duration,
            summary=summary,
            key_findings=key_findings,
            recommendations=recommendations
        )
        
        # Add benchmark comparison if enabled
        if self.config.enable_benchmark_comparison:
            assessment.benchmark_comparison = self._compare_to_benchmarks(dimension_scores)
        
        # Cache the assessment
        if self.config.enable_caching:
            self._cache_assessment(cache_key, assessment)
        
        # Update statistics
        self._update_statistics(assessment)
        
        logger.debug(f"Quality assessment completed in {assessment_duration}ms with score {overall_score:.3f}")

        return assessment

    def _calculate_metrics(
        self,
        answer: str,
        query: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> QualityMetrics:
        """Calculate detailed quality metrics."""

        metrics = QualityMetrics()

        if not answer:
            return metrics

        # Basic text statistics
        words = answer.split()
        sentences = re.split(r'[.!?]+', answer)
        paragraphs = answer.split('\n\n')

        metrics.word_count = len(words)
        metrics.sentence_count = len([s for s in sentences if s.strip()])
        metrics.paragraph_count = len([p for p in paragraphs if p.strip()])

        if metrics.sentence_count > 0:
            metrics.avg_sentence_length = metrics.word_count / metrics.sentence_count

        # Vocabulary metrics
        unique_words = set(word.lower().strip('.,!?;:') for word in words if word.strip())
        metrics.unique_words = len(unique_words)

        if metrics.word_count > 0:
            metrics.vocabulary_richness = metrics.unique_words / metrics.word_count

        # Readability metrics
        metrics.flesch_reading_ease = self._calculate_flesch_reading_ease(answer)
        metrics.flesch_kincaid_grade = self._calculate_flesch_kincaid_grade(answer)

        # Citation metrics
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, answer)
        metrics.citation_count = len(citations)
        metrics.unique_sources = len(set(citations))

        if metrics.word_count > 0:
            metrics.citation_density = metrics.citation_count / metrics.word_count

        # Structure metrics
        metrics.has_introduction = self._has_introduction(answer)
        metrics.has_conclusion = self._has_conclusion(answer)
        metrics.has_headings = self._has_headings(answer)

        # Query alignment metrics
        metrics.query_term_coverage = self._calculate_query_coverage(answer, query)
        metrics.semantic_similarity = self._calculate_semantic_similarity(answer, query)

        return metrics

    def _calculate_flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score."""

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        syllables = sum(self._count_syllables(word) for word in words)

        if not words or not sentences:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0.0, min(100.0, score))

    def _calculate_flesch_kincaid_grade(self, text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level."""

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        syllables = sum(self._count_syllables(word) for word in words)

        if not words or not sentences:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        return max(0.0, grade)

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""

        word = word.lower().strip('.,!?;:')
        if not word:
            return 0

        # Simple syllable counting heuristic
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def _has_introduction(self, text: str) -> bool:
        """Check if text has an introduction."""

        intro_indicators = [
            'introduction', 'overview', 'summary', 'in this', 'this document',
            'to begin', 'first', 'initially', 'the purpose'
        ]

        first_paragraph = text.split('\n\n')[0].lower() if text else ""
        return any(indicator in first_paragraph for indicator in intro_indicators)

    def _has_conclusion(self, text: str) -> bool:
        """Check if text has a conclusion."""

        conclusion_indicators = [
            'conclusion', 'in summary', 'to summarize', 'in conclusion',
            'finally', 'to conclude', 'overall', 'in closing'
        ]

        last_paragraph = text.split('\n\n')[-1].lower() if text else ""
        return any(indicator in last_paragraph for indicator in conclusion_indicators)

    def _has_headings(self, text: str) -> bool:
        """Check if text has headings or structure."""

        # Look for markdown-style headings or numbered sections
        heading_patterns = [
            r'^#+\s+',  # Markdown headings
            r'^\d+\.\s+',  # Numbered sections
            r'^[A-Z][A-Z\s]+:',  # ALL CAPS headings
        ]

        lines = text.split('\n')
        for line in lines:
            for pattern in heading_patterns:
                if re.match(pattern, line.strip()):
                    return True

        return False

    def _calculate_query_coverage(self, answer: str, query: str) -> float:
        """Calculate how well the answer covers query terms."""

        if not query or not answer:
            return 0.0

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = query_words - stop_words

        if not query_words:
            return 1.0

        covered_words = query_words.intersection(answer_words)
        return len(covered_words) / len(query_words)

    def _calculate_semantic_similarity(self, answer: str, query: str) -> float:
        """Calculate semantic similarity between answer and query (simplified)."""

        if not query or not answer:
            return 0.0

        # Simple Jaccard similarity as a proxy for semantic similarity
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        intersection = query_words.intersection(answer_words)
        union = query_words.union(answer_words)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    async def _assess_dimension(
        self,
        dimension: QualityDimension,
        answer: str,
        query: str,
        sources: Optional[List[Dict[str, Any]]],
        metrics: QualityMetrics,
        context: Optional[Dict[str, Any]]
    ) -> QualityScore:
        """Assess a specific quality dimension."""

        if dimension == QualityDimension.COMPLETENESS:
            return self._assess_completeness(answer, query, sources, metrics)
        elif dimension == QualityDimension.ACCURACY:
            return self._assess_accuracy(answer, sources, metrics)
        elif dimension == QualityDimension.RELEVANCE:
            return self._assess_relevance(answer, query, metrics)
        elif dimension == QualityDimension.CLARITY:
            return self._assess_clarity(answer, metrics)
        elif dimension == QualityDimension.COHERENCE:
            return self._assess_coherence(answer, metrics)
        elif dimension == QualityDimension.CITATION_QUALITY:
            return self._assess_citation_quality(answer, sources, metrics)
        elif dimension == QualityDimension.OBJECTIVITY:
            return self._assess_objectivity(answer, metrics)
        elif dimension == QualityDimension.SPECIFICITY:
            return self._assess_specificity(answer, metrics)
        elif dimension == QualityDimension.AUTHORITY:
            return self._assess_authority(answer, sources, metrics)
        elif dimension == QualityDimension.TIMELINESS:
            return self._assess_timeliness(answer, sources, metrics)
        else:
            # Default assessment
            return QualityScore(
                dimension=dimension,
                score=0.5,
                explanation=f"Assessment not implemented for {dimension.value}"
            )

    def _assess_completeness(
        self,
        answer: str,
        query: str,
        sources: Optional[List[Dict[str, Any]]],
        metrics: QualityMetrics
    ) -> QualityScore:
        """Assess completeness of the answer."""

        sub_scores = {}
        evidence = []
        strengths = []
        weaknesses = []
        suggestions = []

        # Query coverage assessment
        query_coverage = metrics.query_term_coverage
        sub_scores["query_coverage"] = query_coverage

        if query_coverage >= 0.8:
            strengths.append("Excellent coverage of query terms")
        elif query_coverage < 0.5:
            weaknesses.append("Poor coverage of query terms")
            suggestions.append("Include more specific terms from the query")

        # Length appropriateness
        word_count = metrics.word_count
        length_score = min(1.0, word_count / 200)  # Assume 200 words is good baseline
        sub_scores["length_appropriateness"] = length_score

        if word_count < 50:
            weaknesses.append("Answer is too brief")
            suggestions.append("Provide more detailed explanation")
        elif word_count > 500:
            evidence.append("Comprehensive length")

        # Source utilization
        if sources:
            source_count = len(sources)
            citation_coverage = min(1.0, metrics.citation_count / source_count)
            sub_scores["source_utilization"] = citation_coverage

            if citation_coverage >= 0.8:
                strengths.append("Good utilization of available sources")
            elif citation_coverage < 0.5:
                weaknesses.append("Underutilization of available sources")
                suggestions.append("Include more citations from provided sources")
        else:
            sub_scores["source_utilization"] = 0.5

        # Structure completeness
        structure_score = 0.0
        if metrics.has_introduction:
            structure_score += 0.3
        if metrics.has_conclusion:
            structure_score += 0.3
        if metrics.paragraph_count > 1:
            structure_score += 0.4

        sub_scores["structure_completeness"] = structure_score

        if structure_score >= 0.7:
            strengths.append("Well-structured answer")
        else:
            suggestions.append("Improve answer structure with clear introduction and conclusion")

        # Calculate overall completeness score
        overall_score = (
            query_coverage * 0.4 +
            length_score * 0.3 +
            sub_scores["source_utilization"] * 0.2 +
            structure_score * 0.1
        )

        explanation = f"Completeness score based on query coverage ({query_coverage:.2f}), length appropriateness ({length_score:.2f}), source utilization ({sub_scores['source_utilization']:.2f}), and structure ({structure_score:.2f})"

        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=overall_score,
            sub_scores=sub_scores,
            explanation=explanation,
            evidence=evidence,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions
        )

    def _assess_accuracy(
        self,
        answer: str,
        sources: Optional[List[Dict[str, Any]]],
        metrics: QualityMetrics
    ) -> QualityScore:
        """Assess accuracy of the answer."""

        sub_scores = {}
        evidence = []
        strengths = []
        weaknesses = []
        suggestions = []

        # Citation density as proxy for verifiability
        citation_density = metrics.citation_density
        citation_score = min(1.0, citation_density * 20)  # Scale citation density
        sub_scores["citation_support"] = citation_score

        if citation_score >= 0.7:
            strengths.append("Well-supported with citations")
        elif citation_score < 0.3:
            weaknesses.append("Insufficient citation support")
            suggestions.append("Add more citations to support claims")

        # Source quality assessment
        if sources:
            # Assess source diversity and quality
            source_types = set()
            total_confidence = 0.0

            for source in sources:
                source_type = source.get('document_type', 'unknown')
                source_types.add(source_type)
                total_confidence += source.get('confidence_score', 0.5)

            source_diversity = min(1.0, len(source_types) / 3)  # Assume 3 types is good
            avg_source_confidence = total_confidence / len(sources) if sources else 0.5

            sub_scores["source_diversity"] = source_diversity
            sub_scores["source_confidence"] = avg_source_confidence

            if source_diversity >= 0.7:
                strengths.append("Good diversity of source types")

            if avg_source_confidence >= 0.8:
                strengths.append("High-confidence sources")
            elif avg_source_confidence < 0.6:
                weaknesses.append("Low-confidence sources")
        else:
            sub_scores["source_diversity"] = 0.5
            sub_scores["source_confidence"] = 0.5

        # Factual consistency indicators
        consistency_score = self._assess_factual_consistency(answer)
        sub_scores["factual_consistency"] = consistency_score

        if consistency_score >= 0.8:
            strengths.append("Factually consistent content")
        elif consistency_score < 0.6:
            weaknesses.append("Potential factual inconsistencies")
            suggestions.append("Review content for factual accuracy")

        # Calculate overall accuracy score
        overall_score = (
            citation_score * 0.3 +
            sub_scores["source_confidence"] * 0.3 +
            consistency_score * 0.25 +
            sub_scores["source_diversity"] * 0.15
        )

        explanation = f"Accuracy score based on citation support ({citation_score:.2f}), source confidence ({sub_scores['source_confidence']:.2f}), factual consistency ({consistency_score:.2f}), and source diversity ({sub_scores['source_diversity']:.2f})"

        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=overall_score,
            sub_scores=sub_scores,
            explanation=explanation,
            evidence=evidence,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions
        )

    def _assess_relevance(
        self,
        answer: str,
        query: str,
        metrics: QualityMetrics
    ) -> QualityScore:
        """Assess relevance of the answer to the query."""

        sub_scores = {}
        evidence = []
        strengths = []
        weaknesses = []
        suggestions = []

        # Semantic similarity
        semantic_similarity = metrics.semantic_similarity
        sub_scores["semantic_similarity"] = semantic_similarity

        # Query term coverage
        query_coverage = metrics.query_term_coverage
        sub_scores["query_term_coverage"] = query_coverage

        # Topic focus assessment
        topic_focus = self._assess_topic_focus(answer, query)
        sub_scores["topic_focus"] = topic_focus

        # Directness assessment
        directness = self._assess_answer_directness(answer, query)
        sub_scores["directness"] = directness

        # Evaluate scores
        if semantic_similarity >= 0.7:
            strengths.append("High semantic similarity to query")
        elif semantic_similarity < 0.4:
            weaknesses.append("Low semantic similarity to query")
            suggestions.append("Focus more directly on the query topic")

        if query_coverage >= 0.8:
            strengths.append("Excellent coverage of query terms")
        elif query_coverage < 0.5:
            weaknesses.append("Poor coverage of query terms")
            suggestions.append("Address more aspects of the query")

        if topic_focus >= 0.8:
            strengths.append("Well-focused on the topic")
        elif topic_focus < 0.6:
            weaknesses.append("Answer lacks focus")
            suggestions.append("Stay more focused on the main topic")

        # Calculate overall relevance score
        overall_score = (
            semantic_similarity * 0.3 +
            query_coverage * 0.3 +
            topic_focus * 0.25 +
            directness * 0.15
        )

        explanation = f"Relevance score based on semantic similarity ({semantic_similarity:.2f}), query coverage ({query_coverage:.2f}), topic focus ({topic_focus:.2f}), and directness ({directness:.2f})"

        return QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=overall_score,
            sub_scores=sub_scores,
            explanation=explanation,
            evidence=evidence,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions
        )

    def _assess_clarity(self, answer: str, metrics: QualityMetrics) -> QualityScore:
        """Assess clarity and readability of the answer."""

        sub_scores = {}
        evidence = []
        strengths = []
        weaknesses = []
        suggestions = []

        # Readability scores
        flesch_score = metrics.flesch_reading_ease
        readability_score = flesch_score / 100.0  # Normalize to 0-1
        sub_scores["readability"] = readability_score

        # Sentence length appropriateness
        avg_sentence_length = metrics.avg_sentence_length
        sentence_score = 1.0 - min(1.0, abs(avg_sentence_length - 20) / 20)  # Optimal around 20 words
        sub_scores["sentence_length"] = sentence_score

        # Structure clarity
        structure_score = 0.0
        if metrics.has_headings:
            structure_score += 0.4
        if metrics.paragraph_count > 1:
            structure_score += 0.3
        if metrics.sentence_count > 3:
            structure_score += 0.3

        sub_scores["structure_clarity"] = structure_score

        # Vocabulary appropriateness
        vocab_score = min(1.0, metrics.vocabulary_richness * 2)  # Scale vocabulary richness
        sub_scores["vocabulary"] = vocab_score

        # Evaluate scores
        if readability_score >= 0.7:
            strengths.append("Good readability")
        elif readability_score < 0.4:
            weaknesses.append("Poor readability")
            suggestions.append("Simplify sentence structure and vocabulary")

        if sentence_score >= 0.8:
            strengths.append("Appropriate sentence length")
        elif sentence_score < 0.6:
            weaknesses.append("Sentence length issues")
            suggestions.append("Vary sentence length for better flow")

        if structure_score >= 0.7:
            strengths.append("Clear structure")
        else:
            suggestions.append("Improve structure with headings and paragraphs")

        # Calculate overall clarity score
        overall_score = (
            readability_score * 0.4 +
            sentence_score * 0.25 +
            structure_score * 0.25 +
            vocab_score * 0.1
        )

        explanation = f"Clarity score based on readability ({readability_score:.2f}), sentence length ({sentence_score:.2f}), structure ({structure_score:.2f}), and vocabulary ({vocab_score:.2f})"

        return QualityScore(
            dimension=QualityDimension.CLARITY,
            score=overall_score,
            sub_scores=sub_scores,
            explanation=explanation,
            evidence=evidence,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions
        )

    def _assess_coherence(self, answer: str, metrics: QualityMetrics) -> QualityScore:
        """Assess logical flow and coherence of the answer."""

        sub_scores = {}
        evidence = []
        strengths = []
        weaknesses = []
        suggestions = []

        # Transition indicators
        transition_score = self._assess_transitions(answer)
        sub_scores["transitions"] = transition_score

        # Logical flow
        flow_score = self._assess_logical_flow(answer)
        sub_scores["logical_flow"] = flow_score

        # Consistency
        consistency_score = self._assess_internal_consistency(answer)
        sub_scores["consistency"] = consistency_score

        # Paragraph coherence
        paragraph_score = self._assess_paragraph_coherence(answer)
        sub_scores["paragraph_coherence"] = paragraph_score

        # Evaluate scores
        if transition_score >= 0.7:
            strengths.append("Good use of transitions")
        elif transition_score < 0.5:
            weaknesses.append("Lacks smooth transitions")
            suggestions.append("Add transition words and phrases")

        if flow_score >= 0.8:
            strengths.append("Logical flow")
        elif flow_score < 0.6:
            weaknesses.append("Poor logical flow")
            suggestions.append("Reorganize content for better logical progression")

        # Calculate overall coherence score
        overall_score = (
            flow_score * 0.35 +
            consistency_score * 0.25 +
            transition_score * 0.25 +
            paragraph_score * 0.15
        )

        explanation = f"Coherence score based on logical flow ({flow_score:.2f}), consistency ({consistency_score:.2f}), transitions ({transition_score:.2f}), and paragraph coherence ({paragraph_score:.2f})"

        return QualityScore(
            dimension=QualityDimension.COHERENCE,
            score=overall_score,
            sub_scores=sub_scores,
            explanation=explanation,
            evidence=evidence,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions
        )

    def _assess_citation_quality(
        self,
        answer: str,
        sources: Optional[List[Dict[str, Any]]],
        metrics: QualityMetrics
    ) -> QualityScore:
        """Assess quality of citations in the answer."""

        sub_scores = {}
        evidence = []
        strengths = []
        weaknesses = []
        suggestions = []

        # Citation coverage
        if sources and len(sources) > 0:
            citation_coverage = min(1.0, metrics.citation_count / len(sources))
            sub_scores["citation_coverage"] = citation_coverage

            if citation_coverage >= 0.8:
                strengths.append("Good citation coverage")
            elif citation_coverage < 0.5:
                weaknesses.append("Poor citation coverage")
                suggestions.append("Add more citations to support claims")
        else:
            sub_scores["citation_coverage"] = 0.5

        # Citation format consistency
        format_score = self._assess_citation_format(answer)
        sub_scores["format_consistency"] = format_score

        # Citation placement appropriateness
        placement_score = self._assess_citation_placement(answer)
        sub_scores["placement"] = placement_score

        # Citation density appropriateness
        density_score = 1.0 - min(1.0, abs(metrics.citation_density - 0.05) / 0.05)  # Optimal around 5%
        sub_scores["density"] = density_score

        # Evaluate scores
        if format_score >= 0.8:
            strengths.append("Consistent citation format")
        elif format_score < 0.6:
            weaknesses.append("Inconsistent citation format")
            suggestions.append("Use consistent citation format throughout")

        if placement_score >= 0.7:
            strengths.append("Appropriate citation placement")
        else:
            suggestions.append("Improve citation placement for better flow")

        # Calculate overall citation quality score
        overall_score = (
            sub_scores["citation_coverage"] * 0.4 +
            format_score * 0.25 +
            placement_score * 0.2 +
            density_score * 0.15
        )

        explanation = f"Citation quality score based on coverage ({sub_scores['citation_coverage']:.2f}), format consistency ({format_score:.2f}), placement ({placement_score:.2f}), and density ({density_score:.2f})"

        return QualityScore(
            dimension=QualityDimension.CITATION_QUALITY,
            score=overall_score,
            sub_scores=sub_scores,
            explanation=explanation,
            evidence=evidence,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions
        )

    # Simplified implementations for remaining dimensions
    def _assess_objectivity(self, answer: str, metrics: QualityMetrics) -> QualityScore:
        """Assess objectivity and bias in the answer."""

        # Simple heuristic-based assessment
        bias_indicators = ['definitely', 'obviously', 'clearly the best', 'without doubt', 'always', 'never']
        bias_count = sum(1 for indicator in bias_indicators if indicator.lower() in answer.lower())

        objectivity_score = max(0.0, 1.0 - (bias_count * 0.2))

        return QualityScore(
            dimension=QualityDimension.OBJECTIVITY,
            score=objectivity_score,
            explanation=f"Objectivity assessment based on bias indicators (found {bias_count})"
        )

    def _assess_specificity(self, answer: str, metrics: QualityMetrics) -> QualityScore:
        """Assess level of detail and specificity."""

        # Assess based on numbers, dates, specific terms
        specific_patterns = [
            r'\d+%',  # Percentages
            r'\$\d+',  # Dollar amounts
            r'\d{4}',  # Years
            r'\d+\.\d+',  # Decimal numbers
        ]

        specificity_count = sum(len(re.findall(pattern, answer)) for pattern in specific_patterns)
        specificity_score = min(1.0, specificity_count / 5)  # Normalize

        return QualityScore(
            dimension=QualityDimension.SPECIFICITY,
            score=specificity_score,
            explanation=f"Specificity based on concrete details (found {specificity_count} specific elements)"
        )

    def _assess_authority(self, answer: str, sources: Optional[List[Dict[str, Any]]], metrics: QualityMetrics) -> QualityScore:
        """Assess authority and credibility of sources."""

        if not sources:
            return QualityScore(
                dimension=QualityDimension.AUTHORITY,
                score=0.5,
                explanation="No sources provided for authority assessment"
            )

        # Assess based on source types and confidence
        authority_score = 0.0
        for source in sources:
            doc_type = source.get('document_type', 'unknown')
            confidence = source.get('confidence_score', 0.5)

            # Weight different document types
            type_weights = {
                'official': 1.0,
                'policy': 0.9,
                'report': 0.8,
                'rfq': 0.7,
                'offer': 0.6,
                'unknown': 0.3
            }

            type_weight = type_weights.get(doc_type, 0.5)
            authority_score += (confidence * type_weight)

        authority_score = authority_score / len(sources) if sources else 0.5

        return QualityScore(
            dimension=QualityDimension.AUTHORITY,
            score=authority_score,
            explanation=f"Authority score based on {len(sources)} sources with average weighted confidence"
        )

    def _assess_timeliness(self, answer: str, sources: Optional[List[Dict[str, Any]]], metrics: QualityMetrics) -> QualityScore:
        """Assess timeliness and currency of information."""

        if not sources:
            return QualityScore(
                dimension=QualityDimension.TIMELINESS,
                score=0.5,
                explanation="No sources provided for timeliness assessment"
            )

        # Assess based on source dates
        current_year = datetime.now().year
        timeliness_scores = []

        for source in sources:
            pub_date = source.get('publication_date')
            if pub_date:
                if isinstance(pub_date, str):
                    try:
                        pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    except:
                        continue

                years_old = current_year - pub_date.year
                timeliness = max(0.0, 1.0 - (years_old * 0.1))  # Decay by 10% per year
                timeliness_scores.append(timeliness)

        avg_timeliness = sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 0.5

        return QualityScore(
            dimension=QualityDimension.TIMELINESS,
            score=avg_timeliness,
            explanation=f"Timeliness based on {len(timeliness_scores)} dated sources"
        )

    # Utility methods for assessment
    def _assess_factual_consistency(self, answer: str) -> float:
        """Assess factual consistency (simplified heuristic)."""

        # Look for contradictory statements (simplified approach)
        contradiction_indicators = [
            'however', 'but', 'although', 'despite', 'contrary to',
            'on the other hand', 'in contrast', 'nevertheless'
        ]

        # Count potential contradiction indicators
        contradiction_count = 0
        for indicator in contradiction_indicators:
            if indicator in answer.lower():
                contradiction_count += 1

        # Simple consistency assessment based on contradiction indicators
        # Fewer contradictions = higher consistency
        consistency_score = max(0.0, 1.0 - (contradiction_count * 0.1))
        return min(1.0, consistency_score)

    def _assess_topic_focus(self, answer: str, query: str) -> float:
        """Assess how well the answer stays focused on the topic."""

        # Simple implementation based on topic word frequency
        query_words = set(query.lower().split())
        answer_words = answer.lower().split()

        if not query_words or not answer_words:
            return 0.5

        # Calculate topic word density
        topic_word_count = sum(1 for word in answer_words if word in query_words)
        topic_density = topic_word_count / len(answer_words)

        # Normalize to reasonable range
        focus_score = min(1.0, topic_density * 10)
        return focus_score

    def _assess_answer_directness(self, answer: str, query: str) -> float:
        """Assess how directly the answer addresses the query."""

        # Look for direct answer indicators
        direct_indicators = ['the answer is', 'yes,', 'no,', 'in summary', 'to answer']

        directness_score = 0.5  # Base score

        for indicator in direct_indicators:
            if indicator in answer.lower():
                directness_score += 0.1

        # Check if answer starts directly
        first_sentence = answer.split('.')[0].lower() if answer else ""
        query_words = query.lower().split()

        if any(word in first_sentence for word in query_words[:3]):  # First 3 query words
            directness_score += 0.2

        return min(1.0, directness_score)

    def _assess_transitions(self, answer: str) -> float:
        """Assess use of transition words and phrases."""

        transition_words = [
            'however', 'therefore', 'furthermore', 'additionally', 'moreover',
            'consequently', 'meanwhile', 'nevertheless', 'in contrast',
            'on the other hand', 'as a result', 'in conclusion'
        ]

        transition_count = sum(1 for word in transition_words if word in answer.lower())

        # Normalize based on text length
        sentences = len(re.split(r'[.!?]+', answer))
        if sentences > 0:
            transition_density = transition_count / sentences
            return min(1.0, transition_density * 3)  # Scale appropriately

        return 0.0

    def _assess_logical_flow(self, answer: str) -> float:
        """Assess logical flow of the answer."""

        # Simple heuristic based on paragraph structure and flow indicators
        paragraphs = answer.split('\n\n')

        if len(paragraphs) <= 1:
            return 0.6  # Single paragraph gets moderate score

        # Look for logical progression indicators
        flow_indicators = [
            'first', 'second', 'third', 'finally', 'next', 'then',
            'initially', 'subsequently', 'lastly'
        ]

        flow_score = 0.5  # Base score

        for indicator in flow_indicators:
            if indicator in answer.lower():
                flow_score += 0.1

        # Bonus for multiple paragraphs
        if len(paragraphs) > 2:
            flow_score += 0.2

        return min(1.0, flow_score)

    def _assess_internal_consistency(self, answer: str) -> float:
        """Assess internal consistency of the answer."""

        # Look for consistent terminology and concepts
        # This is a simplified implementation

        sentences = re.split(r'[.!?]+', answer)
        if len(sentences) < 2:
            return 1.0  # Single sentence is consistent by default

        # Check for consistent use of key terms
        key_terms = []
        for sentence in sentences:
            words = sentence.split()
            # Extract potential key terms (capitalized words, longer words)
            terms = [word for word in words if len(word) > 5 and word[0].isupper()]
            key_terms.extend(terms)

        if not key_terms:
            return 0.8  # No key terms found, assume reasonable consistency

        # Simple consistency check based on term repetition
        unique_terms = set(key_terms)
        consistency_score = len(unique_terms) / len(key_terms) if key_terms else 1.0

        # Invert because more repetition = more consistency
        return 1.0 - consistency_score + 0.5  # Adjust range

    def _assess_paragraph_coherence(self, answer: str) -> float:
        """Assess coherence within paragraphs."""

        paragraphs = answer.split('\n\n')

        if len(paragraphs) <= 1:
            return 0.7  # Single paragraph gets moderate score

        coherence_scores = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            sentences = re.split(r'[.!?]+', paragraph)
            if len(sentences) < 2:
                coherence_scores.append(0.8)
                continue

            # Simple coherence assessment based on sentence length variation
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

            if sentence_lengths:
                avg_length = sum(sentence_lengths) / len(sentence_lengths)
                variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)

                # Lower variance = better coherence (more consistent sentence structure)
                coherence = max(0.0, 1.0 - (variance / 100))
                coherence_scores.append(coherence)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5

    def _assess_citation_format(self, answer: str) -> float:
        """Assess consistency of citation format."""

        # Look for citation patterns
        patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d+\)',  # (1), (2), etc.
            r'\(\w+,\s*\d{4}\)',  # (Author, 2023)
        ]

        citation_counts = {}
        for i, pattern in enumerate(patterns):
            citations = re.findall(pattern, answer)
            if citations:
                citation_counts[i] = len(citations)

        if not citation_counts:
            return 0.5  # No citations found

        # Check if one format dominates
        total_citations = sum(citation_counts.values())
        max_format_count = max(citation_counts.values())

        consistency = max_format_count / total_citations if total_citations > 0 else 0.5
        return consistency

    def _assess_citation_placement(self, answer: str) -> float:
        """Assess appropriateness of citation placement."""

        citations = re.findall(r'\[\d+\]', answer)
        if not citations:
            return 0.5

        sentences = re.split(r'[.!?]+', answer)
        placement_score = 0.0

        for sentence in sentences:
            if any(citation in sentence for citation in citations):
                # Check if citation is at the end of sentence (preferred)
                if sentence.strip().endswith(tuple(citations)):
                    placement_score += 1.0
                else:
                    placement_score += 0.7  # Mid-sentence is okay but not ideal

        return placement_score / len(sentences) if sentences else 0.5

    def _calculate_overall_score(self, dimension_scores: List[QualityScore]) -> float:
        """Calculate weighted overall quality score."""

        if not dimension_scores:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for score in dimension_scores:
            weight = self.config.dimension_weights.get(score.dimension, 0.1)
            weighted_sum += score.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_overall_confidence(self, dimension_scores: List[QualityScore]) -> float:
        """Calculate overall confidence in the assessment."""

        if not dimension_scores:
            return 0.0

        confidences = [score.confidence for score in dimension_scores]
        return sum(confidences) / len(confidences)

    def _generate_summary(self, dimension_scores: List[QualityScore], metrics: QualityMetrics) -> str:
        """Generate quality assessment summary."""

        if not dimension_scores:
            return "No quality assessment performed."

        overall_score = self._calculate_overall_score(dimension_scores)
        quality_level = QualityLevel.EXCELLENT if overall_score >= 0.9 else \
                      QualityLevel.GOOD if overall_score >= 0.7 else \
                      QualityLevel.SATISFACTORY if overall_score >= 0.5 else \
                      QualityLevel.POOR if overall_score >= 0.3 else \
                      QualityLevel.UNACCEPTABLE

        summary_parts = [
            f"Overall quality: {quality_level.value.title()} ({overall_score:.2f})",
            f"Answer length: {metrics.word_count} words, {metrics.sentence_count} sentences",
            f"Citations: {metrics.citation_count} citations from {metrics.unique_sources} sources"
        ]

        # Add top strengths and weaknesses
        all_strengths = []
        all_weaknesses = []

        for score in dimension_scores:
            all_strengths.extend(score.strengths)
            all_weaknesses.extend(score.weaknesses)

        if all_strengths:
            summary_parts.append(f"Key strengths: {', '.join(all_strengths[:3])}")

        if all_weaknesses:
            summary_parts.append(f"Areas for improvement: {', '.join(all_weaknesses[:3])}")

        return ". ".join(summary_parts) + "."

    def _extract_key_findings(self, dimension_scores: List[QualityScore]) -> List[str]:
        """Extract key findings from dimension assessments."""

        findings = []

        # Find highest and lowest scoring dimensions
        if dimension_scores:
            highest_score = max(dimension_scores, key=lambda x: x.score)
            lowest_score = min(dimension_scores, key=lambda x: x.score)

            findings.append(f"Strongest dimension: {highest_score.dimension.value} ({highest_score.score:.2f})")
            findings.append(f"Weakest dimension: {lowest_score.dimension.value} ({lowest_score.score:.2f})")

            # Add specific findings from high-confidence assessments
            for score in dimension_scores:
                if score.confidence >= 0.8 and score.evidence:
                    findings.extend(score.evidence[:2])  # Add top 2 evidence items

        return findings[:5]  # Limit to top 5 findings

    def _generate_recommendations(self, dimension_scores: List[QualityScore]) -> List[str]:
        """Generate improvement recommendations."""

        recommendations = []

        # Collect suggestions from low-scoring dimensions
        low_scoring_dimensions = [score for score in dimension_scores if score.score < 0.6]

        for score in low_scoring_dimensions:
            recommendations.extend(score.suggestions[:2])  # Add top 2 suggestions per dimension

        # Add general recommendations based on overall patterns
        if any(score.dimension == QualityDimension.CITATION_QUALITY and score.score < 0.6 for score in dimension_scores):
            recommendations.append("Improve citation practices and source attribution")

        if any(score.dimension == QualityDimension.CLARITY and score.score < 0.6 for score in dimension_scores):
            recommendations.append("Enhance readability and structure")

        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:8]

    def _compare_to_benchmarks(self, dimension_scores: List[QualityScore]) -> Dict[str, float]:
        """Compare scores to established benchmarks."""

        # Default benchmarks (these could be loaded from configuration or database)
        default_benchmarks = {
            QualityDimension.COMPLETENESS: 0.75,
            QualityDimension.ACCURACY: 0.80,
            QualityDimension.RELEVANCE: 0.78,
            QualityDimension.CLARITY: 0.70,
            QualityDimension.COHERENCE: 0.72,
            QualityDimension.CITATION_QUALITY: 0.65
        }

        comparison = {}

        for score in dimension_scores:
            benchmark = default_benchmarks.get(score.dimension, 0.7)
            comparison[score.dimension.value] = score.score - benchmark

        return comparison

    def _generate_cache_key(self, answer: str, query: str) -> str:
        """Generate cache key for assessment."""

        import hashlib

        key_components = [answer[:100], query[:50]]  # Use truncated versions for key
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_assessment(self, cache_key: str) -> Optional[QualityAssessment]:
        """Get cached assessment if available and not expired."""

        if cache_key not in self._cache:
            return None

        assessment, cached_time = self._cache[cache_key]

        # Check if cache is expired
        cache_age_hours = (datetime.now(timezone.utc) - cached_time).total_seconds() / 3600
        if cache_age_hours > self.config.cache_ttl_hours:
            del self._cache[cache_key]
            return None

        return assessment

    def _cache_assessment(self, cache_key: str, assessment: QualityAssessment) -> None:
        """Cache the quality assessment."""

        # Clean cache if it's getting too large
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entries
            sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:self._max_cache_size // 4]:
                del self._cache[key]

        self._cache[cache_key] = (assessment, datetime.now(timezone.utc))

    def _update_statistics(self, assessment: QualityAssessment) -> None:
        """Update service statistics."""

        self._stats["assessments_performed"] += 1
        self._stats["quality_distribution"][assessment.quality_level.value] += 1

        # Update average assessment time
        current_avg = self._stats["average_assessment_time_ms"]
        total_assessments = self._stats["assessments_performed"]

        if total_assessments == 1:
            self._stats["average_assessment_time_ms"] = assessment.assessment_duration_ms
        else:
            self._stats["average_assessment_time_ms"] = (
                (current_avg * (total_assessments - 1) + assessment.assessment_duration_ms) / total_assessments
            )

    def get_assessment_statistics(self) -> Dict[str, Any]:
        """Get quality assessment service statistics."""
        return self._stats.copy()

    def clear_cache(self) -> None:
        """Clear the assessment cache."""
        self._cache.clear()
        logger.info("Quality assessment cache cleared")

    def set_benchmarks(self, benchmarks: Dict[str, Dict[QualityDimension, float]]) -> None:
        """Set quality benchmarks for comparison."""
        self._benchmarks = benchmarks
        logger.info(f"Updated quality benchmarks for {len(benchmarks)} categories")


# Global service instance
_quality_assessment_service: Optional[QualityAssessmentService] = None


def get_quality_assessment_service(config: Optional[QualityConfig] = None) -> QualityAssessmentService:
    """Get or create the global quality assessment service instance."""
    global _quality_assessment_service

    if _quality_assessment_service is None:
        _quality_assessment_service = QualityAssessmentService(config)

    return _quality_assessment_service


def reset_quality_assessment_service() -> None:
    """Reset the global quality assessment service instance."""
    global _quality_assessment_service
    _quality_assessment_service = None
