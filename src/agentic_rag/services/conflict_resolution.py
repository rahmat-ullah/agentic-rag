"""
Conflict and Incompleteness Handling System

This module implements comprehensive conflict detection, resolution strategies,
and uncertainty communication for answer synthesis.
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


class ConflictType(str, Enum):
    """Types of conflicts that can be detected."""
    
    FACTUAL = "factual"                    # Contradictory facts
    TEMPORAL = "temporal"                  # Time-based conflicts
    NUMERICAL = "numerical"                # Conflicting numbers/quantities
    CATEGORICAL = "categorical"            # Different categories/classifications
    OPINION = "opinion"                    # Conflicting opinions/assessments
    PROCEDURAL = "procedural"              # Different procedures/processes
    REGULATORY = "regulatory"              # Conflicting regulations/requirements
    PRICING = "pricing"                    # Price conflicts
    SPECIFICATION = "specification"        # Technical specification conflicts
    AVAILABILITY = "availability"          # Availability/timeline conflicts


class ConflictSeverity(str, Enum):
    """Severity levels for conflicts."""
    
    CRITICAL = "critical"                  # Major conflicts requiring resolution
    HIGH = "high"                         # Significant conflicts
    MEDIUM = "medium"                     # Moderate conflicts
    LOW = "low"                          # Minor conflicts
    NEGLIGIBLE = "negligible"             # Very minor conflicts


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""
    
    PRESENT_ALL = "present_all"           # Present all conflicting information
    PRIORITIZE_RECENT = "prioritize_recent"  # Favor more recent information
    PRIORITIZE_AUTHORITATIVE = "prioritize_authoritative"  # Favor authoritative sources
    SYNTHESIZE = "synthesize"             # Attempt to synthesize/reconcile
    FLAG_UNCERTAINTY = "flag_uncertainty"  # Flag as uncertain/conflicting
    SEEK_CLARIFICATION = "seek_clarification"  # Request additional information
    MAJORITY_RULE = "majority_rule"       # Go with majority consensus
    EXPERT_JUDGMENT = "expert_judgment"   # Defer to expert sources


class IncompletenessType(str, Enum):
    """Types of incompleteness that can be detected."""
    
    MISSING_INFORMATION = "missing_information"  # Key information missing
    PARTIAL_COVERAGE = "partial_coverage"        # Incomplete coverage of topic
    INSUFFICIENT_DETAIL = "insufficient_detail"  # Lacks necessary detail
    MISSING_CONTEXT = "missing_context"          # Missing contextual information
    INCOMPLETE_ANALYSIS = "incomplete_analysis"  # Analysis is incomplete
    MISSING_SOURCES = "missing_sources"          # Insufficient source material
    TEMPORAL_GAPS = "temporal_gaps"              # Missing time periods
    SCOPE_LIMITATIONS = "scope_limitations"      # Limited scope coverage


class UncertaintyLevel(str, Enum):
    """Levels of uncertainty in information."""
    
    CERTAIN = "certain"                   # High confidence
    LIKELY = "likely"                    # Probable but not certain
    UNCERTAIN = "uncertain"              # Significant uncertainty
    CONFLICTING = "conflicting"          # Conflicting information
    UNKNOWN = "unknown"                  # Information not available


@dataclass
class ConflictEvidence:
    """Evidence supporting a conflict detection."""
    
    source_id: str
    content: str
    confidence: float
    timestamp: Optional[datetime] = None
    authority_score: float = 0.5
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class ConflictInfo(BaseModel):
    """Information about a detected conflict."""
    
    conflict_id: str = Field(default_factory=lambda: str(uuid4()))
    conflict_type: ConflictType = Field(..., description="Type of conflict")
    severity: ConflictSeverity = Field(..., description="Severity of conflict")
    
    # Conflict details
    description: str = Field(..., description="Description of the conflict")
    conflicting_statements: List[str] = Field(..., description="Conflicting statements")
    evidence: List[ConflictEvidence] = Field(default_factory=list, description="Supporting evidence")
    
    # Resolution information
    suggested_strategy: ResolutionStrategy = Field(..., description="Suggested resolution strategy")
    resolution_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in resolution")
    
    # Context
    affected_topics: List[str] = Field(default_factory=list, description="Topics affected by conflict")
    query_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Relevance to original query")
    
    # Metadata
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detection_method: str = Field(default="heuristic", description="Method used for detection")


class IncompletenessInfo(BaseModel):
    """Information about detected incompleteness."""
    
    incompleteness_id: str = Field(default_factory=lambda: str(uuid4()))
    incompleteness_type: IncompletenessType = Field(..., description="Type of incompleteness")
    
    # Incompleteness details
    description: str = Field(..., description="Description of what's missing")
    missing_aspects: List[str] = Field(..., description="Specific missing aspects")
    impact_assessment: str = Field(..., description="Impact of the incompleteness")
    
    # Severity and priority
    severity: ConflictSeverity = Field(..., description="Severity of incompleteness")
    priority: int = Field(default=1, ge=1, le=5, description="Priority for addressing (1=highest)")
    
    # Suggestions
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested actions to address")
    potential_sources: List[str] = Field(default_factory=list, description="Potential sources for missing info")
    
    # Context
    affected_topics: List[str] = Field(default_factory=list, description="Topics affected")
    query_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Relevance to original query")
    
    # Metadata
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detection_method: str = Field(default="heuristic", description="Method used for detection")


class UncertaintyInfo(BaseModel):
    """Information about uncertainty in the answer."""
    
    uncertainty_id: str = Field(default_factory=lambda: str(uuid4()))
    uncertainty_level: UncertaintyLevel = Field(..., description="Level of uncertainty")
    
    # Uncertainty details
    description: str = Field(..., description="Description of the uncertainty")
    uncertain_statements: List[str] = Field(..., description="Statements with uncertainty")
    confidence_range: Tuple[float, float] = Field(..., description="Confidence range (min, max)")
    
    # Sources of uncertainty
    contributing_factors: List[str] = Field(default_factory=list, description="Factors contributing to uncertainty")
    conflicting_sources: List[str] = Field(default_factory=list, description="Sources with conflicting information")
    
    # Communication
    communication_strategy: str = Field(..., description="How to communicate this uncertainty")
    hedge_words: List[str] = Field(default_factory=list, description="Suggested hedge words")
    
    # Context
    affected_topics: List[str] = Field(default_factory=list, description="Topics affected")
    query_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Relevance to original query")
    
    # Metadata
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConflictResolutionConfig(BaseModel):
    """Configuration for conflict resolution system."""
    
    # Detection settings
    enable_conflict_detection: bool = Field(default=True)
    enable_incompleteness_detection: bool = Field(default=True)
    enable_uncertainty_analysis: bool = Field(default=True)
    
    # Sensitivity thresholds
    conflict_detection_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    incompleteness_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    uncertainty_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Resolution preferences
    default_resolution_strategy: ResolutionStrategy = Field(default=ResolutionStrategy.PRESENT_ALL)
    prioritize_recent_sources: bool = Field(default=True)
    prioritize_authoritative_sources: bool = Field(default=True)
    
    # Communication settings
    include_uncertainty_indicators: bool = Field(default=True)
    use_hedge_words: bool = Field(default=True)
    provide_confidence_scores: bool = Field(default=False)
    
    # Performance settings
    max_conflicts_to_detect: int = Field(default=10, gt=0)
    max_incompleteness_items: int = Field(default=5, gt=0)
    enable_caching: bool = Field(default=True)


class ConflictResolutionResult(BaseModel):
    """Result of conflict resolution analysis."""
    
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Analysis results
    conflicts_detected: List[ConflictInfo] = Field(default_factory=list)
    incompleteness_detected: List[IncompletenessInfo] = Field(default_factory=list)
    uncertainty_analysis: List[UncertaintyInfo] = Field(default_factory=list)
    
    # Overall assessment
    overall_conflict_level: ConflictSeverity = Field(default=ConflictSeverity.LOW)
    overall_completeness_score: float = Field(default=0.8, ge=0.0, le=1.0)
    overall_uncertainty_level: UncertaintyLevel = Field(default=UncertaintyLevel.CERTAIN)
    
    # Recommendations
    recommended_actions: List[str] = Field(default_factory=list)
    communication_adjustments: List[str] = Field(default_factory=list)
    additional_research_needed: List[str] = Field(default_factory=list)
    
    # Summary
    summary: str = Field(default="", description="Summary of analysis")
    confidence_in_analysis: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Performance metrics
    analysis_duration_ms: int = Field(default=0, description="Analysis duration in milliseconds")


class ConflictResolutionService:
    """Service for detecting and resolving conflicts and incompleteness."""
    
    def __init__(self, config: Optional[ConflictResolutionConfig] = None):
        self.config = config or ConflictResolutionConfig()
        
        # Analysis cache
        self._cache: Dict[str, Tuple[ConflictResolutionResult, datetime]] = {}
        self._max_cache_size = 500
        
        # Statistics
        self._stats = {
            "analyses_performed": 0,
            "conflicts_detected": 0,
            "incompleteness_detected": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_analysis_time_ms": 0
        }
        
        logger.info(f"Conflict resolution service initialized")
    
    async def analyze_conflicts_and_completeness(
        self,
        answer: str,
        query: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictResolutionResult:
        """Analyze conflicts, incompleteness, and uncertainty in an answer."""
        
        start_time = datetime.now()
        
        # Check cache
        cache_key = self._generate_cache_key(answer, query)
        if self.config.enable_caching:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self._stats["cache_hits"] += 1
                return cached_result
        
        self._stats["cache_misses"] += 1
        
        # Initialize result
        result = ConflictResolutionResult()
        
        # Detect conflicts
        if self.config.enable_conflict_detection:
            result.conflicts_detected = await self._detect_conflicts(answer, query, sources, context)
        
        # Detect incompleteness
        if self.config.enable_incompleteness_detection:
            result.incompleteness_detected = await self._detect_incompleteness(answer, query, sources, context)
        
        # Analyze uncertainty
        if self.config.enable_uncertainty_analysis:
            result.uncertainty_analysis = await self._analyze_uncertainty(answer, query, sources, context)
        
        # Calculate overall assessments
        result.overall_conflict_level = self._calculate_overall_conflict_level(result.conflicts_detected)
        result.overall_completeness_score = self._calculate_completeness_score(result.incompleteness_detected)
        result.overall_uncertainty_level = self._calculate_uncertainty_level(result.uncertainty_analysis)
        
        # Generate recommendations
        result.recommended_actions = self._generate_recommendations(result)
        result.communication_adjustments = self._generate_communication_adjustments(result)
        result.additional_research_needed = self._identify_research_needs(result)
        
        # Generate summary
        result.summary = self._generate_summary(result)
        result.confidence_in_analysis = self._calculate_analysis_confidence(result)
        
        # Set performance metrics
        analysis_duration = int((datetime.now() - start_time).total_seconds() * 1000)
        result.analysis_duration_ms = analysis_duration
        
        # Cache the result
        if self.config.enable_caching:
            self._cache_result(cache_key, result)
        
        # Update statistics
        self._update_statistics(result)
        
        logger.debug(f"Conflict resolution analysis completed in {analysis_duration}ms")

        return result

    async def _detect_conflicts(
        self,
        answer: str,
        query: str,
        sources: Optional[List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]]
    ) -> List[ConflictInfo]:
        """Detect conflicts in the answer."""

        conflicts = []

        # Detect numerical conflicts
        numerical_conflicts = self._detect_numerical_conflicts(answer, sources)
        conflicts.extend(numerical_conflicts)

        # Detect temporal conflicts
        temporal_conflicts = self._detect_temporal_conflicts(answer, sources)
        conflicts.extend(temporal_conflicts)

        # Detect factual conflicts
        factual_conflicts = self._detect_factual_conflicts(answer, sources)
        conflicts.extend(factual_conflicts)

        # Detect pricing conflicts
        pricing_conflicts = self._detect_pricing_conflicts(answer, sources)
        conflicts.extend(pricing_conflicts)

        # Detect specification conflicts
        spec_conflicts = self._detect_specification_conflicts(answer, sources)
        conflicts.extend(spec_conflicts)

        # Limit number of conflicts
        conflicts = conflicts[:self.config.max_conflicts_to_detect]

        return conflicts

    def _detect_numerical_conflicts(
        self,
        answer: str,
        sources: Optional[List[Dict[str, Any]]]
    ) -> List[ConflictInfo]:
        """Detect conflicts in numerical values."""

        conflicts = []

        # Extract numbers from answer
        number_pattern = r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        numbers = re.findall(number_pattern, answer)

        if len(numbers) < 2:
            return conflicts

        # Look for conflicting numbers in similar contexts
        sentences = re.split(r'[.!?]+', answer)

        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                # Check if sentences discuss similar topics
                if self._sentences_similar_topic(sentence1, sentence2):
                    nums1 = re.findall(number_pattern, sentence1)
                    nums2 = re.findall(number_pattern, sentence2)

                    if nums1 and nums2 and nums1[0] != nums2[0]:
                        # Potential numerical conflict
                        conflict = ConflictInfo(
                            conflict_type=ConflictType.NUMERICAL,
                            severity=ConflictSeverity.MEDIUM,
                            description=f"Conflicting numerical values: {nums1[0]} vs {nums2[0]}",
                            conflicting_statements=[sentence1.strip(), sentence2.strip()],
                            suggested_strategy=ResolutionStrategy.SEEK_CLARIFICATION,
                            query_relevance=0.7
                        )
                        conflicts.append(conflict)

        return conflicts

    def _detect_temporal_conflicts(
        self,
        answer: str,
        sources: Optional[List[Dict[str, Any]]]
    ) -> List[ConflictInfo]:
        """Detect conflicts in temporal information."""

        conflicts = []

        # Extract temporal expressions
        temporal_patterns = [
            r'(\d{1,2})\s+days?',
            r'(\d{1,2})\s+weeks?',
            r'(\d{1,2})\s+months?',
            r'by\s+(\w+\s+\d{1,2})',
            r'within\s+(\d+\s+\w+)',
        ]

        temporal_info = []
        sentences = re.split(r'[.!?]+', answer)

        for sentence in sentences:
            for pattern in temporal_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                if matches:
                    temporal_info.append((sentence.strip(), matches[0]))

        # Check for conflicts
        if len(temporal_info) >= 2:
            for i, (sent1, time1) in enumerate(temporal_info):
                for sent2, time2 in temporal_info[i+1:]:
                    if time1 != time2 and self._sentences_similar_topic(sent1, sent2):
                        conflict = ConflictInfo(
                            conflict_type=ConflictType.TEMPORAL,
                            severity=ConflictSeverity.HIGH,
                            description=f"Conflicting temporal information: {time1} vs {time2}",
                            conflicting_statements=[sent1, sent2],
                            suggested_strategy=ResolutionStrategy.PRIORITIZE_RECENT,
                            query_relevance=0.8
                        )
                        conflicts.append(conflict)

        return conflicts

    def _detect_factual_conflicts(
        self,
        answer: str,
        sources: Optional[List[Dict[str, Any]]]
    ) -> List[ConflictInfo]:
        """Detect factual conflicts in the answer."""

        conflicts = []

        # Look for contradictory indicators (simplified approach)
        contradiction_indicators = [
            'however', 'but', 'although', 'despite', 'contrary to',
            'on the other hand', 'in contrast', 'nevertheless', 'conflict'
        ]

        sentences = re.split(r'[.!?]+', answer)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Check for contradiction indicators
            for indicator in contradiction_indicators:
                if indicator in sentence_lower:
                    # Found potential factual conflict
                    conflict = ConflictInfo(
                        conflict_type=ConflictType.FACTUAL,
                        severity=ConflictSeverity.MEDIUM,
                        description=f"Potential contradiction indicated by '{indicator}'",
                        conflicting_statements=[sentence.strip()],
                        suggested_strategy=ResolutionStrategy.SEEK_CLARIFICATION,
                        query_relevance=0.7
                    )
                    conflicts.append(conflict)
                    break  # Only one conflict per sentence

        return conflicts

    def _detect_pricing_conflicts(
        self,
        answer: str,
        sources: Optional[List[Dict[str, Any]]]
    ) -> List[ConflictInfo]:
        """Detect conflicts in pricing information."""

        conflicts = []

        # Extract price information
        price_pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        prices = []
        sentences = re.split(r'[.!?]+', answer)

        for sentence in sentences:
            price_matches = re.findall(price_pattern, sentence)
            if price_matches:
                prices.append((sentence.strip(), price_matches))

        # Check for conflicting prices
        if len(prices) >= 2:
            for i, (sent1, prices1) in enumerate(prices):
                for sent2, prices2 in prices[i+1:]:
                    if (prices1 != prices2 and
                        self._sentences_similar_topic(sent1, sent2)):

                        conflict = ConflictInfo(
                            conflict_type=ConflictType.PRICING,
                            severity=ConflictSeverity.HIGH,
                            description=f"Conflicting pricing information: ${prices1[0]} vs ${prices2[0]}",
                            conflicting_statements=[sent1, sent2],
                            suggested_strategy=ResolutionStrategy.PRIORITIZE_AUTHORITATIVE,
                            query_relevance=0.9
                        )
                        conflicts.append(conflict)

        return conflicts

    def _detect_specification_conflicts(
        self,
        answer: str,
        sources: Optional[List[Dict[str, Any]]]
    ) -> List[ConflictInfo]:
        """Detect conflicts in specifications."""

        conflicts = []

        # Look for specification keywords
        spec_keywords = ['specification', 'requirement', 'standard', 'criteria', 'parameter']

        sentences = re.split(r'[.!?]+', answer)
        spec_sentences = []

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in spec_keywords):
                spec_sentences.append(sentence.strip())

        # Simple conflict detection based on contradictory words
        contradictory_pairs = [
            ('required', 'optional'),
            ('mandatory', 'voluntary'),
            ('must', 'may'),
            ('minimum', 'maximum'),
            ('include', 'exclude')
        ]

        for i, sent1 in enumerate(spec_sentences):
            for sent2 in spec_sentences[i+1:]:
                for word1, word2 in contradictory_pairs:
                    if (word1 in sent1.lower() and word2 in sent2.lower() and
                        self._sentences_similar_topic(sent1, sent2)):

                        conflict = ConflictInfo(
                            conflict_type=ConflictType.SPECIFICATION,
                            severity=ConflictSeverity.HIGH,
                            description=f"Conflicting specifications: {word1} vs {word2}",
                            conflicting_statements=[sent1, sent2],
                            suggested_strategy=ResolutionStrategy.PRIORITIZE_AUTHORITATIVE,
                            query_relevance=0.8
                        )
                        conflicts.append(conflict)

        return conflicts

    def _sentences_similar_topic(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences discuss similar topics."""

        # Simple similarity check based on common words
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return False

        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union) if union else 0.0
        return similarity >= 0.3  # Threshold for similar topics

    async def _detect_incompleteness(
        self,
        answer: str,
        query: str,
        sources: Optional[List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]]
    ) -> List[IncompletenessInfo]:
        """Detect incompleteness in the answer."""

        incompleteness_items = []

        # Analyze query coverage
        query_coverage_issues = self._analyze_query_coverage(answer, query)
        incompleteness_items.extend(query_coverage_issues)

        # Check for missing key information
        missing_info_issues = self._detect_missing_information(answer, query, sources)
        incompleteness_items.extend(missing_info_issues)

        # Analyze source utilization
        source_issues = self._analyze_source_utilization(answer, sources)
        incompleteness_items.extend(source_issues)

        # Check for insufficient detail
        detail_issues = self._detect_insufficient_detail(answer, query)
        incompleteness_items.extend(detail_issues)

        # Limit number of items
        incompleteness_items = incompleteness_items[:self.config.max_incompleteness_items]

        return incompleteness_items

    def _analyze_query_coverage(self, answer: str, query: str) -> List[IncompletenessInfo]:
        """Analyze how well the answer covers the query."""

        issues = []

        # Extract key terms from query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'when', 'where', 'why', 'who'}
        query_words = query_words - stop_words

        # Check coverage
        covered_words = query_words.intersection(answer_words)
        uncovered_words = query_words - covered_words

        if len(uncovered_words) > 0 and len(query_words) > 0:
            coverage_ratio = len(covered_words) / len(query_words)

            if coverage_ratio < self.config.incompleteness_threshold:
                issue = IncompletenessInfo(
                    incompleteness_type=IncompletenessType.PARTIAL_COVERAGE,
                    description=f"Answer covers only {coverage_ratio:.1%} of query terms",
                    missing_aspects=list(uncovered_words),
                    impact_assessment="May not fully address the user's question",
                    severity=ConflictSeverity.MEDIUM if coverage_ratio < 0.5 else ConflictSeverity.LOW,
                    priority=2,
                    suggested_actions=[f"Address missing aspects: {', '.join(list(uncovered_words)[:3])}"],
                    query_relevance=1.0
                )
                issues.append(issue)

        return issues

    def _detect_missing_information(
        self,
        answer: str,
        query: str,
        sources: Optional[List[Dict[str, Any]]]
    ) -> List[IncompletenessInfo]:
        """Detect missing key information."""

        issues = []

        # Common information categories for procurement queries
        info_categories = {
            'pricing': ['price', 'cost', 'budget', 'fee', 'rate'],
            'timeline': ['delivery', 'deadline', 'schedule', 'timeline', 'date'],
            'specifications': ['specification', 'requirement', 'standard', 'criteria'],
            'vendor': ['vendor', 'supplier', 'contractor', 'provider'],
            'location': ['location', 'address', 'site', 'facility'],
            'contact': ['contact', 'phone', 'email', 'representative']
        }

        query_lower = query.lower()
        answer_lower = answer.lower()

        for category, keywords in info_categories.items():
            # Check if query asks about this category
            query_mentions = any(keyword in query_lower for keyword in keywords)
            answer_mentions = any(keyword in answer_lower for keyword in keywords)

            if query_mentions and not answer_mentions:
                issue = IncompletenessInfo(
                    incompleteness_type=IncompletenessType.MISSING_INFORMATION,
                    description=f"Missing {category} information",
                    missing_aspects=[category],
                    impact_assessment=f"User specifically asked about {category} but answer doesn't address it",
                    severity=ConflictSeverity.HIGH,
                    priority=1,
                    suggested_actions=[f"Include {category} information in the answer"],
                    query_relevance=0.9
                )
                issues.append(issue)

        return issues

    def _analyze_source_utilization(
        self,
        answer: str,
        sources: Optional[List[Dict[str, Any]]]
    ) -> List[IncompletenessInfo]:
        """Analyze utilization of available sources."""

        issues = []

        if not sources:
            return issues

        # Count citations in answer
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, answer)
        unique_citations = set(citations)

        # Calculate utilization ratio
        utilization_ratio = len(unique_citations) / len(sources) if sources else 0.0

        if utilization_ratio < 0.5:  # Less than 50% of sources used
            unused_count = len(sources) - len(unique_citations)

            issue = IncompletenessInfo(
                incompleteness_type=IncompletenessType.MISSING_SOURCES,
                description=f"Only {len(unique_citations)} of {len(sources)} available sources utilized",
                missing_aspects=[f"{unused_count} unused sources"],
                impact_assessment="May be missing relevant information from unused sources",
                severity=ConflictSeverity.MEDIUM,
                priority=3,
                suggested_actions=["Review and incorporate information from unused sources"],
                query_relevance=0.6
            )
            issues.append(issue)

        return issues

    def _detect_insufficient_detail(self, answer: str, query: str) -> List[IncompletenessInfo]:
        """Detect insufficient detail in the answer."""

        issues = []

        # Check answer length
        word_count = len(answer.split())
        sentence_count = len(re.split(r'[.!?]+', answer))

        # Heuristic: very short answers may lack detail
        if word_count < 50:
            issue = IncompletenessInfo(
                incompleteness_type=IncompletenessType.INSUFFICIENT_DETAIL,
                description=f"Answer is very brief ({word_count} words)",
                missing_aspects=["detailed explanation", "supporting information"],
                impact_assessment="May not provide sufficient detail to fully answer the question",
                severity=ConflictSeverity.MEDIUM,
                priority=2,
                suggested_actions=["Expand answer with more detailed information"],
                query_relevance=0.7
            )
            issues.append(issue)

        # Check for lack of examples or specifics
        if not re.search(r'\d+', answer):  # No numbers
            issue = IncompletenessInfo(
                incompleteness_type=IncompletenessType.INSUFFICIENT_DETAIL,
                description="Answer lacks specific numerical information",
                missing_aspects=["specific numbers", "quantitative data"],
                impact_assessment="May be too general without specific details",
                severity=ConflictSeverity.LOW,
                priority=4,
                suggested_actions=["Include specific numbers, dates, or quantities"],
                query_relevance=0.5
            )
            issues.append(issue)

        return issues

    async def _analyze_uncertainty(
        self,
        answer: str,
        query: str,
        sources: Optional[List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]]
    ) -> List[UncertaintyInfo]:
        """Analyze uncertainty in the answer."""

        uncertainties = []

        # Detect hedge words and uncertainty indicators
        hedge_words = [
            'might', 'may', 'could', 'possibly', 'probably', 'likely',
            'appears', 'seems', 'suggests', 'indicates', 'approximately',
            'roughly', 'about', 'around', 'potentially', 'presumably'
        ]

        uncertainty_phrases = [
            'it is unclear', 'uncertain', 'not specified', 'not clear',
            'conflicting information', 'varies', 'depends on', 'subject to'
        ]

        answer_lower = answer.lower()

        # Count uncertainty indicators
        hedge_count = sum(1 for word in hedge_words if word in answer_lower)
        phrase_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)

        total_uncertainty_indicators = hedge_count + phrase_count

        if total_uncertainty_indicators > 0:
            # Calculate uncertainty level
            word_count = len(answer.split())
            uncertainty_density = total_uncertainty_indicators / word_count if word_count > 0 else 0

            if uncertainty_density > 0.05:  # More than 5% uncertainty indicators
                uncertainty_level = UncertaintyLevel.UNCERTAIN
            elif uncertainty_density > 0.02:
                uncertainty_level = UncertaintyLevel.LIKELY
            else:
                uncertainty_level = UncertaintyLevel.CERTAIN

            if uncertainty_level != UncertaintyLevel.CERTAIN:
                uncertainty = UncertaintyInfo(
                    uncertainty_level=uncertainty_level,
                    description=f"Answer contains {total_uncertainty_indicators} uncertainty indicators",
                    uncertain_statements=[answer[:200] + "..." if len(answer) > 200 else answer],
                    confidence_range=(0.5, 0.8) if uncertainty_level == UncertaintyLevel.UNCERTAIN else (0.7, 0.9),
                    contributing_factors=[f"Hedge words: {hedge_count}", f"Uncertainty phrases: {phrase_count}"],
                    communication_strategy="Use clear uncertainty indicators and provide confidence ranges",
                    hedge_words=hedge_words[:3],
                    query_relevance=0.8
                )
                uncertainties.append(uncertainty)

        return uncertainties

    def _calculate_overall_conflict_level(self, conflicts: List[ConflictInfo]) -> ConflictSeverity:
        """Calculate overall conflict level."""

        if not conflicts:
            return ConflictSeverity.NEGLIGIBLE

        # Find highest severity
        severity_order = {
            ConflictSeverity.CRITICAL: 5,
            ConflictSeverity.HIGH: 4,
            ConflictSeverity.MEDIUM: 3,
            ConflictSeverity.LOW: 2,
            ConflictSeverity.NEGLIGIBLE: 1
        }

        max_severity = max(conflicts, key=lambda c: severity_order[c.severity])
        return max_severity.severity

    def _calculate_completeness_score(self, incompleteness_items: List[IncompletenessInfo]) -> float:
        """Calculate overall completeness score."""

        if not incompleteness_items:
            return 1.0

        # Calculate score based on severity and count
        severity_weights = {
            ConflictSeverity.CRITICAL: 0.4,
            ConflictSeverity.HIGH: 0.3,
            ConflictSeverity.MEDIUM: 0.2,
            ConflictSeverity.LOW: 0.1,
            ConflictSeverity.NEGLIGIBLE: 0.05
        }

        total_penalty = sum(severity_weights.get(item.severity, 0.1) for item in incompleteness_items)
        completeness_score = max(0.0, 1.0 - total_penalty)

        return completeness_score

    def _calculate_uncertainty_level(self, uncertainties: List[UncertaintyInfo]) -> UncertaintyLevel:
        """Calculate overall uncertainty level."""

        if not uncertainties:
            return UncertaintyLevel.CERTAIN

        # Find highest uncertainty level
        uncertainty_order = {
            UncertaintyLevel.UNKNOWN: 5,
            UncertaintyLevel.CONFLICTING: 4,
            UncertaintyLevel.UNCERTAIN: 3,
            UncertaintyLevel.LIKELY: 2,
            UncertaintyLevel.CERTAIN: 1
        }

        max_uncertainty = max(uncertainties, key=lambda u: uncertainty_order[u.uncertainty_level])
        return max_uncertainty.uncertainty_level

    def _generate_recommendations(self, result: ConflictResolutionResult) -> List[str]:
        """Generate recommendations based on analysis."""

        recommendations = []

        # Conflict-based recommendations
        if result.conflicts_detected:
            critical_conflicts = [c for c in result.conflicts_detected if c.severity == ConflictSeverity.CRITICAL]
            if critical_conflicts:
                recommendations.append("Resolve critical conflicts before finalizing answer")

            high_conflicts = [c for c in result.conflicts_detected if c.severity == ConflictSeverity.HIGH]
            if high_conflicts:
                recommendations.append("Address high-severity conflicts with additional research")

        # Incompleteness-based recommendations
        if result.incompleteness_detected:
            high_priority = [i for i in result.incompleteness_detected if i.priority <= 2]
            if high_priority:
                recommendations.append("Address high-priority missing information")

            missing_info = [i for i in result.incompleteness_detected
                          if i.incompleteness_type == IncompletenessType.MISSING_INFORMATION]
            if missing_info:
                recommendations.append("Include missing key information categories")

        # Uncertainty-based recommendations
        if result.uncertainty_analysis:
            high_uncertainty = [u for u in result.uncertainty_analysis
                              if u.uncertainty_level in [UncertaintyLevel.UNCERTAIN, UncertaintyLevel.CONFLICTING]]
            if high_uncertainty:
                recommendations.append("Clearly communicate uncertainty and provide confidence ranges")

        # General recommendations
        if result.overall_completeness_score < 0.7:
            recommendations.append("Expand answer with more comprehensive information")

        return recommendations[:5]  # Limit to top 5

    def _generate_communication_adjustments(self, result: ConflictResolutionResult) -> List[str]:
        """Generate communication adjustments."""

        adjustments = []

        # Based on conflicts
        if result.conflicts_detected:
            adjustments.append("Present conflicting information transparently")
            adjustments.append("Indicate sources for conflicting claims")

        # Based on uncertainty
        if result.uncertainty_analysis:
            adjustments.append("Use appropriate hedge words for uncertain information")
            adjustments.append("Provide confidence indicators where possible")

        # Based on incompleteness
        if result.incompleteness_detected:
            adjustments.append("Acknowledge limitations and missing information")
            adjustments.append("Suggest where additional information might be found")

        return adjustments

    def _identify_research_needs(self, result: ConflictResolutionResult) -> List[str]:
        """Identify additional research needs."""

        research_needs = []

        # From conflicts
        for conflict in result.conflicts_detected:
            if conflict.severity in [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH]:
                research_needs.append(f"Resolve {conflict.conflict_type.value} conflict: {conflict.description}")

        # From incompleteness
        for item in result.incompleteness_detected:
            if item.priority <= 2:
                research_needs.extend(item.suggested_actions)

        return list(set(research_needs))[:5]  # Remove duplicates and limit

    def _generate_summary(self, result: ConflictResolutionResult) -> str:
        """Generate analysis summary."""

        summary_parts = []

        # Conflicts summary
        if result.conflicts_detected:
            conflict_count = len(result.conflicts_detected)
            summary_parts.append(f"{conflict_count} conflict(s) detected with {result.overall_conflict_level.value} severity")
        else:
            summary_parts.append("No conflicts detected")

        # Completeness summary
        completeness_pct = int(result.overall_completeness_score * 100)
        summary_parts.append(f"Answer completeness: {completeness_pct}%")

        # Uncertainty summary
        if result.uncertainty_analysis:
            summary_parts.append(f"Uncertainty level: {result.overall_uncertainty_level.value}")
        else:
            summary_parts.append("No significant uncertainty detected")

        return ". ".join(summary_parts) + "."

    def _calculate_analysis_confidence(self, result: ConflictResolutionResult) -> float:
        """Calculate confidence in the analysis."""

        # Base confidence
        confidence = 0.8

        # Adjust based on number of items analyzed
        total_items = len(result.conflicts_detected) + len(result.incompleteness_detected) + len(result.uncertainty_analysis)

        if total_items == 0:
            confidence = 0.9  # High confidence when no issues found
        elif total_items > 10:
            confidence = 0.6  # Lower confidence with many issues (may be noisy)

        return confidence

    def _generate_cache_key(self, answer: str, query: str) -> str:
        """Generate cache key for analysis."""

        import hashlib

        key_components = [answer[:100], query[:50]]  # Use truncated versions
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[ConflictResolutionResult]:
        """Get cached result if available and not expired."""

        if cache_key not in self._cache:
            return None

        result, cached_time = self._cache[cache_key]

        # Check if cache is expired (1 hour TTL)
        cache_age_hours = (datetime.now(timezone.utc) - cached_time).total_seconds() / 3600
        if cache_age_hours > 1:
            del self._cache[cache_key]
            return None

        return result

    def _cache_result(self, cache_key: str, result: ConflictResolutionResult) -> None:
        """Cache the analysis result."""

        # Clean cache if it's getting too large
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entries
            sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:self._max_cache_size // 4]:
                del self._cache[key]

        self._cache[cache_key] = (result, datetime.now(timezone.utc))

    def _update_statistics(self, result: ConflictResolutionResult) -> None:
        """Update service statistics."""

        self._stats["analyses_performed"] += 1
        self._stats["conflicts_detected"] += len(result.conflicts_detected)
        self._stats["incompleteness_detected"] += len(result.incompleteness_detected)

        # Update average analysis time
        current_avg = self._stats["average_analysis_time_ms"]
        total_analyses = self._stats["analyses_performed"]

        if total_analyses == 1:
            self._stats["average_analysis_time_ms"] = result.analysis_duration_ms
        else:
            self._stats["average_analysis_time_ms"] = (
                (current_avg * (total_analyses - 1) + result.analysis_duration_ms) / total_analyses
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self._stats.copy()

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
        logger.info("Conflict resolution cache cleared")


# Global service instance
_conflict_resolution_service: Optional[ConflictResolutionService] = None


def get_conflict_resolution_service(config: Optional[ConflictResolutionConfig] = None) -> ConflictResolutionService:
    """Get or create the global conflict resolution service instance."""
    global _conflict_resolution_service

    if _conflict_resolution_service is None:
        _conflict_resolution_service = ConflictResolutionService(config)

    return _conflict_resolution_service


def reset_conflict_resolution_service() -> None:
    """Reset the global conflict resolution service instance."""
    global _conflict_resolution_service
    _conflict_resolution_service = None
