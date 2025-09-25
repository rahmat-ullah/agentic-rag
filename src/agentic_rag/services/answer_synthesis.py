"""
Answer Synthesis Service

This module implements comprehensive LLM-based answer synthesis from retrieved document chunks.
It includes prompt templates, chunk consolidation, quality control, citation generation,
source attribution, and conflict handling for procurement-focused queries.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field, validator

from agentic_rag.services.llm_client import (
    get_llm_client_service, LLMRequest, LLMModel, LLMRequestType
)
from agentic_rag.services.vector_search import VectorSearchResult
from agentic_rag.services.three_hop_search import ThreeHopResults
from agentic_rag.config import get_settings

logger = structlog.get_logger(__name__)


class SynthesisStrategy(str, Enum):
    """Answer synthesis strategies."""
    
    COMPREHENSIVE = "comprehensive"      # Full synthesis with all sources
    FOCUSED = "focused"                 # Focused on most relevant sources
    COMPARATIVE = "comparative"         # Compare and contrast sources
    ANALYTICAL = "analytical"           # Deep analysis with reasoning
    SUMMARY = "summary"                # Concise summary format


class AnswerFormat(str, Enum):
    """Answer formatting options."""
    
    STRUCTURED = "structured"          # Structured with sections
    NARRATIVE = "narrative"            # Flowing narrative text
    BULLET_POINTS = "bullet_points"    # Bullet point format
    FAQ = "faq"                       # Question and answer format
    EXECUTIVE_SUMMARY = "executive_summary"  # Executive summary format


class ConflictResolutionStrategy(str, Enum):
    """Strategies for handling conflicting information."""
    
    PRESENT_ALL = "present_all"        # Present all conflicting views
    PRIORITIZE_RECENT = "prioritize_recent"  # Favor more recent information
    PRIORITIZE_AUTHORITATIVE = "prioritize_authoritative"  # Favor authoritative sources
    SYNTHESIZE = "synthesize"          # Attempt to synthesize conflicts
    FLAG_UNCERTAINTY = "flag_uncertainty"  # Clearly flag uncertain areas


class CitationStyle(str, Enum):
    """Citation formatting styles."""
    
    NUMBERED = "numbered"              # [1], [2], [3]
    AUTHOR_YEAR = "author_year"        # (Smith, 2023)
    FOOTNOTE = "footnote"             # Footnote style
    INLINE = "inline"                 # Inline document references


class QualityDimension(str, Enum):
    """Quality assessment dimensions."""
    
    COMPLETENESS = "completeness"      # Coverage of query aspects
    ACCURACY = "accuracy"             # Factual correctness
    RELEVANCE = "relevance"           # Query alignment
    CLARITY = "clarity"               # Readability and structure
    CITATION_QUALITY = "citation_quality"  # Proper attribution
    COHERENCE = "coherence"           # Logical flow
    OBJECTIVITY = "objectivity"       # Balanced presentation


@dataclass
class SourceChunk:
    """Represents a source chunk with metadata for synthesis."""
    
    chunk_id: str
    content: str
    document_id: str
    document_title: str
    document_type: str
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    chunk_index: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Citation(BaseModel):
    """Citation information for a source."""
    
    id: int = Field(..., description="Citation number")
    chunk_id: str = Field(..., description="Source chunk ID")
    document_id: str = Field(..., description="Document ID")
    document_title: str = Field(..., description="Document title")
    section_title: Optional[str] = Field(None, description="Section title")
    page_number: Optional[int] = Field(None, description="Page number")
    chunk_index: int = Field(default=0, description="Chunk index in document")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    
    # Additional metadata
    document_type: Optional[str] = Field(None, description="Document type")
    author: Optional[str] = Field(None, description="Document author")
    date_created: Optional[datetime] = Field(None, description="Document creation date")
    url: Optional[str] = Field(None, description="Document URL if available")


class ConflictInfo(BaseModel):
    """Information about conflicting sources."""
    
    description: str = Field(..., description="Description of the conflict")
    conflicting_citations: List[int] = Field(..., description="Citation IDs in conflict")
    resolution_strategy: ConflictResolutionStrategy = Field(..., description="How conflict was resolved")
    confidence_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in resolution")
    details: Optional[str] = Field(None, description="Additional details about the conflict")


class QualityScore(BaseModel):
    """Quality assessment score for a dimension."""
    
    dimension: QualityDimension = Field(..., description="Quality dimension")
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score")
    explanation: Optional[str] = Field(None, description="Explanation of the score")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class QualityAssessment(BaseModel):
    """Comprehensive quality assessment of synthesized answer."""
    
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    dimension_scores: List[QualityScore] = Field(..., description="Scores by dimension")
    strengths: List[str] = Field(default_factory=list, description="Answer strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Answer weaknesses")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    
    @validator('overall_score')
    def calculate_overall_score(cls, v, values):
        """Calculate overall score from dimension scores."""
        if 'dimension_scores' in values and values['dimension_scores']:
            scores = [score.score for score in values['dimension_scores']]
            return sum(scores) / len(scores)
        return v


class SynthesisConfig(BaseModel):
    """Configuration for answer synthesis."""
    
    # Strategy and format
    strategy: SynthesisStrategy = Field(default=SynthesisStrategy.COMPREHENSIVE)
    format: AnswerFormat = Field(default=AnswerFormat.STRUCTURED)
    citation_style: CitationStyle = Field(default=CitationStyle.NUMBERED)
    
    # LLM settings
    model: LLMModel = Field(default=LLMModel.GPT_4_TURBO)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)
    
    # Content limits
    max_sources: int = Field(default=20, gt=0)
    max_answer_words: int = Field(default=2000, gt=0)
    min_answer_words: int = Field(default=50, gt=0)
    
    # Quality thresholds
    min_relevance_score: float = Field(default=0.3, ge=0.0, le=1.0)
    min_confidence_score: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # Conflict handling
    conflict_resolution: ConflictResolutionStrategy = Field(default=ConflictResolutionStrategy.PRESENT_ALL)
    max_conflicts_to_show: int = Field(default=5, gt=0)
    
    # Processing options
    enable_quality_assessment: bool = Field(default=True)
    enable_conflict_detection: bool = Field(default=True)
    enable_gap_identification: bool = Field(default=True)
    include_reasoning: bool = Field(default=True)
    
    # Performance settings
    timeout_seconds: int = Field(default=30, gt=0)
    enable_caching: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=24, gt=0)


class SynthesizedAnswer(BaseModel):
    """Complete synthesized answer with all metadata."""
    
    # Core answer
    answer: str = Field(..., description="Synthesized answer text")
    citations: List[Citation] = Field(..., description="Source citations")
    
    # Quality and assessment
    quality_assessment: QualityAssessment = Field(..., description="Quality scores and assessment")
    
    # Conflict and gap handling
    conflicts: List[ConflictInfo] = Field(default_factory=list, description="Identified conflicts")
    information_gaps: List[str] = Field(default_factory=list, description="Identified information gaps")
    uncertainty_areas: List[str] = Field(default_factory=list, description="Areas of uncertainty")
    
    # Metadata
    synthesis_strategy: SynthesisStrategy = Field(..., description="Strategy used for synthesis")
    answer_format: AnswerFormat = Field(..., description="Format used for answer")
    sources_used: int = Field(..., description="Number of sources used")
    total_sources_available: int = Field(..., description="Total sources available")
    
    # Processing metadata
    synthesis_time_ms: int = Field(..., description="Time taken for synthesis")
    model_used: LLMModel = Field(..., description="LLM model used")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Additional context
    reasoning: Optional[str] = Field(None, description="Reasoning behind the synthesis")
    alternative_perspectives: List[str] = Field(default_factory=list, description="Alternative viewpoints")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")


class SynthesisRequest(BaseModel):
    """Request for answer synthesis."""
    
    query: str = Field(..., description="Original user query")
    sources: List[SourceChunk] = Field(..., description="Source chunks for synthesis")
    config: SynthesisConfig = Field(default_factory=SynthesisConfig)
    
    # Context
    tenant_id: UUID = Field(..., description="Tenant ID")
    user_id: UUID = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    
    # Additional context
    user_context: Dict[str, Any] = Field(default_factory=dict, description="Additional user context")
    domain_context: Optional[str] = Field(None, description="Domain context (procurement, legal, etc.)")
    
    @validator('sources')
    def validate_sources(cls, v):
        """Validate that sources are provided."""
        if not v:
            raise ValueError("At least one source must be provided")
        return v


class SynthesisResponse(BaseModel):
    """Response from answer synthesis."""
    
    success: bool = Field(..., description="Whether synthesis was successful")
    answer: Optional[SynthesizedAnswer] = Field(None, description="Synthesized answer if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Performance metrics
    processing_time_ms: int = Field(..., description="Total processing time")
    sources_processed: int = Field(..., description="Number of sources processed")
    
    # Cache information
    from_cache: bool = Field(default=False, description="Whether result was from cache")
    cache_key: Optional[str] = Field(None, description="Cache key used")


class AnswerSynthesisService:
    """Service for synthesizing comprehensive answers from retrieved document chunks."""

    def __init__(self):
        self.settings = get_settings()
        self._llm_client = None

        # Performance tracking
        self._stats = {
            "total_syntheses": 0,
            "successful_syntheses": 0,
            "failed_syntheses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_synthesis_time_ms": 0.0,
            "total_sources_processed": 0
        }

        # Caching
        self._cache: Dict[str, Tuple[SynthesizedAnswer, datetime]] = {}
        self._max_cache_size = 1000

        # Prompt templates
        self._prompt_templates = self._initialize_prompt_templates()

        logger.info("Answer synthesis service initialized")

    async def initialize(self) -> None:
        """Initialize the answer synthesis service."""
        try:
            # Initialize LLM client
            self._llm_client = await get_llm_client_service()

            logger.info("Answer synthesis service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize answer synthesis service: {e}")
            raise

    async def synthesize_answer(self, request: SynthesisRequest) -> SynthesisResponse:
        """Synthesize a comprehensive answer from source chunks."""
        start_time = time.time()

        try:
            self._stats["total_syntheses"] += 1

            # Check cache first
            if request.config.enable_caching:
                cache_key = self._generate_cache_key(request)
                cached_answer = self._get_cached_answer(cache_key, request.config)

                if cached_answer:
                    self._stats["cache_hits"] += 1
                    processing_time_ms = int((time.time() - start_time) * 1000)

                    return SynthesisResponse(
                        success=True,
                        answer=cached_answer,
                        processing_time_ms=processing_time_ms,
                        sources_processed=len(request.sources),
                        from_cache=True,
                        cache_key=cache_key
                    )

            self._stats["cache_misses"] += 1

            # Prepare sources for synthesis
            prepared_sources = await self._prepare_sources(request.sources, request.config)

            # Generate citations
            citations = self._generate_citations(prepared_sources, request.config.citation_style)

            # Synthesize answer using LLM
            synthesized_text = await self._synthesize_with_llm(
                request.query, prepared_sources, citations, request.config,
                request.tenant_id, request.user_id
            )

            # Assess quality if enabled
            quality_assessment = None
            if request.config.enable_quality_assessment:
                quality_assessment = await self._assess_quality(
                    request.query, synthesized_text, prepared_sources, request.config
                )
            else:
                # Create basic quality assessment
                quality_assessment = QualityAssessment(
                    overall_score=0.8,
                    dimension_scores=[
                        QualityScore(dimension=QualityDimension.COMPLETENESS, score=0.8),
                        QualityScore(dimension=QualityDimension.RELEVANCE, score=0.8)
                    ]
                )

            # Detect conflicts if enabled
            conflicts = []
            if request.config.enable_conflict_detection:
                conflicts = await self._detect_conflicts(prepared_sources, request.config)

            # Identify information gaps if enabled
            information_gaps = []
            uncertainty_areas = []
            if request.config.enable_gap_identification:
                information_gaps, uncertainty_areas = await self._identify_gaps_and_uncertainty(
                    request.query, prepared_sources, request.config
                )

            # Generate reasoning if enabled
            reasoning = None
            if request.config.include_reasoning:
                reasoning = await self._generate_reasoning(
                    request.query, prepared_sources, synthesized_text, request.config
                )

            # Create synthesized answer
            synthesis_time_ms = int((time.time() - start_time) * 1000)

            synthesized_answer = SynthesizedAnswer(
                answer=synthesized_text,
                citations=citations,
                quality_assessment=quality_assessment,
                conflicts=conflicts,
                information_gaps=information_gaps,
                uncertainty_areas=uncertainty_areas,
                synthesis_strategy=request.config.strategy,
                answer_format=request.config.format,
                sources_used=len(prepared_sources),
                total_sources_available=len(request.sources),
                synthesis_time_ms=synthesis_time_ms,
                model_used=request.config.model,
                reasoning=reasoning
            )

            # Cache the result
            if request.config.enable_caching:
                self._cache_answer(cache_key, synthesized_answer, request.config)

            # Update statistics
            self._stats["successful_syntheses"] += 1
            self._stats["total_sources_processed"] += len(prepared_sources)
            self._update_average_synthesis_time(synthesis_time_ms)

            processing_time_ms = int((time.time() - start_time) * 1000)

            return SynthesisResponse(
                success=True,
                answer=synthesized_answer,
                processing_time_ms=processing_time_ms,
                sources_processed=len(prepared_sources),
                from_cache=False,
                cache_key=cache_key if request.config.enable_caching else None
            )

        except Exception as e:
            self._stats["failed_syntheses"] += 1
            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.error(f"Answer synthesis failed: {e}")

            return SynthesisResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms,
                sources_processed=len(request.sources) if request.sources else 0
            )

    async def _prepare_sources(self, sources: List[SourceChunk], config: SynthesisConfig) -> List[SourceChunk]:
        """Prepare and filter sources for synthesis."""

        # Filter by relevance and confidence thresholds
        filtered_sources = [
            source for source in sources
            if (source.relevance_score >= config.min_relevance_score and
                source.confidence_score >= config.min_confidence_score)
        ]

        # Sort by relevance score (descending)
        filtered_sources.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit to max sources
        if len(filtered_sources) > config.max_sources:
            filtered_sources = filtered_sources[:config.max_sources]

        logger.info(f"Prepared {len(filtered_sources)} sources from {len(sources)} available")

        return filtered_sources

    def _generate_citations(self, sources: List[SourceChunk], citation_style: CitationStyle) -> List[Citation]:
        """Generate citations for the sources."""
        citations = []

        for i, source in enumerate(sources, 1):
            citation = Citation(
                id=i,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                document_title=source.document_title,
                section_title=source.section_title,
                page_number=source.page_number,
                chunk_index=source.chunk_index,
                confidence_score=source.confidence_score,
                document_type=source.document_type
            )
            citations.append(citation)

        return citations

    async def _synthesize_with_llm(
        self,
        query: str,
        sources: List[SourceChunk],
        citations: List[Citation],
        config: SynthesisConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> str:
        """Synthesize answer using LLM."""

        # Generate synthesis prompt
        prompt = self._generate_synthesis_prompt(query, sources, citations, config)

        # Create LLM request
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_type=LLMRequestType.COMPLETION,
            tenant_id=str(tenant_id),
            user_id=str(user_id)
        )

        # Make LLM request with timeout
        try:
            llm_response = await asyncio.wait_for(
                self._llm_client.generate_completion(llm_request),
                timeout=config.timeout_seconds
            )

            if not llm_response.success:
                raise Exception(f"LLM request failed: {llm_response.error}")

            return llm_response.content

        except asyncio.TimeoutError:
            raise Exception(f"Answer synthesis timed out after {config.timeout_seconds} seconds")

    def _generate_synthesis_prompt(
        self,
        query: str,
        sources: List[SourceChunk],
        citations: List[Citation],
        config: SynthesisConfig
    ) -> str:
        """Generate synthesis prompt based on strategy and format."""

        # Get base template
        template = self._prompt_templates.get(config.strategy, self._prompt_templates[SynthesisStrategy.COMPREHENSIVE])

        # Prepare source information
        source_info = []
        for i, source in enumerate(sources):
            citation_id = i + 1
            source_text = f"[{citation_id}] {source.content}"
            if source.section_title:
                source_text += f" (Section: {source.section_title})"
            if source.page_number:
                source_text += f" (Page: {source.page_number})"
            source_info.append(source_text)

        sources_text = "\n\n".join(source_info)

        # Format citation list
        citation_list = []
        for citation in citations:
            citation_text = f"[{citation.id}] {citation.document_title}"
            if citation.section_title:
                citation_text += f", {citation.section_title}"
            if citation.page_number:
                citation_text += f", Page {citation.page_number}"
            citation_list.append(citation_text)

        citations_text = "\n".join(citation_list)

        # Generate format-specific instructions
        format_instructions = self._get_format_instructions(config.format)

        # Fill template
        prompt = template.format(
            query=query,
            sources=sources_text,
            citations=citations_text,
            format_instructions=format_instructions,
            max_words=config.max_answer_words,
            min_words=config.min_answer_words
        )

        return prompt

    def _get_format_instructions(self, format: AnswerFormat) -> str:
        """Get format-specific instructions."""

        format_instructions = {
            AnswerFormat.STRUCTURED: """
Structure your answer with clear sections and headings. Use proper citations [1], [2], etc.
Include an introduction, main content organized by topic, and a conclusion.
""",
            AnswerFormat.NARRATIVE: """
Provide a flowing narrative answer that tells a complete story. Use proper citations [1], [2], etc.
Ensure smooth transitions between ideas and maintain a logical flow.
""",
            AnswerFormat.BULLET_POINTS: """
Organize your answer using bullet points for key information. Use proper citations [1], [2], etc.
Group related points under appropriate headings.
""",
            AnswerFormat.FAQ: """
Structure your answer as a series of questions and answers. Use proper citations [1], [2], etc.
Address the most important aspects of the query through Q&A format.
""",
            AnswerFormat.EXECUTIVE_SUMMARY: """
Provide a concise executive summary format with key findings and recommendations.
Use proper citations [1], [2], etc. Include executive summary, key findings, and recommendations.
"""
        }

        return format_instructions.get(format, format_instructions[AnswerFormat.STRUCTURED])

    async def _assess_quality(
        self,
        query: str,
        answer: str,
        sources: List[SourceChunk],
        config: SynthesisConfig
    ) -> QualityAssessment:
        """Assess the quality of the synthesized answer."""

        dimension_scores = []

        # Assess completeness
        completeness_score = await self._assess_completeness(query, answer, sources)
        dimension_scores.append(QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=completeness_score,
            explanation="Coverage of query aspects"
        ))

        # Assess relevance
        relevance_score = await self._assess_relevance(query, answer)
        dimension_scores.append(QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=relevance_score,
            explanation="Alignment with user query"
        ))

        # Assess clarity
        clarity_score = await self._assess_clarity(answer)
        dimension_scores.append(QualityScore(
            dimension=QualityDimension.CLARITY,
            score=clarity_score,
            explanation="Readability and structure"
        ))

        # Assess citation quality
        citation_score = await self._assess_citation_quality(answer, sources)
        dimension_scores.append(QualityScore(
            dimension=QualityDimension.CITATION_QUALITY,
            score=citation_score,
            explanation="Proper source attribution"
        ))

        # Calculate overall score
        overall_score = sum(score.score for score in dimension_scores) / len(dimension_scores)

        return QualityAssessment(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            strengths=self._identify_strengths(dimension_scores),
            weaknesses=self._identify_weaknesses(dimension_scores),
            improvement_suggestions=self._generate_improvement_suggestions(dimension_scores)
        )

    async def _assess_completeness(self, query: str, answer: str, sources: List[SourceChunk]) -> float:
        """Assess how completely the answer addresses the query."""
        # Simple heuristic-based assessment
        # In a full implementation, this could use LLM-based evaluation

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Check coverage of query terms
        coverage = len(query_words.intersection(answer_words)) / len(query_words) if query_words else 0

        # Check answer length relative to sources
        source_length = sum(len(source.content) for source in sources)
        answer_length = len(answer)
        length_ratio = min(answer_length / (source_length * 0.1), 1.0) if source_length > 0 else 0.5

        # Combine metrics
        completeness = (coverage * 0.6 + length_ratio * 0.4)
        return min(completeness, 1.0)

    async def _assess_relevance(self, query: str, answer: str) -> float:
        """Assess how relevant the answer is to the query."""
        # Simple keyword-based relevance assessment
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            return 0.5

        # Calculate Jaccard similarity
        intersection = query_words.intersection(answer_words)
        union = query_words.union(answer_words)

        relevance = len(intersection) / len(union) if union else 0
        return min(relevance * 2, 1.0)  # Scale up since Jaccard tends to be low

    async def _assess_clarity(self, answer: str) -> float:
        """Assess the clarity and readability of the answer."""
        # Simple heuristic-based clarity assessment

        if not answer:
            return 0.0

        # Check sentence structure
        sentences = answer.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        # Prefer moderate sentence lengths (10-25 words)
        sentence_score = 1.0 if 10 <= avg_sentence_length <= 25 else max(0.5, 1.0 - abs(avg_sentence_length - 17.5) / 17.5)

        # Check for structure indicators
        structure_indicators = ['however', 'therefore', 'furthermore', 'additionally', 'in conclusion']
        structure_score = min(sum(1 for indicator in structure_indicators if indicator in answer.lower()) / 3, 1.0)

        # Check for citations
        citation_count = answer.count('[') + answer.count('(')
        citation_score = min(citation_count / 5, 1.0)

        clarity = (sentence_score * 0.4 + structure_score * 0.3 + citation_score * 0.3)
        return clarity

    async def _assess_citation_quality(self, answer: str, sources: List[SourceChunk]) -> float:
        """Assess the quality of citations in the answer."""

        if not sources:
            return 0.0

        # Count citations in answer
        citation_count = 0
        for i in range(1, len(sources) + 1):
            if f'[{i}]' in answer:
                citation_count += 1

        # Calculate citation coverage
        citation_coverage = citation_count / len(sources) if sources else 0

        # Check for proper citation format
        proper_format_score = 1.0 if '[' in answer and ']' in answer else 0.5

        citation_quality = (citation_coverage * 0.7 + proper_format_score * 0.3)
        return citation_quality

    def _identify_strengths(self, dimension_scores: List[QualityScore]) -> List[str]:
        """Identify strengths based on dimension scores."""
        strengths = []

        for score in dimension_scores:
            if score.score >= 0.8:
                strengths.append(f"Strong {score.dimension.value}")

        return strengths

    def _identify_weaknesses(self, dimension_scores: List[QualityScore]) -> List[str]:
        """Identify weaknesses based on dimension scores."""
        weaknesses = []

        for score in dimension_scores:
            if score.score < 0.6:
                weaknesses.append(f"Weak {score.dimension.value}")

        return weaknesses

    def _generate_improvement_suggestions(self, dimension_scores: List[QualityScore]) -> List[str]:
        """Generate improvement suggestions based on dimension scores."""
        suggestions = []

        for score in dimension_scores:
            if score.score < 0.7:
                if score.dimension == QualityDimension.COMPLETENESS:
                    suggestions.append("Include more comprehensive coverage of the query aspects")
                elif score.dimension == QualityDimension.RELEVANCE:
                    suggestions.append("Focus more directly on the specific query requirements")
                elif score.dimension == QualityDimension.CLARITY:
                    suggestions.append("Improve structure and readability of the answer")
                elif score.dimension == QualityDimension.CITATION_QUALITY:
                    suggestions.append("Add more proper citations and source attributions")

        return suggestions

    async def _detect_conflicts(self, sources: List[SourceChunk], config: SynthesisConfig) -> List[ConflictInfo]:
        """Detect conflicts between sources."""
        conflicts = []

        # Simple keyword-based conflict detection
        # In a full implementation, this could use more sophisticated NLP techniques

        conflict_keywords = [
            'however', 'but', 'although', 'despite', 'contrary', 'different',
            'disagree', 'conflict', 'contradict', 'oppose'
        ]

        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources[i+1:], i+1):
                # Check for conflicting keywords in proximity
                combined_text = f"{source1.content} {source2.content}".lower()

                conflict_indicators = sum(1 for keyword in conflict_keywords if keyword in combined_text)

                if conflict_indicators >= 2:  # Threshold for conflict detection
                    conflicts.append(ConflictInfo(
                        description=f"Potential conflict between sources {i+1} and {j+1}",
                        conflicting_citations=[i+1, j+1],
                        resolution_strategy=config.conflict_resolution,
                        confidence_level=min(conflict_indicators / 5, 1.0)
                    ))

        return conflicts[:config.max_conflicts_to_show]

    async def _identify_gaps_and_uncertainty(
        self,
        query: str,
        sources: List[SourceChunk],
        config: SynthesisConfig
    ) -> Tuple[List[str], List[str]]:
        """Identify information gaps and uncertainty areas."""

        information_gaps = []
        uncertainty_areas = []

        # Simple gap identification based on query terms not covered in sources
        query_terms = set(query.lower().split())
        source_terms = set()
        for source in sources:
            source_terms.update(source.content.lower().split())

        missing_terms = query_terms - source_terms
        if missing_terms:
            information_gaps.append(f"Limited information about: {', '.join(missing_terms)}")

        # Identify uncertainty indicators
        uncertainty_keywords = ['may', 'might', 'possibly', 'unclear', 'uncertain', 'unknown', 'estimated']

        for source in sources:
            for keyword in uncertainty_keywords:
                if keyword in source.content.lower():
                    uncertainty_areas.append(f"Uncertainty in source {source.document_title}")
                    break

        return information_gaps, uncertainty_areas

    async def _generate_reasoning(
        self,
        query: str,
        sources: List[SourceChunk],
        answer: str,
        config: SynthesisConfig
    ) -> str:
        """Generate reasoning explanation for the synthesis."""

        reasoning_parts = []

        # Explain source selection
        reasoning_parts.append(f"Selected {len(sources)} most relevant sources based on relevance scores.")

        # Explain synthesis strategy
        strategy_explanations = {
            SynthesisStrategy.COMPREHENSIVE: "Used comprehensive synthesis to cover all aspects of the query.",
            SynthesisStrategy.FOCUSED: "Focused on the most relevant information to provide a targeted answer.",
            SynthesisStrategy.COMPARATIVE: "Compared and contrasted information from different sources.",
            SynthesisStrategy.ANALYTICAL: "Provided deep analysis with reasoning and implications.",
            SynthesisStrategy.SUMMARY: "Summarized key information in a concise format."
        }

        reasoning_parts.append(strategy_explanations.get(config.strategy, "Applied standard synthesis approach."))

        # Explain format choice
        reasoning_parts.append(f"Formatted answer as {config.format.value} for optimal readability.")

        return " ".join(reasoning_parts)

    def _generate_cache_key(self, request: SynthesisRequest) -> str:
        """Generate cache key for the synthesis request."""

        # Create a hash of the key components
        import hashlib

        key_components = [
            request.query,
            str(sorted([source.chunk_id for source in request.sources])),
            str(request.config.strategy.value),
            str(request.config.format.value),
            str(request.config.model.value)
        ]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_answer(self, cache_key: str, config: SynthesisConfig) -> Optional[SynthesizedAnswer]:
        """Get cached answer if available and not expired."""

        if cache_key not in self._cache:
            return None

        answer, cached_time = self._cache[cache_key]

        # Check if cache is expired
        cache_age_hours = (datetime.now(timezone.utc) - cached_time).total_seconds() / 3600
        if cache_age_hours > config.cache_ttl_hours:
            del self._cache[cache_key]
            return None

        return answer

    def _cache_answer(self, cache_key: str, answer: SynthesizedAnswer, config: SynthesisConfig) -> None:
        """Cache the synthesized answer."""

        # Clean cache if it's getting too large
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entries
            sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:self._max_cache_size // 4]:
                del self._cache[key]

        self._cache[cache_key] = (answer, datetime.now(timezone.utc))

    def _update_average_synthesis_time(self, synthesis_time_ms: int) -> None:
        """Update average synthesis time statistics."""

        current_avg = self._stats["average_synthesis_time_ms"]
        total_syntheses = self._stats["successful_syntheses"]

        if total_syntheses == 1:
            self._stats["average_synthesis_time_ms"] = synthesis_time_ms
        else:
            # Calculate running average
            self._stats["average_synthesis_time_ms"] = (
                (current_avg * (total_syntheses - 1) + synthesis_time_ms) / total_syntheses
            )

    def _initialize_prompt_templates(self) -> Dict[SynthesisStrategy, str]:
        """Initialize prompt templates for different synthesis strategies."""

        templates = {
            SynthesisStrategy.COMPREHENSIVE: """
Based on the following retrieved information, provide a comprehensive answer to the user's question.

Question: {query}

Retrieved Information:
{sources}

Citation References:
{citations}

Instructions:
1. Synthesize information from all relevant sources
2. Include proper citations using [1], [2], etc. format
3. Provide a complete and thorough answer covering all aspects
4. Highlight any conflicting information found
5. Note if information is incomplete or uncertain
6. Structure your answer clearly with appropriate sections
7. Aim for {min_words}-{max_words} words

{format_instructions}

Answer:
""",

            SynthesisStrategy.FOCUSED: """
Based on the following retrieved information, provide a focused answer to the user's question.

Question: {query}

Retrieved Information:
{sources}

Citation References:
{citations}

Instructions:
1. Focus on the most relevant information for the specific query
2. Include proper citations using [1], [2], etc. format
3. Provide a direct and targeted answer
4. Prioritize accuracy and relevance over comprehensiveness
5. Note any important limitations or uncertainties
6. Aim for {min_words}-{max_words} words

{format_instructions}

Answer:
""",

            SynthesisStrategy.COMPARATIVE: """
Based on the following retrieved information, provide a comparative analysis to answer the user's question.

Question: {query}

Retrieved Information:
{sources}

Citation References:
{citations}

Instructions:
1. Compare and contrast information from different sources
2. Include proper citations using [1], [2], etc. format
3. Highlight similarities and differences between sources
4. Present multiple perspectives where they exist
5. Identify any conflicting viewpoints and explain them
6. Provide a balanced synthesis of the information
7. Aim for {min_words}-{max_words} words

{format_instructions}

Answer:
""",

            SynthesisStrategy.ANALYTICAL: """
Based on the following retrieved information, provide an analytical answer to the user's question.

Question: {query}

Retrieved Information:
{sources}

Citation References:
{citations}

Instructions:
1. Analyze the information deeply with reasoning and implications
2. Include proper citations using [1], [2], etc. format
3. Explain the significance and context of the information
4. Draw logical conclusions and inferences
5. Identify patterns, trends, or relationships
6. Discuss implications and potential outcomes
7. Aim for {min_words}-{max_words} words

{format_instructions}

Answer:
""",

            SynthesisStrategy.SUMMARY: """
Based on the following retrieved information, provide a concise summary to answer the user's question.

Question: {query}

Retrieved Information:
{sources}

Citation References:
{citations}

Instructions:
1. Summarize the key information concisely
2. Include proper citations using [1], [2], etc. format
3. Focus on the most important points
4. Maintain accuracy while being brief
5. Highlight critical facts and findings
6. Aim for {min_words}-{max_words} words

{format_instructions}

Answer:
"""
        }

        return templates

    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get synthesis service statistics."""
        return self._stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the synthesis service."""

        health_status = {
            "status": "healthy",
            "llm_client_available": self._llm_client is not None,
            "cache_size": len(self._cache),
            "total_syntheses": self._stats["total_syntheses"],
            "success_rate": (
                self._stats["successful_syntheses"] / self._stats["total_syntheses"]
                if self._stats["total_syntheses"] > 0 else 0
            ),
            "average_synthesis_time_ms": self._stats["average_synthesis_time_ms"]
        }

        # Check LLM client health if available
        if self._llm_client:
            try:
                llm_health = await self._llm_client.health_check()
                health_status["llm_client_health"] = llm_health.status
            except Exception as e:
                health_status["llm_client_health"] = f"error: {e}"
                health_status["status"] = "degraded"

        return health_status

    async def shutdown(self) -> None:
        """Shutdown the synthesis service."""
        logger.info("Shutting down answer synthesis service")
        self._cache.clear()


# Global service instance
_answer_synthesis_service: Optional[AnswerSynthesisService] = None


async def get_answer_synthesis_service() -> AnswerSynthesisService:
    """Get or create the global answer synthesis service instance."""
    global _answer_synthesis_service

    if _answer_synthesis_service is None:
        _answer_synthesis_service = AnswerSynthesisService()
        await _answer_synthesis_service.initialize()

    return _answer_synthesis_service


async def close_answer_synthesis_service() -> None:
    """Close the global answer synthesis service instance."""
    global _answer_synthesis_service

    if _answer_synthesis_service:
        await _answer_synthesis_service.shutdown()
        _answer_synthesis_service = None


# Utility functions for integration with existing services

def convert_vector_search_results_to_sources(results: List[VectorSearchResult]) -> List[SourceChunk]:
    """Convert VectorSearchResult objects to SourceChunk objects."""

    sources = []

    for i, result in enumerate(results):
        source = SourceChunk(
            chunk_id=result.chunk_id,
            content=result.content,
            document_id=result.document_id,
            document_title=result.metadata.get('document_title', f'Document {result.document_id}'),
            document_type=result.metadata.get('document_type', 'unknown'),
            section_title=result.metadata.get('section_title'),
            page_number=result.metadata.get('page_number'),
            confidence_score=result.score,
            relevance_score=result.score,
            chunk_index=i,
            metadata=result.metadata
        )
        sources.append(source)

    return sources


def convert_three_hop_results_to_sources(results: ThreeHopResults) -> List[SourceChunk]:
    """Convert ThreeHopResults to SourceChunk objects."""

    sources = []
    chunk_index = 0

    # Convert H3 chunks (most relevant for answer synthesis)
    if results.h3_chunks:
        for chunk in results.h3_chunks:
            source = SourceChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                document_id=chunk.document_id,
                document_title=chunk.metadata.get('document_title', f'Document {chunk.document_id}'),
                document_type=chunk.metadata.get('document_type', 'unknown'),
                section_title=chunk.metadata.get('section_title'),
                page_number=chunk.metadata.get('page_number'),
                confidence_score=chunk.score,
                relevance_score=chunk.score,
                chunk_index=chunk_index,
                metadata=chunk.metadata
            )
            sources.append(source)
            chunk_index += 1

    # Also include H2 offers if they contain useful information
    if results.h2_offers:
        for offer in results.h2_offers:
            source = SourceChunk(
                chunk_id=offer.chunk_id,
                content=offer.content,
                document_id=offer.document_id,
                document_title=offer.metadata.get('document_title', f'Offer {offer.document_id}'),
                document_type='offer',
                section_title=offer.metadata.get('section_title'),
                page_number=offer.metadata.get('page_number'),
                confidence_score=offer.score,
                relevance_score=offer.score,
                chunk_index=chunk_index,
                metadata=offer.metadata
            )
            sources.append(source)
            chunk_index += 1

    return sources


async def synthesize_answer_from_search_results(
    query: str,
    search_results: List[VectorSearchResult],
    tenant_id: UUID,
    user_id: UUID,
    config: Optional[SynthesisConfig] = None
) -> SynthesisResponse:
    """Convenience function to synthesize answer from vector search results."""

    # Convert search results to source chunks
    sources = convert_vector_search_results_to_sources(search_results)

    # Create synthesis request
    request = SynthesisRequest(
        query=query,
        sources=sources,
        config=config or SynthesisConfig(),
        tenant_id=tenant_id,
        user_id=user_id
    )

    # Get synthesis service and synthesize answer
    synthesis_service = await get_answer_synthesis_service()
    return await synthesis_service.synthesize_answer(request)


async def synthesize_answer_from_three_hop_results(
    query: str,
    three_hop_results: ThreeHopResults,
    tenant_id: UUID,
    user_id: UUID,
    config: Optional[SynthesisConfig] = None
) -> SynthesisResponse:
    """Convenience function to synthesize answer from three-hop search results."""

    # Convert three-hop results to source chunks
    sources = convert_three_hop_results_to_sources(three_hop_results)

    # Create synthesis request
    request = SynthesisRequest(
        query=query,
        sources=sources,
        config=config or SynthesisConfig(),
        tenant_id=tenant_id,
        user_id=user_id
    )

    # Get synthesis service and synthesize answer
    synthesis_service = await get_answer_synthesis_service()
    return await synthesis_service.synthesize_answer(request)
