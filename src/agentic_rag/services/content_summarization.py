"""
Content Summarization Service

This module implements intelligent content summarization capabilities including:
- Extractive summarization algorithms
- Abstractive summarization with LLM
- Key point identification and ranking
- Summary length control
- Summarization quality assessment
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from enum import Enum
from dataclasses import dataclass
import structlog
from pydantic import BaseModel, Field
import nltk
from collections import Counter
import math

from agentic_rag.services.llm_client import LLMClientService

logger = structlog.get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class SummarizationType(str, Enum):
    """Types of summarization."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"
    KEY_POINTS = "key_points"
    EXECUTIVE = "executive"


class SummaryLength(str, Enum):
    """Summary length options."""
    BRIEF = "brief"          # 1-2 sentences
    SHORT = "short"          # 3-5 sentences
    MEDIUM = "medium"        # 1-2 paragraphs
    LONG = "long"           # 3-4 paragraphs
    DETAILED = "detailed"    # 5+ paragraphs


class QualityMetric(str, Enum):
    """Quality assessment metrics."""
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    CONCISENESS = "conciseness"
    ACCURACY = "accuracy"


@dataclass
class KeyPoint:
    """Represents a key point extracted from content."""
    point_id: str
    content: str
    importance_score: float
    source_sentence: str
    position: int
    category: str


@dataclass
class SentenceScore:
    """Represents a scored sentence for extractive summarization."""
    sentence: str
    score: float
    position: int
    word_count: int
    features: Dict[str, float]


class SummaryResult(BaseModel):
    """Result of content summarization."""
    summary_id: str = Field(default_factory=lambda: str(uuid4()))
    summary_type: SummarizationType
    summary_length: SummaryLength
    content: str
    key_points: List[KeyPoint]
    word_count: int
    sentence_count: int
    compression_ratio: float
    quality_scores: Dict[QualityMetric, float]
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any]


class ContentSummarizationConfig(BaseModel):
    """Configuration for content summarization."""
    default_summary_type: SummarizationType = SummarizationType.HYBRID
    default_length: SummaryLength = SummaryLength.MEDIUM
    max_key_points: int = 10
    min_sentence_length: int = 10
    max_sentence_length: int = 500
    enable_quality_assessment: bool = True
    openai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.3


class ContentSummarizationService:
    """Service for intelligent content summarization."""
    
    def __init__(self, config: Optional[ContentSummarizationConfig] = None):
        self.config = config or ContentSummarizationConfig()
        self.llm_client = LLMClientService()
        
        # Load stopwords
        try:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Statistics tracking
        self._stats = {
            "total_summarizations": 0,
            "successful_summarizations": 0,
            "failed_summarizations": 0,
            "total_processing_time_ms": 0.0,
            "average_processing_time_ms": 0.0,
            "total_words_processed": 0,
            "total_words_generated": 0,
            "average_compression_ratio": 0.0
        }
        
        # Cache for repeated summarizations
        self._summary_cache: Dict[str, SummaryResult] = {}
        
        logger.info("Content summarization service initialized")
    
    async def summarize_content(self, content: str, 
                              summary_type: Optional[SummarizationType] = None,
                              length: Optional[SummaryLength] = None) -> SummaryResult:
        """Summarize content using specified type and length."""
        
        start_time = time.time()
        
        try:
            # Use defaults if not specified
            summary_type = summary_type or self.config.default_summary_type
            length = length or self.config.default_length
            
            # Generate cache key
            cache_key = f"{hash(content)}:{summary_type.value}:{length.value}"
            
            # Check cache
            if cache_key in self._summary_cache:
                logger.info("Returning cached summary result")
                return self._summary_cache[cache_key]
            
            # Preprocess content
            preprocessed_content = self._preprocess_content(content)
            
            # Extract key points first
            key_points = await self._extract_key_points(preprocessed_content)
            
            # Generate summary based on type
            if summary_type == SummarizationType.EXTRACTIVE:
                summary_content = await self._extractive_summarization(preprocessed_content, length)
            elif summary_type == SummarizationType.ABSTRACTIVE:
                summary_content = await self._abstractive_summarization(preprocessed_content, length)
            elif summary_type == SummarizationType.HYBRID:
                summary_content = await self._hybrid_summarization(preprocessed_content, length)
            elif summary_type == SummarizationType.KEY_POINTS:
                summary_content = self._key_points_summarization(key_points, length)
            elif summary_type == SummarizationType.EXECUTIVE:
                summary_content = await self._executive_summarization(preprocessed_content, key_points, length)
            else:
                raise ValueError(f"Unsupported summary type: {summary_type}")
            
            # Calculate metrics
            word_count = len(summary_content.split())
            sentence_count = len(nltk.sent_tokenize(summary_content))
            original_word_count = len(content.split())
            compression_ratio = word_count / original_word_count if original_word_count > 0 else 0
            
            # Assess quality
            quality_scores = {}
            if self.config.enable_quality_assessment:
                quality_scores = await self._assess_quality(content, summary_content, key_points)
            
            # Calculate confidence
            confidence = self._calculate_confidence(quality_scores, compression_ratio, word_count)
            
            # Create result
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = SummaryResult(
                summary_type=summary_type,
                summary_length=length,
                content=summary_content,
                key_points=key_points,
                word_count=word_count,
                sentence_count=sentence_count,
                compression_ratio=compression_ratio,
                quality_scores=quality_scores,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                metadata={
                    "original_word_count": original_word_count,
                    "original_sentence_count": len(nltk.sent_tokenize(content)),
                    "key_points_count": len(key_points)
                }
            )
            
            # Cache result
            self._summary_cache[cache_key] = result
            
            # Update statistics
            self._update_stats("success", processing_time_ms, original_word_count, word_count, compression_ratio)
            
            logger.info(
                "Content summarization completed",
                summary_type=summary_type.value,
                length=length.value,
                word_count=word_count,
                compression_ratio=compression_ratio,
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats("failure", processing_time_ms, 0, 0, 0)
            
            logger.error(
                "Content summarization failed",
                error=str(e),
                summary_type=summary_type.value if summary_type else "unknown",
                processing_time_ms=processing_time_ms
            )
            raise
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for summarization."""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove very short lines (likely formatting artifacts)
        lines = content.split('\n')
        filtered_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(filtered_lines)
    
    async def _extract_key_points(self, content: str) -> List[KeyPoint]:
        """Extract key points from content."""
        
        sentences = nltk.sent_tokenize(content)
        key_points = []
        
        # Score sentences for importance
        sentence_scores = self._score_sentences(sentences)
        
        # Select top sentences as key points
        top_sentences = sorted(sentence_scores, key=lambda x: x.score, reverse=True)[:self.config.max_key_points]
        
        for i, scored_sentence in enumerate(top_sentences):
            # Categorize the key point
            category = self._categorize_key_point(scored_sentence.sentence)
            
            key_point = KeyPoint(
                point_id=str(uuid4()),
                content=scored_sentence.sentence.strip(),
                importance_score=scored_sentence.score,
                source_sentence=scored_sentence.sentence,
                position=scored_sentence.position,
                category=category
            )
            key_points.append(key_point)
        
        # Sort by position to maintain document order
        key_points.sort(key=lambda x: x.position)
        
        return key_points

    def _score_sentences(self, sentences: List[str]) -> List[SentenceScore]:
        """Score sentences for extractive summarization."""

        scored_sentences = []

        # Calculate word frequencies
        all_words = []
        for sentence in sentences:
            words = [word.lower() for word in re.findall(r'\b\w+\b', sentence)
                    if word.lower() not in self.stopwords and len(word) > 2]
            all_words.extend(words)

        word_freq = Counter(all_words)
        max_freq = max(word_freq.values()) if word_freq else 1

        for i, sentence in enumerate(sentences):
            if len(sentence) < self.config.min_sentence_length:
                continue

            words = [word.lower() for word in re.findall(r'\b\w+\b', sentence)
                    if word.lower() not in self.stopwords and len(word) > 2]

            # Calculate features
            features = {}

            # Word frequency score
            features['word_freq'] = sum(word_freq[word] / max_freq for word in words) / len(words) if words else 0

            # Position score (earlier sentences get higher scores)
            features['position'] = 1.0 - (i / len(sentences))

            # Length score (prefer medium-length sentences)
            word_count = len(words)
            if 10 <= word_count <= 30:
                features['length'] = 1.0
            elif word_count < 10:
                features['length'] = word_count / 10
            else:
                features['length'] = 30 / word_count

            # Keyword score (look for important keywords)
            important_keywords = ['requirement', 'important', 'critical', 'key', 'main', 'primary', 'essential']
            features['keywords'] = sum(1 for keyword in important_keywords
                                     if keyword in sentence.lower()) / len(important_keywords)

            # Numerical data score (sentences with numbers are often important)
            features['numerical'] = len(re.findall(r'\d+', sentence)) / 10

            # Composite score
            score = (features['word_freq'] * 0.4 +
                    features['position'] * 0.2 +
                    features['length'] * 0.2 +
                    features['keywords'] * 0.15 +
                    features['numerical'] * 0.05)

            scored_sentences.append(SentenceScore(
                sentence=sentence,
                score=score,
                position=i,
                word_count=word_count,
                features=features
            ))

        return scored_sentences

    def _categorize_key_point(self, sentence: str) -> str:
        """Categorize a key point based on its content."""

        sentence_lower = sentence.lower()

        if any(word in sentence_lower for word in ['requirement', 'must', 'shall', 'should']):
            return "requirement"
        elif any(word in sentence_lower for word in ['cost', 'price', 'budget', 'fee', '$']):
            return "financial"
        elif any(word in sentence_lower for word in ['technical', 'system', 'software', 'hardware']):
            return "technical"
        elif any(word in sentence_lower for word in ['timeline', 'schedule', 'deadline', 'delivery']):
            return "timeline"
        elif any(word in sentence_lower for word in ['risk', 'issue', 'problem', 'concern']):
            return "risk"
        else:
            return "general"

    async def _extractive_summarization(self, content: str, length: SummaryLength) -> str:
        """Generate extractive summary by selecting important sentences."""

        sentences = nltk.sent_tokenize(content)
        scored_sentences = self._score_sentences(sentences)

        # Determine number of sentences based on length
        target_sentences = self._get_target_sentence_count(len(sentences), length)

        # Select top sentences
        top_sentences = sorted(scored_sentences, key=lambda x: x.score, reverse=True)[:target_sentences]

        # Sort by original position to maintain flow
        top_sentences.sort(key=lambda x: x.position)

        # Combine sentences
        summary = ' '.join([s.sentence for s in top_sentences])

        return summary

    async def _abstractive_summarization(self, content: str, length: SummaryLength) -> str:
        """Generate abstractive summary using LLM."""

        # Determine target word count
        target_words = self._get_target_word_count(length)

        prompt = f"""Please provide a concise summary of the following content in approximately {target_words} words.
Focus on the main points, key requirements, and important details. Maintain a professional tone.

Content:
{content}

Summary:"""

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.openai_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Abstractive summarization failed, falling back to extractive: {e}")
            return await self._extractive_summarization(content, length)

    async def _hybrid_summarization(self, content: str, length: SummaryLength) -> str:
        """Generate hybrid summary combining extractive and abstractive approaches."""

        # First, get key sentences using extractive method
        sentences = nltk.sent_tokenize(content)
        scored_sentences = self._score_sentences(sentences)

        # Select top sentences for abstractive processing
        target_sentences = min(10, len(scored_sentences))  # Limit input for LLM
        top_sentences = sorted(scored_sentences, key=lambda x: x.score, reverse=True)[:target_sentences]
        top_sentences.sort(key=lambda x: x.position)

        # Combine top sentences
        key_content = ' '.join([s.sentence for s in top_sentences])

        # Use abstractive summarization on the key content
        return await self._abstractive_summarization(key_content, length)

    def _key_points_summarization(self, key_points: List[KeyPoint], length: SummaryLength) -> str:
        """Generate summary as a list of key points."""

        # Determine number of points based on length
        if length == SummaryLength.BRIEF:
            max_points = 3
        elif length == SummaryLength.SHORT:
            max_points = 5
        elif length == SummaryLength.MEDIUM:
            max_points = 8
        else:
            max_points = len(key_points)

        # Select top key points
        selected_points = key_points[:max_points]

        # Format as bullet points
        summary_lines = []
        for point in selected_points:
            summary_lines.append(f"• {point.content}")

        return '\n'.join(summary_lines)

    async def _executive_summarization(self, content: str, key_points: List[KeyPoint],
                                     length: SummaryLength) -> str:
        """Generate executive summary with structured format."""

        target_words = self._get_target_word_count(length)

        # Categorize key points
        categorized_points = {}
        for point in key_points:
            category = point.category
            if category not in categorized_points:
                categorized_points[category] = []
            categorized_points[category].append(point.content)

        # Build structured prompt
        prompt = f"""Create an executive summary of approximately {target_words} words based on the following content and key points.

Structure the summary with clear sections and focus on business-critical information.

Original Content:
{content[:2000]}...

Key Points by Category:
"""

        for category, points in categorized_points.items():
            prompt += f"\n{category.title()}:\n"
            for point in points[:3]:  # Limit points per category
                prompt += f"- {point}\n"

        prompt += "\nExecutive Summary:"

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.openai_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Executive summarization failed, falling back to hybrid: {e}")
            return await self._hybrid_summarization(content, length)

    def _get_target_sentence_count(self, total_sentences: int, length: SummaryLength) -> int:
        """Get target sentence count based on length and total sentences."""

        if length == SummaryLength.BRIEF:
            return min(2, max(1, total_sentences // 10))
        elif length == SummaryLength.SHORT:
            return min(5, max(2, total_sentences // 8))
        elif length == SummaryLength.MEDIUM:
            return min(10, max(3, total_sentences // 5))
        elif length == SummaryLength.LONG:
            return min(15, max(5, total_sentences // 3))
        else:  # DETAILED
            return min(20, max(8, total_sentences // 2))

    def _get_target_word_count(self, length: SummaryLength) -> int:
        """Get target word count based on length."""

        if length == SummaryLength.BRIEF:
            return 50
        elif length == SummaryLength.SHORT:
            return 100
        elif length == SummaryLength.MEDIUM:
            return 200
        elif length == SummaryLength.LONG:
            return 400
        else:  # DETAILED
            return 600

    async def _assess_quality(self, original_content: str, summary: str,
                            key_points: List[KeyPoint]) -> Dict[QualityMetric, float]:
        """Assess the quality of the generated summary."""

        quality_scores = {}

        # Relevance: How well does the summary capture key information
        original_words = set(re.findall(r'\b\w+\b', original_content.lower()))
        summary_words = set(re.findall(r'\b\w+\b', summary.lower()))
        key_point_words = set()
        for point in key_points:
            key_point_words.update(re.findall(r'\b\w+\b', point.content.lower()))

        if original_words:
            relevance = len(summary_words & key_point_words) / len(key_point_words) if key_point_words else 0.5
            quality_scores[QualityMetric.RELEVANCE] = min(1.0, relevance)
        else:
            quality_scores[QualityMetric.RELEVANCE] = 0.5

        # Coherence: Basic check for sentence structure
        sentences = nltk.sent_tokenize(summary)
        coherence = 1.0 if all(len(s.split()) >= 3 for s in sentences) else 0.7
        quality_scores[QualityMetric.COHERENCE] = coherence

        # Completeness: Coverage of key topics
        key_topics = len(set(point.category for point in key_points))
        summary_topics = 0
        for category in ['requirement', 'financial', 'technical', 'timeline', 'risk']:
            if any(word in summary.lower() for word in [category]):
                summary_topics += 1

        completeness = summary_topics / key_topics if key_topics > 0 else 0.8
        quality_scores[QualityMetric.COMPLETENESS] = min(1.0, completeness)

        # Conciseness: Appropriate length
        original_words_count = len(original_content.split())
        summary_words_count = len(summary.split())
        compression_ratio = summary_words_count / original_words_count if original_words_count > 0 else 0

        # Ideal compression ratio is between 0.1 and 0.3
        if 0.1 <= compression_ratio <= 0.3:
            conciseness = 1.0
        elif compression_ratio < 0.1:
            conciseness = compression_ratio / 0.1
        else:
            conciseness = 0.3 / compression_ratio

        quality_scores[QualityMetric.CONCISENESS] = min(1.0, conciseness)

        # Accuracy: Basic check (simplified)
        quality_scores[QualityMetric.ACCURACY] = 0.8  # Default assumption

        return quality_scores

    def _calculate_confidence(self, quality_scores: Dict[QualityMetric, float],
                            compression_ratio: float, word_count: int) -> float:
        """Calculate overall confidence in the summary."""

        if not quality_scores:
            return 0.7  # Default confidence

        # Average quality scores
        avg_quality = sum(quality_scores.values()) / len(quality_scores)

        # Adjust for compression ratio
        compression_factor = 1.0
        if compression_ratio < 0.05:  # Too compressed
            compression_factor = 0.8
        elif compression_ratio > 0.5:  # Not compressed enough
            compression_factor = 0.9

        # Adjust for word count
        word_count_factor = 1.0
        if word_count < 20:  # Too short
            word_count_factor = 0.8
        elif word_count > 1000:  # Too long
            word_count_factor = 0.9

        confidence = avg_quality * compression_factor * word_count_factor
        return min(1.0, max(0.0, confidence))

    def _update_stats(self, result: str, processing_time_ms: float,
                     words_processed: int, words_generated: int, compression_ratio: float) -> None:
        """Update service statistics."""

        self._stats["total_summarizations"] += 1
        self._stats["total_processing_time_ms"] += processing_time_ms
        self._stats["total_words_processed"] += words_processed
        self._stats["total_words_generated"] += words_generated

        if result == "success":
            self._stats["successful_summarizations"] += 1
        else:
            self._stats["failed_summarizations"] += 1

        # Update averages
        if self._stats["total_summarizations"] > 0:
            self._stats["average_processing_time_ms"] = (
                self._stats["total_processing_time_ms"] / self._stats["total_summarizations"]
            )

            total_compressions = self._stats.get("total_compressions", 0) + compression_ratio
            self._stats["total_compressions"] = total_compressions
            self._stats["average_compression_ratio"] = total_compressions / self._stats["total_summarizations"]

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""

        return {
            **self._stats,
            "cache_size": len(self._summary_cache),
            "config": self.config.dict()
        }

    def clear_cache(self) -> None:
        """Clear summary cache."""

        self._summary_cache.clear()
        logger.info("Summary cache cleared")

    async def batch_summarize(self, contents: List[str],
                            summary_type: Optional[SummarizationType] = None,
                            length: Optional[SummaryLength] = None) -> List[SummaryResult]:
        """Summarize multiple contents in batch."""

        results = []

        # Process summaries in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent summarizations

        async def summarize_with_semaphore(content: str) -> SummaryResult:
            async with semaphore:
                return await self.summarize_content(content, summary_type, length)

        tasks = [summarize_with_semaphore(content) for content in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch summarization failed for content {i}", error=str(result))
            else:
                valid_results.append(result)

        logger.info(f"Batch summarization completed: {len(valid_results)}/{len(contents)} successful")

        return valid_results


# Global service instance
_content_summarization_service: Optional[ContentSummarizationService] = None


def get_content_summarization_service() -> ContentSummarizationService:
    """Get or create the global content summarization service instance."""
    global _content_summarization_service

    if _content_summarization_service is None:
        _content_summarization_service = ContentSummarizationService()

    return _content_summarization_service


def reset_content_summarization_service() -> None:
    """Reset the global content summarization service instance."""
    global _content_summarization_service
    _content_summarization_service = None
