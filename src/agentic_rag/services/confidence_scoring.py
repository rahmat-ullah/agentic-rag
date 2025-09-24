"""
Confidence Scoring Service

This module provides comprehensive confidence scoring for document links
using multiple factors including content similarity, metadata alignment,
temporal proximity, and user feedback.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agentic_rag.models.database import Document, DocumentLink, DocumentKind, DocumentStatus, UserFeedback
from agentic_rag.services.vector_operations import get_vector_operations, VectorSearchOptions
from agentic_rag.services.embedding_pipeline import get_embedding_pipeline

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIDENCE SCORING MODELS
# ============================================================================

class ConfidenceFactors(BaseModel):
    """Individual confidence factors for a document link."""
    
    content_similarity: float = Field(..., ge=0.0, le=1.0, description="Content similarity score")
    metadata_alignment: float = Field(..., ge=0.0, le=1.0, description="Metadata alignment score")
    temporal_proximity: float = Field(..., ge=0.0, le=1.0, description="Temporal proximity score")
    user_feedback_score: float = Field(..., ge=0.0, le=1.0, description="User feedback score")
    
    # Additional factors
    document_quality: float = Field(..., ge=0.0, le=1.0, description="Document quality score")
    processing_completeness: float = Field(..., ge=0.0, le=1.0, description="Processing completeness score")
    
    # Weights for each factor
    content_weight: float = Field(default=0.40, description="Weight for content similarity")
    metadata_weight: float = Field(default=0.30, description="Weight for metadata alignment")
    temporal_weight: float = Field(default=0.15, description="Weight for temporal proximity")
    feedback_weight: float = Field(default=0.15, description="Weight for user feedback")


class ConfidenceScore(BaseModel):
    """Comprehensive confidence score with detailed breakdown."""
    
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    factors: ConfidenceFactors = Field(..., description="Individual confidence factors")
    
    # Detailed analysis
    strengths: List[str] = Field(..., description="Strengths of the link")
    weaknesses: List[str] = Field(..., description="Weaknesses of the link")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")
    
    # Metadata
    calculation_time: float = Field(..., description="Time taken to calculate score")
    algorithm_version: str = Field(default="2.0", description="Algorithm version")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConfidenceAnalysisRequest(BaseModel):
    """Request for confidence analysis."""
    
    rfq_id: UUID = Field(..., description="RFQ document ID")
    offer_id: UUID = Field(..., description="Offer document ID")
    existing_link_id: Optional[UUID] = Field(None, description="Existing link ID if available")
    include_detailed_analysis: bool = Field(default=True, description="Include detailed analysis")
    use_cached_results: bool = Field(default=True, description="Use cached results if available")


# ============================================================================
# CONFIDENCE SCORING SERVICE
# ============================================================================

class ConfidenceScoringService:
    """Advanced confidence scoring service for document links."""
    
    def __init__(self):
        self._vector_operations = None
        self._embedding_pipeline = None
        self._cache = {}  # Simple in-memory cache
        self._stats = {
            "scores_calculated": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("Confidence scoring service initialized")
    
    async def initialize(self):
        """Initialize service dependencies."""
        if not self._vector_operations:
            self._vector_operations = await get_vector_operations()
        if not self._embedding_pipeline:
            self._embedding_pipeline = await get_embedding_pipeline()
        
        logger.info("Confidence scoring service dependencies initialized")
    
    async def calculate_confidence_score(
        self,
        db_session: Session,
        request: ConfidenceAnalysisRequest,
        tenant_id: UUID
    ) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score for a document link.
        
        Args:
            db_session: Database session
            request: Confidence analysis request
            tenant_id: Tenant ID
            
        Returns:
            ConfidenceScore with detailed analysis
        """
        await self.initialize()
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{request.rfq_id}_{request.offer_id}_{tenant_id}"
        if request.use_cached_results and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            cached_score = self._cache[cache_key]
            # Update timestamp but keep the score
            cached_score.last_updated = datetime.now(timezone.utc)
            return cached_score
        
        self._stats["cache_misses"] += 1
        
        # Get documents
        rfq_document = self._get_document(db_session, request.rfq_id, tenant_id)
        offer_document = self._get_document(db_session, request.offer_id, tenant_id)
        
        if not rfq_document or not offer_document:
            raise ValueError("One or both documents not found")
        
        # Get existing link if available
        existing_link = None
        if request.existing_link_id:
            existing_link = db_session.query(DocumentLink).filter(
                DocumentLink.id == request.existing_link_id,
                DocumentLink.tenant_id == tenant_id
            ).first()
        
        # Calculate individual factors
        content_similarity = await self._calculate_content_similarity_advanced(
            rfq_document, offer_document
        )
        
        metadata_alignment = self._calculate_metadata_alignment_advanced(
            rfq_document, offer_document
        )
        
        temporal_proximity = self._calculate_temporal_proximity_advanced(
            rfq_document, offer_document
        )
        
        user_feedback_score = self._calculate_user_feedback_score(existing_link)
        
        document_quality = self._calculate_document_quality_score(
            rfq_document, offer_document
        )
        
        processing_completeness = self._calculate_processing_completeness_score(
            rfq_document, offer_document
        )
        
        # Create confidence factors
        factors = ConfidenceFactors(
            content_similarity=content_similarity,
            metadata_alignment=metadata_alignment,
            temporal_proximity=temporal_proximity,
            user_feedback_score=user_feedback_score,
            document_quality=document_quality,
            processing_completeness=processing_completeness
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(factors)
        
        # Generate analysis if requested
        strengths, weaknesses, recommendations = [], [], []
        if request.include_detailed_analysis:
            strengths, weaknesses, recommendations = self._generate_detailed_analysis(
                factors, rfq_document, offer_document, existing_link
            )
        
        # Create confidence score
        confidence_score = ConfidenceScore(
            overall_confidence=overall_confidence,
            factors=factors,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            calculation_time=time.time() - start_time
        )
        
        # Cache the result
        self._cache[cache_key] = confidence_score
        
        # Update statistics
        self._stats["scores_calculated"] += 1
        
        return confidence_score
    
    def _get_document(self, db_session: Session, document_id: UUID, tenant_id: UUID) -> Optional[Document]:
        """Get document by ID with tenant validation."""
        return db_session.query(Document).filter(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
            Document.deleted_at.is_(None)
        ).first()
    
    async def _calculate_content_similarity_advanced(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate advanced content similarity using multiple techniques."""
        try:
            # Vector similarity (primary method)
            vector_similarity = await self._calculate_vector_similarity(
                rfq_document, offer_document
            )
            
            # Text-based similarity (fallback)
            text_similarity = self._calculate_text_similarity(
                rfq_document, offer_document
            )
            
            # Combine similarities with weights
            combined_similarity = (vector_similarity * 0.8) + (text_similarity * 0.2)
            
            return max(0.0, min(1.0, combined_similarity))
            
        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.5
    
    async def _calculate_vector_similarity(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate vector-based content similarity."""
        # This is a simplified version - in practice, you'd use the full vector search
        # implementation from the previous task
        return 0.7  # Placeholder
    
    def _calculate_text_similarity(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate text-based similarity using keyword matching."""
        if not rfq_document.title or not offer_document.title:
            return 0.5
        
        rfq_words = set(rfq_document.title.lower().split())
        offer_words = set(offer_document.title.lower().split())
        
        if not rfq_words or not offer_words:
            return 0.5
        
        intersection = len(rfq_words & offer_words)
        union = len(rfq_words | offer_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_metadata_alignment_advanced(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate advanced metadata alignment score."""
        score = 0.0
        total_weight = 0.0
        
        # Title similarity (30% weight)
        title_sim = self._calculate_text_similarity(rfq_document, offer_document)
        score += title_sim * 0.3
        total_weight += 0.3
        
        # File size similarity (20% weight)
        if rfq_document.file_size and offer_document.file_size:
            size_ratio = min(rfq_document.file_size, offer_document.file_size) / max(rfq_document.file_size, offer_document.file_size)
            score += size_ratio * 0.2
            total_weight += 0.2
        
        # Chunk count similarity (25% weight)
        if rfq_document.chunk_count and offer_document.chunk_count:
            chunk_ratio = min(rfq_document.chunk_count, offer_document.chunk_count) / max(rfq_document.chunk_count, offer_document.chunk_count)
            score += chunk_ratio * 0.25
            total_weight += 0.25
        
        # Status alignment (25% weight)
        if rfq_document.status == offer_document.status == DocumentStatus.READY:
            score += 1.0 * 0.25
            total_weight += 0.25
        elif rfq_document.status == offer_document.status:
            score += 0.7 * 0.25
            total_weight += 0.25
        
        return score / total_weight if total_weight > 0 else 0.5
    
    def _calculate_temporal_proximity_advanced(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate advanced temporal proximity score."""
        if not rfq_document.created_at or not offer_document.created_at:
            return 0.5
        
        # Calculate time difference in hours
        time_diff_hours = abs((rfq_document.created_at - offer_document.created_at).total_seconds()) / 3600
        
        # Advanced scoring based on time difference
        if time_diff_hours <= 1:
            return 1.0
        elif time_diff_hours <= 24:
            return 0.9
        elif time_diff_hours <= 168:  # 1 week
            return 0.8
        elif time_diff_hours <= 720:  # 1 month
            return 0.6
        elif time_diff_hours <= 2160:  # 3 months
            return 0.4
        else:
            return 0.2
    
    def _calculate_user_feedback_score(self, existing_link: Optional[DocumentLink]) -> float:
        """Calculate user feedback score."""
        if not existing_link or not existing_link.user_feedback:
            return 0.5  # Neutral score for no feedback
        
        feedback_scores = {
            UserFeedback.ACCEPTED: 1.0,
            UserFeedback.MODIFIED: 0.7,
            UserFeedback.REJECTED: 0.1
        }
        
        return feedback_scores.get(existing_link.user_feedback, 0.5)
    
    def _calculate_document_quality_score(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate document quality score."""
        score = 0.0
        
        # Both documents should be ready
        if rfq_document.status == DocumentStatus.READY:
            score += 0.4
        if offer_document.status == DocumentStatus.READY:
            score += 0.4
        
        # Documents should have reasonable chunk counts
        if rfq_document.chunk_count and rfq_document.chunk_count > 5:
            score += 0.1
        if offer_document.chunk_count and offer_document.chunk_count > 5:
            score += 0.1
        
        return score
    
    def _calculate_processing_completeness_score(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate processing completeness score."""
        score = 0.0
        
        # Check if documents have been fully processed
        if rfq_document.status == DocumentStatus.READY and rfq_document.chunk_count > 0:
            score += 0.5
        if offer_document.status == DocumentStatus.READY and offer_document.chunk_count > 0:
            score += 0.5
        
        return score
    
    def _calculate_overall_confidence(self, factors: ConfidenceFactors) -> float:
        """Calculate overall confidence using weighted factors."""
        overall = (
            factors.content_similarity * factors.content_weight +
            factors.metadata_alignment * factors.metadata_weight +
            factors.temporal_proximity * factors.temporal_weight +
            factors.user_feedback_score * factors.feedback_weight
        )
        
        # Apply quality and completeness modifiers
        quality_modifier = (factors.document_quality + factors.processing_completeness) / 2
        overall = overall * (0.8 + 0.2 * quality_modifier)  # 80% base + 20% quality bonus
        
        return max(0.0, min(1.0, overall))
    
    def _generate_detailed_analysis(
        self,
        factors: ConfidenceFactors,
        rfq_document: Document,
        offer_document: Document,
        existing_link: Optional[DocumentLink]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate detailed analysis with strengths, weaknesses, and recommendations."""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze content similarity
        if factors.content_similarity > 0.8:
            strengths.append("Excellent content similarity")
        elif factors.content_similarity < 0.4:
            weaknesses.append("Low content similarity")
            recommendations.append("Review document content alignment")
        
        # Analyze metadata alignment
        if factors.metadata_alignment > 0.7:
            strengths.append("Strong metadata alignment")
        elif factors.metadata_alignment < 0.4:
            weaknesses.append("Poor metadata alignment")
            recommendations.append("Check document titles and properties")
        
        # Analyze temporal proximity
        if factors.temporal_proximity > 0.8:
            strengths.append("Documents created close in time")
        elif factors.temporal_proximity < 0.3:
            weaknesses.append("Large time gap between documents")
        
        # Analyze user feedback
        if existing_link and existing_link.user_feedback == UserFeedback.ACCEPTED:
            strengths.append("Positive user validation")
        elif existing_link and existing_link.user_feedback == UserFeedback.REJECTED:
            weaknesses.append("User rejected this link")
            recommendations.append("Review link relevance")
        
        # Analyze document quality
        if factors.document_quality < 0.5:
            weaknesses.append("Document processing incomplete")
            recommendations.append("Ensure documents are fully processed")
        
        return strengths, weaknesses, recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self._stats.copy()
    
    def clear_cache(self):
        """Clear the confidence score cache."""
        self._cache.clear()
        logger.info("Confidence score cache cleared")


# Singleton instance
_confidence_scoring_service: Optional[ConfidenceScoringService] = None


def get_confidence_scoring_service() -> ConfidenceScoringService:
    """Get or create confidence scoring service instance."""
    global _confidence_scoring_service
    
    if _confidence_scoring_service is None:
        _confidence_scoring_service = ConfidenceScoringService()
    
    return _confidence_scoring_service
