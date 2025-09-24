"""
Document Similarity Service.

This service provides document similarity analysis for automatic
link suggestions between RFQ and Offer documents.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from uuid import UUID
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from agentic_rag.models.database import Document, DocumentKind, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentSuggestion:
    """Represents a document link suggestion."""
    
    def __init__(
        self,
        offer_document_id: UUID,
        offer_title: str,
        confidence_score: float,
        similarity_factors: Dict[str, float],
        created_at: datetime
    ):
        self.offer_document_id = offer_document_id
        self.offer_title = offer_title
        self.confidence_score = confidence_score
        self.similarity_factors = similarity_factors
        self.created_at = created_at


class DocumentSimilarityService:
    """
    Service for analyzing document similarity and generating link suggestions.
    
    This service provides:
    - Title-based similarity analysis
    - Content-based similarity (when available)
    - Temporal proximity analysis
    - Automatic link suggestions for RFQ documents
    - Confidence scoring for suggested links
    """
    
    def __init__(self):
        self.min_confidence_threshold = 0.3
        self.max_suggestions = 10
        self.temporal_window_days = 90
    
    def calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two document titles.
        
        Args:
            title1: First document title
            title2: Second document title
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not title1 or not title2:
            return 0.0
        
        # Normalize titles
        title1_norm = self._normalize_title(title1)
        title2_norm = self._normalize_title(title2)
        
        if title1_norm == title2_norm:
            return 1.0
        
        # Extract keywords
        keywords1 = self._extract_keywords(title1_norm)
        keywords2 = self._extract_keywords(title2_norm)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        # Boost score for exact phrase matches
        phrase_boost = self._calculate_phrase_similarity(title1_norm, title2_norm)
        
        # Combine scores
        final_score = min(1.0, jaccard_score + phrase_boost * 0.3)
        
        return final_score
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        # Convert to lowercase
        title = title.lower()
        
        # Remove common prefixes/suffixes
        prefixes = ['rfq', 'request for quote', 'offer', 'proposal', 'quote']
        for prefix in prefixes:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        # Remove special characters and extra spaces
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def _extract_keywords(self, title: str) -> set:
        """Extract meaningful keywords from title."""
        # Common stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        words = title.split()
        keywords = set()
        
        for word in words:
            word = word.strip()
            if len(word) >= 3 and word not in stop_words:
                keywords.add(word)
        
        return keywords
    
    def _calculate_phrase_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity based on common phrases."""
        # Split into phrases (2-3 words)
        words1 = title1.split()
        words2 = title2.split()
        
        phrases1 = set()
        phrases2 = set()
        
        # Generate 2-word phrases
        for i in range(len(words1) - 1):
            phrases1.add(f"{words1[i]} {words1[i+1]}")
        
        for i in range(len(words2) - 1):
            phrases2.add(f"{words2[i]} {words2[i+1]}")
        
        # Generate 3-word phrases
        for i in range(len(words1) - 2):
            phrases1.add(f"{words1[i]} {words1[i+1]} {words1[i+2]}")
        
        for i in range(len(words2) - 2):
            phrases2.add(f"{words2[i]} {words2[i+1]} {words2[i+2]}")
        
        if not phrases1 or not phrases2:
            return 0.0
        
        intersection = len(phrases1.intersection(phrases2))
        union = len(phrases1.union(phrases2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_temporal_proximity(self, date1: datetime, date2: datetime) -> float:
        """
        Calculate temporal proximity score between two dates.
        
        Args:
            date1: First document date
            date2: Second document date
            
        Returns:
            Proximity score between 0.0 and 1.0
        """
        if not date1 or not date2:
            return 0.0
        
        # Calculate time difference in days
        time_diff = abs((date1 - date2).days)
        
        if time_diff == 0:
            return 1.0
        
        # Use exponential decay with configurable window
        decay_factor = 0.1  # Adjust for steeper/gentler decay
        proximity_score = max(0.0, 1.0 - (time_diff * decay_factor / self.temporal_window_days))
        
        return proximity_score
    
    async def suggest_document_links(
        self,
        rfq_document_id: UUID,
        tenant_id: UUID,
        db_session: Session,
        max_suggestions: Optional[int] = None
    ) -> List[DocumentSuggestion]:
        """
        Generate automatic link suggestions for an RFQ document.
        
        Args:
            rfq_document_id: RFQ document ID
            tenant_id: Tenant ID for security
            db_session: Database session
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of DocumentSuggestion objects
        """
        max_suggestions = max_suggestions or self.max_suggestions
        
        # Get the RFQ document
        rfq_document = db_session.query(Document).filter(
            and_(
                Document.id == rfq_document_id,
                Document.tenant_id == tenant_id,
                Document.deleted_at.is_(None)
            )
        ).first()
        
        if not rfq_document:
            return []
        
        # Define temporal window
        cutoff_date = rfq_document.created_at - timedelta(days=self.temporal_window_days)
        
        # Find potential offer documents
        potential_offers = db_session.query(Document).filter(
            and_(
                Document.tenant_id == tenant_id,
                Document.kind == DocumentKind.OFFER,
                Document.status == DocumentStatus.READY,
                Document.deleted_at.is_(None),
                Document.id != rfq_document_id,
                Document.created_at >= cutoff_date
            )
        ).all()
        
        suggestions = []
        
        for offer_doc in potential_offers:
            # Calculate similarity factors
            title_similarity = self.calculate_title_similarity(
                rfq_document.title, offer_doc.title
            )
            
            temporal_proximity = self.calculate_temporal_proximity(
                rfq_document.created_at, offer_doc.created_at
            )
            
            # Calculate overall confidence score
            # Weight title similarity more heavily than temporal proximity
            confidence_score = (title_similarity * 0.7) + (temporal_proximity * 0.3)
            
            # Only include suggestions above threshold
            if confidence_score >= self.min_confidence_threshold:
                similarity_factors = {
                    "title_similarity": title_similarity,
                    "temporal_proximity": temporal_proximity,
                    "days_apart": abs((rfq_document.created_at - offer_doc.created_at).days)
                }
                
                suggestion = DocumentSuggestion(
                    offer_document_id=offer_doc.id,
                    offer_title=offer_doc.title,
                    confidence_score=confidence_score,
                    similarity_factors=similarity_factors,
                    created_at=offer_doc.created_at
                )
                
                suggestions.append(suggestion)
        
        # Sort by confidence score (descending) and limit results
        suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
        suggestions = suggestions[:max_suggestions]
        
        logger.info(
            f"Generated {len(suggestions)} link suggestions for RFQ {rfq_document_id} "
            f"from {len(potential_offers)} potential offers"
        )
        
        return suggestions


# Global instance
_similarity_service: Optional[DocumentSimilarityService] = None


def get_document_similarity_service() -> DocumentSimilarityService:
    """Get the global document similarity service instance."""
    global _similarity_service
    if _similarity_service is None:
        _similarity_service = DocumentSimilarityService()
    return _similarity_service
