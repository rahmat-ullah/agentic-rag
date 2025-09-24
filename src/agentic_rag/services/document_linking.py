"""
Document Linking Service

This module provides comprehensive document linking functionality including
manual linking, automatic suggestions, confidence scoring, and bulk operations.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.orm import Session, joinedload

from agentic_rag.models.database import (
    Document, DocumentLink, DocumentKind, DocumentStatus, 
    LinkType, UserFeedback, User
)
from agentic_rag.services.vector_operations import get_vector_operations, VectorSearchOptions
from agentic_rag.services.embedding_pipeline import get_embedding_pipeline

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class DocumentLinkCreateRequest(BaseModel):
    """Enhanced request model for creating document links."""
    
    offer_id: UUID = Field(..., description="Offer document ID")
    offer_type: str = Field(..., description="Offer type (technical, commercial, pricing)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Link confidence score")
    link_type: LinkType = Field(default=LinkType.MANUAL, description="Type of link")
    notes: Optional[str] = Field(None, max_length=1000, description="User notes about the link")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('offer_type')
    def validate_offer_type(cls, v):
        if v not in ['technical', 'commercial', 'pricing']:
            raise ValueError('offer_type must be one of: technical, commercial, pricing')
        return v


class DocumentLinkUpdateRequest(BaseModel):
    """Request model for updating document links."""
    
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Updated confidence score")
    notes: Optional[str] = Field(None, max_length=1000, description="Updated notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    user_feedback: Optional[UserFeedback] = Field(None, description="User feedback")


class DocumentLinkValidationRequest(BaseModel):
    """Request model for validating document links."""
    
    validated: bool = Field(..., description="Whether the link is validated")
    notes: Optional[str] = Field(None, max_length=1000, description="Validation notes")


class DocumentLinkInfo(BaseModel):
    """Enhanced document link information."""
    
    id: UUID = Field(..., description="Link ID")
    rfq_id: UUID = Field(..., description="RFQ document ID")
    offer_id: UUID = Field(..., description="Offer document ID")
    offer_type: str = Field(..., description="Offer type")
    confidence: float = Field(..., description="Link confidence score")
    link_type: LinkType = Field(..., description="Type of link")
    quality_score: Optional[float] = Field(None, description="Quality assessment score")
    user_feedback: Optional[UserFeedback] = Field(None, description="User feedback")
    notes: Optional[str] = Field(None, description="User notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    # Timestamps and user info
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    created_by: Optional[UUID] = Field(None, description="Creator user ID")
    validated_by: Optional[UUID] = Field(None, description="Validator user ID")
    validated_at: Optional[datetime] = Field(None, description="Validation timestamp")
    feedback_at: Optional[datetime] = Field(None, description="Feedback timestamp")
    
    # Offer document details
    offer_title: str = Field(..., description="Offer document title")
    offer_kind: DocumentKind = Field(..., description="Offer document kind")
    offer_status: DocumentStatus = Field(..., description="Offer document status")
    
    class Config:
        from_attributes = True


class DocumentLinksResponse(BaseModel):
    """Response model for document links listing."""
    
    rfq_id: UUID = Field(..., description="RFQ document ID")
    rfq_title: str = Field(..., description="RFQ document title")
    links: List[DocumentLinkInfo] = Field(..., description="List of document links")
    total_count: int = Field(..., description="Total number of links")
    statistics: Dict[str, Any] = Field(..., description="Link statistics")


class LinkSuggestion(BaseModel):
    """Document link suggestion with confidence scoring."""
    
    offer_id: UUID = Field(..., description="Suggested offer document ID")
    offer_title: str = Field(..., description="Offer document title")
    offer_kind: DocumentKind = Field(..., description="Offer document kind")
    offer_type: str = Field(..., description="Suggested offer type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Suggestion confidence")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Content similarity score")
    metadata_score: float = Field(..., ge=0.0, le=1.0, description="Metadata alignment score")
    temporal_score: float = Field(..., ge=0.0, le=1.0, description="Temporal proximity score")
    reasons: List[str] = Field(..., description="Reasons for the suggestion")
    metadata: Dict[str, Any] = Field(..., description="Suggestion metadata")


class LinkSuggestionsResponse(BaseModel):
    """Response model for link suggestions."""
    
    rfq_id: UUID = Field(..., description="RFQ document ID")
    rfq_title: str = Field(..., description="RFQ document title")
    suggestions: List[LinkSuggestion] = Field(..., description="List of link suggestions")
    total_count: int = Field(..., description="Total number of suggestions")
    processing_time: float = Field(..., description="Processing time in seconds")


class BulkLinkOperation(BaseModel):
    """Bulk link operation request."""
    
    operation: str = Field(..., description="Operation type: create, update, delete, validate")
    links: List[Dict[str, Any]] = Field(..., description="List of link operations")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch processing size")
    validate_before_operation: bool = Field(default=True, description="Validate before processing")
    
    @validator('operation')
    def validate_operation(cls, v):
        if v not in ['create', 'update', 'delete', 'validate']:
            raise ValueError('operation must be one of: create, update, delete, validate')
        return v


class BulkLinkResult(BaseModel):
    """Result of bulk link operation."""
    
    operation: str = Field(..., description="Operation type")
    total_requested: int = Field(..., description="Total operations requested")
    successful: int = Field(..., description="Successful operations")
    failed: int = Field(..., description="Failed operations")
    errors: List[Dict[str, Any]] = Field(..., description="Error details")
    processing_time: float = Field(..., description="Total processing time")
    batch_results: List[Dict[str, Any]] = Field(..., description="Batch processing results")


# ============================================================================
# DOCUMENT LINKING SERVICE
# ============================================================================

class DocumentLinkingService:
    """Comprehensive document linking service with advanced features."""
    
    def __init__(self):
        self._vector_operations = None
        self._embedding_pipeline = None
        self._stats = {
            "links_created": 0,
            "links_updated": 0,
            "links_deleted": 0,
            "suggestions_generated": 0,
            "bulk_operations": 0
        }
        
        logger.info("Document linking service initialized")
    
    async def initialize(self):
        """Initialize the service dependencies."""
        if not self._vector_operations:
            self._vector_operations = await get_vector_operations()
        if not self._embedding_pipeline:
            self._embedding_pipeline = await get_embedding_pipeline()
        
        logger.info("Document linking service dependencies initialized")
    
    async def create_link(
        self,
        db_session: Session,
        rfq_id: UUID,
        request: DocumentLinkCreateRequest,
        user_id: UUID,
        tenant_id: UUID
    ) -> DocumentLinkInfo:
        """
        Create a new document link with enhanced validation and audit trail.
        
        Args:
            db_session: Database session
            rfq_id: RFQ document ID
            request: Link creation request
            user_id: User creating the link
            tenant_id: Tenant ID
            
        Returns:
            DocumentLinkInfo with created link details
        """
        await self.initialize()
        
        # Validate RFQ document
        rfq_document = self._validate_rfq_document(db_session, rfq_id, tenant_id)
        
        # Validate offer document
        offer_document = self._validate_offer_document(db_session, request.offer_id, tenant_id)
        
        # Check for existing link
        existing_link = db_session.query(DocumentLink).filter(
            and_(
                DocumentLink.tenant_id == tenant_id,
                DocumentLink.rfq_id == rfq_id,
                DocumentLink.offer_id == request.offer_id
            )
        ).first()
        
        if existing_link:
            raise ValueError(f"Link already exists between RFQ {rfq_id} and Offer {request.offer_id}")
        
        # Calculate quality score if not provided
        quality_score = await self._calculate_link_quality(
            rfq_document, offer_document, request.confidence
        )
        
        # Create the link
        document_link = DocumentLink(
            tenant_id=tenant_id,
            rfq_id=rfq_id,
            offer_id=request.offer_id,
            offer_type=request.offer_type,
            confidence=request.confidence,
            link_type=request.link_type,
            created_by=user_id,
            notes=request.notes,
            link_metadata=request.metadata or {},
            quality_score=quality_score
        )
        
        db_session.add(document_link)
        db_session.commit()
        db_session.refresh(document_link)
        
        # Update statistics
        self._stats["links_created"] += 1
        
        # Create response
        return self._create_link_info(document_link, offer_document)
    
    def _validate_rfq_document(self, db_session: Session, rfq_id: UUID, tenant_id: UUID) -> Document:
        """Validate RFQ document exists and is accessible."""
        rfq_document = db_session.query(Document).filter(
            and_(
                Document.id == rfq_id,
                Document.tenant_id == tenant_id,
                Document.deleted_at.is_(None),
                Document.kind == DocumentKind.RFQ
            )
        ).first()
        
        if not rfq_document:
            raise ValueError("RFQ document not found or not accessible")
        
        return rfq_document
    
    def _validate_offer_document(self, db_session: Session, offer_id: UUID, tenant_id: UUID) -> Document:
        """Validate offer document exists and is accessible."""
        offer_document = db_session.query(Document).filter(
            and_(
                Document.id == offer_id,
                Document.tenant_id == tenant_id,
                Document.deleted_at.is_(None),
                Document.kind.in_([DocumentKind.OFFER_TECH, DocumentKind.OFFER_COMM, DocumentKind.PRICING])
            )
        ).first()
        
        if not offer_document:
            raise ValueError("Offer document not found or not accessible")
        
        return offer_document
    
    async def _calculate_link_quality(
        self, 
        rfq_document: Document, 
        offer_document: Document, 
        confidence: float
    ) -> float:
        """Calculate quality score for a document link."""
        # Basic quality calculation based on confidence and document status
        quality_score = confidence * 0.7  # Base score from confidence
        
        # Boost for completed documents
        if rfq_document.status == DocumentStatus.READY:
            quality_score += 0.1
        if offer_document.status == DocumentStatus.READY:
            quality_score += 0.1
        
        # Boost for documents with chunks
        if rfq_document.chunk_count > 0:
            quality_score += 0.05
        if offer_document.chunk_count > 0:
            quality_score += 0.05
        
        return min(quality_score, 1.0)
    
    def _create_link_info(self, link: DocumentLink, offer_document: Document) -> DocumentLinkInfo:
        """Create DocumentLinkInfo from database objects."""
        return DocumentLinkInfo(
            id=link.id,
            rfq_id=link.rfq_id,
            offer_id=link.offer_id,
            offer_type=link.offer_type,
            confidence=link.confidence,
            link_type=link.link_type,
            quality_score=link.quality_score,
            user_feedback=link.user_feedback,
            notes=link.notes,
            metadata=link.link_metadata,
            created_at=link.created_at,
            updated_at=link.updated_at,
            created_by=link.created_by,
            validated_by=link.validated_by,
            validated_at=link.validated_at,
            feedback_at=link.feedback_at,
            offer_title=offer_document.title,
            offer_kind=offer_document.kind,
            offer_status=offer_document.status
        )
    
    async def update_link(
        self,
        db_session: Session,
        link_id: UUID,
        request: DocumentLinkUpdateRequest,
        user_id: UUID,
        tenant_id: UUID
    ) -> DocumentLinkInfo:
        """Update an existing document link."""
        # Find the link
        link = db_session.query(DocumentLink).filter(
            and_(
                DocumentLink.id == link_id,
                DocumentLink.tenant_id == tenant_id
            )
        ).first()

        if not link:
            raise ValueError("Document link not found")

        # Update fields
        if request.confidence is not None:
            link.confidence = request.confidence
        if request.notes is not None:
            link.notes = request.notes
        if request.metadata is not None:
            link.link_metadata = request.metadata
        if request.user_feedback is not None:
            link.user_feedback = request.user_feedback
            link.feedback_at = datetime.now(timezone.utc)

        # Recalculate quality score if confidence changed
        if request.confidence is not None:
            rfq_doc = db_session.query(Document).get(link.rfq_id)
            offer_doc = db_session.query(Document).get(link.offer_id)
            link.quality_score = await self._calculate_link_quality(
                rfq_doc, offer_doc, link.confidence
            )

        db_session.commit()
        db_session.refresh(link)

        # Update statistics
        self._stats["links_updated"] += 1

        # Get offer document for response
        offer_document = db_session.query(Document).get(link.offer_id)
        return self._create_link_info(link, offer_document)

    async def delete_link(
        self,
        db_session: Session,
        link_id: UUID,
        tenant_id: UUID
    ) -> bool:
        """Delete a document link."""
        link = db_session.query(DocumentLink).filter(
            and_(
                DocumentLink.id == link_id,
                DocumentLink.tenant_id == tenant_id
            )
        ).first()

        if not link:
            raise ValueError("Document link not found")

        db_session.delete(link)
        db_session.commit()

        # Update statistics
        self._stats["links_deleted"] += 1

        return True

    async def validate_link(
        self,
        db_session: Session,
        link_id: UUID,
        request: DocumentLinkValidationRequest,
        user_id: UUID,
        tenant_id: UUID
    ) -> DocumentLinkInfo:
        """Validate a document link."""
        link = db_session.query(DocumentLink).filter(
            and_(
                DocumentLink.id == link_id,
                DocumentLink.tenant_id == tenant_id
            )
        ).first()

        if not link:
            raise ValueError("Document link not found")

        if request.validated:
            link.validated_by = user_id
            link.validated_at = datetime.now(timezone.utc)
            if request.notes:
                link.notes = request.notes
        else:
            link.validated_by = None
            link.validated_at = None

        db_session.commit()
        db_session.refresh(link)

        # Get offer document for response
        offer_document = db_session.query(Document).get(link.offer_id)
        return self._create_link_info(link, offer_document)

    async def get_links_for_rfq(
        self,
        db_session: Session,
        rfq_id: UUID,
        tenant_id: UUID,
        include_suggestions: bool = False
    ) -> DocumentLinksResponse:
        """Get all links for an RFQ document."""
        # Validate RFQ document
        rfq_document = self._validate_rfq_document(db_session, rfq_id, tenant_id)

        # Get links with offer document details
        links_query = db_session.query(DocumentLink).options(
            joinedload(DocumentLink.offer)
        ).filter(
            and_(
                DocumentLink.tenant_id == tenant_id,
                DocumentLink.rfq_id == rfq_id
            )
        ).order_by(desc(DocumentLink.confidence))

        if not include_suggestions:
            links_query = links_query.filter(
                DocumentLink.link_type != LinkType.SUGGESTED
            )

        links = links_query.all()

        # Create link info objects
        link_infos = []
        for link in links:
            link_info = self._create_link_info(link, link.offer)
            link_infos.append(link_info)

        # Calculate statistics
        statistics = self._calculate_link_statistics(links)

        return DocumentLinksResponse(
            rfq_id=rfq_id,
            rfq_title=rfq_document.title,
            links=link_infos,
            total_count=len(link_infos),
            statistics=statistics
        )

    def _calculate_link_statistics(self, links: List[DocumentLink]) -> Dict[str, Any]:
        """Calculate statistics for a list of links."""
        if not links:
            return {
                "total_links": 0,
                "by_type": {},
                "by_offer_type": {},
                "average_confidence": 0.0,
                "validated_count": 0
            }

        by_type = {}
        by_offer_type = {}
        total_confidence = 0.0
        validated_count = 0

        for link in links:
            # Count by link type
            link_type_str = link.link_type.value if hasattr(link.link_type, 'value') else str(link.link_type)
            by_type[link_type_str] = by_type.get(link_type_str, 0) + 1

            # Count by offer type
            by_offer_type[link.offer_type] = by_offer_type.get(link.offer_type, 0) + 1

            # Sum confidence
            total_confidence += link.confidence

            # Count validated
            if link.validated_at:
                validated_count += 1

        return {
            "total_links": len(links),
            "by_type": by_type,
            "by_offer_type": by_offer_type,
            "average_confidence": total_confidence / len(links),
            "validated_count": validated_count,
            "validation_rate": validated_count / len(links) if links else 0.0
        }

    async def generate_link_suggestions(
        self,
        db_session: Session,
        rfq_id: UUID,
        tenant_id: UUID,
        max_suggestions: int = 10,
        min_confidence: float = 0.3
    ) -> LinkSuggestionsResponse:
        """
        Generate automatic linking suggestions for an RFQ document.

        Args:
            db_session: Database session
            rfq_id: RFQ document ID
            tenant_id: Tenant ID
            max_suggestions: Maximum number of suggestions to return
            min_confidence: Minimum confidence threshold

        Returns:
            LinkSuggestionsResponse with suggestions
        """
        await self.initialize()
        start_time = time.time()

        # Validate RFQ document
        rfq_document = self._validate_rfq_document(db_session, rfq_id, tenant_id)

        # Get available offer documents
        offer_documents = db_session.query(Document).filter(
            and_(
                Document.tenant_id == tenant_id,
                Document.deleted_at.is_(None),
                Document.status == DocumentStatus.READY,
                Document.kind.in_([DocumentKind.OFFER_TECH, DocumentKind.OFFER_COMM, DocumentKind.PRICING])
            )
        ).all()

        if not offer_documents:
            return LinkSuggestionsResponse(
                rfq_id=rfq_id,
                rfq_title=rfq_document.title,
                suggestions=[],
                total_count=0,
                processing_time=time.time() - start_time
            )

        # Get existing links to avoid duplicates
        existing_links = db_session.query(DocumentLink).filter(
            and_(
                DocumentLink.tenant_id == tenant_id,
                DocumentLink.rfq_id == rfq_id
            )
        ).all()

        existing_offer_ids = {link.offer_id for link in existing_links}

        # Filter out already linked offers
        candidate_offers = [doc for doc in offer_documents if doc.id not in existing_offer_ids]

        # Generate suggestions for each candidate
        suggestions = []
        for offer_doc in candidate_offers:
            suggestion = await self._generate_single_suggestion(
                rfq_document, offer_doc, db_session
            )

            if suggestion and suggestion.confidence >= min_confidence:
                suggestions.append(suggestion)

        # Sort by confidence and limit results
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        suggestions = suggestions[:max_suggestions]

        # Update statistics
        self._stats["suggestions_generated"] += len(suggestions)

        return LinkSuggestionsResponse(
            rfq_id=rfq_id,
            rfq_title=rfq_document.title,
            suggestions=suggestions,
            total_count=len(suggestions),
            processing_time=time.time() - start_time
        )

    async def _generate_single_suggestion(
        self,
        rfq_document: Document,
        offer_document: Document,
        db_session: Session
    ) -> Optional[LinkSuggestion]:
        """Generate a single link suggestion between RFQ and offer documents."""
        try:
            # Calculate content similarity using vector search
            similarity_score = await self._calculate_content_similarity(
                rfq_document, offer_document
            )

            # Calculate metadata alignment score
            metadata_score = self._calculate_metadata_alignment(
                rfq_document, offer_document
            )

            # Calculate temporal proximity score
            temporal_score = self._calculate_temporal_proximity(
                rfq_document, offer_document
            )

            # Determine offer type based on document kind
            offer_type = self._determine_offer_type(offer_document.kind)

            # Calculate overall confidence using weighted factors
            confidence = self._calculate_suggestion_confidence(
                similarity_score, metadata_score, temporal_score
            )

            # Generate reasons for the suggestion
            reasons = self._generate_suggestion_reasons(
                similarity_score, metadata_score, temporal_score, offer_document
            )

            # Create suggestion metadata
            suggestion_metadata = {
                "similarity_score": similarity_score,
                "metadata_score": metadata_score,
                "temporal_score": temporal_score,
                "rfq_chunk_count": rfq_document.chunk_count,
                "offer_chunk_count": offer_document.chunk_count,
                "algorithm_version": "1.0"
            }

            return LinkSuggestion(
                offer_id=offer_document.id,
                offer_title=offer_document.title,
                offer_kind=offer_document.kind,
                offer_type=offer_type,
                confidence=confidence,
                similarity_score=similarity_score,
                metadata_score=metadata_score,
                temporal_score=temporal_score,
                reasons=reasons,
                metadata=suggestion_metadata
            )

        except Exception as e:
            logger.error(f"Error generating suggestion for offer {offer_document.id}: {e}")
            return None

    async def _calculate_content_similarity(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate content similarity between RFQ and offer documents using vector search."""
        try:
            if not self._vector_operations or not self._embedding_pipeline:
                return 0.5  # Default similarity if vector operations not available

            # Get RFQ document content for embedding
            rfq_content = f"{rfq_document.title} {rfq_document.description or ''}"

            # Generate embedding for RFQ content
            embedding_request = {
                "texts": [rfq_content],
                "tenant_id": str(rfq_document.tenant_id),
                "batch_id": f"similarity_{uuid4()}"
            }

            embedding_result = await self._embedding_pipeline.process_embeddings(embedding_request)

            if not embedding_result.embeddings:
                return 0.5

            rfq_embedding = embedding_result.embeddings[0].embedding

            # Search for similar content in offer documents
            search_results = await self._vector_operations.search_vectors(
                query_embedding=rfq_embedding,
                document_kind=offer_document.kind.value,
                tenant_id=str(offer_document.tenant_id),
                options=VectorSearchOptions(
                    n_results=5,
                    where_filter={"document_id": str(offer_document.id)},
                    include_metadata=True,
                    include_documents=True
                )
            )

            if not search_results:
                return 0.3  # Low similarity if no vector matches

            # Calculate average similarity from top results
            similarities = [1.0 - result.distance for result in search_results[:3]]
            avg_similarity = sum(similarities) / len(similarities)

            return max(0.0, min(1.0, avg_similarity))

        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.5  # Default similarity on error

    def _calculate_metadata_alignment(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate metadata alignment score between documents."""
        score = 0.0
        factors = 0

        # Title similarity (basic keyword matching)
        if rfq_document.title and offer_document.title:
            rfq_words = set(rfq_document.title.lower().split())
            offer_words = set(offer_document.title.lower().split())

            if rfq_words and offer_words:
                title_similarity = len(rfq_words & offer_words) / len(rfq_words | offer_words)
                score += title_similarity * 0.4
                factors += 0.4

        # File size similarity (documents of similar size might be related)
        if rfq_document.file_size and offer_document.file_size:
            size_ratio = min(rfq_document.file_size, offer_document.file_size) / max(rfq_document.file_size, offer_document.file_size)
            score += size_ratio * 0.2
            factors += 0.2

        # Chunk count similarity
        if rfq_document.chunk_count and offer_document.chunk_count:
            chunk_ratio = min(rfq_document.chunk_count, offer_document.chunk_count) / max(rfq_document.chunk_count, offer_document.chunk_count)
            score += chunk_ratio * 0.2
            factors += 0.2

        # Processing status alignment
        if rfq_document.status == offer_document.status == DocumentStatus.READY:
            score += 0.2
            factors += 0.2

        return score / factors if factors > 0 else 0.5

    def _calculate_temporal_proximity(
        self,
        rfq_document: Document,
        offer_document: Document
    ) -> float:
        """Calculate temporal proximity score based on document creation times."""
        if not rfq_document.created_at or not offer_document.created_at:
            return 0.5

        # Calculate time difference in days
        time_diff = abs((rfq_document.created_at - offer_document.created_at).days)

        # Score decreases with time difference
        if time_diff <= 1:
            return 1.0
        elif time_diff <= 7:
            return 0.8
        elif time_diff <= 30:
            return 0.6
        elif time_diff <= 90:
            return 0.4
        else:
            return 0.2

    def _determine_offer_type(self, document_kind: DocumentKind) -> str:
        """Determine offer type from document kind."""
        if document_kind == DocumentKind.OFFER_TECH:
            return "technical"
        elif document_kind == DocumentKind.OFFER_COMM:
            return "commercial"
        elif document_kind == DocumentKind.PRICING:
            return "pricing"
        else:
            return "technical"  # Default

    def _calculate_suggestion_confidence(
        self,
        similarity_score: float,
        metadata_score: float,
        temporal_score: float
    ) -> float:
        """Calculate overall confidence score using weighted factors."""
        # Weights based on story requirements
        content_weight = 0.40
        metadata_weight = 0.30
        temporal_weight = 0.15
        base_weight = 0.15

        confidence = (
            similarity_score * content_weight +
            metadata_score * metadata_weight +
            temporal_score * temporal_weight +
            0.5 * base_weight  # Base confidence
        )

        return max(0.0, min(1.0, confidence))

    def _generate_suggestion_reasons(
        self,
        similarity_score: float,
        metadata_score: float,
        temporal_score: float,
        offer_document: Document
    ) -> List[str]:
        """Generate human-readable reasons for the suggestion."""
        reasons = []

        if similarity_score > 0.7:
            reasons.append("High content similarity detected")
        elif similarity_score > 0.5:
            reasons.append("Moderate content similarity found")

        if metadata_score > 0.7:
            reasons.append("Strong metadata alignment")
        elif metadata_score > 0.5:
            reasons.append("Good metadata match")

        if temporal_score > 0.8:
            reasons.append("Documents created close in time")
        elif temporal_score > 0.6:
            reasons.append("Recent document pairing")

        if offer_document.chunk_count > 10:
            reasons.append("Comprehensive offer document")

        if offer_document.status == DocumentStatus.READY:
            reasons.append("Offer document fully processed")

        if not reasons:
            reasons.append("Basic compatibility detected")

        return reasons

    async def bulk_create_links(
        self,
        db_session: Session,
        operation: BulkLinkOperation,
        user_id: UUID,
        tenant_id: UUID
    ) -> BulkLinkResult:
        """
        Perform bulk link creation operations.

        Args:
            db_session: Database session
            operation: Bulk operation request
            user_id: User performing the operation
            tenant_id: Tenant ID

        Returns:
            BulkLinkResult with operation results
        """
        await self.initialize()
        start_time = time.time()

        if operation.operation != "create":
            raise ValueError("This method only supports create operations")

        total_requested = len(operation.links)
        successful = 0
        failed = 0
        errors = []
        batch_results = []

        # Process in batches
        for i in range(0, total_requested, operation.batch_size):
            batch = operation.links[i:i + operation.batch_size]
            batch_start_time = time.time()
            batch_successful = 0
            batch_failed = 0
            batch_errors = []

            for link_data in batch:
                try:
                    # Validate required fields
                    if not all(key in link_data for key in ['rfq_id', 'offer_id', 'offer_type', 'confidence']):
                        raise ValueError("Missing required fields in link data")

                    # Create link request
                    link_request = DocumentLinkCreateRequest(
                        offer_id=UUID(link_data['offer_id']),
                        offer_type=link_data['offer_type'],
                        confidence=float(link_data['confidence']),
                        link_type=LinkType(link_data.get('link_type', 'manual')),
                        notes=link_data.get('notes'),
                        metadata=link_data.get('metadata')
                    )

                    # Validate before operation if requested
                    if operation.validate_before_operation:
                        await self._validate_bulk_link_data(
                            db_session, UUID(link_data['rfq_id']), link_request, tenant_id
                        )

                    # Create the link
                    await self.create_link(
                        db_session=db_session,
                        rfq_id=UUID(link_data['rfq_id']),
                        request=link_request,
                        user_id=user_id,
                        tenant_id=tenant_id
                    )

                    batch_successful += 1
                    successful += 1

                except Exception as e:
                    batch_failed += 1
                    failed += 1
                    error_detail = {
                        "link_data": link_data,
                        "error": str(e),
                        "batch_index": len(batch_results),
                        "item_index": len(batch_errors)
                    }
                    batch_errors.append(error_detail)
                    errors.append(error_detail)

            # Record batch result
            batch_result = {
                "batch_index": len(batch_results),
                "batch_size": len(batch),
                "successful": batch_successful,
                "failed": batch_failed,
                "processing_time": time.time() - batch_start_time,
                "errors": batch_errors
            }
            batch_results.append(batch_result)

            # Commit batch if any successful operations
            if batch_successful > 0:
                try:
                    db_session.commit()
                except Exception as e:
                    db_session.rollback()
                    logger.error(f"Failed to commit batch {len(batch_results)}: {e}")

        # Update statistics
        self._stats["bulk_operations"] += 1
        self._stats["links_created"] += successful

        return BulkLinkResult(
            operation=operation.operation,
            total_requested=total_requested,
            successful=successful,
            failed=failed,
            errors=errors,
            processing_time=time.time() - start_time,
            batch_results=batch_results
        )

    async def bulk_update_links(
        self,
        db_session: Session,
        operation: BulkLinkOperation,
        user_id: UUID,
        tenant_id: UUID
    ) -> BulkLinkResult:
        """Perform bulk link update operations."""
        await self.initialize()
        start_time = time.time()

        if operation.operation != "update":
            raise ValueError("This method only supports update operations")

        total_requested = len(operation.links)
        successful = 0
        failed = 0
        errors = []
        batch_results = []

        # Process in batches
        for i in range(0, total_requested, operation.batch_size):
            batch = operation.links[i:i + operation.batch_size]
            batch_start_time = time.time()
            batch_successful = 0
            batch_failed = 0
            batch_errors = []

            for link_data in batch:
                try:
                    # Validate required fields
                    if 'link_id' not in link_data:
                        raise ValueError("Missing link_id in update data")

                    # Create update request
                    update_request = DocumentLinkUpdateRequest(
                        confidence=link_data.get('confidence'),
                        notes=link_data.get('notes'),
                        metadata=link_data.get('metadata'),
                        user_feedback=UserFeedback(link_data['user_feedback']) if 'user_feedback' in link_data else None
                    )

                    # Update the link
                    await self.update_link(
                        db_session=db_session,
                        link_id=UUID(link_data['link_id']),
                        request=update_request,
                        user_id=user_id,
                        tenant_id=tenant_id
                    )

                    batch_successful += 1
                    successful += 1

                except Exception as e:
                    batch_failed += 1
                    failed += 1
                    error_detail = {
                        "link_data": link_data,
                        "error": str(e),
                        "batch_index": len(batch_results),
                        "item_index": len(batch_errors)
                    }
                    batch_errors.append(error_detail)
                    errors.append(error_detail)

            # Record batch result
            batch_result = {
                "batch_index": len(batch_results),
                "batch_size": len(batch),
                "successful": batch_successful,
                "failed": batch_failed,
                "processing_time": time.time() - batch_start_time,
                "errors": batch_errors
            }
            batch_results.append(batch_result)

            # Commit batch if any successful operations
            if batch_successful > 0:
                try:
                    db_session.commit()
                except Exception as e:
                    db_session.rollback()
                    logger.error(f"Failed to commit batch {len(batch_results)}: {e}")

        # Update statistics
        self._stats["bulk_operations"] += 1
        self._stats["links_updated"] += successful

        return BulkLinkResult(
            operation=operation.operation,
            total_requested=total_requested,
            successful=successful,
            failed=failed,
            errors=errors,
            processing_time=time.time() - start_time,
            batch_results=batch_results
        )

    async def bulk_delete_links(
        self,
        db_session: Session,
        operation: BulkLinkOperation,
        tenant_id: UUID
    ) -> BulkLinkResult:
        """Perform bulk link deletion operations."""
        await self.initialize()
        start_time = time.time()

        if operation.operation != "delete":
            raise ValueError("This method only supports delete operations")

        total_requested = len(operation.links)
        successful = 0
        failed = 0
        errors = []
        batch_results = []

        # Process in batches
        for i in range(0, total_requested, operation.batch_size):
            batch = operation.links[i:i + operation.batch_size]
            batch_start_time = time.time()
            batch_successful = 0
            batch_failed = 0
            batch_errors = []

            for link_data in batch:
                try:
                    # Validate required fields
                    if 'link_id' not in link_data:
                        raise ValueError("Missing link_id in delete data")

                    # Delete the link
                    await self.delete_link(
                        db_session=db_session,
                        link_id=UUID(link_data['link_id']),
                        tenant_id=tenant_id
                    )

                    batch_successful += 1
                    successful += 1

                except Exception as e:
                    batch_failed += 1
                    failed += 1
                    error_detail = {
                        "link_data": link_data,
                        "error": str(e),
                        "batch_index": len(batch_results),
                        "item_index": len(batch_errors)
                    }
                    batch_errors.append(error_detail)
                    errors.append(error_detail)

            # Record batch result
            batch_result = {
                "batch_index": len(batch_results),
                "batch_size": len(batch),
                "successful": batch_successful,
                "failed": batch_failed,
                "processing_time": time.time() - batch_start_time,
                "errors": batch_errors
            }
            batch_results.append(batch_result)

            # Commit batch if any successful operations
            if batch_successful > 0:
                try:
                    db_session.commit()
                except Exception as e:
                    db_session.rollback()
                    logger.error(f"Failed to commit batch {len(batch_results)}: {e}")

        # Update statistics
        self._stats["bulk_operations"] += 1
        self._stats["links_deleted"] += successful

        return BulkLinkResult(
            operation=operation.operation,
            total_requested=total_requested,
            successful=successful,
            failed=failed,
            errors=errors,
            processing_time=time.time() - start_time,
            batch_results=batch_results
        )

    async def _validate_bulk_link_data(
        self,
        db_session: Session,
        rfq_id: UUID,
        request: DocumentLinkCreateRequest,
        tenant_id: UUID
    ):
        """Validate bulk link data before processing."""
        # Check if RFQ exists
        rfq_document = self._validate_rfq_document(db_session, rfq_id, tenant_id)

        # Check if offer exists
        offer_document = self._validate_offer_document(db_session, request.offer_id, tenant_id)

        # Check for existing link
        existing_link = db_session.query(DocumentLink).filter(
            and_(
                DocumentLink.tenant_id == tenant_id,
                DocumentLink.rfq_id == rfq_id,
                DocumentLink.offer_id == request.offer_id
            )
        ).first()

        if existing_link:
            raise ValueError(f"Link already exists between RFQ {rfq_id} and Offer {request.offer_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self._stats.copy()


# Singleton instance
_document_linking_service: Optional[DocumentLinkingService] = None


def get_document_linking_service() -> DocumentLinkingService:
    """Get or create document linking service instance."""
    global _document_linking_service

    if _document_linking_service is None:
        _document_linking_service = DocumentLinkingService()

    return _document_linking_service
