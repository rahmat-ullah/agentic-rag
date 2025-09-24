"""
Document Management API routes.

This module defines FastAPI routes for managing uploaded documents,
including listing, viewing, and deleting documents.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.orm import Session

from agentic_rag.api.dependencies.auth import get_current_user, require_permission
from agentic_rag.api.dependencies.database import get_db_session
from agentic_rag.api.models.responses import ErrorResponse, SuccessResponse
from agentic_rag.config import get_settings
from agentic_rag.models.database import Document, DocumentKind, DocumentStatus, DocumentLink, ChunkMeta, User
from agentic_rag.services.storage import get_storage_service
import hashlib
import hmac
import time
from urllib.parse import urlencode

router = APIRouter(prefix="/documents", tags=["Documents"])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_secure_download_url(document_id: UUID, source_uri: str, user_id: UUID) -> str:
    """
    Generate a secure, time-limited download URL for a document.

    Args:
        document_id: Document ID
        source_uri: Original source URI
        user_id: User ID for access control

    Returns:
        Secure download URL with signature and expiration
    """
    settings = get_settings()

    # Create expiration timestamp (1 hour from now)
    expires = int(time.time()) + 3600

    # Create signature payload
    payload = f"{document_id}:{source_uri}:{user_id}:{expires}"

    # Generate HMAC signature
    secret_key = settings.SECRET_KEY.encode('utf-8')
    signature = hmac.new(
        secret_key,
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # Build secure download URL
    params = {
        'document_id': str(document_id),
        'expires': expires,
        'signature': signature,
        'user_id': str(user_id)
    }

    base_url = f"{settings.API_BASE_URL}/documents/{document_id}/download"
    return f"{base_url}?{urlencode(params)}"


# ============================================================================
# DOCUMENT MODELS
# ============================================================================

from pydantic import BaseModel, Field


class DocumentListItem(BaseModel):
    """Document list item for document listing."""

    id: UUID = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    kind: DocumentKind = Field(..., description="Document kind/type")
    status: DocumentStatus = Field(..., description="Document processing status")
    sha256: str = Field(..., description="SHA256 hash")
    version: int = Field(..., description="Document version")
    pages: Optional[int] = Field(None, description="Number of pages")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    chunk_count: int = Field(0, description="Number of chunks created")
    processing_progress: float = Field(0.0, description="Processing progress (0.0-1.0)")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    created_by: Optional[UUID] = Field(None, description="Creator user ID")


class ChunkStatistics(BaseModel):
    """Chunk statistics for a document."""

    total_chunks: int = Field(0, description="Total number of chunks")
    total_tokens: int = Field(0, description="Total token count across all chunks")
    average_tokens: float = Field(0.0, description="Average tokens per chunk")
    table_chunks: int = Field(0, description="Number of table chunks")
    retired_chunks: int = Field(0, description="Number of retired chunks")


class DocumentDetail(BaseModel):
    """Detailed document information."""

    id: UUID = Field(..., description="Document ID")
    tenant_id: UUID = Field(..., description="Tenant ID")
    title: str = Field(..., description="Document title")
    kind: DocumentKind = Field(..., description="Document kind/type")
    status: DocumentStatus = Field(..., description="Document processing status")
    source_uri: Optional[str] = Field(None, description="Source file URI")
    sha256: str = Field(..., description="SHA256 hash")
    version: int = Field(..., description="Document version")
    pages: Optional[int] = Field(None, description="Number of pages")

    # Processing and metadata
    processing_progress: float = Field(0.0, description="Processing progress (0.0-1.0)")
    processing_error: Optional[str] = Field(None, description="Processing error message")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    chunk_count: int = Field(0, description="Number of chunks created")

    # Chunk statistics
    chunk_statistics: Optional[ChunkStatistics] = Field(None, description="Detailed chunk statistics")

    # Timestamps and user tracking
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    created_by: Optional[UUID] = Field(None, description="Creator user ID")
    deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")

    # Additional metadata
    download_url: Optional[str] = Field(None, description="Download URL")


class DocumentListResponse(BaseModel):
    """Document list response with pagination."""
    
    documents: List[DocumentListItem] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    total_pages: int = Field(..., description="Total number of pages")


# ============================================================================
# DOCUMENT ENDPOINTS
# ============================================================================

@router.get(
    "/",
    response_model=SuccessResponse[DocumentListResponse],
    summary="List documents",
    description="List documents with filtering, pagination, and sorting"
)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    kind: Optional[DocumentKind] = Query(None, description="Filter by document kind"),
    status: Optional[DocumentStatus] = Query(None, description="Filter by document status"),
    search: Optional[str] = Query(None, description="Search in document titles"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    include_deleted: bool = Query(False, description="Include soft-deleted documents"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """List documents with filtering and pagination."""
    try:
        # Build base query with tenant isolation
        query = db_session.query(Document).filter(
            Document.tenant_id == current_user.tenant_id
        )

        # Exclude soft-deleted documents by default
        if not include_deleted:
            query = query.filter(Document.deleted_at.is_(None))

        # Apply filters
        if kind:
            query = query.filter(Document.kind == kind)

        if status:
            query = query.filter(Document.status == status)

        if search:
            query = query.filter(
                Document.title.ilike(f"%{search}%")
            )

        if date_from:
            query = query.filter(Document.created_at >= date_from)

        if date_to:
            query = query.filter(Document.created_at <= date_to)
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply sorting
        sort_column = getattr(Document, sort_by, Document.created_at)
        if sort_order == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(sort_column)
        
        # Apply pagination
        offset = (page - 1) * page_size
        documents = query.offset(offset).limit(page_size).all()
        
        # Convert to response format
        document_items = [
            DocumentListItem(
                id=doc.id,
                title=doc.title,
                kind=doc.kind,
                status=doc.status,
                sha256=doc.sha256,
                version=doc.version,
                pages=doc.pages,
                file_size=doc.file_size,
                chunk_count=doc.chunk_count or 0,
                processing_progress=doc.processing_progress or 0.0,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                created_by=doc.created_by
            )
            for doc in documents
        ]
        
        total_pages = (total_count + page_size - 1) // page_size
        
        response_data = DocumentListResponse(
            documents=document_items,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
        return SuccessResponse(
            data=response_data,
            message=f"Retrieved {len(document_items)} documents"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get(
    "/{document_id}",
    response_model=SuccessResponse[DocumentDetail],
    summary="Get document details",
    description="Get detailed information about a specific document"
)
async def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Get detailed document information."""
    try:
        # Get document with tenant isolation (exclude soft-deleted)
        document = db_session.query(Document).filter(
            and_(
                Document.id == document_id,
                Document.tenant_id == current_user.tenant_id,
                Document.deleted_at.is_(None)
            )
        ).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Calculate chunk statistics
        chunk_stats_query = db_session.query(
            func.count(ChunkMeta.id).label('total_chunks'),
            func.coalesce(func.sum(ChunkMeta.token_count), 0).label('total_tokens'),
            func.coalesce(func.avg(ChunkMeta.token_count), 0.0).label('average_tokens'),
            func.count(func.case([(ChunkMeta.is_table == True, 1)])).label('table_chunks'),
            func.count(func.case([(ChunkMeta.retired == True, 1)])).label('retired_chunks')
        ).filter(
            and_(
                ChunkMeta.document_id == document_id,
                ChunkMeta.tenant_id == current_user.tenant_id
            )
        ).first()

        chunk_statistics = None
        if chunk_stats_query:
            chunk_statistics = ChunkStatistics(
                total_chunks=chunk_stats_query.total_chunks or 0,
                total_tokens=chunk_stats_query.total_tokens or 0,
                average_tokens=float(chunk_stats_query.average_tokens or 0.0),
                table_chunks=chunk_stats_query.table_chunks or 0,
                retired_chunks=chunk_stats_query.retired_chunks or 0
            )

        # Generate secure download URL if source_uri exists
        download_url = None
        if document.source_uri:
            download_url = generate_secure_download_url(
                document_id=document.id,
                source_uri=document.source_uri,
                user_id=current_user.id
            )
        
        document_detail = DocumentDetail(
            id=document.id,
            tenant_id=document.tenant_id,
            title=document.title,
            kind=document.kind,
            status=document.status,
            source_uri=document.source_uri,
            sha256=document.sha256,
            version=document.version,
            pages=document.pages,
            processing_progress=document.processing_progress or 0.0,
            processing_error=document.processing_error,
            file_size=document.file_size,
            chunk_count=document.chunk_count or 0,
            chunk_statistics=chunk_statistics,
            created_at=document.created_at,
            updated_at=document.updated_at,
            created_by=document.created_by,
            deleted_at=document.deleted_at,
            download_url=download_url
        )
        
        return SuccessResponse(
            data=document_detail,
            message="Document details retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.get(
    "/{document_id}/download",
    summary="Download document",
    description="Download document file with secure access control"
)
async def download_document(
    document_id: UUID,
    expires: int = Query(..., description="Expiration timestamp"),
    signature: str = Query(..., description="Security signature"),
    user_id: UUID = Query(..., description="User ID"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Download document file with secure access control."""
    try:
        # Verify the user matches the signed URL
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: User mismatch"
            )

        # Check if URL has expired
        current_time = int(time.time())
        if current_time > expires:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Download URL has expired"
            )

        # Get document with tenant isolation
        document = db_session.query(Document).filter(
            and_(
                Document.id == document_id,
                Document.tenant_id == current_user.tenant_id,
                Document.deleted_at.is_(None)
            )
        ).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Verify signature
        settings = get_settings()
        payload = f"{document_id}:{document.source_uri}:{user_id}:{expires}"
        secret_key = settings.SECRET_KEY.encode('utf-8')
        expected_signature = hmac.new(
            secret_key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid signature"
            )

        # TODO: Implement actual file download from storage service
        # For now, redirect to the source URI or return file info
        from fastapi.responses import RedirectResponse

        if document.source_uri:
            return RedirectResponse(url=document.source_uri)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download document: {str(e)}"
        )


@router.delete(
    "/{document_id}",
    response_model=SuccessResponse[dict],
    summary="Delete document",
    description="Delete a document and clean up associated data"
)
async def delete_document(
    document_id: UUID,
    hard_delete: bool = Query(False, description="Perform hard delete with complete cleanup"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Delete a document and clean up associated data."""
    try:
        from agentic_rag.services.document_cleanup import get_document_cleanup_service

        cleanup_service = get_document_cleanup_service()

        if hard_delete:
            # Perform hard delete with complete cleanup
            result = await cleanup_service.hard_delete_document(
                document_id=document_id,
                tenant_id=current_user.tenant_id,
                db_session=db_session,
                remove_file=True
            )
        else:
            # Perform soft delete
            result = await cleanup_service.soft_delete_document(
                document_id=document_id,
                tenant_id=current_user.tenant_id,
                db_session=db_session,
                user_id=current_user.id
            )

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error or "Document not found"
            )

        response_data = {
            "document_id": document_id,
            "deletion_type": "hard" if hard_delete else "soft",
            "cleanup_time": result.cleanup_time,
            "chunks_removed": result.chunks_removed,
            "links_removed": result.links_removed,
            "file_removed": result.file_removed
        }

        if hard_delete:
            message = f"Document permanently deleted with complete cleanup"
        else:
            message = f"Document soft deleted (will be permanently removed after retention period)"

        return SuccessResponse(
            data=response_data,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


# ============================================================================
# DOCUMENT LINKING MODELS
# ============================================================================

class DocumentLinkCreate(BaseModel):
    """Request model for creating document links."""

    offer_id: UUID = Field(..., description="Offer document ID")
    offer_type: str = Field(..., description="Offer type (technical, commercial, pricing)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Link confidence score")


class DocumentLinkInfo(BaseModel):
    """Document link information."""

    id: UUID = Field(..., description="Link ID")
    rfq_id: UUID = Field(..., description="RFQ document ID")
    offer_id: UUID = Field(..., description="Offer document ID")
    offer_type: str = Field(..., description="Offer type")
    confidence: float = Field(..., description="Link confidence score")
    created_at: datetime = Field(..., description="Creation timestamp")

    # Offer document details
    offer_title: str = Field(..., description="Offer document title")
    offer_kind: DocumentKind = Field(..., description="Offer document kind")
    offer_status: DocumentStatus = Field(..., description="Offer document status")


class DocumentLinksResponse(BaseModel):
    """Response model for document links listing."""

    rfq_id: UUID = Field(..., description="RFQ document ID")
    rfq_title: str = Field(..., description="RFQ document title")
    links: List[DocumentLinkInfo] = Field(..., description="List of document links")
    total_count: int = Field(..., description="Total number of links")


# ============================================================================
# DOCUMENT LINKING ENDPOINTS
# ============================================================================

@router.post(
    "/{rfq_id}/links",
    response_model=SuccessResponse[DocumentLinkInfo],
    summary="Create document link",
    description="Create a link between RFQ and Offer documents"
)
async def create_document_link(
    rfq_id: UUID,
    link_data: DocumentLinkCreate,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Create a link between RFQ and Offer documents."""
    try:
        # Validate RFQ document exists and is accessible
        rfq_document = db_session.query(Document).filter(
            and_(
                Document.id == rfq_id,
                Document.tenant_id == current_user.tenant_id,
                Document.deleted_at.is_(None),
                Document.kind == DocumentKind.RFQ
            )
        ).first()

        if not rfq_document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="RFQ document not found"
            )

        # Validate Offer document exists and is accessible
        offer_document = db_session.query(Document).filter(
            and_(
                Document.id == link_data.offer_id,
                Document.tenant_id == current_user.tenant_id,
                Document.deleted_at.is_(None)
            )
        ).first()

        if not offer_document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Offer document not found"
            )

        # Validate offer type
        valid_offer_types = ["technical", "commercial", "pricing"]
        if link_data.offer_type not in valid_offer_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid offer type. Must be one of: {', '.join(valid_offer_types)}"
            )

        # Check if link already exists
        existing_link = db_session.query(DocumentLink).filter(
            and_(
                DocumentLink.rfq_id == rfq_id,
                DocumentLink.offer_id == link_data.offer_id,
                DocumentLink.offer_type == link_data.offer_type,
                DocumentLink.tenant_id == current_user.tenant_id
            )
        ).first()

        if existing_link:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document link already exists"
            )

        # Create new document link
        document_link = DocumentLink(
            tenant_id=current_user.tenant_id,
            rfq_id=rfq_id,
            offer_id=link_data.offer_id,
            offer_type=link_data.offer_type,
            confidence=link_data.confidence
        )

        db_session.add(document_link)
        db_session.commit()
        db_session.refresh(document_link)

        # Create response
        link_info = DocumentLinkInfo(
            id=document_link.id,
            rfq_id=document_link.rfq_id,
            offer_id=document_link.offer_id,
            offer_type=document_link.offer_type,
            confidence=document_link.confidence,
            created_at=document_link.created_at,
            offer_title=offer_document.title,
            offer_kind=offer_document.kind,
            offer_status=offer_document.status
        )

        return SuccessResponse(
            data=link_info,
            message="Document link created successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document link: {str(e)}"
        )


@router.get(
    "/{rfq_id}/links",
    response_model=SuccessResponse[DocumentLinksResponse],
    summary="List document links",
    description="List all links for an RFQ document"
)
async def list_document_links(
    rfq_id: UUID,
    offer_type: Optional[str] = Query(None, description="Filter by offer type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence score"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """List all links for an RFQ document."""
    try:
        # Validate RFQ document exists and is accessible
        rfq_document = db_session.query(Document).filter(
            and_(
                Document.id == rfq_id,
                Document.tenant_id == current_user.tenant_id,
                Document.deleted_at.is_(None),
                Document.kind == DocumentKind.RFQ
            )
        ).first()

        if not rfq_document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="RFQ document not found"
            )

        # Build query for document links
        query = db_session.query(DocumentLink, Document).join(
            Document, DocumentLink.offer_id == Document.id
        ).filter(
            and_(
                DocumentLink.rfq_id == rfq_id,
                DocumentLink.tenant_id == current_user.tenant_id,
                DocumentLink.confidence >= min_confidence,
                Document.deleted_at.is_(None)  # Exclude links to deleted offers
            )
        )

        # Apply offer type filter
        if offer_type:
            query = query.filter(DocumentLink.offer_type == offer_type)

        # Order by confidence descending
        query = query.order_by(desc(DocumentLink.confidence))

        # Execute query
        results = query.all()

        # Convert to response format
        links = [
            DocumentLinkInfo(
                id=link.id,
                rfq_id=link.rfq_id,
                offer_id=link.offer_id,
                offer_type=link.offer_type,
                confidence=link.confidence,
                created_at=link.created_at,
                offer_title=offer_doc.title,
                offer_kind=offer_doc.kind,
                offer_status=offer_doc.status
            )
            for link, offer_doc in results
        ]

        response_data = DocumentLinksResponse(
            rfq_id=rfq_id,
            rfq_title=rfq_document.title,
            links=links,
            total_count=len(links)
        )

        return SuccessResponse(
            data=response_data,
            message=f"Retrieved {len(links)} document links"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list document links: {str(e)}"
        )


class DocumentSuggestionInfo(BaseModel):
    """Document link suggestion information."""

    offer_document_id: UUID = Field(..., description="Suggested offer document ID")
    offer_title: str = Field(..., description="Offer document title")
    confidence_score: float = Field(..., description="Confidence score (0.0-1.0)")
    similarity_factors: dict = Field(..., description="Factors contributing to similarity")
    created_at: datetime = Field(..., description="Offer document creation date")


class DocumentSuggestionsResponse(BaseModel):
    """Response model for document link suggestions."""

    rfq_document_id: UUID = Field(..., description="RFQ document ID")
    suggestions: List[DocumentSuggestionInfo] = Field(..., description="Link suggestions")
    total_suggestions: int = Field(..., description="Total number of suggestions")


@router.get(
    "/{rfq_id}/suggestions",
    response_model=SuccessResponse[DocumentSuggestionsResponse],
    summary="Get document link suggestions",
    description="Get automatic link suggestions for an RFQ document"
)
async def get_document_link_suggestions(
    rfq_id: UUID,
    max_suggestions: int = Query(10, ge=1, le=50, description="Maximum number of suggestions"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Get automatic link suggestions for an RFQ document."""
    try:
        # Verify RFQ document exists and is accessible
        rfq_document = db_session.query(Document).filter(
            and_(
                Document.id == rfq_id,
                Document.tenant_id == current_user.tenant_id,
                Document.deleted_at.is_(None)
            )
        ).first()

        if not rfq_document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="RFQ document not found"
            )

        # Verify it's an RFQ document
        if rfq_document.kind != DocumentKind.RFQ:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document must be of type RFQ to get link suggestions"
            )

        # Get suggestions using similarity service
        from agentic_rag.services.document_similarity import get_document_similarity_service

        similarity_service = get_document_similarity_service()
        suggestions = await similarity_service.suggest_document_links(
            rfq_document_id=rfq_id,
            tenant_id=current_user.tenant_id,
            db_session=db_session,
            max_suggestions=max_suggestions
        )

        # Convert to response format
        suggestion_infos = [
            DocumentSuggestionInfo(
                offer_document_id=suggestion.offer_document_id,
                offer_title=suggestion.offer_title,
                confidence_score=suggestion.confidence_score,
                similarity_factors=suggestion.similarity_factors,
                created_at=suggestion.created_at
            )
            for suggestion in suggestions
        ]

        response_data = DocumentSuggestionsResponse(
            rfq_document_id=rfq_id,
            suggestions=suggestion_infos,
            total_suggestions=len(suggestion_infos)
        )

        return SuccessResponse(
            data=response_data,
            message=f"Found {len(suggestion_infos)} link suggestions"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document link suggestions: {str(e)}"
        )


# ============================================================================
# BULK OPERATIONS
# ============================================================================

class BulkDeleteRequest(BaseModel):
    """Request model for bulk document deletion."""

    document_ids: List[UUID] = Field(..., description="List of document IDs to delete")


class BulkDeleteResponse(BaseModel):
    """Response model for bulk document deletion."""

    deleted_count: int = Field(..., description="Number of documents deleted")
    failed_count: int = Field(..., description="Number of documents that failed to delete")
    deleted_documents: List[UUID] = Field(..., description="List of successfully deleted document IDs")
    failed_documents: List[dict] = Field(..., description="List of failed deletions with reasons")


@router.delete(
    "/bulk",
    response_model=SuccessResponse[BulkDeleteResponse],
    summary="Bulk delete documents",
    description="Delete multiple documents in a single operation"
)
async def bulk_delete_documents(
    request: BulkDeleteRequest,
    hard_delete: bool = Query(False, description="Perform hard delete with complete cleanup"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Delete multiple documents in a single operation."""
    try:
        from agentic_rag.services.document_cleanup import get_document_cleanup_service

        cleanup_service = get_document_cleanup_service()

        # Perform bulk deletion using cleanup service
        results = await cleanup_service.bulk_delete_documents(
            document_ids=request.document_ids,
            tenant_id=current_user.tenant_id,
            db_session=db_session,
            hard_delete=hard_delete,
            user_id=current_user.id
        )

        # Process results
        deleted_documents = []
        failed_documents = []

        for result in results:
            if result.success:
                deleted_documents.append(result.document_id)
            else:
                failed_documents.append({
                    "document_id": str(result.document_id),
                    "reason": result.error or "Unknown error"
                })

        response_data = BulkDeleteResponse(
            deleted_count=len(deleted_documents),
            failed_count=len(failed_documents),
            deleted_documents=deleted_documents,
            failed_documents=failed_documents
        )

        deletion_type = "hard" if hard_delete else "soft"
        message = f"Bulk {deletion_type} delete completed: {len(deleted_documents)} deleted, {len(failed_documents)} failed"

        return SuccessResponse(
            data=response_data,
            message=message
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to bulk delete documents: {str(e)}"
        )


# ============================================================================
# DOCUMENT PROCESSING ENDPOINTS
# ============================================================================

class ProcessingRequest(BaseModel):
    """Request model for document processing."""

    processing_options: Optional[dict] = Field(None, description="Optional processing configuration")


class ProcessingResponse(BaseModel):
    """Response model for document processing."""

    document_id: UUID = Field(..., description="Document ID")
    status: DocumentStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Processing message")
    processing_started: bool = Field(..., description="Whether processing was started")


@router.post(
    "/{document_id}/process",
    response_model=SuccessResponse[ProcessingResponse],
    summary="Process document",
    description="Trigger document processing through the parsing and chunking pipeline"
)
async def process_document(
    document_id: UUID,
    request: ProcessingRequest,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Trigger document processing through the parsing and chunking pipeline."""
    try:
        # Get document with tenant isolation
        document = db_session.query(Document).filter(
            and_(
                Document.id == document_id,
                Document.tenant_id == current_user.tenant_id,
                Document.deleted_at.is_(None)
            )
        ).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Check if document is in a processable state
        if document.status == DocumentStatus.PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is already being processed"
            )

        if document.status == DocumentStatus.READY:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document has already been processed"
            )

        # Start async processing
        from agentic_rag.services.document_processor import process_document_async

        # Start processing in background
        asyncio.create_task(
            process_document_async(
                document_id=document_id,
                tenant_id=current_user.tenant_id,
                user_id=current_user.id,
                db_session=db_session,
                processing_options=request.processing_options
            )
        )

        response_data = ProcessingResponse(
            document_id=document_id,
            status=DocumentStatus.PROCESSING,
            message="Document processing started",
            processing_started=True
        )

        return SuccessResponse(
            data=response_data,
            message="Document processing started successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start document processing: {str(e)}"
        )


class StatusHistoryItem(BaseModel):
    """Document status history item."""

    status: DocumentStatus = Field(..., description="Document status")
    progress: float = Field(..., description="Processing progress")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message")
    timestamp: datetime = Field(..., description="Status update timestamp")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class StatusHistoryResponse(BaseModel):
    """Response model for document status history."""

    document_id: UUID = Field(..., description="Document ID")
    current_status: DocumentStatus = Field(..., description="Current document status")
    history: List[StatusHistoryItem] = Field(..., description="Status history")


@router.get(
    "/{document_id}/status",
    response_model=SuccessResponse[StatusHistoryResponse],
    summary="Get document status history",
    description="Get the processing status history for a document"
)
async def get_document_status_history(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Get the processing status history for a document."""
    try:
        # Get document with tenant isolation
        document = db_session.query(Document).filter(
            and_(
                Document.id == document_id,
                Document.tenant_id == current_user.tenant_id,
                Document.deleted_at.is_(None)
            )
        ).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Get status history from tracker
        from agentic_rag.services.document_status_tracker import get_document_status_tracker
        status_tracker = get_document_status_tracker()

        history = status_tracker.get_status_history(document_id)

        # Convert to response format
        history_items = [
            StatusHistoryItem(
                status=update.status,
                progress=update.progress,
                message=update.message,
                error=update.error,
                timestamp=update.timestamp,
                metadata=update.metadata
            )
            for update in history
        ]

        response_data = StatusHistoryResponse(
            document_id=document_id,
            current_status=document.status,
            history=history_items
        )

        return SuccessResponse(
            data=response_data,
            message=f"Retrieved {len(history_items)} status history items"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document status history: {str(e)}"
        )


# ============================================================================
# DOCUMENT CLEANUP ENDPOINTS
# ============================================================================

class CleanupResponse(BaseModel):
    """Response model for cleanup operations."""

    cleaned_count: int = Field(..., description="Number of documents cleaned up")
    total_chunks_removed: int = Field(..., description="Total chunks removed")
    total_links_removed: int = Field(..., description="Total links removed")
    total_files_removed: int = Field(..., description="Total files removed")
    cleanup_time: float = Field(..., description="Total cleanup time in seconds")
    failed_count: int = Field(0, description="Number of failed cleanups")


@router.post(
    "/cleanup/expired",
    response_model=SuccessResponse[CleanupResponse],
    summary="Clean up expired documents",
    description="Clean up documents that have exceeded the retention period"
)
async def cleanup_expired_documents(
    batch_size: int = Query(100, ge=1, le=1000, description="Number of documents to process"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Clean up documents that have exceeded the retention period."""
    try:
        from agentic_rag.services.document_cleanup import get_document_cleanup_service

        cleanup_service = get_document_cleanup_service()

        # Perform cleanup of expired documents
        results = await cleanup_service.cleanup_expired_documents(
            tenant_id=current_user.tenant_id,
            db_session=db_session,
            batch_size=batch_size
        )

        # Aggregate results
        cleaned_count = sum(1 for r in results if r.success)
        failed_count = sum(1 for r in results if not r.success)
        total_chunks_removed = sum(r.chunks_removed for r in results)
        total_links_removed = sum(r.links_removed for r in results)
        total_files_removed = sum(1 for r in results if r.file_removed)
        total_cleanup_time = sum(r.cleanup_time for r in results)

        response_data = CleanupResponse(
            cleaned_count=cleaned_count,
            total_chunks_removed=total_chunks_removed,
            total_links_removed=total_links_removed,
            total_files_removed=total_files_removed,
            cleanup_time=total_cleanup_time,
            failed_count=failed_count
        )

        return SuccessResponse(
            data=response_data,
            message=f"Cleanup completed: {cleaned_count} documents cleaned, {failed_count} failed"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup expired documents: {str(e)}"
        )
