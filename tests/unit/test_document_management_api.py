"""
Unit tests for Document Management API.

This module contains comprehensive tests for the document management API endpoints,
including listing, details, status tracking, deletion, and linking functionality.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from agentic_rag.api.routes.documents import (
    DocumentListItem,
    DocumentDetail,
    DocumentLinkCreate,
    DocumentLinkInfo,
    BulkDeleteRequest
)
from agentic_rag.models.database import Document, DocumentKind, DocumentStatus, User, Tenant


class TestDocumentModels:
    """Test document API models."""
    
    def test_document_list_item_model(self):
        """Test DocumentListItem model validation."""
        doc_id = uuid4()
        created_at = datetime.now(timezone.utc)
        
        item = DocumentListItem(
            id=doc_id,
            title="Test Document",
            kind=DocumentKind.RFQ,
            status=DocumentStatus.READY,
            sha256="abc123",
            version=1,
            pages=10,
            file_size=1024,
            chunk_count=5,
            processing_progress=1.0,
            created_at=created_at,
            updated_at=created_at,
            created_by=uuid4()
        )
        
        assert item.id == doc_id
        assert item.title == "Test Document"
        assert item.kind == DocumentKind.RFQ
        assert item.status == DocumentStatus.READY
        assert item.chunk_count == 5
        assert item.processing_progress == 1.0
    
    def test_document_detail_model(self):
        """Test DocumentDetail model validation."""
        doc_id = uuid4()
        tenant_id = uuid4()
        created_at = datetime.now(timezone.utc)
        
        detail = DocumentDetail(
            id=doc_id,
            tenant_id=tenant_id,
            title="Test Document",
            kind=DocumentKind.RFQ,
            status=DocumentStatus.PROCESSING,
            sha256="abc123",
            version=1,
            pages=10,
            processing_progress=0.5,
            processing_error=None,
            file_size=1024,
            chunk_count=3,
            created_at=created_at,
            updated_at=created_at,
            created_by=uuid4(),
            deleted_at=None,
            download_url="https://example.com/download"
        )
        
        assert detail.id == doc_id
        assert detail.tenant_id == tenant_id
        assert detail.status == DocumentStatus.PROCESSING
        assert detail.processing_progress == 0.5
        assert detail.chunk_count == 3
    
    def test_document_link_create_model(self):
        """Test DocumentLinkCreate model validation."""
        offer_id = uuid4()
        
        link_create = DocumentLinkCreate(
            offer_id=offer_id,
            offer_type="technical",
            confidence=0.85
        )
        
        assert link_create.offer_id == offer_id
        assert link_create.offer_type == "technical"
        assert link_create.confidence == 0.85
    
    def test_document_link_create_validation(self):
        """Test DocumentLinkCreate model validation constraints."""
        offer_id = uuid4()
        
        # Test confidence range validation
        with pytest.raises(ValueError):
            DocumentLinkCreate(
                offer_id=offer_id,
                offer_type="technical",
                confidence=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValueError):
            DocumentLinkCreate(
                offer_id=offer_id,
                offer_type="technical",
                confidence=-0.1  # Invalid: < 0.0
            )
    
    def test_bulk_delete_request_model(self):
        """Test BulkDeleteRequest model validation."""
        doc_ids = [uuid4(), uuid4(), uuid4()]
        
        request = BulkDeleteRequest(document_ids=doc_ids)
        
        assert len(request.document_ids) == 3
        assert all(isinstance(doc_id, type(uuid4())) for doc_id in request.document_ids)


class TestDocumentStatusEnum:
    """Test DocumentStatus enum functionality."""
    
    def test_document_status_values(self):
        """Test DocumentStatus enum values."""
        assert DocumentStatus.UPLOADED.value == "uploaded"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.READY.value == "ready"
        assert DocumentStatus.FAILED.value == "failed"
        assert DocumentStatus.DELETED.value == "deleted"
    
    def test_document_status_iteration(self):
        """Test DocumentStatus enum iteration."""
        statuses = list(DocumentStatus)
        assert len(statuses) == 5
        assert DocumentStatus.UPLOADED in statuses
        assert DocumentStatus.PROCESSING in statuses
        assert DocumentStatus.READY in statuses
        assert DocumentStatus.FAILED in statuses
        assert DocumentStatus.DELETED in statuses


class TestDocumentAPIValidation:
    """Test document API input validation."""
    
    def test_valid_offer_types(self):
        """Test valid offer types for document linking."""
        valid_types = ["technical", "commercial", "pricing"]
        
        for offer_type in valid_types:
            link_create = DocumentLinkCreate(
                offer_id=uuid4(),
                offer_type=offer_type,
                confidence=0.8
            )
            assert link_create.offer_type == offer_type
    
    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        offer_id = uuid4()
        
        # Valid confidence scores
        valid_scores = [0.0, 0.5, 1.0, 0.123, 0.999]
        for score in valid_scores:
            link_create = DocumentLinkCreate(
                offer_id=offer_id,
                offer_type="technical",
                confidence=score
            )
            assert link_create.confidence == score


if __name__ == "__main__":
    pytest.main([__file__])
