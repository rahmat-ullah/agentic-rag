"""
Unit tests for database models.

Tests the database model definitions, relationships, and validation.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock

from agentic_rag.models.database import (
    Base,
    Tenant,
    User,
    Document,
    DocumentLink,
    ChunkMeta,
    Feedback,
    DocumentKind,
    UserRole,
    FeedbackLabel
)
from tests.base import BaseUnitTest


class TestEnums(BaseUnitTest):
    """Test database enums."""
    
    def test_document_kind_enum(self):
        """Test DocumentKind enum values."""
        assert DocumentKind.RFQ == "RFQ"
        assert DocumentKind.RFP == "RFP"
        assert DocumentKind.TENDER == "Tender"
        assert DocumentKind.OFFER_TECH == "OfferTech"
        assert DocumentKind.OFFER_COMM == "OfferComm"
        assert DocumentKind.PRICING == "Pricing"
    
    def test_user_role_enum(self):
        """Test UserRole enum values."""
        assert UserRole.ADMIN == "admin"
        assert UserRole.ANALYST == "analyst"
        assert UserRole.VIEWER == "viewer"
    
    def test_feedback_label_enum(self):
        """Test FeedbackLabel enum values."""
        assert FeedbackLabel.UP == "up"
        assert FeedbackLabel.DOWN == "down"
        assert FeedbackLabel.EDIT == "edit"
        assert FeedbackLabel.BAD_LINK == "bad_link"
        assert FeedbackLabel.GOOD_LINK == "good_link"


class TestTenantModel(BaseUnitTest):
    """Test Tenant model."""
    
    def test_tenant_creation(self):
        """Test creating a tenant instance."""
        tenant_id = uuid.uuid4()
        tenant = Tenant(
            id=tenant_id,
            name="Test Tenant"
        )
        
        assert tenant.id == tenant_id
        assert tenant.name == "Test Tenant"
        assert tenant.created_at is None  # Not set until saved to DB
        assert tenant.updated_at is None  # Not set until saved to DB
    
    def test_tenant_repr(self):
        """Test tenant string representation."""
        tenant = Tenant(
            id=uuid.uuid4(),
            name="Test Tenant"
        )
        
        repr_str = repr(tenant)
        assert "Tenant" in repr_str
        assert "Test Tenant" in repr_str


class TestUserModel(BaseUnitTest):
    """Test User model."""

    def test_user_creation(self):
        """Test creating a user instance."""
        user_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        user = User(
            id=user_id,
            tenant_id=tenant_id,
            email="test@example.com",
            password_hash="$2b$12$test.hash.value",
            role=UserRole.ANALYST,
            is_active=True
        )
        
        assert user.id == user_id
        assert user.tenant_id == tenant_id
        assert user.email == "test@example.com"
        assert user.password_hash == "$2b$12$test.hash.value"
        assert user.role == UserRole.ANALYST
        assert user.is_active is True
    
    def test_user_with_explicit_values(self):
        """Test user with explicit values."""
        user = User(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            email="test@example.com",
            password_hash="hash",
            role=UserRole.VIEWER,
            is_active=True
        )

        assert user.role == UserRole.VIEWER
        assert user.is_active is True

    def test_user_repr(self):
        """Test user string representation."""
        user = User(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            email="test@example.com",
            password_hash="hash"
        )

        repr_str = repr(user)
        assert "User" in repr_str
        assert "test@example.com" in repr_str


class TestDocumentModel(BaseUnitTest):
    """Test Document model."""
    
    def test_document_creation(self):
        """Test creating a document instance."""
        doc_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        document = Document(
            id=doc_id,
            tenant_id=tenant_id,
            title="Test Document",
            kind=DocumentKind.RFQ,
            source_uri="s3://bucket/document.pdf",
            sha256="abc123def456",
            pages=10
        )

        assert document.id == doc_id
        assert document.tenant_id == tenant_id
        assert document.title == "Test Document"
        assert document.kind == DocumentKind.RFQ
        assert document.source_uri == "s3://bucket/document.pdf"
        assert document.sha256 == "abc123def456"
        assert document.pages == 10
    
    def test_document_optional_fields(self):
        """Test document with optional fields."""
        document = Document(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            title="Test Document",
            kind=DocumentKind.RFQ,
            source_uri="s3://bucket/document.pdf",
            sha256="abc123def456",
            version=2,
            pages=10
        )

        assert document.version == 2
        assert document.pages == 10
    
    def test_document_repr(self):
        """Test document string representation."""
        document = Document(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            title="Test Document",
            kind=DocumentKind.RFQ,
            source_uri="s3://bucket/document.pdf",
            sha256="abc123def456"
        )

        repr_str = repr(document)
        assert "Document" in repr_str
        assert "Test Document" in repr_str


class TestDocumentLinkModel(BaseUnitTest):
    """Test DocumentLink model."""
    
    def test_document_link_creation(self):
        """Test creating a document link instance."""
        link_id = uuid.uuid4()
        tenant_id = uuid.uuid4()
        rfq_id = uuid.uuid4()
        offer_id = uuid.uuid4()

        link = DocumentLink(
            id=link_id,
            tenant_id=tenant_id,
            rfq_id=rfq_id,
            offer_id=offer_id,
            offer_type="technical",
            confidence=0.85
        )

        assert link.id == link_id
        assert link.tenant_id == tenant_id
        assert link.rfq_id == rfq_id
        assert link.offer_id == offer_id
        assert link.offer_type == "technical"
        assert link.confidence == 0.85

    def test_document_link_offer_types(self):
        """Test document link with different offer types."""
        for offer_type in ["technical", "commercial", "pricing"]:
            link = DocumentLink(
                id=uuid.uuid4(),
                tenant_id=uuid.uuid4(),
                rfq_id=uuid.uuid4(),
                offer_id=uuid.uuid4(),
                offer_type=offer_type,
                confidence=0.75
            )
            assert link.offer_type == offer_type


class TestChunkMetaModel(BaseUnitTest):
    """Test ChunkMeta model."""

    def test_chunk_meta_creation(self):
        """Test creating a chunk meta instance."""
        chunk_id = uuid.uuid4()
        tenant_id = uuid.uuid4()
        doc_id = uuid.uuid4()

        chunk = ChunkMeta(
            id=chunk_id,
            tenant_id=tenant_id,
            document_id=doc_id,
            page_from=1,
            page_to=2,
            section_path=["Section 1", "Introduction"],
            token_count=500,
            hash="abc123def456",
            is_table=False,
            embedding_model_version="text-embedding-3-large"
        )

        assert chunk.id == chunk_id
        assert chunk.tenant_id == tenant_id
        assert chunk.document_id == doc_id
        assert chunk.page_from == 1
        assert chunk.page_to == 2
        assert chunk.section_path == ["Section 1", "Introduction"]
        assert chunk.token_count == 500
        assert chunk.hash == "abc123def456"
        assert chunk.is_table is False
        assert chunk.embedding_model_version == "text-embedding-3-large"

    def test_chunk_meta_defaults(self):
        """Test chunk meta default values."""
        chunk = ChunkMeta(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            is_table=False,
            retired=False
        )

        assert chunk.is_table is False
        assert chunk.retired is False


class TestFeedbackModel(BaseUnitTest):
    """Test Feedback model."""

    def test_feedback_creation(self):
        """Test creating a feedback instance."""
        feedback_id = uuid.uuid4()
        tenant_id = uuid.uuid4()
        created_by = uuid.uuid4()

        feedback = Feedback(
            id=feedback_id,
            tenant_id=tenant_id,
            query="test query",
            label=FeedbackLabel.UP,
            created_by=created_by
        )

        assert feedback.id == feedback_id
        assert feedback.tenant_id == tenant_id
        assert feedback.query == "test query"
        assert feedback.label == FeedbackLabel.UP
        assert feedback.created_by == created_by

    def test_feedback_optional_fields(self):
        """Test feedback with optional fields."""
        feedback = Feedback(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            query="test query",
            label=FeedbackLabel.EDIT,
            rfq_id=uuid.uuid4(),
            offer_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            notes="This is a test note",
            created_by=uuid.uuid4()
        )

        assert feedback.notes == "This is a test note"
        assert feedback.rfq_id is not None
        assert feedback.offer_id is not None
        assert feedback.chunk_id is not None
