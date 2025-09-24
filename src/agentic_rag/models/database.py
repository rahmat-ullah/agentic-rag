"""
Database models for Agentic RAG System.

This module contains SQLAlchemy models that define the database schema
for the Agentic RAG system, including tenancy, documents, chunks, and feedback.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    UUID,
    ARRAY,
    CheckConstraint,
    Index,
    UniqueConstraint,
    Float,
    TypeDecorator,
    JSON,
    func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, ARRAY as PostgresARRAY
import json


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type when available, otherwise uses String(36).
    """
    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PostgresUUID())
        else:
            return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


class JSONArray(TypeDecorator):
    """Platform-independent ARRAY type.

    Uses PostgreSQL's ARRAY type when available, otherwise uses JSON.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PostgresARRAY(String))
        else:
            return dialect.type_descriptor(Text)

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value  # PostgreSQL handles arrays natively
        else:
            return json.dumps(value)  # Store as JSON string for other databases

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value  # PostgreSQL returns arrays natively
        else:
            return json.loads(value)  # Parse JSON string for other databases


Base = declarative_base()


class DocumentKind(str, Enum):
    """Document type enumeration."""

    RFQ = "RFQ"
    RFP = "RFP"
    TENDER = "Tender"
    OFFER_TECH = "OfferTech"
    OFFER_COMM = "OfferComm"
    PRICING = "Pricing"


class DocumentStatus(str, Enum):
    """Document processing status enumeration."""

    UPLOADED = "uploaded"           # File uploaded, not yet processed
    PROCESSING = "processing"       # Currently being processed (parsing, chunking)
    READY = "ready"                # Processing complete, ready for use
    FAILED = "failed"              # Processing failed
    DELETED = "deleted"            # Soft deleted


class UserRole(str, Enum):
    """User role enumeration."""

    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class LinkType(str, Enum):
    """Document link type enumeration."""

    MANUAL = "manual"
    AUTOMATIC = "automatic"
    SUGGESTED = "suggested"


class UserFeedback(str, Enum):
    """User feedback on document links enumeration."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MODIFIED = "modified"


class FeedbackLabel(str, Enum):
    """Feedback label enumeration."""
    
    UP = "up"
    DOWN = "down"
    EDIT = "edit"
    BAD_LINK = "bad_link"
    GOOD_LINK = "good_link"


class Tenant(Base):
    """Tenant model for multi-tenancy support."""
    
    __tablename__ = "tenant"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    users = relationship("User", back_populates="tenant")
    documents = relationship("Document", back_populates="tenant")
    
    def __repr__(self):
        return f"<Tenant(id={self.id}, name='{self.name}')>"


class User(Base):
    """User model with role-based access control."""

    __tablename__ = "app_user"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    created_documents = relationship("Document", back_populates="created_by_user")
    feedback = relationship("Feedback", back_populates="created_by_user")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"


class Document(Base):
    """Document model for storing document metadata."""

    __tablename__ = "document"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    kind = Column(SQLEnum(DocumentKind), nullable=False)
    title = Column(String(500))
    source_uri = Column(String(1000))  # Object store URI
    sha256 = Column(String(64), nullable=False)
    version = Column(Integer, default=1)
    pages = Column(Integer)

    # Status and processing metadata
    status = Column(SQLEnum(DocumentStatus), nullable=False, default=DocumentStatus.UPLOADED)
    processing_progress = Column(Float, default=0.0)  # 0.0 to 1.0
    processing_error = Column(Text)  # Error message if processing failed
    file_size = Column(BigInteger)  # File size in bytes
    chunk_count = Column(Integer, default=0)  # Number of chunks created

    # Timestamps and user tracking
    created_by = Column(GUID, ForeignKey("app_user.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True))  # For soft delete
    
    # Relationships
    tenant = relationship("Tenant", back_populates="documents")
    created_by_user = relationship("User", back_populates="created_documents")
    chunks = relationship("DocumentChunk", back_populates="document")
    rfq_links = relationship("DocumentLink", foreign_keys="DocumentLink.rfq_id", back_populates="rfq")
    offer_links = relationship("DocumentLink", foreign_keys="DocumentLink.offer_id", back_populates="offer")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "sha256", name="uq_document_tenant_sha256"),
        CheckConstraint("processing_progress BETWEEN 0 AND 1", name="ck_processing_progress_range"),
        CheckConstraint("chunk_count >= 0", name="ck_chunk_count_positive"),
        Index("idx_document_tenant_kind", "tenant_id", "kind"),
        Index("idx_document_tenant_status", "tenant_id", "status"),
        Index("idx_document_created_at", "created_at"),
        Index("idx_document_deleted_at", "deleted_at"),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, kind='{self.kind}', title='{self.title}')>"


class DocumentLink(Base):
    """Enhanced document link model for RFQ-Offer relationships with advanced features."""

    __tablename__ = "document_link"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    rfq_id = Column(GUID, ForeignKey("document.id"), nullable=False)
    offer_id = Column(GUID, ForeignKey("document.id"), nullable=False)
    offer_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)

    # Enhanced linking features
    link_type = Column(SQLEnum(LinkType), nullable=False, default=LinkType.MANUAL)
    created_by = Column(GUID, ForeignKey("app_user.id"), nullable=True)
    validated_by = Column(GUID, ForeignKey("app_user.id"), nullable=True)
    validated_at = Column(DateTime(timezone=True), nullable=True)
    link_metadata = Column(JSON, nullable=True)  # Additional metadata for the link
    quality_score = Column(Float, nullable=True)  # Quality assessment score
    user_feedback = Column(SQLEnum(UserFeedback), nullable=True)
    feedback_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)  # User notes about the link

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    rfq = relationship("Document", foreign_keys=[rfq_id], back_populates="rfq_links")
    offer = relationship("Document", foreign_keys=[offer_id], back_populates="offer_links")
    created_by_user = relationship("User", foreign_keys=[created_by])
    validated_by_user = relationship("User", foreign_keys=[validated_by])

    # Constraints
    __table_args__ = (
        CheckConstraint("offer_type IN ('technical', 'commercial', 'pricing')", name="ck_offer_type"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_confidence_range"),
        CheckConstraint("link_type IN ('manual', 'automatic', 'suggested')", name="ck_link_type"),
        CheckConstraint("quality_score IS NULL OR (quality_score BETWEEN 0 AND 1)", name="ck_quality_score_range"),
        CheckConstraint("user_feedback IS NULL OR user_feedback IN ('accepted', 'rejected', 'modified')", name="ck_user_feedback"),
        UniqueConstraint("tenant_id", "rfq_id", "offer_id", name="uq_document_link_tenant_rfq_offer"),
        Index("idx_document_link_tenant_rfq_type", "tenant_id", "rfq_id", "offer_type"),
        Index("idx_document_link_confidence", "confidence"),
        Index("idx_document_link_link_type", "tenant_id", "link_type"),
        Index("idx_document_link_created_by", "created_by"),
        Index("idx_document_link_quality_score", "quality_score"),
        Index("idx_document_link_validated", "validated_by", "validated_at"),
    )

    def __repr__(self):
        return f"<DocumentLink(rfq_id={self.rfq_id}, offer_id={self.offer_id}, type={self.link_type}, confidence={self.confidence})>"


class DocumentChunk(Base):
    """Document chunk model for storing text chunks with metadata."""

    __tablename__ = "chunk_meta"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    document_id = Column(GUID, ForeignKey("document.id"), nullable=False)
    page_from = Column(Integer)
    page_to = Column(Integer)
    section_path = Column(JSONArray)  # ["Section 2", "Scope", "2.1 Electrical"]
    token_count = Column(Integer)
    hash = Column(String(64))  # For deduplication
    is_table = Column(Boolean, default=False)
    retired = Column(Boolean, default=False)
    embedding_model_version = Column(String(50))  # Track embedding model version
    embedding_created_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")
    feedback = relationship("Feedback", back_populates="chunk")

    # Constraints
    __table_args__ = (
        Index("idx_chunk_meta_tenant_document", "tenant_id", "document_id"),
        UniqueConstraint("tenant_id", "hash", name="uq_chunk_meta_tenant_hash"),
        Index("idx_chunk_meta_embedding_version", "embedding_model_version", "tenant_id"),
        Index("idx_chunk_meta_retired", "retired"),
    )

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, token_count={self.token_count})>"


# Alias for backward compatibility
ChunkMeta = DocumentChunk


class Feedback(Base):
    """Feedback model for continuous learning."""
    
    __tablename__ = "feedback"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    query = Column(Text, nullable=False)
    rfq_id = Column(GUID, ForeignKey("document.id"))
    offer_id = Column(GUID, ForeignKey("document.id"))
    chunk_id = Column(GUID, ForeignKey("chunk_meta.id"))
    label = Column(SQLEnum(FeedbackLabel), nullable=False)
    notes = Column(Text)
    created_by = Column(GUID, ForeignKey("app_user.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    created_by_user = relationship("User", back_populates="feedback")
    rfq = relationship("Document", foreign_keys=[rfq_id])
    offer = relationship("Document", foreign_keys=[offer_id])
    chunk = relationship("ChunkMeta", back_populates="feedback")
    
    # Constraints
    __table_args__ = (
        Index("idx_feedback_tenant_query", "tenant_id", "query"),
        Index("idx_feedback_label", "label"),
        Index("idx_feedback_created_at", "created_at"),
    )
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, label='{self.label}', query='{self.query[:50]}...')>"
