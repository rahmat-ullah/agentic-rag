"""
Content Correction Models for Sprint 6 Story 6-02

This module contains comprehensive models for user content corrections,
version control, and expert review workflow.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    Float,
    Index,
    CheckConstraint,
    JSON,
    func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from .database import Base, GUID


class CorrectionType(str, Enum):
    """Content correction type enumeration."""
    
    FACTUAL = "factual"
    FORMATTING = "formatting"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    GRAMMAR = "grammar"
    TERMINOLOGY = "terminology"


class CorrectionStatus(str, Enum):
    """Correction workflow status enumeration."""
    
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    REVERTED = "reverted"


class CorrectionPriority(str, Enum):
    """Correction priority enumeration."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewDecision(str, Enum):
    """Expert review decision enumeration."""
    
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"


class ContentCorrection(Base):
    """Content correction submission model."""
    
    __tablename__ = "content_corrections"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    chunk_id = Column(GUID, ForeignKey("chunk_meta.id"), nullable=False)
    user_id = Column(GUID, ForeignKey("app_user.id"), nullable=False)
    
    # Content information
    original_content = Column(Text, nullable=False)
    corrected_content = Column(Text, nullable=False)
    correction_reason = Column(Text, nullable=True)
    correction_type = Column(SQLEnum(CorrectionType), nullable=False)
    
    # Workflow information
    status = Column(SQLEnum(CorrectionStatus), default=CorrectionStatus.PENDING)
    priority = Column(SQLEnum(CorrectionPriority), default=CorrectionPriority.MEDIUM)
    
    # Review information
    reviewer_id = Column(GUID, ForeignKey("app_user.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    review_decision = Column(SQLEnum(ReviewDecision), nullable=True)
    review_notes = Column(Text, nullable=True)
    
    # Quality metrics
    confidence_score = Column(Float, nullable=True)  # Submitter's confidence in correction
    impact_score = Column(Float, nullable=True)  # Estimated impact of correction
    quality_score = Column(Float, nullable=True)  # Reviewer's quality assessment
    
    # Metadata and context
    correction_metadata = Column(JSONB, nullable=True)
    source_references = Column(JSONB, nullable=True)  # Supporting references
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    implemented_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    submitter = relationship("User", foreign_keys=[user_id])
    reviewer = relationship("User", foreign_keys=[reviewer_id])
    chunk = relationship("DocumentChunk")
    versions = relationship("ContentVersion", back_populates="correction")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_content_corrections_tenant_chunk", "tenant_id", "chunk_id"),
        Index("idx_content_corrections_status", "status"),
        Index("idx_content_corrections_type", "correction_type"),
        Index("idx_content_corrections_created_at", "created_at"),
        Index("idx_content_corrections_reviewer", "reviewer_id"),
        CheckConstraint("confidence_score IS NULL OR confidence_score BETWEEN 0.0 AND 1.0", name="check_confidence_score_range"),
        CheckConstraint("impact_score IS NULL OR impact_score BETWEEN 0.0 AND 1.0", name="check_impact_score_range"),
        CheckConstraint("quality_score IS NULL OR quality_score BETWEEN 0.0 AND 1.0", name="check_quality_score_range"),
    )
    
    def __repr__(self):
        return f"<ContentCorrection(id={self.id}, chunk_id={self.chunk_id}, type={self.correction_type}, status={self.status})>"


class ContentVersion(Base):
    """Content version tracking model."""
    
    __tablename__ = "content_versions"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    chunk_id = Column(GUID, ForeignKey("chunk_meta.id"), nullable=False)
    correction_id = Column(GUID, ForeignKey("content_corrections.id"), nullable=True)
    
    # Version information
    version_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA-256 hash for deduplication
    
    # Change tracking
    change_summary = Column(Text, nullable=True)
    change_metadata = Column(JSONB, nullable=True)
    diff_data = Column(JSONB, nullable=True)  # Structured diff information
    
    # Version status
    is_active = Column(Boolean, default=False)
    is_published = Column(Boolean, default=False)
    
    # Creator information
    created_by = Column(GUID, ForeignKey("app_user.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)
    readability_score = Column(Float, nullable=True)
    
    # Relationships
    chunk = relationship("DocumentChunk")
    correction = relationship("ContentCorrection", back_populates="versions")
    creator = relationship("User")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_content_versions_chunk_version", "chunk_id", "version_number"),
        Index("idx_content_versions_active", "chunk_id", "is_active"),
        Index("idx_content_versions_created_at", "created_at"),
        Index("idx_content_versions_hash", "content_hash"),
        CheckConstraint("quality_score IS NULL OR quality_score BETWEEN 0.0 AND 1.0", name="check_version_quality_score_range"),
        CheckConstraint("readability_score IS NULL OR readability_score BETWEEN 0.0 AND 1.0", name="check_readability_score_range"),
    )
    
    def __repr__(self):
        return f"<ContentVersion(id={self.id}, chunk_id={self.chunk_id}, version={self.version_number}, active={self.is_active})>"


class CorrectionReview(Base):
    """Expert review tracking model."""
    
    __tablename__ = "correction_reviews"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    correction_id = Column(GUID, ForeignKey("content_corrections.id"), nullable=False)
    reviewer_id = Column(GUID, ForeignKey("app_user.id"), nullable=False)
    
    # Review information
    decision = Column(SQLEnum(ReviewDecision), nullable=False)
    review_notes = Column(Text, nullable=True)
    quality_assessment = Column(JSONB, nullable=True)  # Structured quality metrics
    
    # Review criteria scores
    accuracy_score = Column(Float, nullable=True)
    clarity_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    
    # Review metadata
    review_duration_minutes = Column(Integer, nullable=True)
    review_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    correction = relationship("ContentCorrection")
    reviewer = relationship("User")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_correction_reviews_correction", "correction_id"),
        Index("idx_correction_reviews_reviewer", "reviewer_id"),
        Index("idx_correction_reviews_decision", "decision"),
        Index("idx_correction_reviews_completed", "completed_at"),
        CheckConstraint("accuracy_score IS NULL OR accuracy_score BETWEEN 0.0 AND 1.0", name="check_accuracy_score_range"),
        CheckConstraint("clarity_score IS NULL OR clarity_score BETWEEN 0.0 AND 1.0", name="check_clarity_score_range"),
        CheckConstraint("completeness_score IS NULL OR completeness_score BETWEEN 0.0 AND 1.0", name="check_completeness_score_range"),
        CheckConstraint("overall_score IS NULL OR overall_score BETWEEN 0.0 AND 1.0", name="check_overall_score_range"),
    )
    
    def __repr__(self):
        return f"<CorrectionReview(id={self.id}, correction_id={self.correction_id}, decision={self.decision})>"


class CorrectionWorkflow(Base):
    """Correction workflow state tracking model."""
    
    __tablename__ = "correction_workflows"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    correction_id = Column(GUID, ForeignKey("content_corrections.id"), nullable=False)
    
    # Workflow state
    current_step = Column(String(50), nullable=False)  # 'submission', 'review', 'approval', 'implementation'
    workflow_data = Column(JSONB, nullable=True)  # Workflow-specific data
    
    # Assignment information
    assigned_to = Column(GUID, ForeignKey("app_user.id"), nullable=True)
    assigned_at = Column(DateTime(timezone=True), nullable=True)
    due_date = Column(DateTime(timezone=True), nullable=True)
    
    # Progress tracking
    steps_completed = Column(JSONB, nullable=True)  # List of completed steps
    next_steps = Column(JSONB, nullable=True)  # List of upcoming steps
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    correction = relationship("ContentCorrection")
    assignee = relationship("User")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_correction_workflows_correction", "correction_id"),
        Index("idx_correction_workflows_step", "current_step"),
        Index("idx_correction_workflows_assigned", "assigned_to"),
        Index("idx_correction_workflows_due", "due_date"),
    )
    
    def __repr__(self):
        return f"<CorrectionWorkflow(id={self.id}, correction_id={self.correction_id}, step={self.current_step})>"


class CorrectionImpact(Base):
    """Track the impact of content corrections on system performance."""
    
    __tablename__ = "correction_impacts"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    correction_id = Column(GUID, ForeignKey("content_corrections.id"), nullable=False)
    
    # Impact metrics
    search_improvement = Column(Float, nullable=True)  # Search quality improvement
    user_satisfaction = Column(Float, nullable=True)  # User satisfaction change
    accuracy_improvement = Column(Float, nullable=True)  # Content accuracy improvement
    
    # Usage metrics
    chunk_access_before = Column(Integer, default=0)  # Access count before correction
    chunk_access_after = Column(Integer, default=0)  # Access count after correction
    feedback_improvement = Column(Float, nullable=True)  # Feedback score improvement
    
    # Implementation details
    re_embedding_completed = Column(Boolean, default=False)
    re_embedding_at = Column(DateTime(timezone=True), nullable=True)
    
    # Measurement period
    measurement_start = Column(DateTime(timezone=True), nullable=True)
    measurement_end = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    correction = relationship("ContentCorrection")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_correction_impacts_correction", "correction_id"),
        Index("idx_correction_impacts_re_embedding", "re_embedding_completed"),
        Index("idx_correction_impacts_created", "created_at"),
    )
    
    def __repr__(self):
        return f"<CorrectionImpact(id={self.id}, correction_id={self.correction_id}, re_embedded={self.re_embedding_completed})>"
