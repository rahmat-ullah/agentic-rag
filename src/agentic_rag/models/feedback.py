"""
Enhanced Feedback Models for Sprint 6

This module contains comprehensive feedback models that support the full range
of feedback types specified in Sprint 6, Story 6-01: Feedback Collection System.
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


class FeedbackType(str, Enum):
    """Enhanced feedback type enumeration."""
    
    SEARCH_RESULT = "search_result"
    LINK_QUALITY = "link_quality"
    ANSWER_QUALITY = "answer_quality"
    GENERAL = "general"
    SYSTEM_USABILITY = "system_usability"


class FeedbackCategory(str, Enum):
    """Feedback category enumeration."""
    
    # Search result categories
    NOT_RELEVANT = "not_relevant"
    MISSING_INFORMATION = "missing_information"
    OUTDATED_CONTENT = "outdated_content"
    WRONG_DOCUMENT_TYPE = "wrong_document_type"
    
    # Link quality categories
    INCORRECT_LINK = "incorrect_link"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_LINK = "missing_link"
    DUPLICATE_LINK = "duplicate_link"
    
    # Answer quality categories
    INACCURATE_INFORMATION = "inaccurate_information"
    INCOMPLETE_ANSWER = "incomplete_answer"
    POOR_FORMATTING = "poor_formatting"
    MISSING_CITATIONS = "missing_citations"
    
    # General categories
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    PERFORMANCE_ISSUE = "performance_issue"
    USABILITY_ISSUE = "usability_issue"


class FeedbackStatus(str, Enum):
    """Feedback processing status enumeration."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    REVIEWED = "reviewed"
    IMPLEMENTED = "implemented"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class FeedbackPriority(str, Enum):
    """Feedback priority enumeration."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UserFeedbackSubmission(Base):
    """Enhanced user feedback submission model."""
    
    __tablename__ = "user_feedback"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    user_id = Column(GUID, ForeignKey("app_user.id"), nullable=False)
    
    # Feedback classification
    feedback_type = Column(SQLEnum(FeedbackType), nullable=False)
    feedback_category = Column(SQLEnum(FeedbackCategory), nullable=True)
    
    # Target identification
    target_id = Column(GUID, nullable=True)  # search_result_id, link_id, answer_id, etc.
    target_type = Column(String(50), nullable=True)  # "search_result", "document_link", "answer", etc.
    
    # Feedback content
    rating = Column(Integer, nullable=True)  # 1-5 scale or -1/1 for thumbs down/up
    feedback_text = Column(Text, nullable=True)
    
    # Context information
    query = Column(Text, nullable=True)  # Original query that led to this feedback
    session_id = Column(String(255), nullable=True)  # User session identifier
    context_metadata = Column(JSONB, nullable=True)  # Additional context data
    
    # Processing information
    status = Column(SQLEnum(FeedbackStatus), default=FeedbackStatus.PENDING)
    priority = Column(SQLEnum(FeedbackPriority), default=FeedbackPriority.MEDIUM)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Processing results
    processing_notes = Column(Text, nullable=True)
    assigned_to = Column(GUID, ForeignKey("app_user.id"), nullable=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    assigned_user = relationship("User", foreign_keys=[assigned_to])
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_user_feedback_tenant_type", "tenant_id", "feedback_type"),
        Index("idx_user_feedback_status", "status"),
        Index("idx_user_feedback_created_at", "created_at"),
        Index("idx_user_feedback_target", "target_id", "target_type"),
        Index("idx_user_feedback_user_session", "user_id", "session_id"),
        CheckConstraint("rating IS NULL OR rating BETWEEN -1 AND 5", name="check_rating_range"),
    )
    
    def __repr__(self):
        return f"<UserFeedbackSubmission(id={self.id}, type={self.feedback_type}, status={self.status})>"


class FeedbackAggregation(Base):
    """Aggregated feedback statistics for targets."""
    
    __tablename__ = "feedback_aggregation"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    
    # Target identification
    target_id = Column(GUID, nullable=False)
    target_type = Column(String(50), nullable=False)
    feedback_type = Column(SQLEnum(FeedbackType), nullable=False)
    
    # Aggregated metrics
    total_feedback_count = Column(Integer, default=0)
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    average_rating = Column(Float, nullable=True)
    
    # Category breakdown
    category_counts = Column(JSONB, nullable=True)  # {"not_relevant": 5, "outdated_content": 2}
    
    # Quality scores
    quality_score = Column(Float, nullable=True)  # Calculated quality score (0.0 - 1.0)
    confidence_score = Column(Float, nullable=True)  # Confidence in the quality score
    
    # Timestamps
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    first_feedback_at = Column(DateTime(timezone=True), nullable=True)
    last_feedback_at = Column(DateTime(timezone=True), nullable=True)
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_feedback_agg_tenant_target", "tenant_id", "target_id", "target_type"),
        Index("idx_feedback_agg_type", "feedback_type"),
        Index("idx_feedback_agg_quality", "quality_score"),
        CheckConstraint("quality_score IS NULL OR quality_score BETWEEN 0.0 AND 1.0", name="check_quality_score_range"),
        CheckConstraint("confidence_score IS NULL OR confidence_score BETWEEN 0.0 AND 1.0", name="check_confidence_score_range"),
    )
    
    def __repr__(self):
        return f"<FeedbackAggregation(target_id={self.target_id}, type={self.feedback_type}, quality_score={self.quality_score})>"


class FeedbackImpact(Base):
    """Track the impact of feedback on system improvements."""
    
    __tablename__ = "feedback_impact"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    feedback_id = Column(GUID, ForeignKey("user_feedback.id"), nullable=False)
    
    # Impact details
    improvement_type = Column(String(100), nullable=False)  # "link_confidence_adjustment", "chunk_reranking", etc.
    improvement_description = Column(Text, nullable=True)
    
    # Metrics
    before_metric = Column(Float, nullable=True)
    after_metric = Column(Float, nullable=True)
    improvement_percentage = Column(Float, nullable=True)
    
    # Implementation details
    implemented_at = Column(DateTime(timezone=True), server_default=func.now())
    implementation_notes = Column(Text, nullable=True)
    
    # Relationships
    feedback = relationship("UserFeedbackSubmission")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_feedback_impact_tenant_feedback", "tenant_id", "feedback_id"),
        Index("idx_feedback_impact_type", "improvement_type"),
        Index("idx_feedback_impact_implemented", "implemented_at"),
    )
    
    def __repr__(self):
        return f"<FeedbackImpact(feedback_id={self.feedback_id}, type={self.improvement_type})>"


class FeedbackSession(Base):
    """Track user feedback sessions for analytics."""
    
    __tablename__ = "feedback_session"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    user_id = Column(GUID, ForeignKey("app_user.id"), nullable=False)
    session_id = Column(String(255), nullable=False)
    
    # Session metrics
    total_interactions = Column(Integer, default=0)
    feedback_submissions = Column(Integer, default=0)
    session_duration_seconds = Column(Integer, nullable=True)
    
    # Session context
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_feedback_session_tenant_user", "tenant_id", "user_id"),
        Index("idx_feedback_session_session_id", "session_id"),
        Index("idx_feedback_session_started", "started_at"),
    )
    
    def __repr__(self):
        return f"<FeedbackSession(id={self.id}, user_id={self.user_id}, submissions={self.feedback_submissions})>"
