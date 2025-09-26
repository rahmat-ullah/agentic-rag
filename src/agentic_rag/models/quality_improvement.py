"""
Quality Improvement Models for Sprint 6 Story 6-05: Automated Quality Improvement System

This module defines the database models for automated quality improvement,
including quality assessments, improvement actions, monitoring, and automation rules.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from agentic_rag.database.base import Base


class QualityDimension(str, Enum):
    """Quality assessment dimensions."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    FRESHNESS = "freshness"
    RELEVANCE = "relevance"
    USABILITY = "usability"


class QualityIssueType(str, Enum):
    """Types of quality issues."""
    LOW_QUALITY_LINK = "low_quality_link"
    FREQUENT_CORRECTIONS = "frequent_corrections"
    POOR_CONTENT_QUALITY = "poor_content_quality"
    PROCESSING_ERRORS = "processing_errors"
    LOW_USER_SATISFACTION = "low_user_satisfaction"
    HIGH_BOUNCE_RATE = "high_bounce_rate"


class ImprovementActionType(str, Enum):
    """Types of improvement actions."""
    LINK_REVALIDATION = "link_revalidation"
    CONTENT_REPROCESSING = "content_reprocessing"
    EMBEDDING_UPDATE = "embedding_update"
    METADATA_REFRESH = "metadata_refresh"
    ALGORITHM_TUNING = "algorithm_tuning"
    CONTENT_REMOVAL = "content_removal"
    QUALITY_FLAGGING = "quality_flagging"


class ImprovementStatus(str, Enum):
    """Status of improvement actions."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityAssessment(Base):
    """Model for quality assessments of content, links, and system components."""
    
    __tablename__ = "quality_assessments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Assessment target
    target_type = Column(String(50), nullable=False)  # 'content', 'link', 'system', 'query'
    target_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Quality scores
    overall_quality_score = Column(Float, nullable=False)
    accuracy_score = Column(Float)
    completeness_score = Column(Float)
    freshness_score = Column(Float)
    relevance_score = Column(Float)
    usability_score = Column(Float)
    
    # Assessment metadata
    assessment_method = Column(String(100), nullable=False)
    confidence_level = Column(Float)
    sample_size = Column(Integer)
    assessment_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Quality dimensions and weights
    dimension_weights = Column(JSON)  # Dict[QualityDimension, float]
    dimension_scores = Column(JSON)   # Dict[QualityDimension, float]
    
    # Assessment context
    assessment_context = Column(JSON)  # Additional context data
    quality_issues = Column(JSON)      # List of identified issues
    improvement_suggestions = Column(JSON)  # List of suggested improvements
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_quality_assessments_tenant_target', 'tenant_id', 'target_type', 'target_id'),
        Index('idx_quality_assessments_score', 'tenant_id', 'overall_quality_score'),
        Index('idx_quality_assessments_date', 'tenant_id', 'assessment_date'),
    )


class QualityImprovement(Base):
    """Model for quality improvement actions and their tracking."""
    
    __tablename__ = "quality_improvements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Improvement target
    improvement_type = Column(String(50), nullable=False)  # QualityIssueType
    target_type = Column(String(50), nullable=False)
    target_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Trigger information
    trigger_reason = Column(String(200), nullable=False)
    trigger_threshold = Column(Float)
    trigger_metadata = Column(JSON)
    
    # Improvement action
    improvement_action = Column(String(100), nullable=False)  # ImprovementActionType
    action_parameters = Column(JSON)
    status = Column(String(20), default=ImprovementStatus.PENDING, nullable=False)
    
    # Quality tracking
    quality_before = Column(Float)
    quality_after = Column(Float)
    improvement_delta = Column(Float)
    effectiveness_score = Column(Float)
    
    # Execution tracking
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    failed_at = Column(DateTime)
    failure_reason = Column(Text)
    
    # Results and validation
    validation_results = Column(JSON)
    impact_metrics = Column(JSON)
    rollback_data = Column(JSON)  # Data needed for rollback if necessary
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_quality_improvements_tenant_type', 'tenant_id', 'improvement_type'),
        Index('idx_quality_improvements_status', 'tenant_id', 'status'),
        Index('idx_quality_improvements_target', 'tenant_id', 'target_type', 'target_id'),
    )


class QualityMonitoring(Base):
    """Model for quality monitoring and alerting."""
    
    __tablename__ = "quality_monitoring"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Monitoring configuration
    monitor_name = Column(String(100), nullable=False)
    monitor_type = Column(String(50), nullable=False)  # 'threshold', 'trend', 'pattern'
    target_type = Column(String(50), nullable=False)
    
    # Monitoring rules
    quality_threshold = Column(Float)
    trend_threshold = Column(Float)
    pattern_rules = Column(JSON)
    alert_conditions = Column(JSON)
    
    # Monitoring status
    is_active = Column(Boolean, default=True, nullable=False)
    last_check = Column(DateTime)
    next_check = Column(DateTime)
    check_interval_minutes = Column(Integer, default=60)
    
    # Alert configuration
    alert_enabled = Column(Boolean, default=True, nullable=False)
    alert_recipients = Column(JSON)  # List of user IDs or email addresses
    alert_severity = Column(String(20), default="medium")  # low, medium, high, critical
    
    # Monitoring results
    current_value = Column(Float)
    trend_direction = Column(String(20))  # increasing, decreasing, stable
    alert_count = Column(Integer, default=0)
    last_alert = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_quality_monitoring_tenant_active', 'tenant_id', 'is_active'),
        Index('idx_quality_monitoring_next_check', 'next_check'),
    )


class AutomationRule(Base):
    """Model for automated quality improvement rules."""
    
    __tablename__ = "automation_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Rule configuration
    rule_name = Column(String(100), nullable=False)
    rule_type = Column(String(50), nullable=False)  # 'quality_threshold', 'pattern_detection', 'trend_analysis'
    target_type = Column(String(50), nullable=False)
    
    # Rule conditions
    trigger_conditions = Column(JSON, nullable=False)  # Conditions that trigger the rule
    condition_logic = Column(String(20), default="AND")  # AND, OR
    
    # Rule actions
    improvement_actions = Column(JSON, nullable=False)  # List of actions to take
    action_parameters = Column(JSON)
    
    # Rule execution
    is_active = Column(Boolean, default=True, nullable=False)
    execution_count = Column(Integer, default=0)
    last_execution = Column(DateTime)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    
    # Rule validation
    dry_run_mode = Column(Boolean, default=False)  # Test mode without actual execution
    approval_required = Column(Boolean, default=False)
    max_executions_per_day = Column(Integer, default=100)
    
    # Rule metadata
    rule_description = Column(Text)
    rule_priority = Column(Integer, default=50)  # 1-100, higher = more important
    rule_metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_automation_rules_tenant_active', 'tenant_id', 'is_active'),
        Index('idx_automation_rules_type', 'tenant_id', 'rule_type'),
    )


class QualityAlert(Base):
    """Model for quality alerts and notifications."""
    
    __tablename__ = "quality_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Alert source
    monitor_id = Column(UUID(as_uuid=True), ForeignKey('quality_monitoring.id'), index=True)
    rule_id = Column(UUID(as_uuid=True), ForeignKey('automation_rules.id'), index=True)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # 'threshold_breach', 'trend_alert', 'pattern_detected'
    alert_severity = Column(String(20), nullable=False)
    alert_title = Column(String(200), nullable=False)
    alert_message = Column(Text, nullable=False)
    
    # Alert context
    target_type = Column(String(50))
    target_id = Column(UUID(as_uuid=True))
    quality_value = Column(Float)
    threshold_value = Column(Float)
    alert_metadata = Column(JSON)
    
    # Alert status
    status = Column(String(20), default="active", nullable=False)  # active, acknowledged, resolved, dismissed
    acknowledged_by = Column(UUID(as_uuid=True))
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    monitor = relationship("QualityMonitoring", backref="alerts")
    rule = relationship("AutomationRule", backref="alerts")
    
    # Indexes
    __table_args__ = (
        Index('idx_quality_alerts_tenant_status', 'tenant_id', 'status'),
        Index('idx_quality_alerts_severity', 'tenant_id', 'alert_severity'),
        Index('idx_quality_alerts_created', 'tenant_id', 'created_at'),
    )
