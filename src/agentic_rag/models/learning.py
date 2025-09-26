"""
Learning Algorithm Models for Sprint 6 Story 6-03

This module contains comprehensive models for learning algorithms,
performance tracking, and adaptive system improvement.
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


class LearningAlgorithmType(str, Enum):
    """Learning algorithm type enumeration."""
    
    LINK_CONFIDENCE = "link_confidence"
    CHUNK_RANKING = "chunk_ranking"
    QUERY_EXPANSION = "query_expansion"
    NEGATIVE_FEEDBACK = "negative_feedback"
    CONTENT_QUALITY = "content_quality"


class LearningModelType(str, Enum):
    """Learning model type enumeration."""
    
    EXPONENTIAL_MOVING_AVERAGE = "exponential_moving_average"
    BAYESIAN_UPDATE = "bayesian_update"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED_FILTERING = "content_based_filtering"
    NEURAL_LANGUAGE_MODEL = "neural_language_model"


class LearningStatus(str, Enum):
    """Learning algorithm status enumeration."""
    
    ACTIVE = "active"
    PAUSED = "paused"
    TRAINING = "training"
    VALIDATING = "validating"
    DISABLED = "disabled"


class FeedbackSignalType(str, Enum):
    """Feedback signal type enumeration."""
    
    CLICK_THROUGH = "click_through"
    DWELL_TIME = "dwell_time"
    EXPLICIT_RATING = "explicit_rating"
    BOUNCE_RATE = "bounce_rate"
    CONVERSION_RATE = "conversion_rate"
    CORRECTION_FEEDBACK = "correction_feedback"


class LearningAlgorithm(Base):
    """Learning algorithm configuration and state model."""
    
    __tablename__ = "learning_algorithms"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    
    # Algorithm configuration
    algorithm_type = Column(SQLEnum(LearningAlgorithmType), nullable=False)
    model_type = Column(SQLEnum(LearningModelType), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Learning parameters
    learning_rate = Column(Float, default=0.01, nullable=False)
    validation_threshold = Column(Float, default=0.05, nullable=False)
    decay_factor = Column(Float, default=0.95, nullable=True)
    regularization_strength = Column(Float, default=0.001, nullable=True)
    
    # Algorithm state
    status = Column(SQLEnum(LearningStatus), default=LearningStatus.ACTIVE)
    current_version = Column(String(20), default="1.0.0")
    model_parameters = Column(JSONB, nullable=True)
    training_data_size = Column(Integer, default=0)
    
    # Performance metrics
    accuracy_score = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Configuration
    is_enabled = Column(Boolean, default=True)
    auto_update = Column(Boolean, default=True)
    validation_frequency_hours = Column(Integer, default=24)
    
    # Metadata
    algorithm_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_trained_at = Column(DateTime(timezone=True), nullable=True)
    last_validated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    performance_metrics = relationship("LearningPerformanceMetric", back_populates="algorithm")
    learning_sessions = relationship("LearningSession", back_populates="algorithm")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_learning_algorithms_tenant_type", "tenant_id", "algorithm_type"),
        Index("idx_learning_algorithms_status", "status"),
        Index("idx_learning_algorithms_enabled", "is_enabled"),
        Index("idx_learning_algorithms_updated", "updated_at"),
        CheckConstraint("learning_rate > 0.0 AND learning_rate <= 1.0", name="check_learning_rate_range"),
        CheckConstraint("validation_threshold >= 0.0 AND validation_threshold <= 1.0", name="check_validation_threshold_range"),
        CheckConstraint("accuracy_score IS NULL OR accuracy_score BETWEEN 0.0 AND 1.0", name="check_accuracy_score_range"),
        CheckConstraint("precision_score IS NULL OR precision_score BETWEEN 0.0 AND 1.0", name="check_precision_score_range"),
        CheckConstraint("recall_score IS NULL OR recall_score BETWEEN 0.0 AND 1.0", name="check_recall_score_range"),
        CheckConstraint("f1_score IS NULL OR f1_score BETWEEN 0.0 AND 1.0", name="check_f1_score_range"),
    )
    
    def __repr__(self):
        return f"<LearningAlgorithm(id={self.id}, type={self.algorithm_type}, status={self.status})>"


class LearningSession(Base):
    """Learning session tracking model."""
    
    __tablename__ = "learning_sessions"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    algorithm_id = Column(GUID, ForeignKey("learning_algorithms.id"), nullable=False)
    
    # Session information
    session_type = Column(String(50), nullable=False)  # 'training', 'validation', 'inference'
    session_name = Column(String(100), nullable=True)
    
    # Data information
    input_data_size = Column(Integer, nullable=False)
    processed_records = Column(Integer, default=0)
    successful_updates = Column(Integer, default=0)
    failed_updates = Column(Integer, default=0)
    
    # Performance metrics
    processing_time_seconds = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    
    # Results
    performance_improvement = Column(Float, nullable=True)
    accuracy_change = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)
    
    # Session metadata
    session_metadata = Column(JSONB, nullable=True)
    error_details = Column(JSONB, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    algorithm = relationship("LearningAlgorithm", back_populates="learning_sessions")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_learning_sessions_algorithm", "algorithm_id"),
        Index("idx_learning_sessions_type", "session_type"),
        Index("idx_learning_sessions_started", "started_at"),
        Index("idx_learning_sessions_completed", "completed_at"),
    )
    
    def __repr__(self):
        return f"<LearningSession(id={self.id}, algorithm_id={self.algorithm_id}, type={self.session_type})>"


class LearningPerformanceMetric(Base):
    """Learning algorithm performance tracking model."""
    
    __tablename__ = "learning_performance_metrics"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    algorithm_id = Column(GUID, ForeignKey("learning_algorithms.id"), nullable=False)
    
    # Metric information
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)  # 'accuracy', 'precision', 'recall', 'custom'
    
    # Context information
    measurement_period_start = Column(DateTime(timezone=True), nullable=False)
    measurement_period_end = Column(DateTime(timezone=True), nullable=False)
    sample_size = Column(Integer, nullable=False)
    
    # Comparison metrics
    baseline_value = Column(Float, nullable=True)
    improvement_percentage = Column(Float, nullable=True)
    statistical_significance = Column(Float, nullable=True)
    
    # Metadata
    metric_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    algorithm = relationship("LearningAlgorithm", back_populates="performance_metrics")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_learning_performance_algorithm", "algorithm_id"),
        Index("idx_learning_performance_metric", "metric_name"),
        Index("idx_learning_performance_recorded", "recorded_at"),
        Index("idx_learning_performance_period", "measurement_period_start", "measurement_period_end"),
    )
    
    def __repr__(self):
        return f"<LearningPerformanceMetric(id={self.id}, metric={self.metric_name}, value={self.metric_value})>"


class FeedbackSignal(Base):
    """Feedback signal processing model."""
    
    __tablename__ = "feedback_signals"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    
    # Signal information
    signal_type = Column(SQLEnum(FeedbackSignalType), nullable=False)
    target_type = Column(String(50), nullable=False)  # 'chunk', 'link', 'query', 'document'
    target_id = Column(GUID, nullable=False)
    
    # Signal data
    signal_value = Column(Float, nullable=False)
    signal_strength = Column(Float, default=1.0)
    signal_confidence = Column(Float, default=1.0)
    
    # Context information
    user_id = Column(GUID, ForeignKey("app_user.id"), nullable=True)
    session_id = Column(String(100), nullable=True)
    query_context = Column(Text, nullable=True)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    processing_algorithm_id = Column(GUID, ForeignKey("learning_algorithms.id"), nullable=True)
    
    # Signal metadata
    signal_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    processing_algorithm = relationship("LearningAlgorithm")
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_feedback_signals_tenant_type", "tenant_id", "signal_type"),
        Index("idx_feedback_signals_target", "target_type", "target_id"),
        Index("idx_feedback_signals_processed", "is_processed"),
        Index("idx_feedback_signals_created", "created_at"),
        Index("idx_feedback_signals_user", "user_id"),
        CheckConstraint("signal_strength >= 0.0 AND signal_strength <= 1.0", name="check_signal_strength_range"),
        CheckConstraint("signal_confidence >= 0.0 AND signal_confidence <= 1.0", name="check_signal_confidence_range"),
    )
    
    def __repr__(self):
        return f"<FeedbackSignal(id={self.id}, type={self.signal_type}, target={self.target_type}:{self.target_id})>"


class LearningModelState(Base):
    """Learning model state storage model."""
    
    __tablename__ = "learning_model_states"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    algorithm_id = Column(GUID, ForeignKey("learning_algorithms.id"), nullable=False)
    
    # State information
    state_name = Column(String(100), nullable=False)
    state_version = Column(String(20), nullable=False)
    state_type = Column(String(50), nullable=False)  # 'weights', 'parameters', 'embeddings', 'rules'
    
    # State data
    state_data = Column(JSONB, nullable=False)
    state_size_bytes = Column(Integer, nullable=True)
    compression_type = Column(String(20), nullable=True)
    
    # Validation information
    is_validated = Column(Boolean, default=False)
    validation_score = Column(Float, nullable=True)
    validation_date = Column(DateTime(timezone=True), nullable=True)
    
    # Backup and versioning
    is_active = Column(Boolean, default=True)
    is_backup = Column(Boolean, default=False)
    parent_state_id = Column(GUID, ForeignKey("learning_model_states.id"), nullable=True)
    
    # Metadata
    state_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    algorithm = relationship("LearningAlgorithm")
    parent_state = relationship("LearningModelState", remote_side=[id])
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_learning_model_states_algorithm", "algorithm_id"),
        Index("idx_learning_model_states_name_version", "state_name", "state_version"),
        Index("idx_learning_model_states_active", "is_active"),
        Index("idx_learning_model_states_validated", "is_validated"),
        Index("idx_learning_model_states_created", "created_at"),
        CheckConstraint("validation_score IS NULL OR validation_score BETWEEN 0.0 AND 1.0", name="check_validation_score_range"),
    )
    
    def __repr__(self):
        return f"<LearningModelState(id={self.id}, name={self.state_name}, version={self.state_version})>"


class ABTestExperiment(Base):
    """A/B testing experiment model for learning algorithms."""
    
    __tablename__ = "ab_test_experiments"
    
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id = Column(GUID, ForeignKey("tenant.id"), nullable=False)
    
    # Experiment information
    experiment_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    hypothesis = Column(Text, nullable=True)
    
    # Experiment configuration
    control_algorithm_id = Column(GUID, ForeignKey("learning_algorithms.id"), nullable=False)
    treatment_algorithm_id = Column(GUID, ForeignKey("learning_algorithms.id"), nullable=False)
    traffic_split_percentage = Column(Float, default=50.0)
    
    # Experiment status
    status = Column(String(20), default="draft")  # 'draft', 'running', 'paused', 'completed', 'cancelled'
    is_active = Column(Boolean, default=False)
    
    # Success criteria
    primary_metric = Column(String(100), nullable=False)
    success_threshold = Column(Float, nullable=False)
    minimum_sample_size = Column(Integer, default=1000)
    confidence_level = Column(Float, default=0.95)
    
    # Results
    control_metric_value = Column(Float, nullable=True)
    treatment_metric_value = Column(Float, nullable=True)
    statistical_significance = Column(Float, nullable=True)
    effect_size = Column(Float, nullable=True)
    
    # Experiment metadata
    experiment_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    control_algorithm = relationship("LearningAlgorithm", foreign_keys=[control_algorithm_id])
    treatment_algorithm = relationship("LearningAlgorithm", foreign_keys=[treatment_algorithm_id])
    
    # Constraints and indexes
    __table_args__ = (
        Index("idx_ab_test_experiments_tenant", "tenant_id"),
        Index("idx_ab_test_experiments_status", "status"),
        Index("idx_ab_test_experiments_active", "is_active"),
        Index("idx_ab_test_experiments_started", "started_at"),
        CheckConstraint("traffic_split_percentage > 0.0 AND traffic_split_percentage <= 100.0", name="check_traffic_split_range"),
        CheckConstraint("confidence_level > 0.0 AND confidence_level < 1.0", name="check_confidence_level_range"),
        CheckConstraint("statistical_significance IS NULL OR statistical_significance BETWEEN 0.0 AND 1.0", name="check_statistical_significance_range"),
    )
    
    def __repr__(self):
        return f"<ABTestExperiment(id={self.id}, name={self.experiment_name}, status={self.status})>"
