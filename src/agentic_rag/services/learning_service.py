"""
Core Learning Service for Sprint 6 Story 6-03

This service implements learning algorithms for link confidence adjustment,
chunk ranking improvement, query expansion learning, and negative feedback handling.
"""

import uuid
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from agentic_rag.models.learning import (
    LearningAlgorithm,
    LearningSession,
    LearningPerformanceMetric,
    FeedbackSignal,
    LearningModelState,
    LearningAlgorithmType,
    LearningModelType,
    LearningStatus,
    FeedbackSignalType
)
from agentic_rag.models.feedback import UserFeedbackSubmission
from agentic_rag.models.corrections import ContentCorrection
from agentic_rag.database.connection import get_database_session
from agentic_rag.api.exceptions import ValidationError, ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class LearningUpdate:
    """Result of a learning algorithm update."""
    algorithm_id: uuid.UUID
    old_value: float
    new_value: float
    improvement: float
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class LearningValidationResult:
    """Result of learning algorithm validation."""
    is_valid: bool
    validation_score: float
    improvement_percentage: float
    statistical_significance: float
    validation_metadata: Dict[str, Any]


class BaseLearningAlgorithm(ABC):
    """Base class for all learning algorithms."""
    
    def __init__(self, algorithm_id: uuid.UUID, learning_rate: float = 0.01):
        self.algorithm_id = algorithm_id
        self.learning_rate = learning_rate
        self.validation_threshold = 0.05
        self.logger = logger.bind(algorithm_id=algorithm_id)
    
    @abstractmethod
    async def update(self, signal: FeedbackSignal, current_value: float) -> LearningUpdate:
        """Update value based on feedback signal."""
        pass
    
    @abstractmethod
    async def validate_update(self, old_value: float, new_value: float) -> bool:
        """Validate that update improves performance."""
        pass
    
    def _calculate_confidence(self, signal_strength: float, signal_confidence: float) -> float:
        """Calculate confidence for learning update."""
        return min(signal_strength * signal_confidence, 1.0)
    
    def _apply_bounds(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Apply bounds to learning update."""
        return max(min_val, min(max_val, value))


class LinkConfidenceLearningAlgorithm(BaseLearningAlgorithm):
    """Learning algorithm for link confidence adjustment."""
    
    async def update(self, signal: FeedbackSignal, current_value: float) -> LearningUpdate:
        """Update link confidence based on feedback signal."""
        try:
            # Calculate adjustment based on signal type and value
            if signal.signal_type == FeedbackSignalType.EXPLICIT_RATING:
                # Rating scale 1-5, normalize to -1 to 1
                normalized_rating = (signal.signal_value - 3) / 2
                adjustment = self.learning_rate * normalized_rating
            elif signal.signal_type == FeedbackSignalType.CLICK_THROUGH:
                # Click-through rate improvement
                adjustment = self.learning_rate * (signal.signal_value - 0.5)
            elif signal.signal_type == FeedbackSignalType.DWELL_TIME:
                # Dwell time above 30 seconds is positive
                normalized_dwell = min((signal.signal_value - 30) / 60, 1.0)
                adjustment = self.learning_rate * normalized_dwell
            elif signal.signal_type == FeedbackSignalType.BOUNCE_RATE:
                # Lower bounce rate is better
                adjustment = -self.learning_rate * signal.signal_value
            else:
                adjustment = 0.0
            
            # Apply signal strength and confidence
            weighted_adjustment = adjustment * signal.signal_strength * signal.signal_confidence
            
            # Calculate new value with bounds
            new_value = self._apply_bounds(current_value + weighted_adjustment)
            
            # Calculate improvement and confidence
            improvement = new_value - current_value
            confidence = self._calculate_confidence(signal.signal_strength, signal.signal_confidence)
            
            self.logger.info(
                "link_confidence_updated",
                target_id=signal.target_id,
                old_value=current_value,
                new_value=new_value,
                improvement=improvement,
                signal_type=signal.signal_type
            )
            
            return LearningUpdate(
                algorithm_id=self.algorithm_id,
                old_value=current_value,
                new_value=new_value,
                improvement=improvement,
                confidence=confidence,
                metadata={
                    "signal_type": signal.signal_type.value,
                    "signal_value": signal.signal_value,
                    "adjustment": weighted_adjustment,
                    "learning_rate": self.learning_rate
                }
            )
            
        except Exception as e:
            self.logger.error("link_confidence_update_failed", error=str(e), signal_id=signal.id)
            raise ServiceError(f"Failed to update link confidence: {str(e)}")
    
    async def validate_update(self, old_value: float, new_value: float) -> bool:
        """Validate link confidence update."""
        # Simple validation: ensure change is not too drastic
        change_magnitude = abs(new_value - old_value)
        return change_magnitude <= self.validation_threshold


class ChunkRankingLearningAlgorithm(BaseLearningAlgorithm):
    """Learning algorithm for chunk ranking improvement."""
    
    def __init__(self, algorithm_id: uuid.UUID, learning_rate: float = 0.01):
        super().__init__(algorithm_id, learning_rate)
        self.ranking_factors = {
            "click_through": 0.3,
            "dwell_time": 0.25,
            "explicit_rating": 0.25,
            "bounce_rate": -0.2
        }
    
    async def update(self, signal: FeedbackSignal, current_value: float) -> LearningUpdate:
        """Update chunk ranking score based on user interactions."""
        try:
            # Calculate ranking adjustment based on signal type
            if signal.signal_type == FeedbackSignalType.CLICK_THROUGH:
                # Higher click-through rate improves ranking
                factor = self.ranking_factors["click_through"]
                adjustment = factor * self.learning_rate * signal.signal_value
            elif signal.signal_type == FeedbackSignalType.DWELL_TIME:
                # Longer dwell time improves ranking
                normalized_dwell = min(signal.signal_value / 120, 1.0)  # Normalize to 2 minutes
                factor = self.ranking_factors["dwell_time"]
                adjustment = factor * self.learning_rate * normalized_dwell
            elif signal.signal_type == FeedbackSignalType.EXPLICIT_RATING:
                # Direct rating feedback
                normalized_rating = (signal.signal_value - 3) / 2
                factor = self.ranking_factors["explicit_rating"]
                adjustment = factor * self.learning_rate * normalized_rating
            elif signal.signal_type == FeedbackSignalType.BOUNCE_RATE:
                # High bounce rate hurts ranking
                factor = self.ranking_factors["bounce_rate"]
                adjustment = factor * self.learning_rate * signal.signal_value
            else:
                adjustment = 0.0
            
            # Apply signal strength and confidence
            weighted_adjustment = adjustment * signal.signal_strength * signal.signal_confidence
            
            # Calculate new ranking score
            new_value = self._apply_bounds(current_value + weighted_adjustment)
            
            # Calculate improvement and confidence
            improvement = new_value - current_value
            confidence = self._calculate_confidence(signal.signal_strength, signal.signal_confidence)
            
            self.logger.info(
                "chunk_ranking_updated",
                target_id=signal.target_id,
                old_value=current_value,
                new_value=new_value,
                improvement=improvement,
                signal_type=signal.signal_type
            )
            
            return LearningUpdate(
                algorithm_id=self.algorithm_id,
                old_value=current_value,
                new_value=new_value,
                improvement=improvement,
                confidence=confidence,
                metadata={
                    "signal_type": signal.signal_type.value,
                    "signal_value": signal.signal_value,
                    "adjustment": weighted_adjustment,
                    "ranking_factors": self.ranking_factors
                }
            )
            
        except Exception as e:
            self.logger.error("chunk_ranking_update_failed", error=str(e), signal_id=signal.id)
            raise ServiceError(f"Failed to update chunk ranking: {str(e)}")
    
    async def validate_update(self, old_value: float, new_value: float) -> bool:
        """Validate chunk ranking update."""
        # Ensure ranking score stays within reasonable bounds
        change_magnitude = abs(new_value - old_value)
        return change_magnitude <= self.validation_threshold and 0.0 <= new_value <= 1.0


class QueryExpansionLearningAlgorithm(BaseLearningAlgorithm):
    """Learning algorithm for query expansion optimization."""
    
    def __init__(self, algorithm_id: uuid.UUID, learning_rate: float = 0.01):
        super().__init__(algorithm_id, learning_rate)
        self.expansion_terms = {}  # term -> effectiveness score
        self.query_patterns = {}   # pattern -> success rate
    
    async def update(self, signal: FeedbackSignal, current_value: float) -> LearningUpdate:
        """Update query expansion effectiveness based on search success."""
        try:
            # Extract query context and expansion terms
            query_context = signal.query_context or ""
            signal_metadata = signal.signal_metadata or {}
            expansion_terms = signal_metadata.get("expansion_terms", [])
            
            # Calculate effectiveness based on signal type
            if signal.signal_type == FeedbackSignalType.CONVERSION_RATE:
                # High conversion rate indicates successful expansion
                effectiveness = signal.signal_value
            elif signal.signal_type == FeedbackSignalType.CLICK_THROUGH:
                # Click-through rate indicates relevance
                effectiveness = signal.signal_value
            elif signal.signal_type == FeedbackSignalType.EXPLICIT_RATING:
                # Direct feedback on search results
                effectiveness = (signal.signal_value - 1) / 4  # Normalize 1-5 to 0-1
            else:
                effectiveness = 0.5  # Neutral
            
            # Update expansion term effectiveness
            for term in expansion_terms:
                if term not in self.expansion_terms:
                    self.expansion_terms[term] = 0.5  # Start neutral
                
                # Apply exponential moving average
                alpha = self.learning_rate
                self.expansion_terms[term] = (
                    alpha * effectiveness + (1 - alpha) * self.expansion_terms[term]
                )
            
            # Calculate overall improvement
            new_value = self._apply_bounds(current_value + self.learning_rate * (effectiveness - 0.5))
            improvement = new_value - current_value
            confidence = self._calculate_confidence(signal.signal_strength, signal.signal_confidence)
            
            self.logger.info(
                "query_expansion_updated",
                query_context=query_context[:100],  # Truncate for logging
                expansion_terms=expansion_terms,
                effectiveness=effectiveness,
                improvement=improvement
            )
            
            return LearningUpdate(
                algorithm_id=self.algorithm_id,
                old_value=current_value,
                new_value=new_value,
                improvement=improvement,
                confidence=confidence,
                metadata={
                    "expansion_terms": expansion_terms,
                    "effectiveness": effectiveness,
                    "term_scores": {term: self.expansion_terms.get(term, 0.5) for term in expansion_terms}
                }
            )
            
        except Exception as e:
            self.logger.error("query_expansion_update_failed", error=str(e), signal_id=signal.id)
            raise ServiceError(f"Failed to update query expansion: {str(e)}")
    
    async def validate_update(self, old_value: float, new_value: float) -> bool:
        """Validate query expansion update."""
        # Ensure expansion effectiveness stays reasonable
        change_magnitude = abs(new_value - old_value)
        return change_magnitude <= self.validation_threshold


class NegativeFeedbackHandler(BaseLearningAlgorithm):
    """Handler for negative feedback and penalization."""
    
    def __init__(self, algorithm_id: uuid.UUID, learning_rate: float = 0.02):
        super().__init__(algorithm_id, learning_rate)
        self.penalization_factors = {
            "explicit_rating": 0.4,  # Strong penalization for bad ratings
            "bounce_rate": 0.3,      # Moderate penalization for bounces
            "dwell_time": 0.2,       # Light penalization for short dwell
            "click_through": 0.1     # Minimal penalization for no clicks
        }
    
    async def update(self, signal: FeedbackSignal, current_value: float) -> LearningUpdate:
        """Apply negative feedback penalization."""
        try:
            # Determine if feedback is negative
            is_negative = False
            severity = 0.0
            
            if signal.signal_type == FeedbackSignalType.EXPLICIT_RATING:
                is_negative = signal.signal_value < 3
                severity = (3 - signal.signal_value) / 2 if is_negative else 0
            elif signal.signal_type == FeedbackSignalType.BOUNCE_RATE:
                is_negative = signal.signal_value > 0.7
                severity = signal.signal_value if is_negative else 0
            elif signal.signal_type == FeedbackSignalType.DWELL_TIME:
                is_negative = signal.signal_value < 10  # Less than 10 seconds
                severity = (10 - signal.signal_value) / 10 if is_negative else 0
            elif signal.signal_type == FeedbackSignalType.CLICK_THROUGH:
                is_negative = signal.signal_value < 0.1
                severity = (0.1 - signal.signal_value) / 0.1 if is_negative else 0
            
            if not is_negative:
                # No penalization needed
                return LearningUpdate(
                    algorithm_id=self.algorithm_id,
                    old_value=current_value,
                    new_value=current_value,
                    improvement=0.0,
                    confidence=1.0,
                    metadata={"penalization_applied": False}
                )
            
            # Calculate penalization
            factor = self.penalization_factors.get(signal.signal_type.value, 0.1)
            penalization = factor * self.learning_rate * severity
            
            # Apply signal strength and confidence
            weighted_penalization = penalization * signal.signal_strength * signal.signal_confidence
            
            # Apply penalization (reduce value)
            new_value = self._apply_bounds(current_value - weighted_penalization)
            
            improvement = new_value - current_value  # Will be negative
            confidence = self._calculate_confidence(signal.signal_strength, signal.signal_confidence)
            
            self.logger.info(
                "negative_feedback_applied",
                target_id=signal.target_id,
                signal_type=signal.signal_type,
                severity=severity,
                penalization=weighted_penalization,
                old_value=current_value,
                new_value=new_value
            )
            
            return LearningUpdate(
                algorithm_id=self.algorithm_id,
                old_value=current_value,
                new_value=new_value,
                improvement=improvement,
                confidence=confidence,
                metadata={
                    "penalization_applied": True,
                    "severity": severity,
                    "penalization": weighted_penalization,
                    "signal_type": signal.signal_type.value
                }
            )
            
        except Exception as e:
            self.logger.error("negative_feedback_handling_failed", error=str(e), signal_id=signal.id)
            raise ServiceError(f"Failed to handle negative feedback: {str(e)}")
    
    async def validate_update(self, old_value: float, new_value: float) -> bool:
        """Validate negative feedback penalization."""
        # Ensure penalization doesn't reduce value below minimum threshold
        return new_value >= 0.1  # Minimum quality threshold


class LearningService:
    """Core service for managing learning algorithms."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logger.bind(service="learning_service")
        self.algorithms = {}  # Cache for algorithm instances
    
    async def initialize_algorithms(self, tenant_id: uuid.UUID):
        """Initialize learning algorithms for a tenant."""
        try:
            # Get all active algorithms for tenant
            algorithms = self.db.query(LearningAlgorithm).filter(
                and_(
                    LearningAlgorithm.tenant_id == tenant_id,
                    LearningAlgorithm.is_enabled == True,
                    LearningAlgorithm.status == LearningStatus.ACTIVE
                )
            ).all()
            
            for algorithm in algorithms:
                # Create algorithm instance based on type
                if algorithm.algorithm_type == LearningAlgorithmType.LINK_CONFIDENCE:
                    instance = LinkConfidenceLearningAlgorithm(
                        algorithm.id, algorithm.learning_rate
                    )
                elif algorithm.algorithm_type == LearningAlgorithmType.CHUNK_RANKING:
                    instance = ChunkRankingLearningAlgorithm(
                        algorithm.id, algorithm.learning_rate
                    )
                elif algorithm.algorithm_type == LearningAlgorithmType.QUERY_EXPANSION:
                    instance = QueryExpansionLearningAlgorithm(
                        algorithm.id, algorithm.learning_rate
                    )
                elif algorithm.algorithm_type == LearningAlgorithmType.NEGATIVE_FEEDBACK:
                    instance = NegativeFeedbackHandler(
                        algorithm.id, algorithm.learning_rate
                    )
                else:
                    continue
                
                self.algorithms[algorithm.id] = instance
            
            self.logger.info(
                "algorithms_initialized",
                tenant_id=tenant_id,
                algorithm_count=len(self.algorithms)
            )
            
        except Exception as e:
            self.logger.error("algorithm_initialization_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to initialize algorithms: {str(e)}")
    
    async def process_feedback_signal(
        self,
        tenant_id: uuid.UUID,
        signal: FeedbackSignal
    ) -> List[LearningUpdate]:
        """Process a feedback signal through relevant learning algorithms."""
        try:
            updates = []
            
            # Get relevant algorithms for this signal type
            relevant_algorithms = self._get_relevant_algorithms(signal.signal_type)
            
            for algorithm_id in relevant_algorithms:
                if algorithm_id not in self.algorithms:
                    continue
                
                algorithm = self.algorithms[algorithm_id]
                
                # Get current value for the target
                current_value = await self._get_current_value(signal.target_type, signal.target_id)
                
                # Apply learning update
                update = await algorithm.update(signal, current_value)
                
                # Validate update
                if await algorithm.validate_update(update.old_value, update.new_value):
                    # Apply update to target
                    await self._apply_update(signal.target_type, signal.target_id, update.new_value)
                    updates.append(update)
                    
                    # Record performance metric
                    await self._record_performance_metric(algorithm_id, update)
                else:
                    self.logger.warning(
                        "learning_update_validation_failed",
                        algorithm_id=algorithm_id,
                        old_value=update.old_value,
                        new_value=update.new_value
                    )
            
            # Mark signal as processed
            signal.is_processed = True
            signal.processed_at = datetime.now()
            self.db.commit()
            
            self.logger.info(
                "feedback_signal_processed",
                signal_id=signal.id,
                updates_applied=len(updates),
                signal_type=signal.signal_type
            )
            
            return updates
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("feedback_signal_processing_failed", error=str(e), signal_id=signal.id)
            raise ServiceError(f"Failed to process feedback signal: {str(e)}")
    
    def _get_relevant_algorithms(self, signal_type: FeedbackSignalType) -> List[uuid.UUID]:
        """Get algorithms relevant to a signal type."""
        # Map signal types to algorithm types
        signal_algorithm_map = {
            FeedbackSignalType.CLICK_THROUGH: [
                LearningAlgorithmType.LINK_CONFIDENCE,
                LearningAlgorithmType.CHUNK_RANKING
            ],
            FeedbackSignalType.DWELL_TIME: [
                LearningAlgorithmType.CHUNK_RANKING,
                LearningAlgorithmType.NEGATIVE_FEEDBACK
            ],
            FeedbackSignalType.EXPLICIT_RATING: [
                LearningAlgorithmType.LINK_CONFIDENCE,
                LearningAlgorithmType.CHUNK_RANKING,
                LearningAlgorithmType.NEGATIVE_FEEDBACK
            ],
            FeedbackSignalType.BOUNCE_RATE: [
                LearningAlgorithmType.NEGATIVE_FEEDBACK,
                LearningAlgorithmType.CHUNK_RANKING
            ],
            FeedbackSignalType.CONVERSION_RATE: [
                LearningAlgorithmType.QUERY_EXPANSION,
                LearningAlgorithmType.CHUNK_RANKING
            ],
            FeedbackSignalType.CORRECTION_FEEDBACK: [
                LearningAlgorithmType.CONTENT_QUALITY,
                LearningAlgorithmType.NEGATIVE_FEEDBACK
            ]
        }
        
        relevant_types = signal_algorithm_map.get(signal_type, [])
        
        # Return algorithm IDs that match the types
        return [
            alg_id for alg_id, algorithm in self.algorithms.items()
            if any(alg.algorithm_type in relevant_types 
                   for alg in [self.db.query(LearningAlgorithm).get(alg_id)] if alg)
        ]
    
    async def _get_current_value(self, target_type: str, target_id: uuid.UUID) -> float:
        """Get current value for learning target."""
        # This would integrate with existing systems to get current scores
        # For now, return a default value
        return 0.5
    
    async def _apply_update(self, target_type: str, target_id: uuid.UUID, new_value: float):
        """Apply learning update to target."""
        # This would update the actual target (link confidence, ranking score, etc.)
        # Implementation depends on integration with existing systems
        pass
    
    async def _record_performance_metric(self, algorithm_id: uuid.UUID, update: LearningUpdate):
        """Record performance metric for learning update."""
        try:
            metric = LearningPerformanceMetric(
                tenant_id=self.db.query(LearningAlgorithm).get(algorithm_id).tenant_id,
                algorithm_id=algorithm_id,
                metric_name="learning_improvement",
                metric_value=update.improvement,
                metric_type="improvement",
                measurement_period_start=datetime.now() - timedelta(minutes=1),
                measurement_period_end=datetime.now(),
                sample_size=1,
                metric_metadata=update.metadata
            )
            
            self.db.add(metric)
            
        except Exception as e:
            self.logger.error("performance_metric_recording_failed", error=str(e), algorithm_id=algorithm_id)


# Dependency injection
_learning_service_instance = None


def get_learning_service(db_session: Session = None) -> LearningService:
    """Get learning service instance with dependency injection."""
    global _learning_service_instance
    if _learning_service_instance is None:
        if db_session is None:
            db_session = get_database_session()
        _learning_service_instance = LearningService(db_session)
    return _learning_service_instance
