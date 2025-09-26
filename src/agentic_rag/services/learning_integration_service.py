"""
Learning Integration Service for Sprint 6 Story 6-03

This service integrates learning algorithms with feedback collection and
correction systems to enable continuous learning and improvement.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from agentic_rag.models.learning import (
    FeedbackSignal,
    LearningAlgorithm,
    LearningSession,
    FeedbackSignalType,
    LearningAlgorithmType
)
from agentic_rag.models.feedback import UserFeedbackSubmission, FeedbackType
from agentic_rag.models.corrections import ContentCorrection, CorrectionStatus
from agentic_rag.services.learning_service import LearningService, get_learning_service
from agentic_rag.services.feedback_service import FeedbackService, get_feedback_service
from agentic_rag.database.connection import get_database_session
from agentic_rag.api.exceptions import ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class LearningIntegrationResult:
    """Result of learning integration processing."""
    signals_processed: int
    algorithms_updated: int
    improvements_applied: int
    processing_time_seconds: float
    errors: List[str]


@dataclass
class FeedbackSignalBatch:
    """Batch of feedback signals for processing."""
    signals: List[FeedbackSignal]
    total_count: int
    batch_id: str
    created_at: datetime


class LearningIntegrationService:
    """Service for integrating learning algorithms with feedback systems."""
    
    def __init__(
        self,
        db_session: Session,
        learning_service: LearningService = None,
        feedback_service: FeedbackService = None
    ):
        self.db = db_session
        self.learning_service = learning_service or get_learning_service(db_session)
        self.feedback_service = feedback_service or get_feedback_service(db_session)
        self.logger = logger.bind(service="learning_integration_service")
    
    async def process_feedback_for_learning(
        self,
        tenant_id: uuid.UUID,
        batch_size: int = 100
    ) -> LearningIntegrationResult:
        """Process feedback submissions to generate learning signals."""
        try:
            start_time = datetime.now()
            signals_processed = 0
            algorithms_updated = 0
            improvements_applied = 0
            errors = []
            
            # Get unprocessed feedback submissions
            feedback_submissions = self.db.query(UserFeedbackSubmission).filter(
                and_(
                    UserFeedbackSubmission.tenant_id == tenant_id,
                    UserFeedbackSubmission.status == "completed"
                )
            ).limit(batch_size).all()
            
            # Convert feedback to learning signals
            for feedback in feedback_submissions:
                try:
                    signals = await self._convert_feedback_to_signals(feedback)
                    
                    for signal in signals:
                        self.db.add(signal)
                        signals_processed += 1
                    
                    # Mark feedback as processed for learning
                    if not feedback.metadata:
                        feedback.metadata = {}
                    feedback.metadata["learning_processed"] = True
                    feedback.metadata["learning_processed_at"] = datetime.now().isoformat()
                    
                except Exception as e:
                    error_msg = f"Failed to process feedback {feedback.id}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error("feedback_to_signal_conversion_failed", 
                                    feedback_id=feedback.id, error=str(e))
            
            self.db.commit()
            
            # Process learning signals
            if signals_processed > 0:
                learning_result = await self._process_learning_signals(tenant_id)
                algorithms_updated = learning_result.get("algorithms_updated", 0)
                improvements_applied = learning_result.get("improvements_applied", 0)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(
                "feedback_learning_processing_completed",
                tenant_id=tenant_id,
                signals_processed=signals_processed,
                algorithms_updated=algorithms_updated,
                improvements_applied=improvements_applied,
                processing_time=processing_time,
                error_count=len(errors)
            )
            
            return LearningIntegrationResult(
                signals_processed=signals_processed,
                algorithms_updated=algorithms_updated,
                improvements_applied=improvements_applied,
                processing_time_seconds=processing_time,
                errors=errors
            )
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("feedback_learning_processing_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to process feedback for learning: {str(e)}")
    
    async def process_corrections_for_learning(
        self,
        tenant_id: uuid.UUID,
        batch_size: int = 50
    ) -> LearningIntegrationResult:
        """Process content corrections to generate learning signals."""
        try:
            start_time = datetime.now()
            signals_processed = 0
            algorithms_updated = 0
            improvements_applied = 0
            errors = []
            
            # Get implemented corrections that haven't been processed for learning
            corrections = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.status == CorrectionStatus.IMPLEMENTED
                )
            ).limit(batch_size).all()
            
            # Convert corrections to learning signals
            for correction in corrections:
                try:
                    signals = await self._convert_correction_to_signals(correction)
                    
                    for signal in signals:
                        self.db.add(signal)
                        signals_processed += 1
                    
                    # Mark correction as processed for learning
                    if not correction.correction_metadata:
                        correction.correction_metadata = {}
                    correction.correction_metadata["learning_processed"] = True
                    correction.correction_metadata["learning_processed_at"] = datetime.now().isoformat()
                    
                except Exception as e:
                    error_msg = f"Failed to process correction {correction.id}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error("correction_to_signal_conversion_failed", 
                                    correction_id=correction.id, error=str(e))
            
            self.db.commit()
            
            # Process learning signals
            if signals_processed > 0:
                learning_result = await self._process_learning_signals(tenant_id)
                algorithms_updated = learning_result.get("algorithms_updated", 0)
                improvements_applied = learning_result.get("improvements_applied", 0)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(
                "correction_learning_processing_completed",
                tenant_id=tenant_id,
                signals_processed=signals_processed,
                algorithms_updated=algorithms_updated,
                improvements_applied=improvements_applied,
                processing_time=processing_time,
                error_count=len(errors)
            )
            
            return LearningIntegrationResult(
                signals_processed=signals_processed,
                algorithms_updated=algorithms_updated,
                improvements_applied=improvements_applied,
                processing_time_seconds=processing_time,
                errors=errors
            )
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("correction_learning_processing_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to process corrections for learning: {str(e)}")
    
    async def create_feedback_signal(
        self,
        tenant_id: uuid.UUID,
        signal_type: FeedbackSignalType,
        target_type: str,
        target_id: uuid.UUID,
        signal_value: float,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        query_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackSignal:
        """Create a feedback signal for learning algorithms."""
        try:
            signal = FeedbackSignal(
                tenant_id=tenant_id,
                signal_type=signal_type,
                target_type=target_type,
                target_id=target_id,
                signal_value=signal_value,
                user_id=user_id,
                session_id=session_id,
                query_context=query_context,
                signal_metadata=metadata
            )
            
            self.db.add(signal)
            self.db.commit()
            self.db.refresh(signal)
            
            self.logger.info(
                "feedback_signal_created",
                signal_id=signal.id,
                signal_type=signal_type,
                target_type=target_type,
                target_id=target_id
            )
            
            return signal
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("feedback_signal_creation_failed", error=str(e))
            raise ServiceError(f"Failed to create feedback signal: {str(e)}")
    
    async def batch_process_learning_signals(
        self,
        tenant_id: uuid.UUID,
        max_signals: int = 1000
    ) -> Dict[str, Any]:
        """Batch process learning signals for improved performance."""
        try:
            # Initialize learning algorithms
            await self.learning_service.initialize_algorithms(tenant_id)
            
            # Get unprocessed signals
            signals = self.db.query(FeedbackSignal).filter(
                and_(
                    FeedbackSignal.tenant_id == tenant_id,
                    FeedbackSignal.is_processed == False
                )
            ).order_by(FeedbackSignal.created_at).limit(max_signals).all()
            
            if not signals:
                return {
                    "signals_processed": 0,
                    "algorithms_updated": 0,
                    "improvements_applied": 0
                }
            
            # Group signals by target for efficient processing
            signals_by_target = {}
            for signal in signals:
                target_key = f"{signal.target_type}:{signal.target_id}"
                if target_key not in signals_by_target:
                    signals_by_target[target_key] = []
                signals_by_target[target_key].append(signal)
            
            # Process signals in batches
            total_updates = 0
            algorithms_updated = set()
            
            for target_key, target_signals in signals_by_target.items():
                try:
                    for signal in target_signals:
                        updates = await self.learning_service.process_feedback_signal(
                            tenant_id, signal
                        )
                        total_updates += len(updates)
                        algorithms_updated.update(update.algorithm_id for update in updates)
                        
                        # Add small delay to prevent overwhelming the system
                        await asyncio.sleep(0.01)
                        
                except Exception as e:
                    self.logger.error(
                        "signal_batch_processing_error",
                        target_key=target_key,
                        error=str(e)
                    )
                    continue
            
            self.logger.info(
                "batch_learning_processing_completed",
                tenant_id=tenant_id,
                signals_processed=len(signals),
                algorithms_updated=len(algorithms_updated),
                improvements_applied=total_updates
            )
            
            return {
                "signals_processed": len(signals),
                "algorithms_updated": len(algorithms_updated),
                "improvements_applied": total_updates
            }
            
        except Exception as e:
            self.logger.error("batch_learning_processing_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to batch process learning signals: {str(e)}")
    
    async def get_learning_insights(
        self,
        tenant_id: uuid.UUID,
        time_period_hours: int = 24
    ) -> Dict[str, Any]:
        """Get insights from learning algorithm performance."""
        try:
            start_time = datetime.now() - timedelta(hours=time_period_hours)
            
            # Get signal processing statistics
            signal_stats = self.db.query(
                FeedbackSignal.signal_type,
                func.count(FeedbackSignal.id).label('count'),
                func.avg(FeedbackSignal.signal_value).label('avg_value')
            ).filter(
                and_(
                    FeedbackSignal.tenant_id == tenant_id,
                    FeedbackSignal.created_at >= start_time,
                    FeedbackSignal.is_processed == True
                )
            ).group_by(FeedbackSignal.signal_type).all()
            
            # Get algorithm performance
            algorithm_performance = self.db.query(LearningAlgorithm).filter(
                and_(
                    LearningAlgorithm.tenant_id == tenant_id,
                    LearningAlgorithm.is_enabled == True
                )
            ).all()
            
            # Compile insights
            insights = {
                "time_period_hours": time_period_hours,
                "signal_statistics": {
                    stat.signal_type.value: {
                        "count": stat.count,
                        "average_value": float(stat.avg_value) if stat.avg_value else 0.0
                    }
                    for stat in signal_stats
                },
                "algorithm_performance": {
                    alg.algorithm_type.value: {
                        "accuracy_score": alg.accuracy_score,
                        "precision_score": alg.precision_score,
                        "recall_score": alg.recall_score,
                        "f1_score": alg.f1_score,
                        "last_trained": alg.last_trained_at.isoformat() if alg.last_trained_at else None,
                        "training_data_size": alg.training_data_size
                    }
                    for alg in algorithm_performance
                },
                "total_signals_processed": sum(stat.count for stat in signal_stats),
                "active_algorithms": len(algorithm_performance)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error("learning_insights_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to get learning insights: {str(e)}")
    
    # Private helper methods
    
    async def _convert_feedback_to_signals(self, feedback: UserFeedbackSubmission) -> List[FeedbackSignal]:
        """Convert feedback submission to learning signals."""
        signals = []
        
        try:
            # Map feedback types to signal types
            if feedback.feedback_type == FeedbackType.SEARCH_RESULT:
                # Create click-through and rating signals
                if feedback.rating is not None:
                    signals.append(FeedbackSignal(
                        tenant_id=feedback.tenant_id,
                        signal_type=FeedbackSignalType.EXPLICIT_RATING,
                        target_type="search_result",
                        target_id=feedback.target_id,
                        signal_value=float(feedback.rating),
                        user_id=feedback.user_id,
                        session_id=feedback.session_id,
                        signal_metadata={"feedback_id": str(feedback.id)}
                    ))
                
                # Infer click-through from rating
                if feedback.rating and feedback.rating >= 4:
                    signals.append(FeedbackSignal(
                        tenant_id=feedback.tenant_id,
                        signal_type=FeedbackSignalType.CLICK_THROUGH,
                        target_type="search_result",
                        target_id=feedback.target_id,
                        signal_value=1.0,
                        user_id=feedback.user_id,
                        session_id=feedback.session_id,
                        signal_metadata={"inferred_from_rating": True}
                    ))
            
            elif feedback.feedback_type == FeedbackType.LINK_QUALITY:
                # Create link confidence signal
                if feedback.rating is not None:
                    signals.append(FeedbackSignal(
                        tenant_id=feedback.tenant_id,
                        signal_type=FeedbackSignalType.EXPLICIT_RATING,
                        target_type="link",
                        target_id=feedback.target_id,
                        signal_value=float(feedback.rating),
                        user_id=feedback.user_id,
                        session_id=feedback.session_id,
                        signal_metadata={"feedback_id": str(feedback.id)}
                    ))
            
            elif feedback.feedback_type == FeedbackType.ANSWER_QUALITY:
                # Create answer quality signals
                if feedback.rating is not None:
                    signals.append(FeedbackSignal(
                        tenant_id=feedback.tenant_id,
                        signal_type=FeedbackSignalType.EXPLICIT_RATING,
                        target_type="answer",
                        target_id=feedback.target_id,
                        signal_value=float(feedback.rating),
                        user_id=feedback.user_id,
                        session_id=feedback.session_id,
                        query_context=feedback.context.get("query") if feedback.context else None,
                        signal_metadata={"feedback_id": str(feedback.id)}
                    ))
            
            return signals
            
        except Exception as e:
            self.logger.error("feedback_to_signal_conversion_failed", 
                            feedback_id=feedback.id, error=str(e))
            return []
    
    async def _convert_correction_to_signals(self, correction: ContentCorrection) -> List[FeedbackSignal]:
        """Convert content correction to learning signals."""
        signals = []
        
        try:
            # Create correction feedback signal
            signals.append(FeedbackSignal(
                tenant_id=correction.tenant_id,
                signal_type=FeedbackSignalType.CORRECTION_FEEDBACK,
                target_type="chunk",
                target_id=correction.chunk_id,
                signal_value=correction.quality_score or 0.8,  # Default quality improvement
                user_id=correction.user_id,
                signal_metadata={
                    "correction_id": str(correction.id),
                    "correction_type": correction.correction_type.value,
                    "confidence_score": correction.confidence_score,
                    "impact_score": correction.impact_score
                }
            ))
            
            # Create negative feedback signal for original content
            signals.append(FeedbackSignal(
                tenant_id=correction.tenant_id,
                signal_type=FeedbackSignalType.EXPLICIT_RATING,
                target_type="chunk",
                target_id=correction.chunk_id,
                signal_value=2.0,  # Low rating for content that needed correction
                user_id=correction.user_id,
                signal_metadata={
                    "correction_based": True,
                    "original_content_quality": "poor"
                }
            ))
            
            return signals
            
        except Exception as e:
            self.logger.error("correction_to_signal_conversion_failed", 
                            correction_id=correction.id, error=str(e))
            return []
    
    async def _process_learning_signals(self, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Process learning signals and return statistics."""
        try:
            # Get unprocessed signals
            signals = self.db.query(FeedbackSignal).filter(
                and_(
                    FeedbackSignal.tenant_id == tenant_id,
                    FeedbackSignal.is_processed == False
                )
            ).limit(100).all()  # Process in smaller batches
            
            algorithms_updated = set()
            total_improvements = 0
            
            for signal in signals:
                try:
                    updates = await self.learning_service.process_feedback_signal(tenant_id, signal)
                    total_improvements += len(updates)
                    algorithms_updated.update(update.algorithm_id for update in updates)
                except Exception as e:
                    self.logger.error("signal_processing_error", signal_id=signal.id, error=str(e))
                    continue
            
            return {
                "algorithms_updated": len(algorithms_updated),
                "improvements_applied": total_improvements
            }
            
        except Exception as e:
            self.logger.error("learning_signal_processing_failed", error=str(e))
            return {"algorithms_updated": 0, "improvements_applied": 0}


# Dependency injection
_learning_integration_service_instance = None


def get_learning_integration_service(
    db_session: Session = None,
    learning_service: LearningService = None,
    feedback_service: FeedbackService = None
) -> LearningIntegrationService:
    """Get learning integration service instance with dependency injection."""
    global _learning_integration_service_instance
    if _learning_integration_service_instance is None:
        if db_session is None:
            db_session = get_database_session()
        _learning_integration_service_instance = LearningIntegrationService(
            db_session, learning_service, feedback_service
        )
    return _learning_integration_service_instance
