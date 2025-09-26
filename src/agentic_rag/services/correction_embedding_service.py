"""
Correction Re-embedding Service for Sprint 6 Story 6-02

This service handles automatic re-embedding of corrected content to ensure
search quality improvements are reflected in the embedding space.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from agentic_rag.models.corrections import (
    ContentCorrection,
    ContentVersion,
    CorrectionImpact,
    CorrectionStatus
)
from agentic_rag.models.database import DocumentChunk, ChunkEmbedding
from agentic_rag.services.embedding_service import EmbeddingService, get_embedding_service
from agentic_rag.database.connection import get_database_session
from agentic_rag.api.exceptions import ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class ReEmbeddingResult:
    """Result of re-embedding operation."""
    correction_id: uuid.UUID
    chunk_id: uuid.UUID
    version_id: uuid.UUID
    embedding_id: uuid.UUID
    processing_time_seconds: float
    quality_improvement: Optional[float]


@dataclass
class ReEmbeddingBatch:
    """Batch of corrections for re-embedding."""
    corrections: List[ContentCorrection]
    total_count: int
    estimated_processing_time: float


class CorrectionEmbeddingService:
    """Service for managing re-embedding of corrected content."""
    
    def __init__(self, db_session: Session, embedding_service: EmbeddingService = None):
        self.db = db_session
        self.embedding_service = embedding_service or get_embedding_service()
        self.logger = logger.bind(service="correction_embedding_service")
    
    async def trigger_re_embedding(
        self,
        correction_id: uuid.UUID,
        tenant_id: uuid.UUID
    ) -> ReEmbeddingResult:
        """Trigger re-embedding for a specific implemented correction."""
        try:
            start_time = datetime.now()
            
            # Get implemented correction with active version
            correction = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.id == correction_id,
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.status == CorrectionStatus.IMPLEMENTED
                )
            ).first()
            
            if not correction:
                raise ServiceError("Correction not found or not implemented")
            
            # Get the active version for this correction
            active_version = self.db.query(ContentVersion).filter(
                and_(
                    ContentVersion.correction_id == correction_id,
                    ContentVersion.is_active == True
                )
            ).first()
            
            if not active_version:
                raise ServiceError("No active version found for correction")
            
            # Get chunk information
            chunk = self.db.query(DocumentChunk).filter(
                DocumentChunk.id == correction.chunk_id
            ).first()
            
            if not chunk:
                raise ServiceError("Chunk not found")
            
            # Generate new embedding for corrected content
            embedding_vector = await self.embedding_service.generate_embedding(
                text=active_version.content,
                tenant_id=tenant_id
            )
            
            # Create new embedding record
            new_embedding = ChunkEmbedding(
                chunk_id=chunk.id,
                tenant_id=tenant_id,
                embedding_vector=embedding_vector,
                embedding_model=self.embedding_service.model_name,
                created_at=datetime.now(),
                metadata={
                    "correction_id": str(correction_id),
                    "version_id": str(active_version.id),
                    "correction_type": correction.correction_type.value,
                    "re_embedded": True
                }
            )
            
            # Deactivate old embeddings for this chunk
            self.db.query(ChunkEmbedding).filter(
                and_(
                    ChunkEmbedding.chunk_id == chunk.id,
                    ChunkEmbedding.tenant_id == tenant_id
                )
            ).update({"is_active": False})
            
            # Activate new embedding
            new_embedding.is_active = True
            self.db.add(new_embedding)
            
            # Update correction impact tracking
            impact = self.db.query(CorrectionImpact).filter(
                CorrectionImpact.correction_id == correction_id
            ).first()
            
            if impact:
                impact.re_embedding_completed = True
                impact.re_embedding_at = datetime.now()
            else:
                impact = CorrectionImpact(
                    tenant_id=tenant_id,
                    correction_id=correction_id,
                    re_embedding_completed=True,
                    re_embedding_at=datetime.now()
                )
                self.db.add(impact)
            
            self.db.commit()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate quality improvement (simplified)
            quality_improvement = await self._estimate_quality_improvement(correction, active_version)
            
            self.logger.info(
                "re_embedding_completed",
                correction_id=correction_id,
                chunk_id=chunk.id,
                version_id=active_version.id,
                embedding_id=new_embedding.id,
                processing_time=processing_time
            )
            
            return ReEmbeddingResult(
                correction_id=correction_id,
                chunk_id=chunk.id,
                version_id=active_version.id,
                embedding_id=new_embedding.id,
                processing_time_seconds=processing_time,
                quality_improvement=quality_improvement
            )
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("re_embedding_failed", error=str(e), correction_id=correction_id)
            raise ServiceError(f"Failed to re-embed correction: {str(e)}")
    
    async def batch_re_embedding(
        self,
        tenant_id: uuid.UUID,
        max_batch_size: int = 10
    ) -> List[ReEmbeddingResult]:
        """Process batch re-embedding for pending corrections."""
        try:
            # Get corrections that need re-embedding
            pending_corrections = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.status == CorrectionStatus.IMPLEMENTED
                )
            ).join(CorrectionImpact, ContentCorrection.id == CorrectionImpact.correction_id, isouter=True).filter(
                or_(
                    CorrectionImpact.re_embedding_completed == False,
                    CorrectionImpact.re_embedding_completed.is_(None)
                )
            ).limit(max_batch_size).all()
            
            if not pending_corrections:
                self.logger.info("no_pending_re_embeddings", tenant_id=tenant_id)
                return []
            
            results = []
            
            for correction in pending_corrections:
                try:
                    result = await self.trigger_re_embedding(correction.id, tenant_id)
                    results.append(result)
                    
                    # Add small delay between embeddings to avoid rate limits
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(
                        "batch_re_embedding_item_failed",
                        correction_id=correction.id,
                        error=str(e)
                    )
                    continue
            
            self.logger.info(
                "batch_re_embedding_completed",
                tenant_id=tenant_id,
                processed_count=len(results),
                total_pending=len(pending_corrections)
            )
            
            return results
            
        except Exception as e:
            self.logger.error("batch_re_embedding_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to process batch re-embedding: {str(e)}")
    
    async def schedule_re_embedding(
        self,
        correction_id: uuid.UUID,
        tenant_id: uuid.UUID,
        delay_minutes: int = 5
    ):
        """Schedule re-embedding for later processing."""
        try:
            # In a production system, this would use a task queue like Celery
            # For now, we'll simulate scheduling by adding metadata
            
            correction = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.id == correction_id,
                    ContentCorrection.tenant_id == tenant_id
                )
            ).first()
            
            if not correction:
                raise ServiceError("Correction not found")
            
            # Update correction metadata with scheduling information
            if not correction.correction_metadata:
                correction.correction_metadata = {}
            
            correction.correction_metadata.update({
                "re_embedding_scheduled": True,
                "scheduled_at": datetime.now().isoformat(),
                "scheduled_for": (datetime.now() + timedelta(minutes=delay_minutes)).isoformat(),
                "delay_minutes": delay_minutes
            })
            
            self.db.commit()
            
            self.logger.info(
                "re_embedding_scheduled",
                correction_id=correction_id,
                delay_minutes=delay_minutes,
                scheduled_for=correction.correction_metadata["scheduled_for"]
            )
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("re_embedding_scheduling_failed", error=str(e), correction_id=correction_id)
            raise ServiceError(f"Failed to schedule re-embedding: {str(e)}")
    
    async def get_re_embedding_status(
        self,
        correction_id: uuid.UUID,
        tenant_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Get re-embedding status for a correction."""
        try:
            correction = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.id == correction_id,
                    ContentCorrection.tenant_id == tenant_id
                )
            ).first()
            
            if not correction:
                raise ServiceError("Correction not found")
            
            impact = self.db.query(CorrectionImpact).filter(
                CorrectionImpact.correction_id == correction_id
            ).first()
            
            status = {
                "correction_id": str(correction_id),
                "status": correction.status.value,
                "re_embedding_completed": False,
                "re_embedding_at": None,
                "scheduled": False,
                "quality_improvement": None
            }
            
            if impact:
                status.update({
                    "re_embedding_completed": impact.re_embedding_completed,
                    "re_embedding_at": impact.re_embedding_at.isoformat() if impact.re_embedding_at else None,
                    "quality_improvement": impact.search_improvement
                })
            
            # Check scheduling information
            if correction.correction_metadata:
                status["scheduled"] = correction.correction_metadata.get("re_embedding_scheduled", False)
                if status["scheduled"]:
                    status["scheduled_for"] = correction.correction_metadata.get("scheduled_for")
            
            return status
            
        except Exception as e:
            self.logger.error("re_embedding_status_failed", error=str(e), correction_id=correction_id)
            raise ServiceError(f"Failed to get re-embedding status: {str(e)}")
    
    async def get_re_embedding_queue_stats(self, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Get statistics about the re-embedding queue."""
        try:
            # Count pending re-embeddings
            pending_count = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.status == CorrectionStatus.IMPLEMENTED
                )
            ).join(CorrectionImpact, ContentCorrection.id == CorrectionImpact.correction_id, isouter=True).filter(
                or_(
                    CorrectionImpact.re_embedding_completed == False,
                    CorrectionImpact.re_embedding_completed.is_(None)
                )
            ).count()
            
            # Count completed re-embeddings
            completed_count = self.db.query(CorrectionImpact).filter(
                and_(
                    CorrectionImpact.tenant_id == tenant_id,
                    CorrectionImpact.re_embedding_completed == True
                )
            ).count()
            
            # Get average processing time
            avg_processing_time = self.db.query(
                func.avg(
                    func.extract('epoch', CorrectionImpact.updated_at - CorrectionImpact.created_at)
                )
            ).filter(
                and_(
                    CorrectionImpact.tenant_id == tenant_id,
                    CorrectionImpact.re_embedding_completed == True
                )
            ).scalar()
            
            return {
                "pending_re_embeddings": pending_count,
                "completed_re_embeddings": completed_count,
                "average_processing_time_seconds": float(avg_processing_time) if avg_processing_time else 0.0,
                "estimated_queue_time_minutes": pending_count * 0.5  # Simplified estimation
            }
            
        except Exception as e:
            self.logger.error("re_embedding_queue_stats_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to get re-embedding queue statistics: {str(e)}")
    
    # Private helper methods
    
    async def _estimate_quality_improvement(
        self,
        correction: ContentCorrection,
        version: ContentVersion
    ) -> Optional[float]:
        """Estimate quality improvement from correction."""
        # Simplified quality improvement estimation
        # In a real system, this would use more sophisticated metrics
        
        base_improvement = 0.1  # 10% base improvement
        
        # Adjust based on correction type
        type_multipliers = {
            "factual": 1.5,
            "completeness": 1.3,
            "clarity": 1.1,
            "formatting": 0.8,
            "grammar": 0.7,
            "terminology": 1.2
        }
        
        multiplier = type_multipliers.get(correction.correction_type.value, 1.0)
        
        # Adjust based on quality scores
        if correction.quality_score:
            multiplier *= correction.quality_score
        
        return min(base_improvement * multiplier, 1.0)


# Dependency injection
_correction_embedding_service_instance = None


def get_correction_embedding_service(
    db_session: Session = None,
    embedding_service: EmbeddingService = None
) -> CorrectionEmbeddingService:
    """Get correction embedding service instance with dependency injection."""
    global _correction_embedding_service_instance
    if _correction_embedding_service_instance is None:
        if db_session is None:
            db_session = get_database_session()
        _correction_embedding_service_instance = CorrectionEmbeddingService(db_session, embedding_service)
    return _correction_embedding_service_instance
