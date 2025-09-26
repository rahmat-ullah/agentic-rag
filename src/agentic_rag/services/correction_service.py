"""
Content Correction Service for Sprint 6 Story 6-02

This service handles content correction submission, validation, versioning,
workflow management, and expert review processes.
"""

import uuid
import hashlib
import difflib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from agentic_rag.models.corrections import (
    ContentCorrection,
    ContentVersion,
    CorrectionReview,
    CorrectionWorkflow,
    CorrectionImpact,
    CorrectionType,
    CorrectionStatus,
    CorrectionPriority,
    ReviewDecision
)
from agentic_rag.models.database import DocumentChunk, User
from agentic_rag.schemas.corrections import (
    CorrectionSubmissionRequest,
    InlineEditRequest,
    ReviewSubmissionRequest,
    VersionComparisonRequest,
    CorrectionDetails,
    VersionDetails,
    CorrectionTypeEnum,
    CorrectionStatusEnum,
    ReviewDecisionEnum
)
from agentic_rag.schemas.base import PaginationParams, SortParams
from agentic_rag.database.connection import get_database_session
from agentic_rag.api.exceptions import ValidationError, ServiceError

logger = structlog.get_logger(__name__)


@dataclass
class CorrectionSubmissionResult:
    """Result of correction submission."""
    correction_id: uuid.UUID
    status: CorrectionStatusEnum
    workflow_id: uuid.UUID
    estimated_review_time: str
    next_steps: List[str]


@dataclass
class VersionComparisonResult:
    """Result of version comparison."""
    chunk_id: uuid.UUID
    version_1: VersionDetails
    version_2: VersionDetails
    differences: List[Dict[str, Any]]
    similarity_score: float
    change_summary: str


@dataclass
class CorrectionStats:
    """Correction system statistics."""
    total_corrections: int
    pending_corrections: int
    approved_corrections: int
    rejected_corrections: int
    average_review_time_hours: float
    correction_type_breakdown: Dict[str, int]
    quality_improvement_metrics: Dict[str, float]


class CorrectionService:
    """Service for managing content corrections and editing workflow."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logger.bind(service="correction_service")
    
    async def submit_correction(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        correction_data: CorrectionSubmissionRequest
    ) -> CorrectionSubmissionResult:
        """Submit a content correction for review."""
        try:
            # Validate chunk exists and user has permission
            chunk = await self._validate_chunk_access(tenant_id, correction_data.chunk_id, user_id)
            
            # Get current content for comparison
            current_content = await self._get_current_chunk_content(chunk)
            
            # Validate correction is meaningful
            await self._validate_correction_content(current_content, correction_data.corrected_content)
            
            # Create correction record
            correction = ContentCorrection(
                tenant_id=tenant_id,
                chunk_id=correction_data.chunk_id,
                user_id=user_id,
                original_content=current_content,
                corrected_content=correction_data.corrected_content,
                correction_reason=correction_data.correction_reason,
                correction_type=CorrectionType(correction_data.correction_type.value),
                priority=CorrectionPriority(correction_data.priority.value),
                confidence_score=correction_data.confidence_score,
                correction_metadata=correction_data.correction_metadata,
                source_references=correction_data.source_references
            )
            
            # Calculate impact score
            correction.impact_score = await self._calculate_impact_score(correction)
            
            self.db.add(correction)
            self.db.commit()
            self.db.refresh(correction)
            
            # Create workflow
            workflow = await self._create_correction_workflow(correction)
            
            # Create initial version if this is the first correction for this chunk
            await self._ensure_initial_version(chunk, current_content, user_id)
            
            # Assign reviewer if high priority
            if correction.priority in [CorrectionPriority.HIGH, CorrectionPriority.CRITICAL]:
                await self._assign_reviewer(workflow, correction)
            
            # Estimate review time
            estimated_time = self._estimate_review_time(correction.priority, correction.correction_type)
            
            # Determine next steps
            next_steps = self._get_next_steps(workflow.current_step)
            
            self.logger.info(
                "correction_submitted_successfully",
                correction_id=correction.id,
                chunk_id=correction.chunk_id,
                correction_type=correction.correction_type,
                priority=correction.priority,
                user_id=user_id
            )
            
            return CorrectionSubmissionResult(
                correction_id=correction.id,
                status=CorrectionStatusEnum(correction.status.value),
                workflow_id=workflow.id,
                estimated_review_time=estimated_time,
                next_steps=next_steps
            )
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("correction_submission_failed", error=str(e), user_id=user_id)
            raise ServiceError(f"Failed to submit correction: {str(e)}")
    
    async def submit_review(
        self,
        tenant_id: uuid.UUID,
        reviewer_id: uuid.UUID,
        review_data: ReviewSubmissionRequest
    ) -> uuid.UUID:
        """Submit expert review for a correction."""
        try:
            # Get correction and validate reviewer access
            correction = await self._validate_reviewer_access(tenant_id, review_data.correction_id, reviewer_id)
            
            # Create review record
            review = CorrectionReview(
                tenant_id=tenant_id,
                correction_id=review_data.correction_id,
                reviewer_id=reviewer_id,
                decision=ReviewDecision(review_data.decision.value),
                review_notes=review_data.review_notes,
                accuracy_score=review_data.accuracy_score,
                clarity_score=review_data.clarity_score,
                completeness_score=review_data.completeness_score,
                quality_assessment=review_data.quality_assessment,
                review_metadata=review_data.review_metadata,
                completed_at=datetime.now()
            )
            
            # Calculate overall score
            if all(score is not None for score in [review.accuracy_score, review.clarity_score, review.completeness_score]):
                review.overall_score = (review.accuracy_score + review.clarity_score + review.completeness_score) / 3
            
            self.db.add(review)
            
            # Update correction with review information
            correction.reviewer_id = reviewer_id
            correction.reviewed_at = datetime.now()
            correction.review_decision = review.decision
            correction.review_notes = review_data.review_notes
            correction.quality_score = review.overall_score
            
            # Update correction status based on decision
            if review.decision == ReviewDecision.APPROVE:
                correction.status = CorrectionStatus.APPROVED
                # Schedule implementation
                await self._schedule_correction_implementation(correction)
            elif review.decision == ReviewDecision.REJECT:
                correction.status = CorrectionStatus.REJECTED
            elif review.decision == ReviewDecision.REQUEST_CHANGES:
                correction.status = CorrectionStatus.PENDING
            elif review.decision == ReviewDecision.ESCALATE:
                correction.status = CorrectionStatus.UNDER_REVIEW
                await self._escalate_correction(correction)
            
            # Update workflow
            await self._update_workflow_status(correction, review.decision)
            
            self.db.commit()
            
            self.logger.info(
                "correction_review_submitted",
                correction_id=correction.id,
                reviewer_id=reviewer_id,
                decision=review.decision,
                overall_score=review.overall_score
            )
            
            return review.id
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("review_submission_failed", error=str(e), reviewer_id=reviewer_id)
            raise ServiceError(f"Failed to submit review: {str(e)}")
    
    async def implement_correction(
        self,
        tenant_id: uuid.UUID,
        correction_id: uuid.UUID,
        implementer_id: uuid.UUID
    ) -> uuid.UUID:
        """Implement an approved correction."""
        try:
            # Get approved correction
            correction = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.id == correction_id,
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.status == CorrectionStatus.APPROVED
                )
            ).first()
            
            if not correction:
                raise ValidationError("Correction not found or not approved")
            
            # Get current chunk
            chunk = self.db.query(DocumentChunk).filter(
                DocumentChunk.id == correction.chunk_id
            ).first()
            
            if not chunk:
                raise ValidationError("Chunk not found")
            
            # Create new version with corrected content
            new_version = await self._create_new_version(
                chunk=chunk,
                content=correction.corrected_content,
                correction=correction,
                created_by=implementer_id
            )
            
            # Update chunk content (this would typically update the actual chunk storage)
            # For now, we'll track this in the version system
            
            # Mark correction as implemented
            correction.status = CorrectionStatus.IMPLEMENTED
            correction.implemented_at = datetime.now()
            
            # Create impact tracking record
            impact = CorrectionImpact(
                tenant_id=tenant_id,
                correction_id=correction_id,
                measurement_start=datetime.now()
            )
            self.db.add(impact)
            
            # Update workflow
            workflow = self.db.query(CorrectionWorkflow).filter(
                CorrectionWorkflow.correction_id == correction_id
            ).first()
            
            if workflow:
                workflow.current_step = "implemented"
                workflow.completed_at = datetime.now()
            
            self.db.commit()
            
            self.logger.info(
                "correction_implemented",
                correction_id=correction_id,
                chunk_id=correction.chunk_id,
                version_id=new_version.id,
                implementer_id=implementer_id
            )
            
            return new_version.id
            
        except Exception as e:
            self.db.rollback()
            self.logger.error("correction_implementation_failed", error=str(e), correction_id=correction_id)
            raise ServiceError(f"Failed to implement correction: {str(e)}")
    
    async def compare_versions(
        self,
        tenant_id: uuid.UUID,
        comparison_request: VersionComparisonRequest
    ) -> VersionComparisonResult:
        """Compare two versions of content."""
        try:
            # Get both versions
            version_1 = self.db.query(ContentVersion).filter(
                and_(
                    ContentVersion.chunk_id == comparison_request.chunk_id,
                    ContentVersion.version_number == comparison_request.version_1
                )
            ).first()
            
            version_2 = self.db.query(ContentVersion).filter(
                and_(
                    ContentVersion.chunk_id == comparison_request.chunk_id,
                    ContentVersion.version_number == comparison_request.version_2
                )
            ).first()
            
            if not version_1 or not version_2:
                raise ValidationError("One or both versions not found")
            
            # Calculate differences
            differences = self._calculate_content_differences(version_1.content, version_2.content)
            
            # Calculate similarity score
            similarity_score = self._calculate_similarity_score(version_1.content, version_2.content)
            
            # Generate change summary
            change_summary = self._generate_change_summary(differences)
            
            # Convert to response format
            version_1_details = VersionDetails(
                id=version_1.id,
                chunk_id=version_1.chunk_id,
                version_number=version_1.version_number,
                content=version_1.content,
                content_hash=version_1.content_hash,
                change_summary=version_1.change_summary,
                is_active=version_1.is_active,
                is_published=version_1.is_published,
                created_by=version_1.created_by,
                created_at=version_1.created_at,
                quality_score=version_1.quality_score,
                readability_score=version_1.readability_score
            )
            
            version_2_details = VersionDetails(
                id=version_2.id,
                chunk_id=version_2.chunk_id,
                version_number=version_2.version_number,
                content=version_2.content,
                content_hash=version_2.content_hash,
                change_summary=version_2.change_summary,
                is_active=version_2.is_active,
                is_published=version_2.is_published,
                created_by=version_2.created_by,
                created_at=version_2.created_at,
                quality_score=version_2.quality_score,
                readability_score=version_2.readability_score
            )
            
            return VersionComparisonResult(
                chunk_id=comparison_request.chunk_id,
                version_1=version_1_details,
                version_2=version_2_details,
                differences=differences,
                similarity_score=similarity_score,
                change_summary=change_summary
            )
            
        except Exception as e:
            self.logger.error("version_comparison_failed", error=str(e), chunk_id=comparison_request.chunk_id)
            raise ServiceError(f"Failed to compare versions: {str(e)}")
    
    async def get_correction_stats(self, tenant_id: uuid.UUID) -> CorrectionStats:
        """Get correction system statistics."""
        try:
            # Basic counts
            total_corrections = self.db.query(ContentCorrection).filter(
                ContentCorrection.tenant_id == tenant_id
            ).count()
            
            pending_corrections = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.status == CorrectionStatus.PENDING
                )
            ).count()
            
            approved_corrections = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.status == CorrectionStatus.APPROVED
                )
            ).count()
            
            rejected_corrections = self.db.query(ContentCorrection).filter(
                and_(
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.status == CorrectionStatus.REJECTED
                )
            ).count()
            
            # Average review time
            avg_review_time = self.db.query(
                func.avg(
                    func.extract('epoch', ContentCorrection.reviewed_at - ContentCorrection.created_at) / 3600
                )
            ).filter(
                and_(
                    ContentCorrection.tenant_id == tenant_id,
                    ContentCorrection.reviewed_at.isnot(None)
                )
            ).scalar()
            
            average_review_time_hours = float(avg_review_time) if avg_review_time else 0.0
            
            # Correction type breakdown
            type_breakdown = {}
            type_results = self.db.query(
                ContentCorrection.correction_type,
                func.count(ContentCorrection.id)
            ).filter(
                ContentCorrection.tenant_id == tenant_id
            ).group_by(ContentCorrection.correction_type).all()
            
            for correction_type, count in type_results:
                type_breakdown[correction_type.value] = count
            
            # Quality improvement metrics (simplified)
            quality_metrics = {
                "average_accuracy_improvement": 0.15,  # Would be calculated from actual data
                "user_satisfaction_increase": 0.12,
                "search_quality_improvement": 0.08
            }
            
            return CorrectionStats(
                total_corrections=total_corrections,
                pending_corrections=pending_corrections,
                approved_corrections=approved_corrections,
                rejected_corrections=rejected_corrections,
                average_review_time_hours=average_review_time_hours,
                correction_type_breakdown=type_breakdown,
                quality_improvement_metrics=quality_metrics
            )
            
        except Exception as e:
            self.logger.error("correction_stats_failed", error=str(e), tenant_id=tenant_id)
            raise ServiceError(f"Failed to retrieve correction statistics: {str(e)}")
    
    # Private helper methods
    
    async def _validate_chunk_access(self, tenant_id: uuid.UUID, chunk_id: uuid.UUID, user_id: uuid.UUID) -> DocumentChunk:
        """Validate user has access to chunk."""
        chunk = self.db.query(DocumentChunk).filter(
            and_(
                DocumentChunk.id == chunk_id,
                DocumentChunk.tenant_id == tenant_id
            )
        ).first()
        
        if not chunk:
            raise ValidationError("Chunk not found or access denied")
        
        return chunk
    
    async def _get_current_chunk_content(self, chunk: DocumentChunk) -> str:
        """Get current content of chunk."""
        # Get the active version or fall back to original content
        active_version = self.db.query(ContentVersion).filter(
            and_(
                ContentVersion.chunk_id == chunk.id,
                ContentVersion.is_active == True
            )
        ).first()
        
        if active_version:
            return active_version.content
        
        # For now, return a placeholder - in real implementation, this would
        # retrieve the actual chunk content from the storage system
        return f"Current content for chunk {chunk.id}"
    
    async def _validate_correction_content(self, original: str, corrected: str):
        """Validate correction content is meaningful."""
        if original == corrected:
            raise ValidationError("Corrected content is identical to original")
        
        # Check minimum change threshold
        similarity = self._calculate_similarity_score(original, corrected)
        if similarity > 0.99:  # Less than 1% change
            raise ValidationError("Correction changes are too minimal")
    
    def _calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts."""
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio()
    
    def _calculate_content_differences(self, text1: str, text2: str) -> List[Dict[str, Any]]:
        """Calculate structured differences between two texts."""
        differ = difflib.unified_diff(
            text1.splitlines(keepends=True),
            text2.splitlines(keepends=True),
            lineterm=''
        )
        
        differences = []
        for line in differ:
            if line.startswith('@@'):
                continue
            elif line.startswith('-'):
                differences.append({
                    "type": "deletion",
                    "content": line[1:],
                    "line_type": "removed"
                })
            elif line.startswith('+'):
                differences.append({
                    "type": "addition",
                    "content": line[1:],
                    "line_type": "added"
                })
        
        return differences
    
    def _generate_change_summary(self, differences: List[Dict[str, Any]]) -> str:
        """Generate human-readable change summary."""
        if not differences:
            return "No changes detected"
        
        additions = len([d for d in differences if d["type"] == "addition"])
        deletions = len([d for d in differences if d["type"] == "deletion"])
        
        if additions and deletions:
            return f"Modified content with {additions} additions and {deletions} deletions"
        elif additions:
            return f"Added {additions} new lines"
        elif deletions:
            return f"Removed {deletions} lines"
        else:
            return "Content modified"
    
    async def _calculate_impact_score(self, correction: ContentCorrection) -> float:
        """Calculate estimated impact score for correction."""
        # Simplified impact calculation
        base_score = 0.5
        
        # Adjust based on correction type
        type_multipliers = {
            CorrectionType.FACTUAL: 1.5,
            CorrectionType.COMPLETENESS: 1.3,
            CorrectionType.CLARITY: 1.1,
            CorrectionType.FORMATTING: 0.8,
            CorrectionType.GRAMMAR: 0.7,
            CorrectionType.TERMINOLOGY: 1.2
        }
        
        multiplier = type_multipliers.get(correction.correction_type, 1.0)
        
        # Adjust based on confidence
        if correction.confidence_score:
            multiplier *= correction.confidence_score
        
        return min(base_score * multiplier, 1.0)
    
    async def _create_correction_workflow(self, correction: ContentCorrection) -> CorrectionWorkflow:
        """Create workflow for correction."""
        workflow = CorrectionWorkflow(
            tenant_id=correction.tenant_id,
            correction_id=correction.id,
            current_step="submission",
            workflow_data={
                "priority": correction.priority.value,
                "type": correction.correction_type.value
            },
            steps_completed=["submission"],
            next_steps=["validation", "review", "approval"]
        )
        
        self.db.add(workflow)
        return workflow
    
    def _estimate_review_time(self, priority: CorrectionPriority, correction_type: CorrectionType) -> str:
        """Estimate review time based on priority and type."""
        if priority == CorrectionPriority.CRITICAL:
            return "4-8 hours"
        elif priority == CorrectionPriority.HIGH:
            return "1-2 business days"
        elif priority == CorrectionPriority.MEDIUM:
            return "2-3 business days"
        else:
            return "3-5 business days"
    
    def _get_next_steps(self, current_step: str) -> List[str]:
        """Get next steps in workflow."""
        step_map = {
            "submission": ["Expert review assignment", "Quality assessment"],
            "review": ["Approval decision", "Implementation"],
            "approval": ["Content update", "Re-embedding"],
            "implementation": ["Quality verification", "Impact measurement"]
        }

        return step_map.get(current_step, ["Process completion"])

    async def _ensure_initial_version(self, chunk: DocumentChunk, content: str, user_id: uuid.UUID):
        """Ensure initial version exists for chunk."""
        existing_version = self.db.query(ContentVersion).filter(
            and_(
                ContentVersion.chunk_id == chunk.id,
                ContentVersion.version_number == 1
            )
        ).first()

        if not existing_version:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            initial_version = ContentVersion(
                chunk_id=chunk.id,
                version_number=1,
                content=content,
                content_hash=content_hash,
                change_summary="Initial version",
                is_active=True,
                is_published=True,
                created_by=user_id
            )
            self.db.add(initial_version)

    async def _assign_reviewer(self, workflow: CorrectionWorkflow, correction: ContentCorrection):
        """Assign reviewer for high priority corrections."""
        # Simplified reviewer assignment - in real implementation, this would
        # use more sophisticated logic to find appropriate reviewers
        workflow.current_step = "review"
        workflow.assigned_at = datetime.now()
        workflow.due_date = datetime.now() + timedelta(hours=24 if correction.priority == CorrectionPriority.CRITICAL else 48)

    async def _validate_reviewer_access(self, tenant_id: uuid.UUID, correction_id: uuid.UUID, reviewer_id: uuid.UUID) -> ContentCorrection:
        """Validate reviewer has access to correction."""
        correction = self.db.query(ContentCorrection).filter(
            and_(
                ContentCorrection.id == correction_id,
                ContentCorrection.tenant_id == tenant_id,
                ContentCorrection.status.in_([CorrectionStatus.PENDING, CorrectionStatus.UNDER_REVIEW])
            )
        ).first()

        if not correction:
            raise ValidationError("Correction not found or not available for review")

        return correction

    async def _schedule_correction_implementation(self, correction: ContentCorrection):
        """Schedule correction for implementation."""
        # Update workflow to implementation step
        workflow = self.db.query(CorrectionWorkflow).filter(
            CorrectionWorkflow.correction_id == correction.id
        ).first()

        if workflow:
            workflow.current_step = "implementation"
            workflow.steps_completed.append("review")
            workflow.next_steps = ["implementation", "verification"]

    async def _escalate_correction(self, correction: ContentCorrection):
        """Escalate correction to senior reviewer."""
        # Update workflow for escalation
        workflow = self.db.query(CorrectionWorkflow).filter(
            CorrectionWorkflow.correction_id == correction.id
        ).first()

        if workflow:
            workflow.current_step = "escalation"
            workflow.workflow_data["escalated"] = True
            workflow.workflow_data["escalation_reason"] = "Expert review escalation"

    async def _update_workflow_status(self, correction: ContentCorrection, decision: ReviewDecision):
        """Update workflow status based on review decision."""
        workflow = self.db.query(CorrectionWorkflow).filter(
            CorrectionWorkflow.correction_id == correction.id
        ).first()

        if workflow:
            if decision == ReviewDecision.APPROVE:
                workflow.current_step = "approved"
                workflow.steps_completed.append("review")
                workflow.next_steps = ["implementation"]
            elif decision == ReviewDecision.REJECT:
                workflow.current_step = "rejected"
                workflow.completed_at = datetime.now()
            elif decision == ReviewDecision.REQUEST_CHANGES:
                workflow.current_step = "revision_requested"
                workflow.next_steps = ["revision", "resubmission"]
            elif decision == ReviewDecision.ESCALATE:
                workflow.current_step = "escalated"
                workflow.next_steps = ["senior_review"]

    async def _create_new_version(
        self,
        chunk: DocumentChunk,
        content: str,
        correction: ContentCorrection,
        created_by: uuid.UUID
    ) -> ContentVersion:
        """Create new version with corrected content."""
        # Get next version number
        max_version = self.db.query(func.max(ContentVersion.version_number)).filter(
            ContentVersion.chunk_id == chunk.id
        ).scalar()

        next_version = (max_version or 0) + 1

        # Deactivate current active version
        self.db.query(ContentVersion).filter(
            and_(
                ContentVersion.chunk_id == chunk.id,
                ContentVersion.is_active == True
            )
        ).update({"is_active": False})

        # Create new version
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        new_version = ContentVersion(
            chunk_id=chunk.id,
            correction_id=correction.id,
            version_number=next_version,
            content=content,
            content_hash=content_hash,
            change_summary=correction.correction_reason or f"Correction: {correction.correction_type.value}",
            is_active=True,
            is_published=True,
            created_by=created_by,
            change_metadata={
                "correction_type": correction.correction_type.value,
                "correction_id": str(correction.id),
                "reviewer_id": str(correction.reviewer_id) if correction.reviewer_id else None
            }
        )

        self.db.add(new_version)
        return new_version


# Dependency injection
_correction_service_instance = None


def get_correction_service(db_session: Session = None) -> CorrectionService:
    """Get correction service instance with dependency injection."""
    global _correction_service_instance
    if _correction_service_instance is None:
        if db_session is None:
            db_session = get_database_session()
        _correction_service_instance = CorrectionService(db_session)
    return _correction_service_instance
