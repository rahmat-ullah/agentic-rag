"""
Link Validation and Quality Assessment Service

This module provides comprehensive validation rules, quality metrics,
automated assessment, and quality reporting for document links.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from agentic_rag.models.database import (
    Document, DocumentLink, DocumentKind, DocumentStatus, 
    LinkType, UserFeedback
)
from agentic_rag.services.confidence_scoring import (
    get_confidence_scoring_service, ConfidenceAnalysisRequest
)

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class ValidationRule(BaseModel):
    """Individual validation rule."""
    
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(..., description="Rule description")
    severity: str = Field(..., description="Rule severity: error, warning, info")
    category: str = Field(..., description="Rule category")
    is_active: bool = Field(default=True, description="Whether rule is active")


class ValidationResult(BaseModel):
    """Result of a single validation rule."""
    
    rule: ValidationRule = Field(..., description="The validation rule")
    passed: bool = Field(..., description="Whether the rule passed")
    message: str = Field(..., description="Validation message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class LinkValidationReport(BaseModel):
    """Comprehensive validation report for a document link."""
    
    link_id: UUID = Field(..., description="Document link ID")
    rfq_id: UUID = Field(..., description="RFQ document ID")
    offer_id: UUID = Field(..., description="Offer document ID")
    
    # Validation results
    validation_results: List[ValidationResult] = Field(..., description="Individual validation results")
    overall_status: str = Field(..., description="Overall validation status: passed, failed, warning")
    error_count: int = Field(..., description="Number of errors")
    warning_count: int = Field(..., description="Number of warnings")
    info_count: int = Field(..., description="Number of info messages")
    
    # Quality metrics
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    # Recommendations
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    required_actions: List[str] = Field(..., description="Required actions to fix errors")
    
    # Metadata
    validation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validation_duration: float = Field(..., description="Validation duration in seconds")


class QualityMetrics(BaseModel):
    """Quality metrics for document links."""
    
    total_links: int = Field(..., description="Total number of links")
    validated_links: int = Field(..., description="Number of validated links")
    high_quality_links: int = Field(..., description="Number of high quality links (>0.8)")
    medium_quality_links: int = Field(..., description="Number of medium quality links (0.5-0.8)")
    low_quality_links: int = Field(..., description="Number of low quality links (<0.5)")
    
    average_confidence: float = Field(..., description="Average confidence score")
    average_quality: float = Field(..., description="Average quality score")
    
    # By link type
    manual_links: int = Field(..., description="Number of manual links")
    automatic_links: int = Field(..., description="Number of automatic links")
    suggested_links: int = Field(..., description="Number of suggested links")
    
    # User feedback
    accepted_links: int = Field(..., description="Number of accepted links")
    rejected_links: int = Field(..., description="Number of rejected links")
    modified_links: int = Field(..., description="Number of modified links")


# ============================================================================
# VALIDATION SERVICE
# ============================================================================

class LinkValidationService:
    """Comprehensive link validation and quality assessment service."""
    
    def __init__(self):
        self._confidence_service = None
        self._validation_rules = self._initialize_validation_rules()
        self._stats = {
            "validations_performed": 0,
            "errors_detected": 0,
            "warnings_generated": 0
        }
        
        logger.info("Link validation service initialized")
    
    async def initialize(self):
        """Initialize service dependencies."""
        if not self._confidence_service:
            self._confidence_service = get_confidence_scoring_service()
        
        logger.info("Link validation service dependencies initialized")
    
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize the validation rules."""
        return [
            ValidationRule(
                rule_id="LINK_001",
                name="Document Existence",
                description="Both RFQ and Offer documents must exist and be accessible",
                severity="error",
                category="data_integrity"
            ),
            ValidationRule(
                rule_id="LINK_002",
                name="Document Status",
                description="Both documents should be in READY status for optimal linking",
                severity="warning",
                category="document_quality"
            ),
            ValidationRule(
                rule_id="LINK_003",
                name="Confidence Threshold",
                description="Link confidence should be above minimum threshold (0.3)",
                severity="warning",
                category="confidence"
            ),
            ValidationRule(
                rule_id="LINK_004",
                name="Offer Type Validity",
                description="Offer type must be one of: technical, commercial, pricing",
                severity="error",
                category="data_integrity"
            ),
            ValidationRule(
                rule_id="LINK_005",
                name="Duplicate Link Check",
                description="No duplicate links should exist for the same RFQ-Offer pair",
                severity="error",
                category="data_integrity"
            ),
            ValidationRule(
                rule_id="LINK_006",
                name="Document Kind Compatibility",
                description="Offer document kind should be compatible with offer type",
                severity="warning",
                category="logical_consistency"
            ),
            ValidationRule(
                rule_id="LINK_007",
                name="Chunk Availability",
                description="Documents should have processed chunks for better analysis",
                severity="info",
                category="document_quality"
            ),
            ValidationRule(
                rule_id="LINK_008",
                name="Temporal Reasonableness",
                description="Documents should be created within reasonable time frame",
                severity="info",
                category="temporal_analysis"
            ),
            ValidationRule(
                rule_id="LINK_009",
                name="User Feedback Consistency",
                description="User feedback should be consistent with confidence score",
                severity="warning",
                category="user_validation"
            ),
            ValidationRule(
                rule_id="LINK_010",
                name="Quality Score Threshold",
                description="Quality score should meet minimum standards",
                severity="warning",
                category="quality_assessment"
            )
        ]
    
    async def validate_link(
        self,
        db_session: Session,
        link_id: UUID,
        tenant_id: UUID,
        include_confidence_analysis: bool = True
    ) -> LinkValidationReport:
        """
        Perform comprehensive validation of a document link.
        
        Args:
            db_session: Database session
            link_id: Document link ID
            tenant_id: Tenant ID
            include_confidence_analysis: Whether to include confidence analysis
            
        Returns:
            LinkValidationReport with detailed validation results
        """
        await self.initialize()
        start_time = datetime.now(timezone.utc)
        
        # Get the link
        link = db_session.query(DocumentLink).filter(
            and_(
                DocumentLink.id == link_id,
                DocumentLink.tenant_id == tenant_id
            )
        ).first()
        
        if not link:
            raise ValueError("Document link not found")
        
        # Get associated documents
        rfq_document = db_session.query(Document).get(link.rfq_id)
        offer_document = db_session.query(Document).get(link.offer_id)
        
        # Run validation rules
        validation_results = []
        for rule in self._validation_rules:
            if rule.is_active:
                result = await self._execute_validation_rule(
                    rule, link, rfq_document, offer_document, db_session
                )
                validation_results.append(result)
        
        # Calculate counts
        error_count = sum(1 for r in validation_results if not r.passed and r.rule.severity == "error")
        warning_count = sum(1 for r in validation_results if not r.passed and r.rule.severity == "warning")
        info_count = sum(1 for r in validation_results if not r.passed and r.rule.severity == "info")
        
        # Determine overall status
        if error_count > 0:
            overall_status = "failed"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "passed"
        
        # Get confidence analysis if requested
        confidence_score = link.confidence
        if include_confidence_analysis and self._confidence_service:
            try:
                confidence_analysis = await self._confidence_service.calculate_confidence_score(
                    db_session=db_session,
                    request=ConfidenceAnalysisRequest(
                        rfq_id=link.rfq_id,
                        offer_id=link.offer_id,
                        existing_link_id=link.id
                    ),
                    tenant_id=tenant_id
                )
                confidence_score = confidence_analysis.overall_confidence
            except Exception as e:
                logger.warning(f"Failed to get confidence analysis: {e}")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            link, rfq_document, offer_document, validation_results
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, link)
        required_actions = self._generate_required_actions(validation_results)
        
        # Calculate duration
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Update statistics
        self._stats["validations_performed"] += 1
        self._stats["errors_detected"] += error_count
        self._stats["warnings_generated"] += warning_count
        
        return LinkValidationReport(
            link_id=link.id,
            rfq_id=link.rfq_id,
            offer_id=link.offer_id,
            validation_results=validation_results,
            overall_status=overall_status,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            quality_score=quality_score,
            confidence_score=confidence_score,
            recommendations=recommendations,
            required_actions=required_actions,
            validation_duration=duration
        )
    
    async def _execute_validation_rule(
        self,
        rule: ValidationRule,
        link: DocumentLink,
        rfq_document: Optional[Document],
        offer_document: Optional[Document],
        db_session: Session
    ) -> ValidationResult:
        """Execute a single validation rule."""
        try:
            if rule.rule_id == "LINK_001":
                return self._validate_document_existence(rule, rfq_document, offer_document)
            elif rule.rule_id == "LINK_002":
                return self._validate_document_status(rule, rfq_document, offer_document)
            elif rule.rule_id == "LINK_003":
                return self._validate_confidence_threshold(rule, link)
            elif rule.rule_id == "LINK_004":
                return self._validate_offer_type(rule, link)
            elif rule.rule_id == "LINK_005":
                return self._validate_duplicate_links(rule, link, db_session)
            elif rule.rule_id == "LINK_006":
                return self._validate_document_kind_compatibility(rule, link, offer_document)
            elif rule.rule_id == "LINK_007":
                return self._validate_chunk_availability(rule, rfq_document, offer_document)
            elif rule.rule_id == "LINK_008":
                return self._validate_temporal_reasonableness(rule, rfq_document, offer_document)
            elif rule.rule_id == "LINK_009":
                return self._validate_user_feedback_consistency(rule, link)
            elif rule.rule_id == "LINK_010":
                return self._validate_quality_score_threshold(rule, link)
            else:
                return ValidationResult(
                    rule=rule,
                    passed=True,
                    message="Rule not implemented"
                )
        except Exception as e:
            logger.error(f"Error executing validation rule {rule.rule_id}: {e}")
            return ValidationResult(
                rule=rule,
                passed=False,
                message=f"Validation error: {str(e)}"
            )
    
    def _validate_document_existence(
        self, rule: ValidationRule, rfq_document: Optional[Document], offer_document: Optional[Document]
    ) -> ValidationResult:
        """Validate that both documents exist."""
        if not rfq_document:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="RFQ document not found or not accessible"
            )
        
        if not offer_document:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="Offer document not found or not accessible"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message="Both documents exist and are accessible"
        )
    
    def _validate_document_status(
        self, rule: ValidationRule, rfq_document: Optional[Document], offer_document: Optional[Document]
    ) -> ValidationResult:
        """Validate document processing status."""
        if not rfq_document or not offer_document:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="Cannot validate status - documents missing"
            )
        
        if rfq_document.status != DocumentStatus.READY or offer_document.status != DocumentStatus.READY:
            return ValidationResult(
                rule=rule,
                passed=False,
                message=f"Documents not ready - RFQ: {rfq_document.status}, Offer: {offer_document.status}"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message="Both documents are in READY status"
        )
    
    def _validate_confidence_threshold(self, rule: ValidationRule, link: DocumentLink) -> ValidationResult:
        """Validate confidence threshold."""
        min_confidence = 0.3
        
        if link.confidence < min_confidence:
            return ValidationResult(
                rule=rule,
                passed=False,
                message=f"Confidence {link.confidence:.2f} below threshold {min_confidence}"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message=f"Confidence {link.confidence:.2f} meets threshold"
        )
    
    def _validate_offer_type(self, rule: ValidationRule, link: DocumentLink) -> ValidationResult:
        """Validate offer type."""
        valid_types = ["technical", "commercial", "pricing"]
        
        if link.offer_type not in valid_types:
            return ValidationResult(
                rule=rule,
                passed=False,
                message=f"Invalid offer type: {link.offer_type}"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message=f"Valid offer type: {link.offer_type}"
        )
    
    def _validate_duplicate_links(
        self, rule: ValidationRule, link: DocumentLink, db_session: Session
    ) -> ValidationResult:
        """Validate no duplicate links exist."""
        duplicate_count = db_session.query(DocumentLink).filter(
            and_(
                DocumentLink.tenant_id == link.tenant_id,
                DocumentLink.rfq_id == link.rfq_id,
                DocumentLink.offer_id == link.offer_id,
                DocumentLink.id != link.id  # Exclude current link
            )
        ).count()
        
        if duplicate_count > 0:
            return ValidationResult(
                rule=rule,
                passed=False,
                message=f"Found {duplicate_count} duplicate links"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message="No duplicate links found"
        )
    
    def _validate_document_kind_compatibility(
        self, rule: ValidationRule, link: DocumentLink, offer_document: Optional[Document]
    ) -> ValidationResult:
        """Validate document kind compatibility with offer type."""
        if not offer_document:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="Cannot validate compatibility - offer document missing"
            )
        
        # Define compatibility mapping
        compatibility = {
            "technical": [DocumentKind.OFFER_TECH],
            "commercial": [DocumentKind.OFFER_COMM],
            "pricing": [DocumentKind.PRICING]
        }
        
        expected_kinds = compatibility.get(link.offer_type, [])
        
        if offer_document.kind not in expected_kinds:
            return ValidationResult(
                rule=rule,
                passed=False,
                message=f"Document kind {offer_document.kind} incompatible with offer type {link.offer_type}"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message=f"Document kind {offer_document.kind} compatible with offer type {link.offer_type}"
        )
    
    def _validate_chunk_availability(
        self, rule: ValidationRule, rfq_document: Optional[Document], offer_document: Optional[Document]
    ) -> ValidationResult:
        """Validate chunk availability."""
        if not rfq_document or not offer_document:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="Cannot validate chunks - documents missing"
            )
        
        if not rfq_document.chunk_count or rfq_document.chunk_count == 0:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="RFQ document has no processed chunks"
            )
        
        if not offer_document.chunk_count or offer_document.chunk_count == 0:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="Offer document has no processed chunks"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message=f"Both documents have chunks - RFQ: {rfq_document.chunk_count}, Offer: {offer_document.chunk_count}"
        )
    
    def _validate_temporal_reasonableness(
        self, rule: ValidationRule, rfq_document: Optional[Document], offer_document: Optional[Document]
    ) -> ValidationResult:
        """Validate temporal reasonableness."""
        if not rfq_document or not offer_document:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="Cannot validate temporal relationship - documents missing"
            )
        
        if not rfq_document.created_at or not offer_document.created_at:
            return ValidationResult(
                rule=rule,
                passed=True,
                message="Cannot validate temporal relationship - timestamps missing"
            )
        
        # Check if documents are created within reasonable time frame (1 year)
        time_diff_days = abs((rfq_document.created_at - offer_document.created_at).days)
        
        if time_diff_days > 365:
            return ValidationResult(
                rule=rule,
                passed=False,
                message=f"Large time gap between documents: {time_diff_days} days"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message=f"Reasonable time gap: {time_diff_days} days"
        )
    
    def _validate_user_feedback_consistency(self, rule: ValidationRule, link: DocumentLink) -> ValidationResult:
        """Validate user feedback consistency."""
        if not link.user_feedback:
            return ValidationResult(
                rule=rule,
                passed=True,
                message="No user feedback to validate"
            )
        
        # Check consistency between feedback and confidence
        if link.user_feedback == UserFeedback.REJECTED and link.confidence > 0.7:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="High confidence but user rejected the link"
            )
        
        if link.user_feedback == UserFeedback.ACCEPTED and link.confidence < 0.3:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="Low confidence but user accepted the link"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message="User feedback consistent with confidence score"
        )
    
    def _validate_quality_score_threshold(self, rule: ValidationRule, link: DocumentLink) -> ValidationResult:
        """Validate quality score threshold."""
        min_quality = 0.4
        
        if link.quality_score and link.quality_score < min_quality:
            return ValidationResult(
                rule=rule,
                passed=False,
                message=f"Quality score {link.quality_score:.2f} below threshold {min_quality}"
            )
        
        return ValidationResult(
            rule=rule,
            passed=True,
            message=f"Quality score meets threshold"
        )
    
    def _calculate_quality_score(
        self,
        link: DocumentLink,
        rfq_document: Optional[Document],
        offer_document: Optional[Document],
        validation_results: List[ValidationResult]
    ) -> float:
        """Calculate overall quality score based on validation results."""
        # Start with base confidence
        quality = link.confidence * 0.6
        
        # Add validation score
        passed_rules = sum(1 for r in validation_results if r.passed)
        total_rules = len(validation_results)
        validation_score = passed_rules / total_rules if total_rules > 0 else 0.0
        quality += validation_score * 0.3
        
        # Add document quality bonus
        if rfq_document and offer_document:
            if (rfq_document.status == DocumentStatus.READY and 
                offer_document.status == DocumentStatus.READY):
                quality += 0.1
        
        return max(0.0, min(1.0, quality))
    
    def _generate_recommendations(
        self, validation_results: List[ValidationResult], link: DocumentLink
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for result in validation_results:
            if not result.passed:
                if result.rule.rule_id == "LINK_002":
                    recommendations.append("Ensure both documents are fully processed before linking")
                elif result.rule.rule_id == "LINK_003":
                    recommendations.append("Review link relevance to improve confidence score")
                elif result.rule.rule_id == "LINK_006":
                    recommendations.append("Verify offer type matches document content")
                elif result.rule.rule_id == "LINK_007":
                    recommendations.append("Process documents to generate chunks for better analysis")
                elif result.rule.rule_id == "LINK_009":
                    recommendations.append("Review user feedback and confidence score alignment")
        
        if link.confidence < 0.5:
            recommendations.append("Consider reviewing link relevance or removing low-confidence link")
        
        return recommendations
    
    def _generate_required_actions(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate required actions for error fixes."""
        actions = []
        
        for result in validation_results:
            if not result.passed and result.rule.severity == "error":
                if result.rule.rule_id == "LINK_001":
                    actions.append("Restore missing documents or remove invalid link")
                elif result.rule.rule_id == "LINK_004":
                    actions.append("Correct offer type to valid value")
                elif result.rule.rule_id == "LINK_005":
                    actions.append("Remove duplicate links")
        
        return actions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self._stats.copy()


# Singleton instance
_link_validation_service: Optional[LinkValidationService] = None


def get_link_validation_service() -> LinkValidationService:
    """Get or create link validation service instance."""
    global _link_validation_service
    
    if _link_validation_service is None:
        _link_validation_service = LinkValidationService()
    
    return _link_validation_service
