"""
Pricing Analysis and Redaction System Integration

This module provides integration between pricing analysis services and the redaction
system for secure pricing data handling with role-based access control.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from agentic_rag.config import get_settings
from agentic_rag.models.database import UserRole
from agentic_rag.services.pricing_extraction import PricingItem, PricingExtractionResult
from agentic_rag.services.competitive_analysis import CompetitiveAnalysisResult
from agentic_rag.services.cost_modeling import CostModelingResult
from agentic_rag.services.pricing_dashboard import DashboardData
from agentic_rag.services.pricing_masking import PricingMaskingResult

logger = structlog.get_logger(__name__)


class PricingDataSensitivity(str, Enum):
    """Sensitivity levels for pricing data."""
    
    PUBLIC = "public"           # No redaction needed
    INTERNAL = "internal"       # Basic redaction for external users
    CONFIDENTIAL = "confidential"  # Redaction for non-analysts
    RESTRICTED = "restricted"   # Admin-only access


class PricingOperationType(str, Enum):
    """Types of pricing operations."""
    
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    MODELING = "modeling"
    DASHBOARD_VIEW = "dashboard_view"
    REPORT_GENERATION = "report_generation"


@dataclass
class PricingSecurityContext:
    """Security context for pricing operations."""
    
    user_id: str
    user_role: UserRole
    tenant_id: str
    operation_type: PricingOperationType
    data_sensitivity: PricingDataSensitivity
    request_id: str = ""
    session_id: str = ""
    ip_address: str = ""
    user_agent: str = ""


class SecurePricingItem(BaseModel):
    """Secure pricing item with redaction applied."""
    
    # Original item reference
    original_item_id: str = Field(..., description="Original pricing item ID")
    
    # Redacted pricing data
    item_name: str = Field(..., description="Item name (potentially redacted)")
    total_price: Optional[str] = Field(default=None, description="Total price (redacted)")
    unit_price: Optional[str] = Field(default=None, description="Unit price (redacted)")
    currency: Optional[str] = Field(default=None, description="Currency (potentially redacted)")
    vendor: Optional[str] = Field(default=None, description="Vendor (potentially redacted)")
    category: Optional[str] = Field(default=None, description="Category")
    
    # Redaction metadata
    redaction_applied: bool = Field(..., description="Whether redaction was applied")
    redaction_level: str = Field(..., description="Level of redaction applied")
    accessible_fields: List[str] = Field(
        default_factory=list,
        description="Fields accessible to user"
    )
    
    # Security metadata
    access_granted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When access was granted"
    )
    access_expires_at: Optional[datetime] = Field(
        default=None,
        description="When access expires"
    )


class SecurePricingAnalysisResult(BaseModel):
    """Secure pricing analysis result with redaction."""
    
    # Analysis metadata
    analysis_id: str = Field(default_factory=lambda: str(uuid4()), description="Analysis ID")
    operation_type: PricingOperationType = Field(..., description="Type of operation")
    
    # Redacted results
    secure_items: List[SecurePricingItem] = Field(
        default_factory=list,
        description="Secure pricing items"
    )
    
    # Summary statistics (potentially redacted)
    summary_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics"
    )
    
    # Access control
    user_role: UserRole = Field(..., description="User role")
    data_sensitivity: PricingDataSensitivity = Field(..., description="Data sensitivity")
    redaction_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Redaction summary"
    )
    
    # Audit information
    audit_trail_id: str = Field(..., description="Audit trail reference")
    
    # Performance metrics
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )


class PricingRedactionConfig(BaseModel):
    """Configuration for pricing redaction integration."""
    
    # Redaction settings
    enable_redaction: bool = Field(default=True, description="Enable redaction")
    default_sensitivity: PricingDataSensitivity = Field(
        default=PricingDataSensitivity.INTERNAL,
        description="Default data sensitivity"
    )
    
    # Role-based access
    role_permissions: Dict[UserRole, List[PricingDataSensitivity]] = Field(
        default_factory=lambda: {
            UserRole.VIEWER: [PricingDataSensitivity.PUBLIC],
            UserRole.ANALYST: [
                PricingDataSensitivity.PUBLIC,
                PricingDataSensitivity.INTERNAL,
                PricingDataSensitivity.CONFIDENTIAL
            ],
            UserRole.ADMIN: [
                PricingDataSensitivity.PUBLIC,
                PricingDataSensitivity.INTERNAL,
                PricingDataSensitivity.CONFIDENTIAL,
                PricingDataSensitivity.RESTRICTED
            ]
        },
        description="Role-based access permissions"
    )
    
    # Audit settings
    enable_audit_trail: bool = Field(default=True, description="Enable audit trail")
    audit_all_operations: bool = Field(default=True, description="Audit all operations")
    
    # Security settings
    session_timeout_minutes: int = Field(default=60, description="Session timeout")
    max_concurrent_sessions: int = Field(default=5, description="Max concurrent sessions")
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL")


class PricingRedactionIntegrationService:
    """Service for integrating pricing analysis with redaction system."""
    
    def __init__(self, config: Optional[PricingRedactionConfig] = None):
        self.config = config or PricingRedactionConfig()
        self.settings = get_settings()
        
        # Initialize component services (lazy loading to avoid circular imports)
        self._pricing_extraction_service = None
        self._competitive_analysis_service = None
        self._cost_modeling_service = None
        self._dashboard_service = None
        self._pricing_masking_service = None
        
        # Performance tracking
        self._stats = {
            "total_secure_operations": 0,
            "total_redactions_applied": 0,
            "total_access_denied": 0,
            "total_audit_events": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Cache for secure results
        self._cache: Dict[str, Any] = {}
        
        # Active sessions
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Pricing redaction integration service initialized")

    @property
    def pricing_extraction_service(self):
        """Lazy load pricing extraction service."""
        if self._pricing_extraction_service is None:
            from agentic_rag.services.pricing_extraction import PricingExtractionService
            self._pricing_extraction_service = PricingExtractionService()
        return self._pricing_extraction_service

    @property
    def competitive_analysis_service(self):
        """Lazy load competitive analysis service."""
        if self._competitive_analysis_service is None:
            from agentic_rag.services.competitive_analysis import CompetitiveAnalysisService
            self._competitive_analysis_service = CompetitiveAnalysisService()
        return self._competitive_analysis_service

    @property
    def cost_modeling_service(self):
        """Lazy load cost modeling service."""
        if self._cost_modeling_service is None:
            from agentic_rag.services.cost_modeling import CostModelingService
            self._cost_modeling_service = CostModelingService()
        return self._cost_modeling_service

    @property
    def dashboard_service(self):
        """Lazy load dashboard service."""
        if self._dashboard_service is None:
            from agentic_rag.services.pricing_dashboard import PricingDashboardService
            self._dashboard_service = PricingDashboardService()
        return self._dashboard_service

    @property
    def pricing_masking_service(self):
        """Lazy load pricing masking service."""
        if self._pricing_masking_service is None:
            from agentic_rag.services.pricing_masking import PricingMaskingService
            self._pricing_masking_service = PricingMaskingService()
        return self._pricing_masking_service
    
    def secure_pricing_extraction(self, text: str, 
                                 security_context: PricingSecurityContext) -> SecurePricingAnalysisResult:
        """Perform secure pricing extraction with redaction."""
        
        start_time = time.time()
        
        try:
            # Validate access
            if not self._validate_access(security_context, PricingOperationType.EXTRACTION):
                raise PermissionError("Access denied for pricing extraction")
            
            # Perform extraction
            extraction_result = self.pricing_extraction_service.extract_pricing_data(text)
            
            # Apply redaction
            secure_result = self._apply_redaction_to_extraction(
                extraction_result, security_context
            )
            
            # Create audit event
            if self.config.enable_audit_trail:
                audit_event_id = self._log_audit_event(
                    security_context, "pricing_extraction",
                    {"items_extracted": len(extraction_result.pricing_items)}
                )
                secure_result.audit_trail_id = audit_event_id
            
            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            secure_result.processing_time_ms = processing_time_ms
            self._update_stats("extraction", processing_time_ms)
            
            logger.info(
                "Secure pricing extraction completed",
                user_id=security_context.user_id,
                user_role=security_context.user_role.value,
                items_extracted=len(secure_result.secure_items),
                redaction_applied=any(item.redaction_applied for item in secure_result.secure_items)
            )
            
            return secure_result
            
        except Exception as e:
            logger.error(
                "Secure pricing extraction failed",
                error=str(e),
                user_id=security_context.user_id
            )
            raise
    
    def secure_competitive_analysis(self, pricing_items: List[PricingItem],
                                   security_context: PricingSecurityContext) -> SecurePricingAnalysisResult:
        """Perform secure competitive analysis with redaction."""
        
        start_time = time.time()
        
        try:
            # Validate access
            if not self._validate_access(security_context, PricingOperationType.ANALYSIS):
                raise PermissionError("Access denied for competitive analysis")
            
            # Perform analysis
            analysis_result = self.competitive_analysis_service.analyze_competitive_pricing(pricing_items)
            
            # Apply redaction
            secure_result = self._apply_redaction_to_analysis(
                analysis_result, security_context
            )
            
            # Create audit event
            if self.config.enable_audit_trail:
                audit_event_id = self._log_audit_event(
                    security_context, "competitive_analysis",
                    {"items_analyzed": len(pricing_items)}
                )
                secure_result.audit_trail_id = audit_event_id
            
            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            secure_result.processing_time_ms = processing_time_ms
            self._update_stats("analysis", processing_time_ms)
            
            logger.info(
                "Secure competitive analysis completed",
                user_id=security_context.user_id,
                user_role=security_context.user_role.value,
                items_analyzed=len(pricing_items)
            )
            
            return secure_result
            
        except Exception as e:
            logger.error(
                "Secure competitive analysis failed",
                error=str(e),
                user_id=security_context.user_id
            )
            raise

    def secure_dashboard_access(self, security_context: PricingSecurityContext) -> DashboardData:
        """Provide secure access to pricing dashboard with redaction."""

        start_time = time.time()

        try:
            # Validate access
            if not self._validate_access(security_context, PricingOperationType.DASHBOARD_VIEW):
                raise PermissionError("Access denied for dashboard view")

            # Get dashboard data
            dashboard_data = self.dashboard_service.get_dashboard_data(security_context.user_role)

            # Apply redaction to dashboard data
            secure_dashboard = self._apply_redaction_to_dashboard(dashboard_data, security_context)

            # Create audit event
            if self.config.enable_audit_trail:
                self._log_audit_event(
                    security_context, "dashboard_access",
                    {"widgets_accessed": len(dashboard_data.widgets)}
                )

            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats("dashboard", processing_time_ms)

            logger.info(
                "Secure dashboard access completed",
                user_id=security_context.user_id,
                user_role=security_context.user_role.value,
                widgets_count=len(dashboard_data.widgets)
            )

            return secure_dashboard

        except Exception as e:
            logger.error(
                "Secure dashboard access failed",
                error=str(e),
                user_id=security_context.user_id
            )
            raise

    def _validate_access(self, security_context: PricingSecurityContext,
                        operation_type: PricingOperationType) -> bool:
        """Validate user access for pricing operations."""

        try:
            # Check role permissions
            allowed_sensitivities = self.config.role_permissions.get(
                security_context.user_role, []
            )

            if security_context.data_sensitivity not in allowed_sensitivities:
                self._stats["total_access_denied"] += 1
                logger.warning(
                    "Access denied - insufficient permissions",
                    user_id=security_context.user_id,
                    user_role=security_context.user_role.value,
                    required_sensitivity=security_context.data_sensitivity.value
                )
                return False

            # Check session validity
            if not self._validate_session(security_context):
                self._stats["total_access_denied"] += 1
                logger.warning(
                    "Access denied - invalid session",
                    user_id=security_context.user_id,
                    session_id=security_context.session_id
                )
                return False

            return True

        except Exception as e:
            logger.error("Access validation failed", error=str(e))
            return False

    def _validate_session(self, security_context: PricingSecurityContext) -> bool:
        """Validate user session."""

        if not security_context.session_id:
            return True  # No session validation required

        session = self._active_sessions.get(security_context.session_id)
        if not session:
            return False

        # Check session expiry
        if session.get("expires_at", 0) < time.time():
            del self._active_sessions[security_context.session_id]
            return False

        # Check user match
        if session.get("user_id") != security_context.user_id:
            return False

        return True

    def _apply_redaction_to_extraction(self, extraction_result: PricingExtractionResult,
                                     security_context: PricingSecurityContext) -> SecurePricingAnalysisResult:
        """Apply redaction to pricing extraction results."""

        secure_items = []
        redaction_count = 0

        try:
            for item in extraction_result.pricing_items:
                # Determine if redaction is needed
                needs_redaction = self._needs_redaction(item, security_context)

                if needs_redaction:
                    # Apply pricing masking
                    masking_result = self.pricing_masking_service.mask_pricing(
                        str(item.total_price) if item.total_price else "",
                        security_context.user_role
                    )

                    secure_item = SecurePricingItem(
                        original_item_id=str(uuid4()),
                        item_name=item.item_name or "[REDACTED]",
                        total_price=masking_result.masked_text,
                        unit_price=masking_result.masked_text if item.unit_price else None,
                        currency="[REDACTED]" if item.currency else None,
                        vendor=item.vendor if security_context.user_role != UserRole.VIEWER else "[REDACTED]",
                        category=item.category,
                        redaction_applied=True,
                        redaction_level="partial",
                        accessible_fields=["item_name", "category"]
                    )
                    redaction_count += 1
                else:
                    # No redaction needed
                    secure_item = SecurePricingItem(
                        original_item_id=str(uuid4()),
                        item_name=item.item_name or "",
                        total_price=str(item.total_price) if item.total_price else None,
                        unit_price=str(item.unit_price) if item.unit_price else None,
                        currency=item.currency.value if item.currency else None,
                        vendor=item.vendor,
                        category=item.category,
                        redaction_applied=False,
                        redaction_level="none",
                        accessible_fields=["item_name", "total_price", "unit_price", "currency", "vendor", "category"]
                    )

                secure_items.append(secure_item)

            # Create summary statistics
            summary_stats = {
                "total_items": len(extraction_result.pricing_items),
                "items_redacted": redaction_count,
                "redaction_percentage": (redaction_count / len(extraction_result.pricing_items) * 100) if extraction_result.pricing_items else 0,
                "average_confidence": extraction_result.average_confidence
            }

            # Redact summary stats if needed
            if security_context.user_role == UserRole.VIEWER:
                summary_stats = {
                    "total_items": len(extraction_result.pricing_items),
                    "data_available": "Limited view"
                }

            result = SecurePricingAnalysisResult(
                operation_type=PricingOperationType.EXTRACTION,
                secure_items=secure_items,
                summary_stats=summary_stats,
                user_role=security_context.user_role,
                data_sensitivity=security_context.data_sensitivity,
                redaction_summary={
                    "total_redactions": redaction_count,
                    "redaction_method": "pricing_masking"
                },
                audit_trail_id=""  # Will be set by caller
            )

            self._stats["total_redactions_applied"] += redaction_count

            return result

        except Exception as e:
            logger.error("Failed to apply redaction to extraction", error=str(e))
            raise

    def _apply_redaction_to_analysis(self, analysis_result: CompetitiveAnalysisResult,
                                   security_context: PricingSecurityContext) -> SecurePricingAnalysisResult:
        """Apply redaction to competitive analysis results."""

        try:
            # For competitive analysis, we mainly redact the summary statistics
            summary_stats = {}

            if security_context.user_role in [UserRole.ANALYST, UserRole.ADMIN]:
                # Full access to analysis results
                summary_stats = {
                    "total_items_analyzed": analysis_result.total_items_analyzed,
                    "vendors_analyzed": analysis_result.vendors_analyzed,
                    "outliers_detected": len(analysis_result.outliers),
                    "trends_identified": len(analysis_result.pricing_trends),
                    "analysis_confidence": analysis_result.analysis_confidence
                }
            else:
                # Limited access for viewers
                summary_stats = {
                    "analysis_available": "Contact analyst for details",
                    "summary_only": True
                }

            result = SecurePricingAnalysisResult(
                operation_type=PricingOperationType.ANALYSIS,
                secure_items=[],  # Analysis doesn't return individual items
                summary_stats=summary_stats,
                user_role=security_context.user_role,
                data_sensitivity=security_context.data_sensitivity,
                redaction_summary={
                    "analysis_redacted": security_context.user_role == UserRole.VIEWER,
                    "redaction_method": "role_based_filtering"
                },
                audit_trail_id=""  # Will be set by caller
            )

            return result

        except Exception as e:
            logger.error("Failed to apply redaction to analysis", error=str(e))
            raise

    def _needs_redaction(self, item: PricingItem, security_context: PricingSecurityContext) -> bool:
        """Determine if pricing item needs redaction."""

        # Always redact for viewers
        if security_context.user_role == UserRole.VIEWER:
            return True

        # Check data sensitivity
        if security_context.data_sensitivity in [
            PricingDataSensitivity.CONFIDENTIAL,
            PricingDataSensitivity.RESTRICTED
        ]:
            return security_context.user_role != UserRole.ADMIN

        return False

    def _apply_redaction_to_dashboard(self, dashboard_data: DashboardData,
                                    security_context: PricingSecurityContext) -> DashboardData:
        """Apply redaction to dashboard data."""

        try:
            # For now, return dashboard data as-is since role-based filtering
            # is already applied in the dashboard service
            return dashboard_data

        except Exception as e:
            logger.error("Failed to apply redaction to dashboard", error=str(e))
            raise

    def _log_audit_event(self, security_context: PricingSecurityContext,
                        operation: str, metadata: Dict[str, Any]) -> str:
        """Log audit event for pricing operation."""

        event_id = str(uuid4())

        # Log the audit event
        logger.info(
            "Pricing operation audit",
            event_id=event_id,
            event_type="pricing_operation",
            user_id=security_context.user_id,
            tenant_id=security_context.tenant_id,
            operation=operation,
            user_role=security_context.user_role.value,
            operation_type=security_context.operation_type.value,
            data_sensitivity=security_context.data_sensitivity.value,
            ip_address=security_context.ip_address,
            metadata=metadata
        )

        return event_id

    def _update_stats(self, operation_type: str, processing_time_ms: float) -> None:
        """Update service statistics."""

        self._stats["total_secure_operations"] += 1
        self._stats["total_processing_time_ms"] += processing_time_ms
        self._stats["total_audit_events"] += 1

    def create_session(self, user_id: str, user_role: UserRole) -> str:
        """Create a new secure session."""

        session_id = str(uuid4())
        expires_at = time.time() + (self.config.session_timeout_minutes * 60)

        self._active_sessions[session_id] = {
            "user_id": user_id,
            "user_role": user_role.value,
            "created_at": time.time(),
            "expires_at": expires_at
        }

        # Clean up expired sessions
        self._cleanup_expired_sessions()

        logger.info("Secure session created", user_id=user_id, session_id=session_id)
        return session_id

    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""

        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session in self._active_sessions.items()
            if session.get("expires_at", 0) <= current_time
        ]

        for session_id in expired_sessions:
            del self._active_sessions[session_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""

        return {
            **self._stats,
            "active_sessions": len(self._active_sessions),
            "config": self.config.dict()
        }


# Global service instance
_pricing_redaction_integration_service: Optional[PricingRedactionIntegrationService] = None


def get_pricing_redaction_integration_service() -> PricingRedactionIntegrationService:
    """Get or create the global pricing redaction integration service instance."""
    global _pricing_redaction_integration_service

    if _pricing_redaction_integration_service is None:
        _pricing_redaction_integration_service = PricingRedactionIntegrationService()

    return _pricing_redaction_integration_service


def reset_pricing_redaction_integration_service() -> None:
    """Reset the global pricing redaction integration service instance."""
    global _pricing_redaction_integration_service
    _pricing_redaction_integration_service = None
