"""
Audit Trail Implementation Service

This module provides comprehensive audit trail capabilities with event logging,
storage/retrieval, report generation, compliance reporting, and retention policies
for redaction and privacy protection activities.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, validator

from agentic_rag.config import get_settings
from agentic_rag.models.database import UserRole

logger = structlog.get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    
    # Access events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    
    # Redaction events
    PII_DETECTED = "pii_detected"
    CONTENT_REDACTED = "content_redacted"
    REDACTION_BYPASSED = "redaction_bypassed"
    
    # Policy events
    POLICY_APPLIED = "policy_applied"
    POLICY_VIOLATED = "policy_violated"
    RULE_MATCHED = "rule_matched"
    
    # System events
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    CONFIGURATION_CHANGED = "configuration_changed"
    
    # Compliance events
    DATA_EXPORT = "data_export"
    DATA_RETENTION = "data_retention"
    COMPLIANCE_REPORT = "compliance_report"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStandard(str, Enum):
    """Compliance standards for reporting."""
    
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"


@dataclass
class AuditContext:
    """Context information for audit events."""
    
    user_id: Optional[str] = None
    user_role: Optional[UserRole] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    tenant_id: Optional[str] = None
    document_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


class AuditEvent(BaseModel):
    """Individual audit event."""
    
    # Event identification
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique event ID")
    event_type: AuditEventType = Field(..., description="Type of audit event")
    severity: AuditSeverity = Field(default=AuditSeverity.MEDIUM, description="Event severity")
    
    # Event details
    message: str = Field(..., description="Human-readable event message")
    description: str = Field(default="", description="Detailed event description")
    
    # Context information
    user_id: Optional[str] = Field(default=None, description="User ID")
    user_role: Optional[UserRole] = Field(default=None, description="User role")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    request_id: Optional[str] = Field(default=None, description="Request ID")
    client_ip: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="User agent")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")
    
    # Resource information
    resource_type: Optional[str] = Field(default=None, description="Type of resource accessed")
    resource_id: Optional[str] = Field(default=None, description="Resource identifier")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    
    # Event data
    event_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event-specific data"
    )
    
    # Redaction information
    pii_types_detected: List[str] = Field(
        default_factory=list,
        description="Types of PII detected"
    )
    redaction_applied: bool = Field(
        default=False,
        description="Whether redaction was applied"
    )
    redaction_strategy: Optional[str] = Field(
        default=None,
        description="Redaction strategy used"
    )
    
    # Compliance information
    compliance_standards: List[ComplianceStandard] = Field(
        default_factory=list,
        description="Relevant compliance standards"
    )
    retention_period_days: Optional[int] = Field(
        default=None,
        description="Retention period in days"
    )
    
    # Timestamps
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Event expiration timestamp"
    )
    
    # Metadata
    source_service: str = Field(
        default="redaction_service",
        description="Service that generated the event"
    )
    version: str = Field(default="1.0", description="Event schema version")
    tags: List[str] = Field(default_factory=list, description="Event tags")


class AuditQuery(BaseModel):
    """Query parameters for audit trail search."""
    
    # Time range
    start_time: Optional[datetime] = Field(default=None, description="Start time for query")
    end_time: Optional[datetime] = Field(default=None, description="End time for query")
    
    # Filters
    event_types: Optional[List[AuditEventType]] = Field(
        default=None,
        description="Event types to include"
    )
    severities: Optional[List[AuditSeverity]] = Field(
        default=None,
        description="Severities to include"
    )
    user_ids: Optional[List[str]] = Field(default=None, description="User IDs to include")
    user_roles: Optional[List[UserRole]] = Field(default=None, description="User roles to include")
    tenant_ids: Optional[List[str]] = Field(default=None, description="Tenant IDs to include")
    document_ids: Optional[List[str]] = Field(default=None, description="Document IDs to include")
    
    # Search
    search_text: Optional[str] = Field(default=None, description="Text to search in messages")
    
    # Pagination
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    
    # Sorting
    sort_by: str = Field(default="timestamp", description="Field to sort by")
    sort_order: str = Field(default="desc", description="Sort order (asc/desc)")


class AuditReport(BaseModel):
    """Audit report with statistics and events."""
    
    # Report metadata
    report_id: str = Field(default_factory=lambda: str(uuid4()), description="Report ID")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation timestamp"
    )
    query: AuditQuery = Field(..., description="Query used to generate report")
    
    # Statistics
    total_events: int = Field(default=0, description="Total number of events")
    events_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Event counts by type"
    )
    events_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Event counts by severity"
    )
    events_by_user: Dict[str, int] = Field(
        default_factory=dict,
        description="Event counts by user"
    )
    
    # Compliance metrics
    compliance_violations: int = Field(
        default=0,
        description="Number of compliance violations"
    )
    redaction_events: int = Field(
        default=0,
        description="Number of redaction events"
    )
    access_denied_events: int = Field(
        default=0,
        description="Number of access denied events"
    )
    
    # Events
    events: List[AuditEvent] = Field(
        default_factory=list,
        description="Audit events in the report"
    )
    
    # Report summary
    summary: str = Field(default="", description="Report summary")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Security recommendations"
    )


class AuditTrailService:
    """Service for managing audit trail and compliance reporting."""
    
    def __init__(self):
        self.settings = get_settings()
        self._events: List[AuditEvent] = []  # In-memory storage for demo
        self._retention_policies: Dict[AuditEventType, int] = {}
        
        # Performance tracking
        self._stats = {
            "total_events_logged": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "total_queries": 0,
            "total_reports_generated": 0,
            "retention_cleanups": 0
        }
        
        # Initialize default retention policies
        self._initialize_retention_policies()
        
        # Log service start
        self.log_event(
            event_type=AuditEventType.SERVICE_STARTED,
            message="Audit trail service started",
            severity=AuditSeverity.LOW
        )
        
        logger.info("Audit trail service initialized")
    
    def _initialize_retention_policies(self) -> None:
        """Initialize default retention policies for different event types."""
        
        # Default retention periods in days
        self._retention_policies = {
            # Access events - 90 days
            AuditEventType.ACCESS_GRANTED: 90,
            AuditEventType.ACCESS_DENIED: 365,  # Keep access denials longer
            
            # Redaction events - 7 years for compliance
            AuditEventType.PII_DETECTED: 2555,  # 7 years
            AuditEventType.CONTENT_REDACTED: 2555,
            AuditEventType.REDACTION_BYPASSED: 2555,
            
            # Policy events - 1 year
            AuditEventType.POLICY_APPLIED: 365,
            AuditEventType.POLICY_VIOLATED: 2555,  # Keep violations longer
            AuditEventType.RULE_MATCHED: 365,
            
            # System events - 30 days
            AuditEventType.SERVICE_STARTED: 30,
            AuditEventType.SERVICE_STOPPED: 30,
            AuditEventType.CONFIGURATION_CHANGED: 365,
            
            # Compliance events - 7 years
            AuditEventType.DATA_EXPORT: 2555,
            AuditEventType.DATA_RETENTION: 2555,
            AuditEventType.COMPLIANCE_REPORT: 2555
        }
        
        logger.info("Retention policies initialized")

    def log_event(self, event_type: AuditEventType, message: str,
                  context: Optional[AuditContext] = None, **kwargs) -> str:
        """Log an audit event."""

        # Create event
        event_data = {
            "event_type": event_type,
            "message": message,
            **kwargs
        }

        # Add context if provided
        if context:
            event_data.update({
                "user_id": context.user_id,
                "user_role": context.user_role,
                "session_id": context.session_id,
                "request_id": context.request_id,
                "client_ip": context.client_ip,
                "user_agent": context.user_agent,
                "tenant_id": context.tenant_id,
                "document_id": context.document_id,
                "event_data": context.additional_data
            })

        # Set retention period
        retention_days = self._retention_policies.get(event_type, 365)
        expires_at = datetime.now(timezone.utc) + timedelta(days=retention_days)
        event_data["expires_at"] = expires_at
        event_data["retention_period_days"] = retention_days

        # Create event object
        event = AuditEvent(**event_data)

        # Store event
        self._events.append(event)

        # Update statistics
        self._stats["total_events_logged"] += 1
        self._stats["events_by_type"][event_type.value] = (
            self._stats["events_by_type"].get(event_type.value, 0) + 1
        )
        self._stats["events_by_severity"][event.severity.value] = (
            self._stats["events_by_severity"].get(event.severity.value, 0) + 1
        )

        # Log to structured logger
        logger.info(
            "Audit event logged",
            event_id=event.event_id,
            event_type=event_type.value,
            severity=event.severity.value,
            user_id=event.user_id,
            message=message
        )

        return event.event_id

    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events based on criteria."""

        start_time = time.time()

        try:
            # Start with all events
            filtered_events = self._events.copy()

            # Apply time range filter
            if query.start_time:
                filtered_events = [e for e in filtered_events if e.timestamp >= query.start_time]

            if query.end_time:
                filtered_events = [e for e in filtered_events if e.timestamp <= query.end_time]

            # Apply event type filter
            if query.event_types:
                filtered_events = [e for e in filtered_events if e.event_type in query.event_types]

            # Apply severity filter
            if query.severities:
                filtered_events = [e for e in filtered_events if e.severity in query.severities]

            # Apply user filters
            if query.user_ids:
                filtered_events = [e for e in filtered_events if e.user_id in query.user_ids]

            if query.user_roles:
                filtered_events = [e for e in filtered_events if e.user_role in query.user_roles]

            # Apply tenant filter
            if query.tenant_ids:
                filtered_events = [e for e in filtered_events if e.tenant_id in query.tenant_ids]

            # Apply document filter
            if query.document_ids:
                filtered_events = [e for e in filtered_events if e.document_id in query.document_ids]

            # Apply text search
            if query.search_text:
                search_lower = query.search_text.lower()
                filtered_events = [
                    e for e in filtered_events
                    if (search_lower in e.message.lower() or
                        search_lower in e.description.lower())
                ]

            # Sort events
            reverse_order = query.sort_order.lower() == "desc"
            if query.sort_by == "timestamp":
                filtered_events.sort(key=lambda e: e.timestamp, reverse=reverse_order)
            elif query.sort_by == "severity":
                severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                filtered_events.sort(
                    key=lambda e: severity_order.get(e.severity.value, 0),
                    reverse=reverse_order
                )
            elif query.sort_by == "event_type":
                filtered_events.sort(key=lambda e: e.event_type.value, reverse=reverse_order)

            # Apply pagination
            start_idx = query.offset
            end_idx = start_idx + query.limit
            paginated_events = filtered_events[start_idx:end_idx]

            # Update statistics
            self._stats["total_queries"] += 1

            query_time_ms = (time.time() - start_time) * 1000
            logger.debug(
                "Audit query completed",
                total_events=len(self._events),
                filtered_events=len(filtered_events),
                returned_events=len(paginated_events),
                query_time_ms=query_time_ms
            )

            return paginated_events

        except Exception as e:
            logger.error("Audit query failed", error=str(e))
            return []

    def generate_report(self, query: AuditQuery,
                       compliance_standards: Optional[List[ComplianceStandard]] = None) -> AuditReport:
        """Generate an audit report based on query criteria."""

        start_time = time.time()

        try:
            # Query events
            events = self.query_events(query)

            # Calculate statistics
            total_events = len(events)
            events_by_type = {}
            events_by_severity = {}
            events_by_user = {}

            compliance_violations = 0
            redaction_events = 0
            access_denied_events = 0

            for event in events:
                # Count by type
                events_by_type[event.event_type.value] = (
                    events_by_type.get(event.event_type.value, 0) + 1
                )

                # Count by severity
                events_by_severity[event.severity.value] = (
                    events_by_severity.get(event.severity.value, 0) + 1
                )

                # Count by user
                if event.user_id:
                    events_by_user[event.user_id] = (
                        events_by_user.get(event.user_id, 0) + 1
                    )

                # Count specific event types
                if event.event_type == AuditEventType.POLICY_VIOLATED:
                    compliance_violations += 1
                elif event.event_type in [AuditEventType.CONTENT_REDACTED, AuditEventType.PII_DETECTED]:
                    redaction_events += 1
                elif event.event_type == AuditEventType.ACCESS_DENIED:
                    access_denied_events += 1

            # Generate summary
            summary = self._generate_report_summary(
                total_events, compliance_violations, redaction_events, access_denied_events
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                events, compliance_violations, access_denied_events
            )

            # Create report
            report = AuditReport(
                query=query,
                total_events=total_events,
                events_by_type=events_by_type,
                events_by_severity=events_by_severity,
                events_by_user=events_by_user,
                compliance_violations=compliance_violations,
                redaction_events=redaction_events,
                access_denied_events=access_denied_events,
                events=events,
                summary=summary,
                recommendations=recommendations
            )

            # Update statistics
            self._stats["total_reports_generated"] += 1

            # Log report generation
            self.log_event(
                event_type=AuditEventType.COMPLIANCE_REPORT,
                message=f"Audit report generated with {total_events} events",
                severity=AuditSeverity.LOW,
                event_data={
                    "report_id": report.report_id,
                    "total_events": total_events,
                    "compliance_violations": compliance_violations
                }
            )

            generation_time_ms = (time.time() - start_time) * 1000
            logger.info(
                "Audit report generated",
                report_id=report.report_id,
                total_events=total_events,
                generation_time_ms=generation_time_ms
            )

            return report

        except Exception as e:
            logger.error("Report generation failed", error=str(e))
            raise

    def cleanup_expired_events(self) -> int:
        """Clean up expired audit events based on retention policies."""

        start_time = time.time()
        current_time = datetime.now(timezone.utc)

        initial_count = len(self._events)

        # Remove expired events
        self._events = [
            event for event in self._events
            if not event.expires_at or event.expires_at > current_time
        ]

        cleaned_count = initial_count - len(self._events)

        if cleaned_count > 0:
            # Log cleanup event
            self.log_event(
                event_type=AuditEventType.DATA_RETENTION,
                message=f"Cleaned up {cleaned_count} expired audit events",
                severity=AuditSeverity.LOW,
                event_data={
                    "events_cleaned": cleaned_count,
                    "events_remaining": len(self._events)
                }
            )

            self._stats["retention_cleanups"] += 1

        cleanup_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "Audit event cleanup completed",
            events_cleaned=cleaned_count,
            events_remaining=len(self._events),
            cleanup_time_ms=cleanup_time_ms
        )

        return cleaned_count

    def _generate_report_summary(self, total_events: int, violations: int,
                                redactions: int, access_denied: int) -> str:
        """Generate a summary for the audit report."""

        summary_parts = [
            f"Audit report contains {total_events} events"
        ]

        if violations > 0:
            summary_parts.append(f"{violations} compliance violations detected")

        if redactions > 0:
            summary_parts.append(f"{redactions} redaction events recorded")

        if access_denied > 0:
            summary_parts.append(f"{access_denied} access denied events")

        if violations == 0 and access_denied == 0:
            summary_parts.append("No security violations detected")

        return ". ".join(summary_parts) + "."

    def _generate_recommendations(self, events: List[AuditEvent],
                                 violations: int, access_denied: int) -> List[str]:
        """Generate security recommendations based on audit events."""

        recommendations = []

        # Check for high violation rate
        if violations > len(events) * 0.1:  # More than 10% violations
            recommendations.append(
                "High rate of compliance violations detected. Review and update security policies."
            )

        # Check for high access denial rate
        if access_denied > len(events) * 0.05:  # More than 5% access denied
            recommendations.append(
                "High rate of access denials. Review user permissions and access controls."
            )

        # Check for critical events
        critical_events = [e for e in events if e.severity == AuditSeverity.CRITICAL]
        if critical_events:
            recommendations.append(
                f"{len(critical_events)} critical security events require immediate attention."
            )

        # Check for redaction bypasses
        bypass_events = [e for e in events if e.event_type == AuditEventType.REDACTION_BYPASSED]
        if bypass_events:
            recommendations.append(
                f"{len(bypass_events)} redaction bypass events detected. Review bypass policies."
            )

        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("No immediate security concerns identified.")

        return recommendations

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail service statistics."""

        current_time = datetime.now(timezone.utc)

        # Calculate event age distribution
        age_distribution = {"last_hour": 0, "last_day": 0, "last_week": 0, "older": 0}

        for event in self._events:
            age = current_time - event.timestamp
            if age <= timedelta(hours=1):
                age_distribution["last_hour"] += 1
            elif age <= timedelta(days=1):
                age_distribution["last_day"] += 1
            elif age <= timedelta(weeks=1):
                age_distribution["last_week"] += 1
            else:
                age_distribution["older"] += 1

        return {
            **self._stats,
            "current_events": len(self._events),
            "age_distribution": age_distribution,
            "retention_policies": len(self._retention_policies),
            "oldest_event": min([e.timestamp for e in self._events]) if self._events else None,
            "newest_event": max([e.timestamp for e in self._events]) if self._events else None
        }

    def export_events(self, query: AuditQuery, format: str = "json") -> str:
        """Export audit events in specified format."""

        events = self.query_events(query)

        if format.lower() == "json":
            # Convert events to JSON
            events_data = [event.model_dump() for event in events]
            return json.dumps(events_data, indent=2, default=str)

        elif format.lower() == "csv":
            # Convert events to CSV
            if not events:
                return "No events to export"

            # CSV headers
            headers = [
                "event_id", "timestamp", "event_type", "severity", "message",
                "user_id", "user_role", "client_ip", "resource_type", "resource_id"
            ]

            lines = [",".join(headers)]

            for event in events:
                row = [
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.severity.value,
                    f'"{event.message}"',  # Quote message to handle commas
                    event.user_id or "",
                    event.user_role.value if event.user_role else "",
                    event.client_ip or "",
                    event.resource_type or "",
                    event.resource_id or ""
                ]
                lines.append(",".join(str(field) for field in row))

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def set_retention_policy(self, event_type: AuditEventType, retention_days: int) -> None:
        """Set retention policy for an event type."""

        self._retention_policies[event_type] = retention_days

        self.log_event(
            event_type=AuditEventType.CONFIGURATION_CHANGED,
            message=f"Retention policy updated for {event_type.value}: {retention_days} days",
            severity=AuditSeverity.LOW,
            event_data={
                "event_type": event_type.value,
                "retention_days": retention_days
            }
        )

        logger.info(
            "Retention policy updated",
            event_type=event_type.value,
            retention_days=retention_days
        )


# Global service instance
_audit_trail_service: Optional[AuditTrailService] = None


def get_audit_trail_service() -> AuditTrailService:
    """Get or create the global audit trail service instance."""
    global _audit_trail_service

    if _audit_trail_service is None:
        _audit_trail_service = AuditTrailService()

    return _audit_trail_service


def reset_audit_trail_service() -> None:
    """Reset the global audit trail service instance."""
    global _audit_trail_service
    if _audit_trail_service:
        _audit_trail_service.log_event(
            event_type=AuditEventType.SERVICE_STOPPED,
            message="Audit trail service stopped",
            severity=AuditSeverity.LOW
        )
    _audit_trail_service = None
