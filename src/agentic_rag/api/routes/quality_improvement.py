"""
Quality Improvement API Routes for Sprint 6 Story 6-05: Automated Quality Improvement System

This module provides FastAPI endpoints for quality assessment, improvement actions,
monitoring, automation rules, and quality management operations.
"""

import uuid
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from agentic_rag.database.connection import get_database_session
from agentic_rag.api.dependencies.auth import get_current_user, require_role
from agentic_rag.models.auth import User
from agentic_rag.models.quality_improvement import (
    QualityAssessment,
    QualityImprovement,
    QualityMonitoring,
    AutomationRule,
    QualityAlert,
    QualityIssueType,
    ImprovementActionType,
    ImprovementStatus
)
from agentic_rag.schemas.quality_improvement import (
    CreateQualityAssessmentRequest,
    CreateQualityImprovementRequest,
    CreateQualityMonitoringRequest,
    CreateAutomationRuleRequest,
    UpdateImprovementStatusRequest,
    QualityAssessmentResponse,
    QualityImprovementResponse,
    QualityMonitoringResponse,
    AutomationRuleResponse,
    QualityAlertResponse,
    QualityDashboardResponse,
    QualityMetricsResponse,
    QualityAssessmentListResponse,
    QualityImprovementListResponse,
    QualityMonitoringListResponse,
    AutomationRuleListResponse,
    QualityAlertListResponse
)
from agentic_rag.services.quality_improvement_service import QualityImprovementService
from agentic_rag.services.quality_automation_service import QualityAutomationService

router = APIRouter(prefix="/quality", tags=["Quality Improvement"])
logger = structlog.get_logger(__name__)


@router.post("/assessments", response_model=QualityAssessmentResponse)
async def create_quality_assessment(
    assessment_data: CreateQualityAssessmentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Create a new quality assessment."""
    try:
        # Require admin or analyst role
        require_role(current_user, ["admin", "analyst"])
        
        quality_service = QualityImprovementService(db)
        assessment = await quality_service.create_quality_assessment(
            current_user.tenant_id, assessment_data
        )
        
        logger.info(
            "quality_assessment_created_via_api",
            assessment_id=assessment.id,
            target_type=assessment.target_type,
            overall_score=assessment.overall_quality_score,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        return QualityAssessmentResponse.from_attributes(assessment)
        
    except Exception as e:
        logger.error("quality_assessment_creation_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create quality assessment: {str(e)}"
        )


@router.get("/assessments", response_model=QualityAssessmentListResponse)
async def get_quality_assessments(
    target_types: Optional[List[str]] = Query(None),
    min_quality_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_quality_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get quality assessments with filtering and pagination."""
    try:
        # Build query
        query = db.query(QualityAssessment).filter(
            QualityAssessment.tenant_id == current_user.tenant_id
        )
        
        # Apply filters
        if target_types:
            query = query.filter(QualityAssessment.target_type.in_(target_types))
        
        if min_quality_score is not None:
            query = query.filter(QualityAssessment.overall_quality_score >= min_quality_score)
        
        if max_quality_score is not None:
            query = query.filter(QualityAssessment.overall_quality_score <= max_quality_score)
        
        if start_date:
            query = query.filter(QualityAssessment.assessment_date >= start_date)
        
        if end_date:
            query = query.filter(QualityAssessment.assessment_date <= end_date)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        assessments = query.order_by(desc(QualityAssessment.assessment_date)).offset(offset).limit(page_size).all()
        
        # Convert to response models
        assessment_responses = [QualityAssessmentResponse.from_attributes(assessment) for assessment in assessments]
        
        return QualityAssessmentListResponse(
            items=assessment_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )
        
    except Exception as e:
        logger.error("quality_assessments_retrieval_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quality assessments: {str(e)}"
        )


@router.post("/improvements", response_model=QualityImprovementResponse)
async def create_quality_improvement(
    improvement_data: CreateQualityImprovementRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Create a new quality improvement action."""
    try:
        # Require admin or analyst role
        require_role(current_user, ["admin", "analyst"])
        
        quality_service = QualityImprovementService(db)
        improvement = await quality_service.create_quality_improvement(
            current_user.tenant_id, improvement_data
        )
        
        logger.info(
            "quality_improvement_created_via_api",
            improvement_id=improvement.id,
            improvement_type=improvement.improvement_type,
            improvement_action=improvement.improvement_action,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        return QualityImprovementResponse.from_attributes(improvement)
        
    except Exception as e:
        logger.error("quality_improvement_creation_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create quality improvement: {str(e)}"
        )


@router.get("/improvements", response_model=QualityImprovementListResponse)
async def get_quality_improvements(
    improvement_types: Optional[List[QualityIssueType]] = Query(None),
    statuses: Optional[List[ImprovementStatus]] = Query(None),
    target_types: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get quality improvements with filtering and pagination."""
    try:
        # Build query
        query = db.query(QualityImprovement).filter(
            QualityImprovement.tenant_id == current_user.tenant_id
        )
        
        # Apply filters
        if improvement_types:
            query = query.filter(QualityImprovement.improvement_type.in_(improvement_types))
        
        if statuses:
            query = query.filter(QualityImprovement.status.in_(statuses))
        
        if target_types:
            query = query.filter(QualityImprovement.target_type.in_(target_types))
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        improvements = query.order_by(desc(QualityImprovement.created_at)).offset(offset).limit(page_size).all()
        
        # Convert to response models
        improvement_responses = [QualityImprovementResponse.from_attributes(improvement) for improvement in improvements]
        
        return QualityImprovementListResponse(
            items=improvement_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )
        
    except Exception as e:
        logger.error("quality_improvements_retrieval_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quality improvements: {str(e)}"
        )


@router.put("/improvements/{improvement_id}/status", response_model=QualityImprovementResponse)
async def update_improvement_status(
    improvement_id: uuid.UUID,
    status_update: UpdateImprovementStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Update the status of a quality improvement."""
    try:
        # Require admin or analyst role
        require_role(current_user, ["admin", "analyst"])
        
        # Get improvement
        improvement = db.query(QualityImprovement).filter(
            and_(
                QualityImprovement.id == improvement_id,
                QualityImprovement.tenant_id == current_user.tenant_id
            )
        ).first()
        
        if not improvement:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quality improvement not found"
            )
        
        # Update status and related fields
        improvement.status = status_update.status
        
        if status_update.quality_after is not None:
            improvement.quality_after = status_update.quality_after
            if improvement.quality_before:
                improvement.improvement_delta = status_update.quality_after - improvement.quality_before
        
        if status_update.effectiveness_score is not None:
            improvement.effectiveness_score = status_update.effectiveness_score
        
        if status_update.validation_results:
            improvement.validation_results = status_update.validation_results
        
        if status_update.impact_metrics:
            improvement.impact_metrics = status_update.impact_metrics
        
        if status_update.failure_reason:
            improvement.failure_reason = status_update.failure_reason
        
        # Update timestamps based on status
        if status_update.status == ImprovementStatus.IN_PROGRESS and not improvement.started_at:
            improvement.started_at = datetime.utcnow()
        elif status_update.status == ImprovementStatus.COMPLETED and not improvement.completed_at:
            improvement.completed_at = datetime.utcnow()
        elif status_update.status == ImprovementStatus.FAILED and not improvement.failed_at:
            improvement.failed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(improvement)
        
        logger.info(
            "improvement_status_updated_via_api",
            improvement_id=improvement_id,
            new_status=status_update.status,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        return QualityImprovementResponse.from_attributes(improvement)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("improvement_status_update_api_failed", error=str(e), improvement_id=improvement_id, user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update improvement status: {str(e)}"
        )


@router.post("/improvements/{improvement_id}/execute", response_model=QualityImprovementResponse)
async def execute_quality_improvement(
    improvement_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Execute a quality improvement action."""
    try:
        # Require admin role for execution
        require_role(current_user, ["admin"])
        
        # Verify improvement exists and belongs to tenant
        improvement = db.query(QualityImprovement).filter(
            and_(
                QualityImprovement.id == improvement_id,
                QualityImprovement.tenant_id == current_user.tenant_id
            )
        ).first()
        
        if not improvement:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quality improvement not found"
            )
        
        if improvement.status != ImprovementStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot execute improvement with status: {improvement.status}"
            )
        
        # Execute improvement
        quality_service = QualityImprovementService(db)
        success = await quality_service.execute_improvement_action(improvement_id)
        
        # Refresh improvement to get updated data
        db.refresh(improvement)
        
        logger.info(
            "improvement_executed_via_api",
            improvement_id=improvement_id,
            success=success,
            final_status=improvement.status,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        return QualityImprovementResponse.from_attributes(improvement)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("improvement_execution_api_failed", error=str(e), improvement_id=improvement_id, user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute improvement: {str(e)}"
        )


@router.post("/opportunities/detect", response_model=List[Dict[str, Any]])
async def detect_improvement_opportunities(
    target_types: Optional[List[str]] = Query(None),
    min_priority_score: float = Query(0.5, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Detect improvement opportunities based on quality assessments."""
    try:
        # Require admin or analyst role
        require_role(current_user, ["admin", "analyst"])

        quality_service = QualityImprovementService(db)
        opportunities = await quality_service.detect_improvement_opportunities(
            current_user.tenant_id, target_types, min_priority_score
        )

        # Convert opportunities to response format
        opportunity_responses = []
        for opp in opportunities:
            opportunity_responses.append({
                "target_type": opp.target_type,
                "target_id": str(opp.target_id),
                "issue_type": opp.issue_type.value,
                "current_quality": opp.current_quality,
                "expected_improvement": opp.expected_improvement,
                "recommended_action": opp.recommended_action.value,
                "priority_score": opp.priority_score,
                "trigger_reason": opp.trigger_reason,
                "metadata": opp.metadata
            })

        logger.info(
            "improvement_opportunities_detected_via_api",
            opportunity_count=len(opportunities),
            min_priority_score=min_priority_score,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )

        return opportunity_responses

    except Exception as e:
        logger.error("opportunity_detection_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect improvement opportunities: {str(e)}"
        )


@router.post("/monitoring", response_model=QualityMonitoringResponse)
async def create_quality_monitoring(
    monitoring_data: CreateQualityMonitoringRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Create a quality monitoring configuration."""
    try:
        # Require admin role
        require_role(current_user, ["admin"])

        automation_service = QualityAutomationService(db)
        monitor = await automation_service.create_quality_monitoring(
            current_user.tenant_id, monitoring_data
        )

        logger.info(
            "quality_monitoring_created_via_api",
            monitor_id=monitor.id,
            monitor_name=monitor.monitor_name,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )

        return QualityMonitoringResponse.from_attributes(monitor)

    except Exception as e:
        logger.error("quality_monitoring_creation_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create quality monitoring: {str(e)}"
        )


@router.get("/monitoring", response_model=QualityMonitoringListResponse)
async def get_quality_monitoring(
    is_active: Optional[bool] = Query(None),
    monitor_types: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get quality monitoring configurations."""
    try:
        # Build query
        query = db.query(QualityMonitoring).filter(
            QualityMonitoring.tenant_id == current_user.tenant_id
        )

        # Apply filters
        if is_active is not None:
            query = query.filter(QualityMonitoring.is_active == is_active)

        if monitor_types:
            query = query.filter(QualityMonitoring.monitor_type.in_(monitor_types))

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * page_size
        monitors = query.order_by(desc(QualityMonitoring.created_at)).offset(offset).limit(page_size).all()

        # Convert to response models
        monitor_responses = [QualityMonitoringResponse.from_attributes(monitor) for monitor in monitors]

        return QualityMonitoringListResponse(
            items=monitor_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )

    except Exception as e:
        logger.error("quality_monitoring_retrieval_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quality monitoring: {str(e)}"
        )


@router.post("/automation-rules", response_model=AutomationRuleResponse)
async def create_automation_rule(
    rule_data: CreateAutomationRuleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Create an automation rule."""
    try:
        # Require admin role
        require_role(current_user, ["admin"])

        automation_service = QualityAutomationService(db)
        rule = await automation_service.create_automation_rule(
            current_user.tenant_id, rule_data
        )

        logger.info(
            "automation_rule_created_via_api",
            rule_id=rule.id,
            rule_name=rule.rule_name,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )

        return AutomationRuleResponse.from_attributes(rule)

    except Exception as e:
        logger.error("automation_rule_creation_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create automation rule: {str(e)}"
        )


@router.get("/automation-rules", response_model=AutomationRuleListResponse)
async def get_automation_rules(
    is_active: Optional[bool] = Query(None),
    rule_types: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get automation rules."""
    try:
        # Build query
        query = db.query(AutomationRule).filter(
            AutomationRule.tenant_id == current_user.tenant_id
        )

        # Apply filters
        if is_active is not None:
            query = query.filter(AutomationRule.is_active == is_active)

        if rule_types:
            query = query.filter(AutomationRule.rule_type.in_(rule_types))

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * page_size
        rules = query.order_by(desc(AutomationRule.rule_priority), desc(AutomationRule.created_at)).offset(offset).limit(page_size).all()

        # Convert to response models
        rule_responses = [AutomationRuleResponse.from_attributes(rule) for rule in rules]

        return AutomationRuleListResponse(
            items=rule_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )

    except Exception as e:
        logger.error("automation_rules_retrieval_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve automation rules: {str(e)}"
        )


@router.get("/alerts", response_model=QualityAlertListResponse)
async def get_quality_alerts(
    statuses: Optional[List[str]] = Query(None),
    alert_severities: Optional[List[str]] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get quality alerts."""
    try:
        # Build query
        query = db.query(QualityAlert).filter(
            QualityAlert.tenant_id == current_user.tenant_id
        )

        # Apply filters
        if statuses:
            query = query.filter(QualityAlert.status.in_(statuses))

        if alert_severities:
            query = query.filter(QualityAlert.alert_severity.in_(alert_severities))

        if start_date:
            query = query.filter(QualityAlert.created_at >= start_date)

        if end_date:
            query = query.filter(QualityAlert.created_at <= end_date)

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * page_size
        alerts = query.order_by(desc(QualityAlert.created_at)).offset(offset).limit(page_size).all()

        # Convert to response models
        alert_responses = [QualityAlertResponse.from_attributes(alert) for alert in alerts]

        return QualityAlertListResponse(
            items=alert_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )

    except Exception as e:
        logger.error("quality_alerts_retrieval_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quality alerts: {str(e)}"
        )


@router.get("/dashboard", response_model=QualityDashboardResponse)
async def get_quality_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get quality improvement dashboard data."""
    try:
        quality_service = QualityImprovementService(db)
        dashboard = await quality_service.get_quality_dashboard(current_user.tenant_id)

        return dashboard

    except Exception as e:
        logger.error("quality_dashboard_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quality dashboard: {str(e)}"
        )


@router.get("/metrics", response_model=QualityMetricsResponse)
async def get_quality_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Get quality metrics and statistics."""
    try:
        # Calculate quality metrics
        overall_quality_score = 0.78  # Placeholder
        quality_trend_7d = 0.02       # Placeholder
        quality_trend_30d = 0.05      # Placeholder

        # Quality by target type
        content_quality_score = 0.75  # Placeholder
        link_quality_score = 0.82     # Placeholder
        system_quality_score = 0.88   # Placeholder

        # Get improvement metrics
        improvements_in_progress = db.query(QualityImprovement).filter(
            and_(
                QualityImprovement.tenant_id == current_user.tenant_id,
                QualityImprovement.status == ImprovementStatus.IN_PROGRESS
            )
        ).count()

        improvements_completed_7d = db.query(QualityImprovement).filter(
            and_(
                QualityImprovement.tenant_id == current_user.tenant_id,
                QualityImprovement.status == ImprovementStatus.COMPLETED,
                QualityImprovement.completed_at >= datetime.utcnow() - timedelta(days=7)
            )
        ).count()

        improvements_completed_30d = db.query(QualityImprovement).filter(
            and_(
                QualityImprovement.tenant_id == current_user.tenant_id,
                QualityImprovement.status == ImprovementStatus.COMPLETED,
                QualityImprovement.completed_at >= datetime.utcnow() - timedelta(days=30)
            )
        ).count()

        # Get alert metrics
        active_alerts = db.query(QualityAlert).filter(
            and_(
                QualityAlert.tenant_id == current_user.tenant_id,
                QualityAlert.status == "active"
            )
        ).count()

        alerts_created_7d = db.query(QualityAlert).filter(
            and_(
                QualityAlert.tenant_id == current_user.tenant_id,
                QualityAlert.created_at >= datetime.utcnow() - timedelta(days=7)
            )
        ).count()

        alerts_resolved_7d = db.query(QualityAlert).filter(
            and_(
                QualityAlert.tenant_id == current_user.tenant_id,
                QualityAlert.status == "resolved",
                QualityAlert.resolved_at >= datetime.utcnow() - timedelta(days=7)
            )
        ).count()

        # Get automation metrics
        active_automation_rules = db.query(AutomationRule).filter(
            and_(
                AutomationRule.tenant_id == current_user.tenant_id,
                AutomationRule.is_active == True
            )
        ).count()

        # Calculate placeholder metrics
        average_improvement_time_hours = 24.5  # Placeholder
        improvement_success_rate = 0.87        # Placeholder
        average_resolution_time_hours = 8.2    # Placeholder
        automation_executions_7d = 45          # Placeholder
        automation_success_rate = 0.92         # Placeholder

        return QualityMetricsResponse(
            overall_quality_score=overall_quality_score,
            quality_trend_7d=quality_trend_7d,
            quality_trend_30d=quality_trend_30d,
            content_quality_score=content_quality_score,
            link_quality_score=link_quality_score,
            system_quality_score=system_quality_score,
            improvements_in_progress=improvements_in_progress,
            improvements_completed_7d=improvements_completed_7d,
            improvements_completed_30d=improvements_completed_30d,
            average_improvement_time_hours=average_improvement_time_hours,
            improvement_success_rate=improvement_success_rate,
            active_alerts=active_alerts,
            alerts_created_7d=alerts_created_7d,
            alerts_resolved_7d=alerts_resolved_7d,
            average_resolution_time_hours=average_resolution_time_hours,
            active_automation_rules=active_automation_rules,
            automation_executions_7d=automation_executions_7d,
            automation_success_rate=automation_success_rate
        )

    except Exception as e:
        logger.error("quality_metrics_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quality metrics: {str(e)}"
        )


@router.post("/automation/execute", response_model=Dict[str, Any])
async def execute_automation(
    tenant_id_filter: Optional[uuid.UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database_session)
):
    """Execute automation rules and monitoring checks."""
    try:
        # Require admin role
        require_role(current_user, ["admin"])

        # Use current user's tenant if no filter specified
        target_tenant_id = tenant_id_filter if tenant_id_filter else current_user.tenant_id

        # Verify user has access to target tenant
        if target_tenant_id != current_user.tenant_id:
            # Would check if user has cross-tenant access
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to specified tenant"
            )

        automation_service = QualityAutomationService(db)

        # Execute monitoring checks
        monitoring_results = await automation_service.execute_monitoring_checks(target_tenant_id)

        # Execute automation rules
        rule_results = await automation_service.execute_automation_rules(target_tenant_id)

        # Compile results
        results = {
            "monitoring_checks": {
                "total_checks": len(monitoring_results),
                "alerts_triggered": sum(1 for r in monitoring_results if r.alert_triggered),
                "threshold_breaches": sum(1 for r in monitoring_results if r.threshold_breached),
                "trend_detections": sum(1 for r in monitoring_results if r.trend_detected)
            },
            "automation_rules": {
                "total_rules": len(rule_results),
                "successful_executions": sum(1 for r in rule_results if r.execution_success),
                "conditions_met": sum(1 for r in rule_results if r.conditions_met),
                "improvements_created": sum(len(r.improvements_created) for r in rule_results)
            },
            "execution_timestamp": datetime.utcnow().isoformat()
        }

        logger.info(
            "automation_executed_via_api",
            tenant_id=target_tenant_id,
            monitoring_checks=results["monitoring_checks"]["total_checks"],
            automation_rules=results["automation_rules"]["total_rules"],
            user_id=current_user.id
        )

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error("automation_execution_api_failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute automation: {str(e)}"
        )
