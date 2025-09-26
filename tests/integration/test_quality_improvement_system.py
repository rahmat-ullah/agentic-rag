"""
Integration Tests for Sprint 6 Story 6-05: Automated Quality Improvement System

This module contains comprehensive integration tests for the quality improvement system,
including database operations, service functionality, API endpoints, and
end-to-end quality improvement workflows.
"""

import pytest
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, Any

from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from agentic_rag.models.tenant import Tenant
from agentic_rag.models.auth import User
from agentic_rag.models.quality_improvement import (
    QualityAssessment,
    QualityImprovement,
    QualityMonitoring,
    AutomationRule,
    QualityAlert,
    QualityDimension,
    QualityIssueType,
    ImprovementActionType,
    ImprovementStatus
)
from agentic_rag.models.feedback import UserFeedbackSubmission, FeedbackType
from agentic_rag.services.quality_improvement_service import QualityImprovementService
from agentic_rag.services.quality_automation_service import QualityAutomationService
from agentic_rag.schemas.quality_improvement import (
    CreateQualityAssessmentRequest,
    CreateQualityImprovementRequest,
    CreateQualityMonitoringRequest,
    CreateAutomationRuleRequest
)


class TestQualityImprovementDatabase:
    """Test quality improvement database operations."""
    
    def test_create_quality_assessment(self, db_session: Session, test_tenant: Tenant):
        """Test creating quality assessments."""
        assessment = QualityAssessment(
            tenant_id=test_tenant.id,
            target_type="content",
            target_id=uuid.uuid4(),
            overall_quality_score=0.75,
            accuracy_score=0.8,
            completeness_score=0.7,
            freshness_score=0.9,
            relevance_score=0.75,
            usability_score=0.6,
            assessment_method="automated_analysis",
            confidence_level=0.85,
            sample_size=100,
            assessment_date=datetime.utcnow(),
            quality_issues=["formatting_issues"],
            improvement_suggestions=["improve_formatting"]
        )
        
        db_session.add(assessment)
        db_session.commit()
        db_session.refresh(assessment)
        
        assert assessment.id is not None
        assert assessment.target_type == "content"
        assert assessment.overall_quality_score == 0.75
        assert assessment.tenant_id == test_tenant.id
    
    def test_create_quality_improvement(self, db_session: Session, test_tenant: Tenant):
        """Test creating quality improvements."""
        improvement = QualityImprovement(
            tenant_id=test_tenant.id,
            improvement_type=QualityIssueType.LOW_QUALITY_LINK,
            target_type="link",
            target_id=uuid.uuid4(),
            trigger_reason="confidence_score_below_threshold",
            trigger_threshold=0.5,
            improvement_action=ImprovementActionType.LINK_REVALIDATION,
            status=ImprovementStatus.PENDING,
            quality_before=0.4
        )
        
        db_session.add(improvement)
        db_session.commit()
        db_session.refresh(improvement)
        
        assert improvement.id is not None
        assert improvement.improvement_type == QualityIssueType.LOW_QUALITY_LINK
        assert improvement.improvement_action == ImprovementActionType.LINK_REVALIDATION
        assert improvement.status == ImprovementStatus.PENDING
    
    def test_create_quality_monitoring(self, db_session: Session, test_tenant: Tenant):
        """Test creating quality monitoring."""
        monitor = QualityMonitoring(
            tenant_id=test_tenant.id,
            monitor_name="Content Quality Monitor",
            monitor_type="threshold",
            target_type="content",
            quality_threshold=0.7,
            check_interval_minutes=30,
            alert_enabled=True,
            alert_severity="high",
            is_active=True
        )
        
        db_session.add(monitor)
        db_session.commit()
        db_session.refresh(monitor)
        
        assert monitor.id is not None
        assert monitor.monitor_name == "Content Quality Monitor"
        assert monitor.quality_threshold == 0.7
        assert monitor.is_active is True
    
    def test_create_automation_rule(self, db_session: Session, test_tenant: Tenant):
        """Test creating automation rules."""
        rule = AutomationRule(
            tenant_id=test_tenant.id,
            rule_name="Low Quality Link Auto-Fix",
            rule_type="quality_threshold",
            target_type="link",
            trigger_conditions={
                "confidence_score": {"operator": "<", "value": 0.5}
            },
            improvement_actions=["link_revalidation"],
            is_active=True,
            rule_priority=80
        )
        
        db_session.add(rule)
        db_session.commit()
        db_session.refresh(rule)
        
        assert rule.id is not None
        assert rule.rule_name == "Low Quality Link Auto-Fix"
        assert rule.rule_type == "quality_threshold"
        assert rule.is_active is True
    
    def test_create_quality_alert(self, db_session: Session, test_tenant: Tenant):
        """Test creating quality alerts."""
        alert = QualityAlert(
            tenant_id=test_tenant.id,
            alert_type="threshold_breach",
            alert_severity="high",
            alert_title="Quality Alert",
            alert_message="Quality threshold breached",
            target_type="content",
            target_id=uuid.uuid4(),
            quality_value=0.4,
            threshold_value=0.7,
            status="active"
        )
        
        db_session.add(alert)
        db_session.commit()
        db_session.refresh(alert)
        
        assert alert.id is not None
        assert alert.alert_type == "threshold_breach"
        assert alert.alert_severity == "high"
        assert alert.status == "active"


class TestQualityImprovementService:
    """Test quality improvement service functionality."""
    
    @pytest.fixture
    def quality_service(self, db_session: Session):
        """Create quality improvement service instance."""
        return QualityImprovementService(db_session)
    
    @pytest.mark.asyncio
    async def test_create_quality_assessment(self, quality_service: QualityImprovementService, test_tenant: Tenant):
        """Test creating quality assessment via service."""
        assessment_data = CreateQualityAssessmentRequest(
            target_type="content",
            target_id=uuid.uuid4(),
            overall_quality_score=0.75,
            accuracy_score=0.8,
            completeness_score=0.7,
            freshness_score=0.9,
            relevance_score=0.75,
            usability_score=0.6,
            assessment_method="automated_analysis",
            confidence_level=0.85,
            sample_size=100,
            quality_issues=["formatting_issues"],
            improvement_suggestions=["improve_formatting"]
        )
        
        assessment = await quality_service.create_quality_assessment(test_tenant.id, assessment_data)
        
        assert assessment.id is not None
        assert assessment.target_type == "content"
        assert assessment.overall_quality_score == 0.75
        assert assessment.tenant_id == test_tenant.id
    
    @pytest.mark.asyncio
    async def test_assess_quality(self, quality_service: QualityImprovementService, test_tenant: Tenant, db_session: Session):
        """Test quality assessment functionality."""
        target_id = uuid.uuid4()
        
        # Create some test feedback data
        feedback = UserFeedbackSubmission(
            tenant_id=test_tenant.id,
            user_id=uuid.uuid4(),
            feedback_type=FeedbackType.SEARCH_RESULT,
            target_id=target_id,
            rating=4,
            feedback_text="Good content",
            status="completed"
        )
        db_session.add(feedback)
        db_session.commit()
        
        # Assess quality
        quality_result = await quality_service.assess_quality(
            test_tenant.id, "content", target_id, "automated"
        )
        
        assert quality_result is not None
        assert 0.0 <= quality_result.overall_score <= 1.0
        assert isinstance(quality_result.dimension_scores, dict)
        assert isinstance(quality_result.quality_issues, list)
        assert isinstance(quality_result.improvement_suggestions, list)
        assert 0.0 <= quality_result.confidence_level <= 1.0
    
    @pytest.mark.asyncio
    async def test_detect_improvement_opportunities(self, quality_service: QualityImprovementService, test_tenant: Tenant, db_session: Session):
        """Test improvement opportunity detection."""
        # Create a low-quality assessment
        low_quality_assessment = QualityAssessment(
            tenant_id=test_tenant.id,
            target_type="link",
            target_id=uuid.uuid4(),
            overall_quality_score=0.4,  # Below warning threshold
            assessment_method="automated",
            assessment_date=datetime.utcnow()
        )
        db_session.add(low_quality_assessment)
        db_session.commit()
        
        # Detect opportunities
        opportunities = await quality_service.detect_improvement_opportunities(
            test_tenant.id, target_types=["link"], min_priority_score=0.3
        )
        
        assert len(opportunities) > 0
        assert any(opp.target_type == "link" for opp in opportunities)
        assert any(opp.issue_type == QualityIssueType.LOW_QUALITY_LINK for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_execute_improvement_action(self, quality_service: QualityImprovementService, test_tenant: Tenant, db_session: Session):
        """Test improvement action execution."""
        # Create an improvement
        improvement = QualityImprovement(
            tenant_id=test_tenant.id,
            improvement_type=QualityIssueType.LOW_QUALITY_LINK,
            target_type="link",
            target_id=uuid.uuid4(),
            trigger_reason="test_execution",
            improvement_action=ImprovementActionType.LINK_REVALIDATION,
            status=ImprovementStatus.PENDING,
            quality_before=0.4
        )
        db_session.add(improvement)
        db_session.commit()
        db_session.refresh(improvement)
        
        # Execute improvement
        success = await quality_service.execute_improvement_action(improvement.id)
        
        assert success is True
        
        # Refresh and check status
        db_session.refresh(improvement)
        assert improvement.status == ImprovementStatus.COMPLETED
        assert improvement.started_at is not None
        assert improvement.completed_at is not None


class TestQualityAutomationService:
    """Test quality automation service functionality."""
    
    @pytest.fixture
    def automation_service(self, db_session: Session):
        """Create quality automation service instance."""
        return QualityAutomationService(db_session)
    
    @pytest.mark.asyncio
    async def test_create_quality_monitoring(self, automation_service: QualityAutomationService, test_tenant: Tenant):
        """Test creating quality monitoring via service."""
        monitoring_data = CreateQualityMonitoringRequest(
            monitor_name="Test Monitor",
            monitor_type="threshold",
            target_type="content",
            quality_threshold=0.7,
            check_interval_minutes=30,
            alert_enabled=True,
            alert_severity="high"
        )
        
        monitor = await automation_service.create_quality_monitoring(test_tenant.id, monitoring_data)
        
        assert monitor.id is not None
        assert monitor.monitor_name == "Test Monitor"
        assert monitor.quality_threshold == 0.7
        assert monitor.is_active is True
    
    @pytest.mark.asyncio
    async def test_create_automation_rule(self, automation_service: QualityAutomationService, test_tenant: Tenant):
        """Test creating automation rule via service."""
        rule_data = CreateAutomationRuleRequest(
            rule_name="Test Rule",
            rule_type="quality_threshold",
            target_type="link",
            trigger_conditions={
                "confidence_score": {"operator": "<", "value": 0.5}
            },
            improvement_actions=["link_revalidation"],
            rule_priority=80
        )
        
        rule = await automation_service.create_automation_rule(test_tenant.id, rule_data)
        
        assert rule.id is not None
        assert rule.rule_name == "Test Rule"
        assert rule.rule_type == "quality_threshold"
        assert rule.is_active is True
    
    @pytest.mark.asyncio
    async def test_execute_monitoring_checks(self, automation_service: QualityAutomationService, test_tenant: Tenant, db_session: Session):
        """Test monitoring check execution."""
        # Create a monitor
        monitor = QualityMonitoring(
            tenant_id=test_tenant.id,
            monitor_name="Test Monitor",
            monitor_type="threshold",
            target_type="content",
            quality_threshold=0.7,
            is_active=True,
            next_check=datetime.utcnow() - timedelta(minutes=1)  # Due for check
        )
        db_session.add(monitor)
        db_session.commit()
        
        # Execute monitoring checks
        results = await automation_service.execute_monitoring_checks(test_tenant.id)
        
        assert len(results) > 0
        assert any(r.monitor_id == monitor.id for r in results)
    
    @pytest.mark.asyncio
    async def test_execute_automation_rules(self, automation_service: QualityAutomationService, test_tenant: Tenant, db_session: Session):
        """Test automation rule execution."""
        # Create an automation rule
        rule = AutomationRule(
            tenant_id=test_tenant.id,
            rule_name="Test Rule",
            rule_type="quality_threshold",
            target_type="link",
            trigger_conditions={
                "confidence_score": {"operator": "<", "value": 0.5}
            },
            improvement_actions=["link_revalidation"],
            is_active=True,
            dry_run_mode=True  # Use dry run for testing
        )
        db_session.add(rule)
        db_session.commit()
        
        # Execute automation rules
        results = await automation_service.execute_automation_rules(test_tenant.id)
        
        assert len(results) > 0
        assert any(r.rule_id == rule.id for r in results)
