"""
Quality Improvement Schemas for Sprint 6 Story 6-05: Automated Quality Improvement System

This module defines Pydantic schemas for quality improvement API operations,
including quality assessments, improvement actions, monitoring, and automation rules.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator

from agentic_rag.models.quality_improvement import (
    QualityDimension,
    QualityIssueType,
    ImprovementActionType,
    ImprovementStatus
)


# Request Schemas

class CreateQualityAssessmentRequest(BaseModel):
    """Request schema for creating quality assessments."""
    
    target_type: str = Field(..., description="Type of target being assessed")
    target_id: uuid.UUID = Field(..., description="ID of the target being assessed")
    overall_quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Accuracy dimension score")
    completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Completeness dimension score")
    freshness_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Freshness dimension score")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance dimension score")
    usability_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Usability dimension score")
    assessment_method: str = Field(..., description="Method used for assessment")
    confidence_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in assessment")
    sample_size: Optional[int] = Field(None, ge=1, description="Sample size for assessment")
    dimension_weights: Optional[Dict[str, float]] = Field(None, description="Weights for quality dimensions")
    dimension_scores: Optional[Dict[str, float]] = Field(None, description="Individual dimension scores")
    assessment_context: Optional[Dict[str, Any]] = Field(None, description="Additional assessment context")
    quality_issues: Optional[List[str]] = Field(None, description="Identified quality issues")
    improvement_suggestions: Optional[List[str]] = Field(None, description="Suggested improvements")
    
    class Config:
        schema_extra = {
            "example": {
                "target_type": "content",
                "target_id": "123e4567-e89b-12d3-a456-426614174000",
                "overall_quality_score": 0.75,
                "accuracy_score": 0.8,
                "completeness_score": 0.7,
                "freshness_score": 0.9,
                "relevance_score": 0.75,
                "usability_score": 0.6,
                "assessment_method": "automated_analysis",
                "confidence_level": 0.85,
                "sample_size": 100,
                "quality_issues": ["outdated_information", "formatting_issues"],
                "improvement_suggestions": ["update_content", "improve_formatting"]
            }
        }


class CreateQualityImprovementRequest(BaseModel):
    """Request schema for creating quality improvements."""
    
    improvement_type: QualityIssueType = Field(..., description="Type of quality issue")
    target_type: str = Field(..., description="Type of target for improvement")
    target_id: uuid.UUID = Field(..., description="ID of the target for improvement")
    trigger_reason: str = Field(..., description="Reason that triggered the improvement")
    trigger_threshold: Optional[float] = Field(None, description="Threshold that was breached")
    trigger_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional trigger metadata")
    improvement_action: ImprovementActionType = Field(..., description="Action to take for improvement")
    action_parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the improvement action")
    quality_before: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score before improvement")
    
    class Config:
        schema_extra = {
            "example": {
                "improvement_type": "low_quality_link",
                "target_type": "link",
                "target_id": "123e4567-e89b-12d3-a456-426614174000",
                "trigger_reason": "confidence_score_below_threshold",
                "trigger_threshold": 0.5,
                "improvement_action": "link_revalidation",
                "quality_before": 0.4
            }
        }


class CreateQualityMonitoringRequest(BaseModel):
    """Request schema for creating quality monitoring."""
    
    monitor_name: str = Field(..., description="Name of the monitor")
    monitor_type: str = Field(..., description="Type of monitoring")
    target_type: str = Field(..., description="Type of target to monitor")
    quality_threshold: Optional[float] = Field(None, description="Quality threshold for alerts")
    trend_threshold: Optional[float] = Field(None, description="Trend threshold for alerts")
    pattern_rules: Optional[Dict[str, Any]] = Field(None, description="Pattern detection rules")
    alert_conditions: Optional[Dict[str, Any]] = Field(None, description="Conditions for alerts")
    check_interval_minutes: int = Field(60, ge=1, description="Check interval in minutes")
    alert_enabled: bool = Field(True, description="Whether alerts are enabled")
    alert_recipients: Optional[List[str]] = Field(None, description="Alert recipients")
    alert_severity: str = Field("medium", description="Alert severity level")
    
    class Config:
        schema_extra = {
            "example": {
                "monitor_name": "Content Quality Monitor",
                "monitor_type": "threshold",
                "target_type": "content",
                "quality_threshold": 0.7,
                "check_interval_minutes": 30,
                "alert_enabled": True,
                "alert_severity": "high"
            }
        }


class CreateAutomationRuleRequest(BaseModel):
    """Request schema for creating automation rules."""
    
    rule_name: str = Field(..., description="Name of the automation rule")
    rule_type: str = Field(..., description="Type of automation rule")
    target_type: str = Field(..., description="Type of target for the rule")
    trigger_conditions: Dict[str, Any] = Field(..., description="Conditions that trigger the rule")
    condition_logic: str = Field("AND", description="Logic for combining conditions")
    improvement_actions: List[str] = Field(..., description="Actions to take when triggered")
    action_parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for actions")
    dry_run_mode: bool = Field(False, description="Whether to run in test mode")
    approval_required: bool = Field(False, description="Whether approval is required")
    max_executions_per_day: int = Field(100, ge=1, description="Maximum executions per day")
    rule_description: Optional[str] = Field(None, description="Description of the rule")
    rule_priority: int = Field(50, ge=1, le=100, description="Rule priority")
    
    class Config:
        schema_extra = {
            "example": {
                "rule_name": "Low Quality Link Auto-Fix",
                "rule_type": "quality_threshold",
                "target_type": "link",
                "trigger_conditions": {
                    "confidence_score": {"operator": "<", "value": 0.5},
                    "negative_feedback_rate": {"operator": ">", "value": 0.3}
                },
                "improvement_actions": ["link_revalidation"],
                "rule_priority": 80
            }
        }


class UpdateImprovementStatusRequest(BaseModel):
    """Request schema for updating improvement status."""
    
    status: ImprovementStatus = Field(..., description="New status")
    quality_after: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score after improvement")
    effectiveness_score: Optional[float] = Field(None, ge=0.0, le=2.0, description="Effectiveness score")
    validation_results: Optional[Dict[str, Any]] = Field(None, description="Validation results")
    impact_metrics: Optional[Dict[str, Any]] = Field(None, description="Impact metrics")
    failure_reason: Optional[str] = Field(None, description="Reason for failure if applicable")


# Response Schemas

class QualityAssessmentResponse(BaseModel):
    """Response schema for quality assessments."""
    
    id: uuid.UUID
    tenant_id: uuid.UUID
    target_type: str
    target_id: uuid.UUID
    overall_quality_score: float
    accuracy_score: Optional[float]
    completeness_score: Optional[float]
    freshness_score: Optional[float]
    relevance_score: Optional[float]
    usability_score: Optional[float]
    assessment_method: str
    confidence_level: Optional[float]
    sample_size: Optional[int]
    assessment_date: datetime
    dimension_weights: Optional[Dict[str, float]]
    dimension_scores: Optional[Dict[str, float]]
    assessment_context: Optional[Dict[str, Any]]
    quality_issues: Optional[List[str]]
    improvement_suggestions: Optional[List[str]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class QualityImprovementResponse(BaseModel):
    """Response schema for quality improvements."""
    
    id: uuid.UUID
    tenant_id: uuid.UUID
    improvement_type: str
    target_type: str
    target_id: uuid.UUID
    trigger_reason: str
    trigger_threshold: Optional[float]
    trigger_metadata: Optional[Dict[str, Any]]
    improvement_action: str
    action_parameters: Optional[Dict[str, Any]]
    status: str
    quality_before: Optional[float]
    quality_after: Optional[float]
    improvement_delta: Optional[float]
    effectiveness_score: Optional[float]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    failed_at: Optional[datetime]
    failure_reason: Optional[str]
    validation_results: Optional[Dict[str, Any]]
    impact_metrics: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class QualityMonitoringResponse(BaseModel):
    """Response schema for quality monitoring."""
    
    id: uuid.UUID
    tenant_id: uuid.UUID
    monitor_name: str
    monitor_type: str
    target_type: str
    quality_threshold: Optional[float]
    trend_threshold: Optional[float]
    pattern_rules: Optional[Dict[str, Any]]
    alert_conditions: Optional[Dict[str, Any]]
    is_active: bool
    last_check: Optional[datetime]
    next_check: Optional[datetime]
    check_interval_minutes: int
    alert_enabled: bool
    alert_recipients: Optional[List[str]]
    alert_severity: str
    current_value: Optional[float]
    trend_direction: Optional[str]
    alert_count: int
    last_alert: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AutomationRuleResponse(BaseModel):
    """Response schema for automation rules."""
    
    id: uuid.UUID
    tenant_id: uuid.UUID
    rule_name: str
    rule_type: str
    target_type: str
    trigger_conditions: Dict[str, Any]
    condition_logic: str
    improvement_actions: List[str]
    action_parameters: Optional[Dict[str, Any]]
    is_active: bool
    execution_count: int
    last_execution: Optional[datetime]
    success_count: int
    failure_count: int
    dry_run_mode: bool
    approval_required: bool
    max_executions_per_day: int
    rule_description: Optional[str]
    rule_priority: int
    rule_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class QualityAlertResponse(BaseModel):
    """Response schema for quality alerts."""
    
    id: uuid.UUID
    tenant_id: uuid.UUID
    monitor_id: Optional[uuid.UUID]
    rule_id: Optional[uuid.UUID]
    alert_type: str
    alert_severity: str
    alert_title: str
    alert_message: str
    target_type: Optional[str]
    target_id: Optional[uuid.UUID]
    quality_value: Optional[float]
    threshold_value: Optional[float]
    alert_metadata: Optional[Dict[str, Any]]
    status: str
    acknowledged_by: Optional[uuid.UUID]
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    resolution_notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Paginated Response Schemas

class QualityAssessmentListResponse(BaseModel):
    """Paginated response for quality assessments."""
    
    items: List[QualityAssessmentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class QualityImprovementListResponse(BaseModel):
    """Paginated response for quality improvements."""
    
    items: List[QualityImprovementResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class QualityMonitoringListResponse(BaseModel):
    """Paginated response for quality monitoring."""
    
    items: List[QualityMonitoringResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class AutomationRuleListResponse(BaseModel):
    """Paginated response for automation rules."""
    
    items: List[AutomationRuleResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class QualityAlertListResponse(BaseModel):
    """Paginated response for quality alerts."""
    
    items: List[QualityAlertResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# Dashboard and Analytics Schemas

class QualityDashboardResponse(BaseModel):
    """Response schema for quality improvement dashboard."""
    
    overall_quality_score: float
    quality_trend: str  # improving, declining, stable
    total_assessments: int
    active_improvements: int
    completed_improvements: int
    active_alerts: int
    automation_rules_count: int
    
    # Quality breakdown by dimension
    quality_by_dimension: Dict[str, float]
    
    # Recent activity
    recent_assessments: List[QualityAssessmentResponse]
    recent_improvements: List[QualityImprovementResponse]
    recent_alerts: List[QualityAlertResponse]
    
    # Quality statistics
    quality_distribution: Dict[str, int]  # Distribution of quality scores
    improvement_effectiveness: float
    automation_success_rate: float
    
    last_updated: datetime


class QualityMetricsResponse(BaseModel):
    """Response schema for quality metrics."""
    
    overall_quality_score: float
    quality_trend_7d: float
    quality_trend_30d: float
    
    # Quality by target type
    content_quality_score: float
    link_quality_score: float
    system_quality_score: float
    
    # Improvement metrics
    improvements_in_progress: int
    improvements_completed_7d: int
    improvements_completed_30d: int
    average_improvement_time_hours: float
    improvement_success_rate: float
    
    # Alert metrics
    active_alerts: int
    alerts_created_7d: int
    alerts_resolved_7d: int
    average_resolution_time_hours: float
    
    # Automation metrics
    active_automation_rules: int
    automation_executions_7d: int
    automation_success_rate: float
