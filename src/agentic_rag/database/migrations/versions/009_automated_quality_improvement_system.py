"""
Database Migration for Sprint 6 Story 6-05: Automated Quality Improvement System

This migration creates the database schema for automated quality improvement,
including quality assessments, improvement actions, monitoring, and automation rules.

Revision ID: 009_automated_quality_improvement_system
Revises: 008_feedback_analytics_system
Create Date: 2024-12-19 14:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '009_automated_quality_improvement_system'
down_revision = '008_feedback_analytics_system'
branch_labels = None
depends_on = None


def upgrade():
    """Create quality improvement system tables."""
    
    # Create quality_assessments table
    op.create_table(
        'quality_assessments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=False),
        sa.Column('target_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('overall_quality_score', sa.Float, nullable=False),
        sa.Column('accuracy_score', sa.Float),
        sa.Column('completeness_score', sa.Float),
        sa.Column('freshness_score', sa.Float),
        sa.Column('relevance_score', sa.Float),
        sa.Column('usability_score', sa.Float),
        sa.Column('assessment_method', sa.String(100), nullable=False),
        sa.Column('confidence_level', sa.Float),
        sa.Column('sample_size', sa.Integer),
        sa.Column('assessment_date', sa.DateTime, nullable=False),
        sa.Column('dimension_weights', sa.JSON),
        sa.Column('dimension_scores', sa.JSON),
        sa.Column('assessment_context', sa.JSON),
        sa.Column('quality_issues', sa.JSON),
        sa.Column('improvement_suggestions', sa.JSON),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        
        # Constraints
        sa.CheckConstraint('overall_quality_score >= 0.0 AND overall_quality_score <= 1.0', name='ck_quality_assessments_overall_score'),
        sa.CheckConstraint('accuracy_score IS NULL OR (accuracy_score >= 0.0 AND accuracy_score <= 1.0)', name='ck_quality_assessments_accuracy_score'),
        sa.CheckConstraint('completeness_score IS NULL OR (completeness_score >= 0.0 AND completeness_score <= 1.0)', name='ck_quality_assessments_completeness_score'),
        sa.CheckConstraint('freshness_score IS NULL OR (freshness_score >= 0.0 AND freshness_score <= 1.0)', name='ck_quality_assessments_freshness_score'),
        sa.CheckConstraint('relevance_score IS NULL OR (relevance_score >= 0.0 AND relevance_score <= 1.0)', name='ck_quality_assessments_relevance_score'),
        sa.CheckConstraint('usability_score IS NULL OR (usability_score >= 0.0 AND usability_score <= 1.0)', name='ck_quality_assessments_usability_score'),
        sa.CheckConstraint("target_type IN ('content', 'link', 'system', 'query')", name='ck_quality_assessments_target_type'),
    )
    
    # Create quality_improvements table
    op.create_table(
        'quality_improvements',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('improvement_type', sa.String(50), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=False),
        sa.Column('target_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('trigger_reason', sa.String(200), nullable=False),
        sa.Column('trigger_threshold', sa.Float),
        sa.Column('trigger_metadata', sa.JSON),
        sa.Column('improvement_action', sa.String(100), nullable=False),
        sa.Column('action_parameters', sa.JSON),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('quality_before', sa.Float),
        sa.Column('quality_after', sa.Float),
        sa.Column('improvement_delta', sa.Float),
        sa.Column('effectiveness_score', sa.Float),
        sa.Column('started_at', sa.DateTime),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('failed_at', sa.DateTime),
        sa.Column('failure_reason', sa.Text),
        sa.Column('validation_results', sa.JSON),
        sa.Column('impact_metrics', sa.JSON),
        sa.Column('rollback_data', sa.JSON),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        
        # Constraints
        sa.CheckConstraint("improvement_type IN ('low_quality_link', 'frequent_corrections', 'poor_content_quality', 'processing_errors', 'low_user_satisfaction', 'high_bounce_rate')", name='ck_quality_improvements_type'),
        sa.CheckConstraint("improvement_action IN ('link_revalidation', 'content_reprocessing', 'embedding_update', 'metadata_refresh', 'algorithm_tuning', 'content_removal', 'quality_flagging')", name='ck_quality_improvements_action'),
        sa.CheckConstraint("status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')", name='ck_quality_improvements_status'),
    )
    
    # Create quality_monitoring table
    op.create_table(
        'quality_monitoring',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('monitor_name', sa.String(100), nullable=False),
        sa.Column('monitor_type', sa.String(50), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=False),
        sa.Column('quality_threshold', sa.Float),
        sa.Column('trend_threshold', sa.Float),
        sa.Column('pattern_rules', sa.JSON),
        sa.Column('alert_conditions', sa.JSON),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('last_check', sa.DateTime),
        sa.Column('next_check', sa.DateTime),
        sa.Column('check_interval_minutes', sa.Integer, default=60),
        sa.Column('alert_enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('alert_recipients', sa.JSON),
        sa.Column('alert_severity', sa.String(20), default='medium'),
        sa.Column('current_value', sa.Float),
        sa.Column('trend_direction', sa.String(20)),
        sa.Column('alert_count', sa.Integer, default=0),
        sa.Column('last_alert', sa.DateTime),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        
        # Constraints
        sa.CheckConstraint("monitor_type IN ('threshold', 'trend', 'pattern')", name='ck_quality_monitoring_type'),
        sa.CheckConstraint("alert_severity IN ('low', 'medium', 'high', 'critical')", name='ck_quality_monitoring_severity'),
        sa.CheckConstraint("trend_direction IS NULL OR trend_direction IN ('increasing', 'decreasing', 'stable')", name='ck_quality_monitoring_trend'),
    )
    
    # Create automation_rules table
    op.create_table(
        'automation_rules',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rule_name', sa.String(100), nullable=False),
        sa.Column('rule_type', sa.String(50), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=False),
        sa.Column('trigger_conditions', sa.JSON, nullable=False),
        sa.Column('condition_logic', sa.String(20), default='AND'),
        sa.Column('improvement_actions', sa.JSON, nullable=False),
        sa.Column('action_parameters', sa.JSON),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('execution_count', sa.Integer, default=0),
        sa.Column('last_execution', sa.DateTime),
        sa.Column('success_count', sa.Integer, default=0),
        sa.Column('failure_count', sa.Integer, default=0),
        sa.Column('dry_run_mode', sa.Boolean, default=False),
        sa.Column('approval_required', sa.Boolean, default=False),
        sa.Column('max_executions_per_day', sa.Integer, default=100),
        sa.Column('rule_description', sa.Text),
        sa.Column('rule_priority', sa.Integer, default=50),
        sa.Column('rule_metadata', sa.JSON),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        
        # Constraints
        sa.CheckConstraint("rule_type IN ('quality_threshold', 'pattern_detection', 'trend_analysis')", name='ck_automation_rules_type'),
        sa.CheckConstraint("condition_logic IN ('AND', 'OR')", name='ck_automation_rules_logic'),
        sa.CheckConstraint('rule_priority >= 1 AND rule_priority <= 100', name='ck_automation_rules_priority'),
    )
    
    # Create quality_alerts table
    op.create_table(
        'quality_alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('monitor_id', postgresql.UUID(as_uuid=True)),
        sa.Column('rule_id', postgresql.UUID(as_uuid=True)),
        sa.Column('alert_type', sa.String(50), nullable=False),
        sa.Column('alert_severity', sa.String(20), nullable=False),
        sa.Column('alert_title', sa.String(200), nullable=False),
        sa.Column('alert_message', sa.Text, nullable=False),
        sa.Column('target_type', sa.String(50)),
        sa.Column('target_id', postgresql.UUID(as_uuid=True)),
        sa.Column('quality_value', sa.Float),
        sa.Column('threshold_value', sa.Float),
        sa.Column('alert_metadata', sa.JSON),
        sa.Column('status', sa.String(20), nullable=False, default='active'),
        sa.Column('acknowledged_by', postgresql.UUID(as_uuid=True)),
        sa.Column('acknowledged_at', sa.DateTime),
        sa.Column('resolved_at', sa.DateTime),
        sa.Column('resolution_notes', sa.Text),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['monitor_id'], ['quality_monitoring.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['rule_id'], ['automation_rules.id'], ondelete='SET NULL'),
        
        # Constraints
        sa.CheckConstraint("alert_type IN ('threshold_breach', 'trend_alert', 'pattern_detected')", name='ck_quality_alerts_type'),
        sa.CheckConstraint("alert_severity IN ('low', 'medium', 'high', 'critical')", name='ck_quality_alerts_severity'),
        sa.CheckConstraint("status IN ('active', 'acknowledged', 'resolved', 'dismissed')", name='ck_quality_alerts_status'),
    )
    
    # Create indexes for quality_assessments
    op.create_index('idx_quality_assessments_tenant_target', 'quality_assessments', ['tenant_id', 'target_type', 'target_id'])
    op.create_index('idx_quality_assessments_score', 'quality_assessments', ['tenant_id', 'overall_quality_score'])
    op.create_index('idx_quality_assessments_date', 'quality_assessments', ['tenant_id', 'assessment_date'])
    
    # Create indexes for quality_improvements
    op.create_index('idx_quality_improvements_tenant_type', 'quality_improvements', ['tenant_id', 'improvement_type'])
    op.create_index('idx_quality_improvements_status', 'quality_improvements', ['tenant_id', 'status'])
    op.create_index('idx_quality_improvements_target', 'quality_improvements', ['tenant_id', 'target_type', 'target_id'])
    
    # Create indexes for quality_monitoring
    op.create_index('idx_quality_monitoring_tenant_active', 'quality_monitoring', ['tenant_id', 'is_active'])
    op.create_index('idx_quality_monitoring_next_check', 'quality_monitoring', ['next_check'])
    
    # Create indexes for automation_rules
    op.create_index('idx_automation_rules_tenant_active', 'automation_rules', ['tenant_id', 'is_active'])
    op.create_index('idx_automation_rules_type', 'automation_rules', ['tenant_id', 'rule_type'])
    
    # Create indexes for quality_alerts
    op.create_index('idx_quality_alerts_tenant_status', 'quality_alerts', ['tenant_id', 'status'])
    op.create_index('idx_quality_alerts_severity', 'quality_alerts', ['tenant_id', 'alert_severity'])
    op.create_index('idx_quality_alerts_created', 'quality_alerts', ['tenant_id', 'created_at'])


def downgrade():
    """Drop quality improvement system tables."""
    
    # Drop indexes first
    op.drop_index('idx_quality_alerts_created', table_name='quality_alerts')
    op.drop_index('idx_quality_alerts_severity', table_name='quality_alerts')
    op.drop_index('idx_quality_alerts_tenant_status', table_name='quality_alerts')
    
    op.drop_index('idx_automation_rules_type', table_name='automation_rules')
    op.drop_index('idx_automation_rules_tenant_active', table_name='automation_rules')
    
    op.drop_index('idx_quality_monitoring_next_check', table_name='quality_monitoring')
    op.drop_index('idx_quality_monitoring_tenant_active', table_name='quality_monitoring')
    
    op.drop_index('idx_quality_improvements_target', table_name='quality_improvements')
    op.drop_index('idx_quality_improvements_status', table_name='quality_improvements')
    op.drop_index('idx_quality_improvements_tenant_type', table_name='quality_improvements')
    
    op.drop_index('idx_quality_assessments_date', table_name='quality_assessments')
    op.drop_index('idx_quality_assessments_score', table_name='quality_assessments')
    op.drop_index('idx_quality_assessments_tenant_target', table_name='quality_assessments')
    
    # Drop tables in reverse order (respecting foreign key dependencies)
    op.drop_table('quality_alerts')
    op.drop_table('automation_rules')
    op.drop_table('quality_monitoring')
    op.drop_table('quality_improvements')
    op.drop_table('quality_assessments')
