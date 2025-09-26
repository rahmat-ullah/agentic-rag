"""
Feedback Analytics and Insights System Migration

Revision ID: 008_feedback_analytics_system
Revises: 007_learning_algorithms_system
Create Date: 2024-12-19 14:30:00.000000

This migration creates the database schema for Sprint 6 Story 6-04:
Feedback Analytics and Insights System, including analytics metrics,
performance recommendations, dashboard configurations, and metric aggregations.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '008_feedback_analytics_system'
down_revision = '007_learning_algorithms_system'
branch_labels = None
depends_on = None


def upgrade():
    """Create analytics system tables."""
    
    # Create analytics_metrics table
    op.create_table(
        'analytics_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('metric_type', sa.Enum(
            'SEARCH_QUALITY', 'USER_SATISFACTION', 'CONTENT_QUALITY', 
            'SYSTEM_PERFORMANCE', 'USER_ENGAGEMENT', 'LEARNING_EFFECTIVENESS',
            name='analyticsmetrictype'
        ), nullable=False),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_category', sa.String(50), nullable=True),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('baseline_value', sa.Float, nullable=True),
        sa.Column('target_value', sa.Float, nullable=True),
        sa.Column('measurement_date', sa.Date, nullable=False),
        sa.Column('measurement_period_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('measurement_period_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('dimension_values', postgresql.JSONB, nullable=True),
        sa.Column('sample_size', sa.Integer, nullable=True),
        sa.Column('confidence_level', sa.Float, nullable=True),
        sa.Column('previous_value', sa.Float, nullable=True),
        sa.Column('change_percentage', sa.Float, nullable=True),
        sa.Column('trend_direction', sa.String(20), nullable=True),
        sa.Column('data_quality_score', sa.Float, nullable=True),
        sa.Column('statistical_significance', sa.Float, nullable=True),
        sa.Column('calculation_method', sa.String(100), nullable=True),
        sa.Column('data_sources', postgresql.JSONB, nullable=True),
        sa.Column('metric_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.CheckConstraint('metric_value IS NOT NULL', name='ck_analytics_metrics_value_not_null'),
        sa.CheckConstraint('confidence_level >= 0 AND confidence_level <= 1', name='ck_analytics_metrics_confidence_range'),
        sa.CheckConstraint('data_quality_score >= 0 AND data_quality_score <= 1', name='ck_analytics_metrics_quality_range'),
    )
    
    # Create performance_recommendations table
    op.create_table(
        'performance_recommendations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('recommendation_type', sa.Enum(
            'SEARCH_IMPROVEMENT', 'CONTENT_OPTIMIZATION', 'USER_EXPERIENCE',
            'SYSTEM_OPTIMIZATION', 'LEARNING_TUNING', 'QUALITY_ENHANCEMENT',
            name='recommendationtype'
        ), nullable=False),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('rationale', sa.Text, nullable=True),
        sa.Column('priority', sa.Enum(
            'CRITICAL', 'HIGH', 'MEDIUM', 'LOW',
            name='recommendationpriority'
        ), nullable=False),
        sa.Column('estimated_impact', sa.Float, nullable=False),
        sa.Column('implementation_effort', sa.Enum(
            'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH',
            name='implementationeffort'
        ), nullable=False),
        sa.Column('implementation_steps', postgresql.JSONB, nullable=True),
        sa.Column('required_resources', postgresql.JSONB, nullable=True),
        sa.Column('estimated_duration_hours', sa.Integer, nullable=True),
        sa.Column('status', sa.Enum(
            'PENDING', 'IN_PROGRESS', 'COMPLETED', 'REJECTED', 'DEFERRED',
            name='recommendationstatus'
        ), default='PENDING'),
        sa.Column('assigned_to', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('baseline_metrics', postgresql.JSONB, nullable=True),
        sa.Column('target_metrics', postgresql.JSONB, nullable=True),
        sa.Column('actual_metrics', postgresql.JSONB, nullable=True),
        sa.Column('effectiveness_score', sa.Float, nullable=True),
        sa.Column('related_metrics', postgresql.JSONB, nullable=True),
        sa.Column('related_recommendations', postgresql.JSONB, nullable=True),
        sa.Column('recommendation_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('implemented_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['assigned_to'], ['app_user.id']),
        sa.CheckConstraint('estimated_impact >= 0 AND estimated_impact <= 1', name='ck_perf_rec_impact_range'),
        sa.CheckConstraint('effectiveness_score >= 0 AND effectiveness_score <= 1', name='ck_perf_rec_effectiveness_range'),
    )
    
    # Create dashboard_configurations table
    op.create_table(
        'dashboard_configurations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('dashboard_type', sa.String(50), nullable=False),
        sa.Column('layout_config', postgresql.JSONB, nullable=False),
        sa.Column('components', postgresql.JSONB, nullable=False),
        sa.Column('filters', postgresql.JSONB, nullable=True),
        sa.Column('refresh_interval_minutes', sa.Integer, default=5),
        sa.Column('is_public', sa.Boolean, default=False),
        sa.Column('shared_with_users', postgresql.JSONB, nullable=True),
        sa.Column('shared_with_roles', postgresql.JSONB, nullable=True),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('is_default', sa.Boolean, default=False),
        sa.Column('dashboard_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('last_accessed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['user_id'], ['app_user.id']),
        sa.UniqueConstraint('tenant_id', 'user_id', 'name', name='uq_dashboard_configurations_tenant_user_name'),
    )
    
    # Create metric_aggregations table
    op.create_table(
        'metric_aggregations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('aggregation_name', sa.String(100), nullable=False),
        sa.Column('metric_type', sa.Enum(
            'SEARCH_QUALITY', 'USER_SATISFACTION', 'CONTENT_QUALITY', 
            'SYSTEM_PERFORMANCE', 'USER_ENGAGEMENT', 'LEARNING_EFFECTIVENESS',
            name='analyticsmetrictype'
        ), nullable=False),
        sa.Column('aggregation_level', sa.String(50), nullable=False),
        sa.Column('dimensions', postgresql.JSONB, nullable=True),
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('count', sa.Integer, nullable=False),
        sa.Column('sum_value', sa.Float, nullable=True),
        sa.Column('avg_value', sa.Float, nullable=True),
        sa.Column('min_value', sa.Float, nullable=True),
        sa.Column('max_value', sa.Float, nullable=True),
        sa.Column('median_value', sa.Float, nullable=True),
        sa.Column('std_dev', sa.Float, nullable=True),
        sa.Column('p25_value', sa.Float, nullable=True),
        sa.Column('p75_value', sa.Float, nullable=True),
        sa.Column('p90_value', sa.Float, nullable=True),
        sa.Column('p95_value', sa.Float, nullable=True),
        sa.Column('p99_value', sa.Float, nullable=True),
        sa.Column('data_completeness', sa.Float, nullable=True),
        sa.Column('outlier_count', sa.Integer, nullable=True),
        sa.Column('calculation_timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('source_record_count', sa.Integer, nullable=True),
        sa.Column('aggregation_metadata', postgresql.JSONB, nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.UniqueConstraint('tenant_id', 'aggregation_name', 'period_start', 'period_end', 
                           name='uq_metric_aggregations_tenant_name_period'),
        sa.CheckConstraint('period_end > period_start', name='ck_metric_aggregations_period_order'),
        sa.CheckConstraint('count >= 0', name='ck_metric_aggregations_count_positive'),
    )
    
    # Create indexes for analytics_metrics
    op.create_index('idx_analytics_metrics_tenant_type_date', 'analytics_metrics', 
                   ['tenant_id', 'metric_type', 'measurement_date'])
    op.create_index('idx_analytics_metrics_name_date', 'analytics_metrics', 
                   ['metric_name', 'measurement_date'])
    op.create_index('idx_analytics_metrics_category', 'analytics_metrics', ['metric_category'])
    
    # Create indexes for performance_recommendations
    op.create_index('idx_performance_recommendations_tenant_type', 'performance_recommendations', 
                   ['tenant_id', 'recommendation_type'])
    op.create_index('idx_performance_recommendations_status', 'performance_recommendations', ['status'])
    op.create_index('idx_performance_recommendations_priority', 'performance_recommendations', ['priority'])
    op.create_index('idx_performance_recommendations_assigned', 'performance_recommendations', ['assigned_to'])
    
    # Create indexes for dashboard_configurations
    op.create_index('idx_dashboard_configurations_tenant_user', 'dashboard_configurations', 
                   ['tenant_id', 'user_id'])
    op.create_index('idx_dashboard_configurations_type', 'dashboard_configurations', ['dashboard_type'])
    op.create_index('idx_dashboard_configurations_active', 'dashboard_configurations', ['is_active'])
    
    # Create indexes for metric_aggregations
    op.create_index('idx_metric_aggregations_tenant_type_level', 'metric_aggregations', 
                   ['tenant_id', 'metric_type', 'aggregation_level'])
    op.create_index('idx_metric_aggregations_period', 'metric_aggregations', 
                   ['period_start', 'period_end'])
    op.create_index('idx_metric_aggregations_name', 'metric_aggregations', ['aggregation_name'])


def downgrade():
    """Drop analytics system tables."""
    
    # Drop indexes first
    op.drop_index('idx_metric_aggregations_name', 'metric_aggregations')
    op.drop_index('idx_metric_aggregations_period', 'metric_aggregations')
    op.drop_index('idx_metric_aggregations_tenant_type_level', 'metric_aggregations')
    
    op.drop_index('idx_dashboard_configurations_active', 'dashboard_configurations')
    op.drop_index('idx_dashboard_configurations_type', 'dashboard_configurations')
    op.drop_index('idx_dashboard_configurations_tenant_user', 'dashboard_configurations')
    
    op.drop_index('idx_performance_recommendations_assigned', 'performance_recommendations')
    op.drop_index('idx_performance_recommendations_priority', 'performance_recommendations')
    op.drop_index('idx_performance_recommendations_status', 'performance_recommendations')
    op.drop_index('idx_performance_recommendations_tenant_type', 'performance_recommendations')
    
    op.drop_index('idx_analytics_metrics_category', 'analytics_metrics')
    op.drop_index('idx_analytics_metrics_name_date', 'analytics_metrics')
    op.drop_index('idx_analytics_metrics_tenant_type_date', 'analytics_metrics')
    
    # Drop tables
    op.drop_table('metric_aggregations')
    op.drop_table('dashboard_configurations')
    op.drop_table('performance_recommendations')
    op.drop_table('analytics_metrics')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS implementationeffort')
    op.execute('DROP TYPE IF EXISTS recommendationstatus')
    op.execute('DROP TYPE IF EXISTS recommendationpriority')
    op.execute('DROP TYPE IF EXISTS recommendationtype')
    op.execute('DROP TYPE IF EXISTS analyticsmetrictype')
