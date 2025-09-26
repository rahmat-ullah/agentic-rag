"""
Monitoring and Observability System Migration

This migration creates the database schema for the monitoring and observability system,
including tables for metrics, traces, alerts, health checks, and service discovery.

Revision ID: 010_monitoring_observability_system
Revises: 009_automated_quality_improvement_system
Create Date: 2024-12-19 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '010_monitoring_observability_system'
down_revision = '009_automated_quality_improvement_system'
branch_labels = None
depends_on = None


def upgrade():
    """Create monitoring and observability system tables."""
    
    # Create metrics table
    op.create_table(
        'metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('labels', postgresql.JSONB(), nullable=False, default={}),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('source_service', sa.String(100), nullable=True, index=True),
        sa.Column('source_instance', sa.String(100), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    # Create metrics indexes
    op.create_index('idx_metrics_tenant_name_timestamp', 'metrics', ['tenant_id', 'name', 'timestamp'])
    op.create_index('idx_metrics_service_timestamp', 'metrics', ['source_service', 'timestamp'])
    op.create_index('idx_metrics_type_timestamp', 'metrics', ['metric_type', 'timestamp'])
    
    # Create traces table
    op.create_table(
        'traces',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('trace_id', sa.String(32), nullable=False, index=True),
        sa.Column('span_id', sa.String(16), nullable=False, index=True),
        sa.Column('parent_span_id', sa.String(16), nullable=True, index=True),
        sa.Column('operation_name', sa.String(255), nullable=False, index=True),
        sa.Column('service_name', sa.String(100), nullable=False, index=True),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Float(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='ok'),
        sa.Column('tags', postgresql.JSONB(), nullable=False, default={}),
        sa.Column('logs', postgresql.JSONB(), nullable=True, default=[]),
        sa.Column('error', sa.Boolean(), nullable=False, default=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_stack', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    # Create traces indexes
    op.create_index('idx_traces_tenant_trace_id', 'traces', ['tenant_id', 'trace_id'])
    op.create_index('idx_traces_service_start_time', 'traces', ['service_name', 'start_time'])
    op.create_index('idx_traces_operation_start_time', 'traces', ['operation_name', 'start_time'])
    op.create_index('idx_traces_error_start_time', 'traces', ['error', 'start_time'])
    
    # Create alerts table
    op.create_table(
        'alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('rule_name', sa.String(255), nullable=False, index=True),
        sa.Column('alert_name', sa.String(255), nullable=False),
        sa.Column('fingerprint', sa.String(64), nullable=False, index=True),
        sa.Column('severity', sa.String(50), nullable=False, index=True),
        sa.Column('status', sa.String(50), nullable=False, default='active'),
        sa.Column('labels', postgresql.JSONB(), nullable=False, default={}),
        sa.Column('annotations', postgresql.JSONB(), nullable=False, default={}),
        sa.Column('starts_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('ends_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('runbook_url', sa.String(500), nullable=True),
        sa.Column('dashboard_url', sa.String(500), nullable=True),
        sa.Column('escalation_level', sa.Integer(), nullable=False, default=0),
        sa.Column('notification_sent', sa.Boolean(), nullable=False, default=False),
        sa.Column('notification_channels', postgresql.JSONB(), nullable=True, default=[]),
        sa.Column('source_service', sa.String(100), nullable=True, index=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    # Create alerts indexes
    op.create_index('idx_alerts_tenant_status_severity', 'alerts', ['tenant_id', 'status', 'severity'])
    op.create_index('idx_alerts_rule_starts_at', 'alerts', ['rule_name', 'starts_at'])
    op.create_index('idx_alerts_fingerprint_starts_at', 'alerts', ['fingerprint', 'starts_at'])
    
    # Create health_checks table
    op.create_table(
        'health_checks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('check_name', sa.String(255), nullable=False, index=True),
        sa.Column('service_name', sa.String(100), nullable=False, index=True),
        sa.Column('instance_id', sa.String(100), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, index=True),
        sa.Column('response_time_ms', sa.Float(), nullable=True),
        sa.Column('check_timestamp', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('details', postgresql.JSONB(), nullable=True, default={}),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    # Create health_checks indexes
    op.create_index('idx_health_checks_tenant_service_timestamp', 'health_checks', ['tenant_id', 'service_name', 'check_timestamp'])
    op.create_index('idx_health_checks_status_timestamp', 'health_checks', ['status', 'check_timestamp'])
    op.create_index('idx_health_checks_check_name_timestamp', 'health_checks', ['check_name', 'check_timestamp'])
    
    # Create monitoring_configurations table
    op.create_table(
        'monitoring_configurations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('config_name', sa.String(255), nullable=False, index=True),
        sa.Column('config_type', sa.String(100), nullable=False, index=True),
        sa.Column('config_data', postgresql.JSONB(), nullable=False, default={}),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('version', sa.Integer(), nullable=False, default=1),
        sa.Column('previous_version_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('created_by', sa.String(255), nullable=True),
    )
    
    # Create monitoring_configurations indexes
    op.create_index('idx_monitoring_configs_tenant_type_active', 'monitoring_configurations', ['tenant_id', 'config_type', 'is_active'])
    op.create_index('idx_monitoring_configs_name_version', 'monitoring_configurations', ['config_name', 'version'])
    
    # Create service_discovery table
    op.create_table(
        'service_discovery',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('service_name', sa.String(100), nullable=False, index=True),
        sa.Column('instance_id', sa.String(100), nullable=False, index=True),
        sa.Column('host', sa.String(255), nullable=False),
        sa.Column('port', sa.Integer(), nullable=False),
        sa.Column('protocol', sa.String(10), nullable=False, default='http'),
        sa.Column('metrics_path', sa.String(255), nullable=False, default='/metrics'),
        sa.Column('health_path', sa.String(255), nullable=False, default='/health'),
        sa.Column('labels', postgresql.JSONB(), nullable=False, default={}),
        sa.Column('tags', postgresql.JSONB(), nullable=False, default={}),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    # Create service_discovery indexes
    op.create_index('idx_service_discovery_tenant_service', 'service_discovery', ['tenant_id', 'service_name'])
    op.create_index('idx_service_discovery_active_last_seen', 'service_discovery', ['is_active', 'last_seen'])
    op.create_index('idx_service_discovery_instance', 'service_discovery', ['instance_id'])


def downgrade():
    """Drop monitoring and observability system tables."""
    
    # Drop tables in reverse order
    op.drop_table('service_discovery')
    op.drop_table('monitoring_configurations')
    op.drop_table('health_checks')
    op.drop_table('alerts')
    op.drop_table('traces')
    op.drop_table('metrics')
