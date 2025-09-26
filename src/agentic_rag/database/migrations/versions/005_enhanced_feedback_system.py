"""Enhanced feedback system for Sprint 6

Revision ID: 005_enhanced_feedback_system
Revises: 004_enhance_document_link
Create Date: 2024-12-19 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005_enhanced_feedback_system'
down_revision = '004_enhance_document_link'
branch_labels = None
depends_on = None


def upgrade():
    """Create enhanced feedback system tables."""
    
    # Create user_feedback table
    op.create_table(
        'user_feedback',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('user_id', postgresql.UUID(), nullable=False),
        sa.Column('feedback_type', sa.String(50), nullable=False),
        sa.Column('feedback_category', sa.String(50), nullable=True),
        sa.Column('target_id', postgresql.UUID(), nullable=True),
        sa.Column('target_type', sa.String(50), nullable=True),
        sa.Column('rating', sa.Integer(), nullable=True),
        sa.Column('feedback_text', sa.Text(), nullable=True),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('context_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('priority', sa.String(20), nullable=False, server_default='medium'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('processing_notes', sa.Text(), nullable=True),
        sa.Column('assigned_to', postgresql.UUID(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['user_id'], ['app_user.id']),
        sa.ForeignKeyConstraint(['assigned_to'], ['app_user.id']),
        sa.CheckConstraint('rating IS NULL OR rating BETWEEN -1 AND 5', name='check_rating_range'),
    )
    
    # Create feedback_aggregation table
    op.create_table(
        'feedback_aggregation',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('target_id', postgresql.UUID(), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=False),
        sa.Column('feedback_type', sa.String(50), nullable=False),
        sa.Column('total_feedback_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('positive_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('negative_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('average_rating', sa.Float(), nullable=True),
        sa.Column('category_counts', postgresql.JSONB(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('first_feedback_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_feedback_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.CheckConstraint('quality_score IS NULL OR quality_score BETWEEN 0.0 AND 1.0', name='check_quality_score_range'),
        sa.CheckConstraint('confidence_score IS NULL OR confidence_score BETWEEN 0.0 AND 1.0', name='check_confidence_score_range'),
    )
    
    # Create feedback_impact table
    op.create_table(
        'feedback_impact',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('feedback_id', postgresql.UUID(), nullable=False),
        sa.Column('improvement_type', sa.String(100), nullable=False),
        sa.Column('improvement_description', sa.Text(), nullable=True),
        sa.Column('before_metric', sa.Float(), nullable=True),
        sa.Column('after_metric', sa.Float(), nullable=True),
        sa.Column('improvement_percentage', sa.Float(), nullable=True),
        sa.Column('implemented_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('implementation_notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['feedback_id'], ['user_feedback.id']),
    )
    
    # Create feedback_session table
    op.create_table(
        'feedback_session',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('user_id', postgresql.UUID(), nullable=False),
        sa.Column('session_id', sa.String(255), nullable=False),
        sa.Column('total_interactions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('feedback_submissions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('session_duration_seconds', sa.Integer(), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['user_id'], ['app_user.id']),
    )
    
    # Create indexes for user_feedback table
    op.create_index('idx_user_feedback_tenant_type', 'user_feedback', ['tenant_id', 'feedback_type'])
    op.create_index('idx_user_feedback_status', 'user_feedback', ['status'])
    op.create_index('idx_user_feedback_created_at', 'user_feedback', ['created_at'])
    op.create_index('idx_user_feedback_target', 'user_feedback', ['target_id', 'target_type'])
    op.create_index('idx_user_feedback_user_session', 'user_feedback', ['user_id', 'session_id'])
    
    # Create indexes for feedback_aggregation table
    op.create_index('idx_feedback_agg_tenant_target', 'feedback_aggregation', ['tenant_id', 'target_id', 'target_type'])
    op.create_index('idx_feedback_agg_type', 'feedback_aggregation', ['feedback_type'])
    op.create_index('idx_feedback_agg_quality', 'feedback_aggregation', ['quality_score'])
    
    # Create indexes for feedback_impact table
    op.create_index('idx_feedback_impact_tenant_feedback', 'feedback_impact', ['tenant_id', 'feedback_id'])
    op.create_index('idx_feedback_impact_type', 'feedback_impact', ['improvement_type'])
    op.create_index('idx_feedback_impact_implemented', 'feedback_impact', ['implemented_at'])
    
    # Create indexes for feedback_session table
    op.create_index('idx_feedback_session_tenant_user', 'feedback_session', ['tenant_id', 'user_id'])
    op.create_index('idx_feedback_session_session_id', 'feedback_session', ['session_id'])
    op.create_index('idx_feedback_session_started', 'feedback_session', ['started_at'])
    
    # Add check constraints for enum values
    op.create_check_constraint(
        'ck_feedback_type',
        'user_feedback',
        "feedback_type IN ('search_result', 'link_quality', 'answer_quality', 'general', 'system_usability')"
    )
    
    op.create_check_constraint(
        'ck_feedback_category',
        'user_feedback',
        """feedback_category IS NULL OR feedback_category IN (
            'not_relevant', 'missing_information', 'outdated_content', 'wrong_document_type',
            'incorrect_link', 'low_confidence', 'missing_link', 'duplicate_link',
            'inaccurate_information', 'incomplete_answer', 'poor_formatting', 'missing_citations',
            'feature_request', 'bug_report', 'performance_issue', 'usability_issue'
        )"""
    )
    
    op.create_check_constraint(
        'ck_feedback_status',
        'user_feedback',
        "status IN ('pending', 'processing', 'reviewed', 'implemented', 'rejected', 'duplicate')"
    )
    
    op.create_check_constraint(
        'ck_feedback_priority',
        'user_feedback',
        "priority IN ('low', 'medium', 'high', 'critical')"
    )


def downgrade():
    """Drop enhanced feedback system tables."""
    
    # Drop indexes first
    op.drop_index('idx_feedback_session_started', 'feedback_session')
    op.drop_index('idx_feedback_session_session_id', 'feedback_session')
    op.drop_index('idx_feedback_session_tenant_user', 'feedback_session')
    
    op.drop_index('idx_feedback_impact_implemented', 'feedback_impact')
    op.drop_index('idx_feedback_impact_type', 'feedback_impact')
    op.drop_index('idx_feedback_impact_tenant_feedback', 'feedback_impact')
    
    op.drop_index('idx_feedback_agg_quality', 'feedback_aggregation')
    op.drop_index('idx_feedback_agg_type', 'feedback_aggregation')
    op.drop_index('idx_feedback_agg_tenant_target', 'feedback_aggregation')
    
    op.drop_index('idx_user_feedback_user_session', 'user_feedback')
    op.drop_index('idx_user_feedback_target', 'user_feedback')
    op.drop_index('idx_user_feedback_created_at', 'user_feedback')
    op.drop_index('idx_user_feedback_status', 'user_feedback')
    op.drop_index('idx_user_feedback_tenant_type', 'user_feedback')
    
    # Drop tables
    op.drop_table('feedback_session')
    op.drop_table('feedback_impact')
    op.drop_table('feedback_aggregation')
    op.drop_table('user_feedback')
