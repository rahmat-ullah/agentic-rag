"""Content correction and editing system for Sprint 6 Story 6-02

Revision ID: 006_content_correction_system
Revises: 005_enhanced_feedback_system
Create Date: 2024-12-19 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006_content_correction_system'
down_revision = '005_enhanced_feedback_system'
branch_labels = None
depends_on = None


def upgrade():
    """Create content correction and editing system tables."""
    
    # Create content_corrections table
    op.create_table(
        'content_corrections',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('chunk_id', postgresql.UUID(), nullable=False),
        sa.Column('user_id', postgresql.UUID(), nullable=False),
        sa.Column('original_content', sa.Text(), nullable=False),
        sa.Column('corrected_content', sa.Text(), nullable=False),
        sa.Column('correction_reason', sa.Text(), nullable=True),
        sa.Column('correction_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('priority', sa.String(20), nullable=False, server_default='medium'),
        sa.Column('reviewer_id', postgresql.UUID(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('review_decision', sa.String(20), nullable=True),
        sa.Column('review_notes', sa.Text(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('impact_score', sa.Float(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('correction_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('source_references', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('implemented_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['chunk_id'], ['chunk_meta.id']),
        sa.ForeignKeyConstraint(['user_id'], ['app_user.id']),
        sa.ForeignKeyConstraint(['reviewer_id'], ['app_user.id']),
        sa.CheckConstraint('confidence_score IS NULL OR confidence_score BETWEEN 0.0 AND 1.0', name='check_confidence_score_range'),
        sa.CheckConstraint('impact_score IS NULL OR impact_score BETWEEN 0.0 AND 1.0', name='check_impact_score_range'),
        sa.CheckConstraint('quality_score IS NULL OR quality_score BETWEEN 0.0 AND 1.0', name='check_quality_score_range'),
    )
    
    # Create content_versions table
    op.create_table(
        'content_versions',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('chunk_id', postgresql.UUID(), nullable=False),
        sa.Column('correction_id', postgresql.UUID(), nullable=True),
        sa.Column('version_number', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('change_summary', sa.Text(), nullable=True),
        sa.Column('change_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('diff_data', postgresql.JSONB(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_published', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_by', postgresql.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('readability_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['chunk_id'], ['chunk_meta.id']),
        sa.ForeignKeyConstraint(['correction_id'], ['content_corrections.id']),
        sa.ForeignKeyConstraint(['created_by'], ['app_user.id']),
        sa.CheckConstraint('quality_score IS NULL OR quality_score BETWEEN 0.0 AND 1.0', name='check_version_quality_score_range'),
        sa.CheckConstraint('readability_score IS NULL OR readability_score BETWEEN 0.0 AND 1.0', name='check_readability_score_range'),
    )
    
    # Create correction_reviews table
    op.create_table(
        'correction_reviews',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('correction_id', postgresql.UUID(), nullable=False),
        sa.Column('reviewer_id', postgresql.UUID(), nullable=False),
        sa.Column('decision', sa.String(20), nullable=False),
        sa.Column('review_notes', sa.Text(), nullable=True),
        sa.Column('quality_assessment', postgresql.JSONB(), nullable=True),
        sa.Column('accuracy_score', sa.Float(), nullable=True),
        sa.Column('clarity_score', sa.Float(), nullable=True),
        sa.Column('completeness_score', sa.Float(), nullable=True),
        sa.Column('overall_score', sa.Float(), nullable=True),
        sa.Column('review_duration_minutes', sa.Integer(), nullable=True),
        sa.Column('review_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['correction_id'], ['content_corrections.id']),
        sa.ForeignKeyConstraint(['reviewer_id'], ['app_user.id']),
        sa.CheckConstraint('accuracy_score IS NULL OR accuracy_score BETWEEN 0.0 AND 1.0', name='check_accuracy_score_range'),
        sa.CheckConstraint('clarity_score IS NULL OR clarity_score BETWEEN 0.0 AND 1.0', name='check_clarity_score_range'),
        sa.CheckConstraint('completeness_score IS NULL OR completeness_score BETWEEN 0.0 AND 1.0', name='check_completeness_score_range'),
        sa.CheckConstraint('overall_score IS NULL OR overall_score BETWEEN 0.0 AND 1.0', name='check_overall_score_range'),
    )
    
    # Create correction_workflows table
    op.create_table(
        'correction_workflows',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('correction_id', postgresql.UUID(), nullable=False),
        sa.Column('current_step', sa.String(50), nullable=False),
        sa.Column('workflow_data', postgresql.JSONB(), nullable=True),
        sa.Column('assigned_to', postgresql.UUID(), nullable=True),
        sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('due_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('steps_completed', postgresql.JSONB(), nullable=True),
        sa.Column('next_steps', postgresql.JSONB(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['correction_id'], ['content_corrections.id']),
        sa.ForeignKeyConstraint(['assigned_to'], ['app_user.id']),
    )
    
    # Create correction_impacts table
    op.create_table(
        'correction_impacts',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('correction_id', postgresql.UUID(), nullable=False),
        sa.Column('search_improvement', sa.Float(), nullable=True),
        sa.Column('user_satisfaction', sa.Float(), nullable=True),
        sa.Column('accuracy_improvement', sa.Float(), nullable=True),
        sa.Column('chunk_access_before', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('chunk_access_after', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('feedback_improvement', sa.Float(), nullable=True),
        sa.Column('re_embedding_completed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('re_embedding_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('measurement_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('measurement_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['correction_id'], ['content_corrections.id']),
    )
    
    # Create indexes for content_corrections table
    op.create_index('idx_content_corrections_tenant_chunk', 'content_corrections', ['tenant_id', 'chunk_id'])
    op.create_index('idx_content_corrections_status', 'content_corrections', ['status'])
    op.create_index('idx_content_corrections_type', 'content_corrections', ['correction_type'])
    op.create_index('idx_content_corrections_created_at', 'content_corrections', ['created_at'])
    op.create_index('idx_content_corrections_reviewer', 'content_corrections', ['reviewer_id'])
    
    # Create indexes for content_versions table
    op.create_index('idx_content_versions_chunk_version', 'content_versions', ['chunk_id', 'version_number'])
    op.create_index('idx_content_versions_active', 'content_versions', ['chunk_id', 'is_active'])
    op.create_index('idx_content_versions_created_at', 'content_versions', ['created_at'])
    op.create_index('idx_content_versions_hash', 'content_versions', ['content_hash'])
    
    # Create indexes for correction_reviews table
    op.create_index('idx_correction_reviews_correction', 'correction_reviews', ['correction_id'])
    op.create_index('idx_correction_reviews_reviewer', 'correction_reviews', ['reviewer_id'])
    op.create_index('idx_correction_reviews_decision', 'correction_reviews', ['decision'])
    op.create_index('idx_correction_reviews_completed', 'correction_reviews', ['completed_at'])
    
    # Create indexes for correction_workflows table
    op.create_index('idx_correction_workflows_correction', 'correction_workflows', ['correction_id'])
    op.create_index('idx_correction_workflows_step', 'correction_workflows', ['current_step'])
    op.create_index('idx_correction_workflows_assigned', 'correction_workflows', ['assigned_to'])
    op.create_index('idx_correction_workflows_due', 'correction_workflows', ['due_date'])
    
    # Create indexes for correction_impacts table
    op.create_index('idx_correction_impacts_correction', 'correction_impacts', ['correction_id'])
    op.create_index('idx_correction_impacts_re_embedding', 'correction_impacts', ['re_embedding_completed'])
    op.create_index('idx_correction_impacts_created', 'correction_impacts', ['created_at'])
    
    # Add check constraints for enum values
    op.create_check_constraint(
        'ck_correction_type',
        'content_corrections',
        "correction_type IN ('factual', 'formatting', 'clarity', 'completeness', 'grammar', 'terminology')"
    )
    
    op.create_check_constraint(
        'ck_correction_status',
        'content_corrections',
        "status IN ('pending', 'under_review', 'approved', 'rejected', 'implemented', 'reverted')"
    )
    
    op.create_check_constraint(
        'ck_correction_priority',
        'content_corrections',
        "priority IN ('low', 'medium', 'high', 'critical')"
    )
    
    op.create_check_constraint(
        'ck_review_decision',
        'content_corrections',
        "review_decision IS NULL OR review_decision IN ('approve', 'reject', 'request_changes', 'escalate')"
    )
    
    op.create_check_constraint(
        'ck_review_decision_reviews',
        'correction_reviews',
        "decision IN ('approve', 'reject', 'request_changes', 'escalate')"
    )


def downgrade():
    """Drop content correction and editing system tables."""
    
    # Drop indexes first
    op.drop_index('idx_correction_impacts_created', 'correction_impacts')
    op.drop_index('idx_correction_impacts_re_embedding', 'correction_impacts')
    op.drop_index('idx_correction_impacts_correction', 'correction_impacts')
    
    op.drop_index('idx_correction_workflows_due', 'correction_workflows')
    op.drop_index('idx_correction_workflows_assigned', 'correction_workflows')
    op.drop_index('idx_correction_workflows_step', 'correction_workflows')
    op.drop_index('idx_correction_workflows_correction', 'correction_workflows')
    
    op.drop_index('idx_correction_reviews_completed', 'correction_reviews')
    op.drop_index('idx_correction_reviews_decision', 'correction_reviews')
    op.drop_index('idx_correction_reviews_reviewer', 'correction_reviews')
    op.drop_index('idx_correction_reviews_correction', 'correction_reviews')
    
    op.drop_index('idx_content_versions_hash', 'content_versions')
    op.drop_index('idx_content_versions_created_at', 'content_versions')
    op.drop_index('idx_content_versions_active', 'content_versions')
    op.drop_index('idx_content_versions_chunk_version', 'content_versions')
    
    op.drop_index('idx_content_corrections_reviewer', 'content_corrections')
    op.drop_index('idx_content_corrections_created_at', 'content_corrections')
    op.drop_index('idx_content_corrections_type', 'content_corrections')
    op.drop_index('idx_content_corrections_status', 'content_corrections')
    op.drop_index('idx_content_corrections_tenant_chunk', 'content_corrections')
    
    # Drop tables
    op.drop_table('correction_impacts')
    op.drop_table('correction_workflows')
    op.drop_table('correction_reviews')
    op.drop_table('content_versions')
    op.drop_table('content_corrections')
