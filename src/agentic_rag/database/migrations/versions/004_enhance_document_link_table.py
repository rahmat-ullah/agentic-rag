"""Enhance document_link table for advanced linking features

Revision ID: 004_enhance_document_link
Revises: 003_add_chunk_meta_indexes
Create Date: 2024-12-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_enhance_document_link'
down_revision = '003_add_chunk_meta_indexes'
branch_labels = None
depends_on = None


def upgrade():
    """Add enhanced fields to document_link table for advanced linking features."""
    
    # Add new columns to document_link table
    op.add_column('document_link', sa.Column('link_type', sa.String(50), nullable=False, server_default='manual'))
    op.add_column('document_link', sa.Column('created_by', sa.UUID(), nullable=True))
    op.add_column('document_link', sa.Column('validated_by', sa.UUID(), nullable=True))
    op.add_column('document_link', sa.Column('validated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('document_link', sa.Column('link_metadata', postgresql.JSONB(), nullable=True))
    op.add_column('document_link', sa.Column('quality_score', sa.Float(), nullable=True))
    op.add_column('document_link', sa.Column('user_feedback', sa.String(20), nullable=True))
    op.add_column('document_link', sa.Column('feedback_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('document_link', sa.Column('notes', sa.Text(), nullable=True))
    
    # Add foreign key constraints for user references
    op.create_foreign_key(
        'fk_document_link_created_by',
        'document_link', 'app_user',
        ['created_by'], ['id']
    )
    
    op.create_foreign_key(
        'fk_document_link_validated_by',
        'document_link', 'app_user',
        ['validated_by'], ['id']
    )
    
    # Add check constraints
    op.create_check_constraint(
        'ck_link_type',
        'document_link',
        "link_type IN ('manual', 'automatic', 'suggested')"
    )
    
    op.create_check_constraint(
        'ck_quality_score_range',
        'document_link',
        "quality_score IS NULL OR (quality_score BETWEEN 0 AND 1)"
    )
    
    op.create_check_constraint(
        'ck_user_feedback',
        'document_link',
        "user_feedback IS NULL OR user_feedback IN ('accepted', 'rejected', 'modified')"
    )
    
    # Add new indexes for performance
    op.create_index(
        'idx_document_link_link_type',
        'document_link',
        ['tenant_id', 'link_type']
    )
    
    op.create_index(
        'idx_document_link_created_by',
        'document_link',
        ['created_by']
    )
    
    op.create_index(
        'idx_document_link_quality_score',
        'document_link',
        ['quality_score']
    )
    
    op.create_index(
        'idx_document_link_validated',
        'document_link',
        ['validated_by', 'validated_at']
    )
    
    # Add unique constraint for tenant, rfq, offer combination
    op.create_unique_constraint(
        'uq_document_link_tenant_rfq_offer',
        'document_link',
        ['tenant_id', 'rfq_id', 'offer_id']
    )


def downgrade():
    """Remove enhanced fields from document_link table."""
    
    # Drop constraints and indexes
    op.drop_constraint('uq_document_link_tenant_rfq_offer', 'document_link', type_='unique')
    op.drop_index('idx_document_link_validated', table_name='document_link')
    op.drop_index('idx_document_link_quality_score', table_name='document_link')
    op.drop_index('idx_document_link_created_by', table_name='document_link')
    op.drop_index('idx_document_link_link_type', table_name='document_link')
    
    op.drop_constraint('ck_user_feedback', 'document_link', type_='check')
    op.drop_constraint('ck_quality_score_range', 'document_link', type_='check')
    op.drop_constraint('ck_link_type', 'document_link', type_='check')
    
    op.drop_constraint('fk_document_link_validated_by', 'document_link', type_='foreignkey')
    op.drop_constraint('fk_document_link_created_by', 'document_link', type_='foreignkey')
    
    # Drop columns
    op.drop_column('document_link', 'notes')
    op.drop_column('document_link', 'feedback_at')
    op.drop_column('document_link', 'user_feedback')
    op.drop_column('document_link', 'quality_score')
    op.drop_column('document_link', 'link_metadata')
    op.drop_column('document_link', 'validated_at')
    op.drop_column('document_link', 'validated_by')
    op.drop_column('document_link', 'created_by')
    op.drop_column('document_link', 'link_type')
