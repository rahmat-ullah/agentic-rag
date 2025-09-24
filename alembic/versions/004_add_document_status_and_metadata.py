"""Add document status and metadata fields

Revision ID: 004
Revises: 003
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade():
    """Add document status and metadata fields."""
    
    # Create DocumentStatus enum
    document_status_enum = postgresql.ENUM(
        'uploaded', 'processing', 'ready', 'failed', 'deleted',
        name='documentstatus',
        create_type=False
    )
    document_status_enum.create(op.get_bind(), checkfirst=True)
    
    # Add new columns to document table
    op.add_column('document', sa.Column('status', document_status_enum, nullable=False, server_default='uploaded'))
    op.add_column('document', sa.Column('processing_progress', sa.Float(), nullable=True, server_default='0.0'))
    op.add_column('document', sa.Column('processing_error', sa.Text(), nullable=True))
    op.add_column('document', sa.Column('file_size', sa.BigInteger(), nullable=True))
    op.add_column('document', sa.Column('chunk_count', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('document', sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True))
    
    # Add check constraints
    op.create_check_constraint(
        'ck_processing_progress_range',
        'document',
        'processing_progress BETWEEN 0 AND 1'
    )
    op.create_check_constraint(
        'ck_chunk_count_positive',
        'document',
        'chunk_count >= 0'
    )
    
    # Add new indexes
    op.create_index('idx_document_tenant_status', 'document', ['tenant_id', 'status'])
    op.create_index('idx_document_deleted_at', 'document', ['deleted_at'])


def downgrade():
    """Remove document status and metadata fields."""
    
    # Drop indexes
    op.drop_index('idx_document_deleted_at', table_name='document')
    op.drop_index('idx_document_tenant_status', table_name='document')
    
    # Drop check constraints
    op.drop_constraint('ck_chunk_count_positive', 'document', type_='check')
    op.drop_constraint('ck_processing_progress_range', 'document', type_='check')
    
    # Drop columns
    op.drop_column('document', 'deleted_at')
    op.drop_column('document', 'chunk_count')
    op.drop_column('document', 'file_size')
    op.drop_column('document', 'processing_error')
    op.drop_column('document', 'processing_progress')
    op.drop_column('document', 'status')
    
    # Drop enum type
    document_status_enum = postgresql.ENUM(name='documentstatus')
    document_status_enum.drop(op.get_bind(), checkfirst=True)
