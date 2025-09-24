"""Add indexes and constraints for performance optimization

Revision ID: 002
Revises: 001
Create Date: 2025-09-24 14:35:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add indexes and constraints for performance optimization."""
    
    # Document table indexes
    op.create_index('idx_document_tenant_kind', 'document', ['tenant_id', 'kind'])
    op.create_index('idx_document_created_at', 'document', ['created_at'])
    op.create_index('idx_document_sha256', 'document', ['sha256'])
    op.create_index('idx_document_created_by', 'document', ['created_by'])
    
    # Document link table indexes
    op.create_index('idx_document_link_tenant_rfq_type', 'document_link', ['tenant_id', 'rfq_id', 'offer_type'])
    op.create_index('idx_document_link_confidence', 'document_link', ['confidence'])
    op.create_index('idx_document_link_rfq_id', 'document_link', ['rfq_id'])
    op.create_index('idx_document_link_offer_id', 'document_link', ['offer_id'])
    
    # Chunk meta table indexes
    op.create_index('idx_chunk_meta_tenant_document', 'chunk_meta', ['tenant_id', 'document_id'])
    op.create_index('idx_chunk_meta_embedding_version', 'chunk_meta', ['embedding_model_version', 'tenant_id'])
    op.create_index('idx_chunk_meta_retired', 'chunk_meta', ['retired'])
    op.create_index('idx_chunk_meta_document_id', 'chunk_meta', ['document_id'])
    op.create_index('idx_chunk_meta_hash', 'chunk_meta', ['hash'])
    
    # Feedback table indexes
    op.create_index('idx_feedback_tenant_query', 'feedback', ['tenant_id', 'query'])
    op.create_index('idx_feedback_label', 'feedback', ['label'])
    op.create_index('idx_feedback_created_at', 'feedback', ['created_at'])
    op.create_index('idx_feedback_rfq_id', 'feedback', ['rfq_id'])
    op.create_index('idx_feedback_offer_id', 'feedback', ['offer_id'])
    op.create_index('idx_feedback_chunk_id', 'feedback', ['chunk_id'])
    op.create_index('idx_feedback_created_by', 'feedback', ['created_by'])
    
    # App user table indexes
    op.create_index('idx_app_user_tenant_id', 'app_user', ['tenant_id'])
    op.create_index('idx_app_user_email', 'app_user', ['email'])
    op.create_index('idx_app_user_role', 'app_user', ['role'])
    op.create_index('idx_app_user_is_active', 'app_user', ['is_active'])
    
    # Tenant table indexes
    op.create_index('idx_tenant_name', 'tenant', ['name'])
    
    # Composite indexes for common query patterns
    op.create_index('idx_document_tenant_kind_created', 'document', ['tenant_id', 'kind', 'created_at'])
    op.create_index('idx_chunk_meta_tenant_retired_embedding', 'chunk_meta', ['tenant_id', 'retired', 'embedding_model_version'])
    op.create_index('idx_feedback_tenant_label_created', 'feedback', ['tenant_id', 'label', 'created_at'])
    
    # Partial indexes for active records
    op.create_index('idx_app_user_active_tenant', 'app_user', ['tenant_id'], 
                   postgresql_where=sa.text('is_active = true'))
    op.create_index('idx_chunk_meta_active_tenant', 'chunk_meta', ['tenant_id', 'document_id'], 
                   postgresql_where=sa.text('retired = false'))


def downgrade() -> None:
    """Remove indexes and constraints."""
    
    # Drop partial indexes
    op.drop_index('idx_chunk_meta_active_tenant', 'chunk_meta')
    op.drop_index('idx_app_user_active_tenant', 'app_user')
    
    # Drop composite indexes
    op.drop_index('idx_feedback_tenant_label_created', 'feedback')
    op.drop_index('idx_chunk_meta_tenant_retired_embedding', 'chunk_meta')
    op.drop_index('idx_document_tenant_kind_created', 'document')
    
    # Drop tenant table indexes
    op.drop_index('idx_tenant_name', 'tenant')
    
    # Drop app user table indexes
    op.drop_index('idx_app_user_is_active', 'app_user')
    op.drop_index('idx_app_user_role', 'app_user')
    op.drop_index('idx_app_user_email', 'app_user')
    op.drop_index('idx_app_user_tenant_id', 'app_user')
    
    # Drop feedback table indexes
    op.drop_index('idx_feedback_created_by', 'feedback')
    op.drop_index('idx_feedback_chunk_id', 'feedback')
    op.drop_index('idx_feedback_offer_id', 'feedback')
    op.drop_index('idx_feedback_rfq_id', 'feedback')
    op.drop_index('idx_feedback_created_at', 'feedback')
    op.drop_index('idx_feedback_label', 'feedback')
    op.drop_index('idx_feedback_tenant_query', 'feedback')
    
    # Drop chunk meta table indexes
    op.drop_index('idx_chunk_meta_hash', 'chunk_meta')
    op.drop_index('idx_chunk_meta_document_id', 'chunk_meta')
    op.drop_index('idx_chunk_meta_retired', 'chunk_meta')
    op.drop_index('idx_chunk_meta_embedding_version', 'chunk_meta')
    op.drop_index('idx_chunk_meta_tenant_document', 'chunk_meta')
    
    # Drop document link table indexes
    op.drop_index('idx_document_link_offer_id', 'document_link')
    op.drop_index('idx_document_link_rfq_id', 'document_link')
    op.drop_index('idx_document_link_confidence', 'document_link')
    op.drop_index('idx_document_link_tenant_rfq_type', 'document_link')
    
    # Drop document table indexes
    op.drop_index('idx_document_created_by', 'document')
    op.drop_index('idx_document_sha256', 'document')
    op.drop_index('idx_document_created_at', 'document')
    op.drop_index('idx_document_tenant_kind', 'document')
