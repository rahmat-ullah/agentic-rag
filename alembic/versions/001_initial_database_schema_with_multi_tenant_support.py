"""Initial database schema with multi-tenant support

Revision ID: 001
Revises: 
Create Date: 2025-09-24 14:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    
    # Create ENUM types
    document_kind_enum = postgresql.ENUM(
        'RFQ', 'RFP', 'Tender', 'OfferTech', 'OfferComm', 'Pricing',
        name='documentkind'
    )
    document_kind_enum.create(op.get_bind())
    
    user_role_enum = postgresql.ENUM(
        'admin', 'analyst', 'viewer',
        name='userrole'
    )
    user_role_enum.create(op.get_bind())
    
    feedback_label_enum = postgresql.ENUM(
        'up', 'down', 'edit', 'bad_link', 'good_link',
        name='feedbacklabel'
    )
    feedback_label_enum.create(op.get_bind())
    
    # Create tenant table
    op.create_table(
        'tenant',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create app_user table
    op.create_table(
        'app_user',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('role', user_role_enum, nullable=False),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
    )
    
    # Create document table
    op.create_table(
        'document',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('kind', document_kind_enum, nullable=False),
        sa.Column('title', sa.String(500)),
        sa.Column('source_uri', sa.String(1000)),
        sa.Column('sha256', sa.String(64), nullable=False),
        sa.Column('version', sa.Integer, default=1),
        sa.Column('pages', sa.Integer),
        sa.Column('created_by', postgresql.UUID(as_uuid=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['created_by'], ['app_user.id']),
        sa.UniqueConstraint('tenant_id', 'sha256', name='uq_document_tenant_sha256'),
    )
    
    # Create document_link table
    op.create_table(
        'document_link',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rfq_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('offer_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('offer_type', sa.String(50), nullable=False),
        sa.Column('confidence', sa.Float, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['rfq_id'], ['document.id']),
        sa.ForeignKeyConstraint(['offer_id'], ['document.id']),
        sa.CheckConstraint("offer_type IN ('technical', 'commercial', 'pricing')", name='ck_offer_type'),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1", name='ck_confidence_range'),
    )
    
    # Create chunk_meta table
    op.create_table(
        'chunk_meta',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('page_from', sa.Integer),
        sa.Column('page_to', sa.Integer),
        sa.Column('section_path', postgresql.ARRAY(sa.String)),
        sa.Column('token_count', sa.Integer),
        sa.Column('hash', sa.String(64)),
        sa.Column('is_table', sa.Boolean, default=False),
        sa.Column('retired', sa.Boolean, default=False),
        sa.Column('embedding_model_version', sa.String(50)),
        sa.Column('embedding_created_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['document_id'], ['document.id']),
        sa.UniqueConstraint('tenant_id', 'hash', name='uq_chunk_meta_tenant_hash'),
    )
    
    # Create feedback table
    op.create_table(
        'feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('query', sa.Text, nullable=False),
        sa.Column('rfq_id', postgresql.UUID(as_uuid=True)),
        sa.Column('offer_id', postgresql.UUID(as_uuid=True)),
        sa.Column('chunk_id', postgresql.UUID(as_uuid=True)),
        sa.Column('label', feedback_label_enum, nullable=False),
        sa.Column('notes', sa.Text),
        sa.Column('created_by', postgresql.UUID(as_uuid=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['rfq_id'], ['document.id']),
        sa.ForeignKeyConstraint(['offer_id'], ['document.id']),
        sa.ForeignKeyConstraint(['chunk_id'], ['chunk_meta.id']),
        sa.ForeignKeyConstraint(['created_by'], ['app_user.id']),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    
    # Drop tables in reverse order
    op.drop_table('feedback')
    op.drop_table('chunk_meta')
    op.drop_table('document_link')
    op.drop_table('document')
    op.drop_table('app_user')
    op.drop_table('tenant')
    
    # Drop ENUM types
    feedback_label_enum = postgresql.ENUM(name='feedbacklabel')
    feedback_label_enum.drop(op.get_bind())
    
    user_role_enum = postgresql.ENUM(name='userrole')
    user_role_enum.drop(op.get_bind())
    
    document_kind_enum = postgresql.ENUM(name='documentkind')
    document_kind_enum.drop(op.get_bind())
