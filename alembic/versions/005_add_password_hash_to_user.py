"""Add password_hash field to user table

Revision ID: 005
Revises: 004
Create Date: 2025-09-24 14:50:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add password_hash field to app_user table."""
    
    # Add password_hash column
    op.add_column('app_user', sa.Column('password_hash', sa.String(255), nullable=True))
    
    # Set a default password hash for existing users (they'll need to reset)
    # This is a bcrypt hash of "changeme123" - users should change this immediately
    default_hash = "$2b$12$LQv3c1yqBWVHxkd0LQ4YNu3plUcABvzzaCjI5.KMJbpVRdIK5WSMO"
    op.execute(f"UPDATE app_user SET password_hash = '{default_hash}' WHERE password_hash IS NULL")
    
    # Make password_hash non-nullable
    op.alter_column('app_user', 'password_hash', nullable=False)


def downgrade() -> None:
    """Remove password_hash field from app_user table."""
    
    # Drop password_hash column
    op.drop_column('app_user', 'password_hash')
