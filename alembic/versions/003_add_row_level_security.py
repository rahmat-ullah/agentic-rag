"""Add Row-Level Security (RLS) for multi-tenant isolation

Revision ID: 003
Revises: 002
Create Date: 2025-09-24 14:40:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add Row-Level Security policies for multi-tenant isolation."""
    
    # Create function to get current tenant ID from session
    op.execute("""
        CREATE OR REPLACE FUNCTION get_current_tenant_id()
        RETURNS UUID AS $$
        BEGIN
            RETURN COALESCE(
                current_setting('app.current_tenant_id', true)::UUID,
                '00000000-0000-0000-0000-000000000000'::UUID
            );
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Create function to set current tenant ID
    op.execute("""
        CREATE OR REPLACE FUNCTION set_current_tenant_id(tenant_uuid UUID)
        RETURNS VOID AS $$
        BEGIN
            PERFORM set_config('app.current_tenant_id', tenant_uuid::TEXT, false);
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Create function to get current user role
    op.execute("""
        CREATE OR REPLACE FUNCTION get_current_user_role()
        RETURNS TEXT AS $$
        BEGIN
            RETURN COALESCE(
                current_setting('app.current_user_role', true),
                'viewer'
            );
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Create function to set current user role
    op.execute("""
        CREATE OR REPLACE FUNCTION set_current_user_role(user_role TEXT)
        RETURNS VOID AS $$
        BEGIN
            PERFORM set_config('app.current_user_role', user_role, false);
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Enable RLS on tenant table
    op.execute("ALTER TABLE tenant ENABLE ROW LEVEL SECURITY;")
    
    # Tenant RLS policies
    op.execute("""
        CREATE POLICY tenant_isolation_policy ON tenant
        FOR ALL
        TO PUBLIC
        USING (id = get_current_tenant_id());
    """)
    
    # Enable RLS on app_user table
    op.execute("ALTER TABLE app_user ENABLE ROW LEVEL SECURITY;")
    
    # App user RLS policies
    op.execute("""
        CREATE POLICY app_user_isolation_policy ON app_user
        FOR ALL
        TO PUBLIC
        USING (tenant_id = get_current_tenant_id());
    """)
    
    # Enable RLS on document table
    op.execute("ALTER TABLE document ENABLE ROW LEVEL SECURITY;")
    
    # Document RLS policies
    op.execute("""
        CREATE POLICY document_isolation_policy ON document
        FOR ALL
        TO PUBLIC
        USING (tenant_id = get_current_tenant_id());
    """)
    
    # Enable RLS on document_link table
    op.execute("ALTER TABLE document_link ENABLE ROW LEVEL SECURITY;")
    
    # Document link RLS policies
    op.execute("""
        CREATE POLICY document_link_isolation_policy ON document_link
        FOR ALL
        TO PUBLIC
        USING (tenant_id = get_current_tenant_id());
    """)
    
    # Enable RLS on chunk_meta table
    op.execute("ALTER TABLE chunk_meta ENABLE ROW LEVEL SECURITY;")
    
    # Chunk meta RLS policies
    op.execute("""
        CREATE POLICY chunk_meta_isolation_policy ON chunk_meta
        FOR ALL
        TO PUBLIC
        USING (tenant_id = get_current_tenant_id());
    """)
    
    # Enable RLS on feedback table
    op.execute("ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;")
    
    # Feedback RLS policies
    op.execute("""
        CREATE POLICY feedback_isolation_policy ON feedback
        FOR ALL
        TO PUBLIC
        USING (tenant_id = get_current_tenant_id());
    """)
    
    # Role-based access policies for feedback (analysts and admins can modify)
    op.execute("""
        CREATE POLICY feedback_role_based_policy ON feedback
        FOR INSERT
        TO PUBLIC
        WITH CHECK (
            tenant_id = get_current_tenant_id() AND
            get_current_user_role() IN ('admin', 'analyst')
        );
    """)
    
    op.execute("""
        CREATE POLICY feedback_update_role_policy ON feedback
        FOR UPDATE
        TO PUBLIC
        USING (
            tenant_id = get_current_tenant_id() AND
            get_current_user_role() IN ('admin', 'analyst')
        );
    """)
    
    op.execute("""
        CREATE POLICY feedback_delete_role_policy ON feedback
        FOR DELETE
        TO PUBLIC
        USING (
            tenant_id = get_current_tenant_id() AND
            get_current_user_role() = 'admin'
        );
    """)
    
    # Document modification policies (only admins and analysts can modify documents)
    op.execute("""
        CREATE POLICY document_insert_role_policy ON document
        FOR INSERT
        TO PUBLIC
        WITH CHECK (
            tenant_id = get_current_tenant_id() AND
            get_current_user_role() IN ('admin', 'analyst')
        );
    """)
    
    op.execute("""
        CREATE POLICY document_update_role_policy ON document
        FOR UPDATE
        TO PUBLIC
        USING (
            tenant_id = get_current_tenant_id() AND
            get_current_user_role() IN ('admin', 'analyst')
        );
    """)
    
    op.execute("""
        CREATE POLICY document_delete_role_policy ON document
        FOR DELETE
        TO PUBLIC
        USING (
            tenant_id = get_current_tenant_id() AND
            get_current_user_role() = 'admin'
        );
    """)
    
    # User management policies (only admins can manage users)
    op.execute("""
        CREATE POLICY app_user_insert_admin_policy ON app_user
        FOR INSERT
        TO PUBLIC
        WITH CHECK (
            tenant_id = get_current_tenant_id() AND
            get_current_user_role() = 'admin'
        );
    """)
    
    op.execute("""
        CREATE POLICY app_user_update_admin_policy ON app_user
        FOR UPDATE
        TO PUBLIC
        USING (
            tenant_id = get_current_tenant_id() AND
            get_current_user_role() = 'admin'
        );
    """)
    
    op.execute("""
        CREATE POLICY app_user_delete_admin_policy ON app_user
        FOR DELETE
        TO PUBLIC
        USING (
            tenant_id = get_current_tenant_id() AND
            get_current_user_role() = 'admin'
        );
    """)


def downgrade() -> None:
    """Remove Row-Level Security policies."""
    
    # Drop policies
    op.execute("DROP POLICY IF EXISTS app_user_delete_admin_policy ON app_user;")
    op.execute("DROP POLICY IF EXISTS app_user_update_admin_policy ON app_user;")
    op.execute("DROP POLICY IF EXISTS app_user_insert_admin_policy ON app_user;")
    
    op.execute("DROP POLICY IF EXISTS document_delete_role_policy ON document;")
    op.execute("DROP POLICY IF EXISTS document_update_role_policy ON document;")
    op.execute("DROP POLICY IF EXISTS document_insert_role_policy ON document;")
    
    op.execute("DROP POLICY IF EXISTS feedback_delete_role_policy ON feedback;")
    op.execute("DROP POLICY IF EXISTS feedback_update_role_policy ON feedback;")
    op.execute("DROP POLICY IF EXISTS feedback_role_based_policy ON feedback;")
    op.execute("DROP POLICY IF EXISTS feedback_isolation_policy ON feedback;")
    
    op.execute("DROP POLICY IF EXISTS chunk_meta_isolation_policy ON chunk_meta;")
    op.execute("DROP POLICY IF EXISTS document_link_isolation_policy ON document_link;")
    op.execute("DROP POLICY IF EXISTS document_isolation_policy ON document;")
    op.execute("DROP POLICY IF EXISTS app_user_isolation_policy ON app_user;")
    op.execute("DROP POLICY IF EXISTS tenant_isolation_policy ON tenant;")
    
    # Disable RLS
    op.execute("ALTER TABLE feedback DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE chunk_meta DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE document_link DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE document DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE app_user DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE tenant DISABLE ROW LEVEL SECURITY;")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS set_current_user_role(TEXT);")
    op.execute("DROP FUNCTION IF EXISTS get_current_user_role();")
    op.execute("DROP FUNCTION IF EXISTS set_current_tenant_id(UUID);")
    op.execute("DROP FUNCTION IF EXISTS get_current_tenant_id();")
