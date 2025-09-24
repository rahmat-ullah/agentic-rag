"""Add database utility functions and procedures

Revision ID: 004
Revises: 003
Create Date: 2025-09-24 14:45:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add database utility functions and procedures."""
    
    # Function to validate document SHA256 format
    op.execute("""
        CREATE OR REPLACE FUNCTION validate_sha256(hash_value TEXT)
        RETURNS BOOLEAN AS $$
        BEGIN
            RETURN hash_value ~ '^[a-fA-F0-9]{64}$';
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    
    # Function to validate email format
    op.execute("""
        CREATE OR REPLACE FUNCTION validate_email(email_value TEXT)
        RETURNS BOOLEAN AS $$
        BEGIN
            RETURN email_value ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$';
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    
    # Function to generate document version
    op.execute("""
        CREATE OR REPLACE FUNCTION get_next_document_version(
            p_tenant_id UUID,
            p_sha256 TEXT
        )
        RETURNS INTEGER AS $$
        DECLARE
            next_version INTEGER;
        BEGIN
            SELECT COALESCE(MAX(version), 0) + 1
            INTO next_version
            FROM document
            WHERE tenant_id = p_tenant_id AND sha256 = p_sha256;
            
            RETURN next_version;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Function to get document statistics
    op.execute("""
        CREATE OR REPLACE FUNCTION get_document_stats(p_tenant_id UUID)
        RETURNS TABLE(
            kind TEXT,
            count BIGINT,
            total_pages BIGINT,
            avg_pages NUMERIC
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                d.kind::TEXT,
                COUNT(*)::BIGINT,
                COALESCE(SUM(d.pages), 0)::BIGINT,
                COALESCE(AVG(d.pages), 0)::NUMERIC
            FROM document d
            WHERE d.tenant_id = p_tenant_id
            GROUP BY d.kind
            ORDER BY d.kind;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Function to get chunk statistics
    op.execute("""
        CREATE OR REPLACE FUNCTION get_chunk_stats(p_tenant_id UUID)
        RETURNS TABLE(
            document_kind TEXT,
            chunk_count BIGINT,
            total_tokens BIGINT,
            avg_tokens NUMERIC,
            table_chunks BIGINT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                d.kind::TEXT,
                COUNT(c.id)::BIGINT,
                COALESCE(SUM(c.token_count), 0)::BIGINT,
                COALESCE(AVG(c.token_count), 0)::NUMERIC,
                COUNT(CASE WHEN c.is_table THEN 1 END)::BIGINT
            FROM document d
            LEFT JOIN chunk_meta c ON d.id = c.document_id
            WHERE d.tenant_id = p_tenant_id AND (c.retired IS NULL OR c.retired = false)
            GROUP BY d.kind
            ORDER BY d.kind;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Function to clean up retired chunks
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_retired_chunks(
            p_tenant_id UUID,
            p_days_old INTEGER DEFAULT 30
        )
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            DELETE FROM chunk_meta
            WHERE tenant_id = p_tenant_id
              AND retired = true
              AND updated_at < NOW() - INTERVAL '1 day' * p_days_old;
            
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Function to update document link confidence
    op.execute("""
        CREATE OR REPLACE FUNCTION update_document_link_confidence(
            p_tenant_id UUID,
            p_rfq_id UUID,
            p_offer_id UUID,
            p_offer_type TEXT,
            p_new_confidence FLOAT
        )
        RETURNS BOOLEAN AS $$
        DECLARE
            updated_rows INTEGER;
        BEGIN
            -- Validate confidence range
            IF p_new_confidence < 0 OR p_new_confidence > 1 THEN
                RAISE EXCEPTION 'Confidence must be between 0 and 1';
            END IF;
            
            -- Validate offer type
            IF p_offer_type NOT IN ('technical', 'commercial', 'pricing') THEN
                RAISE EXCEPTION 'Invalid offer type: %', p_offer_type;
            END IF;
            
            UPDATE document_link
            SET confidence = p_new_confidence,
                updated_at = NOW()
            WHERE tenant_id = p_tenant_id
              AND rfq_id = p_rfq_id
              AND offer_id = p_offer_id
              AND offer_type = p_offer_type;
            
            GET DIAGNOSTICS updated_rows = ROW_COUNT;
            RETURN updated_rows > 0;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Function to get feedback summary
    op.execute("""
        CREATE OR REPLACE FUNCTION get_feedback_summary(p_tenant_id UUID)
        RETURNS TABLE(
            label TEXT,
            count BIGINT,
            percentage NUMERIC
        ) AS $$
        DECLARE
            total_feedback BIGINT;
        BEGIN
            -- Get total feedback count
            SELECT COUNT(*) INTO total_feedback
            FROM feedback
            WHERE tenant_id = p_tenant_id;
            
            -- Return summary
            RETURN QUERY
            SELECT 
                f.label::TEXT,
                COUNT(*)::BIGINT,
                CASE 
                    WHEN total_feedback > 0 THEN 
                        ROUND((COUNT(*)::NUMERIC / total_feedback::NUMERIC) * 100, 2)
                    ELSE 0
                END::NUMERIC
            FROM feedback f
            WHERE f.tenant_id = p_tenant_id
            GROUP BY f.label
            ORDER BY COUNT(*) DESC;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Trigger function to update updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create triggers for updated_at columns
    op.execute("""
        CREATE TRIGGER update_tenant_updated_at
            BEFORE UPDATE ON tenant
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_app_user_updated_at
            BEFORE UPDATE ON app_user
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_document_updated_at
            BEFORE UPDATE ON document
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_document_link_updated_at
            BEFORE UPDATE ON document_link
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_chunk_meta_updated_at
            BEFORE UPDATE ON chunk_meta
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)
    
    # Add check constraints for data validation
    op.execute("""
        ALTER TABLE document 
        ADD CONSTRAINT ck_document_sha256_format 
        CHECK (validate_sha256(sha256));
    """)
    
    op.execute("""
        ALTER TABLE app_user 
        ADD CONSTRAINT ck_app_user_email_format 
        CHECK (validate_email(email));
    """)
    
    op.execute("""
        ALTER TABLE document 
        ADD CONSTRAINT ck_document_version_positive 
        CHECK (version > 0);
    """)
    
    op.execute("""
        ALTER TABLE document 
        ADD CONSTRAINT ck_document_pages_positive 
        CHECK (pages IS NULL OR pages > 0);
    """)
    
    op.execute("""
        ALTER TABLE chunk_meta 
        ADD CONSTRAINT ck_chunk_meta_token_count_positive 
        CHECK (token_count IS NULL OR token_count > 0);
    """)
    
    op.execute("""
        ALTER TABLE chunk_meta 
        ADD CONSTRAINT ck_chunk_meta_page_range 
        CHECK (page_from IS NULL OR page_to IS NULL OR page_from <= page_to);
    """)


def downgrade() -> None:
    """Remove database utility functions and procedures."""
    
    # Drop check constraints
    op.execute("ALTER TABLE chunk_meta DROP CONSTRAINT IF EXISTS ck_chunk_meta_page_range;")
    op.execute("ALTER TABLE chunk_meta DROP CONSTRAINT IF EXISTS ck_chunk_meta_token_count_positive;")
    op.execute("ALTER TABLE document DROP CONSTRAINT IF EXISTS ck_document_pages_positive;")
    op.execute("ALTER TABLE document DROP CONSTRAINT IF EXISTS ck_document_version_positive;")
    op.execute("ALTER TABLE app_user DROP CONSTRAINT IF EXISTS ck_app_user_email_format;")
    op.execute("ALTER TABLE document DROP CONSTRAINT IF EXISTS ck_document_sha256_format;")
    
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS update_chunk_meta_updated_at ON chunk_meta;")
    op.execute("DROP TRIGGER IF EXISTS update_document_link_updated_at ON document_link;")
    op.execute("DROP TRIGGER IF EXISTS update_document_updated_at ON document;")
    op.execute("DROP TRIGGER IF EXISTS update_app_user_updated_at ON app_user;")
    op.execute("DROP TRIGGER IF EXISTS update_tenant_updated_at ON tenant;")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")
    op.execute("DROP FUNCTION IF EXISTS get_feedback_summary(UUID);")
    op.execute("DROP FUNCTION IF EXISTS update_document_link_confidence(UUID, UUID, UUID, TEXT, FLOAT);")
    op.execute("DROP FUNCTION IF EXISTS cleanup_retired_chunks(UUID, INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS get_chunk_stats(UUID);")
    op.execute("DROP FUNCTION IF EXISTS get_document_stats(UUID);")
    op.execute("DROP FUNCTION IF EXISTS get_next_document_version(UUID, TEXT);")
    op.execute("DROP FUNCTION IF EXISTS validate_email(TEXT);")
    op.execute("DROP FUNCTION IF EXISTS validate_sha256(TEXT);")
