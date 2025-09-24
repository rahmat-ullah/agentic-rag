-- PostgreSQL initialization script for Agentic RAG System
-- This script sets up the database with required extensions and configurations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create application database if it doesn't exist
-- (This is handled by POSTGRES_DB environment variable, but included for completeness)

-- Set up database configuration
ALTER DATABASE agentic_rag SET timezone TO 'UTC';
ALTER DATABASE agentic_rag SET log_statement TO 'all';
ALTER DATABASE agentic_rag SET log_min_duration_statement TO 1000;

-- Create application user with limited privileges (for production)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'agentic_rag_app') THEN
        CREATE ROLE agentic_rag_app WITH LOGIN PASSWORD 'app_password';
    END IF;
END
$$;

-- Grant necessary privileges to application user
GRANT CONNECT ON DATABASE agentic_rag TO agentic_rag_app;
GRANT USAGE ON SCHEMA public TO agentic_rag_app;
GRANT CREATE ON SCHEMA public TO agentic_rag_app;

-- Create read-only user for analytics/reporting
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'agentic_rag_readonly') THEN
        CREATE ROLE agentic_rag_readonly WITH LOGIN PASSWORD 'readonly_password';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE agentic_rag TO agentic_rag_readonly;
GRANT USAGE ON SCHEMA public TO agentic_rag_readonly;

-- Set up Row-Level Security (RLS) helper functions
-- These will be used by the application for tenant isolation

-- Function to get current tenant ID from application context
CREATE OR REPLACE FUNCTION get_current_tenant_id()
RETURNS UUID AS $$
BEGIN
    -- This will be set by the application using SET LOCAL
    RETURN COALESCE(
        NULLIF(current_setting('app.current_tenant_id', true), ''),
        '00000000-0000-0000-0000-000000000000'
    )::UUID;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get current user ID from application context
CREATE OR REPLACE FUNCTION get_current_user_id()
RETURNS UUID AS $$
BEGIN
    -- This will be set by the application using SET LOCAL
    RETURN COALESCE(
        NULLIF(current_setting('app.current_user_id', true), ''),
        '00000000-0000-0000-0000-000000000000'
    )::UUID;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get current user role from application context
CREATE OR REPLACE FUNCTION get_current_user_role()
RETURNS TEXT AS $$
BEGIN
    -- This will be set by the application using SET LOCAL
    RETURN COALESCE(
        NULLIF(current_setting('app.current_user_role', true), ''),
        'viewer'
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to check if current user is admin
CREATE OR REPLACE FUNCTION is_admin()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN get_current_user_role() = 'admin';
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to check if current user can access tenant
CREATE OR REPLACE FUNCTION can_access_tenant(tenant_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    -- Admin can access any tenant, others only their own
    RETURN is_admin() OR get_current_tenant_id() = tenant_id;
END;
$$ LANGUAGE plpgsql STABLE;

-- Create indexes for performance monitoring
CREATE INDEX IF NOT EXISTS idx_pg_stat_statements_query 
ON pg_stat_statements(query);

-- Set up logging for security auditing
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    old_values JSONB,
    new_values JSONB,
    user_id UUID,
    tenant_id UUID,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id TEXT DEFAULT current_setting('application_name', true)
);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, old_values, user_id, tenant_id)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), get_current_user_id(), get_current_tenant_id());
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, old_values, new_values, user_id, tenant_id)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), row_to_json(NEW), get_current_user_id(), get_current_tenant_id());
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, new_values, user_id, tenant_id)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(NEW), get_current_user_id(), get_current_tenant_id());
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Performance optimization settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET pg_stat_statements.max = 10000;
ALTER SYSTEM SET log_min_duration_statement = 1000;
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_lock_waits = on;

-- Memory and performance settings for development
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Connection settings
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET max_prepared_transactions = 0;

-- Security settings
ALTER SYSTEM SET ssl = off;  -- Enable in production
ALTER SYSTEM SET password_encryption = 'md5';
ALTER SYSTEM SET row_security = on;

-- Reload configuration
SELECT pg_reload_conf();

-- Create data directory structure if needed
\echo 'PostgreSQL initialization completed successfully'
\echo 'Database: agentic_rag'
\echo 'Extensions: uuid-ossp, pgcrypto, pg_stat_statements'
\echo 'RLS helper functions created'
\echo 'Audit logging configured'
\echo 'Performance settings optimized for development'
