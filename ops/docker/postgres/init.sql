-- PostgreSQL initialization script for Agentic RAG System
-- This script sets up the initial database configuration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create application database if it doesn't exist
-- (This is handled by POSTGRES_DB environment variable, but kept for reference)

-- Set up basic configuration
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;

-- Create initial schema (will be managed by Alembic migrations)
-- This is just a placeholder for development setup

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE agentic_rag TO postgres;

-- Create development user (optional)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'agentic_rag_dev') THEN
        CREATE ROLE agentic_rag_dev WITH LOGIN PASSWORD 'dev_password';
        GRANT ALL PRIVILEGES ON DATABASE agentic_rag TO agentic_rag_dev;
    END IF;
END
$$;
