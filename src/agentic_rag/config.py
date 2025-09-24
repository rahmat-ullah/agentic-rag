"""
Configuration management for Agentic RAG System.

This module provides centralized configuration management using Pydantic Settings
with support for environment variables, validation, and different environments.
"""

import os
from enum import Enum
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic.networks import AnyHttpUrl, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environment types."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="agentic_rag", env="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    postgres_url: Optional[PostgresDsn] = Field(default=None, env="POSTGRES_URL")
    
    # Pool settings
    db_pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    db_pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    db_echo: bool = Field(default=False, env="DB_ECHO")
    
    @field_validator("postgres_url", mode="before")
    @classmethod
    def build_postgres_url(cls, v, info):
        """Build PostgreSQL URL if not provided."""
        if v:
            return v

        values = info.data if info else {}
        return PostgresDsn.build(
            scheme="postgresql",
            username=values.get("postgres_user"),
            password=values.get("postgres_password"),
            host=values.get("postgres_host"),
            port=values.get("postgres_port"),
            path=f"/{values.get('postgres_db')}",
        )


class VectorDatabaseSettings(BaseSettings):
    """Vector database configuration settings."""
    
    # ChromaDB
    chromadb_host: str = Field(default="localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(default=8001, env="CHROMADB_PORT")
    chromadb_url: Optional[AnyHttpUrl] = Field(default=None, env="CHROMADB_URL")
    
    # Collections
    chromadb_rfq_collection: str = Field(default="rfq_collection", env="CHROMADB_RFQ_COLLECTION")
    chromadb_offer_collection: str = Field(default="offer_collection", env="CHROMADB_OFFER_COLLECTION")
    
    @field_validator("chromadb_url", mode="before")
    @classmethod
    def build_chromadb_url(cls, v, info):
        """Build ChromaDB URL if not provided."""
        if v:
            return v

        values = info.data if info else {}
        host = values.get("chromadb_host")
        port = values.get("chromadb_port")
        return f"http://{host}:{port}"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_url: Optional[RedisDsn] = Field(default=None, env="REDIS_URL")
    
    # Pool settings
    redis_pool_size: int = Field(default=10, env="REDIS_POOL_SIZE")
    redis_pool_timeout: int = Field(default=30, env="REDIS_POOL_TIMEOUT")
    
    @field_validator("redis_url", mode="before")
    @classmethod
    def build_redis_url(cls, v, info):
        """Build Redis URL if not provided."""
        if v:
            return v

        values = info.data if info else {}
        password = values.get("redis_password")
        auth = f":{password}@" if password else ""
        host = values.get("redis_host")
        port = values.get("redis_port")
        db = values.get("redis_db")

        return f"redis://{auth}{host}:{port}/{db}"


class ObjectStorageSettings(BaseSettings):
    """Object storage configuration settings."""

    # MinIO/S3
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, env="MINIO_SECURE")
    minio_region: str = Field(default="us-east-1", env="MINIO_REGION")

    # Buckets
    minio_bucket_documents: str = Field(default="documents", env="MINIO_BUCKET_DOCUMENTS")
    minio_bucket_thumbnails: str = Field(default="thumbnails", env="MINIO_BUCKET_THUMBNAILS")
    minio_bucket_exports: str = Field(default="exports", env="MINIO_BUCKET_EXPORTS")

    # Storage paths and encryption
    storage_path_prefix: str = Field(default="uploads", env="STORAGE_PATH_PREFIX")
    storage_encryption_enabled: bool = Field(default=True, env="STORAGE_ENCRYPTION_ENABLED")
    storage_encryption_key: Optional[str] = Field(default=None, env="STORAGE_ENCRYPTION_KEY")


class AISettings(BaseSettings):
    """AI/ML service configuration settings."""
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")
    openai_embedding_model: str = Field(default="text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    openai_chat_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_CHAT_MODEL")
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    
    # Granite-Docling
    docling_endpoint: str = Field(default="http://localhost:9002", env="DOCLING_ENDPOINT")
    docling_max_pages: int = Field(default=2000, env="DOCLING_MAX_PAGES")
    docling_timeout: int = Field(default=300, env="DOCLING_TIMEOUT")
    docling_batch_size: int = Field(default=1, env="DOCLING_BATCH_SIZE")
    docling_retry_attempts: int = Field(default=3, env="DOCLING_RETRY_ATTEMPTS")
    docling_retry_delay: float = Field(default=1.0, env="DOCLING_RETRY_DELAY")
    docling_circuit_breaker_threshold: int = Field(default=5, env="DOCLING_CIRCUIT_BREAKER_THRESHOLD")
    docling_circuit_breaker_timeout: int = Field(default=60, env="DOCLING_CIRCUIT_BREAKER_TIMEOUT")


class RetrievalSettings(BaseSettings):
    """Retrieval configuration settings."""
    
    # Three-hop search parameters
    k1_rfq_candidates: int = Field(default=12, env="K1_RFQ_CANDIDATES")
    k1_rerank_top: int = Field(default=3, env="K1_RERANK_TOP")
    k3_offer_candidates: int = Field(default=40, env="K3_OFFER_CANDIDATES")
    k3_rerank_top: int = Field(default=12, env="K3_RERANK_TOP")
    link_confidence_threshold: float = Field(default=0.6, env="LINK_CONFIDENCE_THRESHOLD")
    
    # Embedding settings
    embedding_dimension: int = Field(default=3072, env="EMBEDDING_DIMENSION")
    chunk_size: int = Field(default=1024, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, env="CHUNK_OVERLAP")


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    # JWT
    jwt_secret_key: str = Field(default="dev-jwt-secret-key", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Password hashing
    password_hash_algorithm: str = Field(default="bcrypt", env="PASSWORD_HASH_ALGORITHM")
    password_hash_rounds: int = Field(default=12, env="PASSWORD_HASH_ROUNDS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")


class UploadSettings(BaseSettings):
    """File upload configuration settings."""

    # File size limits (in bytes)
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    max_chunk_size: int = Field(default=5 * 1024 * 1024, env="MAX_CHUNK_SIZE")  # 5MB

    # Supported file types
    allowed_mime_types: List[str] = Field(
        default=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
            "application/vnd.oasis.opendocument.text",  # ODT
            "application/rtf",  # RTF
            "image/png",
            "image/jpeg",
            "image/tiff",
            "application/zip"
        ],
        env="ALLOWED_MIME_TYPES"
    )

    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".pptx", ".odt", ".rtf", ".png", ".jpg", ".jpeg", ".tiff", ".zip"],
        env="ALLOWED_EXTENSIONS"
    )

    # Upload quotas per tenant (in bytes)
    tenant_upload_quota: int = Field(default=10 * 1024 * 1024 * 1024, env="TENANT_UPLOAD_QUOTA")  # 10GB

    # Virus scanning
    virus_scan_enabled: bool = Field(default=True, env="VIRUS_SCAN_ENABLED")
    virus_scan_timeout: int = Field(default=30, env="VIRUS_SCAN_TIMEOUT")  # seconds
    clamav_host: str = Field(default="localhost", env="CLAMAV_HOST")
    clamav_port: int = Field(default=3310, env="CLAMAV_PORT")
    clamav_socket_path: Optional[str] = Field(default=None, env="CLAMAV_SOCKET_PATH")

    # Content validation
    content_validation_enabled: bool = Field(default=True, env="CONTENT_VALIDATION_ENABLED")
    magic_bytes_validation: bool = Field(default=True, env="MAGIC_BYTES_VALIDATION")
    structure_validation_enabled: bool = Field(default=True, env="STRUCTURE_VALIDATION_ENABLED")

    # Security and quarantine
    quarantine_enabled: bool = Field(default=True, env="QUARANTINE_ENABLED")
    quarantine_retention_days: int = Field(default=30, env="QUARANTINE_RETENTION_DAYS")
    security_audit_enabled: bool = Field(default=True, env="SECURITY_AUDIT_ENABLED")
    security_violation_alerts: bool = Field(default=True, env="SECURITY_VIOLATION_ALERTS")

    # Upload session settings
    upload_session_timeout: int = Field(default=3600, env="UPLOAD_SESSION_TIMEOUT")  # 1 hour
    upload_cleanup_interval: int = Field(default=300, env="UPLOAD_CLEANUP_INTERVAL")  # 5 minutes

    # Document versioning settings
    enable_document_versioning: bool = Field(default=True, env="ENABLE_DOCUMENT_VERSIONING")
    max_document_versions: int = Field(default=10, env="MAX_DOCUMENT_VERSIONS")

    # Progress tracking and WebSocket settings
    enable_progress_tracking: bool = Field(default=True, env="ENABLE_PROGRESS_TRACKING")
    websocket_enabled: bool = Field(default=True, env="WEBSOCKET_ENABLED")
    websocket_ping_interval: int = Field(default=30, env="WEBSOCKET_PING_INTERVAL")  # seconds
    websocket_connection_timeout: int = Field(default=300, env="WEBSOCKET_CONNECTION_TIMEOUT")  # 5 minutes

    # Chunked upload settings
    enable_chunked_uploads: bool = Field(default=True, env="ENABLE_CHUNKED_UPLOADS")
    chunked_upload_threshold: int = Field(default=5 * 1024 * 1024, env="CHUNKED_UPLOAD_THRESHOLD")  # 5MB
    chunk_storage_cleanup_interval: int = Field(default=600, env="CHUNK_STORAGE_CLEANUP_INTERVAL")  # 10 minutes
    chunk_storage_path: str = Field(default="./temp/chunks", env="CHUNK_STORAGE_PATH")  # Path for temporary chunk storage

    # Upload resumption settings
    enable_upload_resumption: bool = Field(default=True, env="ENABLE_UPLOAD_RESUMPTION")
    resumption_token_expiry: int = Field(default=86400, env="RESUMPTION_TOKEN_EXPIRY")  # 24 hours

    # Batch upload settings
    max_batch_size: int = Field(default=10, env="MAX_BATCH_SIZE")  # Maximum files per batch upload


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # Development features
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    enable_redoc: bool = Field(default=True, env="ENABLE_REDOC")
    enable_openapi: bool = Field(default=True, env="ENABLE_OPENAPI")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    vector_db: VectorDatabaseSettings = VectorDatabaseSettings()
    redis: RedisSettings = RedisSettings()
    storage: ObjectStorageSettings = ObjectStorageSettings()
    ai: AISettings = AISettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    security: SecuritySettings = SecuritySettings()
    upload: UploadSettings = UploadSettings()
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Allow extra fields from .env file
    }
        
    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level value."""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
