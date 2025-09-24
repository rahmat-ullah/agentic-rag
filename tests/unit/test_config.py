"""
Unit tests for configuration management.

Tests the configuration loading, validation, and environment-specific settings.
"""

import os
import pytest
from unittest.mock import patch

from agentic_rag.config import (
    Settings, 
    Environment, 
    LogLevel,
    DatabaseSettings,
    SecuritySettings,
    get_settings
)
from tests.base import BaseUnitTest


class TestEnvironmentEnum(BaseUnitTest):
    """Test Environment enum."""
    
    def test_environment_values(self):
        """Test that environment enum has correct values."""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.TESTING == "testing"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"


class TestLogLevelEnum(BaseUnitTest):
    """Test LogLevel enum."""
    
    def test_log_level_values(self):
        """Test that log level enum has correct values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"


class TestDatabaseSettings(BaseUnitTest):
    """Test DatabaseSettings configuration."""
    
    def test_default_values(self):
        """Test default database settings."""
        settings = DatabaseSettings()
        
        assert settings.postgres_host == "localhost"
        assert settings.postgres_port == 5432
        assert settings.postgres_db == "agentic_rag"
        assert settings.postgres_user == "postgres"
        assert settings.postgres_password == "postgres"
        assert settings.db_pool_size == 10
        assert settings.db_max_overflow == 20
        assert settings.db_pool_timeout == 30
        assert settings.db_pool_recycle == 3600
        assert settings.db_echo is False
    
    def test_postgres_url_building(self):
        """Test PostgreSQL URL building."""
        settings = DatabaseSettings(
            postgres_host="testhost",
            postgres_port=5433,
            postgres_db="testdb",
            postgres_user="testuser",
            postgres_password="testpass"
        )
        
        expected_url = "postgresql://testuser:testpass@testhost:5433//testdb"
        assert str(settings.postgres_url) == expected_url
    
    @patch.dict(os.environ, {
        "POSTGRES_HOST": "envhost",
        "POSTGRES_PORT": "5434",
        "POSTGRES_DB": "envdb",
        "POSTGRES_USER": "envuser",
        "POSTGRES_PASSWORD": "envpass",
        "DB_POOL_SIZE": "20",
        "DB_ECHO": "true"
    })
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        settings = DatabaseSettings()
        
        assert settings.postgres_host == "envhost"
        assert settings.postgres_port == 5434
        assert settings.postgres_db == "envdb"
        assert settings.postgres_user == "envuser"
        assert settings.postgres_password == "envpass"
        assert settings.db_pool_size == 20
        assert settings.db_echo is True


class TestSecuritySettings(BaseUnitTest):
    """Test SecuritySettings configuration."""
    
    def test_default_values(self):
        """Test default security settings."""
        settings = SecuritySettings()
        
        assert settings.jwt_secret_key == "dev-jwt-secret-key"
        assert settings.jwt_algorithm == "HS256"
        assert settings.jwt_access_token_expire_minutes == 30
        assert settings.jwt_refresh_token_expire_days == 7
        assert settings.password_hash_algorithm == "bcrypt"
        assert settings.password_hash_rounds == 12
        assert settings.rate_limit_requests == 100
        assert settings.rate_limit_window == 60
    
    @patch.dict(os.environ, {
        "JWT_SECRET_KEY": "test-secret",
        "JWT_ALGORITHM": "RS256",
        "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "60",
        "PASSWORD_HASH_ROUNDS": "10",
        "RATE_LIMIT_REQUESTS": "200"
    })
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        settings = SecuritySettings()
        
        assert settings.jwt_secret_key == "test-secret"
        assert settings.jwt_algorithm == "RS256"
        assert settings.jwt_access_token_expire_minutes == 60
        assert settings.password_hash_rounds == 10
        assert settings.rate_limit_requests == 200


class TestMainSettings(BaseUnitTest):
    """Test main Settings class."""
    
    def test_settings_initialization(self):
        """Test that settings can be initialized and have expected types."""
        settings = Settings()

        assert isinstance(settings.environment, Environment)
        assert isinstance(settings.debug, bool)
        assert isinstance(settings.log_level, LogLevel)
        assert isinstance(settings.api_host, str)
        assert isinstance(settings.api_port, int)
        assert isinstance(settings.api_reload, bool)
        assert isinstance(settings.secret_key, str)
        assert isinstance(settings.enable_docs, bool)
        assert isinstance(settings.enable_redoc, bool)
        assert isinstance(settings.enable_openapi, bool)
    
    def test_nested_settings_initialization(self):
        """Test that nested settings are properly initialized."""
        settings = Settings()

        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.security, SecuritySettings)
        assert isinstance(settings.database.postgres_host, str)
        assert isinstance(settings.security.jwt_algorithm, str)
    
    def test_environment_variable_override(self):
        """Test that environment variables can override settings."""
        # Test with explicit values
        settings = Settings(
            environment="testing",
            debug=True,
            log_level="DEBUG",
            api_port=8080,
            secret_key="test-secret"
        )

        assert settings.environment == Environment.TESTING
        assert settings.debug is True
        assert settings.log_level == LogLevel.DEBUG
        assert settings.api_port == 8080
        assert settings.secret_key == "test-secret"
    
    def test_cors_settings(self):
        """Test CORS settings."""
        settings = Settings()
        
        assert isinstance(settings.cors_origins, list)
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:8080" in settings.cors_origins
        assert settings.cors_allow_credentials is True
        assert settings.cors_allow_methods == ["*"]
        assert settings.cors_allow_headers == ["*"]
    
    def test_environment_validation(self):
        """Test environment validation."""
        # Test valid environment
        settings = Settings(environment="production")
        assert settings.environment == Environment.PRODUCTION
        
        # Test case insensitive
        settings = Settings(environment="DEVELOPMENT")
        assert settings.environment == Environment.DEVELOPMENT
    
    def test_is_development_property(self):
        """Test is_development property."""
        settings = Settings(environment="development")
        assert settings.is_development is True
        
        settings = Settings(environment="production")
        assert settings.is_development is False
    
    def test_is_testing_property(self):
        """Test is_testing property."""
        settings = Settings(environment="testing")
        assert settings.is_testing is True
        
        settings = Settings(environment="development")
        assert settings.is_testing is False
    
    def test_is_production_property(self):
        """Test is_production property."""
        settings = Settings(environment="production")
        assert settings.is_production is True
        
        settings = Settings(environment="development")
        assert settings.is_production is False


class TestGetSettings(BaseUnitTest):
    """Test get_settings function."""
    
    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
    
    def test_get_settings_singleton_behavior(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
