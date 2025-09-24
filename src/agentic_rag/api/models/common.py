"""
Common API models and schemas.

This module contains shared Pydantic models used across multiple API endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional, Any, Dict, List
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class APIStatus(str, Enum):
    """API response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class ErrorCode(str, Enum):
    """Standard error codes for API responses."""
    # Authentication errors
    MISSING_TOKEN = "MISSING_TOKEN"
    INVALID_TOKEN = "INVALID_TOKEN"
    EXPIRED_TOKEN = "EXPIRED_TOKEN"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    
    # Authorization errors
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    ROLE_REQUIRED = "ROLE_REQUIRED"
    TENANT_ISOLATION_VIOLATION = "TENANT_ISOLATION_VIOLATION"
    
    # Tenant errors
    MISSING_TENANT_CONTEXT = "MISSING_TENANT_CONTEXT"
    INVALID_TENANT = "INVALID_TENANT"
    TENANT_NOT_FOUND = "TENANT_NOT_FOUND"
    
    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    
    # Resource errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    
    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Server errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"


class PaginationParams(BaseModel):
    """Standard pagination parameters."""
    
    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-based)",
        example=1
    )
    
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page (max 100)",
        example=20
    )
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size


class PaginationMeta(BaseModel):
    """Pagination metadata for responses."""
    
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    total_items: int = Field(description="Total number of items")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there are more pages")
    has_previous: bool = Field(description="Whether there are previous pages")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page": 1,
                "page_size": 20,
                "total_items": 150,
                "total_pages": 8,
                "has_next": True,
                "has_previous": False
            }
        }
    )


class SortParams(BaseModel):
    """Standard sorting parameters."""
    
    sort_by: str = Field(
        default="created_at",
        description="Field to sort by",
        example="created_at"
    )
    
    sort_order: str = Field(
        default="desc",
        pattern="^(asc|desc)$",
        description="Sort order: 'asc' or 'desc'",
        example="desc"
    )


class FilterParams(BaseModel):
    """Standard filtering parameters."""
    
    search: Optional[str] = Field(
        default=None,
        description="Search term for text fields",
        example="procurement"
    )
    
    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter items created after this date",
        example="2024-01-01T00:00:00Z"
    )
    
    created_before: Optional[datetime] = Field(
        default=None,
        description="Filter items created before this date",
        example="2024-12-31T23:59:59Z"
    )
    
    is_active: Optional[bool] = Field(
        default=None,
        description="Filter by active status",
        example=True
    )


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheck(BaseModel):
    """Health check response model."""
    
    status: HealthStatus = Field(description="Overall health status")
    timestamp: datetime = Field(description="Health check timestamp")
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="Server uptime in seconds")
    
    # Component health
    database: HealthStatus = Field(description="Database connection status")
    redis: Optional[HealthStatus] = Field(default=None, description="Redis connection status")
    external_apis: Optional[Dict[str, HealthStatus]] = Field(
        default=None,
        description="External API health status"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "database": "healthy",
                "redis": "healthy",
                "external_apis": {
                    "openai": "healthy",
                    "chromadb": "healthy"
                }
            }
        }
    )


class APIMetadata(BaseModel):
    """API metadata for responses."""
    
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier for tracing"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Request processing time in milliseconds"
    )
    
    rate_limit_remaining: Optional[int] = Field(
        default=None,
        description="Remaining rate limit quota"
    )
    
    rate_limit_reset: Optional[datetime] = Field(
        default=None,
        description="Rate limit reset time"
    )


class TenantInfo(BaseModel):
    """Tenant information model."""
    
    id: uuid.UUID = Field(description="Tenant unique identifier")
    name: str = Field(description="Tenant name")
    is_active: bool = Field(description="Whether tenant is active")
    created_at: datetime = Field(description="Tenant creation timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Acme Corporation",
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
    )


class UserInfo(BaseModel):
    """User information model."""
    
    id: uuid.UUID = Field(description="User unique identifier")
    email: str = Field(description="User email address")
    role: str = Field(description="User role (viewer/analyst/admin)")
    tenant_id: uuid.UUID = Field(description="User's tenant ID")
    is_active: bool = Field(description="Whether user is active")
    created_at: datetime = Field(description="User creation timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "email": "user@example.com",
                "role": "analyst",
                "tenant_id": "123e4567-e89b-12d3-a456-426614174000",
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
    )


class ValidationErrorDetail(BaseModel):
    """Validation error detail model."""
    
    field: str = Field(description="Field that failed validation")
    message: str = Field(description="Validation error message")
    value: Any = Field(description="Invalid value that was provided")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "field": "email",
                "message": "Invalid email format",
                "value": "not-an-email"
            }
        }
    )
