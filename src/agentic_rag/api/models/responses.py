"""
Base response models for the Agentic RAG API.

This module defines standard response formats used across all API endpoints
for consistent error handling and data presentation.
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field


T = TypeVar('T')


class BaseResponse(BaseModel):
    """Base response model for all API responses."""
    
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Human-readable message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class SuccessResponse(BaseResponse, Generic[T]):
    """Success response with data payload."""
    
    success: bool = Field(True, description="Always true for success responses")
    data: T = Field(..., description="Response data payload")


class ErrorResponse(BaseResponse):
    """Error response with error details."""
    
    success: bool = Field(False, description="Always false for error responses")
    error_code: str = Field(..., description="Machine-readable error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Validation error occurred",
                "error_code": "VALIDATION_ERROR",
                "error_details": {
                    "field": "email",
                    "issue": "Invalid email format"
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456789"
            }
        }


class PaginationMeta(BaseModel):
    """Pagination metadata."""
    
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Number of items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class PaginatedResponse(SuccessResponse[List[T]]):
    """Paginated response with metadata."""
    
    pagination: PaginationMeta = Field(..., description="Pagination metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Data retrieved successfully",
                "data": [
                    {"id": "123", "name": "Example Item"}
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 20,
                    "total_items": 100,
                    "total_pages": 5,
                    "has_next": True,
                    "has_previous": False
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456789"
            }
        }


class HealthStatus(BaseModel):
    """Health check status for a service component."""
    
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status (healthy/unhealthy/degraded)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional status details")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")


class HealthResponse(SuccessResponse[Dict[str, Any]]):
    """Health check response."""
    
    data: Dict[str, Any] = Field(
        ..., 
        description="Health check data including overall status and component details"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Health check completed",
                "data": {
                    "status": "healthy",
                    "version": "1.0.0",
                    "uptime_seconds": 3600,
                    "components": [
                        {
                            "name": "database",
                            "status": "healthy",
                            "response_time_ms": 5.2
                        },
                        {
                            "name": "vector_store",
                            "status": "healthy", 
                            "response_time_ms": 12.8
                        }
                    ]
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456789"
            }
        }


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""
    
    field: str = Field(..., description="Field name that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(None, description="Invalid value that was provided")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field-specific details."""
    
    error_code: str = Field("VALIDATION_ERROR", description="Always VALIDATION_ERROR")
    validation_errors: List[ValidationErrorDetail] = Field(
        ..., 
        description="List of validation errors"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Request validation failed",
                "error_code": "VALIDATION_ERROR",
                "validation_errors": [
                    {
                        "field": "email",
                        "message": "Invalid email format",
                        "value": "invalid-email"
                    },
                    {
                        "field": "password",
                        "message": "Password must be at least 8 characters",
                        "value": "short"
                    }
                ],
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456789"
            }
        }
