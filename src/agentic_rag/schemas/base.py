"""
Base Pydantic schemas for the Agentic RAG API.

This module contains base response models and common schemas used across
the API endpoints.
"""

from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model for all API responses."""
    
    success: bool = Field(True, description="Indicates if the request was successful")
    message: Optional[str] = Field(None, description="Optional message for additional context")
    error_code: Optional[str] = Field(None, description="Error code if success is False")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully"
            }
        }


class ErrorResponse(BaseResponse):
    """Error response model."""
    
    success: bool = Field(False, description="Always False for error responses")
    message: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Specific error code")

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "message": "Validation error occurred",
                "error_code": "VALIDATION_ERROR",
                "error_details": {
                    "field": "email",
                    "issue": "Invalid email format"
                }
            }
        }


T = TypeVar('T')


class PaginatedResponse(BaseResponse, Generic[T]):
    """Base paginated response model."""
    
    items: List[T] = Field(..., description="List of items for the current page")
    total_count: int = Field(..., description="Total number of items across all pages")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "items": [],
                "total_count": 150,
                "page": 1,
                "page_size": 20,
                "total_pages": 8,
                "has_next": True,
                "has_previous": False
            }
        }


class PaginationParams(BaseModel):
    """Common pagination parameters."""
    
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=100, description="Number of items per page")
    
    class Config:
        schema_extra = {
            "example": {
                "page": 1,
                "page_size": 20
            }
        }


class SortParams(BaseModel):
    """Common sorting parameters."""
    
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", regex="^(asc|desc)$", description="Sort order: asc or desc")
    
    class Config:
        schema_extra = {
            "example": {
                "sort_by": "created_at",
                "sort_order": "desc"
            }
        }


class FilterParams(BaseModel):
    """Common filtering parameters."""
    
    search: Optional[str] = Field(None, description="Search term for text fields")
    date_from: Optional[str] = Field(None, description="Filter from date (ISO format)")
    date_to: Optional[str] = Field(None, description="Filter to date (ISO format)")
    
    class Config:
        schema_extra = {
            "example": {
                "search": "pricing",
                "date_from": "2024-01-01T00:00:00Z",
                "date_to": "2024-12-31T23:59:59Z"
            }
        }
