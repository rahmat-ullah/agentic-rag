"""
User models for the Agentic RAG API.

This module defines Pydantic models for user management.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """User creation model."""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    role: str = Field(..., description="User role (admin/analyst/viewer)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "newuser@example.com",
                "password": "securepassword123",
                "role": "analyst"
            }
        }


class UserUpdate(BaseModel):
    """User update model."""
    
    email: Optional[EmailStr] = Field(None, description="User email address")
    role: Optional[str] = Field(None, description="User role")
    is_active: Optional[bool] = Field(None, description="Whether user is active")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "updated@example.com",
                "role": "admin",
                "is_active": True
            }
        }


class UserResponse(BaseModel):
    """User response model."""
    
    id: UUID = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email")
    role: str = Field(..., description="User role")
    tenant_id: UUID = Field(..., description="Tenant ID")
    is_active: bool = Field(..., description="Whether user is active")
    created_at: datetime = Field(..., description="User creation timestamp")
    updated_at: datetime = Field(..., description="User last update timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "user@example.com",
                "role": "analyst",
                "tenant_id": "123e4567-e89b-12d3-a456-426614174001",
                "is_active": True,
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z"
            }
        }


class UserListResponse(BaseModel):
    """User list response model."""
    
    users: list[UserResponse] = Field(..., description="List of users")
    total: int = Field(..., description="Total number of users")
    
    class Config:
        json_schema_extra = {
            "example": {
                "users": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "email": "user1@example.com",
                        "role": "analyst",
                        "tenant_id": "123e4567-e89b-12d3-a456-426614174001",
                        "is_active": True,
                        "created_at": "2024-01-01T12:00:00Z",
                        "updated_at": "2024-01-01T12:00:00Z"
                    }
                ],
                "total": 1
            }
        }
