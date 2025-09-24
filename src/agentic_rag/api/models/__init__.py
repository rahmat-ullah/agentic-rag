"""
API models package for Agentic RAG System.

This package contains Pydantic models for request/response validation,
authentication, and data transfer objects.
"""

from .responses import (
    BaseResponse,
    ErrorResponse,
    SuccessResponse,
    PaginatedResponse,
    HealthResponse,
)

from .auth import (
    LoginRequest,
    LoginResponse,
    TokenResponse,
    RefreshTokenRequest,
    UserInfo,
)

from .users import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserListResponse,
)

from .search import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchFilters,
    SearchOptions,
    SearchPagination,
    SearchStatistics,
    SearchSuggestionsRequest,
    SearchSuggestionsResponse,
)

__all__ = [
    # Response models
    "BaseResponse",
    "ErrorResponse",
    "SuccessResponse",
    "PaginatedResponse",
    "HealthResponse",

    # Auth models
    "LoginRequest",
    "LoginResponse",
    "TokenResponse",
    "RefreshTokenRequest",
    "UserInfo",

    # User models
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserListResponse",

    # Search models
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "SearchFilters",
    "SearchOptions",
    "SearchPagination",
    "SearchStatistics",
    "SearchSuggestionsRequest",
    "SearchSuggestionsResponse",
]
