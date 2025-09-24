"""
Exception handlers for the Agentic RAG API.

This module defines custom exceptions and their handlers for consistent
error responses across the API.
"""

import json
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import structlog

from .models.responses import ErrorResponse, ValidationErrorResponse, ValidationErrorDetail


logger = structlog.get_logger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def safe_json_response(content: Dict[str, Any], status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> JSONResponse:
    """Create a JSONResponse with safe datetime serialization."""
    json_content = json.dumps(content, cls=DateTimeEncoder)
    return JSONResponse(
        content=json.loads(json_content),
        status_code=status_code,
        headers=headers
    )


class APIException(Exception):
    """Base exception for API errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class AuthenticationError(APIException):
    """Authentication failed."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class AuthorizationError(APIException):
    """Authorization failed."""
    
    def __init__(self, message: str = "Access denied", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR", 
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class TenantError(APIException):
    """Tenant-related error."""
    
    def __init__(self, message: str = "Tenant error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TENANT_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class ResourceNotFoundError(APIException):
    """Resource not found."""
    
    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class ConflictError(APIException):
    """Resource conflict."""
    
    def __init__(self, message: str = "Resource conflict", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFLICT_ERROR",
            status_code=status.HTTP_409_CONFLICT,
            details=details
        )


class RateLimitError(APIException):
    """Rate limit exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup exception handlers for the FastAPI application."""
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
        """Handle custom API exceptions."""
        request_id = getattr(request.state, 'request_id', None)
        
        logger.error(
            "API exception occurred",
            error_code=exc.error_code,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
            request_id=request_id,
            path=request.url.path,
            method=request.method
        )
        
        error_response = ErrorResponse(
            message=exc.message,
            error_code=exc.error_code,
            error_details=exc.details,
            request_id=request_id
        )
        
        return safe_json_response(
            content=error_response.dict(),
            status_code=exc.status_code
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        request_id = getattr(request.state, 'request_id', None)
        
        logger.warning(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            request_id=request_id,
            path=request.url.path,
            method=request.method
        )
        
        error_response = ErrorResponse(
            message=exc.detail,
            error_code="HTTP_ERROR",
            request_id=request_id
        )
        
        return safe_json_response(
            content=error_response.dict(),
            status_code=exc.status_code
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle request validation errors."""
        request_id = getattr(request.state, 'request_id', None)
        
        # Convert Pydantic validation errors to our format
        validation_errors = []
        for error in exc.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            validation_errors.append(
                ValidationErrorDetail(
                    field=field_path,
                    message=error["msg"],
                    value=error.get("input")
                )
            )
        
        logger.warning(
            "Request validation failed",
            validation_errors=[err.dict() for err in validation_errors],
            request_id=request_id,
            path=request.url.path,
            method=request.method
        )
        
        error_response = ValidationErrorResponse(
            message="Request validation failed",
            validation_errors=validation_errors,
            request_id=request_id
        )
        
        return safe_json_response(
            content=error_response.dict(),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, 'request_id', None)
        
        logger.error(
            "Unexpected exception occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            traceback=traceback.format_exc()
        )
        
        error_response = ErrorResponse(
            message="An unexpected error occurred",
            error_code="INTERNAL_SERVER_ERROR",
            request_id=request_id
        )
        
        return safe_json_response(
            content=error_response.dict(),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
