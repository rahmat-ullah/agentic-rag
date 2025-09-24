"""
OpenAPI documentation configuration and examples.

This module contains OpenAPI schema customizations, examples,
and documentation enhancements for the API.
"""

from typing import Dict, Any, List
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI


# OpenAPI tags metadata
OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Health check endpoints for monitoring system status and dependencies.",
    },
    {
        "name": "Authentication",
        "description": """
        Authentication endpoints for user login, token refresh, and session management.
        
        **Authentication Flow:**
        1. Login with email/password to get access and refresh tokens
        2. Use access token in Authorization header: `Bearer <token>`
        3. Refresh access token using refresh token when it expires
        4. Logout to invalidate tokens (optional)
        """,
    },
    {
        "name": "Users",
        "description": """
        User management endpoints for retrieving user information and managing user accounts.
        
        **Permissions:**
        - **Viewer**: Can view own user information
        - **Analyst**: Can view own user information  
        - **Admin**: Can view and manage all users in tenant
        """,
    },
    {
        "name": "Demo & Testing",
        "description": """
        Demonstration endpoints for testing authentication, authorization, and tenant isolation.
        
        **Available Endpoints:**
        - Public endpoints (no authentication required)
        - Role-based endpoints (viewer/analyst/admin)
        - Permission-based endpoints (specific permissions)
        - Tenant context endpoints (multi-tenant features)
        
        These endpoints are useful for testing API functionality and understanding
        the security model.
        """,
    },
    {
        "name": "Documents",
        "description": """
        Document management endpoints for uploading, processing, and retrieving documents.
        
        **Features:**
        - Document upload with metadata extraction
        - Document processing and chunking
        - Full-text and semantic search
        - Document versioning and history
        - Bulk operations
        
        **Coming in Sprint 2**
        """,
    },
    {
        "name": "Knowledge Base",
        "description": """
        Knowledge base management for organizing documents and managing retrieval contexts.
        
        **Features:**
        - Knowledge base creation and management
        - Document organization and categorization
        - Retrieval configuration and tuning
        - Analytics and insights
        
        **Coming in Sprint 3**
        """,
    },
    {
        "name": "Search & Retrieval",
        "description": """
        Advanced search and retrieval endpoints using three-hop retrieval strategy.
        
        **Features:**
        - Semantic search with vector embeddings
        - Three-hop retrieval for complex queries
        - Hybrid search (text + semantic)
        - Search result ranking and filtering
        - Query expansion and refinement
        
        **Coming in Sprint 4**
        """,
    },
    {
        "name": "Agents",
        "description": """
        AI agent orchestration endpoints for complex task automation.
        
        **Features:**
        - Agent creation and configuration
        - Task orchestration and workflow management
        - Agent collaboration and communication
        - Performance monitoring and optimization
        
        **Coming in Sprint 5**
        """,
    },
]


# Common response examples
RESPONSE_EXAMPLES = {
    "success_example": {
        "success": True,
        "message": "Operation completed successfully",
        "data": {"id": "123e4567-e89b-12d3-a456-426614174000"},
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_123456789"
    },
    "error_example": {
        "success": False,
        "message": "Authentication failed",
        "error_code": "INVALID_TOKEN",
        "error_details": {
            "token_expired": True,
            "expires_at": "2024-01-15T09:30:00Z"
        },
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_123456789"
    },
    "validation_error_example": {
        "success": False,
        "message": "Request validation failed",
        "error_code": "VALIDATION_ERROR",
        "validation_errors": [
            {
                "field": "email",
                "message": "Invalid email format",
                "value": "not-an-email"
            }
        ],
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_123456789"
    },
    "unauthorized_example": {
        "success": False,
        "message": "Authorization token required",
        "error_code": "MISSING_TOKEN",
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_123456789"
    },
    "forbidden_example": {
        "success": False,
        "message": "Insufficient permissions for this operation",
        "error_code": "INSUFFICIENT_PERMISSIONS",
        "error_details": {
            "required_permission": "document:write",
            "user_role": "viewer"
        },
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_123456789"
    },
    "not_found_example": {
        "success": False,
        "message": "Resource not found",
        "error_code": "RESOURCE_NOT_FOUND",
        "error_details": {
            "resource_type": "document",
            "resource_id": "123e4567-e89b-12d3-a456-426614174000"
        },
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_123456789"
    },
    "rate_limit_example": {
        "success": False,
        "message": "Rate limit exceeded",
        "error_code": "RATE_LIMIT_EXCEEDED",
        "error_details": {
            "limit": 100,
            "window_seconds": 3600,
            "reset_at": "2024-01-15T11:30:00Z"
        },
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_123456789"
    }
}


# Security scheme definitions
SECURITY_SCHEMES = {
    "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT access token obtained from /auth/login endpoint"
    },
    "TenantHeader": {
        "type": "apiKey",
        "in": "header",
        "name": "X-Tenant-ID",
        "description": "Tenant ID for multi-tenant operations (optional, defaults to user's tenant)"
    }
}


def customize_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Customize OpenAPI schema with enhanced documentation.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Customized OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=app.servers,
        tags=OPENAPI_TAGS
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = SECURITY_SCHEMES
    
    # Add common response examples
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    if "examples" not in openapi_schema["components"]:
        openapi_schema["components"]["examples"] = {}
    
    openapi_schema["components"]["examples"].update(RESPONSE_EXAMPLES)
    
    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    # Customize info section
    openapi_schema["info"]["x-logo"] = {
        "url": "https://your-domain.com/logo.png",
        "altText": "Agentic RAG System"
    }
    
    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Full Documentation",
        "url": "https://docs.your-domain.com"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Common HTTP status code responses for reuse
COMMON_RESPONSES = {
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": RESPONSE_EXAMPLES["validation_error_example"]
            }
        }
    },
    401: {
        "description": "Unauthorized",
        "content": {
            "application/json": {
                "example": RESPONSE_EXAMPLES["unauthorized_example"]
            }
        }
    },
    403: {
        "description": "Forbidden",
        "content": {
            "application/json": {
                "example": RESPONSE_EXAMPLES["forbidden_example"]
            }
        }
    },
    404: {
        "description": "Not Found",
        "content": {
            "application/json": {
                "example": RESPONSE_EXAMPLES["not_found_example"]
            }
        }
    },
    429: {
        "description": "Rate Limit Exceeded",
        "content": {
            "application/json": {
                "example": RESPONSE_EXAMPLES["rate_limit_example"]
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "message": "Internal server error occurred",
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "request_id": "req_123456789"
                }
            }
        }
    }
}
