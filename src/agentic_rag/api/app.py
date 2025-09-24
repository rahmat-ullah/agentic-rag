"""
FastAPI application factory for Agentic RAG System.

This module creates and configures the FastAPI application with all necessary
middleware, error handlers, and route configurations.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from agentic_rag.config import get_settings
from agentic_rag.adapters.database import get_database_adapter

from .middleware.auth import AuthenticationMiddleware
from .middleware.tenant import TenantContextMiddleware
from .middleware.security import SecurityHeadersMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .routes import health, auth, users, demo
from .models.openapi import customize_openapi, OPENAPI_TAGS
from .exceptions import setup_exception_handlers
from .models.responses import ErrorResponse


logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    
    # Startup
    logger.info("Starting Agentic RAG API", version=app.version)
    
    # Initialize database connection (graceful failure in development)
    try:
        db = get_database_adapter()
        if db.health_check():
            logger.info("Database connection established")
        else:
            logger.warning("Database health check failed - continuing in development mode")
    except Exception as e:
        if settings.is_production:
            logger.error("Database connection failed in production", error=str(e))
            raise RuntimeError("Database connection failed")
        else:
            logger.warning("Database connection failed - continuing in development mode", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agentic RAG API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="Agentic RAG System API",
        description="""
        ## Agentic Retrieval-Augmented Generation System API

        A comprehensive API for managing documents, knowledge bases, and AI-powered retrieval
        in a multi-tenant environment with role-based access control.

        ### Features

        - **Multi-tenant Architecture**: Complete tenant isolation with Row-Level Security
        - **Role-based Authorization**: Three-tier role system (viewer/analyst/admin)
        - **JWT Authentication**: Secure token-based authentication with refresh tokens
        - **Document Management**: Upload, process, and manage documents with metadata
        - **Vector Search**: ChromaDB integration for semantic document retrieval
        - **Three-hop Retrieval**: Advanced retrieval strategy for complex queries
        - **Agent Orchestration**: AI agent coordination for complex tasks

        ### Authentication

        Most endpoints require authentication via JWT tokens. Include the token in the
        `Authorization` header as `Bearer <token>`.

        ### Multi-tenant Context

        For tenant-specific operations, provide the tenant ID via:
        - `X-Tenant-ID` header (recommended)
        - `tenant_id` query parameter

        ### Rate Limiting

        API requests are rate-limited per user and tenant. See response headers for
        current limits and remaining quota.
        """,
        version="1.0.0",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_redoc else None,
        openapi_url="/openapi.json" if settings.enable_openapi else None,
        lifespan=lifespan,
        contact={
            "name": "Agentic RAG System",
            "url": "https://github.com/your-org/agentic-rag",
            "email": "support@your-org.com"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.agentic-rag.com",
                "description": "Production server"
            }
        ],
        openapi_tags=OPENAPI_TAGS
    )
    
    # Add middleware (order matters - last added is executed first)
    
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware)
    
    # Tenant context middleware
    app.add_middleware(TenantContextMiddleware)
    
    # Authentication middleware
    app.add_middleware(AuthenticationMiddleware)
    
    # CORS middleware with comprehensive configuration
    cors_origins = settings.cors_origins or []
    if settings.environment == "development":
        # Allow localhost for development
        cors_origins.extend([
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080"
        ])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=settings.cors_allow_headers or [
            "Authorization",
            "Content-Type",
            "X-Tenant-ID",
            "X-Request-ID",
            "Accept",
            "Origin",
            "User-Agent",
            "DNT",
            "Cache-Control",
            "X-Mx-ReqToken",
            "Keep-Alive",
            "X-Requested-With",
            "If-Modified-Since"
        ],
        expose_headers=[
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Request-ID",
            "X-Tenant-ID",
            "X-Tenant-Name"
        ],
        max_age=86400,  # 24 hours
    )

    # Trusted host middleware - skip for development
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure properly in production
        )
    
    # Request ID and logging middleware
    @app.middleware("http")
    async def add_request_id_and_logging(request: Request, call_next):
        """Add request ID and structured logging."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Add request context to logger
        logger_ctx = logger.bind(
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
        )
        
        logger_ctx.info("Request started")
        
        try:
            response = await call_next(request)
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            process_time = time.time() - start_time
            logger_ctx.info(
                "Request completed",
                status_code=response.status_code,
                process_time=process_time,
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger_ctx.error(
                "Request failed",
                error=str(e),
                process_time=process_time,
                exc_info=True,
            )
            raise
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(users.router, prefix="/users", tags=["Users"])
    app.include_router(demo.router, prefix="/demo", tags=["Demo & Testing"])

    # Import and include upload router
    from agentic_rag.api.routes import upload
    app.include_router(upload.router, prefix="/api/v1/upload", tags=["Upload"])

    # Include upload router as /ingest endpoint (Sprint 2 specification)
    app.include_router(upload.router, prefix="/ingest", tags=["Ingest"])

    # Import and include WebSocket router
    from agentic_rag.api.routes import websocket
    app.include_router(websocket.router, prefix="/api/v1", tags=["WebSocket"])

    # Import and include documents router
    from agentic_rag.api.routes import documents
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])

    # Import and include search router
    from agentic_rag.api.routes import search
    app.include_router(search.router, prefix="/api/v1", tags=["Search"])

    # Customize OpenAPI schema
    app.openapi = lambda: customize_openapi(app)

    return app


def get_app() -> FastAPI:
    """Get configured FastAPI application instance."""
    return create_app()
