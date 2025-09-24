"""
API package for Agentic RAG System.

This package contains the FastAPI application and all API-related components
including authentication, authorization, and endpoint definitions.
"""

from .app import create_app

__all__ = ["create_app"]
