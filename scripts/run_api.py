#!/usr/bin/env python3
"""
API server runner for Agentic RAG System.

This script starts the FastAPI application with proper configuration.
"""

import sys
import uvicorn
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_rag.config import get_settings
from agentic_rag.api import create_app


def main():
    """Run the API server."""
    settings = get_settings()
    
    # Create the FastAPI app
    app = create_app()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload and settings.debug,
        log_level=settings.log_level.value.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
