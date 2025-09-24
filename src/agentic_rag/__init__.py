"""
Agentic RAG System

A production-grade, agentic Retrieval-Augmented Generation (RAG) system that:
- Continuously learns from users (feedback, edits, link confirmations)
- Uses IBM Granite-Docling-258M to parse/convert documents
- Uses OpenAI embedding models for vectorization
- Stores vectors in ChromaDB and system truth in PostgreSQL
- Implements contextual retrieval with three-hop search pattern
"""

__version__ = "0.1.0"
__author__ = "Agentic RAG Team"
__email__ = "team@agentic-rag.com"

# Package metadata
__title__ = "agentic-rag"
__description__ = "Production-grade Agentic RAG system for procurement documents"
__url__ = "https://github.com/agentic-rag/agentic-rag"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Agentic RAG Team"
