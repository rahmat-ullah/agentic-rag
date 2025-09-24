"""
Document Chunking Services

This package provides comprehensive document chunking capabilities including:
- Basic text chunking with configurable parameters
- Section-aware chunking that respects document structure
- Table detection and special handling
- Chunk deduplication
- Integrated chunking pipeline
"""

from .basic_chunker import (
    BasicTextChunker,
    ChunkingConfig,
    ChunkMetadata,
    TextChunk,
    SentenceDetector,
    get_basic_chunker
)
from .section_aware_chunker import (
    SectionAwareChunker,
    SectionAwareChunk,
    SectionMetadata,
    SectionInfo,
    DocumentSectionAnalyzer,
    get_section_aware_chunker
)
from .table_aware_chunker import (
    TableAwareChunker,
    TableAwareChunk,
    TableMetadata,
    TableInfo,
    TableDetector,
    get_table_aware_chunker
)
from .deduplication_chunker import (
    DeduplicationChunker,
    DeduplicatedChunk,
    DeduplicationMetadata,
    ContentNormalizer,
    ChunkDeduplicator,
    get_deduplication_chunker
)
from .pipeline import (
    ChunkingPipeline,
    ChunkingPipelineConfig,
    ChunkingResult,
    get_chunking_pipeline,
    process_document_chunks_async,
    process_document_chunks_sync
)

__all__ = [
    "BasicTextChunker",
    "ChunkingConfig",
    "ChunkMetadata",
    "TextChunk",
    "SentenceDetector",
    "get_basic_chunker",
    "SectionAwareChunker",
    "SectionAwareChunk",
    "SectionMetadata",
    "SectionInfo",
    "DocumentSectionAnalyzer",
    "get_section_aware_chunker",
    "TableAwareChunker",
    "TableAwareChunk",
    "TableMetadata",
    "TableInfo",
    "TableDetector",
    "get_table_aware_chunker",
    "DeduplicationChunker",
    "DeduplicatedChunk",
    "DeduplicationMetadata",
    "ContentNormalizer",
    "ChunkDeduplicator",
    "get_deduplication_chunker",
    "ChunkingPipeline",
    "ChunkingPipelineConfig",
    "ChunkingResult",
    "get_chunking_pipeline",
    "process_document_chunks_async",
    "process_document_chunks_sync"
]
