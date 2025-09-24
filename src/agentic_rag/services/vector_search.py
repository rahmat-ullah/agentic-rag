"""
Vector Search Service

This module provides enhanced vector similarity search capabilities using ChromaDB
with configurable parameters, collection selection, and result processing.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.vector_operations import (
    VectorOperationsService, VectorSearchOptions, get_vector_operations_service
)
from agentic_rag.services.vector_store import VectorSearchResult, VectorMetadata
from agentic_rag.models.database import DocumentKind

logger = structlog.get_logger(__name__)


class SearchStrategy(str, Enum):
    """Vector search strategies."""
    SIMILARITY = "similarity"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class CollectionStrategy(str, Enum):
    """Collection selection strategies."""
    AUTO = "auto"
    SINGLE = "single"
    MULTI = "multi"
    ALL = "all"


@dataclass
class SearchConfiguration:
    """Configuration for vector search operations."""
    strategy: SearchStrategy = SearchStrategy.SIMILARITY
    collection_strategy: CollectionStrategy = CollectionStrategy.AUTO
    similarity_threshold: float = 0.7
    max_results: int = 100
    include_metadata: bool = True
    include_documents: bool = True
    boost_recent: bool = False
    boost_factor: float = 1.1
    diversity_threshold: float = 0.9


class VectorSearchRequest(BaseModel):
    """Request model for vector search operations."""
    
    query_embedding: List[float] = Field(..., description="Query embedding vector")
    tenant_id: str = Field(..., description="Tenant identifier")
    document_types: Optional[List[DocumentKind]] = Field(None, description="Document types to search")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    search_config: Optional[SearchConfiguration] = Field(None, description="Search configuration")
    max_results: int = Field(50, ge=1, le=100, description="Maximum results to return")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum score threshold")


class VectorSearchResponse(BaseModel):
    """Response model for vector search operations."""
    
    results: List[VectorSearchResult] = Field(..., description="Search results")
    total_searched: int = Field(..., description="Total vectors searched")
    search_time_ms: int = Field(..., description="Search time in milliseconds")
    collections_searched: List[str] = Field(..., description="Collections that were searched")
    strategy_used: SearchStrategy = Field(..., description="Search strategy used")
    filters_applied: Dict[str, Any] = Field(..., description="Filters that were applied")


class VectorSearchService:
    """Enhanced vector search service with advanced capabilities."""
    
    def __init__(self):
        self._vector_ops: Optional[VectorOperationsService] = None
        self._default_config = SearchConfiguration()
        
        # Collection mapping for document types
        self._collection_mapping = {
            DocumentKind.RFQ: "rfq_chunks",
            DocumentKind.OFFER: "offer_chunks",
            DocumentKind.CONTRACT: "contract_chunks",
            DocumentKind.SPECIFICATION: "specification_chunks",
            DocumentKind.REPORT: "report_chunks",
            DocumentKind.OTHER: "other_chunks"
        }
        
        # Search statistics
        self._stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "average_search_time_ms": 0.0,
            "total_results_returned": 0
        }
        
        logger.info("Vector search service initialized")
    
    async def initialize(self) -> None:
        """Initialize the vector search service."""
        self._vector_ops = await get_vector_operations_service()
        logger.info("Vector search service connected to vector operations")
    
    async def search_vectors(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """
        Perform enhanced vector similarity search.
        
        Args:
            request: Vector search request with parameters
            
        Returns:
            VectorSearchResponse with results and metadata
        """
        start_time = time.time()
        
        if not self._vector_ops:
            await self.initialize()
        
        # Use provided config or default
        config = request.search_config or self._default_config
        
        logger.info(
            f"Starting vector search",
            tenant_id=request.tenant_id,
            document_types=request.document_types,
            max_results=request.max_results,
            strategy=config.strategy.value
        )
        
        try:
            self._stats["total_searches"] += 1
            
            # Determine collections to search
            collections_to_search = self._determine_collections(
                request.document_types, 
                config.collection_strategy
            )
            
            # Build search filters
            search_filters = self._build_search_filters(
                request.tenant_id,
                request.metadata_filters
            )
            
            # Perform search across collections
            all_results = []
            total_searched = 0
            
            for collection_name in collections_to_search:
                # Map collection name to document kind
                document_kind = self._get_document_kind_for_collection(collection_name)
                
                # Configure search options
                search_options = VectorSearchOptions(
                    n_results=request.max_results,
                    score_threshold=request.score_threshold or config.similarity_threshold,
                    where_filter=search_filters,
                    include_metadata=config.include_metadata,
                    include_documents=config.include_documents
                )
                
                # Perform search
                collection_results = await self._vector_ops.search_vectors(
                    query_embedding=request.query_embedding,
                    document_kind=document_kind,
                    tenant_id=request.tenant_id,
                    options=search_options
                )
                
                all_results.extend(collection_results)
                total_searched += len(collection_results)
            
            # Post-process results
            processed_results = self._post_process_results(
                all_results, 
                config, 
                request.max_results
            )
            
            search_time_ms = int((time.time() - start_time) * 1000)
            
            # Update statistics
            self._stats["successful_searches"] += 1
            self._stats["total_results_returned"] += len(processed_results)
            self._update_average_search_time(search_time_ms)
            
            response = VectorSearchResponse(
                results=processed_results,
                total_searched=total_searched,
                search_time_ms=search_time_ms,
                collections_searched=collections_to_search,
                strategy_used=config.strategy,
                filters_applied=search_filters
            )
            
            logger.info(
                f"Vector search completed successfully",
                results_count=len(processed_results),
                total_searched=total_searched,
                search_time_ms=search_time_ms,
                collections_searched=len(collections_to_search)
            )
            
            return response
            
        except Exception as e:
            self._stats["failed_searches"] += 1
            logger.error(f"Vector search failed: {e}", exc_info=True)
            raise
    
    def _determine_collections(
        self, 
        document_types: Optional[List[DocumentKind]], 
        strategy: CollectionStrategy
    ) -> List[str]:
        """Determine which collections to search based on document types and strategy."""
        if strategy == CollectionStrategy.ALL:
            return list(self._collection_mapping.values())
        
        if document_types is None or len(document_types) == 0:
            if strategy == CollectionStrategy.AUTO:
                # Search all collections when no specific types requested
                return list(self._collection_mapping.values())
            else:
                # Default to RFQ collection
                return [self._collection_mapping[DocumentKind.RFQ]]
        
        # Map document types to collections
        collections = []
        for doc_type in document_types:
            if doc_type in self._collection_mapping:
                collections.append(self._collection_mapping[doc_type])
        
        return collections if collections else [self._collection_mapping[DocumentKind.RFQ]]
    
    def _get_document_kind_for_collection(self, collection_name: str) -> str:
        """Get document kind string for a collection name."""
        for doc_kind, coll_name in self._collection_mapping.items():
            if coll_name == collection_name:
                return doc_kind.value if hasattr(doc_kind, 'value') else str(doc_kind)
        return DocumentKind.OTHER.value if hasattr(DocumentKind.OTHER, 'value') else str(DocumentKind.OTHER)
    
    def _build_search_filters(
        self, 
        tenant_id: str, 
        metadata_filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build ChromaDB-compatible search filters."""
        filters = {"tenant_id": tenant_id}
        
        if metadata_filters:
            # Add metadata filters
            for key, value in metadata_filters.items():
                if key not in ["tenant_id"]:  # Don't override tenant isolation
                    filters[key] = value
        
        return filters
    
    def _post_process_results(
        self, 
        results: List[VectorSearchResult], 
        config: SearchConfiguration,
        max_results: int
    ) -> List[VectorSearchResult]:
        """Post-process search results with ranking, deduplication, and filtering."""
        if not results:
            return []
        
        # Apply score threshold
        filtered_results = [
            r for r in results 
            if r.distance <= (1.0 - config.similarity_threshold)
        ]
        
        # Sort by relevance (lower distance = higher relevance)
        sorted_results = sorted(filtered_results, key=lambda x: x.distance)
        
        # Apply diversity filtering if enabled
        if config.diversity_threshold < 1.0:
            sorted_results = self._apply_diversity_filtering(
                sorted_results, 
                config.diversity_threshold
            )
        
        # Apply recency boost if enabled
        if config.boost_recent:
            sorted_results = self._apply_recency_boost(
                sorted_results, 
                config.boost_factor
            )
        
        # Limit results
        return sorted_results[:max_results]
    
    def _apply_diversity_filtering(
        self, 
        results: List[VectorSearchResult], 
        threshold: float
    ) -> List[VectorSearchResult]:
        """Apply diversity filtering to reduce similar results."""
        if not results:
            return results
        
        diverse_results = [results[0]]  # Always include the best result
        
        for result in results[1:]:
            # Check if this result is sufficiently different from existing ones
            is_diverse = True
            for existing in diverse_results:
                # Simple diversity check based on content similarity
                if self._calculate_content_similarity(result.document, existing.document) > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity between two texts."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _apply_recency_boost(
        self, 
        results: List[VectorSearchResult], 
        boost_factor: float
    ) -> List[VectorSearchResult]:
        """Apply recency boost to more recent documents."""
        # This is a simplified implementation
        # In practice, you'd use document creation dates for proper recency scoring
        return results
    
    def _update_average_search_time(self, search_time_ms: int) -> None:
        """Update the running average search time."""
        total_searches = self._stats["successful_searches"]
        current_avg = self._stats["average_search_time_ms"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_searches - 1)) + search_time_ms) / total_searches
        self._stats["average_search_time_ms"] = new_avg
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get search service statistics."""
        return dict(self._stats)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the vector search service."""
        if not self._vector_ops:
            await self.initialize()
        
        # Basic health check
        health_status = {
            "status": "healthy",
            "vector_operations_available": self._vector_ops is not None,
            "collections_configured": len(self._collection_mapping),
            "statistics": self._stats
        }
        
        return health_status


# Global instance
_vector_search_service: Optional[VectorSearchService] = None


async def get_vector_search_service() -> VectorSearchService:
    """Get the global vector search service instance."""
    global _vector_search_service
    
    if _vector_search_service is None:
        _vector_search_service = VectorSearchService()
        await _vector_search_service.initialize()
    
    return _vector_search_service
