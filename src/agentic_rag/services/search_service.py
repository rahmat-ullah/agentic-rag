"""
Search Service

This module provides the main search service that orchestrates query processing,
vector search, result ranking, pagination, and response formatting.
"""

import time
from typing import Dict, List, Optional, Any
from uuid import UUID

import structlog
from pydantic import BaseModel

from agentic_rag.api.models.search import (
    SearchFilters, SearchOptions, SearchResponse, SearchResultItem,
    SearchPagination, SearchStatistics
)
from agentic_rag.services.query_processor import get_query_processor, ProcessedQuery
from agentic_rag.services.vector_search import (
    get_vector_search_service, VectorSearchRequest, SearchConfiguration as VectorSearchConfig
)
from agentic_rag.services.result_ranker import (
    get_result_ranker, RankingConfiguration, FilterConfiguration, RankedResult
)
from agentic_rag.services.search_configuration import (
    get_search_configuration_manager,
    SearchConfiguration,
    SearchMode
)
from agentic_rag.models.database import DocumentKind

logger = structlog.get_logger(__name__)


class SearchService:
    """Main search service orchestrating the complete search pipeline."""
    
    def __init__(self, config_name: Optional[str] = None):
        self._query_processor = None
        self._vector_search = None
        self._result_ranker = None

        # Initialize configuration
        self._config_manager = get_search_configuration_manager()
        self._search_config = self._config_manager.get_configuration(config_name)

        # Search cache (simple in-memory cache for demo)
        self._search_cache: Dict[str, SearchResponse] = {}
        self._cache_ttl_seconds = self._search_config.performance.cache_ttl_seconds
        self._cache_timestamps: Dict[str, float] = {}

        # Search statistics
        self._stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_search_time_ms": 0.0,
            "total_results_returned": 0,
            "configuration_name": config_name or "default"
        }
        
        logger.info("Search service initialized")
    
    async def initialize(self) -> None:
        """Initialize the search service."""
        self._query_processor = await get_query_processor()
        self._vector_search = await get_vector_search_service()
        self._result_ranker = await get_result_ranker()
        logger.info("Search service initialized with all components")
    
    async def search_documents(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        filters: Optional[SearchFilters] = None,
        options: Optional[SearchOptions] = None,
        page: int = 1,
        page_size: int = 20
    ) -> SearchResponse:
        """
        Perform complete document search with ranking and pagination.
        
        Args:
            query: Natural language search query
            tenant_id: Tenant identifier
            user_id: User identifier
            filters: Search filters
            options: Search options
            page: Page number (1-based)
            page_size: Results per page
            
        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.time()
        
        if not self._query_processor:
            await self.initialize()
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, tenant_id, filters, options, page, page_size)
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self._stats["cache_hits"] += 1
            logger.info(f"Search served from cache", query=query[:50], cache_key=cache_key[:20])
            return cached_result
        
        self._stats["cache_misses"] += 1
        self._stats["total_searches"] += 1
        
        logger.info(
            f"Starting document search",
            query=query[:100],
            tenant_id=tenant_id,
            page=page,
            page_size=page_size
        )
        
        try:
            # Step 1: Process query
            query_start = time.time()
            processed_query = await self._query_processor.process_query(
                query=query,
                tenant_id=tenant_id,
                expand_query=True,
                generate_embedding=True
            )
            query_time_ms = int((time.time() - query_start) * 1000)
            
            # Step 2: Perform vector search
            vector_start = time.time()
            vector_request = self._build_vector_search_request(
                processed_query, tenant_id, filters, options
            )
            vector_response = await self._vector_search.search_vectors(vector_request)
            vector_time_ms = int((time.time() - vector_start) * 1000)
            
            # Step 3: Rank and filter results
            ranking_start = time.time()
            ranking_config = self._build_ranking_config(options)
            filter_config = self._build_filter_config(filters)
            
            ranked_results = await self._result_ranker.rank_and_filter_results(
                results=vector_response.results,
                query=query,
                ranking_config=ranking_config,
                filter_config=filter_config
            )
            ranking_time_ms = int((time.time() - ranking_start) * 1000)
            
            # Step 4: Apply pagination
            paginated_results, pagination_info = self._apply_pagination(
                ranked_results, page, page_size
            )
            
            # Step 5: Format response
            search_results = self._format_search_results(
                paginated_results, options
            )
            
            # Step 6: Build statistics
            total_time_ms = int((time.time() - start_time) * 1000)
            statistics = SearchStatistics(
                search_time_ms=total_time_ms,
                query_processing_time_ms=query_time_ms,
                vector_search_time_ms=vector_time_ms,
                ranking_time_ms=ranking_time_ms,
                total_chunks_searched=vector_response.total_searched,
                cache_hit=False
            )
            
            # Step 7: Create response
            response = SearchResponse(
                query=query,
                results=search_results,
                pagination=pagination_info,
                statistics=statistics,
                suggestions=None  # Could add query suggestions here
            )
            
            # Cache the result
            self._cache_result(cache_key, response)
            
            # Update statistics
            self._stats["successful_searches"] += 1
            self._stats["total_results_returned"] += len(search_results)
            self._update_average_search_time(total_time_ms)
            
            # Track result access for popularity scoring
            for result in paginated_results:
                if result.result.metadata:
                    self._result_ranker.track_result_access(
                        result.result.metadata.document_id,
                        result.result.id
                    )
            
            logger.info(
                f"Search completed successfully",
                query=query[:50],
                results_count=len(search_results),
                total_results=pagination_info.total_results,
                search_time_ms=total_time_ms,
                cache_hit=False
            )
            
            return response
            
        except Exception as e:
            self._stats["failed_searches"] += 1
            logger.error(f"Search failed: {e}", exc_info=True)
            raise
    
    def _generate_cache_key(
        self,
        query: str,
        tenant_id: str,
        filters: Optional[SearchFilters],
        options: Optional[SearchOptions],
        page: int,
        page_size: int
    ) -> str:
        """Generate a cache key for the search request."""
        # Simple cache key generation (in production, use proper hashing)
        key_parts = [
            query.lower().strip(),
            tenant_id,
            str(page),
            str(page_size)
        ]
        
        if filters:
            key_parts.append(str(filters.dict()))
        
        if options:
            key_parts.append(str(options.dict()))
        
        return "|".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[SearchResponse]:
        """Get cached search result if available and not expired."""
        if cache_key not in self._search_cache:
            return None
        
        # Check if cache entry is expired
        timestamp = self._cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp > self._cache_ttl_seconds:
            # Remove expired entry
            del self._search_cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
        
        # Update cache hit statistics
        cached_result = self._search_cache[cache_key]
        cached_result.statistics.cache_hit = True
        
        return cached_result
    
    def _cache_result(self, cache_key: str, response: SearchResponse) -> None:
        """Cache a search result."""
        # Simple cache management (in production, use Redis or similar)
        self._search_cache[cache_key] = response
        self._cache_timestamps[cache_key] = time.time()
        
        # Simple cache size management
        if len(self._search_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self._cache_timestamps.keys(),
                key=lambda k: self._cache_timestamps[k]
            )[:100]
            
            for key in oldest_keys:
                del self._search_cache[key]
                del self._cache_timestamps[key]
    
    def _build_vector_search_request(
        self,
        processed_query: ProcessedQuery,
        tenant_id: str,
        filters: Optional[SearchFilters],
        options: Optional[SearchOptions]
    ) -> VectorSearchRequest:
        """Build vector search request from processed query and filters."""
        # Determine document types to search
        document_types = None
        if filters and filters.document_types:
            document_types = filters.document_types
        
        # Build metadata filters
        metadata_filters = {}
        if filters:
            if filters.date_from or filters.date_to:
                date_filter = {}
                if filters.date_from:
                    date_filter['$gte'] = filters.date_from.isoformat()
                if filters.date_to:
                    date_filter['$lte'] = filters.date_to.isoformat()
                metadata_filters['created_at'] = date_filter
            
            if filters.section_path:
                metadata_filters['section_path'] = {'$in': filters.section_path}
            
            if filters.is_table is not None:
                metadata_filters['is_table'] = filters.is_table
            
            if filters.min_token_count:
                metadata_filters['token_count'] = {'$gte': filters.min_token_count}
            
            if filters.max_token_count:
                if 'token_count' in metadata_filters:
                    metadata_filters['token_count']['$lte'] = filters.max_token_count
                else:
                    metadata_filters['token_count'] = {'$lte': filters.max_token_count}
            
            if filters.metadata_filters:
                metadata_filters.update(filters.metadata_filters)
        
        # Configure search using global configuration
        similarity_threshold = self._search_config.similarity_thresholds.good
        if options and options.score_threshold:
            similarity_threshold = options.score_threshold

        max_results = self._search_config.result_limits.max_results
        if options and hasattr(options, 'max_results'):
            max_results = min(options.max_results, self._search_config.result_limits.max_results)

        search_config = VectorSearchConfig(
            similarity_threshold=similarity_threshold,
            max_results=max_results,
            include_metadata=options.include_metadata if options else True,
            include_documents=options.include_content if options else True
        )
        
        return VectorSearchRequest(
            query_embedding=processed_query.embedding,
            tenant_id=tenant_id,
            document_types=document_types,
            metadata_filters=metadata_filters,
            search_config=search_config,
            max_results=100,
            score_threshold=options.score_threshold if options else None
        )

    def _build_ranking_config(self, options: Optional[SearchOptions]) -> RankingConfiguration:
        """Build ranking configuration from search options and global config."""
        config = RankingConfiguration()

        # Apply ranking weights from configuration
        config.weights.similarity_weight = self._search_config.ranking_weights.semantic_similarity
        config.weights.recency_weight = self._search_config.ranking_weights.recency
        config.weights.popularity_weight = self._search_config.ranking_weights.user_interactions
        config.weights.metadata_weight = self._search_config.ranking_weights.section_relevance

        # Apply boost settings from configuration
        config.boost_factors.exact_match_boost = 2.0 if self._search_config.boost_exact_matches else 1.0
        config.boost_factors.title_match_boost = 1.5 if self._search_config.boost_title_matches else 1.0
        config.boost_factors.recency_boost = 1.3 if self._search_config.boost_recent_documents else 1.0

        # Set diversity filtering based on configuration
        if self._search_config.enable_diversity_filtering:
            config.diversity_threshold = 0.8
        else:
            config.diversity_threshold = 1.0

        # Override with user options if provided
        if options:
            if options.score_threshold:
                # Adjust similarity threshold based on score threshold
                config.weights.similarity_weight = 0.8

            if not options.deduplicate_results:
                config.diversity_threshold = 1.0  # Disable diversity filtering

        return config

    def _build_filter_config(self, filters: Optional[SearchFilters]) -> FilterConfiguration:
        """Build filter configuration from search filters."""
        config = FilterConfiguration()

        if filters:
            config.document_types = filters.document_types
            config.date_from = filters.date_from
            config.date_to = filters.date_to
            config.section_paths = filters.section_path
            config.metadata_filters = filters.metadata_filters
            config.exclude_tables = filters.is_table is False if filters.is_table is not None else False
            config.min_token_count = filters.min_token_count or 10

        return config

    def _apply_pagination(
        self,
        results: List[RankedResult],
        page: int,
        page_size: int
    ) -> tuple[List[RankedResult], SearchPagination]:
        """Apply pagination to ranked results."""
        total_results = len(results)
        total_pages = (total_results + page_size - 1) // page_size

        # Calculate offset
        offset = (page - 1) * page_size

        # Get page results
        page_results = results[offset:offset + page_size]

        # Create pagination info
        pagination = SearchPagination(
            page=page,
            page_size=page_size,
            total_results=total_results,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )

        return page_results, pagination

    def _format_search_results(
        self,
        ranked_results: List[RankedResult],
        options: Optional[SearchOptions]
    ) -> List[SearchResultItem]:
        """Format ranked results into search result items."""
        search_results = []

        for ranked_result in ranked_results:
            result = ranked_result.result

            # Build metadata if requested
            metadata = None
            if options is None or options.include_metadata:
                if result.metadata:
                    metadata = {
                        "document_type": result.metadata.document_kind,
                        "section_path": result.metadata.section_path,
                        "page_from": result.metadata.page_from,
                        "page_to": result.metadata.page_to,
                        "token_count": result.metadata.token_count,
                        "is_table": result.metadata.is_table,
                        "created_at": result.metadata.created_at
                    }

            # Build document info if requested
            document_info = None
            if options is None or options.include_document_info:
                if result.metadata:
                    document_info = {
                        "kind": result.metadata.document_kind,
                        "created_at": result.metadata.created_at
                    }

            # Get content
            content = result.document if (options is None or options.include_content) else ""

            # Create search result item
            search_item = SearchResultItem(
                document_id=UUID(result.metadata.document_id) if result.metadata else UUID("00000000-0000-0000-0000-000000000000"),
                chunk_id=UUID(result.metadata.chunk_id) if result.metadata else UUID("00000000-0000-0000-0000-000000000000"),
                title=self._extract_title_from_metadata(result.metadata),
                content=content,
                score=ranked_result.final_score,
                metadata=metadata,
                document_info=document_info,
                highlighted_content=None  # Could implement highlighting here
            )

            search_results.append(search_item)

        return search_results

    def _extract_title_from_metadata(self, metadata) -> str:
        """Extract a title from metadata or generate one."""
        if not metadata:
            return "Document Chunk"

        # Try to build a meaningful title
        title_parts = []

        if metadata.document_kind:
            title_parts.append(metadata.document_kind)

        if metadata.section_path:
            title_parts.append(" - ".join(metadata.section_path))

        if metadata.page_from:
            if metadata.page_to and metadata.page_to != metadata.page_from:
                title_parts.append(f"Pages {metadata.page_from}-{metadata.page_to}")
            else:
                title_parts.append(f"Page {metadata.page_from}")

        return " | ".join(title_parts) if title_parts else "Document Chunk"

    def _update_average_search_time(self, search_time_ms: int) -> None:
        """Update the running average search time."""
        total_searches = self._stats["successful_searches"]
        current_avg = self._stats["average_search_time_ms"]

        # Calculate new average
        new_avg = ((current_avg * (total_searches - 1)) + search_time_ms) / total_searches
        self._stats["average_search_time_ms"] = new_avg

    async def get_search_suggestions(
        self,
        partial_query: str,
        tenant_id: str,
        limit: int = 5
    ) -> List[str]:
        """Get search suggestions for partial query."""
        if not self._query_processor:
            await self.initialize()

        return await self._query_processor.get_query_suggestions(
            partial_query=partial_query,
            tenant_id=tenant_id,
            limit=limit
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the search service."""
        if not self._query_processor:
            await self.initialize()

        # Check all components
        health_status = {
            "status": "healthy",
            "components": {
                "query_processor": self._query_processor is not None,
                "vector_search": self._vector_search is not None,
                "result_ranker": self._result_ranker is not None
            },
            "cache": {
                "entries": len(self._search_cache),
                "hit_rate": self._stats["cache_hits"] / max(self._stats["cache_hits"] + self._stats["cache_misses"], 1)
            },
            "statistics": self._stats
        }

        # Check component health
        try:
            vector_health = await self._vector_search.health_check()
            health_status["components"]["vector_search_detail"] = vector_health
        except Exception as e:
            health_status["components"]["vector_search_error"] = str(e)
            health_status["status"] = "degraded"

        return health_status

    async def get_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """Get search service statistics."""
        stats = dict(self._stats)

        # Add component statistics
        if self._vector_search:
            try:
                vector_stats = await self._vector_search.get_search_statistics()
                stats["vector_search"] = vector_stats
            except Exception:
                pass

        if self._result_ranker:
            try:
                ranking_stats = await self._result_ranker.get_ranking_statistics()
                stats["result_ranking"] = ranking_stats
            except Exception:
                pass

        return stats


# Global instance
_search_service: Optional[SearchService] = None


async def get_search_service() -> SearchService:
    """Get the global search service instance."""
    global _search_service

    if _search_service is None:
        _search_service = SearchService()
        await _search_service.initialize()

    return _search_service
