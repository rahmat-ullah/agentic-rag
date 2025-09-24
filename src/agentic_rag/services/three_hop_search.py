"""
Three-Hop Search Service for implementing the three-hop retrieval pipeline.

This service orchestrates the three-hop search pattern:
H1: RFQ anchor search → H2: Linked offer discovery → H3: Targeted chunk retrieval
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session, joinedload

from agentic_rag.api.models.three_hop_search import (
    ThreeHopConfig, ThreeHopSearchRequest, ThreeHopSearchResponse,
    ThreeHopResults, ThreeHopStatistics, ThreeHopTimings,
    H1AnchorResult, H2OfferResult, H3ChunkResult
)
from agentic_rag.models.database import Document, DocumentKind, DocumentLink, DocumentChunk
from agentic_rag.services.search_service import SearchService, get_search_service
from agentic_rag.services.document_linking import DocumentLinkingService, get_document_linking_service
from agentic_rag.services.vector_search import VectorSearchService, get_vector_search_service
from agentic_rag.services.query_processor import QueryProcessor, get_query_processor
import structlog

logger = structlog.get_logger(__name__)


class ThreeHopSearchService:
    """Service for executing three-hop search queries."""
    
    def __init__(self):
        self._search_service: Optional[SearchService] = None
        self._linking_service: Optional[DocumentLinkingService] = None
        self._vector_search: Optional[VectorSearchService] = None
        self._query_processor: Optional[QueryProcessor] = None
        self._cache: Dict[str, any] = {}
        self._stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "h1_searches": 0,
            "h2_discoveries": 0,
            "h3_retrievals": 0,
            "avg_search_time_ms": 0.0,
            "parallel_executions": 0,
            "sequential_executions": 0,
            "timeout_errors": 0,
            "fallback_searches": 0,
            "reranking_operations": 0,
            "total_rfqs_processed": 0,
            "total_offers_discovered": 0,
            "total_chunks_retrieved": 0,
            "avg_h1_time_ms": 0.0,
            "avg_h2_time_ms": 0.0,
            "avg_h3_time_ms": 0.0
        }
        self._performance_history = []
        self._max_history_size = 1000
    
    async def initialize(self):
        """Initialize the service dependencies."""
        if not self._search_service:
            self._search_service = get_search_service()
            await self._search_service.initialize()
        
        if not self._linking_service:
            self._linking_service = get_document_linking_service()
            await self._linking_service.initialize()
        
        if not self._vector_search:
            self._vector_search = get_vector_search_service()
            await self._vector_search.initialize()
        
        if not self._query_processor:
            self._query_processor = get_query_processor()
            await self._query_processor.initialize()
    
    async def three_hop_search(
        self,
        db_session: Session,
        request: ThreeHopSearchRequest,
        tenant_id: UUID,
        user_id: UUID
    ) -> ThreeHopSearchResponse:
        """
        Execute a complete three-hop search.
        
        Args:
            db_session: Database session
            request: Three-hop search request
            tenant_id: Tenant identifier
            user_id: User identifier
            
        Returns:
            ThreeHopSearchResponse with results and statistics
        """
        start_time = time.time()
        
        await self.initialize()
        
        # Use default config if not provided
        config = request.config or ThreeHopConfig()
        
        # Generate cache key
        cache_key = self._generate_cache_key(request.query, tenant_id, config)
        
        # Check cache if enabled
        if config.enable_caching and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            cached_result = self._cache[cache_key]
            logger.info(f"Three-hop search served from cache", query=request.query[:50])
            return cached_result
        
        self._stats["cache_misses"] += 1
        self._stats["total_searches"] += 1
        
        logger.info(
            f"Starting three-hop search",
            query=request.query[:100],
            tenant_id=str(tenant_id),
            config=config.dict()
        )
        
        try:
            # Execute the three hops
            if config.enable_parallel_processing:
                results, timings = await self._execute_parallel_search(
                    db_session, request, config, tenant_id, user_id
                )
            else:
                results, timings = await self._execute_sequential_search(
                    db_session, request, config, tenant_id, user_id
                )
            
            # Calculate statistics
            total_time_ms = int((time.time() - start_time) * 1000)
            timings.total_time_ms = total_time_ms
            
            statistics = ThreeHopStatistics(
                timings=timings,
                rfqs_searched=len(results.h1_anchors) if results.h1_anchors else 0,
                offers_discovered=len(results.h2_offers) if results.h2_offers else 0,
                chunks_evaluated=len(results.h3_chunks) if results.h3_chunks else 0,
                cache_hits=self._stats["cache_hits"],
                parallel_execution=config.enable_parallel_processing
            )
            
            # Generate explanation if requested
            explanation = None
            if request.explain_results:
                explanation = self._generate_explanation(results, statistics, config)
            
            # Create response
            response = ThreeHopSearchResponse(
                query=request.query,
                three_hop_results=results,
                statistics=statistics,
                explanation=explanation,
                config_used=config
            )
            
            # Cache the result if enabled
            if config.enable_caching:
                self._cache[cache_key] = response
            
            # Update statistics and performance tracking
            self._update_search_stats(total_time_ms)
            self._record_performance_metrics(timings, results, config, config.enable_parallel_processing)

            logger.info(
                f"Three-hop search completed",
                query=request.query[:50],
                total_time_ms=total_time_ms,
                h1_results=len(results.h1_anchors),
                h2_results=len(results.h2_offers),
                h3_results=len(results.h3_chunks),
                parallel_execution=config.enable_parallel_processing,
                cache_hit=cache_key in self._cache if config.enable_caching else False
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Three-hop search failed: {e}", exc_info=True)
            raise
    
    async def _execute_sequential_search(
        self,
        db_session: Session,
        request: ThreeHopSearchRequest,
        config: ThreeHopConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> Tuple[ThreeHopResults, ThreeHopTimings]:
        """Execute three-hop search sequentially."""
        
        # H1: RFQ Anchor Search
        h1_start = time.time()
        h1_anchors = await self._execute_h1_rfq_search(
            db_session, request.query, config, tenant_id, user_id
        )
        h1_time_ms = int((time.time() - h1_start) * 1000)
        
        # H2: Linked Offer Discovery
        h2_start = time.time()
        h2_offers = await self._execute_h2_link_discovery(
            db_session, h1_anchors, config, tenant_id
        )
        h2_time_ms = int((time.time() - h2_start) * 1000)
        
        # H3: Targeted Chunk Retrieval
        h3_start = time.time()
        h3_chunks = await self._execute_h3_chunk_retrieval(
            db_session, h2_offers, request.query, config, tenant_id, user_id
        )
        h3_time_ms = int((time.time() - h3_start) * 1000)
        
        results = ThreeHopResults(
            h1_anchors=h1_anchors,
            h2_offers=h2_offers,
            h3_chunks=h3_chunks
        )
        
        timings = ThreeHopTimings(
            h1_time_ms=h1_time_ms,
            h2_time_ms=h2_time_ms,
            h3_time_ms=h3_time_ms,
            total_time_ms=0  # Will be set by caller
        )
        
        return results, timings
    
    async def _execute_parallel_search(
        self,
        db_session: Session,
        request: ThreeHopSearchRequest,
        config: ThreeHopConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> Tuple[ThreeHopResults, ThreeHopTimings]:
        """Execute three-hop search with parallel processing where possible."""
        
        # H1 must be executed first to get anchors
        h1_start = time.time()
        h1_anchors = await self._execute_h1_rfq_search(
            db_session, request.query, config, tenant_id, user_id
        )
        h1_time_ms = int((time.time() - h1_start) * 1000)
        
        # H2 and H3 can be partially parallelized
        h2_start = time.time()
        h2_offers = await self._execute_h2_link_discovery(
            db_session, h1_anchors, config, tenant_id
        )
        h2_time_ms = int((time.time() - h2_start) * 1000)
        
        # H3 can process multiple offers in parallel
        h3_start = time.time()
        h3_chunks = await self._execute_h3_chunk_retrieval_parallel(
            db_session, h2_offers, request.query, config, tenant_id, user_id
        )
        h3_time_ms = int((time.time() - h3_start) * 1000)
        
        results = ThreeHopResults(
            h1_anchors=h1_anchors,
            h2_offers=h2_offers,
            h3_chunks=h3_chunks
        )
        
        timings = ThreeHopTimings(
            h1_time_ms=h1_time_ms,
            h2_time_ms=h2_time_ms,
            h3_time_ms=h3_time_ms,
            total_time_ms=0  # Will be set by caller
        )
        
        return results, timings
    
    def _generate_cache_key(self, query: str, tenant_id: UUID, config: ThreeHopConfig) -> str:
        """Generate cache key for the search request."""
        import hashlib
        
        # Create a hash of the query, tenant, and key config parameters
        key_data = f"{query}_{tenant_id}_{config.h1_max_results}_{config.h2_min_confidence}_{config.h3_max_chunks_per_offer}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _update_search_stats(self, search_time_ms: int):
        """Update search statistics."""
        current_avg = self._stats["avg_search_time_ms"]
        total_searches = self._stats["total_searches"]

        # Calculate new average
        self._stats["avg_search_time_ms"] = (
            (current_avg * (total_searches - 1) + search_time_ms) / total_searches
        )

    def _record_performance_metrics(
        self,
        timings: ThreeHopTimings,
        results: ThreeHopResults,
        config: ThreeHopConfig,
        parallel_execution: bool
    ):
        """Record detailed performance metrics for analysis."""

        # Update hop-specific timing averages
        self._update_hop_timing_average("h1", timings.h1_time_ms)
        self._update_hop_timing_average("h2", timings.h2_time_ms)
        self._update_hop_timing_average("h3", timings.h3_time_ms)

        # Update execution type counters
        if parallel_execution:
            self._stats["parallel_executions"] += 1
        else:
            self._stats["sequential_executions"] += 1

        # Update result counters
        self._stats["total_rfqs_processed"] += len(results.h1_anchors)
        self._stats["total_offers_discovered"] += len(results.h2_offers)
        self._stats["total_chunks_retrieved"] += len(results.h3_chunks)

        # Record performance history
        performance_record = {
            "timestamp": time.time(),
            "total_time_ms": timings.total_time_ms,
            "h1_time_ms": timings.h1_time_ms,
            "h2_time_ms": timings.h2_time_ms,
            "h3_time_ms": timings.h3_time_ms,
            "parallel_execution": parallel_execution,
            "rfq_count": len(results.h1_anchors),
            "offer_count": len(results.h2_offers),
            "chunk_count": len(results.h3_chunks),
            "config_preset": self._detect_config_preset(config)
        }

        self._performance_history.append(performance_record)

        # Maintain history size limit
        if len(self._performance_history) > self._max_history_size:
            self._performance_history = self._performance_history[-self._max_history_size:]

    def _update_hop_timing_average(self, hop: str, time_ms: int):
        """Update average timing for a specific hop."""
        avg_key = f"avg_{hop}_time_ms"
        count_key = f"{hop}_searches"

        current_avg = self._stats.get(avg_key, 0.0)
        count = self._stats.get(count_key, 0)

        if count > 0:
            self._stats[avg_key] = ((current_avg * count) + time_ms) / (count + 1)
        else:
            self._stats[avg_key] = time_ms

    def _detect_config_preset(self, config: ThreeHopConfig) -> str:
        """Detect which preset the config most closely matches."""
        try:
            presets = ["high_precision", "high_recall", "balanced", "fast", "comprehensive"]

            for preset_name in presets:
                preset_config = ThreeHopConfig.get_preset(preset_name)
                if self._configs_match(config, preset_config):
                    return preset_name

            return "custom"
        except:
            return "unknown"

    def _configs_match(self, config1: ThreeHopConfig, config2: ThreeHopConfig) -> bool:
        """Check if two configurations are approximately the same."""
        key_fields = [
            "h1_max_results", "h1_similarity_threshold",
            "h2_min_confidence", "h2_max_offers_per_rfq",
            "h3_max_chunks_per_offer", "h3_chunk_similarity_threshold"
        ]

        for field in key_fields:
            val1 = getattr(config1, field)
            val2 = getattr(config2, field)

            if isinstance(val1, float) and isinstance(val2, float):
                if abs(val1 - val2) > 0.05:  # 5% tolerance for floats
                    return False
            elif val1 != val2:
                return False

        return True

    def get_performance_summary(self) -> Dict[str, any]:
        """Get comprehensive performance summary."""
        if not self._performance_history:
            return {"message": "No performance data available"}

        recent_records = self._performance_history[-100:]  # Last 100 searches

        # Calculate percentiles
        total_times = [r["total_time_ms"] for r in recent_records]
        total_times.sort()

        p50 = total_times[len(total_times) // 2] if total_times else 0
        p95 = total_times[int(len(total_times) * 0.95)] if total_times else 0
        p99 = total_times[int(len(total_times) * 0.99)] if total_times else 0

        # Calculate success rates
        total_searches = len(recent_records)
        successful_searches = len([r for r in recent_records if r["chunk_count"] > 0])
        success_rate = successful_searches / total_searches if total_searches > 0 else 0

        # Analyze configuration usage
        config_usage = {}
        for record in recent_records:
            preset = record.get("config_preset", "unknown")
            config_usage[preset] = config_usage.get(preset, 0) + 1

        return {
            "recent_performance": {
                "total_searches": total_searches,
                "success_rate": success_rate,
                "avg_time_ms": sum(total_times) / len(total_times) if total_times else 0,
                "p50_time_ms": p50,
                "p95_time_ms": p95,
                "p99_time_ms": p99
            },
            "hop_performance": {
                "avg_h1_time_ms": self._stats.get("avg_h1_time_ms", 0),
                "avg_h2_time_ms": self._stats.get("avg_h2_time_ms", 0),
                "avg_h3_time_ms": self._stats.get("avg_h3_time_ms", 0)
            },
            "execution_patterns": {
                "parallel_executions": self._stats.get("parallel_executions", 0),
                "sequential_executions": self._stats.get("sequential_executions", 0),
                "parallel_usage_rate": self._stats.get("parallel_executions", 0) / max(self._stats.get("total_searches", 1), 1)
            },
            "config_usage": config_usage,
            "cache_performance": {
                "hit_rate": self._stats.get("cache_hits", 0) / max(self._stats.get("total_searches", 1), 1),
                "total_hits": self._stats.get("cache_hits", 0),
                "total_misses": self._stats.get("cache_misses", 0)
            }
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get service statistics."""
        return self._stats.copy()


    async def _execute_h1_rfq_search(
        self,
        db_session: Session,
        query: str,
        config: ThreeHopConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> List[H1AnchorResult]:
        """
        Execute H1: RFQ anchor search to find relevant RFQ documents.

        Args:
            db_session: Database session
            query: Search query
            config: Three-hop configuration
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            List of H1AnchorResult objects
        """
        self._stats["h1_searches"] += 1

        logger.info(f"Executing H1 RFQ anchor search", query=query[:50])

        try:
            # Process the query
            processed_query = await self._query_processor.process_query(
                query=query,
                tenant_id=str(tenant_id),
                expand_query=True,
                generate_embedding=True
            )

            # Search RFQ collection specifically
            from agentic_rag.services.vector_search import VectorSearchRequest, VectorSearchOptions

            vector_request = VectorSearchRequest(
                query_embedding=processed_query.embedding,
                tenant_id=str(tenant_id),
                collections=["rfq"],  # Only search RFQ collection
                max_results=config.h1_max_results * 2,  # Get more to allow for filtering
                score_threshold=config.h1_similarity_threshold,
                filters={
                    "document_kind": DocumentKind.RFQ.value
                }
            )

            # Execute vector search
            vector_response = await self._vector_search.search_vectors(vector_request)

            # Get document details and link counts
            document_ids = [UUID(result.metadata.get("document_id")) for result in vector_response.results if result.metadata.get("document_id")]

            # Query documents with link counts
            documents_query = db_session.query(
                Document,
                func.count(DocumentLink.id).label("linked_offers_count")
            ).outerjoin(
                DocumentLink, and_(
                    DocumentLink.rfq_id == Document.id,
                    DocumentLink.tenant_id == tenant_id
                )
            ).filter(
                and_(
                    Document.id.in_(document_ids),
                    Document.tenant_id == tenant_id,
                    Document.kind == DocumentKind.RFQ
                )
            ).group_by(Document.id)

            documents_data = {doc.id: (doc, link_count) for doc, link_count in documents_query.all()}

            # Create H1 results
            h1_results = []
            for vector_result in vector_response.results:
                if not vector_result.metadata.get("document_id"):
                    continue

                doc_id = UUID(vector_result.metadata["document_id"])
                if doc_id not in documents_data:
                    continue

                document, linked_offers_count = documents_data[doc_id]

                # Apply recency boost if enabled
                score = vector_result.distance
                if config.h1_boost_recent:
                    days_old = (datetime.now(timezone.utc) - document.created_at).days
                    if days_old <= 30:  # Boost documents less than 30 days old
                        score *= config.h1_boost_factor

                h1_result = H1AnchorResult(
                    document_id=document.id,
                    title=document.title or f"RFQ Document {document.id}",
                    score=min(score, 1.0),  # Cap at 1.0
                    linked_offers_count=linked_offers_count,
                    created_at=document.created_at,
                    metadata={
                        "document_type": document.kind.value,
                        "status": document.status.value,
                        "chunk_count": len(document.chunks) if document.chunks else 0,
                        "file_size": document.file_size,
                        "content_type": document.content_type
                    }
                )
                h1_results.append(h1_result)

            # Sort by score (descending) and limit results
            h1_results.sort(key=lambda x: x.score, reverse=True)
            h1_results = h1_results[:config.h1_max_results]

            logger.info(
                f"H1 RFQ search completed",
                query=query[:50],
                results_found=len(h1_results),
                max_score=h1_results[0].score if h1_results else 0.0
            )

            return h1_results

        except Exception as e:
            logger.error(f"H1 RFQ search failed: {e}", exc_info=True)
            return []

    def _generate_explanation(
        self,
        results: ThreeHopResults,
        statistics: ThreeHopStatistics,
        config: ThreeHopConfig
    ) -> str:
        """Generate explanation of the search path and results."""

        explanation_parts = []

        # H1 explanation
        h1_count = len(results.h1_anchors)
        if h1_count > 0:
            explanation_parts.append(f"Found {h1_count} relevant RFQ document(s)")
            if h1_count > 0:
                max_score = max(anchor.score for anchor in results.h1_anchors)
                explanation_parts.append(f"with highest relevance score of {max_score:.2f}")
        else:
            explanation_parts.append("No relevant RFQ documents found")

        # H2 explanation
        h2_count = len(results.h2_offers)
        if h2_count > 0:
            explanation_parts.append(f"discovered {h2_count} linked offer document(s)")
            if h2_count > 0:
                max_confidence = max(offer.link_confidence for offer in results.h2_offers)
                explanation_parts.append(f"with highest link confidence of {max_confidence:.2f}")
        else:
            explanation_parts.append("no linked offers discovered")

        # H3 explanation
        h3_count = len(results.h3_chunks)
        if h3_count > 0:
            explanation_parts.append(f"retrieved {h3_count} highly relevant chunk(s)")
            if h3_count > 0:
                max_chunk_score = max(chunk.score for chunk in results.h3_chunks)
                explanation_parts.append(f"with highest chunk score of {max_chunk_score:.2f}")
        else:
            explanation_parts.append("no relevant chunks retrieved")

        # Performance explanation
        total_time = statistics.timings.total_time_ms
        explanation_parts.append(f"Search completed in {total_time}ms")

        if statistics.cache_hits > 0:
            explanation_parts.append(f"with {statistics.cache_hits} cache hit(s)")

        return ". ".join(explanation_parts) + "."

    async def _execute_h2_link_discovery(
        self,
        db_session: Session,
        h1_anchors: List[H1AnchorResult],
        config: ThreeHopConfig,
        tenant_id: UUID
    ) -> List[H2OfferResult]:
        """
        Execute H2: Linked offer discovery from RFQ anchors.

        Args:
            db_session: Database session
            h1_anchors: H1 RFQ anchor results
            config: Three-hop configuration
            tenant_id: Tenant identifier

        Returns:
            List of H2OfferResult objects
        """
        self._stats["h2_discoveries"] += 1

        if not h1_anchors:
            logger.info("No H1 anchors provided for H2 link discovery")
            return []

        logger.info(f"Executing H2 link discovery", anchor_count=len(h1_anchors))

        try:
            h2_results = []

            for anchor in h1_anchors:
                # Get links for this RFQ
                links_response = await self._linking_service.get_links_for_rfq(
                    db_session=db_session,
                    rfq_id=anchor.document_id,
                    tenant_id=tenant_id,
                    include_suggestions=config.h2_include_suggestions
                )

                # Filter by confidence threshold
                valid_links = [
                    link for link in links_response.links
                    if link.confidence >= config.h2_min_confidence
                ]

                # Sort by confidence and limit per RFQ
                valid_links.sort(key=lambda x: x.confidence, reverse=True)
                valid_links = valid_links[:config.h2_max_offers_per_rfq]

                # Convert to H2 results
                for link in valid_links:
                    # Get offer document details
                    offer_doc = db_session.query(Document).filter(
                        and_(
                            Document.id == link.offer_id,
                            Document.tenant_id == tenant_id
                        )
                    ).first()

                    if not offer_doc:
                        continue

                    # Count chunks for this offer
                    chunk_count = db_session.query(func.count(DocumentChunk.id)).filter(
                        and_(
                            DocumentChunk.document_id == offer_doc.id,
                            DocumentChunk.tenant_id == tenant_id
                        )
                    ).scalar() or 0

                    h2_result = H2OfferResult(
                        document_id=offer_doc.id,
                        title=offer_doc.title or f"Offer Document {offer_doc.id}",
                        link_confidence=link.confidence,
                        chunks_found=chunk_count,
                        source_rfq_id=anchor.document_id,
                        link_type=link.link_type.value if hasattr(link.link_type, 'value') else str(link.link_type),
                        metadata={
                            "document_type": offer_doc.kind.value,
                            "offer_type": link.offer_type,
                            "status": offer_doc.status.value,
                            "file_size": offer_doc.file_size,
                            "content_type": offer_doc.content_type,
                            "created_at": offer_doc.created_at.isoformat()
                        }
                    )
                    h2_results.append(h2_result)

            # Handle fallback search for unlinked RFQs if enabled
            if config.h2_fallback_search:
                unlinked_anchors = [
                    anchor for anchor in h1_anchors
                    if not any(result.source_rfq_id == anchor.document_id for result in h2_results)
                ]

                if unlinked_anchors:
                    fallback_results = await self._execute_h2_fallback_search(
                        db_session, unlinked_anchors, config, tenant_id
                    )
                    h2_results.extend(fallback_results)

            # Sort by link confidence and remove duplicates
            seen_offers = set()
            unique_results = []
            for result in sorted(h2_results, key=lambda x: x.link_confidence, reverse=True):
                if result.document_id not in seen_offers:
                    seen_offers.add(result.document_id)
                    unique_results.append(result)

            logger.info(
                f"H2 link discovery completed",
                anchor_count=len(h1_anchors),
                offers_found=len(unique_results),
                max_confidence=unique_results[0].link_confidence if unique_results else 0.0
            )

            return unique_results

        except Exception as e:
            logger.error(f"H2 link discovery failed: {e}", exc_info=True)
            return []

    async def _execute_h2_fallback_search(
        self,
        db_session: Session,
        unlinked_anchors: List[H1AnchorResult],
        config: ThreeHopConfig,
        tenant_id: UUID
    ) -> List[H2OfferResult]:
        """
        Execute fallback search for RFQs without links.

        This performs content similarity search to find potential offers
        even when no explicit links exist.
        """
        logger.info(f"Executing H2 fallback search", unlinked_count=len(unlinked_anchors))

        fallback_results = []

        try:
            for anchor in unlinked_anchors:
                # Get RFQ document content for similarity search
                rfq_doc = db_session.query(Document).filter(
                    and_(
                        Document.id == anchor.document_id,
                        Document.tenant_id == tenant_id
                    )
                ).first()

                if not rfq_doc or not rfq_doc.title:
                    continue

                # Use document title as query for similarity search
                from agentic_rag.services.vector_search import VectorSearchRequest

                # Process the RFQ title as a query
                processed_query = await self._query_processor.process_query(
                    query=rfq_doc.title,
                    tenant_id=str(tenant_id),
                    expand_query=False,
                    generate_embedding=True
                )

                # Search offer collections
                vector_request = VectorSearchRequest(
                    query_embedding=processed_query.embedding,
                    tenant_id=str(tenant_id),
                    collections=["offer"],  # Only search offer collections
                    max_results=config.h2_max_offers_per_rfq,
                    score_threshold=config.h2_min_confidence,
                    filters={
                        "document_kind": [DocumentKind.OFFER.value]
                    }
                )

                vector_response = await self._vector_search.search_vectors(vector_request)

                # Convert to H2 results
                for vector_result in vector_response.results:
                    if not vector_result.metadata.get("document_id"):
                        continue

                    offer_id = UUID(vector_result.metadata["document_id"])

                    # Get offer document
                    offer_doc = db_session.query(Document).filter(
                        and_(
                            Document.id == offer_id,
                            Document.tenant_id == tenant_id
                        )
                    ).first()

                    if not offer_doc:
                        continue

                    # Count chunks
                    chunk_count = db_session.query(func.count(DocumentChunk.id)).filter(
                        and_(
                            DocumentChunk.document_id == offer_doc.id,
                            DocumentChunk.tenant_id == tenant_id
                        )
                    ).scalar() or 0

                    h2_result = H2OfferResult(
                        document_id=offer_doc.id,
                        title=offer_doc.title or f"Offer Document {offer_doc.id}",
                        link_confidence=vector_result.distance,  # Use similarity as confidence
                        chunks_found=chunk_count,
                        source_rfq_id=anchor.document_id,
                        link_type="fallback_similarity",
                        metadata={
                            "document_type": offer_doc.kind.value,
                            "offer_type": "unknown",
                            "status": offer_doc.status.value,
                            "file_size": offer_doc.file_size,
                            "content_type": offer_doc.content_type,
                            "created_at": offer_doc.created_at.isoformat(),
                            "fallback_search": True
                        }
                    )
                    fallback_results.append(h2_result)

            logger.info(f"H2 fallback search completed", fallback_results=len(fallback_results))
            return fallback_results

        except Exception as e:
            logger.error(f"H2 fallback search failed: {e}", exc_info=True)
            return []


    async def _execute_h3_chunk_retrieval(
        self,
        db_session: Session,
        h2_offers: List[H2OfferResult],
        query: str,
        config: ThreeHopConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> List[H3ChunkResult]:
        """
        Execute H3: Targeted chunk retrieval from linked offers.

        Args:
            db_session: Database session
            h2_offers: H2 offer results
            query: Original search query
            config: Three-hop configuration
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            List of H3ChunkResult objects
        """
        self._stats["h3_retrievals"] += 1

        if not h2_offers:
            logger.info("No H2 offers provided for H3 chunk retrieval")
            return []

        logger.info(f"Executing H3 chunk retrieval", offer_count=len(h2_offers))

        try:
            # Process the original query for chunk search
            processed_query = await self._query_processor.process_query(
                query=query,
                tenant_id=str(tenant_id),
                expand_query=True,
                generate_embedding=True
            )

            h3_results = []

            for offer in h2_offers:
                # Search chunks within this specific offer document
                from agentic_rag.services.vector_search import VectorSearchRequest

                vector_request = VectorSearchRequest(
                    query_embedding=processed_query.embedding,
                    tenant_id=str(tenant_id),
                    collections=["offer"],  # Search offer collection
                    max_results=config.h3_max_chunks_per_offer,
                    score_threshold=config.h3_chunk_similarity_threshold,
                    filters={
                        "document_id": str(offer.document_id)  # Filter to specific document
                    }
                )

                vector_response = await self._vector_search.search_vectors(vector_request)

                # Convert to H3 results
                for vector_result in vector_response.results:
                    if not vector_result.metadata.get("chunk_id"):
                        continue

                    chunk_id = UUID(vector_result.metadata["chunk_id"])

                    # Get chunk details
                    chunk = db_session.query(DocumentChunk).filter(
                        and_(
                            DocumentChunk.id == chunk_id,
                            DocumentChunk.tenant_id == tenant_id
                        )
                    ).first()

                    if not chunk:
                        continue

                    # Build source path
                    source_path = f"{offer.title} → {vector_result.metadata.get('section_path', 'Unknown Section')}"

                    # Get context chunks if enabled
                    context_chunks = None
                    if config.h3_context_window > 0:
                        context_chunks = await self._get_context_chunks(
                            db_session, chunk, config.h3_context_window, tenant_id
                        )

                    h3_result = H3ChunkResult(
                        chunk_id=chunk.id,
                        content=chunk.content,
                        score=vector_result.distance,
                        source_path=source_path,
                        source_document_id=offer.document_id,
                        source_document_title=offer.title,
                        metadata={
                            "section_path": vector_result.metadata.get("section_path", []),
                            "page_from": vector_result.metadata.get("page_from"),
                            "page_to": vector_result.metadata.get("page_to"),
                            "token_count": chunk.token_count,
                            "is_table": vector_result.metadata.get("is_table", False),
                            "chunk_index": chunk.chunk_index,
                            "link_confidence": offer.link_confidence,
                            "offer_type": offer.metadata.get("offer_type")
                        },
                        context_chunks=context_chunks
                    )
                    h3_results.append(h3_result)

            # Apply final reranking if enabled
            if config.h3_enable_reranking and h3_results:
                h3_results = await self._rerank_h3_chunks(h3_results, query, config)

            # Sort by score and limit total results
            h3_results.sort(key=lambda x: x.score, reverse=True)
            h3_results = h3_results[:config.max_total_results]

            logger.info(
                f"H3 chunk retrieval completed",
                offer_count=len(h2_offers),
                chunks_found=len(h3_results),
                max_score=h3_results[0].score if h3_results else 0.0
            )

            return h3_results

        except Exception as e:
            logger.error(f"H3 chunk retrieval failed: {e}", exc_info=True)
            return []

    async def _execute_h3_chunk_retrieval_parallel(
        self,
        db_session: Session,
        h2_offers: List[H2OfferResult],
        query: str,
        config: ThreeHopConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> List[H3ChunkResult]:
        """
        Execute H3 chunk retrieval with parallel processing for multiple offers.
        """
        if not h2_offers:
            return []

        logger.info(f"Executing H3 chunk retrieval (parallel)", offer_count=len(h2_offers))

        # Process offers in parallel batches
        batch_size = min(5, len(h2_offers))  # Process up to 5 offers in parallel
        all_results = []

        for i in range(0, len(h2_offers), batch_size):
            batch = h2_offers[i:i + batch_size]

            # Create tasks for parallel execution
            tasks = [
                self._process_single_offer_chunks(
                    db_session, offer, query, config, tenant_id, user_id
                )
                for offer in batch
            ]

            # Execute batch in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful results
            for result in batch_results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Parallel H3 processing failed: {result}")

        # Apply final reranking if enabled
        if config.h3_enable_reranking and all_results:
            all_results = await self._rerank_h3_chunks(all_results, query, config)

        # Sort and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:config.max_total_results]

    async def _process_single_offer_chunks(
        self,
        db_session: Session,
        offer: H2OfferResult,
        query: str,
        config: ThreeHopConfig,
        tenant_id: UUID,
        user_id: UUID
    ) -> List[H3ChunkResult]:
        """Process chunks for a single offer document."""
        # This is essentially the same logic as the inner loop of _execute_h3_chunk_retrieval
        # but extracted for parallel processing
        try:
            processed_query = await self._query_processor.process_query(
                query=query,
                tenant_id=str(tenant_id),
                expand_query=True,
                generate_embedding=True
            )

            from agentic_rag.services.vector_search import VectorSearchRequest

            vector_request = VectorSearchRequest(
                query_embedding=processed_query.embedding,
                tenant_id=str(tenant_id),
                collections=["offer"],
                max_results=config.h3_max_chunks_per_offer,
                score_threshold=config.h3_chunk_similarity_threshold,
                filters={
                    "document_id": str(offer.document_id)
                }
            )

            vector_response = await self._vector_search.search_vectors(vector_request)

            results = []
            for vector_result in vector_response.results:
                if not vector_result.metadata.get("chunk_id"):
                    continue

                chunk_id = UUID(vector_result.metadata["chunk_id"])

                chunk = db_session.query(DocumentChunk).filter(
                    and_(
                        DocumentChunk.id == chunk_id,
                        DocumentChunk.tenant_id == tenant_id
                    )
                ).first()

                if not chunk:
                    continue

                source_path = f"{offer.title} → {vector_result.metadata.get('section_path', 'Unknown Section')}"

                context_chunks = None
                if config.h3_context_window > 0:
                    context_chunks = await self._get_context_chunks(
                        db_session, chunk, config.h3_context_window, tenant_id
                    )

                h3_result = H3ChunkResult(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    score=vector_result.distance,
                    source_path=source_path,
                    source_document_id=offer.document_id,
                    source_document_title=offer.title,
                    metadata={
                        "section_path": vector_result.metadata.get("section_path", []),
                        "page_from": vector_result.metadata.get("page_from"),
                        "page_to": vector_result.metadata.get("page_to"),
                        "token_count": chunk.token_count,
                        "is_table": vector_result.metadata.get("is_table", False),
                        "chunk_index": chunk.chunk_index,
                        "link_confidence": offer.link_confidence,
                        "offer_type": offer.metadata.get("offer_type")
                    },
                    context_chunks=context_chunks
                )
                results.append(h3_result)

            return results

        except Exception as e:
            logger.error(f"Single offer chunk processing failed: {e}", exc_info=True)
            return []


    async def _get_context_chunks(
        self,
        db_session: Session,
        chunk: DocumentChunk,
        context_window: int,
        tenant_id: UUID
    ) -> List[str]:
        """Get surrounding context chunks for a given chunk."""
        try:
            # Get chunks before and after the current chunk
            context_chunks = db_session.query(DocumentChunk).filter(
                and_(
                    DocumentChunk.document_id == chunk.document_id,
                    DocumentChunk.tenant_id == tenant_id,
                    DocumentChunk.chunk_index >= chunk.chunk_index - context_window,
                    DocumentChunk.chunk_index <= chunk.chunk_index + context_window,
                    DocumentChunk.id != chunk.id  # Exclude the current chunk
                )
            ).order_by(DocumentChunk.chunk_index).all()

            return [ctx_chunk.content for ctx_chunk in context_chunks]

        except Exception as e:
            logger.error(f"Failed to get context chunks: {e}")
            return []

    async def _rerank_h3_chunks(
        self,
        chunks: List[H3ChunkResult],
        query: str,
        config: ThreeHopConfig
    ) -> List[H3ChunkResult]:
        """Apply final reranking to H3 chunks based on multiple factors."""
        try:
            # Enhanced scoring that considers multiple factors
            for chunk in chunks:
                # Base score from vector similarity
                base_score = chunk.score

                # Link confidence boost
                link_confidence_boost = chunk.metadata.get("link_confidence", 0.0) * 0.2

                # Content length normalization (prefer chunks with substantial content)
                content_length = len(chunk.content)
                length_factor = min(1.0, content_length / 500)  # Normalize to 500 chars

                # Table content boost (tables often contain important structured data)
                table_boost = 0.1 if chunk.metadata.get("is_table", False) else 0.0

                # Recent document boost (if available)
                recent_boost = 0.0
                if "created_at" in chunk.metadata:
                    try:
                        created_at = datetime.fromisoformat(chunk.metadata["created_at"].replace('Z', '+00:00'))
                        days_old = (datetime.now(timezone.utc) - created_at).days
                        if days_old <= 30:
                            recent_boost = 0.05
                    except:
                        pass

                # Calculate final score
                final_score = (
                    base_score * 0.6 +  # 60% vector similarity
                    link_confidence_boost +  # 20% link confidence
                    length_factor * 0.1 +  # 10% content length
                    table_boost +  # 10% table boost
                    recent_boost  # 5% recency boost
                )

                # Update the score
                chunk.score = min(final_score, 1.0)  # Cap at 1.0

            # Sort by the new scores
            chunks.sort(key=lambda x: x.score, reverse=True)

            logger.info(f"Reranked {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Chunk reranking failed: {e}")
            return chunks  # Return original order if reranking fails


# Singleton instance
_three_hop_search_service: Optional[ThreeHopSearchService] = None


def get_three_hop_search_service() -> ThreeHopSearchService:
    """Get the singleton three-hop search service instance."""
    global _three_hop_search_service
    if _three_hop_search_service is None:
        _three_hop_search_service = ThreeHopSearchService()
    return _three_hop_search_service
