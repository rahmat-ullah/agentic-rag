"""
Enhanced Query Processing Service

This module provides enhanced query processing capabilities with procurement-specific
terminology expansion, context injection, RFQ hint processing, intent classification,
and intelligent search strategy selection.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.query_processor import (
    get_query_processor, ProcessedQuery, QueryType, QueryIntent
)
from agentic_rag.services.procurement_terminology import (
    get_procurement_terminology_service, TerminologyExpansionConfig, TermExpansion
)
from agentic_rag.services.advanced_query_preprocessor import (
    get_advanced_query_preprocessor, PreprocessingConfig
)
from agentic_rag.services.context_injection import (
    get_context_injection_service, ContextInjectionConfig, InjectedContext
)
from agentic_rag.services.rfq_hint_processor import (
    get_rfq_hint_processor, RFQHintConfig, RFQHint
)
from agentic_rag.services.query_intent_classifier import (
    get_query_intent_classifier, QueryIntentConfig, ClassificationResult, ProcurementIntent
)
from agentic_rag.services.search_strategy_selector import (
    get_search_strategy_selector, SearchStrategyConfig, StrategySelection
)

logger = structlog.get_logger(__name__)


class EnhancementLevel(str, Enum):
    """Levels of query enhancement."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    COMPREHENSIVE = "comprehensive"


class ProcessingMode(str, Enum):
    """Query processing modes."""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    CUSTOM = "custom"


@dataclass
class EnhancementMetrics:
    """Metrics for query enhancement performance."""
    terminology_expansions: int
    context_injections: int
    rfq_hints_processed: int
    intent_confidence: float
    processing_time_ms: int
    enhancement_effectiveness: float


class EnhancedQueryConfig(BaseModel):
    """Configuration for enhanced query processing."""
    
    # Enhancement settings
    enhancement_level: EnhancementLevel = Field(
        EnhancementLevel.STANDARD, 
        description="Level of query enhancement"
    )
    processing_mode: ProcessingMode = Field(
        ProcessingMode.BALANCED,
        description="Processing mode for performance tuning"
    )
    
    # Feature toggles
    enable_terminology_expansion: bool = Field(True, description="Enable procurement terminology expansion")
    enable_context_injection: bool = Field(True, description="Enable context injection")
    enable_rfq_hint_processing: bool = Field(True, description="Enable RFQ hint processing")
    enable_intent_classification: bool = Field(True, description="Enable intent classification")
    enable_strategy_selection: bool = Field(True, description="Enable search strategy selection")
    
    # Performance settings
    max_processing_time_ms: int = Field(200, ge=50, le=1000, description="Maximum processing time")
    enable_caching: bool = Field(True, description="Enable result caching")
    cache_ttl_minutes: int = Field(30, ge=5, le=120, description="Cache TTL in minutes")
    
    # Quality settings
    min_enhancement_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Minimum enhancement confidence")
    max_expansions_per_query: int = Field(10, ge=1, le=20, description="Maximum expansions per query")
    
    # Terminology expansion config
    terminology_config: TerminologyExpansionConfig = Field(
        default_factory=TerminologyExpansionConfig,
        description="Terminology expansion configuration"
    )


class EnhancedProcessedQuery(BaseModel):
    """Enhanced processed query with additional metadata."""
    
    # Base query processing results
    original_query: str = Field(..., description="Original user query")
    processed_query: str = Field(..., description="Processed query")
    expanded_query: str = Field(..., description="Expanded query with terminology")
    enhanced_query: str = Field(..., description="Final enhanced query")
    
    # Embeddings and vectors
    embedding: Optional[List[float]] = Field(None, description="Query embedding vector")
    
    # Classification results
    query_type: QueryType = Field(..., description="Detected query type")
    intent: QueryIntent = Field(..., description="Detected query intent")
    procurement_intent: str = Field(..., description="Procurement-specific intent")
    
    # Enhancement results
    terminology_expansions: List[TermExpansion] = Field(
        default_factory=list, 
        description="Applied terminology expansions"
    )
    context_injections: List[InjectedContext] = Field(
        default_factory=list,
        description="Applied context injections"
    )
    rfq_hints: List[RFQHint] = Field(
        default_factory=list,
        description="Detected RFQ hints"
    )
    
    # Strategy and routing
    recommended_strategy: str = Field(..., description="Recommended search strategy")
    routing_suggestions: List[str] = Field(
        default_factory=list,
        description="Search routing suggestions"
    )
    
    # Quality metrics
    enhancement_confidence: float = Field(..., description="Overall enhancement confidence")
    metrics: EnhancementMetrics = Field(..., description="Processing metrics")
    
    # Processing metadata
    processing_time_ms: int = Field(..., description="Total processing time")
    cache_hit: bool = Field(False, description="Whether result was cached")


class EnhancedQueryProcessor:
    """Enhanced query processor with procurement-specific capabilities."""
    
    def __init__(self):
        self._base_processor = None
        self._terminology_service = None
        self._advanced_preprocessor = None
        self._context_injection_service = None
        self._rfq_hint_processor = None
        self._intent_classifier = None
        self._strategy_selector = None
        
        # Processing cache
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Processing statistics
        self._stats = {
            "total_queries_processed": 0,
            "terminology_expansions_applied": 0,
            "context_injections_applied": 0,
            "rfq_hints_detected": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time_ms": 0.0,
            "enhancement_success_rate": 0.0
        }
        
        logger.info("Enhanced query processor initialized")
    
    async def initialize(self):
        """Initialize the enhanced query processor."""
        if not self._base_processor:
            self._base_processor = get_query_processor()
            await self._base_processor.initialize()
        
        if not self._terminology_service:
            self._terminology_service = get_procurement_terminology_service()
        
        if not self._advanced_preprocessor:
            self._advanced_preprocessor = await get_advanced_query_preprocessor()

        if not self._context_injection_service:
            self._context_injection_service = get_context_injection_service()

        if not self._rfq_hint_processor:
            self._rfq_hint_processor = get_rfq_hint_processor()

        if not self._intent_classifier:
            self._intent_classifier = get_query_intent_classifier()

        if not self._strategy_selector:
            self._strategy_selector = get_search_strategy_selector()

        logger.info("Enhanced query processor services initialized")
    
    async def process_enhanced_query(
        self,
        query: str,
        tenant_id: str,
        user_id: Optional[str] = None,
        config: Optional[EnhancedQueryConfig] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EnhancedProcessedQuery:
        """
        Process query with enhanced procurement-specific capabilities.
        
        Args:
            query: Original user query
            tenant_id: Tenant identifier
            user_id: Optional user identifier
            config: Enhancement configuration
            context: Optional processing context
            
        Returns:
            EnhancedProcessedQuery with all enhancement results
        """
        start_time = time.time()
        
        if not self._base_processor:
            await self.initialize()
        
        # Use default config if not provided
        if config is None:
            config = EnhancedQueryConfig()
        
        # Check cache first
        cache_key = self._generate_cache_key(query, tenant_id, config)
        cached_result = self._get_cached_result(cache_key, config)
        if cached_result:
            self._stats["cache_hits"] += 1
            cached_result.cache_hit = True
            return cached_result
        
        self._stats["cache_misses"] += 1
        self._stats["total_queries_processed"] += 1
        
        logger.info(
            f"Processing enhanced query",
            query=query[:100],
            tenant_id=tenant_id,
            enhancement_level=config.enhancement_level.value
        )
        
        try:
            # Step 1: Base query processing
            base_start = time.time()
            base_processed = await self._base_processor.process_query(
                query=query,
                tenant_id=tenant_id,
                expand_query=True,
                generate_embedding=True
            )
            base_time_ms = int((time.time() - base_start) * 1000)
            
            # Step 2: Terminology expansion
            terminology_start = time.time()
            expanded_query, terminology_expansions = await self._apply_terminology_expansion(
                base_processed.expanded_query, config
            )
            terminology_time_ms = int((time.time() - terminology_start) * 1000)
            
            # Step 3: Context injection
            context_start = time.time()
            enhanced_query, context_injections = await self._apply_context_injection(
                expanded_query, base_processed, config, context
            )
            context_time_ms = int((time.time() - context_start) * 1000)
            
            # Step 4: RFQ hint processing
            rfq_start = time.time()
            rfq_hints = await self._process_rfq_hints(
                enhanced_query, config, context
            )
            rfq_time_ms = int((time.time() - rfq_start) * 1000)
            
            # Step 5: Intent classification
            intent_start = time.time()
            procurement_intent, intent_confidence = await self._classify_procurement_intent(
                enhanced_query, base_processed, config
            )
            intent_time_ms = int((time.time() - intent_start) * 1000)
            
            # Step 6: Search strategy selection
            strategy_start = time.time()
            recommended_strategy, routing_suggestions = await self._select_search_strategy(
                enhanced_query, base_processed, procurement_intent, config
            )
            strategy_time_ms = int((time.time() - strategy_start) * 1000)
            
            # Calculate overall metrics
            total_time_ms = int((time.time() - start_time) * 1000)
            enhancement_confidence = self._calculate_enhancement_confidence(
                terminology_expansions, context_injections, rfq_hints, intent_confidence
            )
            
            metrics = EnhancementMetrics(
                terminology_expansions=len(terminology_expansions),
                context_injections=len(context_injections),
                rfq_hints_processed=len(rfq_hints),
                intent_confidence=intent_confidence,
                processing_time_ms=total_time_ms,
                enhancement_effectiveness=enhancement_confidence
            )
            
            # Build enhanced result
            enhanced_result = EnhancedProcessedQuery(
                original_query=query,
                processed_query=base_processed.processed_query,
                expanded_query=expanded_query,
                enhanced_query=enhanced_query,
                embedding=base_processed.embedding,
                query_type=base_processed.query_type,
                intent=base_processed.intent,
                procurement_intent=procurement_intent,
                terminology_expansions=terminology_expansions,
                context_injections=context_injections,
                rfq_hints=rfq_hints,
                recommended_strategy=recommended_strategy,
                routing_suggestions=routing_suggestions,
                enhancement_confidence=enhancement_confidence,
                metrics=metrics,
                processing_time_ms=total_time_ms,
                cache_hit=False
            )
            
            # Cache result if enabled
            if config.enable_caching:
                self._cache_result(cache_key, enhanced_result, config)
            
            # Update statistics
            self._update_statistics(enhanced_result)
            
            logger.info(
                f"Enhanced query processing complete",
                processing_time_ms=total_time_ms,
                enhancement_confidence=enhancement_confidence,
                terminology_expansions=len(terminology_expansions),
                context_injections=len(context_injections)
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}", exc_info=True)
            raise

    async def _apply_terminology_expansion(
        self,
        query: str,
        config: EnhancedQueryConfig
    ) -> Tuple[str, List[TermExpansion]]:
        """Apply procurement terminology expansion to query."""

        if not config.enable_terminology_expansion:
            return query, []

        try:
            expanded_query, expansions = self._terminology_service.expand_query_terms(
                query, config.terminology_config
            )

            # Limit expansions based on config
            limited_expansions = expansions[:config.max_expansions_per_query]

            self._stats["terminology_expansions_applied"] += len(limited_expansions)

            return expanded_query, limited_expansions

        except Exception as e:
            logger.warning(f"Terminology expansion failed: {e}")
            return query, []

    async def _apply_context_injection(
        self,
        query: str,
        base_processed: ProcessedQuery,
        config: EnhancedQueryConfig,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[str, List[InjectedContext]]:
        """Apply context injection based on query type and intent."""

        if not config.enable_context_injection:
            return query, []

        try:
            # Create context injection config
            context_config = ContextInjectionConfig(
                user_role=getattr(config, 'user_role', None),
                process_stage=getattr(config, 'process_stage', None),
                min_context_confidence=config.min_enhancement_confidence,
                max_contexts_per_query=min(config.max_expansions_per_query, 5)
            )

            # Apply context injection using the dedicated service
            enhanced_query, injected_contexts = self._context_injection_service.inject_context(
                query=query,
                query_type=base_processed.query_type,
                query_intent=base_processed.intent,
                config=context_config,
                user_context=context
            )

            self._stats["context_injections_applied"] += len(injected_contexts)

            return enhanced_query, injected_contexts

        except Exception as e:
            logger.warning(f"Context injection failed: {e}")
            return query, []

    async def _process_rfq_hints(
        self,
        query: str,
        config: EnhancedQueryConfig,
        context: Optional[Dict[str, Any]]
    ) -> List[RFQHint]:
        """Process RFQ hints to guide search strategy."""

        if not config.enable_rfq_hint_processing:
            return []

        try:
            # Create RFQ hint config
            rfq_config = RFQHintConfig(
                max_hints_per_query=min(config.max_expansions_per_query, 10),
                relevance_threshold=config.min_enhancement_confidence
            )

            # Process RFQ hints using the dedicated service
            hints = self._rfq_hint_processor.process_rfq_hints(
                query=query,
                config=rfq_config,
                context=context
            )

            self._stats["rfq_hints_detected"] += len(hints)

            return hints

        except Exception as e:
            logger.warning(f"RFQ hint processing failed: {e}")
            return []

    async def _classify_procurement_intent(
        self,
        query: str,
        base_processed: ProcessedQuery,
        config: EnhancedQueryConfig
    ) -> Tuple[str, float]:
        """Classify procurement-specific intent."""

        if not config.enable_intent_classification:
            return "general", 0.5

        try:
            # Create intent classification config
            intent_config = QueryIntentConfig(
                min_confidence_threshold=config.min_enhancement_confidence,
                enable_explanation_generation=True
            )

            # Classify intent using the dedicated service
            classification_result = self._intent_classifier.classify_intent(
                query=query,
                config=intent_config,
                context={
                    "query_type": base_processed.query_type.value,
                    "query_intent": base_processed.intent.value
                }
            )

            return classification_result.intent.value, classification_result.confidence

        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")
            return "general", 0.5

    async def _select_search_strategy(
        self,
        query: str,
        base_processed: ProcessedQuery,
        procurement_intent: str,
        config: EnhancedQueryConfig
    ) -> Tuple[str, List[str]]:
        """Select optimal search strategy based on query analysis."""

        if not config.enable_strategy_selection:
            return "default", []

        try:
            # Create strategy selection config
            strategy_config = SearchStrategyConfig(
                min_confidence_threshold=config.min_enhancement_confidence,
                enable_explanation=True,
                enable_performance_prediction=True
            )

            # Select strategy using the dedicated service
            strategy_selection = self._strategy_selector.select_strategy(
                query=query,
                query_type=base_processed.query_type,
                query_intent=base_processed.intent,
                procurement_intent=procurement_intent,
                config=strategy_config,
                context={
                    "query_length": len(query.split()),
                    "processing_mode": config.processing_mode.value
                }
            )

            # Generate routing suggestions based on strategy and parameters
            routing_suggestions = []

            if strategy_selection.strategy.value == "three_hop_search":
                routing_suggestions.append("enable_document_linking")
            elif strategy_selection.strategy.value == "comprehensive_search":
                routing_suggestions.append("enable_reranking")
            elif strategy_selection.strategy.value == "priority_search":
                routing_suggestions.append("fast_mode")

            if base_processed.query_type == QueryType.QUESTION:
                routing_suggestions.append("enable_explanation")

            return strategy_selection.strategy.value, routing_suggestions

        except Exception as e:
            logger.warning(f"Strategy selection failed: {e}")
            return "default", []

    def _calculate_enhancement_confidence(
        self,
        terminology_expansions: List[TermExpansion],
        context_injections: List[str],
        rfq_hints: List[str],
        intent_confidence: float
    ) -> float:
        """Calculate overall enhancement confidence score."""

        # Base confidence from intent classification
        confidence = intent_confidence * 0.4

        # Add confidence from terminology expansions
        if terminology_expansions:
            avg_term_confidence = sum(exp.confidence for exp in terminology_expansions) / len(terminology_expansions)
            confidence += avg_term_confidence * 0.3

        # Add confidence from context injections
        if context_injections:
            confidence += min(len(context_injections) * 0.1, 0.2)

        # Add confidence from RFQ hints
        if rfq_hints:
            confidence += min(len(rfq_hints) * 0.05, 0.1)

        return min(confidence, 1.0)

    def _generate_cache_key(
        self,
        query: str,
        tenant_id: str,
        config: EnhancedQueryConfig
    ) -> str:
        """Generate cache key for query processing result."""

        import hashlib

        # Create hash from query, tenant, and key config parameters
        key_data = f"{query}:{tenant_id}:{config.enhancement_level.value}:{config.processing_mode.value}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_result(
        self,
        cache_key: str,
        config: EnhancedQueryConfig
    ) -> Optional[EnhancedProcessedQuery]:
        """Get cached result if available and not expired."""

        if not config.enable_caching or cache_key not in self._cache:
            return None

        # Check if cache entry is expired
        cache_time = self._cache_timestamps.get(cache_key, 0)
        current_time = time.time()
        ttl_seconds = config.cache_ttl_minutes * 60

        if current_time - cache_time > ttl_seconds:
            # Remove expired entry
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._cache[cache_key]

    def _cache_result(
        self,
        cache_key: str,
        result: EnhancedProcessedQuery,
        config: EnhancedQueryConfig
    ):
        """Cache processing result."""

        if config.enable_caching:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()

            # Simple cache cleanup - remove oldest entries if cache is too large
            if len(self._cache) > 1000:
                oldest_key = min(self._cache_timestamps.items(), key=lambda x: x[1])[0]
                del self._cache[oldest_key]
                del self._cache_timestamps[oldest_key]

    def _update_statistics(self, result: EnhancedProcessedQuery):
        """Update processing statistics."""

        # Update average processing time
        current_avg = self._stats["average_processing_time_ms"]
        total_queries = self._stats["total_queries_processed"]

        new_avg = ((current_avg * (total_queries - 1)) + result.processing_time_ms) / total_queries
        self._stats["average_processing_time_ms"] = new_avg

        # Update enhancement success rate
        if result.enhancement_confidence >= 0.7:
            success_count = self._stats.get("successful_enhancements", 0) + 1
            self._stats["successful_enhancements"] = success_count
            self._stats["enhancement_success_rate"] = success_count / total_queries

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()

    def clear_cache(self):
        """Clear processing cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Enhanced query processor cache cleared")


# Singleton instance
_enhanced_query_processor: Optional[EnhancedQueryProcessor] = None


async def get_enhanced_query_processor() -> EnhancedQueryProcessor:
    """Get the singleton enhanced query processor instance."""
    global _enhanced_query_processor
    if _enhanced_query_processor is None:
        _enhanced_query_processor = EnhancedQueryProcessor()
        await _enhanced_query_processor.initialize()
    return _enhanced_query_processor
