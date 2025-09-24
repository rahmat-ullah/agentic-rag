"""
Search Strategy Selection Service

This module provides intelligent search strategy selection based on query analysis,
performance monitoring, adaptive optimization, and explanation systems.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.query_intent_classifier import ProcurementIntent
from agentic_rag.services.query_processor import QueryType, QueryIntent

logger = structlog.get_logger(__name__)


class SearchStrategy(str, Enum):
    """Available search strategies."""
    SEMANTIC_SEARCH = "semantic_search"
    THREE_HOP_SEARCH = "three_hop_search"
    FILTERED_SEARCH = "filtered_search"
    EXACT_MATCH = "exact_match"
    DOCUMENT_SPECIFIC = "document_specific"
    HYBRID_SEARCH = "hybrid_search"
    BALANCED_SEARCH = "balanced_search"
    PRIORITY_SEARCH = "priority_search"
    COMPREHENSIVE_SEARCH = "comprehensive_search"


class StrategyConfidence(str, Enum):
    """Confidence levels for strategy selection."""
    VERY_HIGH = "very_high"  # 0.9+
    HIGH = "high"           # 0.8-0.89
    MEDIUM = "medium"       # 0.6-0.79
    LOW = "low"            # 0.4-0.59
    VERY_LOW = "very_low"  # <0.4


@dataclass
class StrategyFeature:
    """Feature for strategy selection."""
    name: str
    condition: str
    weight: float
    strategy: SearchStrategy
    confidence_boost: float


@dataclass
class StrategySelection:
    """Result of strategy selection."""
    strategy: SearchStrategy
    confidence: float
    confidence_level: StrategyConfidence
    reasoning: str
    alternative_strategies: List[Tuple[SearchStrategy, float]]
    parameters: Dict[str, Any]
    performance_prediction: Dict[str, float]


class SearchStrategyConfig(BaseModel):
    """Configuration for search strategy selection."""
    
    # Selection settings
    min_confidence_threshold: float = Field(0.4, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_alternative_strategies: int = Field(3, ge=1, le=5, description="Maximum alternative strategies")
    enable_performance_prediction: bool = Field(True, description="Enable performance prediction")
    
    # Optimization settings
    enable_adaptive_optimization: bool = Field(True, description="Enable adaptive optimization")
    performance_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for performance in selection")
    accuracy_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight for accuracy in selection")
    cost_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for cost in selection")
    
    # Fallback settings
    enable_fallback_strategies: bool = Field(True, description="Enable fallback strategies")
    fallback_timeout_seconds: int = Field(30, ge=5, le=120, description="Fallback timeout in seconds")
    
    # Explanation settings
    enable_explanation: bool = Field(True, description="Enable strategy explanation")
    detailed_reasoning: bool = Field(False, description="Enable detailed reasoning")


class SearchStrategySelector:
    """Service for intelligent search strategy selection."""
    
    def __init__(self):
        self._strategy_features = self._initialize_strategy_features()
        self._intent_strategy_mapping = self._initialize_intent_strategy_mapping()
        self._query_type_mapping = self._initialize_query_type_mapping()
        self._performance_history = self._initialize_performance_history()
        
        # Selection statistics
        self._stats = {
            "total_selections": 0,
            "strategy_distribution": {strategy.value: 0 for strategy in SearchStrategy},
            "confidence_distribution": {level.value: 0 for level in StrategyConfidence},
            "average_confidence": 0.0,
            "performance_accuracy": 0.0,
            "adaptive_improvements": 0
        }
        
        logger.info("Search strategy selector initialized")
    
    def select_strategy(
        self,
        query: str,
        query_type: QueryType,
        query_intent: QueryIntent,
        procurement_intent: str,
        config: SearchStrategyConfig,
        context: Optional[Dict[str, Any]] = None
    ) -> StrategySelection:
        """
        Select optimal search strategy based on query analysis.
        
        Args:
            query: Query string
            query_type: Detected query type
            query_intent: Detected query intent
            procurement_intent: Procurement-specific intent
            config: Strategy selection configuration
            context: Optional context information
            
        Returns:
            StrategySelection with recommended strategy and metadata
        """
        
        logger.info(f"Selecting search strategy for query: {query[:100]}...")
        
        self._stats["total_selections"] += 1
        
        # Collect strategy scores from different methods
        strategy_scores = {}
        
        # 1. Intent-based strategy selection
        intent_scores = self._select_by_intent(procurement_intent, config)
        self._merge_scores(strategy_scores, intent_scores, weight=0.4)
        
        # 2. Query type-based selection
        type_scores = self._select_by_query_type(query_type, query_intent, config)
        self._merge_scores(strategy_scores, type_scores, weight=0.3)
        
        # 3. Feature-based selection
        feature_scores = self._select_by_features(query, config)
        self._merge_scores(strategy_scores, feature_scores, weight=0.2)
        
        # 4. Performance-based optimization
        if config.enable_adaptive_optimization:
            performance_scores = self._optimize_by_performance(query, config)
            self._merge_scores(strategy_scores, performance_scores, weight=0.1)
        
        # Determine final strategy selection
        selection = self._determine_final_selection(
            query, strategy_scores, config, context
        )
        
        # Update statistics
        self._update_statistics(selection)
        
        logger.info(
            f"Strategy selection complete",
            strategy=selection.strategy.value,
            confidence=selection.confidence,
            reasoning=selection.reasoning[:100]
        )
        
        return selection
    
    def _select_by_intent(
        self,
        procurement_intent: str,
        config: SearchStrategyConfig
    ) -> Dict[SearchStrategy, float]:
        """Select strategy based on procurement intent."""
        
        scores = {strategy: 0.0 for strategy in SearchStrategy}
        
        # Map procurement intent to strategies
        if procurement_intent in self._intent_strategy_mapping:
            strategy_preferences = self._intent_strategy_mapping[procurement_intent]
            
            for strategy, score in strategy_preferences.items():
                scores[strategy] = score
        
        return scores
    
    def _select_by_query_type(
        self,
        query_type: QueryType,
        query_intent: QueryIntent,
        config: SearchStrategyConfig
    ) -> Dict[SearchStrategy, float]:
        """Select strategy based on query type and intent."""
        
        scores = {strategy: 0.0 for strategy in SearchStrategy}
        
        # Query type preferences
        type_key = f"{query_type.value}_{query_intent.value}"
        if type_key in self._query_type_mapping:
            strategy_preferences = self._query_type_mapping[type_key]
            
            for strategy, score in strategy_preferences.items():
                scores[strategy] = score
        
        return scores
    
    def _select_by_features(
        self,
        query: str,
        config: SearchStrategyConfig
    ) -> Dict[SearchStrategy, float]:
        """Select strategy based on query features."""
        
        scores = {strategy: 0.0 for strategy in SearchStrategy}
        query_lower = query.lower()
        
        for feature in self._strategy_features:
            if self._evaluate_condition(query_lower, feature.condition):
                current_score = scores[feature.strategy]
                boost = feature.weight * feature.confidence_boost
                scores[feature.strategy] = min(current_score + boost, 1.0)
        
        return scores
    
    def _optimize_by_performance(
        self,
        query: str,
        config: SearchStrategyConfig
    ) -> Dict[SearchStrategy, float]:
        """Optimize strategy selection based on performance history."""
        
        scores = {strategy: 0.0 for strategy in SearchStrategy}
        
        # Analyze query characteristics for performance prediction
        query_length = len(query.split())
        query_complexity = self._calculate_query_complexity(query)
        
        # Apply performance-based scoring
        for strategy in SearchStrategy:
            if strategy.value in self._performance_history:
                history = self._performance_history[strategy.value]
                
                # Calculate performance score based on historical data
                performance_score = (
                    history["accuracy"] * config.accuracy_weight +
                    (1 - history["avg_response_time"] / 10) * config.performance_weight +
                    (1 - history["cost_factor"]) * config.cost_weight
                )
                
                scores[strategy] = min(performance_score, 1.0)
        
        return scores
    
    def _merge_scores(
        self,
        target_scores: Dict[SearchStrategy, float],
        source_scores: Dict[SearchStrategy, float],
        weight: float
    ):
        """Merge strategy scores with weighting."""
        
        for strategy, score in source_scores.items():
            if strategy not in target_scores:
                target_scores[strategy] = 0.0
            target_scores[strategy] += score * weight
    
    def _determine_final_selection(
        self,
        query: str,
        scores: Dict[SearchStrategy, float],
        config: SearchStrategyConfig,
        context: Optional[Dict[str, Any]]
    ) -> StrategySelection:
        """Determine final strategy selection."""
        
        # Sort strategies by score
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get best strategy and confidence
        best_strategy, best_confidence = sorted_strategies[0] if sorted_strategies else (SearchStrategy.BALANCED_SEARCH, 0.5)
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(best_confidence)
        
        # Get alternative strategies
        alternative_strategies = [
            (strategy, score) for strategy, score in sorted_strategies[1:config.max_alternative_strategies + 1]
            if score >= config.min_confidence_threshold
        ]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            query, best_strategy, best_confidence, config
        ) if config.enable_explanation else ""
        
        # Generate strategy parameters
        parameters = self._generate_strategy_parameters(best_strategy, query, context)
        
        # Predict performance
        performance_prediction = self._predict_performance(
            best_strategy, query, config
        ) if config.enable_performance_prediction else {}
        
        return StrategySelection(
            strategy=best_strategy,
            confidence=best_confidence,
            confidence_level=confidence_level,
            reasoning=reasoning,
            alternative_strategies=alternative_strategies,
            parameters=parameters,
            performance_prediction=performance_prediction
        )
    
    def _evaluate_condition(self, query: str, condition: str) -> bool:
        """Evaluate a condition against the query."""
        
        if condition.startswith("contains:"):
            term = condition.split(":", 1)[1]
            return term in query
        elif condition.startswith("length_gt:"):
            threshold = int(condition.split(":")[1])
            return len(query.split()) > threshold
        elif condition.startswith("length_lt:"):
            threshold = int(condition.split(":")[1])
            return len(query.split()) < threshold
        elif condition.startswith("pattern:"):
            import re
            pattern = condition.split(":", 1)[1]
            return bool(re.search(pattern, query))
        
        return False
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        
        # Simple complexity calculation based on various factors
        word_count = len(query.split())
        unique_words = len(set(query.lower().split()))
        avg_word_length = sum(len(word) for word in query.split()) / max(word_count, 1)
        
        # Normalize to 0-1 scale
        complexity = min((word_count * 0.1 + unique_words * 0.05 + avg_word_length * 0.1) / 3, 1.0)
        
        return complexity

    def _get_confidence_level(self, confidence: float) -> StrategyConfidence:
        """Convert numeric confidence to confidence level."""

        if confidence >= 0.9:
            return StrategyConfidence.VERY_HIGH
        elif confidence >= 0.8:
            return StrategyConfidence.HIGH
        elif confidence >= 0.6:
            return StrategyConfidence.MEDIUM
        elif confidence >= 0.4:
            return StrategyConfidence.LOW
        else:
            return StrategyConfidence.VERY_LOW

    def _generate_reasoning(
        self,
        query: str,
        strategy: SearchStrategy,
        confidence: float,
        config: SearchStrategyConfig
    ) -> str:
        """Generate reasoning for strategy selection."""

        strategy_explanations = {
            SearchStrategy.SEMANTIC_SEARCH: "Best for conceptual and meaning-based queries",
            SearchStrategy.THREE_HOP_SEARCH: "Optimal for RFQ-Offer relationship queries",
            SearchStrategy.FILTERED_SEARCH: "Ideal for queries with specific criteria",
            SearchStrategy.EXACT_MATCH: "Perfect for precise term matching",
            SearchStrategy.DOCUMENT_SPECIFIC: "Suited for document-type specific queries",
            SearchStrategy.HYBRID_SEARCH: "Combines multiple search approaches",
            SearchStrategy.BALANCED_SEARCH: "General-purpose search strategy",
            SearchStrategy.PRIORITY_SEARCH: "Optimized for urgent or high-priority queries",
            SearchStrategy.COMPREHENSIVE_SEARCH: "Thorough search for complex queries"
        }

        base_explanation = strategy_explanations.get(strategy, "Selected based on query analysis")
        confidence_text = f"with {confidence:.1%} confidence"

        if config.detailed_reasoning:
            query_length = len(query.split())
            complexity = self._calculate_query_complexity(query)
            return f"{base_explanation} {confidence_text}. Query length: {query_length} words, complexity: {complexity:.2f}"
        else:
            return f"{base_explanation} {confidence_text}"

    def _generate_strategy_parameters(
        self,
        strategy: SearchStrategy,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate strategy-specific parameters."""

        base_params = {
            "similarity_threshold": 0.7,
            "max_results": 20,
            "timeout_seconds": 30
        }

        # Strategy-specific parameter adjustments
        if strategy == SearchStrategy.THREE_HOP_SEARCH:
            base_params.update({
                "h1_limit": 5,
                "h2_limit": 10,
                "h3_limit": 20,
                "confidence_threshold": 0.6
            })
        elif strategy == SearchStrategy.EXACT_MATCH:
            base_params.update({
                "similarity_threshold": 0.95,
                "fuzzy_matching": False
            })
        elif strategy == SearchStrategy.COMPREHENSIVE_SEARCH:
            base_params.update({
                "max_results": 50,
                "timeout_seconds": 60,
                "enable_reranking": True
            })
        elif strategy == SearchStrategy.PRIORITY_SEARCH:
            base_params.update({
                "timeout_seconds": 15,
                "max_results": 10,
                "fast_mode": True
            })

        return base_params

    def _predict_performance(
        self,
        strategy: SearchStrategy,
        query: str,
        config: SearchStrategyConfig
    ) -> Dict[str, float]:
        """Predict performance metrics for the selected strategy."""

        if strategy.value not in self._performance_history:
            return {
                "predicted_accuracy": 0.7,
                "predicted_response_time": 2.0,
                "predicted_cost": 0.5
            }

        history = self._performance_history[strategy.value]
        query_complexity = self._calculate_query_complexity(query)

        # Adjust predictions based on query complexity
        complexity_factor = 1 + (query_complexity - 0.5) * 0.2

        return {
            "predicted_accuracy": min(history["accuracy"] * (2 - complexity_factor), 1.0),
            "predicted_response_time": history["avg_response_time"] * complexity_factor,
            "predicted_cost": history["cost_factor"] * complexity_factor
        }

    def _update_statistics(self, selection: StrategySelection):
        """Update selection statistics."""

        # Update strategy distribution
        self._stats["strategy_distribution"][selection.strategy.value] += 1

        # Update confidence distribution
        self._stats["confidence_distribution"][selection.confidence_level.value] += 1

        # Update average confidence
        total_selections = self._stats["total_selections"]
        current_avg = self._stats["average_confidence"]
        new_avg = ((current_avg * (total_selections - 1)) + selection.confidence) / total_selections
        self._stats["average_confidence"] = new_avg

    def update_performance_feedback(
        self,
        strategy: SearchStrategy,
        actual_performance: Dict[str, float]
    ):
        """Update performance history with actual results."""

        if strategy.value not in self._performance_history:
            self._performance_history[strategy.value] = {
                "accuracy": 0.7,
                "avg_response_time": 2.0,
                "cost_factor": 0.5,
                "sample_count": 0
            }

        history = self._performance_history[strategy.value]
        sample_count = history["sample_count"]

        # Update running averages
        for metric in ["accuracy", "avg_response_time", "cost_factor"]:
            if metric in actual_performance:
                current_avg = history[metric]
                new_value = actual_performance[metric]
                updated_avg = ((current_avg * sample_count) + new_value) / (sample_count + 1)
                history[metric] = updated_avg

        history["sample_count"] += 1
        self._stats["adaptive_improvements"] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get selection statistics."""
        return self._stats.copy()

    def get_performance_history(self) -> Dict[str, Any]:
        """Get performance history."""
        return self._performance_history.copy()

    def _initialize_strategy_features(self) -> List[StrategyFeature]:
        """Initialize strategy selection features."""

        return [
            # Three-hop search features
            StrategyFeature(
                name="rfq_reference",
                condition="contains:rfq",
                weight=0.9,
                strategy=SearchStrategy.THREE_HOP_SEARCH,
                confidence_boost=0.95
            ),
            StrategyFeature(
                name="vendor_evaluation",
                condition="contains:vendor",
                weight=0.8,
                strategy=SearchStrategy.THREE_HOP_SEARCH,
                confidence_boost=0.9
            ),

            # Exact match features
            StrategyFeature(
                name="exact_term_query",
                condition="pattern:\"[^\"]+\"",
                weight=0.9,
                strategy=SearchStrategy.EXACT_MATCH,
                confidence_boost=0.95
            ),
            StrategyFeature(
                name="specific_id_query",
                condition="pattern:[A-Z0-9]{6,}",
                weight=0.8,
                strategy=SearchStrategy.EXACT_MATCH,
                confidence_boost=0.9
            ),

            # Semantic search features
            StrategyFeature(
                name="conceptual_query",
                condition="length_gt:5",
                weight=0.7,
                strategy=SearchStrategy.SEMANTIC_SEARCH,
                confidence_boost=0.8
            ),
            StrategyFeature(
                name="natural_language",
                condition="contains:what",
                weight=0.8,
                strategy=SearchStrategy.SEMANTIC_SEARCH,
                confidence_boost=0.85
            ),

            # Filtered search features
            StrategyFeature(
                name="cost_criteria",
                condition="contains:cost",
                weight=0.8,
                strategy=SearchStrategy.FILTERED_SEARCH,
                confidence_boost=0.9
            ),
            StrategyFeature(
                name="compliance_criteria",
                condition="contains:compliance",
                weight=0.8,
                strategy=SearchStrategy.FILTERED_SEARCH,
                confidence_boost=0.9
            ),

            # Priority search features
            StrategyFeature(
                name="urgent_query",
                condition="contains:urgent",
                weight=0.9,
                strategy=SearchStrategy.PRIORITY_SEARCH,
                confidence_boost=0.95
            ),
            StrategyFeature(
                name="asap_query",
                condition="contains:asap",
                weight=0.9,
                strategy=SearchStrategy.PRIORITY_SEARCH,
                confidence_boost=0.95
            ),

            # Comprehensive search features
            StrategyFeature(
                name="complex_query",
                condition="length_gt:15",
                weight=0.7,
                strategy=SearchStrategy.COMPREHENSIVE_SEARCH,
                confidence_boost=0.8
            ),
            StrategyFeature(
                name="analysis_request",
                condition="contains:analysis",
                weight=0.8,
                strategy=SearchStrategy.COMPREHENSIVE_SEARCH,
                confidence_boost=0.85
            )
        ]

    def _initialize_intent_strategy_mapping(self) -> Dict[str, Dict[SearchStrategy, float]]:
        """Initialize intent to strategy mapping."""

        return {
            "requirements_analysis": {
                SearchStrategy.SEMANTIC_SEARCH: 0.8,
                SearchStrategy.COMPREHENSIVE_SEARCH: 0.7,
                SearchStrategy.FILTERED_SEARCH: 0.6
            },
            "vendor_evaluation": {
                SearchStrategy.THREE_HOP_SEARCH: 0.9,
                SearchStrategy.HYBRID_SEARCH: 0.7,
                SearchStrategy.COMPREHENSIVE_SEARCH: 0.6
            },
            "cost_analysis": {
                SearchStrategy.FILTERED_SEARCH: 0.9,
                SearchStrategy.SEMANTIC_SEARCH: 0.7,
                SearchStrategy.HYBRID_SEARCH: 0.6
            },
            "compliance_check": {
                SearchStrategy.EXACT_MATCH: 0.8,
                SearchStrategy.FILTERED_SEARCH: 0.9,
                SearchStrategy.DOCUMENT_SPECIFIC: 0.7
            },
            "contract_review": {
                SearchStrategy.DOCUMENT_SPECIFIC: 0.9,
                SearchStrategy.EXACT_MATCH: 0.8,
                SearchStrategy.SEMANTIC_SEARCH: 0.6
            },
            "technical_assessment": {
                SearchStrategy.SEMANTIC_SEARCH: 0.8,
                SearchStrategy.COMPREHENSIVE_SEARCH: 0.7,
                SearchStrategy.HYBRID_SEARCH: 0.6
            },
            "timeline_inquiry": {
                SearchStrategy.PRIORITY_SEARCH: 0.8,
                SearchStrategy.FILTERED_SEARCH: 0.7,
                SearchStrategy.SEMANTIC_SEARCH: 0.6
            },
            "comparison_request": {
                SearchStrategy.COMPREHENSIVE_SEARCH: 0.9,
                SearchStrategy.HYBRID_SEARCH: 0.8,
                SearchStrategy.SEMANTIC_SEARCH: 0.7
            },
            "status_check": {
                SearchStrategy.PRIORITY_SEARCH: 0.8,
                SearchStrategy.EXACT_MATCH: 0.7,
                SearchStrategy.FILTERED_SEARCH: 0.6
            },
            "general_search": {
                SearchStrategy.BALANCED_SEARCH: 0.8,
                SearchStrategy.SEMANTIC_SEARCH: 0.7,
                SearchStrategy.HYBRID_SEARCH: 0.6
            }
        }

    def _initialize_query_type_mapping(self) -> Dict[str, Dict[SearchStrategy, float]]:
        """Initialize query type to strategy mapping."""

        return {
            "question_search": {
                SearchStrategy.SEMANTIC_SEARCH: 0.8,
                SearchStrategy.COMPREHENSIVE_SEARCH: 0.7
            },
            "question_comparison": {
                SearchStrategy.COMPREHENSIVE_SEARCH: 0.9,
                SearchStrategy.HYBRID_SEARCH: 0.8
            },
            "phrase_search": {
                SearchStrategy.SEMANTIC_SEARCH: 0.8,
                SearchStrategy.BALANCED_SEARCH: 0.7
            },
            "phrase_exact": {
                SearchStrategy.EXACT_MATCH: 0.9,
                SearchStrategy.FILTERED_SEARCH: 0.7
            },
            "keywords_search": {
                SearchStrategy.BALANCED_SEARCH: 0.8,
                SearchStrategy.SEMANTIC_SEARCH: 0.7
            },
            "keywords_filter": {
                SearchStrategy.FILTERED_SEARCH: 0.9,
                SearchStrategy.HYBRID_SEARCH: 0.7
            }
        }

    def _initialize_performance_history(self) -> Dict[str, Dict[str, float]]:
        """Initialize performance history with baseline values."""

        return {
            SearchStrategy.SEMANTIC_SEARCH.value: {
                "accuracy": 0.75,
                "avg_response_time": 1.5,
                "cost_factor": 0.6,
                "sample_count": 100
            },
            SearchStrategy.THREE_HOP_SEARCH.value: {
                "accuracy": 0.85,
                "avg_response_time": 3.0,
                "cost_factor": 0.8,
                "sample_count": 50
            },
            SearchStrategy.FILTERED_SEARCH.value: {
                "accuracy": 0.80,
                "avg_response_time": 1.2,
                "cost_factor": 0.4,
                "sample_count": 75
            },
            SearchStrategy.EXACT_MATCH.value: {
                "accuracy": 0.95,
                "avg_response_time": 0.8,
                "cost_factor": 0.2,
                "sample_count": 200
            },
            SearchStrategy.DOCUMENT_SPECIFIC.value: {
                "accuracy": 0.88,
                "avg_response_time": 1.8,
                "cost_factor": 0.5,
                "sample_count": 60
            },
            SearchStrategy.HYBRID_SEARCH.value: {
                "accuracy": 0.82,
                "avg_response_time": 2.5,
                "cost_factor": 0.7,
                "sample_count": 80
            },
            SearchStrategy.BALANCED_SEARCH.value: {
                "accuracy": 0.78,
                "avg_response_time": 2.0,
                "cost_factor": 0.5,
                "sample_count": 150
            },
            SearchStrategy.PRIORITY_SEARCH.value: {
                "accuracy": 0.70,
                "avg_response_time": 0.5,
                "cost_factor": 0.3,
                "sample_count": 40
            },
            SearchStrategy.COMPREHENSIVE_SEARCH.value: {
                "accuracy": 0.90,
                "avg_response_time": 5.0,
                "cost_factor": 1.0,
                "sample_count": 30
            }
        }


# Singleton instance
_search_strategy_selector: Optional[SearchStrategySelector] = None


def get_search_strategy_selector() -> SearchStrategySelector:
    """Get the singleton search strategy selector instance."""
    global _search_strategy_selector
    if _search_strategy_selector is None:
        _search_strategy_selector = SearchStrategySelector()
    return _search_strategy_selector
