"""
Search Configuration Service for Flexible Search Parameters.

This module provides a comprehensive configuration system for search parameters
including similarity thresholds, result counts, search modes, and parameter validation.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)


class SearchMode(str, Enum):
    """Search mode options."""
    STRICT = "strict"
    FUZZY = "fuzzy"
    HYBRID = "hybrid"
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"


class RankingStrategy(str, Enum):
    """Ranking strategy options."""
    SIMILARITY = "similarity"
    RECENCY = "recency"
    POPULARITY = "popularity"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class FilterMode(str, Enum):
    """Filter application modes."""
    STRICT = "strict"
    LENIENT = "lenient"
    ADAPTIVE = "adaptive"


@dataclass
class SimilarityThresholds:
    """Similarity threshold configuration."""
    minimum: float = 0.3
    good: float = 0.7
    excellent: float = 0.9
    adaptive: bool = True
    
    def __post_init__(self):
        """Validate thresholds."""
        if not (0.0 <= self.minimum <= self.good <= self.excellent <= 1.0):
            raise ValueError("Thresholds must be in ascending order between 0.0 and 1.0")


@dataclass
class ResultLimits:
    """Result count and pagination limits."""
    min_results: int = 1
    max_results: int = 100
    default_results: int = 20
    max_page_size: int = 50
    default_page_size: int = 20
    
    def __post_init__(self):
        """Validate limits."""
        if not (1 <= self.min_results <= self.default_results <= self.max_results):
            raise ValueError("Result limits must be in ascending order")
        if not (1 <= self.default_page_size <= self.max_page_size <= self.max_results):
            raise ValueError("Page size limits must be valid")


@dataclass
class RankingWeights:
    """Weights for different ranking factors."""
    semantic_similarity: float = 0.4
    recency: float = 0.2
    document_type: float = 0.15
    section_relevance: float = 0.15
    user_interactions: float = 0.1
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (self.semantic_similarity + self.recency + self.document_type + 
                self.section_relevance + self.user_interactions)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ranking weights must sum to 1.0, got {total}")


@dataclass
class PerformanceSettings:
    """Performance-related search settings."""
    query_timeout_ms: int = 30000
    vector_search_timeout_ms: int = 20000
    ranking_timeout_ms: int = 5000
    max_concurrent_searches: int = 100
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_parallel_processing: bool = True
    
    def __post_init__(self):
        """Validate performance settings."""
        if self.query_timeout_ms <= 0 or self.vector_search_timeout_ms <= 0:
            raise ValueError("Timeout values must be positive")
        if self.max_concurrent_searches <= 0:
            raise ValueError("Max concurrent searches must be positive")


class SearchConfiguration(BaseModel):
    """Comprehensive search configuration."""
    
    # Basic search settings
    search_mode: SearchMode = SearchMode.HYBRID
    ranking_strategy: RankingStrategy = RankingStrategy.HYBRID
    filter_mode: FilterMode = FilterMode.ADAPTIVE
    
    # Similarity and thresholds
    similarity_thresholds: SimilarityThresholds = Field(default_factory=SimilarityThresholds)
    
    # Result limits
    result_limits: ResultLimits = Field(default_factory=ResultLimits)
    
    # Ranking configuration
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)
    
    # Performance settings
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    # Query processing options
    enable_spell_check: bool = True
    enable_query_expansion: bool = True
    enable_synonym_expansion: bool = True
    enable_stop_word_removal: bool = True
    enable_stemming: bool = False
    
    # Advanced options
    boost_exact_matches: bool = True
    boost_title_matches: bool = True
    boost_recent_documents: bool = True
    enable_diversity_filtering: bool = True
    
    # Custom parameters
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
    
    @validator('similarity_thresholds')
    def validate_similarity_thresholds(cls, v):
        """Validate similarity thresholds."""
        if isinstance(v, dict):
            return SimilarityThresholds(**v)
        return v
    
    @validator('result_limits')
    def validate_result_limits(cls, v):
        """Validate result limits."""
        if isinstance(v, dict):
            return ResultLimits(**v)
        return v
    
    @validator('ranking_weights')
    def validate_ranking_weights(cls, v):
        """Validate ranking weights."""
        if isinstance(v, dict):
            return RankingWeights(**v)
        return v
    
    @validator('performance')
    def validate_performance(cls, v):
        """Validate performance settings."""
        if isinstance(v, dict):
            return PerformanceSettings(**v)
        return v


class SearchConfigurationManager:
    """Manager for search configurations with validation and presets."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self._configurations: Dict[str, SearchConfiguration] = {}
        self._default_config = self._create_default_configuration()
        self._presets = self._create_preset_configurations()
        
        logger.info("Search configuration manager initialized")
    
    def _create_default_configuration(self) -> SearchConfiguration:
        """Create the default search configuration."""
        return SearchConfiguration()
    
    def _create_preset_configurations(self) -> Dict[str, SearchConfiguration]:
        """Create preset configurations for common use cases."""
        presets = {}
        
        # High precision configuration
        presets["high_precision"] = SearchConfiguration(
            search_mode=SearchMode.STRICT,
            similarity_thresholds=SimilarityThresholds(
                minimum=0.8,
                good=0.9,
                excellent=0.95
            ),
            result_limits=ResultLimits(
                default_results=10,
                max_results=25,
                max_page_size=25,
                default_page_size=10
            ),
            ranking_weights=RankingWeights(
                semantic_similarity=0.7,
                recency=0.1,
                document_type=0.1,
                section_relevance=0.1,
                user_interactions=0.0
            )
        )
        
        # High recall configuration
        presets["high_recall"] = SearchConfiguration(
            search_mode=SearchMode.FUZZY,
            similarity_thresholds=SimilarityThresholds(
                minimum=0.3,
                good=0.5,
                excellent=0.7
            ),
            result_limits=ResultLimits(
                default_results=50,
                max_results=100,
                max_page_size=50,
                default_page_size=20
            ),
            enable_query_expansion=True,
            enable_synonym_expansion=True
        )
        
        # Fast search configuration
        presets["fast_search"] = SearchConfiguration(
            search_mode=SearchMode.SEMANTIC_ONLY,
            result_limits=ResultLimits(
                default_results=15,
                max_results=30,
                max_page_size=30,
                default_page_size=15
            ),
            performance=PerformanceSettings(
                query_timeout_ms=10000,
                vector_search_timeout_ms=5000,
                ranking_timeout_ms=2000,
                enable_parallel_processing=True
            ),
            enable_spell_check=False,
            enable_query_expansion=False
        )
        
        # Comprehensive search configuration
        presets["comprehensive"] = SearchConfiguration(
            search_mode=SearchMode.HYBRID,
            ranking_strategy=RankingStrategy.HYBRID,
            enable_spell_check=True,
            enable_query_expansion=True,
            enable_synonym_expansion=True,
            boost_exact_matches=True,
            boost_title_matches=True,
            boost_recent_documents=True,
            enable_diversity_filtering=True
        )
        
        return presets
    
    def get_configuration(self, name: Optional[str] = None) -> SearchConfiguration:
        """
        Get a search configuration by name.
        
        Args:
            name: Configuration name (None for default)
            
        Returns:
            SearchConfiguration instance
        """
        if name is None:
            return self._default_config.copy(deep=True)
        
        if name in self._presets:
            return self._presets[name].copy(deep=True)
        
        if name in self._configurations:
            return self._configurations[name].copy(deep=True)
        
        raise ValueError(f"Configuration '{name}' not found")
    
    def register_configuration(self, name: str, config: SearchConfiguration) -> None:
        """
        Register a custom configuration.
        
        Args:
            name: Configuration name
            config: SearchConfiguration instance
        """
        self.validate_configuration(config)
        self._configurations[name] = config.copy(deep=True)
        logger.info(f"Registered search configuration: {name}")
    
    def validate_configuration(self, config: SearchConfiguration) -> bool:
        """
        Validate a search configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Pydantic validation happens automatically
            config.dict()
            return True
        except Exception as e:
            raise ValueError(f"Invalid search configuration: {e}")
    
    def get_preset_names(self) -> List[str]:
        """Get list of available preset names."""
        return list(self._presets.keys())
    
    def get_configuration_names(self) -> List[str]:
        """Get list of all configuration names."""
        return list(self._configurations.keys())
    
    def create_configuration_from_dict(self, config_dict: Dict[str, Any]) -> SearchConfiguration:
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SearchConfiguration instance
        """
        return SearchConfiguration(**config_dict)


# Singleton instance
_config_manager_instance: Optional[SearchConfigurationManager] = None


def get_search_configuration_manager() -> SearchConfigurationManager:
    """Get the search configuration manager instance."""
    global _config_manager_instance
    
    if _config_manager_instance is None:
        _config_manager_instance = SearchConfigurationManager()
    
    return _config_manager_instance


def reset_search_configuration_manager():
    """Reset the configuration manager instance (for testing)."""
    global _config_manager_instance
    _config_manager_instance = None
