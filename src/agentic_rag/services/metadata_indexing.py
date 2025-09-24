"""
Metadata Indexing Service

This module provides efficient metadata indexing and filtering capabilities
for vector storage operations.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from agentic_rag.services.vector_store import VectorMetadata
from agentic_rag.services.metadata_validator import validate_chunk_metadata, MetadataValidationLevel

logger = logging.getLogger(__name__)


class IndexingStrategy(str, Enum):
    """Strategies for metadata indexing."""
    
    FULL = "full"          # Index all metadata fields
    SELECTIVE = "selective" # Index only specified fields
    MINIMAL = "minimal"     # Index only essential fields


class FilterOperator(str, Enum):
    """Operators for metadata filtering."""
    
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


@dataclass
class MetadataFilter:
    """Filter specification for metadata queries."""
    
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = True


class MetadataQuery(BaseModel):
    """Query specification for metadata filtering."""
    
    filters: List[MetadataFilter] = Field(default_factory=list, description="List of metadata filters")
    logical_operator: str = Field(default="AND", description="Logical operator between filters (AND/OR)")
    limit: Optional[int] = Field(None, description="Maximum number of results")
    offset: int = Field(default=0, description="Offset for pagination")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field(default="asc", description="Sort order (asc/desc)")


class MetadataIndexingResult(BaseModel):
    """Result of metadata indexing operation."""
    
    indexed_fields: List[str] = Field(default_factory=list, description="Successfully indexed fields")
    failed_fields: List[str] = Field(default_factory=list, description="Failed to index fields")
    total_fields: int = Field(default=0, description="Total fields processed")
    indexing_time: float = Field(default=0.0, description="Time taken for indexing")
    errors: List[str] = Field(default_factory=list, description="Indexing errors")
    warnings: List[str] = Field(default_factory=list, description="Indexing warnings")


class MetadataIndexer:
    """Service for indexing and filtering metadata efficiently."""
    
    def __init__(self, strategy: IndexingStrategy = IndexingStrategy.SELECTIVE):
        self.strategy = strategy
        self._indexed_fields: Set[str] = set()
        self._field_types: Dict[str, str] = {}
        self._field_cardinality: Dict[str, int] = {}
        
        # Essential fields that should always be indexed
        self._essential_fields = {
            "tenant_id", "document_id", "chunk_id", "document_kind", 
            "document_type", "is_table", "is_duplicate", "created_at"
        }
        
        # High-cardinality fields that may need special handling
        self._high_cardinality_fields = {
            "chunk_id", "document_id", "created_at", "updated_at"
        }
        
        logger.info(f"Metadata indexer initialized with {strategy.value} strategy")
    
    async def prepare_metadata_for_indexing(
        self,
        metadata: Dict[str, Any],
        validate: bool = True
    ) -> MetadataIndexingResult:
        """
        Prepare metadata for efficient indexing.
        
        Args:
            metadata: Raw metadata dictionary
            validate: Whether to validate metadata before indexing
            
        Returns:
            MetadataIndexingResult with indexing outcome
        """
        import time
        start_time = time.time()
        
        result = MetadataIndexingResult(total_fields=len(metadata))
        
        try:
            # Validate metadata if requested
            if validate:
                validation_result = await validate_chunk_metadata(
                    metadata, 
                    MetadataValidationLevel.MODERATE
                )
                
                if not validation_result.is_valid:
                    result.errors.extend(validation_result.errors)
                    result.warnings.extend(validation_result.warnings)
                    
                    # Use validated metadata even if not fully valid
                    metadata = validation_result.validated_metadata
                
                result.warnings.extend(validation_result.warnings)
            
            # Determine fields to index based on strategy
            fields_to_index = self._select_fields_for_indexing(metadata)
            
            # Process each field for indexing
            for field_name in fields_to_index:
                if field_name in metadata:
                    try:
                        # Normalize field value for indexing
                        normalized_value = await self._normalize_field_value(
                            field_name, 
                            metadata[field_name]
                        )
                        
                        # Update field tracking
                        self._indexed_fields.add(field_name)
                        self._field_types[field_name] = type(normalized_value).__name__
                        
                        # Update cardinality estimate
                        if field_name not in self._field_cardinality:
                            self._field_cardinality[field_name] = 0
                        self._field_cardinality[field_name] += 1
                        
                        result.indexed_fields.append(field_name)
                        
                    except Exception as e:
                        logger.error(f"Failed to index field {field_name}: {e}")
                        result.failed_fields.append(field_name)
                        result.errors.append(f"Field {field_name}: {str(e)}")
                else:
                    result.warnings.append(f"Field {field_name} selected for indexing but not present in metadata")
            
            result.indexing_time = time.time() - start_time
            
            logger.debug(f"Prepared metadata for indexing: {len(result.indexed_fields)} fields in {result.indexing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Metadata indexing preparation failed: {e}")
            result.errors.append(f"Indexing preparation error: {str(e)}")
            result.indexing_time = time.time() - start_time
        
        return result
    
    def _select_fields_for_indexing(self, metadata: Dict[str, Any]) -> Set[str]:
        """Select which fields to index based on strategy."""
        if self.strategy == IndexingStrategy.FULL:
            return set(metadata.keys())
        
        elif self.strategy == IndexingStrategy.MINIMAL:
            return self._essential_fields.intersection(set(metadata.keys()))
        
        else:  # SELECTIVE
            # Start with essential fields
            selected_fields = self._essential_fields.intersection(set(metadata.keys()))
            
            # Add fields that are good for filtering
            filterable_fields = {
                "document_name", "chunk_index", "token_count", "quality_score",
                "page_from", "page_to", "section_path", "similarity_score"
            }
            
            selected_fields.update(filterable_fields.intersection(set(metadata.keys())))
            
            return selected_fields
    
    async def _normalize_field_value(self, field_name: str, value: Any) -> Any:
        """Normalize field value for consistent indexing."""
        try:
            # Handle None values
            if value is None:
                return None
            
            # Normalize strings
            if isinstance(value, str):
                # Trim whitespace
                value = value.strip()
                
                # Handle empty strings
                if not value:
                    return None
                
                # Normalize case for certain fields
                if field_name in {"document_kind", "document_type"}:
                    value = value.upper()
                
                # Limit string length for indexing efficiency
                if len(value) > 1000:
                    value = value[:1000]
                    logger.warning(f"Truncated long string value for field {field_name}")
            
            # Normalize lists
            elif isinstance(value, list):
                # Remove empty items and normalize
                value = [item for item in value if item is not None and str(item).strip()]
                
                # Limit list size
                if len(value) > 100:
                    value = value[:100]
                    logger.warning(f"Truncated long list for field {field_name}")
            
            # Normalize timestamps
            elif field_name in {"created_at", "updated_at"} and isinstance(value, str):
                try:
                    # Ensure ISO format
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    value = dt.isoformat()
                except ValueError:
                    logger.warning(f"Invalid timestamp format for {field_name}: {value}")
                    value = datetime.utcnow().isoformat()
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to normalize field {field_name}: {e}")
            return value
    
    def build_chromadb_filter(self, query: MetadataQuery) -> Dict[str, Any]:
        """
        Build ChromaDB-compatible filter from metadata query.
        
        Args:
            query: Metadata query specification
            
        Returns:
            ChromaDB filter dictionary
        """
        if not query.filters:
            return {}
        
        try:
            # Build filter conditions
            conditions = []
            
            for filter_spec in query.filters:
                condition = self._build_filter_condition(filter_spec)
                if condition:
                    conditions.append(condition)
            
            if not conditions:
                return {}
            
            # Combine conditions with logical operator
            if len(conditions) == 1:
                return conditions[0]
            
            if query.logical_operator.upper() == "OR":
                return {"$or": conditions}
            else:  # Default to AND
                return {"$and": conditions}
                
        except Exception as e:
            logger.error(f"Failed to build ChromaDB filter: {e}")
            return {}
    
    def _build_filter_condition(self, filter_spec: MetadataFilter) -> Optional[Dict[str, Any]]:
        """Build a single filter condition for ChromaDB."""
        try:
            field = filter_spec.field
            operator = filter_spec.operator
            value = filter_spec.value
            
            # Handle case sensitivity for string operations
            if isinstance(value, str) and not filter_spec.case_sensitive:
                value = value.lower()
                # Note: ChromaDB doesn't have built-in case-insensitive search,
                # so we'd need to store normalized versions
            
            # Map operators to ChromaDB format
            if operator == FilterOperator.EQUALS:
                return {field: {"$eq": value}}
            
            elif operator == FilterOperator.NOT_EQUALS:
                return {field: {"$ne": value}}
            
            elif operator == FilterOperator.GREATER_THAN:
                return {field: {"$gt": value}}
            
            elif operator == FilterOperator.GREATER_EQUAL:
                return {field: {"$gte": value}}
            
            elif operator == FilterOperator.LESS_THAN:
                return {field: {"$lt": value}}
            
            elif operator == FilterOperator.LESS_EQUAL:
                return {field: {"$lte": value}}
            
            elif operator == FilterOperator.IN:
                if isinstance(value, list):
                    return {field: {"$in": value}}
                else:
                    return {field: {"$eq": value}}
            
            elif operator == FilterOperator.NOT_IN:
                if isinstance(value, list):
                    return {field: {"$nin": value}}
                else:
                    return {field: {"$ne": value}}
            
            elif operator == FilterOperator.CONTAINS:
                # ChromaDB doesn't have native contains, use regex-like approach
                return {field: {"$eq": value}}  # Simplified for now
            
            elif operator == FilterOperator.STARTS_WITH:
                # ChromaDB doesn't have native starts_with, use regex-like approach
                return {field: {"$eq": value}}  # Simplified for now
            
            elif operator == FilterOperator.ENDS_WITH:
                # ChromaDB doesn't have native ends_with, use regex-like approach
                return {field: {"$eq": value}}  # Simplified for now
            
            else:
                logger.warning(f"Unsupported filter operator: {operator}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to build filter condition for {filter_spec.field}: {e}")
            return None
    
    def get_indexing_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed fields."""
        return {
            "strategy": self.strategy.value,
            "indexed_fields_count": len(self._indexed_fields),
            "indexed_fields": list(self._indexed_fields),
            "field_types": self._field_types.copy(),
            "field_cardinality": self._field_cardinality.copy(),
            "essential_fields": list(self._essential_fields),
            "high_cardinality_fields": list(self._high_cardinality_fields)
        }
    
    def optimize_indexing_strategy(self) -> Dict[str, Any]:
        """Analyze and suggest indexing optimizations."""
        recommendations = []
        
        # Check for high-cardinality fields
        high_cardinality = {
            field: count for field, count in self._field_cardinality.items()
            if count > 10000  # Threshold for high cardinality
        }
        
        if high_cardinality:
            recommendations.append({
                "type": "high_cardinality",
                "message": f"High cardinality fields detected: {list(high_cardinality.keys())}",
                "suggestion": "Consider using hash-based indexing or range partitioning"
            })
        
        # Check for unused indexed fields
        if len(self._indexed_fields) > 20:
            recommendations.append({
                "type": "too_many_fields",
                "message": f"Large number of indexed fields: {len(self._indexed_fields)}",
                "suggestion": "Consider switching to SELECTIVE or MINIMAL indexing strategy"
            })
        
        # Check for missing essential fields
        missing_essential = self._essential_fields - self._indexed_fields
        if missing_essential:
            recommendations.append({
                "type": "missing_essential",
                "message": f"Missing essential fields: {list(missing_essential)}",
                "suggestion": "Ensure essential fields are included in metadata"
            })
        
        return {
            "current_strategy": self.strategy.value,
            "indexed_fields_count": len(self._indexed_fields),
            "recommendations": recommendations,
            "optimization_score": self._calculate_optimization_score()
        }
    
    def _calculate_optimization_score(self) -> float:
        """Calculate optimization score (0-1, higher is better)."""
        score = 1.0
        
        # Penalize for too many fields
        if len(self._indexed_fields) > 15:
            score -= 0.2
        
        # Penalize for missing essential fields
        missing_essential = len(self._essential_fields - self._indexed_fields)
        score -= missing_essential * 0.1
        
        # Penalize for high cardinality fields
        high_cardinality_count = sum(
            1 for count in self._field_cardinality.values() if count > 10000
        )
        score -= high_cardinality_count * 0.1
        
        return max(0.0, min(1.0, score))


# Global indexer instance
_metadata_indexer: Optional[MetadataIndexer] = None


async def get_metadata_indexer(strategy: IndexingStrategy = IndexingStrategy.SELECTIVE) -> MetadataIndexer:
    """Get or create the global metadata indexer instance."""
    global _metadata_indexer
    
    if _metadata_indexer is None:
        _metadata_indexer = MetadataIndexer(strategy)
    
    return _metadata_indexer


async def prepare_metadata_for_vector_storage(
    metadata: Dict[str, Any],
    validate: bool = True,
    strategy: IndexingStrategy = IndexingStrategy.SELECTIVE
) -> MetadataIndexingResult:
    """
    Convenience function to prepare metadata for vector storage.
    
    Args:
        metadata: Raw metadata dictionary
        validate: Whether to validate metadata
        strategy: Indexing strategy to use
        
    Returns:
        MetadataIndexingResult with preparation outcome
    """
    indexer = await get_metadata_indexer(strategy)
    return await indexer.prepare_metadata_for_indexing(metadata, validate)
