"""
ChromaDB Collection Management Service

This module provides utilities for managing ChromaDB collections including
initialization, validation, metadata management, and collection utilities.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Set
from enum import Enum

from pydantic import BaseModel, Field

from agentic_rag.services.vector_store import ChromaDBClient, get_vector_store
from agentic_rag.models.database import DocumentKind

logger = logging.getLogger(__name__)


class CollectionType(str, Enum):
    """Collection type enumeration."""
    
    RFQ = "rfq"
    OFFER = "offer"


class CollectionInfo(BaseModel):
    """Information about a ChromaDB collection."""
    
    name: str = Field(..., description="Collection name")
    type: CollectionType = Field(..., description="Collection type")
    document_kinds: List[str] = Field(..., description="Supported document kinds")
    vector_count: int = Field(default=0, description="Number of vectors in collection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Collection metadata")
    created_at: Optional[float] = Field(None, description="Creation timestamp")
    last_updated: Optional[float] = Field(None, description="Last update timestamp")
    health_status: str = Field(default="unknown", description="Health status")


class CollectionValidationResult(BaseModel):
    """Result of collection validation."""
    
    collection_name: str = Field(..., description="Collection name")
    is_valid: bool = Field(..., description="Whether collection is valid")
    issues: List[str] = Field(default_factory=list, description="Validation issues found")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for fixes")


class CollectionManager:
    """Manager for ChromaDB collections with validation and utilities."""
    
    def __init__(self):
        self._client: Optional[ChromaDBClient] = None
        
        # Collection configuration
        self._collection_configs = {
            CollectionType.RFQ: {
                "document_kinds": [DocumentKind.RFQ, DocumentKind.RFP, DocumentKind.TENDER],
                "description": "RFQ, RFP, and Tender documents for procurement requests",
                "expected_metadata_fields": [
                    "tenant_id", "document_id", "chunk_id", "section_path",
                    "page_from", "page_to", "document_kind", "created_at"
                ]
            },
            CollectionType.OFFER: {
                "document_kinds": [DocumentKind.OFFER_TECH, DocumentKind.OFFER_COMM, DocumentKind.PRICING],
                "description": "Technical, commercial, and pricing offer documents",
                "expected_metadata_fields": [
                    "tenant_id", "document_id", "chunk_id", "section_path",
                    "page_from", "page_to", "document_kind", "created_at"
                ]
            }
        }
        
        logger.info("Collection manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the collection manager with ChromaDB client."""
        self._client = await get_vector_store()
        logger.info("Collection manager connected to ChromaDB")
    
    def get_collection_type_for_document_kind(self, document_kind: str) -> CollectionType:
        """Get the collection type for a document kind."""
        # Handle different input formats
        kind_mapping = {
            "RFQ": DocumentKind.RFQ,
            "RFP": DocumentKind.RFP,
            "TENDER": DocumentKind.TENDER,
            "OFFERTECH": DocumentKind.OFFER_TECH,
            "OFFER_TECH": DocumentKind.OFFER_TECH,
            "OFFERCOMM": DocumentKind.OFFER_COMM,
            "OFFER_COMM": DocumentKind.OFFER_COMM,
            "PRICING": DocumentKind.PRICING
        }

        document_kind_enum = kind_mapping.get(document_kind.upper())
        if not document_kind_enum:
            raise ValueError(f"Unknown document kind: {document_kind}")

        for collection_type, config in self._collection_configs.items():
            if document_kind_enum in config["document_kinds"]:
                return collection_type

        raise ValueError(f"No collection found for document kind: {document_kind}")
    
    def get_supported_document_kinds(self, collection_type: CollectionType) -> List[str]:
        """Get supported document kinds for a collection type."""
        config = self._collection_configs.get(collection_type)
        if not config:
            return []
        
        return [kind.value for kind in config["document_kinds"]]
    
    async def get_collection_info(self, collection_type: CollectionType) -> CollectionInfo:
        """Get detailed information about a collection."""
        if not self._client:
            await self.initialize()
        
        try:
            # Get collection name
            if collection_type == CollectionType.RFQ:
                collection_name = self._client.rfq_collection_name
                sample_document_kind = "RFQ"
            else:
                collection_name = self._client.offer_collection_name
                sample_document_kind = "OfferTech"
            
            # Get collection stats
            stats = await self._client.get_collection_stats(sample_document_kind)
            
            # Get configuration
            config = self._collection_configs[collection_type]
            
            return CollectionInfo(
                name=collection_name,
                type=collection_type,
                document_kinds=self.get_supported_document_kinds(collection_type),
                vector_count=stats.get("vector_count", 0),
                metadata=stats.get("metadata", {}),
                created_at=stats.get("metadata", {}).get("created_at"),
                last_updated=time.time(),
                health_status="healthy" if "error" not in stats else "unhealthy"
            )
            
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_type}: {e}")
            
            return CollectionInfo(
                name="unknown",
                type=collection_type,
                document_kinds=self.get_supported_document_kinds(collection_type),
                vector_count=0,
                metadata={},
                health_status="unhealthy"
            )
    
    async def validate_collection(self, collection_type: CollectionType) -> CollectionValidationResult:
        """Validate a collection configuration and health."""
        if not self._client:
            await self.initialize()
        
        collection_info = await self.get_collection_info(collection_type)
        
        issues = []
        recommendations = []
        
        # Check if collection exists and is accessible
        if collection_info.health_status == "unhealthy":
            issues.append("Collection is not accessible or healthy")
            recommendations.append("Check ChromaDB service status and connectivity")
        
        # Check vector count
        if collection_info.vector_count == 0:
            issues.append("Collection contains no vectors")
            recommendations.append("Add sample vectors to test collection functionality")
        
        # Check metadata configuration
        config = self._collection_configs[collection_type]
        collection_metadata = collection_info.metadata
        
        if not collection_metadata.get("description"):
            issues.append("Collection missing description metadata")
            recommendations.append(f"Add description: {config['description']}")
        
        # Check document kinds configuration
        expected_kinds = set(self.get_supported_document_kinds(collection_type))
        configured_kinds = set(collection_metadata.get("document_types", []))
        
        if expected_kinds != configured_kinds:
            issues.append(f"Document kinds mismatch: expected {expected_kinds}, got {configured_kinds}")
            recommendations.append("Update collection metadata with correct document types")
        
        is_valid = len(issues) == 0
        
        return CollectionValidationResult(
            collection_name=collection_info.name,
            is_valid=is_valid,
            issues=issues,
            recommendations=recommendations
        )
    
    async def validate_all_collections(self) -> Dict[CollectionType, CollectionValidationResult]:
        """Validate all collections."""
        results = {}
        
        for collection_type in CollectionType:
            results[collection_type] = await self.validate_collection(collection_type)
        
        return results
    
    async def get_collection_summary(self) -> Dict[str, Any]:
        """Get a summary of all collections."""
        if not self._client:
            await self.initialize()
        
        try:
            # Get collection information
            rfq_info = await self.get_collection_info(CollectionType.RFQ)
            offer_info = await self.get_collection_info(CollectionType.OFFER)
            
            # Get validation results
            validations = await self.validate_all_collections()
            
            # Get client statistics
            stats = self._client.get_operation_stats()
            
            return {
                "collections": {
                    "rfq": rfq_info.model_dump(),
                    "offer": offer_info.model_dump()
                },
                "validation": {
                    collection_type.value: result.model_dump()
                    for collection_type, result in validations.items()
                },
                "statistics": stats,
                "total_vectors": rfq_info.vector_count + offer_info.vector_count,
                "healthy_collections": sum(
                    1 for info in [rfq_info, offer_info]
                    if info.health_status == "healthy"
                ),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection summary: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def repair_collection_metadata(self, collection_type: CollectionType) -> bool:
        """Repair collection metadata if possible."""
        try:
            # This would require ChromaDB API support for metadata updates
            # For now, log the repair attempt
            logger.info(f"Collection metadata repair requested for {collection_type}")
            logger.warning("Collection metadata repair not implemented - requires ChromaDB API support")
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to repair collection metadata: {e}")
            return False
    
    def get_collection_config(self, collection_type: CollectionType) -> Dict[str, Any]:
        """Get configuration for a collection type."""
        return self._collection_configs.get(collection_type, {}).copy()


# Global collection manager instance
_collection_manager: Optional[CollectionManager] = None


async def get_collection_manager() -> CollectionManager:
    """Get or create the global collection manager instance."""
    global _collection_manager
    
    if _collection_manager is None:
        _collection_manager = CollectionManager()
        await _collection_manager.initialize()
    
    return _collection_manager
