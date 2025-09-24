"""
ChromaDB Vector Store Service

This module provides a comprehensive interface to ChromaDB for vector storage and retrieval
operations with multi-tenant isolation and performance optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID, uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection
from chromadb.api.types import (
    Documents, Embeddings, IDs, Metadatas, QueryResult, GetResult
)
from pydantic import BaseModel, Field

from agentic_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorMetadata(BaseModel):
    """Metadata for vector storage."""
    
    tenant_id: str = Field(..., description="Tenant identifier")
    document_id: str = Field(..., description="Document identifier")
    chunk_id: str = Field(..., description="Chunk identifier")
    section_path: List[str] = Field(default_factory=list, description="Document section path")
    page_from: Optional[int] = Field(None, description="Starting page number")
    page_to: Optional[int] = Field(None, description="Ending page number")
    token_count: Optional[int] = Field(None, description="Number of tokens in chunk")
    is_table: bool = Field(default=False, description="Whether chunk contains table data")
    document_kind: str = Field(..., description="Document type (RFQ, OfferTech, etc.)")
    created_at: str = Field(..., description="Creation timestamp")
    embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model used")


class VectorSearchResult(BaseModel):
    """Result from vector search operation."""
    
    id: str = Field(..., description="Vector ID")
    document: str = Field(..., description="Document text content")
    metadata: VectorMetadata = Field(..., description="Vector metadata")
    distance: float = Field(..., description="Distance/similarity score")


class VectorOperationResult(BaseModel):
    """Result from vector operation."""
    
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation type")
    count: int = Field(default=0, description="Number of vectors affected")
    duration: float = Field(..., description="Operation duration in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class ChromaDBClient:
    """Async ChromaDB client with multi-tenant support and performance optimization."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[chromadb.Client] = None
        self._collections: Dict[str, Collection] = {}
        
        # Collection names
        self.rfq_collection_name = settings.vector_db.chromadb_rfq_collection
        self.offer_collection_name = settings.vector_db.chromadb_offer_collection
        
        # Performance settings
        self._batch_size = 100
        self._max_retries = 3
        self._retry_delay = 1.0
        
        # Statistics
        self._operation_stats = {
            "add_operations": 0,
            "query_operations": 0,
            "delete_operations": 0,
            "errors": 0,
            "total_vectors": 0,
            "last_health_check": None
        }
        
        logger.info("ChromaDB client initialized")
    
    async def start(self) -> None:
        """Initialize ChromaDB client and collections."""
        try:
            # Create ChromaDB client
            if self.settings.vector_db.chromadb_url:
                # Use HTTP client for remote ChromaDB
                self._client = chromadb.HttpClient(
                    host=self.settings.vector_db.chromadb_url.host,
                    port=self.settings.vector_db.chromadb_url.port or 8000,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=False
                    )
                )
            else:
                # Use local client
                self._client = chromadb.Client(
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=False
                    )
                )
            
            # Initialize collections
            await self._initialize_collections()
            
            logger.info("ChromaDB client started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start ChromaDB client: {e}")
            raise
    
    async def stop(self) -> None:
        """Clean up ChromaDB client resources."""
        try:
            self._collections.clear()
            self._client = None
            logger.info("ChromaDB client stopped")
        except Exception as e:
            logger.error(f"Error stopping ChromaDB client: {e}")
    
    async def _initialize_collections(self) -> None:
        """Initialize ChromaDB collections for RFQ and Offer documents."""
        if not self._client:
            raise RuntimeError("ChromaDB client not initialized")
        
        try:
            # Create or get RFQ collection
            try:
                rfq_collection = self._client.get_collection(self.rfq_collection_name)
                logger.info(f"Found existing RFQ collection: {self.rfq_collection_name}")
            except Exception:
                rfq_collection = self._client.create_collection(
                    name=self.rfq_collection_name,
                    metadata={
                        "description": "RFQ, RFP, and Tender documents",
                        "document_types": ["RFQ", "RFP", "Tender"],
                        "created_at": time.time()
                    }
                )
                logger.info(f"Created RFQ collection: {self.rfq_collection_name}")
            
            self._collections[self.rfq_collection_name] = rfq_collection
            
            # Create or get Offer collection
            try:
                offer_collection = self._client.get_collection(self.offer_collection_name)
                logger.info(f"Found existing Offer collection: {self.offer_collection_name}")
            except Exception:
                offer_collection = self._client.create_collection(
                    name=self.offer_collection_name,
                    metadata={
                        "description": "Offer technical, commercial, and pricing documents",
                        "document_types": ["OfferTech", "OfferComm", "Pricing"],
                        "created_at": time.time()
                    }
                )
                logger.info(f"Created Offer collection: {self.offer_collection_name}")
            
            self._collections[self.offer_collection_name] = offer_collection
            
            logger.info("All collections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize collections: {e}")
            raise
    
    def _get_collection_for_document_kind(self, document_kind: str) -> Collection:
        """Get the appropriate collection for a document kind."""
        if document_kind.upper() in ["RFQ", "RFP", "TENDER"]:
            collection = self._collections.get(self.rfq_collection_name)
        elif document_kind.upper() in ["OFFERTECH", "OFFERCOMM", "PRICING"]:
            collection = self._collections.get(self.offer_collection_name)
        else:
            raise ValueError(f"Unknown document kind: {document_kind}")
        
        if not collection:
            raise RuntimeError(f"Collection not initialized for document kind: {document_kind}")
        
        return collection
    
    async def add_vectors(
        self,
        vectors: List[Tuple[str, List[float], str, VectorMetadata]]
    ) -> VectorOperationResult:
        """
        Add vectors to ChromaDB with metadata.
        
        Args:
            vectors: List of (id, embedding, document, metadata) tuples
            
        Returns:
            VectorOperationResult with operation details
        """
        start_time = time.time()
        
        try:
            if not vectors:
                return VectorOperationResult(
                    success=True,
                    operation="add_vectors",
                    count=0,
                    duration=time.time() - start_time
                )
            
            # Group vectors by collection
            rfq_vectors = []
            offer_vectors = []
            
            for vector_id, embedding, document, metadata in vectors:
                vector_data = (vector_id, embedding, document, metadata.model_dump())
                
                if metadata.document_kind.upper() in ["RFQ", "RFP", "TENDER"]:
                    rfq_vectors.append(vector_data)
                elif metadata.document_kind.upper() in ["OFFERTECH", "OFFERCOMM", "PRICING"]:
                    offer_vectors.append(vector_data)
                else:
                    logger.warning(f"Unknown document kind: {metadata.document_kind}")
            
            total_added = 0
            
            # Add RFQ vectors
            if rfq_vectors:
                total_added += await self._add_vectors_to_collection(
                    self._collections[self.rfq_collection_name],
                    rfq_vectors
                )
            
            # Add Offer vectors
            if offer_vectors:
                total_added += await self._add_vectors_to_collection(
                    self._collections[self.offer_collection_name],
                    offer_vectors
                )
            
            self._operation_stats["add_operations"] += 1
            self._operation_stats["total_vectors"] += total_added
            
            duration = time.time() - start_time
            logger.info(f"Added {total_added} vectors in {duration:.2f}s")
            
            return VectorOperationResult(
                success=True,
                operation="add_vectors",
                count=total_added,
                duration=duration
            )
            
        except Exception as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Failed to add vectors: {e}")
            
            return VectorOperationResult(
                success=False,
                operation="add_vectors",
                count=0,
                duration=time.time() - start_time,
                error=str(e)
            )
    
    async def _add_vectors_to_collection(
        self,
        collection: Collection,
        vectors: List[Tuple[str, List[float], str, Dict]]
    ) -> int:
        """Add vectors to a specific collection in batches."""
        total_added = 0
        
        # Process in batches
        for i in range(0, len(vectors), self._batch_size):
            batch = vectors[i:i + self._batch_size]
            
            ids = [v[0] for v in batch]
            embeddings = [v[1] for v in batch]
            documents = [v[2] for v in batch]
            metadatas = [v[3] for v in batch]
            
            # Add batch to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            total_added += len(batch)
            logger.debug(f"Added batch of {len(batch)} vectors to {collection.name}")
        
        return total_added

    async def query_vectors(
        self,
        query_embedding: List[float],
        document_kind: str,
        tenant_id: str,
        n_results: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Query vectors from ChromaDB with tenant isolation.

        Args:
            query_embedding: Query vector embedding
            document_kind: Document type to search in
            tenant_id: Tenant identifier for isolation
            n_results: Number of results to return
            where_filter: Additional metadata filters

        Returns:
            List of VectorSearchResult objects
        """
        start_time = time.time()

        try:
            collection = self._get_collection_for_document_kind(document_kind)

            # Build where filter with tenant isolation
            where = {"tenant_id": tenant_id}
            if where_filter:
                where.update(where_filter)

            # Query collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            # Convert to VectorSearchResult objects
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, vector_id in enumerate(results["ids"][0]):
                    metadata_dict = results["metadatas"][0][i] if results["metadatas"] else {}

                    search_result = VectorSearchResult(
                        id=vector_id,
                        document=results["documents"][0][i] if results["documents"] else "",
                        metadata=VectorMetadata(**metadata_dict),
                        distance=results["distances"][0][i] if results["distances"] else 0.0
                    )
                    search_results.append(search_result)

            self._operation_stats["query_operations"] += 1

            duration = time.time() - start_time
            logger.info(f"Queried {len(search_results)} vectors in {duration:.2f}s")

            return search_results

        except Exception as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Failed to query vectors: {e}")
            raise

    async def delete_vectors(
        self,
        vector_ids: List[str],
        document_kind: str,
        tenant_id: str
    ) -> VectorOperationResult:
        """
        Delete vectors from ChromaDB with tenant isolation.

        Args:
            vector_ids: List of vector IDs to delete
            document_kind: Document type
            tenant_id: Tenant identifier for isolation

        Returns:
            VectorOperationResult with operation details
        """
        start_time = time.time()

        try:
            if not vector_ids:
                return VectorOperationResult(
                    success=True,
                    operation="delete_vectors",
                    count=0,
                    duration=time.time() - start_time
                )

            collection = self._get_collection_for_document_kind(document_kind)

            # Verify vectors belong to tenant before deletion
            existing_vectors = collection.get(
                ids=vector_ids,
                where={"tenant_id": tenant_id},
                include=["metadatas"]
            )

            verified_ids = existing_vectors["ids"] if existing_vectors["ids"] else []

            if verified_ids:
                collection.delete(ids=verified_ids)

            self._operation_stats["delete_operations"] += 1
            self._operation_stats["total_vectors"] -= len(verified_ids)

            duration = time.time() - start_time
            logger.info(f"Deleted {len(verified_ids)} vectors in {duration:.2f}s")

            return VectorOperationResult(
                success=True,
                operation="delete_vectors",
                count=len(verified_ids),
                duration=duration
            )

        except Exception as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Failed to delete vectors: {e}")

            return VectorOperationResult(
                success=False,
                operation="delete_vectors",
                count=0,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def update_vectors(
        self,
        vectors: List[Tuple[str, List[float], str, VectorMetadata]]
    ) -> VectorOperationResult:
        """
        Update vectors in ChromaDB.

        Args:
            vectors: List of (id, embedding, document, metadata) tuples

        Returns:
            VectorOperationResult with operation details
        """
        start_time = time.time()

        try:
            if not vectors:
                return VectorOperationResult(
                    success=True,
                    operation="update_vectors",
                    count=0,
                    duration=time.time() - start_time
                )

            # Group vectors by collection
            rfq_vectors = []
            offer_vectors = []

            for vector_id, embedding, document, metadata in vectors:
                vector_data = (vector_id, embedding, document, metadata.model_dump())

                if metadata.document_kind.upper() in ["RFQ", "RFP", "TENDER"]:
                    rfq_vectors.append(vector_data)
                elif metadata.document_kind.upper() in ["OFFERTECH", "OFFERCOMM", "PRICING"]:
                    offer_vectors.append(vector_data)

            total_updated = 0

            # Update RFQ vectors
            if rfq_vectors:
                total_updated += await self._update_vectors_in_collection(
                    self._collections[self.rfq_collection_name],
                    rfq_vectors
                )

            # Update Offer vectors
            if offer_vectors:
                total_updated += await self._update_vectors_in_collection(
                    self._collections[self.offer_collection_name],
                    offer_vectors
                )

            duration = time.time() - start_time
            logger.info(f"Updated {total_updated} vectors in {duration:.2f}s")

            return VectorOperationResult(
                success=True,
                operation="update_vectors",
                count=total_updated,
                duration=duration
            )

        except Exception as e:
            self._operation_stats["errors"] += 1
            logger.error(f"Failed to update vectors: {e}")

            return VectorOperationResult(
                success=False,
                operation="update_vectors",
                count=0,
                duration=time.time() - start_time,
                error=str(e)
            )

    async def _update_vectors_in_collection(
        self,
        collection: Collection,
        vectors: List[Tuple[str, List[float], str, Dict]]
    ) -> int:
        """Update vectors in a specific collection."""
        total_updated = 0

        # Process in batches
        for i in range(0, len(vectors), self._batch_size):
            batch = vectors[i:i + self._batch_size]

            ids = [v[0] for v in batch]
            embeddings = [v[1] for v in batch]
            documents = [v[2] for v in batch]
            metadatas = [v[3] for v in batch]

            # Update batch in collection
            collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            total_updated += len(batch)
            logger.debug(f"Updated batch of {len(batch)} vectors in {collection.name}")

        return total_updated

    async def get_collection_stats(self, document_kind: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        try:
            collection = self._get_collection_for_document_kind(document_kind)
            count = collection.count()

            return {
                "collection_name": collection.name,
                "document_kind": document_kind,
                "vector_count": count,
                "metadata": collection.metadata
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "collection_name": "unknown",
                "document_kind": document_kind,
                "vector_count": 0,
                "error": str(e)
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on ChromaDB client."""
        try:
            if not self._client:
                return {
                    "status": "unhealthy",
                    "error": "Client not initialized"
                }

            # Test basic operations
            start_time = time.time()

            # Get collection counts
            rfq_stats = await self.get_collection_stats("RFQ")
            offer_stats = await self.get_collection_stats("OfferTech")

            health_data = {
                "status": "healthy",
                "response_time": time.time() - start_time,
                "collections": {
                    "rfq": rfq_stats,
                    "offer": offer_stats
                },
                "statistics": self._operation_stats,
                "timestamp": time.time()
            }

            self._operation_stats["last_health_check"] = time.time()

            return health_data

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return self._operation_stats.copy()

    async def reset_collections(self, confirm: bool = False) -> bool:
        """Reset all collections (for testing only)."""
        if not confirm:
            logger.warning("Collection reset requires confirmation")
            return False

        try:
            if not self._client:
                return False

            # Delete existing collections
            for collection_name in [self.rfq_collection_name, self.offer_collection_name]:
                try:
                    self._client.delete_collection(collection_name)
                    logger.info(f"Deleted collection: {collection_name}")
                except Exception as e:
                    logger.warning(f"Could not delete collection {collection_name}: {e}")

            # Reinitialize collections
            self._collections.clear()
            await self._initialize_collections()

            # Reset stats
            self._operation_stats = {
                "add_operations": 0,
                "query_operations": 0,
                "delete_operations": 0,
                "errors": 0,
                "total_vectors": 0,
                "last_health_check": None
            }

            logger.info("Collections reset successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to reset collections: {e}")
            return False


# Global ChromaDB client instance
_chroma_client: Optional[ChromaDBClient] = None


async def get_vector_store() -> ChromaDBClient:
    """Get or create the global ChromaDB client instance."""
    global _chroma_client

    if _chroma_client is None:
        settings = get_settings()
        _chroma_client = ChromaDBClient(settings)
        await _chroma_client.start()

    return _chroma_client


async def close_vector_store() -> None:
    """Close the global ChromaDB client instance."""
    global _chroma_client

    if _chroma_client:
        await _chroma_client.stop()
        _chroma_client = None
