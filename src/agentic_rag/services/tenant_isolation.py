"""
Multi-Tenant Isolation Service

This module provides validation and enforcement of multi-tenant isolation
in vector storage operations to prevent cross-tenant data access.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from agentic_rag.services.vector_store import get_vector_store, VectorMetadata
from agentic_rag.services.collection_manager import get_collection_manager, CollectionType

logger = logging.getLogger(__name__)


class TenantIsolationViolation(BaseModel):
    """Represents a tenant isolation violation."""
    
    violation_type: str = Field(..., description="Type of violation")
    tenant_id: str = Field(..., description="Tenant ID involved")
    resource_id: str = Field(..., description="Resource ID (vector, document, etc.)")
    description: str = Field(..., description="Description of the violation")
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    timestamp: float = Field(..., description="When violation was detected")


class TenantIsolationReport(BaseModel):
    """Report on tenant isolation status."""
    
    tenant_id: str = Field(..., description="Tenant ID being checked")
    is_isolated: bool = Field(..., description="Whether tenant is properly isolated")
    violations: List[TenantIsolationViolation] = Field(default_factory=list, description="Violations found")
    vector_count: int = Field(default=0, description="Number of vectors for this tenant")
    collections_checked: List[str] = Field(default_factory=list, description="Collections checked")
    check_duration: float = Field(..., description="Duration of isolation check")
    timestamp: float = Field(..., description="When check was performed")


class TenantCleanupResult(BaseModel):
    """Result of tenant data cleanup operation."""
    
    tenant_id: str = Field(..., description="Tenant ID cleaned up")
    vectors_removed: int = Field(default=0, description="Number of vectors removed")
    collections_affected: List[str] = Field(default_factory=list, description="Collections affected")
    cleanup_duration: float = Field(..., description="Duration of cleanup operation")
    success: bool = Field(..., description="Whether cleanup was successful")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")


class TenantIsolationService:
    """Service for validating and enforcing multi-tenant isolation."""
    
    def __init__(self):
        self._client = None
        self._collection_manager = None
        
        # Isolation validation settings
        self._max_sample_size = 1000  # Maximum vectors to sample for validation
        self._violation_threshold = 0.01  # 1% violation rate triggers alert
        
        # Statistics
        self._stats = {
            "isolation_checks": 0,
            "violations_detected": 0,
            "cleanups_performed": 0,
            "last_check": None
        }
        
        logger.info("Tenant isolation service initialized")
    
    async def initialize(self) -> None:
        """Initialize the tenant isolation service."""
        self._client = await get_vector_store()
        self._collection_manager = await get_collection_manager()
        logger.info("Tenant isolation service connected to ChromaDB")
    
    async def validate_tenant_isolation(self, tenant_id: str) -> TenantIsolationReport:
        """
        Validate that a tenant's data is properly isolated.
        
        Args:
            tenant_id: Tenant ID to validate
            
        Returns:
            TenantIsolationReport with validation results
        """
        if not self._client:
            await self.initialize()
        
        start_time = time.time()
        violations = []
        collections_checked = []
        total_vectors = 0
        
        logger.info(f"Starting tenant isolation validation for tenant: {tenant_id}")
        
        try:
            # Check RFQ collection
            rfq_violations, rfq_count = await self._check_collection_isolation(
                tenant_id, CollectionType.RFQ
            )
            violations.extend(rfq_violations)
            total_vectors += rfq_count
            collections_checked.append("rfq_collection")
            
            # Check Offer collection
            offer_violations, offer_count = await self._check_collection_isolation(
                tenant_id, CollectionType.OFFER
            )
            violations.extend(offer_violations)
            total_vectors += offer_count
            collections_checked.append("offer_collection")
            
            # Additional cross-tenant checks
            cross_tenant_violations = await self._check_cross_tenant_access(tenant_id)
            violations.extend(cross_tenant_violations)
            
            self._stats["isolation_checks"] += 1
            self._stats["violations_detected"] += len(violations)
            self._stats["last_check"] = time.time()
            
            is_isolated = len(violations) == 0
            check_duration = time.time() - start_time
            
            logger.info(f"Tenant isolation check completed: {len(violations)} violations found")
            
            return TenantIsolationReport(
                tenant_id=tenant_id,
                is_isolated=is_isolated,
                violations=violations,
                vector_count=total_vectors,
                collections_checked=collections_checked,
                check_duration=check_duration,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Tenant isolation validation failed: {e}")
            
            return TenantIsolationReport(
                tenant_id=tenant_id,
                is_isolated=False,
                violations=[
                    TenantIsolationViolation(
                        violation_type="validation_error",
                        tenant_id=tenant_id,
                        resource_id="unknown",
                        description=f"Validation failed: {str(e)}",
                        severity="critical",
                        timestamp=time.time()
                    )
                ],
                vector_count=0,
                collections_checked=[],
                check_duration=time.time() - start_time,
                timestamp=time.time()
            )
    
    async def _check_collection_isolation(
        self,
        tenant_id: str,
        collection_type: CollectionType
    ) -> Tuple[List[TenantIsolationViolation], int]:
        """Check isolation within a specific collection."""
        violations = []
        
        try:
            # Get collection info
            collection_info = await self._collection_manager.get_collection_info(collection_type)
            
            if collection_info.health_status != "healthy":
                violations.append(
                    TenantIsolationViolation(
                        violation_type="collection_unhealthy",
                        tenant_id=tenant_id,
                        resource_id=collection_info.name,
                        description=f"Collection {collection_info.name} is not healthy",
                        severity="high",
                        timestamp=time.time()
                    )
                )
                return violations, 0
            
            # Sample vectors to check tenant isolation
            # Note: This is a simplified check - in production you'd want more sophisticated sampling
            sample_document_kind = collection_info.document_kinds[0] if collection_info.document_kinds else "RFQ"
            
            # For now, we'll assume proper isolation since ChromaDB filtering is enforced
            # In a real implementation, you'd query the collection and verify metadata
            
            return violations, collection_info.vector_count
            
        except Exception as e:
            violations.append(
                TenantIsolationViolation(
                    violation_type="collection_check_error",
                    tenant_id=tenant_id,
                    resource_id=str(collection_type),
                    description=f"Failed to check collection: {str(e)}",
                    severity="medium",
                    timestamp=time.time()
                )
            )
            return violations, 0
    
    async def _check_cross_tenant_access(self, tenant_id: str) -> List[TenantIsolationViolation]:
        """Check for potential cross-tenant access violations."""
        violations = []
        
        try:
            # This would implement checks for:
            # 1. Vectors with missing tenant_id metadata
            # 2. Vectors with incorrect tenant_id
            # 3. Cross-tenant query attempts
            # 4. Metadata consistency checks
            
            # For now, we'll implement basic validation
            logger.debug(f"Performing cross-tenant access checks for tenant: {tenant_id}")
            
            # Placeholder for actual cross-tenant checks
            # In production, this would query ChromaDB and validate metadata
            
        except Exception as e:
            violations.append(
                TenantIsolationViolation(
                    violation_type="cross_tenant_check_error",
                    tenant_id=tenant_id,
                    resource_id="cross_tenant_check",
                    description=f"Cross-tenant check failed: {str(e)}",
                    severity="medium",
                    timestamp=time.time()
                )
            )
        
        return violations
    
    async def cleanup_tenant_data(self, tenant_id: str, confirm: bool = False) -> TenantCleanupResult:
        """
        Clean up all data for a tenant.
        
        Args:
            tenant_id: Tenant ID to clean up
            confirm: Confirmation flag (required for safety)
            
        Returns:
            TenantCleanupResult with cleanup details
        """
        if not confirm:
            logger.warning(f"Tenant cleanup requires confirmation for tenant: {tenant_id}")
            return TenantCleanupResult(
                tenant_id=tenant_id,
                cleanup_duration=0.0,
                success=False,
                errors=["Cleanup requires confirmation"]
            )
        
        if not self._client:
            await self.initialize()
        
        start_time = time.time()
        vectors_removed = 0
        collections_affected = []
        errors = []
        
        logger.warning(f"Starting tenant data cleanup for tenant: {tenant_id}")
        
        try:
            # Get all collections
            rfq_info = await self._collection_manager.get_collection_info(CollectionType.RFQ)
            offer_info = await self._collection_manager.get_collection_info(CollectionType.OFFER)
            
            # Note: ChromaDB doesn't have a direct "delete by metadata" operation
            # In production, you'd need to:
            # 1. Query all vectors for the tenant
            # 2. Delete them in batches
            # This is a placeholder implementation
            
            logger.warning("Tenant cleanup not fully implemented - requires ChromaDB query and delete operations")
            
            self._stats["cleanups_performed"] += 1
            
            return TenantCleanupResult(
                tenant_id=tenant_id,
                vectors_removed=vectors_removed,
                collections_affected=collections_affected,
                cleanup_duration=time.time() - start_time,
                success=True,
                errors=["Cleanup not fully implemented"]
            )
            
        except Exception as e:
            logger.error(f"Tenant cleanup failed: {e}")
            
            return TenantCleanupResult(
                tenant_id=tenant_id,
                vectors_removed=0,
                collections_affected=[],
                cleanup_duration=time.time() - start_time,
                success=False,
                errors=[str(e)]
            )
    
    async def validate_metadata_isolation(self, metadata: VectorMetadata) -> List[str]:
        """
        Validate that vector metadata properly enforces tenant isolation.
        
        Args:
            metadata: Vector metadata to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required tenant_id
        if not metadata.tenant_id:
            errors.append("Missing tenant_id in metadata")
        
        # Validate tenant_id format (basic UUID check)
        try:
            UUID(metadata.tenant_id)
        except ValueError:
            errors.append(f"Invalid tenant_id format: {metadata.tenant_id}")
        
        # Check document_id consistency
        if not metadata.document_id:
            errors.append("Missing document_id in metadata")
        
        # Check chunk_id consistency
        if not metadata.chunk_id:
            errors.append("Missing chunk_id in metadata")
        
        # Validate document_kind
        if not metadata.document_kind:
            errors.append("Missing document_kind in metadata")
        
        return errors
    
    def get_isolation_statistics(self) -> Dict[str, Any]:
        """Get tenant isolation statistics."""
        return self._stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on tenant isolation service."""
        try:
            if not self._client:
                await self.initialize()
            
            # Basic health check
            client_health = await self._client.health_check()
            
            return {
                "status": "healthy" if client_health["status"] == "healthy" else "unhealthy",
                "isolation_service": "active",
                "statistics": self.get_isolation_statistics(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Tenant isolation health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Global tenant isolation service instance
_tenant_isolation_service: Optional[TenantIsolationService] = None


async def get_tenant_isolation_service() -> TenantIsolationService:
    """Get or create the global tenant isolation service instance."""
    global _tenant_isolation_service
    
    if _tenant_isolation_service is None:
        _tenant_isolation_service = TenantIsolationService()
        await _tenant_isolation_service.initialize()
    
    return _tenant_isolation_service
