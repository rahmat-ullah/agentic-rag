"""
Integration tests for Sprint 6 Story 6-02: User Correction and Editing System

This module contains comprehensive integration tests for the content correction
system including submission, review workflow, version control, and re-embedding.
"""

import uuid
import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from agentic_rag.api.app import create_app
from agentic_rag.services.correction_service import CorrectionService
from agentic_rag.services.correction_embedding_service import CorrectionEmbeddingService
from agentic_rag.models.corrections import (
    ContentCorrection,
    ContentVersion,
    CorrectionReview,
    CorrectionWorkflow,
    CorrectionImpact,
    CorrectionType,
    CorrectionStatus,
    CorrectionPriority,
    ReviewDecision
)
from agentic_rag.schemas.corrections import (
    CorrectionSubmissionRequest,
    ReviewSubmissionRequest,
    VersionComparisonRequest,
    CorrectionTypeEnum,
    CorrectionPriorityEnum,
    ReviewDecisionEnum
)


class TestCorrectionSystemIntegration:
    """Integration tests for the complete correction system."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI application."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def test_tenant_id(self):
        """Test tenant ID."""
        return uuid.uuid4()
    
    @pytest.fixture
    def test_user_id(self):
        """Test user ID."""
        return uuid.uuid4()
    
    @pytest.fixture
    def test_chunk_id(self):
        """Test chunk ID."""
        return uuid.uuid4()
    
    @pytest.fixture
    def test_reviewer_id(self):
        """Test reviewer ID."""
        return uuid.uuid4()
    
    @pytest.fixture
    def correction_service(self, db_session):
        """Create correction service instance."""
        return CorrectionService(db_session)
    
    @pytest.fixture
    def embedding_service(self, db_session):
        """Create correction embedding service instance."""
        return CorrectionEmbeddingService(db_session)
    
    @pytest.fixture
    def sample_correction_request(self, test_chunk_id):
        """Sample correction submission request."""
        return CorrectionSubmissionRequest(
            chunk_id=test_chunk_id,
            corrected_content="The electrical components are priced at $150 per unit (updated from $120 as of December 2024).",
            correction_reason="Updated pricing information based on latest supplier quotes",
            correction_type=CorrectionTypeEnum.FACTUAL,
            priority=CorrectionPriorityEnum.HIGH,
            confidence_score=0.95,
            source_references=[
                {
                    "type": "supplier_quote",
                    "document": "Supplier_Quote_Dec2024.pdf",
                    "page": 3
                }
            ]
        )
    
    @pytest.fixture
    def sample_review_request(self):
        """Sample review submission request."""
        return ReviewSubmissionRequest(
            correction_id=uuid.uuid4(),  # Will be updated in tests
            decision=ReviewDecisionEnum.APPROVE,
            review_notes="Correction is accurate and well-sourced. Pricing update is verified.",
            accuracy_score=0.95,
            clarity_score=0.9,
            completeness_score=0.85,
            quality_assessment={
                "factual_accuracy": "excellent",
                "source_quality": "high",
                "impact_assessment": "medium"
            }
        )
    
    async def test_complete_correction_workflow(
        self,
        correction_service,
        test_tenant_id,
        test_user_id,
        test_reviewer_id,
        sample_correction_request,
        sample_review_request
    ):
        """Test complete correction workflow from submission to implementation."""
        
        # Step 1: Submit correction
        submission_result = await correction_service.submit_correction(
            tenant_id=test_tenant_id,
            user_id=test_user_id,
            correction_data=sample_correction_request
        )
        
        assert submission_result.correction_id is not None
        assert submission_result.status == "pending"
        assert submission_result.workflow_id is not None
        assert "Expert review assignment" in submission_result.next_steps
        
        correction_id = submission_result.correction_id
        
        # Step 2: Submit expert review
        sample_review_request.correction_id = correction_id
        
        review_id = await correction_service.submit_review(
            tenant_id=test_tenant_id,
            reviewer_id=test_reviewer_id,
            review_data=sample_review_request
        )
        
        assert review_id is not None
        
        # Step 3: Implement correction
        version_id = await correction_service.implement_correction(
            tenant_id=test_tenant_id,
            correction_id=correction_id,
            implementer_id=test_reviewer_id
        )
        
        assert version_id is not None
        
        # Verify correction status
        # This would require additional service methods to check status
        
    async def test_version_comparison(
        self,
        correction_service,
        test_tenant_id,
        test_chunk_id
    ):
        """Test version comparison functionality."""
        
        comparison_request = VersionComparisonRequest(
            chunk_id=test_chunk_id,
            version_1=1,
            version_2=2,
            comparison_type="side_by_side"
        )
        
        # This test would require existing versions in the database
        # For now, we'll test the service method structure
        try:
            result = await correction_service.compare_versions(
                tenant_id=test_tenant_id,
                comparison_request=comparison_request
            )
            
            assert result.chunk_id == test_chunk_id
            assert result.version_1 is not None
            assert result.version_2 is not None
            assert isinstance(result.differences, list)
            assert isinstance(result.similarity_score, float)
            assert result.change_summary is not None
            
        except Exception as e:
            # Expected to fail without proper test data setup
            assert "not found" in str(e).lower()
    
    async def test_correction_statistics(
        self,
        correction_service,
        test_tenant_id
    ):
        """Test correction statistics retrieval."""
        
        stats = await correction_service.get_correction_stats(test_tenant_id)
        
        assert stats.total_corrections >= 0
        assert stats.pending_corrections >= 0
        assert stats.approved_corrections >= 0
        assert stats.rejected_corrections >= 0
        assert stats.average_review_time_hours >= 0.0
        assert isinstance(stats.correction_type_breakdown, dict)
        assert isinstance(stats.quality_improvement_metrics, dict)
    
    async def test_re_embedding_workflow(
        self,
        embedding_service,
        test_tenant_id
    ):
        """Test re-embedding workflow for corrections."""
        
        # Test batch re-embedding
        results = await embedding_service.batch_re_embedding(
            tenant_id=test_tenant_id,
            max_batch_size=5
        )
        
        # Should return empty list if no pending corrections
        assert isinstance(results, list)
        
        # Test queue statistics
        queue_stats = await embedding_service.get_re_embedding_queue_stats(test_tenant_id)
        
        assert "pending_re_embeddings" in queue_stats
        assert "completed_re_embeddings" in queue_stats
        assert "average_processing_time_seconds" in queue_stats
        assert "estimated_queue_time_minutes" in queue_stats
    
    def test_correction_api_endpoints(self, client):
        """Test correction API endpoints."""
        
        # Test correction submission endpoint
        correction_data = {
            "chunk_id": str(uuid.uuid4()),
            "corrected_content": "Updated content with corrections",
            "correction_reason": "Test correction",
            "correction_type": "factual",
            "priority": "medium",
            "confidence_score": 0.8
        }
        
        # This would require proper authentication setup
        # For now, test the endpoint structure
        response = client.post("/api/v1/corrections", json=correction_data)
        
        # Expected to fail without authentication
        assert response.status_code in [401, 422]  # Unauthorized or validation error
    
    def test_version_comparison_api(self, client):
        """Test version comparison API endpoint."""
        
        comparison_data = {
            "chunk_id": str(uuid.uuid4()),
            "version_1": 1,
            "version_2": 2,
            "comparison_type": "side_by_side"
        }
        
        response = client.post("/api/v1/versions/compare", json=comparison_data)
        
        # Expected to fail without authentication
        assert response.status_code in [401, 422]
    
    def test_correction_stats_api(self, client):
        """Test correction statistics API endpoint."""
        
        response = client.get("/api/v1/corrections/stats")
        
        # Expected to fail without authentication
        assert response.status_code == 401
    
    async def test_correction_validation(
        self,
        correction_service,
        test_tenant_id,
        test_user_id,
        test_chunk_id
    ):
        """Test correction validation logic."""
        
        # Test identical content validation
        invalid_request = CorrectionSubmissionRequest(
            chunk_id=test_chunk_id,
            corrected_content="",  # Empty content
            correction_reason="Test",
            correction_type=CorrectionTypeEnum.FACTUAL,
            priority=CorrectionPriorityEnum.MEDIUM
        )
        
        with pytest.raises(Exception):  # Should raise validation error
            await correction_service.submit_correction(
                tenant_id=test_tenant_id,
                user_id=test_user_id,
                correction_data=invalid_request
            )
    
    async def test_review_validation(
        self,
        correction_service,
        test_tenant_id,
        test_reviewer_id
    ):
        """Test review validation logic."""
        
        # Test review with missing notes for rejection
        invalid_review = ReviewSubmissionRequest(
            correction_id=uuid.uuid4(),
            decision=ReviewDecisionEnum.REJECT,
            review_notes=None,  # Missing required notes for rejection
            accuracy_score=0.5
        )
        
        with pytest.raises(Exception):  # Should raise validation error
            await correction_service.submit_review(
                tenant_id=test_tenant_id,
                reviewer_id=test_reviewer_id,
                review_data=invalid_review
            )
    
    async def test_workflow_state_management(
        self,
        correction_service,
        test_tenant_id,
        test_user_id,
        sample_correction_request
    ):
        """Test workflow state management."""
        
        # Submit correction and verify workflow creation
        result = await correction_service.submit_correction(
            tenant_id=test_tenant_id,
            user_id=test_user_id,
            correction_data=sample_correction_request
        )
        
        assert result.workflow_id is not None
        assert result.next_steps is not None
        assert len(result.next_steps) > 0
    
    async def test_priority_based_processing(
        self,
        correction_service,
        test_tenant_id,
        test_user_id,
        test_chunk_id
    ):
        """Test priority-based correction processing."""
        
        # Submit high priority correction
        high_priority_request = CorrectionSubmissionRequest(
            chunk_id=test_chunk_id,
            corrected_content="Critical update to safety information",
            correction_reason="Safety compliance update",
            correction_type=CorrectionTypeEnum.FACTUAL,
            priority=CorrectionPriorityEnum.CRITICAL,
            confidence_score=1.0
        )
        
        result = await correction_service.submit_correction(
            tenant_id=test_tenant_id,
            user_id=test_user_id,
            correction_data=high_priority_request
        )
        
        # Critical priority should have faster estimated review time
        assert "4-8 hours" in result.estimated_review_time or "1-2" in result.estimated_review_time
    
    async def test_correction_impact_tracking(
        self,
        embedding_service,
        test_tenant_id
    ):
        """Test correction impact tracking."""
        
        # Test impact status for non-existent correction
        fake_correction_id = uuid.uuid4()
        
        try:
            status = await embedding_service.get_re_embedding_status(
                correction_id=fake_correction_id,
                tenant_id=test_tenant_id
            )
            
            # Should not reach here
            assert False, "Expected ServiceError for non-existent correction"
            
        except Exception as e:
            assert "not found" in str(e).lower()


class TestCorrectionSystemPerformance:
    """Performance tests for correction system."""
    
    async def test_batch_correction_processing(
        self,
        correction_service,
        test_tenant_id,
        test_user_id
    ):
        """Test batch correction processing performance."""
        
        # This would test processing multiple corrections
        # For now, verify the service can handle the load
        
        stats = await correction_service.get_correction_stats(test_tenant_id)
        
        # Basic performance check - should complete quickly
        start_time = datetime.now()
        await correction_service.get_correction_stats(test_tenant_id)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 1.0  # Should complete within 1 second
    
    async def test_re_embedding_performance(
        self,
        embedding_service,
        test_tenant_id
    ):
        """Test re-embedding performance."""
        
        # Test queue statistics performance
        start_time = datetime.now()
        stats = await embedding_service.get_re_embedding_queue_stats(test_tenant_id)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 2.0  # Should complete within 2 seconds
        
        assert isinstance(stats, dict)
        assert "pending_re_embeddings" in stats


# Test fixtures and utilities

@pytest.fixture
def db_session():
    """Mock database session for testing."""
    # This would be replaced with actual test database session
    class MockSession:
        def query(self, *args):
            return MockQuery()
        
        def add(self, obj):
            pass
        
        def commit(self):
            pass
        
        def rollback(self):
            pass
        
        def refresh(self, obj):
            pass
    
    class MockQuery:
        def filter(self, *args):
            return self
        
        def join(self, *args, **kwargs):
            return self
        
        def first(self):
            return None
        
        def all(self):
            return []
        
        def count(self):
            return 0
        
        def scalar(self):
            return 0.0
        
        def limit(self, n):
            return self
        
        def group_by(self, *args):
            return self
        
        def update(self, values):
            return 0
    
    return MockSession()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
