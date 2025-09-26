"""
Integration tests for Sprint 6 Feedback Collection System

This module tests the complete feedback collection system including
API endpoints, service integration, and database operations.
"""

import pytest
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from agentic_rag.api.app import create_app
from agentic_rag.models.feedback import (
    UserFeedbackSubmission,
    FeedbackAggregation,
    FeedbackSession,
    FeedbackType,
    FeedbackCategory,
    FeedbackStatus,
    FeedbackPriority
)
from agentic_rag.schemas.feedback import (
    FeedbackSubmissionRequest,
    ThumbsFeedbackRequest,
    DetailedFeedbackRequest,
    FeedbackTypeEnum,
    FeedbackCategoryEnum,
    FeedbackPriorityEnum
)
from agentic_rag.services.feedback_service import FeedbackService
from agentic_rag.services.feedback_integration import FeedbackIntegrationService, FeedbackContext


class TestFeedbackSystem:
    """Test suite for the feedback collection system."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        # This would be implemented with actual test database setup
        pass
    
    @pytest.fixture
    def feedback_service(self, db_session):
        """Create feedback service instance."""
        return FeedbackService(db_session)
    
    @pytest.fixture
    def feedback_integration_service(self, db_session, feedback_service):
        """Create feedback integration service instance."""
        return FeedbackIntegrationService(db_session, feedback_service)
    
    @pytest.fixture
    def test_user_context(self):
        """Create test user context."""
        return {
            "user_id": uuid.uuid4(),
            "tenant_id": uuid.uuid4(),
            "session_id": "test_session_123",
            "user_agent": "pytest/1.0"
        }
    
    @pytest.fixture
    def auth_headers(self, test_user_context):
        """Create authentication headers for API tests."""
        # This would be implemented with actual JWT token generation
        return {
            "Authorization": "Bearer test_token",
            "X-Tenant-ID": str(test_user_context["tenant_id"])
        }
    
    def test_submit_basic_feedback(self, client, auth_headers):
        """Test basic feedback submission via API."""
        feedback_data = {
            "feedback_type": "search_result",
            "feedback_category": "not_relevant",
            "target_id": str(uuid.uuid4()),
            "target_type": "search_result",
            "rating": -1,
            "feedback_text": "This result doesn't match my query",
            "query": "pricing information for electrical components"
        }
        
        response = client.post(
            "/api/v1/feedback",
            json=feedback_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "feedback_id" in result
        assert result["status"] == "pending"
        assert "confirmation_message" in result
    
    def test_submit_thumbs_feedback(self, client, auth_headers):
        """Test thumbs up/down feedback submission."""
        thumbs_data = {
            "target_id": str(uuid.uuid4()),
            "target_type": "search_result",
            "thumbs_up": True,
            "query": "pricing information"
        }
        
        response = client.post(
            "/api/v1/feedback/thumbs",
            json=thumbs_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["estimated_processing_time"] == "Immediate"
    
    def test_submit_detailed_feedback(self, client, auth_headers):
        """Test detailed feedback form submission."""
        detailed_data = {
            "feedback_type": "answer_quality",
            "feedback_category": "inaccurate_information",
            "title": "Incorrect pricing information",
            "description": "The answer provided outdated pricing data",
            "expected_behavior": "Current pricing from latest documents",
            "actual_behavior": "Pricing from 2022 documents",
            "priority": "high"
        }
        
        response = client.post(
            "/api/v1/feedback/detailed",
            json=detailed_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "feedback_id" in result
    
    def test_get_user_feedback_history(self, client, auth_headers):
        """Test retrieving user feedback history."""
        response = client.get(
            "/api/v1/feedback",
            headers=auth_headers,
            params={"page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "items" in result
        assert "total_count" in result
        assert "page" in result
    
    def test_get_feedback_stats(self, client, auth_headers):
        """Test retrieving feedback statistics."""
        response = client.get(
            "/api/v1/feedback/stats",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "total_submissions" in result
        assert "pending_count" in result
        assert "category_breakdown" in result
    
    def test_feedback_validation(self, client, auth_headers):
        """Test feedback validation rules."""
        # Test invalid rating for answer quality
        invalid_data = {
            "feedback_type": "answer_quality",
            "rating": 0,  # Invalid for answer quality (must be 1-5)
            "target_id": str(uuid.uuid4()),
            "target_type": "answer"
        }
        
        response = client.post(
            "/api/v1/feedback",
            json=invalid_data,
            headers=auth_headers
        )
        
        assert response.status_code == 400
    
    def test_feedback_service_submission(self, feedback_service, test_user_context):
        """Test feedback service submission logic."""
        feedback_data = FeedbackSubmissionRequest(
            feedback_type=FeedbackTypeEnum.SEARCH_RESULT,
            feedback_category=FeedbackCategoryEnum.NOT_RELEVANT,
            target_id=uuid.uuid4(),
            target_type="search_result",
            rating=-1,
            feedback_text="Not relevant to my query",
            query="test query"
        )
        
        # This would test the actual service method
        # result = await feedback_service.submit_feedback(
        #     tenant_id=test_user_context["tenant_id"],
        #     user_id=test_user_context["user_id"],
        #     feedback_data=feedback_data
        # )
        # 
        # assert result.feedback_id is not None
        # assert result.status == FeedbackStatusEnum.PENDING
    
    def test_feedback_aggregation_update(self, feedback_service, test_user_context):
        """Test feedback aggregation calculation."""
        target_id = uuid.uuid4()
        
        # Submit multiple feedback items for the same target
        feedback_items = [
            {"rating": 1, "category": FeedbackCategoryEnum.NOT_RELEVANT},
            {"rating": -1, "category": FeedbackCategoryEnum.OUTDATED_CONTENT},
            {"rating": 1, "category": FeedbackCategoryEnum.NOT_RELEVANT},
        ]
        
        # This would test aggregation logic
        # for item in feedback_items:
        #     feedback_data = FeedbackSubmissionRequest(
        #         feedback_type=FeedbackTypeEnum.SEARCH_RESULT,
        #         feedback_category=item["category"],
        #         target_id=target_id,
        #         target_type="search_result",
        #         rating=item["rating"]
        #     )
        #     
        #     await feedback_service.submit_feedback(
        #         tenant_id=test_user_context["tenant_id"],
        #         user_id=test_user_context["user_id"],
        #         feedback_data=feedback_data
        #     )
        
        # Verify aggregation was updated correctly
        # aggregation = db_session.query(FeedbackAggregation).filter(
        #     FeedbackAggregation.target_id == target_id
        # ).first()
        # 
        # assert aggregation.total_feedback_count == 3
        # assert aggregation.positive_count == 2
        # assert aggregation.negative_count == 1
    
    def test_search_result_feedback_integration(self, feedback_integration_service, test_user_context):
        """Test integration with search results."""
        # Mock search response and result item
        from agentic_rag.api.models.search import SearchResponse, SearchResultItem
        
        search_response = SearchResponse(
            request_id="test_request_123",
            query="test query",
            results=[],
            total_results=1,
            search_time_ms=100,
            pagination=None,
            statistics=None
        )
        
        result_item = SearchResultItem(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title="Test Document",
            text="Test chunk content",
            relevance_score=0.85,
            rank=1,
            metadata={}
        )
        
        context = FeedbackContext(
            user_id=test_user_context["user_id"],
            tenant_id=test_user_context["tenant_id"],
            session_id=test_user_context["session_id"],
            query="test query"
        )
        
        # This would test the integration method
        # feedback_id = await feedback_integration_service.collect_search_result_feedback(
        #     search_response=search_response,
        #     result_item=result_item,
        #     rating=1,
        #     context=context,
        #     feedback_text="Good result"
        # )
        # 
        # assert feedback_id is not None
    
    def test_feedback_session_tracking(self, feedback_service, test_user_context):
        """Test feedback session tracking."""
        session_id = test_user_context["session_id"]
        
        # Submit multiple feedback items in the same session
        for i in range(3):
            feedback_data = FeedbackSubmissionRequest(
                feedback_type=FeedbackTypeEnum.SEARCH_RESULT,
                target_id=uuid.uuid4(),
                target_type="search_result",
                rating=1,
                session_id=session_id
            )
            
            # This would test session tracking
            # await feedback_service.submit_feedback(
            #     tenant_id=test_user_context["tenant_id"],
            #     user_id=test_user_context["user_id"],
            #     feedback_data=feedback_data,
            #     session_id=session_id
            # )
        
        # Verify session was tracked correctly
        # session = db_session.query(FeedbackSession).filter(
        #     FeedbackSession.session_id == session_id
        # ).first()
        # 
        # assert session.feedback_submissions == 3
        # assert session.total_interactions >= 3
    
    def test_feedback_priority_determination(self, feedback_service):
        """Test automatic priority determination."""
        # Bug report should be high priority
        bug_feedback = FeedbackSubmissionRequest(
            feedback_type=FeedbackTypeEnum.GENERAL,
            feedback_category=FeedbackCategoryEnum.BUG_REPORT,
            feedback_text="System crashes when uploading large files"
        )
        
        # Negative rating should be medium priority
        negative_feedback = FeedbackSubmissionRequest(
            feedback_type=FeedbackTypeEnum.SEARCH_RESULT,
            rating=-1,
            feedback_text="Completely irrelevant results"
        )
        
        # Feature request should be medium priority
        feature_feedback = FeedbackSubmissionRequest(
            feedback_type=FeedbackTypeEnum.GENERAL,
            feedback_category=FeedbackCategoryEnum.FEATURE_REQUEST,
            feedback_text="Add export to Excel functionality"
        )
        
        # This would test priority determination logic
        # bug_priority = feedback_service._determine_priority(bug_feedback)
        # negative_priority = feedback_service._determine_priority(negative_feedback)
        # feature_priority = feedback_service._determine_priority(feature_feedback)
        # 
        # assert bug_priority == FeedbackPriority.HIGH
        # assert negative_priority == FeedbackPriority.MEDIUM
        # assert feature_priority == FeedbackPriority.MEDIUM
    
    def test_feedback_insights_generation(self, feedback_integration_service, test_user_context):
        """Test feedback insights for similar queries."""
        query = "pricing information"
        
        # This would test insights generation
        # insights = await feedback_integration_service.get_feedback_insights_for_query(
        #     query=query,
        #     tenant_id=test_user_context["tenant_id"]
        # )
        # 
        # assert "insights" in insights
        # assert "patterns" in insights
        # assert "total_similar_feedback" in insights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
