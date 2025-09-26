"""
Integration tests for the Learning Algorithms System (Sprint 6 Story 6-03).

This module contains comprehensive integration tests for the learning algorithms
system including database operations, API endpoints, and service integration.
"""

import uuid
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from agentic_rag.models.learning import (
    LearningAlgorithm,
    FeedbackSignal,
    LearningPerformanceMetric,
    ABTestExperiment,
    LearningAlgorithmType,
    LearningModelType,
    LearningStatus,
    FeedbackSignalType
)
from agentic_rag.models.feedback import UserFeedbackSubmission, FeedbackType
from agentic_rag.services.learning_service import LearningService
from agentic_rag.services.learning_integration_service import LearningIntegrationService
from agentic_rag.services.learning_monitoring_service import LearningMonitoringService


class TestLearningAlgorithmDatabase:
    """Test learning algorithm database operations."""
    
    def test_create_learning_algorithm(self, db_session: Session, test_tenant):
        """Test creating a learning algorithm."""
        algorithm = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.LINK_CONFIDENCE,
            model_type=LearningModelType.EXPONENTIAL_MOVING_AVERAGE,
            name="Test Link Confidence Algorithm",
            description="Test algorithm for link confidence learning",
            learning_rate=0.01,
            validation_threshold=0.05
        )
        
        db_session.add(algorithm)
        db_session.commit()
        db_session.refresh(algorithm)
        
        assert algorithm.id is not None
        assert algorithm.algorithm_type == LearningAlgorithmType.LINK_CONFIDENCE
        assert algorithm.model_type == LearningModelType.EXPONENTIAL_MOVING_AVERAGE
        assert algorithm.name == "Test Link Confidence Algorithm"
        assert algorithm.learning_rate == 0.01
        assert algorithm.status == LearningStatus.ACTIVE
        assert algorithm.is_enabled is True
    
    def test_create_feedback_signal(self, db_session: Session, test_tenant, test_user):
        """Test creating a feedback signal."""
        target_id = uuid.uuid4()
        
        signal = FeedbackSignal(
            tenant_id=test_tenant.id,
            signal_type=FeedbackSignalType.EXPLICIT_RATING,
            target_type="chunk",
            target_id=target_id,
            signal_value=4.0,
            signal_strength=1.0,
            signal_confidence=0.9,
            user_id=test_user.id,
            session_id="test-session-123",
            query_context="What is the pricing for this service?"
        )
        
        db_session.add(signal)
        db_session.commit()
        db_session.refresh(signal)
        
        assert signal.id is not None
        assert signal.signal_type == FeedbackSignalType.EXPLICIT_RATING
        assert signal.target_type == "chunk"
        assert signal.target_id == target_id
        assert signal.signal_value == 4.0
        assert signal.is_processed is False
    
    def test_create_performance_metric(self, db_session: Session, test_tenant):
        """Test creating a performance metric."""
        # Create algorithm first
        algorithm = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.CHUNK_RANKING,
            model_type=LearningModelType.BAYESIAN_UPDATE,
            name="Test Chunk Ranking Algorithm"
        )
        db_session.add(algorithm)
        db_session.commit()
        
        # Create performance metric
        metric = LearningPerformanceMetric(
            tenant_id=test_tenant.id,
            algorithm_id=algorithm.id,
            metric_name="accuracy",
            metric_value=0.85,
            metric_type="accuracy",
            measurement_period_start=datetime.now() - timedelta(hours=1),
            measurement_period_end=datetime.now(),
            sample_size=100
        )
        
        db_session.add(metric)
        db_session.commit()
        db_session.refresh(metric)
        
        assert metric.id is not None
        assert metric.algorithm_id == algorithm.id
        assert metric.metric_name == "accuracy"
        assert metric.metric_value == 0.85
        assert metric.sample_size == 100
    
    def test_create_ab_test_experiment(self, db_session: Session, test_tenant):
        """Test creating an A/B test experiment."""
        # Create control and treatment algorithms
        control_alg = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.QUERY_EXPANSION,
            model_type=LearningModelType.COLLABORATIVE_FILTERING,
            name="Control Algorithm"
        )
        treatment_alg = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.QUERY_EXPANSION,
            model_type=LearningModelType.NEURAL_LANGUAGE_MODEL,
            name="Treatment Algorithm"
        )
        
        db_session.add_all([control_alg, treatment_alg])
        db_session.commit()
        
        # Create experiment
        experiment = ABTestExperiment(
            tenant_id=test_tenant.id,
            experiment_name="Query Expansion A/B Test",
            description="Testing neural vs collaborative filtering",
            control_algorithm_id=control_alg.id,
            treatment_algorithm_id=treatment_alg.id,
            primary_metric="click_through_rate",
            success_threshold=0.05,
            traffic_split_percentage=50.0
        )
        
        db_session.add(experiment)
        db_session.commit()
        db_session.refresh(experiment)
        
        assert experiment.id is not None
        assert experiment.experiment_name == "Query Expansion A/B Test"
        assert experiment.control_algorithm_id == control_alg.id
        assert experiment.treatment_algorithm_id == treatment_alg.id
        assert experiment.status == "draft"
        assert experiment.is_active is False


class TestLearningService:
    """Test learning service functionality."""
    
    @pytest.fixture
    def learning_service(self, db_session: Session):
        """Create learning service instance."""
        return LearningService(db_session)
    
    @pytest.fixture
    def test_algorithm(self, db_session: Session, test_tenant):
        """Create test learning algorithm."""
        algorithm = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.LINK_CONFIDENCE,
            model_type=LearningModelType.EXPONENTIAL_MOVING_AVERAGE,
            name="Test Algorithm",
            learning_rate=0.01
        )
        db_session.add(algorithm)
        db_session.commit()
        db_session.refresh(algorithm)
        return algorithm
    
    @pytest.mark.asyncio
    async def test_initialize_algorithms(
        self,
        learning_service: LearningService,
        test_tenant,
        test_algorithm
    ):
        """Test algorithm initialization."""
        await learning_service.initialize_algorithms(test_tenant.id)
        
        assert test_algorithm.id in learning_service.algorithms
        algorithm_instance = learning_service.algorithms[test_algorithm.id]
        assert algorithm_instance.algorithm_id == test_algorithm.id
        assert algorithm_instance.learning_rate == 0.01
    
    @pytest.mark.asyncio
    async def test_process_feedback_signal(
        self,
        learning_service: LearningService,
        db_session: Session,
        test_tenant,
        test_user,
        test_algorithm
    ):
        """Test processing feedback signal."""
        # Initialize algorithms
        await learning_service.initialize_algorithms(test_tenant.id)
        
        # Create feedback signal
        signal = FeedbackSignal(
            tenant_id=test_tenant.id,
            signal_type=FeedbackSignalType.EXPLICIT_RATING,
            target_type="link",
            target_id=uuid.uuid4(),
            signal_value=4.0,
            user_id=test_user.id
        )
        db_session.add(signal)
        db_session.commit()
        
        # Process signal
        updates = await learning_service.process_feedback_signal(test_tenant.id, signal)
        
        assert len(updates) > 0
        assert signal.is_processed is True
        assert signal.processed_at is not None


class TestLearningIntegrationService:
    """Test learning integration service functionality."""
    
    @pytest.fixture
    def integration_service(self, db_session: Session):
        """Create learning integration service instance."""
        return LearningIntegrationService(db_session)
    
    @pytest.mark.asyncio
    async def test_create_feedback_signal(
        self,
        integration_service: LearningIntegrationService,
        test_tenant,
        test_user
    ):
        """Test creating feedback signal."""
        target_id = uuid.uuid4()
        
        signal = await integration_service.create_feedback_signal(
            tenant_id=test_tenant.id,
            signal_type=FeedbackSignalType.CLICK_THROUGH,
            target_type="chunk",
            target_id=target_id,
            signal_value=1.0,
            user_id=test_user.id,
            session_id="test-session",
            query_context="test query"
        )
        
        assert signal.id is not None
        assert signal.signal_type == FeedbackSignalType.CLICK_THROUGH
        assert signal.target_id == target_id
        assert signal.user_id == test_user.id
    
    @pytest.mark.asyncio
    async def test_process_feedback_for_learning(
        self,
        integration_service: LearningIntegrationService,
        db_session: Session,
        test_tenant,
        test_user
    ):
        """Test processing feedback for learning."""
        # Create feedback submission
        feedback = UserFeedbackSubmission(
            tenant_id=test_tenant.id,
            user_id=test_user.id,
            feedback_type=FeedbackType.SEARCH_RESULT,
            target_id=uuid.uuid4(),
            rating=4,
            status="completed"
        )
        db_session.add(feedback)
        db_session.commit()
        
        # Process feedback
        result = await integration_service.process_feedback_for_learning(
            test_tenant.id, batch_size=10
        )
        
        assert result.signals_processed >= 0
        assert result.processing_time_seconds > 0
        assert isinstance(result.errors, list)


class TestLearningMonitoringService:
    """Test learning monitoring service functionality."""
    
    @pytest.fixture
    def monitoring_service(self, db_session: Session):
        """Create learning monitoring service instance."""
        return LearningMonitoringService(db_session)
    
    @pytest.mark.asyncio
    async def test_create_ab_test_experiment(
        self,
        monitoring_service: LearningMonitoringService,
        db_session: Session,
        test_tenant
    ):
        """Test creating A/B test experiment."""
        # Create algorithms
        control_alg = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.CHUNK_RANKING,
            model_type=LearningModelType.EXPONENTIAL_MOVING_AVERAGE,
            name="Control Algorithm"
        )
        treatment_alg = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.CHUNK_RANKING,
            model_type=LearningModelType.BAYESIAN_UPDATE,
            name="Treatment Algorithm"
        )
        db_session.add_all([control_alg, treatment_alg])
        db_session.commit()
        
        # Create experiment
        experiment = await monitoring_service.create_ab_test_experiment(
            tenant_id=test_tenant.id,
            experiment_name="Test Experiment",
            control_algorithm_id=control_alg.id,
            treatment_algorithm_id=treatment_alg.id,
            primary_metric="accuracy",
            success_threshold=0.05,
            description="Test A/B experiment"
        )
        
        assert experiment.id is not None
        assert experiment.experiment_name == "Test Experiment"
        assert experiment.status == "draft"
    
    @pytest.mark.asyncio
    async def test_perform_learning_health_check(
        self,
        monitoring_service: LearningMonitoringService,
        db_session: Session,
        test_tenant
    ):
        """Test learning health check."""
        # Create test algorithm
        algorithm = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.NEGATIVE_FEEDBACK,
            model_type=LearningModelType.REINFORCEMENT_LEARNING,
            name="Test Algorithm",
            accuracy_score=0.8
        )
        db_session.add(algorithm)
        db_session.commit()
        
        # Perform health check
        health_checks = await monitoring_service.perform_learning_health_check(
            test_tenant.id
        )
        
        assert len(health_checks) > 0
        health_check = health_checks[0]
        assert health_check.algorithm_id == algorithm.id
        assert health_check.health_score > 0
        assert health_check.status in ["healthy", "warning", "critical"]


class TestLearningAPI:
    """Test learning API endpoints."""
    
    def test_create_learning_algorithm_api(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        test_tenant
    ):
        """Test creating learning algorithm via API."""
        request_data = {
            "algorithm_type": "link_confidence",
            "model_type": "exponential_moving_average",
            "name": "API Test Algorithm",
            "description": "Test algorithm created via API",
            "learning_rate": 0.02,
            "validation_threshold": 0.1
        }
        
        response = client.post(
            "/api/v1/learning/algorithms",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "API Test Algorithm"
        assert data["algorithm_type"] == "link_confidence"
        assert data["learning_rate"] == 0.02
    
    def test_list_learning_algorithms_api(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        db_session: Session,
        test_tenant
    ):
        """Test listing learning algorithms via API."""
        # Create test algorithm
        algorithm = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.CHUNK_RANKING,
            model_type=LearningModelType.BAYESIAN_UPDATE,
            name="API List Test Algorithm"
        )
        db_session.add(algorithm)
        db_session.commit()
        
        response = client.get(
            "/api/v1/learning/algorithms",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["total"] > 0
        assert len(data["items"]) > 0
    
    def test_create_feedback_signal_api(
        self,
        client: TestClient,
        auth_headers: Dict[str, str]
    ):
        """Test creating feedback signal via API."""
        request_data = {
            "signal_type": "explicit_rating",
            "target_type": "chunk",
            "target_id": str(uuid.uuid4()),
            "signal_value": 3.5,
            "signal_strength": 0.8,
            "signal_confidence": 0.9,
            "query_context": "API test query"
        }
        
        response = client.post(
            "/api/v1/learning/signals",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["signal_type"] == "explicit_rating"
        assert data["signal_value"] == 3.5
        assert data["query_context"] == "API test query"
    
    def test_learning_health_check_api(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        db_session: Session,
        test_tenant
    ):
        """Test learning health check via API."""
        # Create test algorithm
        algorithm = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.CONTENT_QUALITY,
            model_type=LearningModelType.CONTENT_BASED_FILTERING,
            name="Health Check Test Algorithm"
        )
        db_session.add(algorithm)
        db_session.commit()
        
        response = client.get(
            "/api/v1/learning/health-check",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            health_check = data[0]
            assert "algorithm_id" in health_check
            assert "health_score" in health_check
            assert "status" in health_check
    
    def test_get_learning_insights_api(
        self,
        client: TestClient,
        auth_headers: Dict[str, str]
    ):
        """Test getting learning insights via API."""
        response = client.get(
            "/api/v1/learning/insights?time_period_hours=24",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "time_period_hours" in data
        assert "signal_statistics" in data
        assert "algorithm_performance" in data
        assert "total_signals_processed" in data
        assert "active_algorithms" in data


class TestLearningSystemIntegration:
    """Test end-to-end learning system integration."""
    
    @pytest.mark.asyncio
    async def test_complete_learning_workflow(
        self,
        db_session: Session,
        test_tenant,
        test_user
    ):
        """Test complete learning workflow from feedback to algorithm update."""
        # 1. Create learning algorithm
        algorithm = LearningAlgorithm(
            tenant_id=test_tenant.id,
            algorithm_type=LearningAlgorithmType.LINK_CONFIDENCE,
            model_type=LearningModelType.EXPONENTIAL_MOVING_AVERAGE,
            name="Integration Test Algorithm",
            learning_rate=0.01
        )
        db_session.add(algorithm)
        db_session.commit()
        
        # 2. Create feedback submission
        feedback = UserFeedbackSubmission(
            tenant_id=test_tenant.id,
            user_id=test_user.id,
            feedback_type=FeedbackType.LINK_QUALITY,
            target_id=uuid.uuid4(),
            rating=5,
            status="completed"
        )
        db_session.add(feedback)
        db_session.commit()
        
        # 3. Process feedback through integration service
        integration_service = LearningIntegrationService(db_session)
        result = await integration_service.process_feedback_for_learning(
            test_tenant.id, batch_size=10
        )
        
        # 4. Verify signals were created
        signals = db_session.query(FeedbackSignal).filter(
            FeedbackSignal.tenant_id == test_tenant.id
        ).all()
        
        assert len(signals) > 0
        assert result.signals_processed > 0
        
        # 5. Initialize and run learning service
        learning_service = LearningService(db_session)
        await learning_service.initialize_algorithms(test_tenant.id)
        
        # 6. Process signals through learning algorithms
        for signal in signals:
            if not signal.is_processed:
                updates = await learning_service.process_feedback_signal(
                    test_tenant.id, signal
                )
                assert len(updates) >= 0  # May be 0 if no relevant algorithms
        
        # 7. Verify learning occurred
        processed_signals = db_session.query(FeedbackSignal).filter(
            FeedbackSignal.tenant_id == test_tenant.id,
            FeedbackSignal.is_processed == True
        ).count()
        
        assert processed_signals > 0
