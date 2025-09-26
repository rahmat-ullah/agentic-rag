"""
Learning Algorithm API Routes for Sprint 6 Story 6-03

This module contains FastAPI routes for learning algorithm management,
monitoring, and configuration.
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from agentic_rag.api.dependencies.auth import get_current_user, require_permissions
from agentic_rag.api.dependencies.database import get_db_session
from agentic_rag.api.dependencies.tenant import get_tenant_context
from agentic_rag.models.auth import User
from agentic_rag.models.tenant import TenantContext
from agentic_rag.models.learning import (
    LearningAlgorithm,
    FeedbackSignal,
    LearningPerformanceMetric,
    ABTestExperiment,
    LearningAlgorithmType,
    LearningStatus
)
from agentic_rag.schemas.learning import (
    CreateLearningAlgorithmRequest,
    UpdateLearningAlgorithmRequest,
    CreateFeedbackSignalRequest,
    CreateABTestExperimentRequest,
    LearningAlgorithmResponse,
    FeedbackSignalResponse,
    LearningPerformanceMetricResponse,
    ABTestExperimentResponse,
    LearningValidationResultResponse,
    ABTestResultResponse,
    LearningHealthCheckResponse,
    LearningIntegrationResultResponse,
    LearningInsightsResponse,
    PaginatedLearningAlgorithmsResponse,
    PaginatedFeedbackSignalsResponse,
    PaginatedPerformanceMetricsResponse,
    PaginatedABTestExperimentsResponse
)
from agentic_rag.services.learning_service import get_learning_service
from agentic_rag.services.learning_integration_service import get_learning_integration_service
from agentic_rag.services.learning_monitoring_service import get_learning_monitoring_service
from agentic_rag.api.exceptions import ValidationError, NotFoundError

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/learning", tags=["Learning Algorithms"])


@router.post(
    "/algorithms",
    response_model=LearningAlgorithmResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Learning Algorithm",
    description="Create a new learning algorithm configuration"
)
async def create_learning_algorithm(
    request: CreateLearningAlgorithmRequest,
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:write"]))
):
    """Create a new learning algorithm."""
    try:
        # Create algorithm
        algorithm = LearningAlgorithm(
            tenant_id=tenant_context.tenant_id,
            algorithm_type=request.algorithm_type,
            model_type=request.model_type,
            name=request.name,
            description=request.description,
            learning_rate=request.learning_rate,
            validation_threshold=request.validation_threshold,
            decay_factor=request.decay_factor,
            regularization_strength=request.regularization_strength,
            is_enabled=request.is_enabled,
            auto_update=request.auto_update,
            validation_frequency_hours=request.validation_frequency_hours,
            algorithm_metadata=request.algorithm_metadata
        )
        
        db.add(algorithm)
        db.commit()
        db.refresh(algorithm)
        
        logger.info(
            "learning_algorithm_created",
            algorithm_id=algorithm.id,
            algorithm_type=algorithm.algorithm_type,
            user_id=current_user.id,
            tenant_id=tenant_context.tenant_id
        )
        
        return LearningAlgorithmResponse.model_validate(algorithm)
        
    except Exception as e:
        db.rollback()
        logger.error("learning_algorithm_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create learning algorithm: {str(e)}"
        )


@router.get(
    "/algorithms",
    response_model=PaginatedLearningAlgorithmsResponse,
    summary="List Learning Algorithms",
    description="Get paginated list of learning algorithms with filtering"
)
async def list_learning_algorithms(
    algorithm_type: Optional[LearningAlgorithmType] = Query(None, description="Filter by algorithm type"),
    status: Optional[LearningStatus] = Query(None, description="Filter by status"),
    is_enabled: Optional[bool] = Query(None, description="Filter by enabled status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:read"]))
):
    """Get paginated list of learning algorithms."""
    try:
        # Build query
        query = db.query(LearningAlgorithm).filter(
            LearningAlgorithm.tenant_id == tenant_context.tenant_id
        )
        
        if algorithm_type:
            query = query.filter(LearningAlgorithm.algorithm_type == algorithm_type)
        if status:
            query = query.filter(LearningAlgorithm.status == status)
        if is_enabled is not None:
            query = query.filter(LearningAlgorithm.is_enabled == is_enabled)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        algorithms = query.order_by(desc(LearningAlgorithm.created_at)).offset(offset).limit(page_size).all()
        
        return PaginatedLearningAlgorithmsResponse(
            items=[LearningAlgorithmResponse.model_validate(alg) for alg in algorithms],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )
        
    except Exception as e:
        logger.error("learning_algorithms_listing_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list learning algorithms: {str(e)}"
        )


@router.get(
    "/algorithms/{algorithm_id}",
    response_model=LearningAlgorithmResponse,
    summary="Get Learning Algorithm",
    description="Get details of a specific learning algorithm"
)
async def get_learning_algorithm(
    algorithm_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:read"]))
):
    """Get details of a specific learning algorithm."""
    try:
        algorithm = db.query(LearningAlgorithm).filter(
            and_(
                LearningAlgorithm.id == algorithm_id,
                LearningAlgorithm.tenant_id == tenant_context.tenant_id
            )
        ).first()
        
        if not algorithm:
            raise NotFoundError(f"Learning algorithm {algorithm_id} not found")
        
        return LearningAlgorithmResponse.model_validate(algorithm)
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error("learning_algorithm_retrieval_failed", error=str(e), algorithm_id=algorithm_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get learning algorithm: {str(e)}"
        )


@router.put(
    "/algorithms/{algorithm_id}",
    response_model=LearningAlgorithmResponse,
    summary="Update Learning Algorithm",
    description="Update learning algorithm configuration"
)
async def update_learning_algorithm(
    algorithm_id: uuid.UUID,
    request: UpdateLearningAlgorithmRequest,
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:write"]))
):
    """Update learning algorithm configuration."""
    try:
        algorithm = db.query(LearningAlgorithm).filter(
            and_(
                LearningAlgorithm.id == algorithm_id,
                LearningAlgorithm.tenant_id == tenant_context.tenant_id
            )
        ).first()
        
        if not algorithm:
            raise NotFoundError(f"Learning algorithm {algorithm_id} not found")
        
        # Update fields
        update_data = request.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(algorithm, field, value)
        
        algorithm.updated_at = datetime.now()
        
        db.commit()
        db.refresh(algorithm)
        
        logger.info(
            "learning_algorithm_updated",
            algorithm_id=algorithm_id,
            user_id=current_user.id,
            updated_fields=list(update_data.keys())
        )
        
        return LearningAlgorithmResponse.model_validate(algorithm)
        
    except NotFoundError:
        raise
    except Exception as e:
        db.rollback()
        logger.error("learning_algorithm_update_failed", error=str(e), algorithm_id=algorithm_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update learning algorithm: {str(e)}"
        )


@router.delete(
    "/algorithms/{algorithm_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Learning Algorithm",
    description="Delete a learning algorithm"
)
async def delete_learning_algorithm(
    algorithm_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:delete"]))
):
    """Delete a learning algorithm."""
    try:
        algorithm = db.query(LearningAlgorithm).filter(
            and_(
                LearningAlgorithm.id == algorithm_id,
                LearningAlgorithm.tenant_id == tenant_context.tenant_id
            )
        ).first()
        
        if not algorithm:
            raise NotFoundError(f"Learning algorithm {algorithm_id} not found")
        
        db.delete(algorithm)
        db.commit()
        
        logger.info(
            "learning_algorithm_deleted",
            algorithm_id=algorithm_id,
            user_id=current_user.id
        )
        
    except NotFoundError:
        raise
    except Exception as e:
        db.rollback()
        logger.error("learning_algorithm_deletion_failed", error=str(e), algorithm_id=algorithm_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete learning algorithm: {str(e)}"
        )


@router.post(
    "/algorithms/{algorithm_id}/validate",
    response_model=LearningValidationResultResponse,
    summary="Validate Algorithm Performance",
    description="Validate learning algorithm performance against baseline"
)
async def validate_algorithm_performance(
    algorithm_id: uuid.UUID,
    validation_period_hours: int = Query(24, ge=1, description="Validation period in hours"),
    baseline_period_hours: int = Query(168, ge=1, description="Baseline period in hours"),
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:validate"]))
):
    """Validate learning algorithm performance."""
    try:
        monitoring_service = get_learning_monitoring_service(db)
        
        result = await monitoring_service.validate_algorithm_performance(
            algorithm_id, validation_period_hours, baseline_period_hours
        )
        
        logger.info(
            "algorithm_performance_validated",
            algorithm_id=algorithm_id,
            status=result.status,
            user_id=current_user.id
        )
        
        return LearningValidationResultResponse(
            status=result.status.value,
            score=result.score,
            improvement_percentage=result.improvement_percentage,
            statistical_significance=result.statistical_significance,
            confidence_interval=list(result.confidence_interval),
            validation_metadata=result.validation_metadata,
            recommendations=result.recommendations
        )
        
    except Exception as e:
        logger.error("algorithm_validation_failed", error=str(e), algorithm_id=algorithm_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate algorithm performance: {str(e)}"
        )


@router.post(
    "/signals",
    response_model=FeedbackSignalResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Feedback Signal",
    description="Create a feedback signal for learning algorithms"
)
async def create_feedback_signal(
    request: CreateFeedbackSignalRequest,
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:write"]))
):
    """Create a feedback signal for learning algorithms."""
    try:
        integration_service = get_learning_integration_service(db)
        
        signal = await integration_service.create_feedback_signal(
            tenant_id=tenant_context.tenant_id,
            signal_type=request.signal_type,
            target_type=request.target_type,
            target_id=request.target_id,
            signal_value=request.signal_value,
            user_id=request.user_id or current_user.id,
            session_id=request.session_id,
            query_context=request.query_context,
            metadata=request.signal_metadata
        )
        
        logger.info(
            "feedback_signal_created",
            signal_id=signal.id,
            signal_type=signal.signal_type,
            user_id=current_user.id
        )
        
        return FeedbackSignalResponse.model_validate(signal)
        
    except Exception as e:
        logger.error("feedback_signal_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create feedback signal: {str(e)}"
        )


@router.post(
    "/process-feedback",
    response_model=LearningIntegrationResultResponse,
    summary="Process Feedback for Learning",
    description="Process feedback submissions to generate learning signals"
)
async def process_feedback_for_learning(
    batch_size: int = Query(100, ge=1, le=1000, description="Batch size for processing"),
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:process"]))
):
    """Process feedback submissions to generate learning signals."""
    try:
        integration_service = get_learning_integration_service(db)
        
        result = await integration_service.process_feedback_for_learning(
            tenant_context.tenant_id, batch_size
        )
        
        logger.info(
            "feedback_processing_completed",
            signals_processed=result.signals_processed,
            algorithms_updated=result.algorithms_updated,
            user_id=current_user.id
        )
        
        return LearningIntegrationResultResponse.model_validate(result)
        
    except Exception as e:
        logger.error("feedback_processing_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process feedback for learning: {str(e)}"
        )


@router.post(
    "/experiments",
    response_model=ABTestExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create A/B Test Experiment",
    description="Create a new A/B test experiment for learning algorithms"
)
async def create_ab_test_experiment(
    request: CreateABTestExperimentRequest,
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:experiment"]))
):
    """Create a new A/B test experiment."""
    try:
        monitoring_service = get_learning_monitoring_service(db)
        
        experiment = await monitoring_service.create_ab_test_experiment(
            tenant_id=tenant_context.tenant_id,
            experiment_name=request.experiment_name,
            control_algorithm_id=request.control_algorithm_id,
            treatment_algorithm_id=request.treatment_algorithm_id,
            primary_metric=request.primary_metric,
            success_threshold=request.success_threshold,
            traffic_split_percentage=request.traffic_split_percentage,
            minimum_sample_size=request.minimum_sample_size,
            confidence_level=request.confidence_level,
            description=request.description,
            hypothesis=request.hypothesis
        )
        
        logger.info(
            "ab_test_experiment_created",
            experiment_id=experiment.id,
            experiment_name=experiment.experiment_name,
            user_id=current_user.id
        )
        
        return ABTestExperimentResponse.model_validate(experiment)
        
    except Exception as e:
        logger.error("ab_test_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create A/B test experiment: {str(e)}"
        )


@router.post(
    "/experiments/{experiment_id}/start",
    response_model=ABTestExperimentResponse,
    summary="Start A/B Test Experiment",
    description="Start an A/B test experiment"
)
async def start_ab_test_experiment(
    experiment_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:experiment"]))
):
    """Start an A/B test experiment."""
    try:
        monitoring_service = get_learning_monitoring_service(db)
        
        experiment = await monitoring_service.start_ab_test_experiment(experiment_id)
        
        logger.info(
            "ab_test_experiment_started",
            experiment_id=experiment_id,
            user_id=current_user.id
        )
        
        return ABTestExperimentResponse.model_validate(experiment)
        
    except Exception as e:
        logger.error("ab_test_start_failed", error=str(e), experiment_id=experiment_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start A/B test experiment: {str(e)}"
        )


@router.get(
    "/experiments/{experiment_id}/results",
    response_model=ABTestResultResponse,
    summary="Get A/B Test Results",
    description="Get results of an A/B test experiment"
)
async def get_ab_test_results(
    experiment_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:read"]))
):
    """Get results of an A/B test experiment."""
    try:
        monitoring_service = get_learning_monitoring_service(db)
        
        result = await monitoring_service.analyze_ab_test_results(experiment_id)
        
        return ABTestResultResponse.model_validate(result)
        
    except Exception as e:
        logger.error("ab_test_results_failed", error=str(e), experiment_id=experiment_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get A/B test results: {str(e)}"
        )


@router.get(
    "/health-check",
    response_model=List[LearningHealthCheckResponse],
    summary="Learning Health Check",
    description="Perform health check on all learning algorithms"
)
async def learning_health_check(
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:read"]))
):
    """Perform health check on all learning algorithms."""
    try:
        monitoring_service = get_learning_monitoring_service(db)
        
        health_checks = await monitoring_service.perform_learning_health_check(
            tenant_context.tenant_id
        )
        
        return [LearningHealthCheckResponse.model_validate(hc) for hc in health_checks]
        
    except Exception as e:
        logger.error("learning_health_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform learning health check: {str(e)}"
        )


@router.get(
    "/insights",
    response_model=LearningInsightsResponse,
    summary="Get Learning Insights",
    description="Get insights from learning algorithm performance"
)
async def get_learning_insights(
    time_period_hours: int = Query(24, ge=1, description="Time period for insights in hours"),
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db_session),
    _: None = Depends(require_permissions(["learning:read"]))
):
    """Get insights from learning algorithm performance."""
    try:
        integration_service = get_learning_integration_service(db)
        
        insights = await integration_service.get_learning_insights(
            tenant_context.tenant_id, time_period_hours
        )
        
        return LearningInsightsResponse.model_validate(insights)
        
    except Exception as e:
        logger.error("learning_insights_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get learning insights: {str(e)}"
        )
