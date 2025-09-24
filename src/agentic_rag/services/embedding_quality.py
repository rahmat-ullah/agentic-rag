"""
Embedding Quality Validation and Monitoring

This module provides comprehensive quality validation, monitoring,
and alerting for embedding operations and results.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QualityMetric(str, Enum):
    """Types of quality metrics for embeddings."""
    
    DIMENSION_CONSISTENCY = "dimension_consistency"
    MAGNITUDE_RANGE = "magnitude_range"
    SIMILARITY_COHERENCE = "similarity_coherence"
    DISTRIBUTION_NORMALITY = "distribution_normality"
    OUTLIER_DETECTION = "outlier_detection"
    SEMANTIC_CONSISTENCY = "semantic_consistency"


class QualityStatus(str, Enum):
    """Quality validation status."""
    
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class QualityThresholds(BaseModel):
    """Thresholds for quality validation."""
    
    min_dimension: int = Field(default=1024, description="Minimum embedding dimension")
    max_dimension: int = Field(default=4096, description="Maximum embedding dimension")
    min_magnitude: float = Field(default=0.1, description="Minimum embedding magnitude")
    max_magnitude: float = Field(default=100.0, description="Maximum embedding magnitude")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity for related texts")
    outlier_threshold: float = Field(default=3.0, description="Standard deviations for outlier detection")
    consistency_threshold: float = Field(default=0.8, description="Minimum consistency score")


class QualityResult(BaseModel):
    """Result from quality validation."""
    
    metric: QualityMetric = Field(..., description="Quality metric evaluated")
    status: QualityStatus = Field(..., description="Validation status")
    score: float = Field(..., description="Quality score (0-1)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed results")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")


class EmbeddingQualityReport(BaseModel):
    """Comprehensive quality report for embeddings."""
    
    batch_id: str = Field(..., description="Batch identifier")
    total_embeddings: int = Field(..., description="Total number of embeddings")
    overall_status: QualityStatus = Field(..., description="Overall quality status")
    overall_score: float = Field(..., description="Overall quality score")
    metric_results: List[QualityResult] = Field(default_factory=list, description="Individual metric results")
    issues: List[str] = Field(default_factory=list, description="Quality issues found")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    processing_time: float = Field(..., description="Time taken for quality validation")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Report creation time")


class EmbeddingQualityValidator:
    """Validator for embedding quality with comprehensive metrics."""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        self.validation_history: deque = deque(maxlen=1000)
        
        logger.info(f"Embedding quality validator initialized with thresholds: {self.thresholds.model_dump()}")
    
    async def validate_embeddings(
        self,
        embeddings: List[List[float]],
        texts: Optional[List[str]] = None,
        batch_id: Optional[str] = None
    ) -> EmbeddingQualityReport:
        """
        Validate quality of embeddings with comprehensive metrics.
        
        Args:
            embeddings: List of embedding vectors
            texts: Optional list of source texts
            batch_id: Optional batch identifier
            
        Returns:
            EmbeddingQualityReport with validation results
        """
        start_time = time.time()
        batch_id = batch_id or f"batch_{int(time.time())}"
        
        logger.info(f"Starting quality validation for {len(embeddings)} embeddings")
        
        # Convert to numpy array for efficient computation
        embeddings_array = np.array(embeddings)
        
        # Run all quality checks
        metric_results = []
        
        # 1. Dimension consistency
        dim_result = await self._check_dimension_consistency(embeddings_array)
        metric_results.append(dim_result)
        
        # 2. Magnitude range
        mag_result = await self._check_magnitude_range(embeddings_array)
        metric_results.append(mag_result)
        
        # 3. Distribution normality
        dist_result = await self._check_distribution_normality(embeddings_array)
        metric_results.append(dist_result)
        
        # 4. Outlier detection
        outlier_result = await self._check_outliers(embeddings_array)
        metric_results.append(outlier_result)
        
        # 5. Similarity coherence (if texts provided)
        if texts and len(texts) == len(embeddings):
            sim_result = await self._check_similarity_coherence(embeddings_array, texts)
            metric_results.append(sim_result)
        
        # Calculate overall quality
        overall_score = np.mean([result.score for result in metric_results])
        overall_status = self._determine_overall_status(overall_score)
        
        # Generate issues and recommendations
        issues = []
        recommendations = []
        
        for result in metric_results:
            if result.status in [QualityStatus.POOR, QualityStatus.FAILED]:
                issues.extend(result.details.get("issues", []))
                recommendations.extend(result.details.get("recommendations", []))
        
        processing_time = time.time() - start_time
        
        # Create quality report
        report = EmbeddingQualityReport(
            batch_id=batch_id,
            total_embeddings=len(embeddings),
            overall_status=overall_status,
            overall_score=overall_score,
            metric_results=metric_results,
            issues=issues,
            recommendations=recommendations,
            processing_time=processing_time
        )
        
        # Store in history
        self.validation_history.append(report)
        
        logger.info(f"Quality validation completed: {overall_status.value} (score: {overall_score:.3f})")
        
        return report
    
    async def _check_dimension_consistency(self, embeddings: np.ndarray) -> QualityResult:
        """Check if all embeddings have consistent dimensions."""
        try:
            if len(embeddings.shape) != 2:
                return QualityResult(
                    metric=QualityMetric.DIMENSION_CONSISTENCY,
                    status=QualityStatus.FAILED,
                    score=0.0,
                    details={
                        "error": "Invalid embedding array shape",
                        "issues": ["Embeddings array is not 2-dimensional"],
                        "recommendations": ["Ensure embeddings are properly formatted"]
                    }
                )
            
            num_embeddings, dimension = embeddings.shape
            
            # Check dimension range
            if dimension < self.thresholds.min_dimension:
                status = QualityStatus.POOR
                score = 0.3
                issues = [f"Dimension {dimension} below minimum {self.thresholds.min_dimension}"]
                recommendations = ["Consider using a higher-dimensional embedding model"]
            elif dimension > self.thresholds.max_dimension:
                status = QualityStatus.ACCEPTABLE
                score = 0.7
                issues = [f"Dimension {dimension} above recommended maximum {self.thresholds.max_dimension}"]
                recommendations = ["Consider dimensionality reduction for efficiency"]
            else:
                status = QualityStatus.EXCELLENT
                score = 1.0
                issues = []
                recommendations = []
            
            return QualityResult(
                metric=QualityMetric.DIMENSION_CONSISTENCY,
                status=status,
                score=score,
                details={
                    "dimension": dimension,
                    "num_embeddings": num_embeddings,
                    "issues": issues,
                    "recommendations": recommendations
                }
            )
            
        except Exception as e:
            logger.error(f"Dimension consistency check failed: {e}")
            return QualityResult(
                metric=QualityMetric.DIMENSION_CONSISTENCY,
                status=QualityStatus.FAILED,
                score=0.0,
                details={"error": str(e)}
            )
    
    async def _check_magnitude_range(self, embeddings: np.ndarray) -> QualityResult:
        """Check if embedding magnitudes are within expected range."""
        try:
            # Calculate L2 norms (magnitudes)
            magnitudes = np.linalg.norm(embeddings, axis=1)
            
            min_mag = np.min(magnitudes)
            max_mag = np.max(magnitudes)
            mean_mag = np.mean(magnitudes)
            std_mag = np.std(magnitudes)
            
            issues = []
            recommendations = []
            
            # Check magnitude range
            if min_mag < self.thresholds.min_magnitude:
                issues.append(f"Some embeddings have very low magnitude: {min_mag:.4f}")
                recommendations.append("Check for zero or near-zero embeddings")
            
            if max_mag > self.thresholds.max_magnitude:
                issues.append(f"Some embeddings have very high magnitude: {max_mag:.4f}")
                recommendations.append("Consider normalizing embeddings")
            
            # Determine status based on issues
            if len(issues) == 0:
                status = QualityStatus.EXCELLENT
                score = 1.0
            elif len(issues) == 1:
                status = QualityStatus.GOOD
                score = 0.8
            else:
                status = QualityStatus.ACCEPTABLE
                score = 0.6
            
            return QualityResult(
                metric=QualityMetric.MAGNITUDE_RANGE,
                status=status,
                score=score,
                details={
                    "min_magnitude": float(min_mag),
                    "max_magnitude": float(max_mag),
                    "mean_magnitude": float(mean_mag),
                    "std_magnitude": float(std_mag),
                    "issues": issues,
                    "recommendations": recommendations
                }
            )
            
        except Exception as e:
            logger.error(f"Magnitude range check failed: {e}")
            return QualityResult(
                metric=QualityMetric.MAGNITUDE_RANGE,
                status=QualityStatus.FAILED,
                score=0.0,
                details={"error": str(e)}
            )
    
    async def _check_distribution_normality(self, embeddings: np.ndarray) -> QualityResult:
        """Check if embedding values follow a reasonable distribution."""
        try:
            # Flatten all embedding values
            all_values = embeddings.flatten()
            
            # Calculate distribution statistics
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            skewness = self._calculate_skewness(all_values)
            kurtosis = self._calculate_kurtosis(all_values)
            
            issues = []
            recommendations = []
            
            # Check for extreme skewness
            if abs(skewness) > 2.0:
                issues.append(f"High skewness detected: {skewness:.3f}")
                recommendations.append("Check for bias in embedding generation")
            
            # Check for extreme kurtosis
            if abs(kurtosis) > 5.0:
                issues.append(f"High kurtosis detected: {kurtosis:.3f}")
                recommendations.append("Check for outliers or distribution issues")
            
            # Determine status
            if len(issues) == 0:
                status = QualityStatus.EXCELLENT
                score = 1.0
            elif len(issues) == 1:
                status = QualityStatus.GOOD
                score = 0.7
            else:
                status = QualityStatus.ACCEPTABLE
                score = 0.5
            
            return QualityResult(
                metric=QualityMetric.DISTRIBUTION_NORMALITY,
                status=status,
                score=score,
                details={
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "issues": issues,
                    "recommendations": recommendations
                }
            )
            
        except Exception as e:
            logger.error(f"Distribution normality check failed: {e}")
            return QualityResult(
                metric=QualityMetric.DISTRIBUTION_NORMALITY,
                status=QualityStatus.FAILED,
                score=0.0,
                details={"error": str(e)}
            )
    
    async def _check_outliers(self, embeddings: np.ndarray) -> QualityResult:
        """Detect outlier embeddings using statistical methods."""
        try:
            # Calculate centroid
            centroid = np.mean(embeddings, axis=0)
            
            # Calculate distances from centroid
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            
            # Calculate outlier threshold using standard deviation
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            outlier_threshold = mean_distance + (self.thresholds.outlier_threshold * std_distance)
            
            # Find outliers
            outlier_indices = np.where(distances > outlier_threshold)[0]
            outlier_count = len(outlier_indices)
            outlier_percentage = (outlier_count / len(embeddings)) * 100
            
            issues = []
            recommendations = []
            
            if outlier_percentage > 10:
                issues.append(f"High outlier percentage: {outlier_percentage:.1f}%")
                recommendations.append("Review input texts for quality issues")
                status = QualityStatus.POOR
                score = 0.4
            elif outlier_percentage > 5:
                issues.append(f"Moderate outlier percentage: {outlier_percentage:.1f}%")
                recommendations.append("Consider reviewing outlier embeddings")
                status = QualityStatus.ACCEPTABLE
                score = 0.7
            else:
                status = QualityStatus.EXCELLENT
                score = 1.0
            
            return QualityResult(
                metric=QualityMetric.OUTLIER_DETECTION,
                status=status,
                score=score,
                details={
                    "outlier_count": int(outlier_count),
                    "outlier_percentage": float(outlier_percentage),
                    "outlier_threshold": float(outlier_threshold),
                    "mean_distance": float(mean_distance),
                    "std_distance": float(std_distance),
                    "issues": issues,
                    "recommendations": recommendations
                }
            )
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return QualityResult(
                metric=QualityMetric.OUTLIER_DETECTION,
                status=QualityStatus.FAILED,
                score=0.0,
                details={"error": str(e)}
            )
    
    async def _check_similarity_coherence(self, embeddings: np.ndarray, texts: List[str]) -> QualityResult:
        """Check if similar texts have similar embeddings."""
        try:
            # Sample pairs for similarity checking (limit for performance)
            max_pairs = min(100, len(texts) // 2)
            coherence_scores = []
            
            for i in range(max_pairs):
                # Compare consecutive texts (assuming some similarity)
                if i + 1 < len(texts):
                    text_similarity = self._calculate_text_similarity(texts[i], texts[i + 1])
                    embedding_similarity = self._calculate_cosine_similarity(
                        embeddings[i], embeddings[i + 1]
                    )
                    
                    # Coherence is correlation between text and embedding similarity
                    coherence_scores.append(abs(text_similarity - embedding_similarity))
            
            if coherence_scores:
                avg_coherence = 1.0 - np.mean(coherence_scores)  # Invert for score
                
                if avg_coherence >= self.thresholds.consistency_threshold:
                    status = QualityStatus.EXCELLENT
                    score = avg_coherence
                elif avg_coherence >= 0.6:
                    status = QualityStatus.GOOD
                    score = avg_coherence
                elif avg_coherence >= 0.4:
                    status = QualityStatus.ACCEPTABLE
                    score = avg_coherence
                else:
                    status = QualityStatus.POOR
                    score = avg_coherence
            else:
                status = QualityStatus.ACCEPTABLE
                score = 0.5
                avg_coherence = 0.5
            
            return QualityResult(
                metric=QualityMetric.SIMILARITY_COHERENCE,
                status=status,
                score=score,
                details={
                    "coherence_score": float(avg_coherence),
                    "pairs_checked": len(coherence_scores),
                    "issues": [] if status != QualityStatus.POOR else ["Low similarity coherence"],
                    "recommendations": [] if status != QualityStatus.POOR else ["Review embedding model performance"]
                }
            )
            
        except Exception as e:
            logger.error(f"Similarity coherence check failed: {e}")
            return QualityResult(
                metric=QualityMetric.SIMILARITY_COHERENCE,
                status=QualityStatus.FAILED,
                score=0.0,
                details={"error": str(e)}
            )
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard similarity)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _determine_overall_status(self, score: float) -> QualityStatus:
        """Determine overall quality status from score."""
        if score >= 0.9:
            return QualityStatus.EXCELLENT
        elif score >= 0.8:
            return QualityStatus.GOOD
        elif score >= 0.6:
            return QualityStatus.ACCEPTABLE
        elif score >= 0.4:
            return QualityStatus.POOR
        else:
            return QualityStatus.FAILED
    
    def get_validation_history(self, limit: Optional[int] = None) -> List[EmbeddingQualityReport]:
        """Get validation history."""
        history = list(self.validation_history)
        if limit:
            history = history[-limit:]
        return history
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality statistics from validation history."""
        if not self.validation_history:
            return {"message": "No validation history available"}
        
        recent_reports = list(self.validation_history)[-100:]  # Last 100 reports
        
        # Calculate statistics
        overall_scores = [report.overall_score for report in recent_reports]
        status_counts = {}
        
        for report in recent_reports:
            status = report.overall_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_validations": len(self.validation_history),
            "recent_validations": len(recent_reports),
            "average_score": np.mean(overall_scores) if overall_scores else 0.0,
            "min_score": np.min(overall_scores) if overall_scores else 0.0,
            "max_score": np.max(overall_scores) if overall_scores else 0.0,
            "status_distribution": status_counts,
            "last_validation": recent_reports[-1].created_at.isoformat() if recent_reports else None
        }


# Global quality validator instance
_quality_validator: Optional[EmbeddingQualityValidator] = None


async def get_quality_validator() -> EmbeddingQualityValidator:
    """Get or create the global quality validator instance."""
    global _quality_validator
    
    if _quality_validator is None:
        _quality_validator = EmbeddingQualityValidator()
    
    return _quality_validator


async def close_quality_validator() -> None:
    """Close the global quality validator instance."""
    global _quality_validator
    
    if _quality_validator:
        _quality_validator = None
