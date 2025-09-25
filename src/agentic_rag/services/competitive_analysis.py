"""
Competitive Analysis Engine

This module provides competitive pricing analysis with market comparison,
trend analysis, outlier detection, and benchmarking capabilities for
procurement scenarios.
"""

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, validator

from agentic_rag.config import get_settings
from agentic_rag.models.database import UserRole
from agentic_rag.services.pricing_extraction import PricingItem, Currency

logger = structlog.get_logger(__name__)


class ComparisonType(str, Enum):
    """Types of competitive comparisons."""
    
    PRICE_COMPARISON = "price_comparison"
    VENDOR_COMPARISON = "vendor_comparison"
    CATEGORY_COMPARISON = "category_comparison"
    HISTORICAL_COMPARISON = "historical_comparison"
    MARKET_COMPARISON = "market_comparison"


class TrendDirection(str, Enum):
    """Direction of pricing trends."""
    
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class OutlierType(str, Enum):
    """Types of pricing outliers."""
    
    HIGH_OUTLIER = "high_outlier"      # Significantly higher than market
    LOW_OUTLIER = "low_outlier"        # Significantly lower than market
    QUALITY_OUTLIER = "quality_outlier"  # Different quality/specification
    VENDOR_OUTLIER = "vendor_outlier"    # Vendor-specific pricing anomaly


class BenchmarkCategory(str, Enum):
    """Categories for benchmarking."""
    
    BEST_PRICE = "best_price"
    AVERAGE_PRICE = "average_price"
    PREMIUM_PRICE = "premium_price"
    MARKET_LEADER = "market_leader"
    VALUE_FOR_MONEY = "value_for_money"


@dataclass
class CompetitiveMetrics:
    """Competitive analysis metrics."""
    
    # Basic statistics
    min_price: Decimal
    max_price: Decimal
    avg_price: Decimal
    median_price: Decimal
    std_deviation: Decimal
    
    # Market position
    market_position: str  # "below_market", "at_market", "above_market"
    percentile_rank: float  # 0-100
    
    # Competitive advantage
    price_advantage: Decimal  # Difference from average
    savings_potential: Decimal  # Potential savings vs highest
    
    # Risk indicators
    price_volatility: float
    outlier_risk: bool
    
    # Metadata
    sample_size: int
    confidence_level: float


class PricingTrend(BaseModel):
    """Pricing trend analysis."""
    
    # Trend identification
    trend_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique trend ID")
    item_category: str = Field(..., description="Item category")
    vendor: Optional[str] = Field(default=None, description="Vendor name")
    
    # Trend data
    direction: TrendDirection = Field(..., description="Trend direction")
    magnitude: float = Field(..., description="Trend magnitude (percentage change)")
    duration_days: int = Field(..., description="Trend duration in days")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Trend confidence")
    
    # Price points
    start_price: Decimal = Field(..., description="Starting price")
    end_price: Decimal = Field(..., description="Ending price")
    peak_price: Optional[Decimal] = Field(default=None, description="Peak price in period")
    trough_price: Optional[Decimal] = Field(default=None, description="Lowest price in period")
    
    # Analysis period
    start_date: datetime = Field(..., description="Trend start date")
    end_date: datetime = Field(..., description="Trend end date")
    
    # Predictions
    predicted_next_price: Optional[Decimal] = Field(default=None, description="Predicted next price")
    prediction_confidence: float = Field(default=0.0, description="Prediction confidence")
    
    # Metadata
    data_points: int = Field(..., description="Number of data points used")
    analysis_method: str = Field(default="statistical", description="Analysis method used")


class PricingOutlier(BaseModel):
    """Pricing outlier detection result."""
    
    # Outlier identification
    outlier_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique outlier ID")
    item: PricingItem = Field(..., description="Outlier pricing item")
    outlier_type: OutlierType = Field(..., description="Type of outlier")
    
    # Statistical measures
    z_score: float = Field(..., description="Z-score from mean")
    percentile: float = Field(..., description="Percentile position")
    deviation_amount: Decimal = Field(..., description="Deviation from expected price")
    deviation_percentage: float = Field(..., description="Deviation percentage")
    
    # Context
    market_average: Decimal = Field(..., description="Market average price")
    expected_range_min: Decimal = Field(..., description="Expected minimum price")
    expected_range_max: Decimal = Field(..., description="Expected maximum price")
    
    # Analysis
    confidence: float = Field(..., ge=0.0, le=1.0, description="Outlier detection confidence")
    explanation: str = Field(..., description="Explanation for outlier status")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    
    # Metadata
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp"
    )


class CompetitiveBenchmark(BaseModel):
    """Competitive benchmarking result."""
    
    # Benchmark identification
    benchmark_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique benchmark ID")
    category: BenchmarkCategory = Field(..., description="Benchmark category")
    item_category: str = Field(..., description="Item category")
    
    # Benchmark values
    benchmark_price: Decimal = Field(..., description="Benchmark price")
    benchmark_vendor: Optional[str] = Field(default=None, description="Benchmark vendor")
    
    # Competitive position
    your_price: Decimal = Field(..., description="Your current price")
    price_difference: Decimal = Field(..., description="Difference from benchmark")
    percentage_difference: float = Field(..., description="Percentage difference")
    competitive_position: str = Field(..., description="Competitive position description")
    
    # Market context
    market_share_estimate: Optional[float] = Field(default=None, description="Estimated market share")
    quality_adjustment: Optional[float] = Field(default=None, description="Quality adjustment factor")
    
    # Recommendations
    action_required: bool = Field(..., description="Whether action is required")
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")
    priority: str = Field(default="medium", description="Priority level")
    
    # Metadata
    analysis_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis date"
    )
    data_freshness: int = Field(..., description="Data freshness in hours")


class CompetitiveAnalysisConfig(BaseModel):
    """Configuration for competitive analysis."""
    
    # Analysis settings
    enabled_comparisons: Set[ComparisonType] = Field(
        default_factory=lambda: set(ComparisonType),
        description="Enabled comparison types"
    )
    
    # Statistical settings
    outlier_threshold_z_score: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Z-score threshold for outlier detection"
    )
    trend_min_data_points: int = Field(
        default=5,
        ge=3,
        description="Minimum data points for trend analysis"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    
    # Time settings
    trend_analysis_days: int = Field(
        default=90,
        ge=7,
        description="Days to analyze for trends"
    )
    benchmark_freshness_hours: int = Field(
        default=24,
        ge=1,
        description="Maximum age of benchmark data in hours"
    )
    
    # Currency settings
    base_currency: Currency = Field(
        default=Currency.USD,
        description="Base currency for analysis"
    )
    currency_conversion_enabled: bool = Field(
        default=True,
        description="Enable currency conversion"
    )
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=1800, description="Cache TTL in seconds")
    max_processing_time_seconds: int = Field(
        default=60,
        description="Maximum processing time"
    )


class CompetitiveAnalysisResult(BaseModel):
    """Result of competitive analysis."""
    
    # Analysis results
    competitive_metrics: Optional[CompetitiveMetrics] = Field(
        default=None,
        description="Competitive metrics"
    )
    pricing_trends: List[PricingTrend] = Field(
        default_factory=list,
        description="Identified pricing trends"
    )
    outliers: List[PricingOutlier] = Field(
        default_factory=list,
        description="Detected pricing outliers"
    )
    benchmarks: List[CompetitiveBenchmark] = Field(
        default_factory=list,
        description="Competitive benchmarks"
    )
    
    # Summary statistics
    total_items_analyzed: int = Field(default=0, description="Total items analyzed")
    vendors_analyzed: int = Field(default=0, description="Number of vendors analyzed")
    categories_analyzed: int = Field(default=0, description="Number of categories analyzed")
    
    # Quality indicators
    analysis_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall analysis confidence"
    )
    data_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Data completeness score"
    )
    
    # Recommendations
    key_insights: List[str] = Field(
        default_factory=list,
        description="Key insights from analysis"
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Recommended action items"
    )
    
    # Performance metrics
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )
    
    # Metadata
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    config_used: Optional[CompetitiveAnalysisConfig] = Field(
        default=None,
        description="Configuration used for analysis"
    )


class CompetitiveAnalysisService:
    """Service for competitive pricing analysis."""
    
    def __init__(self, config: Optional[CompetitiveAnalysisConfig] = None):
        self.config = config or CompetitiveAnalysisConfig()
        self.settings = get_settings()
        
        # Performance tracking
        self._stats = {
            "total_analyses": 0,
            "total_items_analyzed": 0,
            "total_outliers_detected": 0,
            "total_trends_identified": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Cache for results
        self._cache: Dict[str, Any] = {}
        
        logger.info("Competitive analysis service initialized")
    
    def analyze_competitive_pricing(self, pricing_items: List[PricingItem], 
                                  comparison_type: ComparisonType = ComparisonType.PRICE_COMPARISON) -> CompetitiveAnalysisResult:
        """Perform competitive pricing analysis."""
        
        start_time = time.time()
        
        try:
            # Check cache
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(pricing_items, comparison_type)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self._stats["cache_hits"] += 1
                    return cached_result
                self._stats["cache_misses"] += 1
            
            # Initialize result
            result = CompetitiveAnalysisResult(config_used=self.config)
            
            # Validate input
            if not pricing_items:
                logger.warning("No pricing items provided for analysis")
                return result
            
            # Perform analysis based on comparison type
            if comparison_type == ComparisonType.PRICE_COMPARISON:
                result = self._analyze_price_comparison(pricing_items, result)
            elif comparison_type == ComparisonType.VENDOR_COMPARISON:
                result = self._analyze_vendor_comparison(pricing_items, result)
            elif comparison_type == ComparisonType.CATEGORY_COMPARISON:
                result = self._analyze_category_comparison(pricing_items, result)
            elif comparison_type == ComparisonType.HISTORICAL_COMPARISON:
                result = self._analyze_historical_comparison(pricing_items, result)
            elif comparison_type == ComparisonType.MARKET_COMPARISON:
                result = self._analyze_market_comparison(pricing_items, result)
            
            # Calculate summary statistics
            result.total_items_analyzed = len(pricing_items)
            result.vendors_analyzed = len(set(item.vendor for item in pricing_items if item.vendor))
            result.categories_analyzed = len(set(item.category for item in pricing_items if item.category))
            
            # Generate insights and recommendations
            result.key_insights = self._generate_insights(result)
            result.action_items = self._generate_action_items(result)
            
            # Calculate quality metrics
            result.analysis_confidence = self._calculate_analysis_confidence(result)
            result.data_completeness = self._calculate_data_completeness(pricing_items)
            
            # Performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time_ms
            
            # Update statistics
            self._stats["total_analyses"] += 1
            self._stats["total_items_analyzed"] += len(pricing_items)
            self._stats["total_outliers_detected"] += len(result.outliers)
            self._stats["total_trends_identified"] += len(result.pricing_trends)
            self._stats["total_processing_time_ms"] += processing_time_ms
            
            # Cache result
            if self.config.enable_caching:
                self._cache_result(cache_key, result)
            
            logger.info(
                "Competitive analysis completed",
                comparison_type=comparison_type.value,
                items_analyzed=len(pricing_items),
                outliers_detected=len(result.outliers),
                trends_identified=len(result.pricing_trends),
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error("Competitive analysis failed", error=str(e), comparison_type=comparison_type.value)
            raise

    def _analyze_price_comparison(self, pricing_items: List[PricingItem],
                                result: CompetitiveAnalysisResult) -> CompetitiveAnalysisResult:
        """Analyze price comparison across items."""

        try:
            # Extract prices for analysis
            prices = []
            for item in pricing_items:
                if item.total_price:
                    prices.append(float(item.total_price))
                elif item.unit_price and item.quantity:
                    prices.append(float(item.unit_price * item.quantity))

            if not prices:
                logger.warning("No valid prices found for comparison")
                return result

            # Calculate competitive metrics
            result.competitive_metrics = self._calculate_competitive_metrics(prices)

            # Detect outliers
            result.outliers = self._detect_outliers(pricing_items, prices)

            # Generate benchmarks
            result.benchmarks = self._generate_benchmarks(pricing_items, prices)

        except Exception as e:
            logger.error("Price comparison analysis failed", error=str(e))

        return result

    def _analyze_vendor_comparison(self, pricing_items: List[PricingItem],
                                 result: CompetitiveAnalysisResult) -> CompetitiveAnalysisResult:
        """Analyze pricing comparison across vendors."""

        try:
            # Group items by vendor
            vendor_groups = {}
            for item in pricing_items:
                vendor = item.vendor or "Unknown"
                if vendor not in vendor_groups:
                    vendor_groups[vendor] = []
                vendor_groups[vendor].append(item)

            # Analyze each vendor
            vendor_benchmarks = []
            for vendor, items in vendor_groups.items():
                vendor_prices = []
                for item in items:
                    if item.total_price:
                        vendor_prices.append(float(item.total_price))

                if vendor_prices:
                    avg_price = statistics.mean(vendor_prices)
                    benchmark = CompetitiveBenchmark(
                        category=BenchmarkCategory.AVERAGE_PRICE,
                        item_category=f"Vendor: {vendor}",
                        benchmark_price=Decimal(str(avg_price)),
                        benchmark_vendor=vendor,
                        your_price=Decimal(str(avg_price)),
                        price_difference=Decimal('0'),
                        percentage_difference=0.0,
                        competitive_position="baseline",
                        action_required=False,
                        data_freshness=1
                    )
                    vendor_benchmarks.append(benchmark)

            result.benchmarks.extend(vendor_benchmarks)

        except Exception as e:
            logger.error("Vendor comparison analysis failed", error=str(e))

        return result

    def _analyze_category_comparison(self, pricing_items: List[PricingItem],
                                   result: CompetitiveAnalysisResult) -> CompetitiveAnalysisResult:
        """Analyze pricing comparison across categories."""

        try:
            # Group items by category
            category_groups = {}
            for item in pricing_items:
                category = item.category or "Uncategorized"
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(item)

            # Analyze each category
            for category, items in category_groups.items():
                category_prices = []
                for item in items:
                    if item.total_price:
                        category_prices.append(float(item.total_price))

                if len(category_prices) >= self.config.trend_min_data_points:
                    # Detect trends within category
                    trend = self._detect_category_trend(category, items, category_prices)
                    if trend:
                        result.pricing_trends.append(trend)

        except Exception as e:
            logger.error("Category comparison analysis failed", error=str(e))

        return result

    def _analyze_historical_comparison(self, pricing_items: List[PricingItem],
                                     result: CompetitiveAnalysisResult) -> CompetitiveAnalysisResult:
        """Analyze historical pricing trends."""

        try:
            # Sort items by extraction date
            sorted_items = sorted(pricing_items, key=lambda x: x.extracted_at)

            # Group by time periods
            time_groups = self._group_by_time_periods(sorted_items)

            # Analyze trends across time periods
            for period, items in time_groups.items():
                if len(items) >= self.config.trend_min_data_points:
                    trend = self._analyze_time_period_trend(period, items)
                    if trend:
                        result.pricing_trends.append(trend)

        except Exception as e:
            logger.error("Historical comparison analysis failed", error=str(e))

        return result

    def _analyze_market_comparison(self, pricing_items: List[PricingItem],
                                 result: CompetitiveAnalysisResult) -> CompetitiveAnalysisResult:
        """Analyze market-wide pricing comparison."""

        try:
            # This would typically integrate with external market data
            # For now, we'll use internal data as a proxy

            # Calculate market statistics
            all_prices = []
            for item in pricing_items:
                if item.total_price:
                    all_prices.append(float(item.total_price))

            if all_prices:
                result.competitive_metrics = self._calculate_competitive_metrics(all_prices)

                # Generate market benchmarks
                market_benchmarks = self._generate_market_benchmarks(pricing_items, all_prices)
                result.benchmarks.extend(market_benchmarks)

        except Exception as e:
            logger.error("Market comparison analysis failed", error=str(e))

        return result

    def _calculate_competitive_metrics(self, prices: List[float]) -> CompetitiveMetrics:
        """Calculate competitive metrics from price data."""

        if not prices:
            raise ValueError("No prices provided for metrics calculation")

        # Basic statistics
        min_price = Decimal(str(min(prices)))
        max_price = Decimal(str(max(prices)))
        avg_price = Decimal(str(statistics.mean(prices)))
        median_price = Decimal(str(statistics.median(prices)))

        # Standard deviation
        std_dev = Decimal(str(statistics.stdev(prices) if len(prices) > 1 else 0))

        # Market position (using median as market reference)
        median_val = float(median_price)
        if avg_price < median_price * Decimal('0.9'):
            market_position = "below_market"
        elif avg_price > median_price * Decimal('1.1'):
            market_position = "above_market"
        else:
            market_position = "at_market"

        # Percentile rank (simplified)
        sorted_prices = sorted(prices)
        avg_val = float(avg_price)
        percentile_rank = (sorted_prices.index(min(sorted_prices, key=lambda x: abs(x - avg_val))) / len(sorted_prices)) * 100

        # Price advantage
        price_advantage = avg_price - median_price

        # Savings potential
        savings_potential = max_price - min_price

        # Volatility (coefficient of variation)
        price_volatility = float(std_dev / avg_price) if avg_price > 0 else 0

        # Outlier risk
        outlier_risk = price_volatility > 0.3  # 30% volatility threshold

        return CompetitiveMetrics(
            min_price=min_price,
            max_price=max_price,
            avg_price=avg_price,
            median_price=median_price,
            std_deviation=std_dev,
            market_position=market_position,
            percentile_rank=percentile_rank,
            price_advantage=price_advantage,
            savings_potential=savings_potential,
            price_volatility=price_volatility,
            outlier_risk=outlier_risk,
            sample_size=len(prices),
            confidence_level=min(0.95, len(prices) / 30)  # Higher confidence with more data
        )

    def _detect_outliers(self, pricing_items: List[PricingItem], prices: List[float]) -> List[PricingOutlier]:
        """Detect pricing outliers using statistical methods."""

        outliers = []

        if len(prices) < 3:
            return outliers  # Need at least 3 points for outlier detection

        try:
            mean_price = statistics.mean(prices)
            std_dev = statistics.stdev(prices)

            for i, item in enumerate(pricing_items):
                if i >= len(prices):
                    continue

                price = prices[i]
                z_score = (price - mean_price) / std_dev if std_dev > 0 else 0

                # Check if outlier
                if abs(z_score) > self.config.outlier_threshold_z_score:
                    # Determine outlier type
                    outlier_type = OutlierType.HIGH_OUTLIER if z_score > 0 else OutlierType.LOW_OUTLIER

                    # Calculate percentile
                    sorted_prices = sorted(prices)
                    percentile = (sorted_prices.index(min(sorted_prices, key=lambda x: abs(x - price))) / len(sorted_prices)) * 100

                    # Calculate expected range
                    expected_min = Decimal(str(mean_price - 2 * std_dev))
                    expected_max = Decimal(str(mean_price + 2 * std_dev))

                    # Create outlier
                    outlier = PricingOutlier(
                        item=item,
                        outlier_type=outlier_type,
                        z_score=z_score,
                        percentile=percentile,
                        deviation_amount=Decimal(str(abs(price - mean_price))),
                        deviation_percentage=abs((price - mean_price) / mean_price * 100),
                        market_average=Decimal(str(mean_price)),
                        expected_range_min=expected_min,
                        expected_range_max=expected_max,
                        confidence=min(0.95, abs(z_score) / 4),  # Higher confidence for more extreme outliers
                        explanation=self._generate_outlier_explanation(outlier_type, z_score, price, mean_price),
                        recommendations=self._generate_outlier_recommendations(outlier_type, item)
                    )

                    outliers.append(outlier)

        except Exception as e:
            logger.error("Outlier detection failed", error=str(e))

        return outliers

    def _generate_outlier_explanation(self, outlier_type: OutlierType, z_score: float,
                                    price: float, mean_price: float) -> str:
        """Generate explanation for outlier detection."""

        if outlier_type == OutlierType.HIGH_OUTLIER:
            return f"Price ${price:.2f} is {abs(z_score):.1f} standard deviations above the mean of ${mean_price:.2f}"
        else:
            return f"Price ${price:.2f} is {abs(z_score):.1f} standard deviations below the mean of ${mean_price:.2f}"

    def _generate_outlier_recommendations(self, outlier_type: OutlierType, item: PricingItem) -> List[str]:
        """Generate recommendations for outlier items."""

        recommendations = []

        if outlier_type == OutlierType.HIGH_OUTLIER:
            recommendations.extend([
                "Investigate reasons for high pricing",
                "Consider negotiating with vendor for better rates",
                "Evaluate alternative suppliers",
                "Assess if premium pricing reflects higher quality"
            ])
        else:
            recommendations.extend([
                "Verify pricing accuracy and completeness",
                "Assess quality and specification differences",
                "Consider potential hidden costs",
                "Evaluate vendor reliability and service quality"
            ])

        return recommendations

    def _generate_benchmarks(self, pricing_items: List[PricingItem], prices: List[float]) -> List[CompetitiveBenchmark]:
        """Generate competitive benchmarks."""

        benchmarks = []

        if not prices:
            return benchmarks

        try:
            # Best price benchmark
            min_price = min(prices)
            min_item = pricing_items[prices.index(min_price)]

            best_price_benchmark = CompetitiveBenchmark(
                category=BenchmarkCategory.BEST_PRICE,
                item_category="All Items",
                benchmark_price=Decimal(str(min_price)),
                benchmark_vendor=min_item.vendor,
                your_price=Decimal(str(statistics.mean(prices))),
                price_difference=Decimal(str(statistics.mean(prices) - min_price)),
                percentage_difference=((statistics.mean(prices) - min_price) / min_price * 100),
                competitive_position="above_best" if statistics.mean(prices) > min_price else "at_best",
                action_required=statistics.mean(prices) > min_price * 1.1,
                recommendations=["Consider switching to best price vendor", "Negotiate better rates"],
                data_freshness=1
            )
            benchmarks.append(best_price_benchmark)

            # Average price benchmark
            avg_price = statistics.mean(prices)
            avg_benchmark = CompetitiveBenchmark(
                category=BenchmarkCategory.AVERAGE_PRICE,
                item_category="All Items",
                benchmark_price=Decimal(str(avg_price)),
                your_price=Decimal(str(avg_price)),
                price_difference=Decimal('0'),
                percentage_difference=0.0,
                competitive_position="at_market",
                action_required=False,
                recommendations=["Monitor market changes", "Maintain current pricing strategy"],
                data_freshness=1
            )
            benchmarks.append(avg_benchmark)

        except Exception as e:
            logger.error("Benchmark generation failed", error=str(e))

        return benchmarks

    def _detect_category_trend(self, category: str, items: List[PricingItem], prices: List[float]) -> Optional[PricingTrend]:
        """Detect pricing trend within a category."""

        try:
            if len(prices) < self.config.trend_min_data_points:
                return None

            # Simple trend detection using linear regression approximation
            n = len(prices)
            x_values = list(range(n))

            # Calculate slope
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(prices)

            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, prices))
            denominator = sum((x - x_mean) ** 2 for x in x_values)

            if denominator == 0:
                return None

            slope = numerator / denominator

            # Determine trend direction
            if abs(slope) < 0.01:  # Threshold for stability
                direction = TrendDirection.STABLE
            elif slope > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING

            # Calculate magnitude
            magnitude = abs(slope / y_mean * 100) if y_mean > 0 else 0

            # Create trend
            trend = PricingTrend(
                item_category=category,
                direction=direction,
                magnitude=magnitude,
                duration_days=30,  # Simplified
                confidence=min(0.9, n / 10),  # Higher confidence with more data points
                start_price=Decimal(str(prices[0])),
                end_price=Decimal(str(prices[-1])),
                peak_price=Decimal(str(max(prices))),
                trough_price=Decimal(str(min(prices))),
                start_date=datetime.now(timezone.utc) - timedelta(days=30),
                end_date=datetime.now(timezone.utc),
                data_points=n,
                analysis_method="linear_regression"
            )

            return trend

        except Exception as e:
            logger.error("Category trend detection failed", error=str(e), category=category)
            return None

    def _group_by_time_periods(self, items: List[PricingItem]) -> Dict[str, List[PricingItem]]:
        """Group items by time periods for trend analysis."""

        time_groups = {}

        for item in items:
            # Group by month
            period_key = item.extracted_at.strftime("%Y-%m")
            if period_key not in time_groups:
                time_groups[period_key] = []
            time_groups[period_key].append(item)

        return time_groups

    def _analyze_time_period_trend(self, period: str, items: List[PricingItem]) -> Optional[PricingTrend]:
        """Analyze trend for a specific time period."""

        try:
            prices = []
            for item in items:
                if item.total_price:
                    prices.append(float(item.total_price))

            if len(prices) < self.config.trend_min_data_points:
                return None

            # Simple trend analysis
            start_price = prices[0]
            end_price = prices[-1]

            if abs(end_price - start_price) / start_price < 0.05:  # 5% threshold
                direction = TrendDirection.STABLE
            elif end_price > start_price:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING

            magnitude = abs((end_price - start_price) / start_price * 100)

            trend = PricingTrend(
                item_category=f"Period: {period}",
                direction=direction,
                magnitude=magnitude,
                duration_days=30,
                confidence=0.7,
                start_price=Decimal(str(start_price)),
                end_price=Decimal(str(end_price)),
                peak_price=Decimal(str(max(prices))),
                trough_price=Decimal(str(min(prices))),
                start_date=items[0].extracted_at,
                end_date=items[-1].extracted_at,
                data_points=len(prices)
            )

            return trend

        except Exception as e:
            logger.error("Time period trend analysis failed", error=str(e), period=period)
            return None

    def _generate_market_benchmarks(self, items: List[PricingItem], prices: List[float]) -> List[CompetitiveBenchmark]:
        """Generate market-wide benchmarks."""

        benchmarks = []

        try:
            if not prices:
                return benchmarks

            # Market leader benchmark (top 10th percentile)
            sorted_prices = sorted(prices)
            top_10_idx = int(len(sorted_prices) * 0.9)
            market_leader_price = sorted_prices[top_10_idx] if top_10_idx < len(sorted_prices) else sorted_prices[-1]

            market_leader_benchmark = CompetitiveBenchmark(
                category=BenchmarkCategory.MARKET_LEADER,
                item_category="Market Analysis",
                benchmark_price=Decimal(str(market_leader_price)),
                your_price=Decimal(str(statistics.mean(prices))),
                price_difference=Decimal(str(statistics.mean(prices) - market_leader_price)),
                percentage_difference=((statistics.mean(prices) - market_leader_price) / market_leader_price * 100),
                competitive_position="below_leader" if statistics.mean(prices) < market_leader_price else "at_leader",
                action_required=False,
                recommendations=["Analyze market leader strategies", "Consider premium positioning"],
                data_freshness=1
            )
            benchmarks.append(market_leader_benchmark)

        except Exception as e:
            logger.error("Market benchmark generation failed", error=str(e))

        return benchmarks

    def _generate_insights(self, result: CompetitiveAnalysisResult) -> List[str]:
        """Generate key insights from analysis results."""

        insights = []

        try:
            # Metrics insights
            if result.competitive_metrics:
                metrics = result.competitive_metrics
                insights.append(f"Market position: {metrics.market_position}")
                insights.append(f"Price volatility: {metrics.price_volatility:.1%}")

                if metrics.outlier_risk:
                    insights.append("High price volatility detected - market instability")

            # Outlier insights
            if result.outliers:
                high_outliers = sum(1 for o in result.outliers if o.outlier_type == OutlierType.HIGH_OUTLIER)
                low_outliers = sum(1 for o in result.outliers if o.outlier_type == OutlierType.LOW_OUTLIER)

                if high_outliers > 0:
                    insights.append(f"{high_outliers} items priced significantly above market")
                if low_outliers > 0:
                    insights.append(f"{low_outliers} items priced significantly below market")

            # Trend insights
            if result.pricing_trends:
                increasing_trends = sum(1 for t in result.pricing_trends if t.direction == TrendDirection.INCREASING)
                decreasing_trends = sum(1 for t in result.pricing_trends if t.direction == TrendDirection.DECREASING)

                if increasing_trends > decreasing_trends:
                    insights.append("Overall upward pricing trend detected")
                elif decreasing_trends > increasing_trends:
                    insights.append("Overall downward pricing trend detected")
                else:
                    insights.append("Mixed pricing trends across categories")

        except Exception as e:
            logger.error("Insight generation failed", error=str(e))

        return insights

    def _generate_action_items(self, result: CompetitiveAnalysisResult) -> List[str]:
        """Generate action items from analysis results."""

        action_items = []

        try:
            # High-priority outliers
            critical_outliers = [o for o in result.outliers if o.confidence > 0.8]
            if critical_outliers:
                action_items.append(f"Review {len(critical_outliers)} critical pricing outliers")

            # Benchmark actions
            action_required_benchmarks = [b for b in result.benchmarks if b.action_required]
            if action_required_benchmarks:
                action_items.append(f"Address {len(action_required_benchmarks)} competitive gaps")

            # Trend actions
            strong_trends = [t for t in result.pricing_trends if t.confidence > 0.7]
            if strong_trends:
                action_items.append(f"Monitor {len(strong_trends)} significant pricing trends")

            # General recommendations
            if result.competitive_metrics and result.competitive_metrics.outlier_risk:
                action_items.append("Implement price monitoring and alerts")

        except Exception as e:
            logger.error("Action item generation failed", error=str(e))

        return action_items

    def _calculate_analysis_confidence(self, result: CompetitiveAnalysisResult) -> float:
        """Calculate overall analysis confidence."""

        try:
            confidence_factors = []

            # Data volume factor
            if result.total_items_analyzed > 0:
                volume_factor = min(1.0, result.total_items_analyzed / 50)  # Optimal at 50+ items
                confidence_factors.append(volume_factor)

            # Metrics confidence
            if result.competitive_metrics:
                confidence_factors.append(result.competitive_metrics.confidence_level)

            # Outlier detection confidence
            if result.outliers:
                outlier_confidences = [o.confidence for o in result.outliers]
                confidence_factors.append(statistics.mean(outlier_confidences))

            # Trend confidence
            if result.pricing_trends:
                trend_confidences = [t.confidence for t in result.pricing_trends]
                confidence_factors.append(statistics.mean(trend_confidences))

            return statistics.mean(confidence_factors) if confidence_factors else 0.0

        except Exception as e:
            logger.error("Confidence calculation failed", error=str(e))
            return 0.0

    def _calculate_data_completeness(self, items: List[PricingItem]) -> float:
        """Calculate data completeness score."""

        if not items:
            return 0.0

        try:
            completeness_scores = []

            for item in items:
                score = 0.0

                # Required fields
                if item.item_name:
                    score += 0.3
                if item.total_price or item.unit_price:
                    score += 0.3
                if item.currency:
                    score += 0.2

                # Optional but valuable fields
                if item.vendor:
                    score += 0.1
                if item.category:
                    score += 0.1

                completeness_scores.append(score)

            return statistics.mean(completeness_scores)

        except Exception as e:
            logger.error("Data completeness calculation failed", error=str(e))
            return 0.0

    def _generate_cache_key(self, items: List[PricingItem], comparison_type: ComparisonType) -> str:
        """Generate cache key for analysis result."""

        import hashlib

        # Create a signature from items and config
        item_signatures = []
        for item in items[:10]:  # Limit to first 10 items for performance
            sig = f"{item.item_name}:{item.total_price}:{item.vendor}:{item.category}"
            item_signatures.append(sig)

        content = f"{'|'.join(item_signatures)}:{comparison_type.value}:{str(self.config.dict())}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[CompetitiveAnalysisResult]:
        """Get cached analysis result."""

        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if cached_data.get("expires_at", 0) > time.time():
                return cached_data.get("result")

        return None

    def _cache_result(self, cache_key: str, result: CompetitiveAnalysisResult) -> None:
        """Cache analysis result."""

        self._cache[cache_key] = {
            "result": result,
            "expires_at": time.time() + self.config.cache_ttl_seconds
        }

        # Simple cache cleanup
        if len(self._cache) > 100:
            current_time = time.time()
            expired_keys = [
                key for key, data in self._cache.items()
                if data.get("expires_at", 0) <= current_time
            ]
            for key in expired_keys:
                del self._cache[key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""

        return {
            **self._stats,
            "cache_size": len(self._cache),
            "config": self.config.dict()
        }


# Global service instance
_competitive_analysis_service: Optional[CompetitiveAnalysisService] = None


def get_competitive_analysis_service() -> CompetitiveAnalysisService:
    """Get or create the global competitive analysis service instance."""
    global _competitive_analysis_service

    if _competitive_analysis_service is None:
        _competitive_analysis_service = CompetitiveAnalysisService()

    return _competitive_analysis_service


def reset_competitive_analysis_service() -> None:
    """Reset the global competitive analysis service instance."""
    global _competitive_analysis_service
    _competitive_analysis_service = None
