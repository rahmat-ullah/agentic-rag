"""
Cost Modeling and Estimation Service

This module provides cost modeling with component breakdown, estimation algorithms,
scenario analysis, and risk assessment for procurement scenarios.
"""

import math
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


class CostCategory(str, Enum):
    """Categories for cost breakdown."""
    
    DIRECT_MATERIALS = "direct_materials"
    DIRECT_LABOR = "direct_labor"
    MANUFACTURING_OVERHEAD = "manufacturing_overhead"
    SHIPPING_LOGISTICS = "shipping_logistics"
    TAXES_DUTIES = "taxes_duties"
    VENDOR_MARGIN = "vendor_margin"
    ADMINISTRATIVE = "administrative"
    CONTINGENCY = "contingency"
    OTHER = "other"


class EstimationMethod(str, Enum):
    """Methods for cost estimation."""
    
    HISTORICAL_AVERAGE = "historical_average"
    PARAMETRIC = "parametric"
    ANALOGOUS = "analogous"
    BOTTOM_UP = "bottom_up"
    THREE_POINT = "three_point"
    MONTE_CARLO = "monte_carlo"
    REGRESSION = "regression"


class RiskLevel(str, Enum):
    """Risk levels for cost estimates."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScenarioType(str, Enum):
    """Types of cost scenarios."""
    
    OPTIMISTIC = "optimistic"
    MOST_LIKELY = "most_likely"
    PESSIMISTIC = "pessimistic"
    WORST_CASE = "worst_case"
    BEST_CASE = "best_case"


@dataclass
class CostComponent:
    """Individual cost component."""
    
    component_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    category: CostCategory = CostCategory.OTHER
    base_cost: Decimal = Decimal('0')
    quantity: float = 1.0
    unit: str = "each"
    
    # Cost factors
    labor_hours: Optional[float] = None
    material_cost: Optional[Decimal] = None
    overhead_rate: Optional[float] = None
    
    # Risk and uncertainty
    uncertainty_percentage: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    
    # Metadata
    description: str = ""
    vendor: Optional[str] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CostEstimate(BaseModel):
    """Cost estimation result."""
    
    # Estimate identification
    estimate_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique estimate ID")
    item_name: str = Field(..., description="Item being estimated")
    estimation_method: EstimationMethod = Field(..., description="Method used for estimation")
    
    # Cost breakdown
    components: List[CostComponent] = Field(
        default_factory=list,
        description="Cost components"
    )
    
    # Estimated costs
    base_cost: Decimal = Field(..., description="Base estimated cost")
    total_cost: Decimal = Field(..., description="Total estimated cost including all factors")
    
    # Cost ranges
    optimistic_cost: Decimal = Field(..., description="Optimistic cost estimate")
    pessimistic_cost: Decimal = Field(..., description="Pessimistic cost estimate")
    
    # Confidence and risk
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Estimation confidence")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    uncertainty_range: Tuple[Decimal, Decimal] = Field(..., description="Uncertainty range")
    
    # Analysis metadata
    estimation_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Estimation date"
    )
    valid_until: Optional[datetime] = Field(default=None, description="Estimate validity")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions")
    limitations: List[str] = Field(default_factory=list, description="Estimation limitations")


class CostScenario(BaseModel):
    """Cost scenario analysis."""
    
    # Scenario identification
    scenario_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique scenario ID")
    scenario_type: ScenarioType = Field(..., description="Type of scenario")
    name: str = Field(..., description="Scenario name")
    description: str = Field(default="", description="Scenario description")
    
    # Scenario parameters
    cost_adjustments: Dict[CostCategory, float] = Field(
        default_factory=dict,
        description="Cost adjustments by category (percentage)"
    )
    volume_multiplier: float = Field(default=1.0, description="Volume adjustment multiplier")
    timeline_adjustment: float = Field(default=1.0, description="Timeline adjustment factor")
    
    # Results
    total_cost: Decimal = Field(..., description="Total cost for scenario")
    cost_difference: Decimal = Field(..., description="Difference from baseline")
    percentage_change: float = Field(..., description="Percentage change from baseline")
    
    # Risk assessment
    probability: float = Field(..., ge=0.0, le=1.0, description="Scenario probability")
    impact_level: RiskLevel = Field(..., description="Impact level")
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )


class CostModelingConfig(BaseModel):
    """Configuration for cost modeling."""
    
    # Estimation settings
    default_estimation_method: EstimationMethod = Field(
        default=EstimationMethod.HISTORICAL_AVERAGE,
        description="Default estimation method"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    
    # Risk settings
    default_uncertainty_percentage: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Default uncertainty percentage"
    )
    risk_tolerance: RiskLevel = Field(
        default=RiskLevel.MEDIUM,
        description="Risk tolerance level"
    )
    
    # Scenario settings
    enable_scenario_analysis: bool = Field(
        default=True,
        description="Enable scenario analysis"
    )
    default_scenarios: List[ScenarioType] = Field(
        default_factory=lambda: [
            ScenarioType.OPTIMISTIC,
            ScenarioType.MOST_LIKELY,
            ScenarioType.PESSIMISTIC
        ],
        description="Default scenarios to generate"
    )
    
    # Component settings
    enable_component_breakdown: bool = Field(
        default=True,
        description="Enable detailed component breakdown"
    )
    min_component_percentage: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Minimum percentage for component inclusion"
    )
    
    # Currency and units
    base_currency: Currency = Field(
        default=Currency.USD,
        description="Base currency for modeling"
    )
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_processing_time_seconds: int = Field(
        default=30,
        description="Maximum processing time"
    )


class CostModelingResult(BaseModel):
    """Result of cost modeling analysis."""
    
    # Primary estimate
    primary_estimate: CostEstimate = Field(..., description="Primary cost estimate")
    
    # Alternative estimates
    alternative_estimates: List[CostEstimate] = Field(
        default_factory=list,
        description="Alternative estimation methods"
    )
    
    # Scenario analysis
    scenarios: List[CostScenario] = Field(
        default_factory=list,
        description="Cost scenarios"
    )
    
    # Component analysis
    component_breakdown: Dict[CostCategory, Decimal] = Field(
        default_factory=dict,
        description="Cost breakdown by category"
    )
    component_percentages: Dict[CostCategory, float] = Field(
        default_factory=dict,
        description="Component percentages of total cost"
    )
    
    # Risk assessment
    overall_risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Identified risk factors"
    )
    mitigation_strategies: List[str] = Field(
        default_factory=list,
        description="Risk mitigation strategies"
    )
    
    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Cost optimization recommendations"
    )
    cost_drivers: List[str] = Field(
        default_factory=list,
        description="Key cost drivers identified"
    )
    
    # Quality metrics
    estimation_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated accuracy of the model"
    )
    data_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality of input data"
    )
    
    # Performance metrics
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )
    
    # Metadata
    modeling_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Modeling timestamp"
    )
    config_used: Optional[CostModelingConfig] = Field(
        default=None,
        description="Configuration used for modeling"
    )


class CostModelingService:
    """Service for cost modeling and estimation."""
    
    def __init__(self, config: Optional[CostModelingConfig] = None):
        self.config = config or CostModelingConfig()
        self.settings = get_settings()
        
        # Performance tracking
        self._stats = {
            "total_estimates": 0,
            "total_scenarios_generated": 0,
            "total_components_analyzed": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Cache for results
        self._cache: Dict[str, Any] = {}
        
        # Historical data for estimation
        self._historical_data: Dict[str, List[PricingItem]] = {}
        
        logger.info("Cost modeling service initialized")
    
    def estimate_cost(self, item_name: str, 
                     historical_items: Optional[List[PricingItem]] = None,
                     estimation_method: Optional[EstimationMethod] = None) -> CostModelingResult:
        """Estimate cost for an item using specified method."""
        
        start_time = time.time()
        
        try:
            # Use default method if not specified
            method = estimation_method or self.config.default_estimation_method
            
            # Check cache
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(item_name, method, historical_items)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self._stats["cache_hits"] += 1
                    return cached_result
                self._stats["cache_misses"] += 1
            
            # Initialize result
            result = CostModelingResult(
                primary_estimate=CostEstimate(
                    item_name=item_name,
                    estimation_method=method,
                    base_cost=Decimal('0'),
                    total_cost=Decimal('0'),
                    optimistic_cost=Decimal('0'),
                    pessimistic_cost=Decimal('0'),
                    confidence_level=0.0,
                    risk_level=RiskLevel.MEDIUM,
                    uncertainty_range=(Decimal('0'), Decimal('0'))
                ),
                overall_risk_level=RiskLevel.MEDIUM,
                config_used=self.config
            )
            
            # Perform estimation based on method
            if method == EstimationMethod.HISTORICAL_AVERAGE:
                result = self._estimate_historical_average(item_name, historical_items, result)
            elif method == EstimationMethod.PARAMETRIC:
                result = self._estimate_parametric(item_name, historical_items, result)
            elif method == EstimationMethod.THREE_POINT:
                result = self._estimate_three_point(item_name, historical_items, result)
            elif method == EstimationMethod.BOTTOM_UP:
                result = self._estimate_bottom_up(item_name, result)
            else:
                # Fallback to historical average
                result = self._estimate_historical_average(item_name, historical_items, result)
            
            # Generate component breakdown if enabled
            if self.config.enable_component_breakdown:
                result = self._generate_component_breakdown(result)
            
            # Generate scenarios if enabled
            if self.config.enable_scenario_analysis:
                result = self._generate_scenarios(result)
            
            # Assess risks and generate recommendations
            result = self._assess_risks(result)
            result = self._generate_recommendations(result)
            
            # Calculate quality metrics
            result.estimation_accuracy = self._calculate_estimation_accuracy(result)
            result.data_quality_score = self._calculate_data_quality(historical_items)
            
            # Performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time_ms
            
            # Update statistics
            self._stats["total_estimates"] += 1
            self._stats["total_scenarios_generated"] += len(result.scenarios)
            self._stats["total_processing_time_ms"] += processing_time_ms
            
            # Cache result
            if self.config.enable_caching:
                self._cache_result(cache_key, result)
            
            logger.info(
                "Cost estimation completed",
                item_name=item_name,
                method=method.value,
                total_cost=float(result.primary_estimate.total_cost),
                confidence=result.primary_estimate.confidence_level,
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error("Cost estimation failed", error=str(e), item_name=item_name)
            raise

    def _estimate_historical_average(self, item_name: str,
                                   historical_items: Optional[List[PricingItem]],
                                   result: CostModelingResult) -> CostModelingResult:
        """Estimate cost using historical average method."""

        try:
            if not historical_items:
                # Use default estimation
                base_cost = Decimal('100.00')  # Default fallback
                confidence = 0.3
            else:
                # Calculate average from historical data
                prices = []
                for item in historical_items:
                    if item.total_price:
                        prices.append(float(item.total_price))
                    elif item.unit_price and item.quantity:
                        prices.append(float(item.unit_price * item.quantity))

                if prices:
                    avg_price = statistics.mean(prices)
                    base_cost = Decimal(str(avg_price))
                    confidence = min(0.9, len(prices) / 10)  # Higher confidence with more data
                else:
                    base_cost = Decimal('100.00')
                    confidence = 0.3

            # Calculate uncertainty range
            uncertainty = self.config.default_uncertainty_percentage / 100
            optimistic = base_cost * Decimal(str(1 - uncertainty))
            pessimistic = base_cost * Decimal(str(1 + uncertainty))

            # Update primary estimate
            result.primary_estimate.base_cost = base_cost
            result.primary_estimate.total_cost = base_cost
            result.primary_estimate.optimistic_cost = optimistic
            result.primary_estimate.pessimistic_cost = pessimistic
            result.primary_estimate.confidence_level = confidence
            result.primary_estimate.uncertainty_range = (optimistic, pessimistic)

        except Exception as e:
            logger.error("Historical average estimation failed", error=str(e))

        return result

    def _estimate_parametric(self, item_name: str,
                           historical_items: Optional[List[PricingItem]],
                           result: CostModelingResult) -> CostModelingResult:
        """Estimate cost using parametric method."""

        try:
            # Simple parametric model based on item characteristics
            base_cost = Decimal('100.00')

            # Adjust based on item name characteristics
            if any(keyword in item_name.lower() for keyword in ['premium', 'high-end', 'enterprise']):
                base_cost *= Decimal('1.5')
            elif any(keyword in item_name.lower() for keyword in ['basic', 'standard', 'economy']):
                base_cost *= Decimal('0.8')

            # Adjust based on historical data if available
            if historical_items:
                prices = [float(item.total_price) for item in historical_items if item.total_price]
                if prices:
                    historical_avg = statistics.mean(prices)
                    base_cost = (base_cost + Decimal(str(historical_avg))) / 2

            # Calculate ranges
            uncertainty = 0.15  # 15% uncertainty for parametric
            optimistic = base_cost * Decimal('0.85')
            pessimistic = base_cost * Decimal('1.15')

            # Update estimate
            result.primary_estimate.base_cost = base_cost
            result.primary_estimate.total_cost = base_cost
            result.primary_estimate.optimistic_cost = optimistic
            result.primary_estimate.pessimistic_cost = pessimistic
            result.primary_estimate.confidence_level = 0.7
            result.primary_estimate.uncertainty_range = (optimistic, pessimistic)

        except Exception as e:
            logger.error("Parametric estimation failed", error=str(e))

        return result

    def _estimate_three_point(self, item_name: str,
                            historical_items: Optional[List[PricingItem]],
                            result: CostModelingResult) -> CostModelingResult:
        """Estimate cost using three-point estimation method."""

        try:
            # Get optimistic, most likely, and pessimistic estimates
            if historical_items and len(historical_items) >= 3:
                prices = [float(item.total_price) for item in historical_items if item.total_price]
                if len(prices) >= 3:
                    prices.sort()
                    optimistic = Decimal(str(prices[0]))  # Lowest price
                    pessimistic = Decimal(str(prices[-1]))  # Highest price
                    most_likely = Decimal(str(statistics.median(prices)))  # Median price
                else:
                    # Default values
                    optimistic = Decimal('80.00')
                    most_likely = Decimal('100.00')
                    pessimistic = Decimal('120.00')
            else:
                # Default three-point estimates
                optimistic = Decimal('80.00')
                most_likely = Decimal('100.00')
                pessimistic = Decimal('120.00')

            # Calculate expected value using PERT formula: (O + 4M + P) / 6
            expected_cost = (optimistic + 4 * most_likely + pessimistic) / 6

            # Update estimate
            result.primary_estimate.base_cost = most_likely
            result.primary_estimate.total_cost = expected_cost
            result.primary_estimate.optimistic_cost = optimistic
            result.primary_estimate.pessimistic_cost = pessimistic
            result.primary_estimate.confidence_level = 0.8
            result.primary_estimate.uncertainty_range = (optimistic, pessimistic)

        except Exception as e:
            logger.error("Three-point estimation failed", error=str(e))

        return result

    def _estimate_bottom_up(self, item_name: str, result: CostModelingResult) -> CostModelingResult:
        """Estimate cost using bottom-up method with component breakdown."""

        try:
            # Create default components
            components = [
                CostComponent(
                    name="Materials",
                    category=CostCategory.DIRECT_MATERIALS,
                    base_cost=Decimal('40.00'),
                    uncertainty_percentage=10.0
                ),
                CostComponent(
                    name="Labor",
                    category=CostCategory.DIRECT_LABOR,
                    base_cost=Decimal('30.00'),
                    uncertainty_percentage=15.0
                ),
                CostComponent(
                    name="Overhead",
                    category=CostCategory.MANUFACTURING_OVERHEAD,
                    base_cost=Decimal('20.00'),
                    uncertainty_percentage=20.0
                ),
                CostComponent(
                    name="Shipping",
                    category=CostCategory.SHIPPING_LOGISTICS,
                    base_cost=Decimal('10.00'),
                    uncertainty_percentage=25.0
                )
            ]

            # Calculate total cost
            total_cost = sum(comp.base_cost for comp in components)

            # Calculate uncertainty ranges
            optimistic_total = Decimal('0')
            pessimistic_total = Decimal('0')

            for comp in components:
                uncertainty = comp.uncertainty_percentage / 100
                optimistic_total += comp.base_cost * Decimal(str(1 - uncertainty))
                pessimistic_total += comp.base_cost * Decimal(str(1 + uncertainty))

            # Update estimate
            result.primary_estimate.components = components
            result.primary_estimate.base_cost = total_cost
            result.primary_estimate.total_cost = total_cost
            result.primary_estimate.optimistic_cost = optimistic_total
            result.primary_estimate.pessimistic_cost = pessimistic_total
            result.primary_estimate.confidence_level = 0.75
            result.primary_estimate.uncertainty_range = (optimistic_total, pessimistic_total)

        except Exception as e:
            logger.error("Bottom-up estimation failed", error=str(e))

        return result

    def _generate_component_breakdown(self, result: CostModelingResult) -> CostModelingResult:
        """Generate detailed component breakdown."""

        try:
            if not result.primary_estimate.components:
                # Create default breakdown if no components exist
                total_cost = result.primary_estimate.total_cost

                result.component_breakdown = {
                    CostCategory.DIRECT_MATERIALS: total_cost * Decimal('0.4'),
                    CostCategory.DIRECT_LABOR: total_cost * Decimal('0.3'),
                    CostCategory.MANUFACTURING_OVERHEAD: total_cost * Decimal('0.2'),
                    CostCategory.SHIPPING_LOGISTICS: total_cost * Decimal('0.1')
                }
            else:
                # Calculate breakdown from components
                breakdown = {}
                for component in result.primary_estimate.components:
                    if component.category not in breakdown:
                        breakdown[component.category] = Decimal('0')
                    breakdown[component.category] += component.base_cost

                result.component_breakdown = breakdown

            # Calculate percentages
            total = sum(result.component_breakdown.values())
            if total > 0:
                result.component_percentages = {
                    category: float(amount / total * 100)
                    for category, amount in result.component_breakdown.items()
                }

        except Exception as e:
            logger.error("Component breakdown generation failed", error=str(e))

        return result

    def _generate_scenarios(self, result: CostModelingResult) -> CostModelingResult:
        """Generate cost scenarios."""

        try:
            base_cost = result.primary_estimate.total_cost
            scenarios = []

            for scenario_type in self.config.default_scenarios:
                if scenario_type == ScenarioType.OPTIMISTIC:
                    adjustments = {
                        CostCategory.DIRECT_MATERIALS: -10.0,
                        CostCategory.DIRECT_LABOR: -5.0,
                        CostCategory.SHIPPING_LOGISTICS: -15.0
                    }
                    probability = 0.2

                elif scenario_type == ScenarioType.MOST_LIKELY:
                    adjustments = {}  # No adjustments for most likely
                    probability = 0.6

                elif scenario_type == ScenarioType.PESSIMISTIC:
                    adjustments = {
                        CostCategory.DIRECT_MATERIALS: 15.0,
                        CostCategory.DIRECT_LABOR: 10.0,
                        CostCategory.SHIPPING_LOGISTICS: 25.0,
                        CostCategory.CONTINGENCY: 10.0
                    }
                    probability = 0.2

                else:
                    adjustments = {}
                    probability = 0.1

                # Calculate scenario cost
                scenario_cost = base_cost
                total_adjustment = 0.0

                for category, adjustment in adjustments.items():
                    if category in result.component_breakdown:
                        component_cost = result.component_breakdown[category]
                        adjustment_amount = component_cost * Decimal(str(adjustment / 100))
                        scenario_cost += adjustment_amount
                        total_adjustment += float(adjustment_amount)

                # Create scenario
                scenario = CostScenario(
                    scenario_type=scenario_type,
                    name=scenario_type.value.replace('_', ' ').title(),
                    description=f"{scenario_type.value.replace('_', ' ').title()} cost scenario",
                    cost_adjustments=adjustments,
                    total_cost=scenario_cost,
                    cost_difference=scenario_cost - base_cost,
                    percentage_change=float((scenario_cost - base_cost) / base_cost * 100) if base_cost > 0 else 0,
                    probability=probability,
                    impact_level=RiskLevel.LOW if scenario_type == ScenarioType.OPTIMISTIC else RiskLevel.HIGH
                )

                scenarios.append(scenario)

            result.scenarios = scenarios

        except Exception as e:
            logger.error("Scenario generation failed", error=str(e))

        return result

    def _assess_risks(self, result: CostModelingResult) -> CostModelingResult:
        """Assess risks and determine overall risk level."""

        try:
            risk_factors = []
            risk_score = 0

            # Confidence-based risk
            if result.primary_estimate.confidence_level < 0.5:
                risk_factors.append("Low estimation confidence")
                risk_score += 2

            # Uncertainty range risk
            uncertainty_range = result.primary_estimate.uncertainty_range
            if uncertainty_range[1] > uncertainty_range[0] * Decimal('1.5'):
                risk_factors.append("High cost uncertainty")
                risk_score += 2

            # Component risk
            if result.primary_estimate.components:
                high_risk_components = [
                    comp for comp in result.primary_estimate.components
                    if comp.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                ]
                if high_risk_components:
                    risk_factors.append(f"{len(high_risk_components)} high-risk components")
                    risk_score += len(high_risk_components)

            # Scenario risk
            if result.scenarios:
                pessimistic_scenarios = [
                    s for s in result.scenarios
                    if s.scenario_type == ScenarioType.PESSIMISTIC and s.percentage_change > 20
                ]
                if pessimistic_scenarios:
                    risk_factors.append("High pessimistic scenario impact")
                    risk_score += 1

            # Determine overall risk level
            if risk_score >= 5:
                overall_risk = RiskLevel.CRITICAL
            elif risk_score >= 3:
                overall_risk = RiskLevel.HIGH
            elif risk_score >= 1:
                overall_risk = RiskLevel.MEDIUM
            else:
                overall_risk = RiskLevel.LOW

            result.overall_risk_level = overall_risk
            result.risk_factors = risk_factors

            # Generate mitigation strategies
            mitigation_strategies = []
            if "Low estimation confidence" in risk_factors:
                mitigation_strategies.append("Gather more historical data for better estimates")
            if "High cost uncertainty" in risk_factors:
                mitigation_strategies.append("Implement cost monitoring and early warning systems")
            if any("high-risk components" in factor for factor in risk_factors):
                mitigation_strategies.append("Focus on risk mitigation for high-risk components")

            result.mitigation_strategies = mitigation_strategies

        except Exception as e:
            logger.error("Risk assessment failed", error=str(e))

        return result

    def _generate_recommendations(self, result: CostModelingResult) -> CostModelingResult:
        """Generate cost optimization recommendations."""

        try:
            recommendations = []
            cost_drivers = []

            # Component-based recommendations
            if result.component_breakdown:
                # Find largest cost components
                sorted_components = sorted(
                    result.component_breakdown.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for category, amount in sorted_components[:3]:  # Top 3 components
                    percentage = result.component_percentages.get(category, 0)
                    if percentage > 20:  # Significant component
                        cost_drivers.append(f"{category.value}: {percentage:.1f}% of total cost")

                        if category == CostCategory.DIRECT_MATERIALS:
                            recommendations.append("Negotiate better material prices or find alternative suppliers")
                        elif category == CostCategory.DIRECT_LABOR:
                            recommendations.append("Optimize labor efficiency or consider automation")
                        elif category == CostCategory.SHIPPING_LOGISTICS:
                            recommendations.append("Optimize shipping routes or consolidate shipments")

            # Risk-based recommendations
            if result.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                recommendations.append("Implement comprehensive risk management plan")
                recommendations.append("Consider alternative suppliers or backup options")

            # Scenario-based recommendations
            if result.scenarios:
                pessimistic_impact = max(
                    (s.percentage_change for s in result.scenarios if s.scenario_type == ScenarioType.PESSIMISTIC),
                    default=0
                )
                if pessimistic_impact > 25:
                    recommendations.append("Prepare contingency budget for cost overruns")

            result.recommendations = recommendations
            result.cost_drivers = cost_drivers

        except Exception as e:
            logger.error("Recommendation generation failed", error=str(e))

        return result

    def _calculate_estimation_accuracy(self, result: CostModelingResult) -> float:
        """Calculate estimated accuracy of the cost model."""

        try:
            accuracy_factors = []

            # Confidence factor
            accuracy_factors.append(result.primary_estimate.confidence_level)

            # Data quality factor
            if hasattr(result, 'data_quality_score'):
                accuracy_factors.append(result.data_quality_score)

            # Method factor
            method_accuracy = {
                EstimationMethod.HISTORICAL_AVERAGE: 0.7,
                EstimationMethod.PARAMETRIC: 0.6,
                EstimationMethod.THREE_POINT: 0.8,
                EstimationMethod.BOTTOM_UP: 0.9,
                EstimationMethod.MONTE_CARLO: 0.85
            }
            method_score = method_accuracy.get(result.primary_estimate.estimation_method, 0.5)
            accuracy_factors.append(method_score)

            # Component detail factor
            if result.primary_estimate.components:
                component_factor = min(1.0, len(result.primary_estimate.components) / 5)
                accuracy_factors.append(component_factor)

            return statistics.mean(accuracy_factors) if accuracy_factors else 0.5

        except Exception as e:
            logger.error("Accuracy calculation failed", error=str(e))
            return 0.5

    def _calculate_data_quality(self, historical_items: Optional[List[PricingItem]]) -> float:
        """Calculate quality score of input data."""

        if not historical_items:
            return 0.3  # Low quality without historical data

        try:
            quality_scores = []

            for item in historical_items:
                score = 0.0

                # Required fields
                if item.item_name:
                    score += 0.3
                if item.total_price or item.unit_price:
                    score += 0.4
                if item.currency:
                    score += 0.1

                # Optional but valuable fields
                if item.vendor:
                    score += 0.1
                if item.category:
                    score += 0.1

                quality_scores.append(score)

            # Volume bonus
            volume_factor = min(1.0, len(historical_items) / 20)  # Optimal at 20+ items

            return statistics.mean(quality_scores) * volume_factor if quality_scores else 0.3

        except Exception as e:
            logger.error("Data quality calculation failed", error=str(e))
            return 0.3

    def _generate_cache_key(self, item_name: str, method: EstimationMethod,
                          historical_items: Optional[List[PricingItem]]) -> str:
        """Generate cache key for cost estimation."""

        import hashlib

        # Create signature
        historical_sig = ""
        if historical_items:
            item_sigs = []
            for item in historical_items[:5]:  # Limit for performance
                sig = f"{item.item_name}:{item.total_price}:{item.vendor}"
                item_sigs.append(sig)
            historical_sig = "|".join(item_sigs)

        content = f"{item_name}:{method.value}:{historical_sig}:{str(self.config.dict())}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[CostModelingResult]:
        """Get cached cost modeling result."""

        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if cached_data.get("expires_at", 0) > time.time():
                return cached_data.get("result")

        return None

    def _cache_result(self, cache_key: str, result: CostModelingResult) -> None:
        """Cache cost modeling result."""

        self._cache[cache_key] = {
            "result": result,
            "expires_at": time.time() + self.config.cache_ttl_seconds
        }

        # Simple cache cleanup
        if len(self._cache) > 50:
            current_time = time.time()
            expired_keys = [
                key for key, data in self._cache.items()
                if data.get("expires_at", 0) <= current_time
            ]
            for key in expired_keys:
                del self._cache[key]

    def add_historical_data(self, category: str, items: List[PricingItem]) -> None:
        """Add historical data for improved estimation."""

        if category not in self._historical_data:
            self._historical_data[category] = []

        self._historical_data[category].extend(items)

        # Keep only recent data (last 100 items per category)
        if len(self._historical_data[category]) > 100:
            self._historical_data[category] = self._historical_data[category][-100:]

        logger.info(f"Added {len(items)} historical items to category '{category}'")

    def get_historical_data(self, category: str) -> List[PricingItem]:
        """Get historical data for a category."""

        return self._historical_data.get(category, [])

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""

        return {
            **self._stats,
            "cache_size": len(self._cache),
            "historical_categories": len(self._historical_data),
            "total_historical_items": sum(len(items) for items in self._historical_data.values()),
            "config": self.config.dict()
        }


# Global service instance
_cost_modeling_service: Optional[CostModelingService] = None


def get_cost_modeling_service() -> CostModelingService:
    """Get or create the global cost modeling service instance."""
    global _cost_modeling_service

    if _cost_modeling_service is None:
        _cost_modeling_service = CostModelingService()

    return _cost_modeling_service


def reset_cost_modeling_service() -> None:
    """Reset the global cost modeling service instance."""
    global _cost_modeling_service
    _cost_modeling_service = None
