"""
Pricing Intelligence Dashboard Service

This module provides pricing intelligence dashboard with visualization,
alerts, reporting, and decision support for procurement scenarios.
"""

import json
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
from agentic_rag.services.competitive_analysis import CompetitiveAnalysisResult, CompetitiveMetrics
from agentic_rag.services.cost_modeling import CostModelingResult, CostEstimate

logger = structlog.get_logger(__name__)


class DashboardWidgetType(str, Enum):
    """Types of dashboard widgets."""
    
    PRICE_TREND_CHART = "price_trend_chart"
    COMPETITIVE_COMPARISON = "competitive_comparison"
    COST_BREAKDOWN_PIE = "cost_breakdown_pie"
    OUTLIER_DETECTION = "outlier_detection"
    SAVINGS_OPPORTUNITIES = "savings_opportunities"
    RISK_ASSESSMENT = "risk_assessment"
    VENDOR_PERFORMANCE = "vendor_performance"
    MARKET_INTELLIGENCE = "market_intelligence"
    KPI_METRICS = "kpi_metrics"
    ALERT_SUMMARY = "alert_summary"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"


class AlertType(str, Enum):
    """Types of pricing alerts."""
    
    PRICE_SPIKE = "price_spike"
    PRICE_DROP = "price_drop"
    OUTLIER_DETECTED = "outlier_detected"
    BUDGET_EXCEEDED = "budget_exceeded"
    SAVINGS_OPPORTUNITY = "savings_opportunity"
    VENDOR_RISK = "vendor_risk"
    MARKET_CHANGE = "market_change"
    QUALITY_ISSUE = "quality_issue"


class ReportType(str, Enum):
    """Types of pricing reports."""
    
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    VENDOR_COMPARISON = "vendor_comparison"
    COST_OPTIMIZATION = "cost_optimization"
    MARKET_INTELLIGENCE = "market_intelligence"
    COMPLIANCE_REPORT = "compliance_report"
    TREND_ANALYSIS = "trend_analysis"


class ChartType(str, Enum):
    """Types of charts for visualization."""
    
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    GAUGE_CHART = "gauge_chart"
    TABLE = "table"


@dataclass
class ChartData:
    """Chart data structure."""
    
    chart_type: ChartType
    title: str
    data: Dict[str, Any]
    labels: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)


class DashboardWidget(BaseModel):
    """Dashboard widget configuration."""
    
    # Widget identification
    widget_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique widget ID")
    widget_type: DashboardWidgetType = Field(..., description="Type of widget")
    title: str = Field(..., description="Widget title")
    description: str = Field(default="", description="Widget description")
    
    # Layout properties
    position: Tuple[int, int] = Field(default=(0, 0), description="Widget position (row, col)")
    size: Tuple[int, int] = Field(default=(1, 1), description="Widget size (height, width)")
    
    # Data configuration
    data_source: str = Field(..., description="Data source identifier")
    refresh_interval: int = Field(default=300, description="Refresh interval in seconds")
    
    # Visualization settings
    chart_config: Optional[ChartData] = Field(default=None, description="Chart configuration")
    
    # Access control
    required_roles: List[UserRole] = Field(
        default_factory=lambda: [UserRole.VIEWER],
        description="Required roles to view widget"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )


class PricingAlert(BaseModel):
    """Pricing alert definition."""
    
    # Alert identification
    alert_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique alert ID")
    alert_type: AlertType = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity")
    
    # Alert content
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    
    # Trigger information
    triggered_by: str = Field(..., description="What triggered the alert")
    threshold_value: Optional[Decimal] = Field(default=None, description="Threshold that was crossed")
    current_value: Optional[Decimal] = Field(default=None, description="Current value")
    
    # Context
    affected_items: List[str] = Field(default_factory=list, description="Affected items")
    vendor: Optional[str] = Field(default=None, description="Related vendor")
    category: Optional[str] = Field(default=None, description="Related category")
    
    # Actions
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )
    action_required: bool = Field(default=False, description="Whether action is required")
    
    # Status
    acknowledged: bool = Field(default=False, description="Whether alert is acknowledged")
    resolved: bool = Field(default=False, description="Whether alert is resolved")
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert creation time"
    )
    expires_at: Optional[datetime] = Field(default=None, description="Alert expiration time")


class DashboardReport(BaseModel):
    """Dashboard report definition."""
    
    # Report identification
    report_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique report ID")
    report_type: ReportType = Field(..., description="Type of report")
    title: str = Field(..., description="Report title")
    
    # Report content
    summary: str = Field(..., description="Executive summary")
    sections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Report sections"
    )
    
    # Data and metrics
    key_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key metrics"
    )
    charts: List[ChartData] = Field(
        default_factory=list,
        description="Report charts"
    )
    
    # Insights and recommendations
    insights: List[str] = Field(
        default_factory=list,
        description="Key insights"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )
    
    # Metadata
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation time"
    )
    generated_by: str = Field(..., description="Report generator")
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    
    # Dashboard settings
    dashboard_title: str = Field(default="Pricing Intelligence Dashboard", description="Dashboard title")
    refresh_interval: int = Field(default=300, description="Global refresh interval in seconds")
    
    # Widget settings
    default_widgets: List[DashboardWidgetType] = Field(
        default_factory=lambda: [
            DashboardWidgetType.KPI_METRICS,
            DashboardWidgetType.PRICE_TREND_CHART,
            DashboardWidgetType.COMPETITIVE_COMPARISON,
            DashboardWidgetType.ALERT_SUMMARY
        ],
        description="Default widgets to display"
    )
    
    # Alert settings
    enable_alerts: bool = Field(default=True, description="Enable alert system")
    alert_retention_days: int = Field(default=30, description="Alert retention period")
    
    # Report settings
    enable_reports: bool = Field(default=True, description="Enable report generation")
    report_retention_days: int = Field(default=90, description="Report retention period")
    
    # Visualization settings
    chart_theme: str = Field(default="light", description="Chart theme")
    color_palette: List[str] = Field(
        default_factory=lambda: [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
        ],
        description="Color palette for charts"
    )
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable data caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")


class DashboardData(BaseModel):
    """Dashboard data container."""
    
    # Widgets
    widgets: List[DashboardWidget] = Field(
        default_factory=list,
        description="Dashboard widgets"
    )
    
    # Alerts
    active_alerts: List[PricingAlert] = Field(
        default_factory=list,
        description="Active alerts"
    )
    
    # Reports
    recent_reports: List[DashboardReport] = Field(
        default_factory=list,
        description="Recent reports"
    )
    
    # KPIs
    key_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )
    
    # Metadata
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    data_freshness: int = Field(default=0, description="Data freshness in minutes")


class PricingDashboardService:
    """Service for pricing intelligence dashboard."""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.settings = get_settings()
        
        # Performance tracking
        self._stats = {
            "total_dashboard_loads": 0,
            "total_alerts_generated": 0,
            "total_reports_generated": 0,
            "total_widgets_rendered": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Cache for dashboard data
        self._cache: Dict[str, Any] = {}
        
        # Alert storage
        self._alerts: List[PricingAlert] = []
        
        # Report storage
        self._reports: List[DashboardReport] = []
        
        logger.info("Pricing dashboard service initialized")
    
    def get_dashboard_data(self, user_role: UserRole = UserRole.VIEWER) -> DashboardData:
        """Get complete dashboard data for a user role."""
        
        start_time = time.time()
        
        try:
            # Check cache
            if self.config.enable_caching:
                cache_key = f"dashboard_data_{user_role.value}"
                cached_data = self._get_cached_data(cache_key)
                if cached_data:
                    self._stats["cache_hits"] += 1
                    return cached_data
                self._stats["cache_misses"] += 1
            
            # Generate dashboard data
            dashboard_data = DashboardData()
            
            # Generate widgets
            dashboard_data.widgets = self._generate_widgets(user_role)
            
            # Get active alerts
            dashboard_data.active_alerts = self._get_active_alerts(user_role)
            
            # Get recent reports
            dashboard_data.recent_reports = self._get_recent_reports(user_role)
            
            # Calculate key metrics
            dashboard_data.key_metrics = self._calculate_key_metrics()
            
            # Set data freshness
            dashboard_data.data_freshness = 0  # Real-time for now
            
            # Cache result
            if self.config.enable_caching:
                self._cache_data(cache_key, dashboard_data)
            
            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            self._stats["total_dashboard_loads"] += 1
            self._stats["total_widgets_rendered"] += len(dashboard_data.widgets)
            self._stats["total_processing_time_ms"] += processing_time_ms
            
            logger.info(
                "Dashboard data generated",
                user_role=user_role.value,
                widgets_count=len(dashboard_data.widgets),
                alerts_count=len(dashboard_data.active_alerts),
                processing_time_ms=processing_time_ms
            )
            
            return dashboard_data
            
        except Exception as e:
            logger.error("Dashboard data generation failed", error=str(e), user_role=user_role.value)
            raise

    def _generate_widgets(self, user_role: UserRole) -> List[DashboardWidget]:
        """Generate dashboard widgets based on user role."""

        widgets = []

        try:
            for widget_type in self.config.default_widgets:
                # Check role permissions
                if self._has_widget_permission(user_role, widget_type):
                    widget = self._create_widget(widget_type, user_role)
                    if widget:
                        widgets.append(widget)

        except Exception as e:
            logger.error("Widget generation failed", error=str(e))

        return widgets

    def _has_widget_permission(self, user_role: UserRole, widget_type: DashboardWidgetType) -> bool:
        """Check if user role has permission for widget type."""

        # Define role-based widget permissions
        widget_permissions = {
            DashboardWidgetType.KPI_METRICS: [UserRole.VIEWER, UserRole.ANALYST, UserRole.ADMIN],
            DashboardWidgetType.PRICE_TREND_CHART: [UserRole.VIEWER, UserRole.ANALYST, UserRole.ADMIN],
            DashboardWidgetType.COMPETITIVE_COMPARISON: [UserRole.ANALYST, UserRole.ADMIN],
            DashboardWidgetType.COST_BREAKDOWN_PIE: [UserRole.ANALYST, UserRole.ADMIN],
            DashboardWidgetType.OUTLIER_DETECTION: [UserRole.ANALYST, UserRole.ADMIN],
            DashboardWidgetType.SAVINGS_OPPORTUNITIES: [UserRole.ANALYST, UserRole.ADMIN],
            DashboardWidgetType.RISK_ASSESSMENT: [UserRole.ADMIN],
            DashboardWidgetType.VENDOR_PERFORMANCE: [UserRole.ANALYST, UserRole.ADMIN],
            DashboardWidgetType.MARKET_INTELLIGENCE: [UserRole.ADMIN],
            DashboardWidgetType.ALERT_SUMMARY: [UserRole.VIEWER, UserRole.ANALYST, UserRole.ADMIN]
        }

        return user_role in widget_permissions.get(widget_type, [])

    def _create_widget(self, widget_type: DashboardWidgetType, user_role: UserRole) -> Optional[DashboardWidget]:
        """Create a specific widget based on type."""

        try:
            if widget_type == DashboardWidgetType.KPI_METRICS:
                return self._create_kpi_widget()
            elif widget_type == DashboardWidgetType.PRICE_TREND_CHART:
                return self._create_price_trend_widget()
            elif widget_type == DashboardWidgetType.COMPETITIVE_COMPARISON:
                return self._create_competitive_comparison_widget()
            elif widget_type == DashboardWidgetType.COST_BREAKDOWN_PIE:
                return self._create_cost_breakdown_widget()
            elif widget_type == DashboardWidgetType.OUTLIER_DETECTION:
                return self._create_outlier_detection_widget()
            elif widget_type == DashboardWidgetType.SAVINGS_OPPORTUNITIES:
                return self._create_savings_opportunities_widget()
            elif widget_type == DashboardWidgetType.RISK_ASSESSMENT:
                return self._create_risk_assessment_widget()
            elif widget_type == DashboardWidgetType.VENDOR_PERFORMANCE:
                return self._create_vendor_performance_widget()
            elif widget_type == DashboardWidgetType.MARKET_INTELLIGENCE:
                return self._create_market_intelligence_widget()
            elif widget_type == DashboardWidgetType.ALERT_SUMMARY:
                return self._create_alert_summary_widget()

        except Exception as e:
            logger.error("Widget creation failed", error=str(e), widget_type=widget_type.value)

        return None

    def _create_kpi_widget(self) -> DashboardWidget:
        """Create KPI metrics widget."""

        # Sample KPI data
        kpi_data = {
            "total_items_analyzed": 1250,
            "average_savings": "12.5%",
            "cost_reduction": "$45,000",
            "vendor_count": 25,
            "outliers_detected": 8,
            "alerts_active": 3
        }

        chart_data = ChartData(
            chart_type=ChartType.TABLE,
            title="Key Performance Indicators",
            data=kpi_data
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.KPI_METRICS,
            title="Key Metrics",
            description="Overview of key pricing intelligence metrics",
            data_source="pricing_analytics",
            chart_config=chart_data,
            position=(0, 0),
            size=(1, 2)
        )

    def _create_price_trend_widget(self) -> DashboardWidget:
        """Create price trend chart widget."""

        # Sample trend data
        trend_data = {
            "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "datasets": [
                {
                    "label": "Average Price",
                    "data": [100, 105, 98, 110, 108, 95],
                    "borderColor": "#1f77b4",
                    "backgroundColor": "rgba(31, 119, 180, 0.1)"
                }
            ]
        }

        chart_data = ChartData(
            chart_type=ChartType.LINE_CHART,
            title="Price Trends Over Time",
            data=trend_data,
            labels=trend_data["labels"],
            colors=["#1f77b4"]
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.PRICE_TREND_CHART,
            title="Price Trends",
            description="Historical price trends analysis",
            data_source="price_history",
            chart_config=chart_data,
            position=(0, 2),
            size=(2, 2)
        )

    def _create_competitive_comparison_widget(self) -> DashboardWidget:
        """Create competitive comparison widget."""

        # Sample comparison data
        comparison_data = {
            "labels": ["Vendor A", "Vendor B", "Vendor C", "Vendor D"],
            "datasets": [
                {
                    "label": "Price",
                    "data": [120, 95, 110, 105],
                    "backgroundColor": ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
                }
            ]
        }

        chart_data = ChartData(
            chart_type=ChartType.BAR_CHART,
            title="Vendor Price Comparison",
            data=comparison_data,
            labels=comparison_data["labels"],
            colors=comparison_data["datasets"][0]["backgroundColor"]
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.COMPETITIVE_COMPARISON,
            title="Competitive Analysis",
            description="Vendor price comparison and market position",
            data_source="competitive_analysis",
            chart_config=chart_data,
            position=(1, 0),
            size=(1, 2)
        )

    def _create_cost_breakdown_widget(self) -> DashboardWidget:
        """Create cost breakdown pie chart widget."""

        # Sample cost breakdown data
        breakdown_data = {
            "labels": ["Materials", "Labor", "Overhead", "Shipping"],
            "datasets": [
                {
                    "data": [40, 30, 20, 10],
                    "backgroundColor": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                }
            ]
        }

        chart_data = ChartData(
            chart_type=ChartType.PIE_CHART,
            title="Cost Component Breakdown",
            data=breakdown_data,
            labels=breakdown_data["labels"],
            colors=breakdown_data["datasets"][0]["backgroundColor"]
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.COST_BREAKDOWN_PIE,
            title="Cost Breakdown",
            description="Analysis of cost components",
            data_source="cost_modeling",
            chart_config=chart_data,
            position=(2, 0),
            size=(1, 1)
        )

    def _create_outlier_detection_widget(self) -> DashboardWidget:
        """Create outlier detection widget."""

        # Sample outlier data
        outlier_data = {
            "outliers": [
                {"item": "Product X", "price": 250, "expected": 150, "deviation": 66.7},
                {"item": "Product Y", "price": 75, "expected": 120, "deviation": -37.5},
                {"item": "Product Z", "price": 300, "expected": 180, "deviation": 66.7}
            ],
            "total_outliers": 3,
            "high_outliers": 2,
            "low_outliers": 1
        }

        chart_data = ChartData(
            chart_type=ChartType.TABLE,
            title="Pricing Outliers Detected",
            data=outlier_data
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.OUTLIER_DETECTION,
            title="Outlier Detection",
            description="Items with unusual pricing patterns",
            data_source="outlier_analysis",
            chart_config=chart_data,
            position=(2, 1),
            size=(1, 1)
        )

    def _create_savings_opportunities_widget(self) -> DashboardWidget:
        """Create savings opportunities widget."""

        # Sample savings data
        savings_data = {
            "opportunities": [
                {"category": "Office Supplies", "potential_savings": 15000, "confidence": 85},
                {"category": "IT Equipment", "potential_savings": 25000, "confidence": 92},
                {"category": "Facilities", "potential_savings": 8000, "confidence": 78}
            ],
            "total_potential": 48000,
            "high_confidence_total": 40000
        }

        chart_data = ChartData(
            chart_type=ChartType.BAR_CHART,
            title="Savings Opportunities",
            data={
                "labels": [opp["category"] for opp in savings_data["opportunities"]],
                "datasets": [{
                    "label": "Potential Savings ($)",
                    "data": [opp["potential_savings"] for opp in savings_data["opportunities"]],
                    "backgroundColor": ["#2ca02c", "#1f77b4", "#ff7f0e"]
                }]
            }
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.SAVINGS_OPPORTUNITIES,
            title="Savings Opportunities",
            description="Identified cost reduction opportunities",
            data_source="savings_analysis",
            chart_config=chart_data,
            position=(2, 2),
            size=(1, 2)
        )

    def _create_risk_assessment_widget(self) -> DashboardWidget:
        """Create risk assessment widget."""

        # Sample risk data
        risk_data = {
            "overall_risk": "Medium",
            "risk_factors": [
                {"factor": "Price Volatility", "level": "High", "impact": 8},
                {"factor": "Vendor Concentration", "level": "Medium", "impact": 6},
                {"factor": "Market Uncertainty", "level": "Low", "impact": 3}
            ],
            "risk_score": 65
        }

        chart_data = ChartData(
            chart_type=ChartType.GAUGE_CHART,
            title="Risk Assessment",
            data=risk_data
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.RISK_ASSESSMENT,
            title="Risk Assessment",
            description="Overall procurement risk analysis",
            data_source="risk_analysis",
            chart_config=chart_data,
            position=(3, 0),
            size=(1, 1)
        )

    def _create_vendor_performance_widget(self) -> DashboardWidget:
        """Create vendor performance widget."""

        # Sample vendor performance data
        vendor_data = {
            "vendors": [
                {"name": "Vendor A", "score": 85, "price_competitiveness": 90, "reliability": 80},
                {"name": "Vendor B", "score": 92, "price_competitiveness": 88, "reliability": 96},
                {"name": "Vendor C", "score": 78, "price_competitiveness": 75, "reliability": 81}
            ]
        }

        chart_data = ChartData(
            chart_type=ChartType.SCATTER_PLOT,
            title="Vendor Performance Matrix",
            data=vendor_data
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.VENDOR_PERFORMANCE,
            title="Vendor Performance",
            description="Vendor performance and competitiveness analysis",
            data_source="vendor_analysis",
            chart_config=chart_data,
            position=(3, 1),
            size=(1, 1)
        )

    def _create_market_intelligence_widget(self) -> DashboardWidget:
        """Create market intelligence widget."""

        # Sample market data
        market_data = {
            "market_trends": "Stable with slight upward pressure",
            "price_index": 102.5,
            "volatility": "Low",
            "key_insights": [
                "Raw material costs increasing 3%",
                "New suppliers entering market",
                "Demand expected to grow 5% next quarter"
            ]
        }

        chart_data = ChartData(
            chart_type=ChartType.TABLE,
            title="Market Intelligence Summary",
            data=market_data
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.MARKET_INTELLIGENCE,
            title="Market Intelligence",
            description="Market trends and intelligence insights",
            data_source="market_analysis",
            chart_config=chart_data,
            position=(3, 2),
            size=(1, 2)
        )

    def _create_alert_summary_widget(self) -> DashboardWidget:
        """Create alert summary widget."""

        # Sample alert data
        alert_data = {
            "total_alerts": len(self._alerts),
            "critical_alerts": len([a for a in self._alerts if a.severity == AlertSeverity.CRITICAL]),
            "warning_alerts": len([a for a in self._alerts if a.severity == AlertSeverity.WARNING]),
            "recent_alerts": [
                {"title": "Price spike detected", "severity": "warning", "time": "2 hours ago"},
                {"title": "New vendor opportunity", "severity": "info", "time": "4 hours ago"}
            ]
        }

        chart_data = ChartData(
            chart_type=ChartType.TABLE,
            title="Active Alerts Summary",
            data=alert_data
        )

        return DashboardWidget(
            widget_type=DashboardWidgetType.ALERT_SUMMARY,
            title="Alert Summary",
            description="Summary of active pricing alerts",
            data_source="alert_system",
            chart_config=chart_data,
            position=(0, 4),
            size=(1, 1)
        )

    def _get_active_alerts(self, user_role: UserRole) -> List[PricingAlert]:
        """Get active alerts for user role."""

        try:
            # Filter alerts based on user role and active status
            active_alerts = []

            for alert in self._alerts:
                if not alert.resolved and not self._is_alert_expired(alert):
                    # Check role-based access
                    if self._has_alert_permission(user_role, alert):
                        active_alerts.append(alert)

            # Sort by severity and creation time
            active_alerts.sort(
                key=lambda x: (
                    self._get_severity_priority(x.severity),
                    x.created_at
                ),
                reverse=True
            )

            return active_alerts

        except Exception as e:
            logger.error("Failed to get active alerts", error=str(e))
            return []

    def _has_alert_permission(self, user_role: UserRole, alert: PricingAlert) -> bool:
        """Check if user has permission to view alert."""

        # Define role-based alert permissions
        if alert.severity == AlertSeverity.CRITICAL:
            return user_role in [UserRole.ANALYST, UserRole.ADMIN]
        elif alert.severity == AlertSeverity.WARNING:
            return user_role in [UserRole.VIEWER, UserRole.ANALYST, UserRole.ADMIN]
        else:
            return True  # INFO alerts visible to all

    def _is_alert_expired(self, alert: PricingAlert) -> bool:
        """Check if alert has expired."""

        if alert.expires_at:
            return datetime.now(timezone.utc) > alert.expires_at
        return False

    def _get_severity_priority(self, severity: AlertSeverity) -> int:
        """Get numeric priority for severity sorting."""

        priorities = {
            AlertSeverity.URGENT: 4,
            AlertSeverity.CRITICAL: 3,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 1
        }
        return priorities.get(severity, 0)

    def _get_recent_reports(self, user_role: UserRole) -> List[DashboardReport]:
        """Get recent reports for user role."""

        try:
            # Filter reports based on user role
            accessible_reports = []

            for report in self._reports:
                if self._has_report_permission(user_role, report):
                    accessible_reports.append(report)

            # Sort by generation time and return recent ones
            accessible_reports.sort(key=lambda x: x.generated_at, reverse=True)
            return accessible_reports[:10]  # Return last 10 reports

        except Exception as e:
            logger.error("Failed to get recent reports", error=str(e))
            return []

    def _has_report_permission(self, user_role: UserRole, report: DashboardReport) -> bool:
        """Check if user has permission to view report."""

        # Define role-based report permissions
        report_permissions = {
            ReportType.EXECUTIVE_SUMMARY: [UserRole.ANALYST, UserRole.ADMIN],
            ReportType.DETAILED_ANALYSIS: [UserRole.ADMIN],
            ReportType.VENDOR_COMPARISON: [UserRole.ANALYST, UserRole.ADMIN],
            ReportType.COST_OPTIMIZATION: [UserRole.ANALYST, UserRole.ADMIN],
            ReportType.MARKET_INTELLIGENCE: [UserRole.ADMIN],
            ReportType.COMPLIANCE_REPORT: [UserRole.ADMIN],
            ReportType.TREND_ANALYSIS: [UserRole.VIEWER, UserRole.ANALYST, UserRole.ADMIN]
        }

        return user_role in report_permissions.get(report.report_type, [])

    def _calculate_key_metrics(self) -> Dict[str, Any]:
        """Calculate key performance metrics."""

        try:
            metrics = {
                "total_items_analyzed": 1250,
                "total_vendors": 25,
                "average_savings_percentage": 12.5,
                "total_cost_reduction": 45000,
                "active_alerts": len(self._alerts),
                "critical_alerts": len([a for a in self._alerts if a.severity == AlertSeverity.CRITICAL]),
                "outliers_detected": 8,
                "reports_generated": len(self._reports),
                "data_freshness_minutes": 5,
                "system_health": "Good"
            }

            return metrics

        except Exception as e:
            logger.error("Failed to calculate key metrics", error=str(e))
            return {}

    def create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                    title: str, message: str, **kwargs) -> PricingAlert:
        """Create a new pricing alert."""

        try:
            alert = PricingAlert(
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                triggered_by=kwargs.get("triggered_by", "system"),
                threshold_value=kwargs.get("threshold_value"),
                current_value=kwargs.get("current_value"),
                affected_items=kwargs.get("affected_items", []),
                vendor=kwargs.get("vendor"),
                category=kwargs.get("category"),
                recommended_actions=kwargs.get("recommended_actions", []),
                action_required=kwargs.get("action_required", False),
                expires_at=kwargs.get("expires_at")
            )

            self._alerts.append(alert)
            self._stats["total_alerts_generated"] += 1

            logger.info(
                "Alert created",
                alert_id=alert.alert_id,
                alert_type=alert_type.value,
                severity=severity.value,
                title=title
            )

            return alert

        except Exception as e:
            logger.error("Failed to create alert", error=str(e))
            raise

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""

        try:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info("Alert acknowledged", alert_id=alert_id, user_id=user_id)
                    return True

            logger.warning("Alert not found for acknowledgment", alert_id=alert_id)
            return False

        except Exception as e:
            logger.error("Failed to acknowledge alert", error=str(e), alert_id=alert_id)
            return False

    def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """Resolve an alert."""

        try:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info("Alert resolved", alert_id=alert_id, user_id=user_id)
                    return True

            logger.warning("Alert not found for resolution", alert_id=alert_id)
            return False

        except Exception as e:
            logger.error("Failed to resolve alert", error=str(e), alert_id=alert_id)
            return False

    def generate_report(self, report_type: ReportType, period_start: datetime,
                       period_end: datetime, generated_by: str) -> DashboardReport:
        """Generate a pricing intelligence report."""

        try:
            if report_type == ReportType.EXECUTIVE_SUMMARY:
                report = self._generate_executive_summary(period_start, period_end, generated_by)
            elif report_type == ReportType.DETAILED_ANALYSIS:
                report = self._generate_detailed_analysis(period_start, period_end, generated_by)
            elif report_type == ReportType.VENDOR_COMPARISON:
                report = self._generate_vendor_comparison(period_start, period_end, generated_by)
            elif report_type == ReportType.COST_OPTIMIZATION:
                report = self._generate_cost_optimization(period_start, period_end, generated_by)
            elif report_type == ReportType.TREND_ANALYSIS:
                report = self._generate_trend_analysis(period_start, period_end, generated_by)
            else:
                # Default to executive summary
                report = self._generate_executive_summary(period_start, period_end, generated_by)

            self._reports.append(report)
            self._stats["total_reports_generated"] += 1

            logger.info(
                "Report generated",
                report_id=report.report_id,
                report_type=report_type.value,
                generated_by=generated_by
            )

            return report

        except Exception as e:
            logger.error("Failed to generate report", error=str(e), report_type=report_type.value)
            raise

    def _generate_executive_summary(self, period_start: datetime, period_end: datetime,
                                  generated_by: str) -> DashboardReport:
        """Generate executive summary report."""

        summary = """
        Executive Summary: Pricing Intelligence Report

        During the reporting period, our pricing analysis system processed 1,250 items
        across 25 vendors, identifying $45,000 in potential cost savings opportunities.

        Key achievements:
        - 12.5% average savings identified
        - 8 pricing outliers detected and addressed
        - 3 critical alerts resolved
        - Market position improved in 4 categories
        """

        key_metrics = {
            "total_savings_identified": 45000,
            "average_savings_percentage": 12.5,
            "items_analyzed": 1250,
            "vendors_evaluated": 25,
            "outliers_detected": 8,
            "alerts_generated": 15
        }

        insights = [
            "Office supplies category shows highest savings potential",
            "Vendor consolidation could reduce costs by 8%",
            "Market trends indicate stable pricing for next quarter",
            "New vendor partnerships improved competitive position"
        ]

        recommendations = [
            "Implement vendor consolidation strategy",
            "Negotiate volume discounts for high-volume categories",
            "Monitor emerging suppliers in IT equipment category",
            "Establish quarterly pricing reviews"
        ]

        return DashboardReport(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            title="Executive Summary - Pricing Intelligence",
            summary=summary.strip(),
            key_metrics=key_metrics,
            insights=insights,
            recommendations=recommendations,
            generated_by=generated_by,
            period_start=period_start,
            period_end=period_end
        )

    def _generate_detailed_analysis(self, period_start: datetime, period_end: datetime,
                                  generated_by: str) -> DashboardReport:
        """Generate detailed analysis report."""

        summary = "Comprehensive detailed analysis of pricing data and market conditions."

        sections = [
            {
                "title": "Market Analysis",
                "content": "Detailed market condition analysis and trends"
            },
            {
                "title": "Vendor Performance",
                "content": "Individual vendor performance metrics and comparisons"
            },
            {
                "title": "Cost Optimization",
                "content": "Detailed cost optimization opportunities and strategies"
            }
        ]

        return DashboardReport(
            report_type=ReportType.DETAILED_ANALYSIS,
            title="Detailed Pricing Analysis Report",
            summary=summary,
            sections=sections,
            generated_by=generated_by,
            period_start=period_start,
            period_end=period_end
        )

    def _generate_vendor_comparison(self, period_start: datetime, period_end: datetime,
                                  generated_by: str) -> DashboardReport:
        """Generate vendor comparison report."""

        summary = "Comprehensive comparison of vendor performance and pricing competitiveness."

        key_metrics = {
            "top_performing_vendor": "Vendor B",
            "most_competitive_pricing": "Vendor A",
            "highest_reliability": "Vendor C",
            "best_value_vendor": "Vendor B"
        }

        return DashboardReport(
            report_type=ReportType.VENDOR_COMPARISON,
            title="Vendor Comparison Report",
            summary=summary,
            key_metrics=key_metrics,
            generated_by=generated_by,
            period_start=period_start,
            period_end=period_end
        )

    def _generate_cost_optimization(self, period_start: datetime, period_end: datetime,
                                  generated_by: str) -> DashboardReport:
        """Generate cost optimization report."""

        summary = "Analysis of cost optimization opportunities and implementation strategies."

        recommendations = [
            "Consolidate suppliers in office supplies category",
            "Negotiate volume discounts for high-frequency purchases",
            "Implement just-in-time ordering for non-critical items",
            "Explore alternative suppliers for premium categories"
        ]

        return DashboardReport(
            report_type=ReportType.COST_OPTIMIZATION,
            title="Cost Optimization Report",
            summary=summary,
            recommendations=recommendations,
            generated_by=generated_by,
            period_start=period_start,
            period_end=period_end
        )

    def _generate_trend_analysis(self, period_start: datetime, period_end: datetime,
                               generated_by: str) -> DashboardReport:
        """Generate trend analysis report."""

        summary = "Analysis of pricing trends and market movements over the reporting period."

        insights = [
            "Overall pricing trend shows 2% increase over period",
            "Technology category experiencing price deflation",
            "Raw materials showing seasonal price variations",
            "Service categories remain stable with low volatility"
        ]

        return DashboardReport(
            report_type=ReportType.TREND_ANALYSIS,
            title="Pricing Trend Analysis Report",
            summary=summary,
            insights=insights,
            generated_by=generated_by,
            period_start=period_start,
            period_end=period_end
        )

    def _get_cached_data(self, cache_key: str) -> Optional[DashboardData]:
        """Get cached dashboard data."""

        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if cached_data.get("expires_at", 0) > time.time():
                return cached_data.get("data")

        return None

    def _cache_data(self, cache_key: str, data: DashboardData) -> None:
        """Cache dashboard data."""

        self._cache[cache_key] = {
            "data": data,
            "expires_at": time.time() + self.config.cache_ttl_seconds
        }

        # Simple cache cleanup
        if len(self._cache) > 20:
            current_time = time.time()
            expired_keys = [
                key for key, cached_data in self._cache.items()
                if cached_data.get("expires_at", 0) <= current_time
            ]
            for key in expired_keys:
                del self._cache[key]

    def cleanup_expired_data(self) -> None:
        """Clean up expired alerts and reports."""

        try:
            current_time = datetime.now(timezone.utc)

            # Clean up expired alerts
            retention_date = current_time - timedelta(days=self.config.alert_retention_days)
            self._alerts = [
                alert for alert in self._alerts
                if alert.created_at > retention_date and not self._is_alert_expired(alert)
            ]

            # Clean up old reports
            report_retention_date = current_time - timedelta(days=self.config.report_retention_days)
            self._reports = [
                report for report in self._reports
                if report.generated_at > report_retention_date
            ]

            logger.info("Expired data cleanup completed")

        except Exception as e:
            logger.error("Failed to cleanup expired data", error=str(e))

    def get_widget_data(self, widget_id: str, user_role: UserRole) -> Optional[Dict[str, Any]]:
        """Get data for a specific widget."""

        try:
            # Find widget
            widget = None
            dashboard_data = self.get_dashboard_data(user_role)

            for w in dashboard_data.widgets:
                if w.widget_id == widget_id:
                    widget = w
                    break

            if not widget:
                return None

            # Return widget chart data
            if widget.chart_config:
                return {
                    "chart_type": widget.chart_config.chart_type.value,
                    "title": widget.chart_config.title,
                    "data": widget.chart_config.data,
                    "labels": widget.chart_config.labels,
                    "colors": widget.chart_config.colors,
                    "options": widget.chart_config.options
                }

            return None

        except Exception as e:
            logger.error("Failed to get widget data", error=str(e), widget_id=widget_id)
            return None

    def update_widget_config(self, widget_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update widget configuration."""

        try:
            # This would typically update widget configuration in database
            # For now, just log the update
            logger.info("Widget configuration updated", widget_id=widget_id, updates=config_updates)
            return True

        except Exception as e:
            logger.error("Failed to update widget config", error=str(e), widget_id=widget_id)
            return False

    def export_dashboard_data(self, user_role: UserRole, format: str = "json") -> Optional[str]:
        """Export dashboard data in specified format."""

        try:
            dashboard_data = self.get_dashboard_data(user_role)

            if format.lower() == "json":
                return dashboard_data.json(indent=2)
            else:
                logger.warning("Unsupported export format", format=format)
                return None

        except Exception as e:
            logger.error("Failed to export dashboard data", error=str(e), format=format)
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""

        return {
            **self._stats,
            "cache_size": len(self._cache),
            "active_alerts": len([a for a in self._alerts if not a.resolved]),
            "total_reports": len(self._reports),
            "config": self.config.dict()
        }


# Global service instance
_pricing_dashboard_service: Optional[PricingDashboardService] = None


def get_pricing_dashboard_service() -> PricingDashboardService:
    """Get or create the global pricing dashboard service instance."""
    global _pricing_dashboard_service

    if _pricing_dashboard_service is None:
        _pricing_dashboard_service = PricingDashboardService()

    return _pricing_dashboard_service


def reset_pricing_dashboard_service() -> None:
    """Reset the global pricing dashboard service instance."""
    global _pricing_dashboard_service
    _pricing_dashboard_service = None
