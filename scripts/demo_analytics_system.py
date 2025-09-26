#!/usr/bin/env python3
"""
Feedback Analytics and Insights System Demonstration Script for Sprint 6 Story 6-04

This script demonstrates the complete analytics system functionality including
metric calculation, trend analysis, recommendation generation, and dashboard
data aggregation capabilities.
"""

import asyncio
import uuid
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

import structlog
from sqlalchemy.orm import Session

from agentic_rag.database.connection import get_database_session
from agentic_rag.models.tenant import Tenant
from agentic_rag.models.auth import User
from agentic_rag.models.analytics import (
    AnalyticsMetric,
    PerformanceRecommendation,
    DashboardConfiguration,
    MetricAggregation,
    AnalyticsMetricType,
    RecommendationType,
    RecommendationStatus
)
from agentic_rag.models.feedback import UserFeedbackSubmission, FeedbackType
from agentic_rag.services.analytics_service import AnalyticsService
from agentic_rag.services.recommendation_service import RecommendationService
from agentic_rag.schemas.analytics import (
    CreateAnalyticsMetricRequest,
    CreatePerformanceRecommendationRequest
)

logger = structlog.get_logger(__name__)


class AnalyticsSystemDemo:
    """Demonstration of the analytics and insights system."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.analytics_service = AnalyticsService(db_session)
        self.recommendation_service = RecommendationService(db_session)
        self.demo_tenant = None
        self.demo_user = None
        self.demo_metrics = []
        self.demo_recommendations = []
    
    async def run_complete_demo(self):
        """Run complete analytics system demonstration."""
        print("🚀 Starting Feedback Analytics and Insights System Demonstration")
        print("=" * 70)
        
        try:
            # Setup demo environment
            await self.setup_demo_environment()
            
            # Demonstrate core functionality
            await self.demo_metric_creation()
            await self.demo_search_quality_analytics()
            await self.demo_user_satisfaction_analysis()
            await self.demo_content_quality_assessment()
            await self.demo_trend_analysis()
            await self.demo_recommendation_generation()
            await self.demo_dashboard_configuration()
            
            # Show final results
            await self.show_demo_results()
            
            print("\n✅ Analytics System Demonstration Completed Successfully!")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ Demo failed: {str(e)}")
            logger.error("demo_failed", error=str(e), exc_info=True)
            raise
    
    async def setup_demo_environment(self):
        """Set up demo tenant and user."""
        print("\n📋 Setting up demo environment...")
        
        # Create demo tenant
        self.demo_tenant = Tenant(
            name="Analytics Demo Tenant",
            slug="analytics-demo",
            description="Tenant for analytics system demonstration"
        )
        self.db.add(self.demo_tenant)
        
        # Create demo user
        self.demo_user = User(
            tenant_id=self.demo_tenant.id,
            email="demo@analyticsdemo.com",
            username="analytics_demo_user",
            full_name="Analytics Demo User",
            role="admin",
            is_active=True
        )
        self.db.add(self.demo_user)
        
        self.db.commit()
        print(f"✓ Created demo tenant: {self.demo_tenant.name}")
        print(f"✓ Created demo user: {self.demo_user.username}")
        
        # Create sample feedback data
        await self.create_sample_feedback_data()
    
    async def create_sample_feedback_data(self):
        """Create sample feedback data for analytics."""
        print("✓ Creating sample feedback data...")
        
        # Create various types of feedback over the past 30 days
        feedback_data = []
        for i in range(100):
            days_ago = random.randint(0, 30)
            feedback_date = datetime.now() - timedelta(days=days_ago)
            
            feedback = UserFeedbackSubmission(
                tenant_id=self.demo_tenant.id,
                user_id=self.demo_user.id,
                feedback_type=random.choice([FeedbackType.SEARCH_RESULT, FeedbackType.LINK_QUALITY, FeedbackType.ANSWER_QUALITY]),
                target_id=uuid.uuid4(),
                rating=random.randint(1, 5),
                feedback_text=f"Sample feedback {i+1}",
                status="completed",
                created_at=feedback_date,
                context_metadata={"demo": True, "feedback_number": i+1}
            )
            feedback_data.append(feedback)
        
        self.db.add_all(feedback_data)
        self.db.commit()
        print(f"✓ Created {len(feedback_data)} sample feedback submissions")
    
    async def demo_metric_creation(self):
        """Demonstrate analytics metric creation."""
        print("\n📊 Demonstrating Analytics Metric Creation...")
        
        # Create various types of metrics
        metric_configs = [
            {
                "type": AnalyticsMetricType.SEARCH_QUALITY,
                "name": "click_through_rate",
                "category": "user_engagement",
                "value": 0.65,
                "baseline": 0.60,
                "target": 0.70
            },
            {
                "type": AnalyticsMetricType.USER_SATISFACTION,
                "name": "user_satisfaction_rating",
                "category": "feedback",
                "value": 4.2,
                "baseline": 4.0,
                "target": 4.5
            },
            {
                "type": AnalyticsMetricType.CONTENT_QUALITY,
                "name": "overall_content_quality",
                "category": "quality",
                "value": 0.85,
                "baseline": 0.80,
                "target": 0.90
            },
            {
                "type": AnalyticsMetricType.SYSTEM_PERFORMANCE,
                "name": "avg_response_time",
                "category": "performance",
                "value": 0.25,
                "baseline": 0.30,
                "target": 0.20
            },
            {
                "type": AnalyticsMetricType.USER_ENGAGEMENT,
                "name": "daily_active_users",
                "category": "engagement",
                "value": 1250,
                "baseline": 1000,
                "target": 1500
            }
        ]
        
        for config in metric_configs:
            metric_data = CreateAnalyticsMetricRequest(
                metric_type=config["type"],
                metric_name=config["name"],
                metric_category=config["category"],
                metric_value=config["value"],
                baseline_value=config["baseline"],
                target_value=config["target"],
                measurement_date=date.today(),
                sample_size=random.randint(500, 2000),
                confidence_level=0.95,
                calculation_method="demo_calculation",
                data_sources=["demo_data"],
                metric_metadata={"demo": True}
            )
            
            metric = await self.analytics_service.create_analytics_metric(
                self.demo_tenant.id, metric_data
            )
            self.demo_metrics.append(metric)
        
        print(f"✓ Created {len(self.demo_metrics)} analytics metrics:")
        for metric in self.demo_metrics:
            print(f"  - {metric.metric_name}: {metric.metric_value} ({metric.metric_type.value})")
    
    async def demo_search_quality_analytics(self):
        """Demonstrate search quality analytics."""
        print("\n🔍 Demonstrating Search Quality Analytics...")
        
        search_quality = await self.analytics_service.calculate_search_quality_metrics(
            self.demo_tenant.id,
            start_date=date.today() - timedelta(days=30),
            end_date=date.today()
        )
        
        print("✓ Search Quality Metrics:")
        print(f"  - Click-through rate: {search_quality.click_through_rate:.1%}")
        print(f"  - Result relevance score: {search_quality.result_relevance_score:.3f}")
        print(f"  - User satisfaction rating: {search_quality.user_satisfaction_rating:.1f}/5.0")
        print(f"  - Search success rate: {search_quality.search_success_rate:.1%}")
        print(f"  - Average results per query: {search_quality.average_results_per_query:.1f}")
        print(f"  - Zero results rate: {search_quality.zero_results_rate:.1%}")
        print(f"  - Query refinement rate: {search_quality.query_refinement_rate:.1%}")
        print(f"  - Session abandonment rate: {search_quality.session_abandonment_rate:.1%}")
        
        if search_quality.trend_analysis:
            print(f"  - Trend analysis: {len(search_quality.trend_analysis)} trends identified")
        
        if search_quality.quality_alerts:
            print(f"  - Quality alerts: {len(search_quality.quality_alerts)} alerts")
        
        print(f"  - Benchmark comparison: {len(search_quality.benchmark_comparison)} benchmarks")
    
    async def demo_user_satisfaction_analysis(self):
        """Demonstrate user satisfaction analysis."""
        print("\n😊 Demonstrating User Satisfaction Analysis...")
        
        satisfaction = await self.analytics_service.calculate_user_satisfaction_score(
            self.demo_tenant.id,
            start_date=date.today() - timedelta(days=30),
            end_date=date.today()
        )
        
        print("✓ User Satisfaction Analysis:")
        print(f"  - Overall satisfaction score: {satisfaction.overall_satisfaction_score:.1f}/5.0")
        print(f"  - Satisfaction by segment: {len(satisfaction.satisfaction_by_segment)} segments")
        for segment, score in satisfaction.satisfaction_by_segment.items():
            print(f"    • {segment}: {score:.1f}/5.0")
        
        print(f"  - Satisfaction trends: {len(satisfaction.satisfaction_trends)} trends")
        print(f"  - Correlation analysis: {len(satisfaction.correlation_analysis)} correlations")
        for factor, correlation in satisfaction.correlation_analysis.items():
            print(f"    • {factor}: {correlation:.2f}")
        
        print(f"  - Satisfaction drivers: {len(satisfaction.satisfaction_drivers)} drivers")
        print(f"  - Prediction confidence: {satisfaction.prediction_confidence:.1%}")
        print(f"  - Predicted satisfaction: {satisfaction.predicted_satisfaction:.1f}/5.0")
    
    async def demo_content_quality_assessment(self):
        """Demonstrate content quality assessment."""
        print("\n📄 Demonstrating Content Quality Assessment...")
        
        content_quality = await self.analytics_service.assess_content_quality(
            self.demo_tenant.id,
            start_date=date.today() - timedelta(days=30),
            end_date=date.today()
        )
        
        print("✓ Content Quality Assessment:")
        print(f"  - Overall quality score: {content_quality.overall_quality_score:.1%}")
        print(f"  - Quality by category: {len(content_quality.quality_by_category)} categories")
        for category, score in content_quality.quality_by_category.items():
            print(f"    • {category}: {score:.1%}")
        
        print(f"  - Quality trends: {len(content_quality.quality_trends)} trends")
        print(f"  - Quality issues: {len(content_quality.quality_issues)} issues")
        print(f"  - Improvement opportunities: {len(content_quality.improvement_opportunities)} opportunities")
        print(f"  - Quality alerts: {len(content_quality.quality_alerts)} alerts")
        print(f"  - Content freshness score: {content_quality.content_freshness_score:.1%}")
        print(f"  - Accuracy score: {content_quality.accuracy_score:.1%}")
        print(f"  - Completeness score: {content_quality.completeness_score:.1%}")
    
    async def demo_trend_analysis(self):
        """Demonstrate trend analysis functionality."""
        print("\n📈 Demonstrating Trend Analysis...")
        
        # Create historical data for trend analysis
        metric_name = "click_through_rate"
        for i in range(14):  # 14 days of data
            days_ago = 14 - i
            trend_value = 0.6 + (i * 0.005) + random.uniform(-0.02, 0.02)  # Slight upward trend with noise
            
            historical_metric = AnalyticsMetric(
                tenant_id=self.demo_tenant.id,
                metric_type=AnalyticsMetricType.SEARCH_QUALITY,
                metric_name=metric_name,
                metric_value=trend_value,
                measurement_date=date.today() - timedelta(days=days_ago),
                sample_size=random.randint(800, 1200),
                metric_metadata={"demo": True, "historical": True}
            )
            self.db.add(historical_metric)
        
        self.db.commit()
        
        # Analyze trends
        trend_result = await self.analytics_service.analyze_trends(
            self.demo_tenant.id,
            metric_name,
            date.today() - timedelta(days=14),
            date.today()
        )
        
        print(f"✓ Trend Analysis for {metric_name}:")
        print(f"  - Trend direction: {trend_result.trend_direction}")
        print(f"  - Trend strength: {trend_result.trend_strength:.3f}")
        print(f"  - Change percentage: {trend_result.change_percentage:.1f}%")
        print(f"  - Statistical significance: {trend_result.statistical_significance:.3f}")
        print(f"  - Confidence interval: {trend_result.confidence_interval}")
        print(f"  - Sample size: {trend_result.metadata.get('sample_size', 'N/A')}")
    
    async def demo_recommendation_generation(self):
        """Demonstrate recommendation generation."""
        print("\n💡 Demonstrating Recommendation Generation...")
        
        # Detect improvement opportunities
        opportunities = await self.recommendation_service.detect_improvement_opportunities(
            self.demo_tenant.id, analysis_period_days=30
        )
        
        print(f"✓ Detected {len(opportunities)} improvement opportunities:")
        for i, opp in enumerate(opportunities[:3], 1):  # Show first 3
            print(f"  {i}. {opp.title}")
            print(f"     Type: {opp.opportunity_type}")
            print(f"     Impact: {opp.estimated_impact:.1%}")
            print(f"     Effort: {opp.implementation_effort}")
            print(f"     Priority: {opp.priority}")
        
        # Generate recommendations from opportunities
        recommendations = await self.recommendation_service.generate_recommendations_from_opportunities(
            self.demo_tenant.id, opportunities
        )
        
        print(f"\n✓ Generated {len(recommendations)} recommendations:")
        for rec in recommendations:
            print(f"  - {rec.title} ({rec.recommendation_type.value})")
            print(f"    Priority: {rec.priority.value}, Impact: {rec.estimated_impact:.1%}")
            self.demo_recommendations.append(rec)
        
        # Prioritize recommendations
        if recommendations:
            prioritized = await self.recommendation_service.prioritize_recommendations(
                self.demo_tenant.id, recommendations
            )
            print(f"\n✓ Prioritized {len(prioritized)} recommendations by impact and effort")
    
    async def demo_dashboard_configuration(self):
        """Demonstrate dashboard configuration."""
        print("\n📊 Demonstrating Dashboard Configuration...")
        
        # Create a sample dashboard configuration
        dashboard = DashboardConfiguration(
            tenant_id=self.demo_tenant.id,
            user_id=self.demo_user.id,
            name="Executive Analytics Dashboard",
            description="High-level analytics for executives",
            dashboard_type="executive",
            layout_config={"columns": 12, "rows": 10},
            components=[
                {
                    "type": "metric_card",
                    "position": {"x": 0, "y": 0, "w": 3, "h": 2},
                    "config": {"metric": "search_quality_score", "title": "Search Quality"}
                },
                {
                    "type": "metric_card",
                    "position": {"x": 3, "y": 0, "w": 3, "h": 2},
                    "config": {"metric": "user_satisfaction", "title": "User Satisfaction"}
                },
                {
                    "type": "line_chart",
                    "position": {"x": 0, "y": 2, "w": 6, "h": 4},
                    "config": {"metric": "click_through_rate", "title": "CTR Trend", "period": "30d"}
                },
                {
                    "type": "bar_chart",
                    "position": {"x": 6, "y": 0, "w": 6, "h": 6},
                    "config": {"metric": "satisfaction_by_segment", "title": "Satisfaction by User Segment"}
                }
            ],
            refresh_interval_minutes=5,
            is_active=True,
            is_default=True
        )
        
        self.db.add(dashboard)
        self.db.commit()
        self.db.refresh(dashboard)
        
        print(f"✓ Created dashboard configuration: {dashboard.name}")
        print(f"  - Type: {dashboard.dashboard_type}")
        print(f"  - Components: {len(dashboard.components)}")
        print(f"  - Refresh interval: {dashboard.refresh_interval_minutes} minutes")
        print(f"  - Is default: {dashboard.is_default}")
    
    async def show_demo_results(self):
        """Show final demo results and statistics."""
        print("\n📊 Demo Results Summary...")
        
        # Database statistics
        total_metrics = self.db.query(AnalyticsMetric).filter(
            AnalyticsMetric.tenant_id == self.demo_tenant.id
        ).count()
        
        total_recommendations = self.db.query(PerformanceRecommendation).filter(
            PerformanceRecommendation.tenant_id == self.demo_tenant.id
        ).count()
        
        total_dashboards = self.db.query(DashboardConfiguration).filter(
            DashboardConfiguration.tenant_id == self.demo_tenant.id
        ).count()
        
        total_feedback = self.db.query(UserFeedbackSubmission).filter(
            UserFeedbackSubmission.tenant_id == self.demo_tenant.id
        ).count()
        
        print(f"✓ Database Statistics:")
        print(f"  - Analytics metrics: {total_metrics}")
        print(f"  - Performance recommendations: {total_recommendations}")
        print(f"  - Dashboard configurations: {total_dashboards}")
        print(f"  - Feedback submissions: {total_feedback}")
        
        # System capabilities demonstrated
        print(f"\n✓ System Capabilities Demonstrated:")
        print(f"  - ✅ Analytics metric creation and management")
        print(f"  - ✅ Search quality metrics calculation")
        print(f"  - ✅ User satisfaction scoring and analysis")
        print(f"  - ✅ Content quality assessment")
        print(f"  - ✅ Trend analysis with statistical validation")
        print(f"  - ✅ Automated recommendation generation")
        print(f"  - ✅ Recommendation prioritization and tracking")
        print(f"  - ✅ Dashboard configuration and customization")
        print(f"  - ✅ Multi-tenant data isolation")
        print(f"  - ✅ Comprehensive analytics insights")


async def main():
    """Main demonstration function."""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get database session
    db_session = get_database_session()
    
    try:
        # Run demonstration
        demo = AnalyticsSystemDemo(db_session)
        await demo.run_complete_demo()
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        logger.error("demonstration_failed", error=str(e), exc_info=True)
        raise
    finally:
        db_session.close()


if __name__ == "__main__":
    asyncio.run(main())
