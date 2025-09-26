#!/usr/bin/env python3
"""
Learning Algorithms System Demonstration Script for Sprint 6 Story 6-03

This script demonstrates the complete learning algorithms system functionality
including algorithm creation, feedback processing, performance monitoring,
and A/B testing capabilities.
"""

import asyncio
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

import structlog
from sqlalchemy.orm import Session

from agentic_rag.database.connection import get_database_session
from agentic_rag.models.tenant import Tenant
from agentic_rag.models.auth import User
from agentic_rag.models.learning import (
    LearningAlgorithm,
    FeedbackSignal,
    LearningPerformanceMetric,
    ABTestExperiment,
    LearningAlgorithmType,
    LearningModelType,
    FeedbackSignalType
)
from agentic_rag.models.feedback import UserFeedbackSubmission, FeedbackType
from agentic_rag.services.learning_service import LearningService
from agentic_rag.services.learning_integration_service import LearningIntegrationService
from agentic_rag.services.learning_monitoring_service import LearningMonitoringService

logger = structlog.get_logger(__name__)


class LearningSystemDemo:
    """Demonstration of the learning algorithms system."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.learning_service = LearningService(db_session)
        self.integration_service = LearningIntegrationService(db_session)
        self.monitoring_service = LearningMonitoringService(db_session)
        self.demo_tenant = None
        self.demo_user = None
        self.demo_algorithms = []
        self.demo_signals = []
    
    async def run_complete_demo(self):
        """Run complete learning system demonstration."""
        print("🚀 Starting Learning Algorithms System Demonstration")
        print("=" * 60)
        
        try:
            # Setup demo environment
            await self.setup_demo_environment()
            
            # Demonstrate core functionality
            await self.demo_algorithm_creation()
            await self.demo_feedback_signal_processing()
            await self.demo_learning_integration()
            await self.demo_performance_monitoring()
            await self.demo_ab_testing()
            await self.demo_health_monitoring()
            
            # Show final results
            await self.show_demo_results()
            
            print("\n✅ Learning Algorithms System Demonstration Completed Successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Demo failed: {str(e)}")
            logger.error("demo_failed", error=str(e), exc_info=True)
            raise
    
    async def setup_demo_environment(self):
        """Set up demo tenant and user."""
        print("\n📋 Setting up demo environment...")
        
        # Create demo tenant
        self.demo_tenant = Tenant(
            name="Learning Demo Tenant",
            slug="learning-demo",
            description="Tenant for learning algorithms demonstration"
        )
        self.db.add(self.demo_tenant)
        
        # Create demo user
        self.demo_user = User(
            tenant_id=self.demo_tenant.id,
            email="demo@learningdemo.com",
            username="learning_demo_user",
            full_name="Learning Demo User",
            role="admin",
            is_active=True
        )
        self.db.add(self.demo_user)
        
        self.db.commit()
        print(f"✓ Created demo tenant: {self.demo_tenant.name}")
        print(f"✓ Created demo user: {self.demo_user.username}")
    
    async def demo_algorithm_creation(self):
        """Demonstrate learning algorithm creation."""
        print("\n🧠 Demonstrating Learning Algorithm Creation...")
        
        # Create different types of learning algorithms
        algorithms_config = [
            {
                "type": LearningAlgorithmType.LINK_CONFIDENCE,
                "model": LearningModelType.EXPONENTIAL_MOVING_AVERAGE,
                "name": "Link Confidence Learner",
                "description": "Learns to adjust link confidence based on user feedback",
                "learning_rate": 0.01
            },
            {
                "type": LearningAlgorithmType.CHUNK_RANKING,
                "model": LearningModelType.BAYESIAN_UPDATE,
                "name": "Chunk Ranking Optimizer",
                "description": "Optimizes chunk ranking based on user interactions",
                "learning_rate": 0.015
            },
            {
                "type": LearningAlgorithmType.QUERY_EXPANSION,
                "model": LearningModelType.NEURAL_LANGUAGE_MODEL,
                "name": "Query Expansion Engine",
                "description": "Learns effective query expansion strategies",
                "learning_rate": 0.005
            },
            {
                "type": LearningAlgorithmType.NEGATIVE_FEEDBACK,
                "model": LearningModelType.REINFORCEMENT_LEARNING,
                "name": "Negative Feedback Handler",
                "description": "Handles negative feedback and penalization",
                "learning_rate": 0.02
            }
        ]
        
        for config in algorithms_config:
            algorithm = LearningAlgorithm(
                tenant_id=self.demo_tenant.id,
                algorithm_type=config["type"],
                model_type=config["model"],
                name=config["name"],
                description=config["description"],
                learning_rate=config["learning_rate"],
                validation_threshold=0.05,
                is_enabled=True,
                auto_update=True
            )
            
            self.db.add(algorithm)
            self.demo_algorithms.append(algorithm)
        
        self.db.commit()
        
        print(f"✓ Created {len(self.demo_algorithms)} learning algorithms:")
        for alg in self.demo_algorithms:
            print(f"  - {alg.name} ({alg.algorithm_type.value})")
        
        # Initialize algorithms in learning service
        await self.learning_service.initialize_algorithms(self.demo_tenant.id)
        print(f"✓ Initialized {len(self.learning_service.algorithms)} algorithms in learning service")
    
    async def demo_feedback_signal_processing(self):
        """Demonstrate feedback signal creation and processing."""
        print("\n📊 Demonstrating Feedback Signal Processing...")
        
        # Create various types of feedback signals
        signal_configs = [
            {
                "type": FeedbackSignalType.EXPLICIT_RATING,
                "target_type": "chunk",
                "value": 4.5,
                "strength": 1.0,
                "confidence": 0.9,
                "context": "User rated search result highly relevant"
            },
            {
                "type": FeedbackSignalType.CLICK_THROUGH,
                "target_type": "link",
                "value": 1.0,
                "strength": 0.8,
                "confidence": 1.0,
                "context": "User clicked on document link"
            },
            {
                "type": FeedbackSignalType.DWELL_TIME,
                "target_type": "chunk",
                "value": 45.0,  # 45 seconds
                "strength": 0.7,
                "confidence": 0.8,
                "context": "User spent significant time reading content"
            },
            {
                "type": FeedbackSignalType.BOUNCE_RATE,
                "target_type": "chunk",
                "value": 0.2,  # Low bounce rate (good)
                "strength": 0.6,
                "confidence": 0.7,
                "context": "Low bounce rate indicates good content"
            },
            {
                "type": FeedbackSignalType.CONVERSION_RATE,
                "target_type": "query",
                "value": 0.8,  # High conversion rate
                "strength": 0.9,
                "confidence": 0.95,
                "context": "Query led to successful task completion"
            }
        ]
        
        for config in signal_configs:
            signal = await self.integration_service.create_feedback_signal(
                tenant_id=self.demo_tenant.id,
                signal_type=config["type"],
                target_type=config["target_type"],
                target_id=uuid.uuid4(),
                signal_value=config["value"],
                user_id=self.demo_user.id,
                session_id=f"demo-session-{random.randint(1000, 9999)}",
                query_context=config["context"],
                metadata={"demo": True, "signal_description": config["context"]}
            )
            self.demo_signals.append(signal)
        
        print(f"✓ Created {len(self.demo_signals)} feedback signals:")
        for signal in self.demo_signals:
            print(f"  - {signal.signal_type.value}: {signal.signal_value} ({signal.target_type})")
        
        # Process signals through learning algorithms
        processed_count = 0
        for signal in self.demo_signals:
            try:
                updates = await self.learning_service.process_feedback_signal(
                    self.demo_tenant.id, signal
                )
                processed_count += len(updates)
                print(f"  ✓ Processed signal {signal.signal_type.value}: {len(updates)} algorithm updates")
            except Exception as e:
                print(f"  ⚠ Failed to process signal {signal.signal_type.value}: {str(e)}")
        
        print(f"✓ Total algorithm updates applied: {processed_count}")
    
    async def demo_learning_integration(self):
        """Demonstrate learning integration with feedback system."""
        print("\n🔗 Demonstrating Learning Integration...")
        
        # Create sample feedback submissions
        feedback_submissions = []
        for i in range(5):
            feedback = UserFeedbackSubmission(
                tenant_id=self.demo_tenant.id,
                user_id=self.demo_user.id,
                feedback_type=random.choice([FeedbackType.SEARCH_RESULT, FeedbackType.LINK_QUALITY, FeedbackType.ANSWER_QUALITY]),
                target_id=uuid.uuid4(),
                rating=random.randint(3, 5),
                comment=f"Demo feedback submission {i+1}",
                status="completed",
                context={"demo": True, "submission_number": i+1}
            )
            self.db.add(feedback)
            feedback_submissions.append(feedback)
        
        self.db.commit()
        print(f"✓ Created {len(feedback_submissions)} feedback submissions")
        
        # Process feedback for learning
        result = await self.integration_service.process_feedback_for_learning(
            self.demo_tenant.id, batch_size=10
        )
        
        print(f"✓ Feedback processing results:")
        print(f"  - Signals processed: {result.signals_processed}")
        print(f"  - Algorithms updated: {result.algorithms_updated}")
        print(f"  - Improvements applied: {result.improvements_applied}")
        print(f"  - Processing time: {result.processing_time_seconds:.2f}s")
        if result.errors:
            print(f"  - Errors: {len(result.errors)}")
    
    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring and validation."""
        print("\n📈 Demonstrating Performance Monitoring...")
        
        # Create sample performance metrics for algorithms
        for algorithm in self.demo_algorithms:
            # Create historical metrics
            for days_ago in range(7, 0, -1):
                metric_date = datetime.now() - timedelta(days=days_ago)
                
                # Simulate improving performance over time
                base_score = 0.6 + (7 - days_ago) * 0.05
                noise = random.uniform(-0.02, 0.02)
                metric_value = min(1.0, max(0.0, base_score + noise))
                
                metric = LearningPerformanceMetric(
                    tenant_id=self.demo_tenant.id,
                    algorithm_id=algorithm.id,
                    metric_name="accuracy",
                    metric_value=metric_value,
                    metric_type="accuracy",
                    measurement_period_start=metric_date - timedelta(hours=1),
                    measurement_period_end=metric_date,
                    sample_size=random.randint(50, 200),
                    baseline_value=0.6,
                    improvement_percentage=(metric_value - 0.6) / 0.6 * 100,
                    metric_metadata={"demo": True, "simulated": True}
                )
                self.db.add(metric)
        
        self.db.commit()
        print("✓ Created historical performance metrics")
        
        # Validate algorithm performance
        for algorithm in self.demo_algorithms[:2]:  # Validate first 2 algorithms
            try:
                validation_result = await self.monitoring_service.validate_algorithm_performance(
                    algorithm.id, validation_period_hours=24, baseline_period_hours=168
                )
                
                print(f"✓ Validation for {algorithm.name}:")
                print(f"  - Status: {validation_result.status.value}")
                print(f"  - Score: {validation_result.score:.3f}")
                print(f"  - Improvement: {validation_result.improvement_percentage:.1f}%")
                print(f"  - Statistical significance: {validation_result.statistical_significance:.3f}")
                print(f"  - Recommendations: {len(validation_result.recommendations)}")
                
            except Exception as e:
                print(f"  ⚠ Validation failed for {algorithm.name}: {str(e)}")
    
    async def demo_ab_testing(self):
        """Demonstrate A/B testing functionality."""
        print("\n🧪 Demonstrating A/B Testing...")
        
        if len(self.demo_algorithms) >= 2:
            # Create A/B test experiment
            control_alg = self.demo_algorithms[0]
            treatment_alg = self.demo_algorithms[1]
            
            experiment = await self.monitoring_service.create_ab_test_experiment(
                tenant_id=self.demo_tenant.id,
                experiment_name="Link Confidence A/B Test",
                control_algorithm_id=control_alg.id,
                treatment_algorithm_id=treatment_alg.id,
                primary_metric="click_through_rate",
                success_threshold=0.05,
                traffic_split_percentage=50.0,
                description="Testing different link confidence algorithms",
                hypothesis="Bayesian update will outperform exponential moving average"
            )
            
            print(f"✓ Created A/B test experiment: {experiment.experiment_name}")
            print(f"  - Control: {control_alg.name}")
            print(f"  - Treatment: {treatment_alg.name}")
            print(f"  - Primary metric: {experiment.primary_metric}")
            print(f"  - Success threshold: {experiment.success_threshold}")
            
            # Start experiment
            started_experiment = await self.monitoring_service.start_ab_test_experiment(experiment.id)
            print(f"✓ Started experiment (Status: {started_experiment.status})")
            
            # Simulate some performance data
            for algorithm_id in [control_alg.id, treatment_alg.id]:
                for _ in range(10):
                    metric = LearningPerformanceMetric(
                        tenant_id=self.demo_tenant.id,
                        algorithm_id=algorithm_id,
                        metric_name="click_through_rate",
                        metric_value=random.uniform(0.15, 0.25),
                        metric_type="rate",
                        measurement_period_start=datetime.now() - timedelta(minutes=30),
                        measurement_period_end=datetime.now(),
                        sample_size=random.randint(100, 500),
                        metric_metadata={"experiment_id": str(experiment.id)}
                    )
                    self.db.add(metric)
            
            self.db.commit()
            print("✓ Generated sample performance data for experiment")
            
            # Analyze results
            try:
                results = await self.monitoring_service.analyze_ab_test_results(experiment.id)
                print(f"✓ A/B test analysis results:")
                print(f"  - Control performance: {results.control_performance:.3f}")
                print(f"  - Treatment performance: {results.treatment_performance:.3f}")
                print(f"  - Improvement: {results.improvement_percentage:.1f}%")
                print(f"  - Statistical significance: {results.statistical_significance:.3f}")
                print(f"  - Is significant: {results.is_significant}")
                print(f"  - Recommendation: {results.recommendation}")
            except Exception as e:
                print(f"  ⚠ A/B test analysis failed: {str(e)}")
        else:
            print("⚠ Not enough algorithms for A/B testing demo")
    
    async def demo_health_monitoring(self):
        """Demonstrate health monitoring functionality."""
        print("\n🏥 Demonstrating Health Monitoring...")
        
        # Perform health check
        health_checks = await self.monitoring_service.perform_learning_health_check(
            self.demo_tenant.id
        )
        
        print(f"✓ Health check completed for {len(health_checks)} algorithms:")
        for health_check in health_checks:
            print(f"  - {health_check.algorithm_type.value}:")
            print(f"    • Health score: {health_check.health_score:.2f}")
            print(f"    • Status: {health_check.status}")
            print(f"    • Performance trend: {health_check.performance_trend}")
            print(f"    • Issues: {len(health_check.issues)}")
            print(f"    • Recommendations: {len(health_check.recommendations)}")
            
            if health_check.issues:
                print(f"    • Top issues: {', '.join(health_check.issues[:2])}")
    
    async def show_demo_results(self):
        """Show final demo results and insights."""
        print("\n📊 Demo Results Summary...")
        
        # Get learning insights
        insights = await self.integration_service.get_learning_insights(
            self.demo_tenant.id, time_period_hours=24
        )
        
        print(f"✓ Learning insights (last 24 hours):")
        print(f"  - Total signals processed: {insights['total_signals_processed']}")
        print(f"  - Active algorithms: {insights['active_algorithms']}")
        
        if insights['signal_statistics']:
            print("  - Signal statistics:")
            for signal_type, stats in insights['signal_statistics'].items():
                print(f"    • {signal_type}: {stats['count']} signals, avg value: {stats['average_value']:.2f}")
        
        if insights['algorithm_performance']:
            print("  - Algorithm performance:")
            for alg_type, perf in insights['algorithm_performance'].items():
                accuracy = perf.get('accuracy_score', 'N/A')
                print(f"    • {alg_type}: accuracy {accuracy}")
        
        # Show database statistics
        total_algorithms = self.db.query(LearningAlgorithm).filter(
            LearningAlgorithm.tenant_id == self.demo_tenant.id
        ).count()
        
        total_signals = self.db.query(FeedbackSignal).filter(
            FeedbackSignal.tenant_id == self.demo_tenant.id
        ).count()
        
        total_metrics = self.db.query(LearningPerformanceMetric).filter(
            LearningPerformanceMetric.tenant_id == self.demo_tenant.id
        ).count()
        
        print(f"\n📈 Database Statistics:")
        print(f"  - Learning algorithms: {total_algorithms}")
        print(f"  - Feedback signals: {total_signals}")
        print(f"  - Performance metrics: {total_metrics}")


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
        demo = LearningSystemDemo(db_session)
        await demo.run_complete_demo()
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        logger.error("demonstration_failed", error=str(e), exc_info=True)
        raise
    finally:
        db_session.close()


if __name__ == "__main__":
    asyncio.run(main())
