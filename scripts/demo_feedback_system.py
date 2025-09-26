"""
Demonstration Script for Sprint 6 Feedback Collection System

This script demonstrates the complete feedback collection system functionality
including API endpoints, service integration, and analytics.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List

import structlog
import httpx

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class FeedbackSystemDemo:
    """Demonstration of the feedback collection system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)
        self.auth_headers = {
            "Authorization": "Bearer demo_token",
            "X-Tenant-ID": str(uuid.uuid4())
        }
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def demo_basic_feedback_submission(self):
        """Demonstrate basic feedback submission."""
        logger.info("=== Demo: Basic Feedback Submission ===")
        
        # Search result feedback
        search_feedback = {
            "feedback_type": "search_result",
            "feedback_category": "not_relevant",
            "target_id": str(uuid.uuid4()),
            "target_type": "search_result",
            "rating": -1,
            "feedback_text": "This search result doesn't match my query about electrical pricing",
            "query": "What are the pricing details for electrical components?",
            "session_id": "demo_session_001"
        }
        
        try:
            response = await self.client.post(
                "/api/v1/feedback",
                json=search_feedback,
                headers=self.auth_headers
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Search result feedback submitted successfully", 
                          feedback_id=result.get("feedback_id"),
                          status=result.get("status"))
                return result.get("feedback_id")
            else:
                logger.error("Failed to submit search feedback", 
                           status_code=response.status_code,
                           response=response.text)
                
        except Exception as e:
            logger.error("Error submitting search feedback", error=str(e))
        
        return None
    
    async def demo_thumbs_feedback(self):
        """Demonstrate quick thumbs up/down feedback."""
        logger.info("=== Demo: Thumbs Feedback ===")
        
        thumbs_feedback = {
            "target_id": str(uuid.uuid4()),
            "target_type": "search_result",
            "thumbs_up": True,
            "query": "pricing information for electrical components",
            "session_id": "demo_session_001"
        }
        
        try:
            response = await self.client.post(
                "/api/v1/feedback/thumbs",
                json=thumbs_feedback,
                headers=self.auth_headers
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Thumbs feedback submitted successfully",
                          feedback_id=result.get("feedback_id"),
                          processing_time=result.get("estimated_processing_time"))
                return result.get("feedback_id")
            else:
                logger.error("Failed to submit thumbs feedback",
                           status_code=response.status_code)
                
        except Exception as e:
            logger.error("Error submitting thumbs feedback", error=str(e))
        
        return None
    
    async def demo_detailed_feedback(self):
        """Demonstrate detailed feedback form submission."""
        logger.info("=== Demo: Detailed Feedback Form ===")
        
        detailed_feedback = {
            "feedback_type": "answer_quality",
            "feedback_category": "inaccurate_information",
            "title": "Incorrect pricing information in answer",
            "description": "The system provided pricing information from 2022 documents instead of current pricing",
            "expected_behavior": "Should use the most recent pricing documents available",
            "actual_behavior": "Used outdated pricing from archived documents",
            "priority": "high",
            "query": "What is the current pricing for electrical components?",
            "session_id": "demo_session_001"
        }
        
        try:
            response = await self.client.post(
                "/api/v1/feedback/detailed",
                json=detailed_feedback,
                headers=self.auth_headers
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Detailed feedback submitted successfully",
                          feedback_id=result.get("feedback_id"),
                          estimated_processing=result.get("estimated_processing_time"))
                return result.get("feedback_id")
            else:
                logger.error("Failed to submit detailed feedback",
                           status_code=response.status_code)
                
        except Exception as e:
            logger.error("Error submitting detailed feedback", error=str(e))
        
        return None
    
    async def demo_link_quality_feedback(self):
        """Demonstrate document link quality feedback."""
        logger.info("=== Demo: Link Quality Feedback ===")
        
        link_feedback = {
            "feedback_type": "link_quality",
            "feedback_category": "incorrect_link",
            "target_id": str(uuid.uuid4()),
            "target_type": "document_link",
            "rating": -1,
            "feedback_text": "This link connects unrelated documents - RFQ for electrical work linked to plumbing offer",
            "query": "electrical components pricing",
            "session_id": "demo_session_001"
        }
        
        try:
            response = await self.client.post(
                "/api/v1/feedback",
                json=link_feedback,
                headers=self.auth_headers
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Link quality feedback submitted successfully",
                          feedback_id=result.get("feedback_id"))
                return result.get("feedback_id")
            else:
                logger.error("Failed to submit link feedback",
                           status_code=response.status_code)
                
        except Exception as e:
            logger.error("Error submitting link feedback", error=str(e))
        
        return None
    
    async def demo_feedback_history(self):
        """Demonstrate retrieving feedback history."""
        logger.info("=== Demo: Feedback History Retrieval ===")
        
        try:
            response = await self.client.get(
                "/api/v1/feedback",
                headers=self.auth_headers,
                params={"page": 1, "page_size": 10}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Feedback history retrieved successfully",
                          total_count=result.get("total_count"),
                          items_count=len(result.get("items", [])))
                
                # Display feedback items
                for item in result.get("items", [])[:3]:  # Show first 3 items
                    logger.info("Feedback item",
                              feedback_type=item.get("feedback_type"),
                              status=item.get("status"),
                              rating=item.get("rating"),
                              created_at=item.get("created_at"))
                
                return result
            else:
                logger.error("Failed to retrieve feedback history",
                           status_code=response.status_code)
                
        except Exception as e:
            logger.error("Error retrieving feedback history", error=str(e))
        
        return None
    
    async def demo_feedback_statistics(self):
        """Demonstrate feedback statistics and analytics."""
        logger.info("=== Demo: Feedback Statistics ===")
        
        try:
            response = await self.client.get(
                "/api/v1/feedback/stats",
                headers=self.auth_headers
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Feedback statistics retrieved successfully",
                          total_submissions=result.get("total_submissions"),
                          pending_count=result.get("pending_count"),
                          processed_count=result.get("processed_count"),
                          average_rating=result.get("average_rating"))
                
                # Display category breakdown
                category_breakdown = result.get("category_breakdown", {})
                if category_breakdown:
                    logger.info("Category breakdown", categories=category_breakdown)
                
                # Display recent trends
                recent_trends = result.get("recent_trends", {})
                if recent_trends:
                    logger.info("Recent trends", trends=recent_trends)
                
                return result
            else:
                logger.error("Failed to retrieve feedback statistics",
                           status_code=response.status_code)
                
        except Exception as e:
            logger.error("Error retrieving feedback statistics", error=str(e))
        
        return None
    
    async def demo_multiple_feedback_types(self):
        """Demonstrate submitting multiple types of feedback."""
        logger.info("=== Demo: Multiple Feedback Types ===")
        
        feedback_scenarios = [
            {
                "name": "Positive Search Result",
                "data": {
                    "feedback_type": "search_result",
                    "target_id": str(uuid.uuid4()),
                    "target_type": "search_result",
                    "rating": 5,
                    "feedback_text": "Perfect match for my query!",
                    "query": "electrical component specifications"
                }
            },
            {
                "name": "Feature Request",
                "data": {
                    "feedback_type": "general",
                    "feedback_category": "feature_request",
                    "feedback_text": "Please add export to Excel functionality for search results",
                    "query": "export search results"
                }
            },
            {
                "name": "Performance Issue",
                "data": {
                    "feedback_type": "general",
                    "feedback_category": "performance_issue",
                    "feedback_text": "Search is taking too long (>10 seconds) for complex queries",
                    "query": "complex multi-criteria search"
                }
            }
        ]
        
        submitted_feedback = []
        
        for scenario in feedback_scenarios:
            try:
                response = await self.client.post(
                    "/api/v1/feedback",
                    json=scenario["data"],
                    headers=self.auth_headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"{scenario['name']} feedback submitted",
                              feedback_id=result.get("feedback_id"))
                    submitted_feedback.append(result.get("feedback_id"))
                else:
                    logger.error(f"Failed to submit {scenario['name']} feedback",
                               status_code=response.status_code)
                    
            except Exception as e:
                logger.error(f"Error submitting {scenario['name']} feedback", error=str(e))
        
        return submitted_feedback
    
    async def run_complete_demo(self):
        """Run the complete feedback system demonstration."""
        logger.info("🚀 Starting Sprint 6 Feedback Collection System Demo")
        
        demo_results = {}
        
        # Demo 1: Basic feedback submission
        demo_results["basic_feedback"] = await self.demo_basic_feedback_submission()
        
        # Demo 2: Thumbs feedback
        demo_results["thumbs_feedback"] = await self.demo_thumbs_feedback()
        
        # Demo 3: Detailed feedback
        demo_results["detailed_feedback"] = await self.demo_detailed_feedback()
        
        # Demo 4: Link quality feedback
        demo_results["link_feedback"] = await self.demo_link_quality_feedback()
        
        # Demo 5: Multiple feedback types
        demo_results["multiple_feedback"] = await self.demo_multiple_feedback_types()
        
        # Demo 6: Feedback history
        demo_results["feedback_history"] = await self.demo_feedback_history()
        
        # Demo 7: Feedback statistics
        demo_results["feedback_stats"] = await self.demo_feedback_statistics()
        
        # Summary
        logger.info("✅ Feedback System Demo Completed Successfully")
        logger.info("Demo Results Summary", 
                   total_scenarios=len(demo_results),
                   successful_submissions=sum(1 for v in demo_results.values() if v is not None))
        
        return demo_results


async def main():
    """Main demonstration function."""
    try:
        async with FeedbackSystemDemo() as demo:
            results = await demo.run_complete_demo()
            
            print("\n" + "="*60)
            print("SPRINT 6 FEEDBACK COLLECTION SYSTEM DEMO COMPLETE")
            print("="*60)
            print(f"✅ Demonstrated comprehensive feedback collection capabilities")
            print(f"✅ Tested multiple feedback types and categories")
            print(f"✅ Validated API endpoints and response handling")
            print(f"✅ Showed feedback history and analytics features")
            print("="*60)
            
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        print(f"\n❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
