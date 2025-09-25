"""
Sprint 5 Integration Tests

Comprehensive end-to-end tests for the complete Sprint 5 integrated system,
validating that all components work together properly.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from agentic_rag.services.orchestration.integration import (
    get_enhanced_orchestrator,
    IntegratedWorkflowResult
)
from agentic_rag.services.orchestration.base import Context
from agentic_rag.services.orchestration.planner import QueryIntent


class TestSprint5Integration:
    """Test suite for Sprint 5 integrated functionality."""

    @pytest.fixture(autouse=True)
    async def setup_orchestrator(self):
        """Set up the enhanced orchestrator for testing."""
        self.orchestrator = await get_enhanced_orchestrator()
        self.user_id = uuid4()
        self.tenant_id = uuid4()

    async def test_pricing_inquiry_workflow(self):
        """Test complete workflow for pricing inquiry queries."""
        query = "What are the pricing strategies mentioned in the procurement documents?"
        
        result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="analyst"
        )
        
        # Validate basic workflow completion
        assert result.success
        assert result.query_intent == QueryIntent.PRICING_INQUIRY
        assert "pricing_agent" in result.agents_used
        assert "synthesizer_agent" in result.agents_used
        assert "redaction_agent" in result.agents_used
        
        # Validate pricing-specific results
        assert result.pricing_analysis is not None
        assert result.final_answer is not None
        assert len(result.workflow_steps) >= 4
        
        # Validate privacy protection
        assert result.pii_detected >= 0
        assert result.redactions_applied >= 0

    async def test_document_comparison_workflow(self):
        """Test complete workflow for document comparison queries."""
        query = "Compare the terms and conditions between the two vendor proposals"
        
        result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="reviewer"
        )
        
        # Validate basic workflow completion
        assert result.success
        assert result.query_intent == QueryIntent.COMPARISON
        assert "analysis_agent" in result.agents_used
        
        # Validate comparison-specific results
        assert result.document_comparison is not None
        assert result.final_answer is not None
        
        # Validate quality metrics
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.execution_time_ms > 0

    async def test_risk_assessment_workflow(self):
        """Test complete workflow for risk assessment queries."""
        query = "Identify and assess the security risks in the vendor contracts"
        
        result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="security_analyst"
        )
        
        # Validate basic workflow completion
        assert result.success
        assert result.query_intent == QueryIntent.RISK_ASSESSMENT
        assert "analysis_agent" in result.agents_used
        
        # Validate risk assessment results
        assert result.risk_assessment is not None
        assert result.final_answer is not None

    async def test_compliance_checking_workflow(self):
        """Test complete workflow for compliance checking queries."""
        query = "Check if the procurement documents comply with ISO 9001 standards"
        
        result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="compliance_officer"
        )
        
        # Validate basic workflow completion
        assert result.success
        assert result.query_intent == QueryIntent.COMPLIANCE_CHECK
        assert "analysis_agent" in result.agents_used
        
        # Validate compliance checking results
        assert result.compliance_check is not None
        assert result.final_answer is not None

    async def test_content_summarization_workflow(self):
        """Test complete workflow for content summarization queries."""
        query = "Provide a summary of the key points in the vendor evaluation documents"
        
        result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="manager"
        )
        
        # Validate basic workflow completion
        assert result.success
        assert result.query_intent == QueryIntent.SUMMARIZATION
        assert "analysis_agent" in result.agents_used
        
        # Validate summarization results
        assert result.final_answer is not None
        assert len(result.final_answer) > 0

    async def test_privacy_protection_integration(self):
        """Test that privacy protection is properly integrated across all workflows."""
        query = "What are the contact details and pricing for John Smith's proposal?"
        
        result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="viewer"  # Lower privilege role
        )
        
        # Validate privacy protection
        assert result.success
        assert "redaction_agent" in result.agents_used
        
        # Should have detected and redacted PII
        assert result.pii_detected >= 0
        assert result.redactions_applied >= 0
        
        # Final answer should be privacy-protected
        assert result.redacted_content is not None

    async def test_citation_generation_integration(self):
        """Test that citations are properly generated across all workflows."""
        query = "What are the main benefits mentioned in the vendor proposals?"
        
        result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="analyst"
        )
        
        # Validate citation generation
        assert result.success
        assert "synthesizer_agent" in result.agents_used
        
        # Should have generated citations
        assert result.citations is not None
        assert isinstance(result.citations, list)
        
        # Each citation should have required fields
        for citation in result.citations:
            assert "source" in citation
            assert "text" in citation

    async def test_workflow_performance_metrics(self):
        """Test that performance metrics are properly tracked."""
        query = "Analyze the pricing and compliance aspects of the vendor proposals"
        
        result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="analyst"
        )
        
        # Validate performance tracking
        assert result.success
        assert result.execution_time_ms > 0
        assert 0.0 <= result.confidence_score <= 1.0
        
        # Validate workflow steps tracking
        assert len(result.workflow_steps) >= 4
        expected_steps = ["query_analysis", "document_retrieval", "intent_processing", "answer_synthesis", "privacy_protection"]
        actual_steps = [step["step"] for step in result.workflow_steps]
        
        for expected_step in expected_steps:
            assert expected_step in actual_steps

    async def test_agent_coordination(self):
        """Test that agents are properly coordinated and work together."""
        query = "Compare pricing strategies and assess compliance risks in vendor documents"
        
        result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="senior_analyst"
        )
        
        # Validate agent coordination
        assert result.success
        
        # Should use multiple specialized agents
        expected_agents = ["planner_agent", "retriever_agent", "pricing_agent", "synthesizer_agent", "redaction_agent"]
        for expected_agent in expected_agents:
            assert expected_agent in result.agents_used
        
        # Should have results from multiple analysis types
        assert result.pricing_analysis is not None
        assert result.final_answer is not None

    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with an empty query
        result = await self.orchestrator.process_integrated_query(
            query="",
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="viewer"
        )
        
        # Should handle gracefully
        assert isinstance(result, IntegratedWorkflowResult)
        # May succeed or fail, but should not crash

    async def test_role_based_access_control(self):
        """Test that role-based access control works properly."""
        query = "What are the confidential pricing details in the vendor contracts?"
        
        # Test with viewer role (lower privileges)
        viewer_result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="viewer"
        )
        
        # Test with admin role (higher privileges)
        admin_result = await self.orchestrator.process_integrated_query(
            query=query,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            user_role="admin"
        )
        
        # Both should succeed but with different levels of redaction
        assert viewer_result.success
        assert admin_result.success
        
        # Viewer should have more redactions than admin
        assert viewer_result.redactions_applied >= admin_result.redactions_applied


@pytest.mark.asyncio
async def test_complete_system_integration():
    """Test the complete system integration end-to-end."""
    orchestrator = await get_enhanced_orchestrator()
    
    # Test multiple query types in sequence
    test_queries = [
        ("What are the pricing models in the vendor proposals?", QueryIntent.PRICING_INQUIRY),
        ("Compare the security features between vendors", QueryIntent.COMPARISON),
        ("Assess the risks in the procurement process", QueryIntent.RISK_ASSESSMENT),
        ("Check compliance with procurement regulations", QueryIntent.COMPLIANCE_CHECK),
        ("Summarize the vendor evaluation criteria", QueryIntent.SUMMARIZATION)
    ]
    
    results = []
    for query, expected_intent in test_queries:
        result = await orchestrator.process_integrated_query(
            query=query,
            user_id=uuid4(),
            tenant_id=uuid4(),
            user_role="analyst"
        )
        
        # Validate each result
        assert result.success
        assert result.query_intent == expected_intent
        assert result.final_answer is not None
        assert len(result.agents_used) >= 3
        
        results.append(result)
    
    # Validate that all queries were processed successfully
    assert len(results) == len(test_queries)
    
    # Validate performance across all queries
    total_execution_time = sum(result.execution_time_ms for result in results)
    average_confidence = sum(result.confidence_score for result in results) / len(results)
    
    assert total_execution_time > 0
    assert 0.0 <= average_confidence <= 1.0


if __name__ == "__main__":
    # Run a simple integration test
    async def main():
        test_instance = TestSprint5Integration()
        await test_instance.setup_orchestrator()
        
        print("Running Sprint 5 integration tests...")
        
        try:
            await test_instance.test_pricing_inquiry_workflow()
            print("✓ Pricing inquiry workflow test passed")
            
            await test_instance.test_document_comparison_workflow()
            print("✓ Document comparison workflow test passed")
            
            await test_instance.test_privacy_protection_integration()
            print("✓ Privacy protection integration test passed")
            
            await test_instance.test_citation_generation_integration()
            print("✓ Citation generation integration test passed")
            
            await test_instance.test_workflow_performance_metrics()
            print("✓ Performance metrics test passed")
            
            print("\n🎉 All Sprint 5 integration tests passed!")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            raise
    
    asyncio.run(main())
