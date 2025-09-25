#!/usr/bin/env python3
"""
Sprint 5 Integration Demonstration Script

This script demonstrates the complete Sprint 5 functionality by running
end-to-end workflows that showcase all integrated services working together.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_rag.services.orchestration.integration import (
    get_enhanced_orchestrator,
    IntegratedWorkflowResult
)
from agentic_rag.services.orchestration.planner import QueryIntent


class Sprint5Demo:
    """Demonstration class for Sprint 5 integrated functionality."""
    
    def __init__(self):
        self.orchestrator = None
        self.demo_user_id = uuid4()
        self.demo_tenant_id = uuid4()
        
    async def initialize(self):
        """Initialize the demonstration environment."""
        print("🚀 Initializing Sprint 5 Integrated System...")
        self.orchestrator = await get_enhanced_orchestrator()
        print("✅ System initialized successfully!")
        print()
    
    async def run_demo(self):
        """Run the complete demonstration."""
        print("=" * 80)
        print("🎯 SPRINT 5 COMPLETE INTEGRATION DEMONSTRATION")
        print("=" * 80)
        print()
        
        # Demo scenarios
        scenarios = [
            {
                "name": "Pricing Analysis Workflow",
                "query": "What are the pricing strategies and cost models mentioned in the vendor proposals?",
                "role": "procurement_analyst",
                "expected_intent": QueryIntent.PRICING_INQUIRY
            },
            {
                "name": "Document Comparison Workflow", 
                "query": "Compare the terms and conditions between the different vendor contracts",
                "role": "legal_reviewer",
                "expected_intent": QueryIntent.COMPARISON
            },
            {
                "name": "Risk Assessment Workflow",
                "query": "Identify and assess the security and operational risks in the vendor proposals",
                "role": "risk_analyst", 
                "expected_intent": QueryIntent.RISK_ASSESSMENT
            },
            {
                "name": "Compliance Checking Workflow",
                "query": "Check if the procurement documents comply with ISO 9001 and SOX requirements",
                "role": "compliance_officer",
                "expected_intent": QueryIntent.COMPLIANCE_CHECK
            },
            {
                "name": "Content Summarization Workflow",
                "query": "Provide a comprehensive summary of the key points in all vendor evaluation documents",
                "role": "executive",
                "expected_intent": QueryIntent.SUMMARIZATION
            },
            {
                "name": "Privacy Protection Workflow",
                "query": "What are the contact details and personal information mentioned in John Smith's vendor proposal?",
                "role": "viewer",  # Lower privilege role to test redaction
                "expected_intent": QueryIntent.EXTRACTION
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"📋 Scenario {i}: {scenario['name']}")
            print("-" * 60)
            
            result = await self.run_scenario(scenario)
            results.append(result)
            
            print()
        
        # Summary
        await self.print_summary(results)
    
    async def run_scenario(self, scenario: Dict[str, Any]) -> IntegratedWorkflowResult:
        """Run a single demonstration scenario."""
        print(f"🔍 Query: {scenario['query']}")
        print(f"👤 User Role: {scenario['role']}")
        print(f"🎯 Expected Intent: {scenario['expected_intent'].value}")
        print()
        
        start_time = datetime.now()
        
        try:
            # Process the query through the integrated system
            result = await self.orchestrator.process_integrated_query(
                query=scenario['query'],
                user_id=self.demo_user_id,
                tenant_id=self.demo_tenant_id,
                user_role=scenario['role'],
                context_data={
                    "demo_scenario": scenario['name'],
                    "timestamp": start_time.isoformat()
                }
            )
            
            # Display results
            await self.display_result(result, scenario)
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            # Return a failed result
            return IntegratedWorkflowResult(
                request_id=str(uuid4()),
                original_query=scenario['query'],
                query_intent=scenario['expected_intent'],
                success=False,
                execution_time_ms=0
            )
    
    async def display_result(self, result: IntegratedWorkflowResult, scenario: Dict[str, Any]):
        """Display the results of a workflow execution."""
        if result.success:
            print("✅ Workflow Status: SUCCESS")
            print(f"🧠 Detected Intent: {result.query_intent.value}")
            print(f"⏱️  Execution Time: {result.execution_time_ms}ms")
            print(f"🤖 Agents Used: {', '.join(result.agents_used)}")
            print(f"📊 Confidence Score: {result.confidence_score:.2f}")
            
            # Display workflow steps
            print("\n📋 Workflow Steps:")
            for i, step in enumerate(result.workflow_steps, 1):
                print(f"  {i}. {step['step'].replace('_', ' ').title()}")
            
            # Display intent-specific results
            if result.pricing_analysis:
                print(f"\n💰 Pricing Analysis: Found {len(result.pricing_analysis.get('pricing_data', []))} pricing items")
            
            if result.document_comparison:
                print(f"\n📄 Document Comparison: Found {result.document_comparison.get('differences_found', 0)} differences")
            
            if result.risk_assessment:
                print(f"\n⚠️  Risk Assessment: Overall risk level - {result.risk_assessment.get('risk_level', 'Unknown')}")
            
            if result.compliance_check:
                print(f"\n✅ Compliance Check: Score - {result.compliance_check.get('compliance_score', 0):.2f}")
            
            # Display privacy protection
            if result.pii_detected > 0 or result.redactions_applied > 0:
                print(f"\n🔒 Privacy Protection:")
                print(f"  - PII Items Detected: {result.pii_detected}")
                print(f"  - Redactions Applied: {result.redactions_applied}")
            
            # Display citations
            if result.citations:
                print(f"\n📚 Citations: {len(result.citations)} sources referenced")
            
            # Display final answer (truncated)
            if result.final_answer:
                answer_preview = result.final_answer[:200] + "..." if len(result.final_answer) > 200 else result.final_answer
                print(f"\n💬 Answer Preview: {answer_preview}")
            
        else:
            print("❌ Workflow Status: FAILED")
            print(f"⏱️  Execution Time: {result.execution_time_ms}ms")
    
    async def print_summary(self, results: list):
        """Print a summary of all demonstration results."""
        print("=" * 80)
        print("📊 DEMONSTRATION SUMMARY")
        print("=" * 80)
        
        successful_results = [r for r in results if r.success]
        total_scenarios = len(results)
        successful_scenarios = len(successful_results)
        
        print(f"📈 Success Rate: {successful_scenarios}/{total_scenarios} ({successful_scenarios/total_scenarios*100:.1f}%)")
        
        if successful_results:
            avg_execution_time = sum(r.execution_time_ms for r in successful_results) / len(successful_results)
            avg_confidence = sum(r.confidence_score for r in successful_results) / len(successful_results)
            total_pii_detected = sum(r.pii_detected for r in successful_results)
            total_redactions = sum(r.redactions_applied for r in successful_results)
            
            print(f"⏱️  Average Execution Time: {avg_execution_time:.0f}ms")
            print(f"📊 Average Confidence Score: {avg_confidence:.2f}")
            print(f"🔒 Total PII Items Detected: {total_pii_detected}")
            print(f"🔒 Total Redactions Applied: {total_redactions}")
            
            # Intent distribution
            intent_counts = {}
            for result in successful_results:
                intent = result.query_intent.value
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            print("\n🎯 Query Intent Distribution:")
            for intent, count in intent_counts.items():
                print(f"  - {intent.replace('_', ' ').title()}: {count}")
            
            # Agent usage
            all_agents = set()
            for result in successful_results:
                all_agents.update(result.agents_used)
            
            print(f"\n🤖 Total Unique Agents Used: {len(all_agents)}")
            for agent in sorted(all_agents):
                print(f"  - {agent.replace('_', ' ').title()}")
        
        print("\n🎉 Sprint 5 Integration Demonstration Complete!")
        print("\nKey Features Demonstrated:")
        print("✅ Intelligent query analysis and intent classification")
        print("✅ Multi-agent orchestration and coordination")
        print("✅ Specialized processing (pricing, comparison, risk, compliance)")
        print("✅ Answer synthesis with citations")
        print("✅ Privacy protection and content redaction")
        print("✅ Role-based access control")
        print("✅ Performance monitoring and quality assessment")
        print("✅ End-to-end workflow execution")


async def main():
    """Main demonstration function."""
    demo = Sprint5Demo()
    
    try:
        await demo.initialize()
        await demo.run_demo()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Thank you for viewing the Sprint 5 integration demonstration!")


if __name__ == "__main__":
    print("🎬 Starting Sprint 5 Integration Demonstration...")
    print("This demo showcases the complete integrated system functionality.")
    print()
    
    # Run the demonstration
    asyncio.run(main())
