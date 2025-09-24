"""
Procurement-Optimized Prompts for LLM Reranking

This module provides specialized prompt templates and generation logic
optimized for procurement content reranking with domain-specific criteria.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.vector_search import VectorSearchResult
from agentic_rag.services.reranking_models import RerankingConfig, ScoringWeights

logger = structlog.get_logger(__name__)


class QueryType(str, Enum):
    """Types of procurement queries."""
    REQUIREMENTS = "requirements"
    SPECIFICATIONS = "specifications"
    COMPLIANCE = "compliance"
    PRICING = "pricing"
    VENDOR = "vendor"
    CONTRACT = "contract"
    TIMELINE = "timeline"
    GENERAL = "general"


class DocumentContext(str, Enum):
    """Document context types for procurement."""
    RFQ_REQUIREMENTS = "rfq_requirements"
    OFFER_RESPONSE = "offer_response"
    CONTRACT_TERMS = "contract_terms"
    SPECIFICATION_DETAILS = "specification_details"
    COMPLIANCE_DOCUMENTATION = "compliance_documentation"
    VENDOR_INFORMATION = "vendor_information"
    PRICING_INFORMATION = "pricing_information"


@dataclass
class PromptTemplate:
    """Template for generating reranking prompts."""
    base_template: str
    few_shot_examples: List[Dict[str, Any]]
    scoring_criteria: Dict[str, str]
    context_adaptations: Dict[str, str]


class ProcurementPromptGenerator:
    """Generator for procurement-optimized reranking prompts."""
    
    def __init__(self):
        self._templates = self._initialize_templates()
        self._few_shot_examples = self._initialize_few_shot_examples()
        self._query_patterns = self._initialize_query_patterns()
        
        logger.info("Procurement prompt generator initialized")
    
    def generate_reranking_prompt(
        self,
        query: str,
        results: List[VectorSearchResult],
        config: RerankingConfig,
        query_type: Optional[QueryType] = None,
        include_few_shot: bool = True
    ) -> str:
        """
        Generate a procurement-optimized reranking prompt.
        
        Args:
            query: Original search query
            results: Search results to rerank
            config: Reranking configuration
            query_type: Detected query type
            include_few_shot: Whether to include few-shot examples
            
        Returns:
            Generated prompt string
        """
        
        # Detect query type if not provided
        if query_type is None:
            query_type = self._detect_query_type(query)
        
        # Get appropriate template
        template = self._get_template_for_query_type(query_type)
        
        # Build prompt components
        prompt_parts = []
        
        # 1. System instruction
        prompt_parts.append(self._build_system_instruction(query_type, config))
        
        # 2. Query context
        prompt_parts.append(self._build_query_context(query, query_type))
        
        # 3. Scoring criteria
        prompt_parts.append(self._build_scoring_criteria(config, query_type))
        
        # 4. Few-shot examples (if enabled)
        if include_few_shot:
            examples = self._get_few_shot_examples(query_type)
            if examples:
                prompt_parts.append(self._build_few_shot_section(examples))
        
        # 5. Results to rank
        prompt_parts.append(self._build_results_section(results, query_type))
        
        # 6. Output format instruction
        prompt_parts.append(self._build_output_format_instruction())
        
        return "\n\n".join(prompt_parts)
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of procurement query."""
        query_lower = query.lower()
        
        # Check for specific patterns
        for query_type, patterns in self._query_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return QueryType(query_type)
        
        return QueryType.GENERAL
    
    def _get_template_for_query_type(self, query_type: QueryType) -> PromptTemplate:
        """Get the appropriate template for the query type."""
        return self._templates.get(query_type, self._templates[QueryType.GENERAL])
    
    def _build_system_instruction(self, query_type: QueryType, config: RerankingConfig) -> str:
        """Build the system instruction part of the prompt."""
        
        base_instruction = """You are an expert procurement analyst with deep knowledge of:
- RFQ (Request for Quotation) analysis and evaluation
- Vendor proposal assessment and comparison
- Contract terms and compliance requirements
- Technical specifications and requirements matching
- Procurement best practices and industry standards

Your task is to rank search results by their relevance and value for procurement decision-making."""
        
        # Add query-specific context
        query_specific = {
            QueryType.REQUIREMENTS: "Focus on technical requirements, specifications, and compliance criteria.",
            QueryType.SPECIFICATIONS: "Prioritize detailed technical specifications and implementation details.",
            QueryType.COMPLIANCE: "Emphasize regulatory compliance, standards adherence, and certification requirements.",
            QueryType.PRICING: "Focus on cost analysis, pricing structures, and value propositions.",
            QueryType.VENDOR: "Prioritize vendor qualifications, experience, and capability assessments.",
            QueryType.CONTRACT: "Emphasize contract terms, legal requirements, and risk factors.",
            QueryType.TIMELINE: "Focus on delivery schedules, milestones, and time-sensitive requirements."
        }
        
        if query_type in query_specific:
            base_instruction += f"\n\nFor this {query_type.value} query: {query_specific[query_type]}"
        
        return base_instruction
    
    def _build_query_context(self, query: str, query_type: QueryType) -> str:
        """Build the query context section."""
        
        context = f"""Query: "{query}"
Query Type: {query_type.value.title()}

Context: This query is seeking information related to procurement processes. Consider the following aspects:
- Business impact and strategic importance
- Technical accuracy and completeness
- Compliance and risk factors
- Cost implications and value delivery
- Implementation feasibility"""
        
        return context
    
    def _build_scoring_criteria(self, config: RerankingConfig, query_type: QueryType) -> str:
        """Build the scoring criteria section."""
        
        weights = config.scoring_weights
        
        criteria = f"""Evaluate each result on these criteria (weights in parentheses):

1. RELEVANCE ({weights.relevance*100:.0f}%): How directly does the content answer the query?
   - Direct answer to the specific question asked
   - Alignment with procurement context and objectives
   - Applicability to the business scenario

2. SPECIFICITY ({weights.specificity*100:.0f}%): How detailed and specific is the information?
   - Level of technical detail and precision
   - Concrete examples, numbers, and specifications
   - Actionable information vs. general statements

3. COMPLETENESS ({weights.completeness*100:.0f}%): How comprehensive is the coverage?
   - Coverage of all relevant aspects of the query
   - Inclusion of important context and dependencies
   - Addressing potential follow-up questions

4. AUTHORITY ({weights.authority*100:.0f}%): How credible and reliable is the source?
   - Source credibility and expertise
   - Official documentation vs. informal content
   - Recency and accuracy of information"""
        
        # Add query-specific criteria
        query_specific_criteria = {
            QueryType.REQUIREMENTS: """
Additional considerations for requirements:
- Clarity and measurability of requirements
- Alignment with business objectives
- Technical feasibility and constraints""",
            
            QueryType.COMPLIANCE: """
Additional considerations for compliance:
- Regulatory accuracy and currency
- Certification and audit requirements
- Risk assessment and mitigation""",
            
            QueryType.PRICING: """
Additional considerations for pricing:
- Cost breakdown and transparency
- Value proposition and ROI
- Market competitiveness"""
        }
        
        if query_type in query_specific_criteria:
            criteria += query_specific_criteria[query_type]
        
        return criteria
    
    def _build_few_shot_section(self, examples: List[Dict[str, Any]]) -> str:
        """Build the few-shot examples section."""
        
        section = "Here are examples of good ranking decisions:\n"
        
        for i, example in enumerate(examples, 1):
            section += f"""
Example {i}:
Query: "{example['query']}"
Top Result: {example['top_result']}
Reasoning: {example['reasoning']}
Scores: Relevance={example['scores']['relevance']}, Specificity={example['scores']['specificity']}, Completeness={example['scores']['completeness']}, Authority={example['scores']['authority']}
"""
        
        return section
    
    def _build_results_section(self, results: List[VectorSearchResult], query_type: QueryType) -> str:
        """Build the results section to be ranked."""
        
        section = "Results to rank:\n"
        
        for idx, result in enumerate(results, 1):
            # Extract relevant metadata
            doc_type = result.metadata.get("document_type", "Unknown")
            section_path = result.metadata.get("section_path", [])
            page_info = result.metadata.get("page_from", "")
            
            # Truncate content for prompt
            content_preview = result.content[:400] + "..." if len(result.content) > 400 else result.content
            
            section += f"""
Result {idx}:
Document Type: {doc_type}
Section: {' > '.join(section_path) if section_path else 'N/A'}
Page: {page_info if page_info else 'N/A'}
Content: {content_preview}
---
"""
        
        return section
    
    def _build_output_format_instruction(self) -> str:
        """Build the output format instruction."""
        
        return """Provide your ranking as a JSON object with the following format:
{
  "rankings": [
    {
      "result_id": 1,
      "relevance": 8.5,
      "specificity": 7.0,
      "completeness": 6.5,
      "authority": 8.0,
      "explanation": "Brief explanation focusing on procurement value and decision-making relevance"
    }
  ]
}

Important:
- Score each criterion from 1-10 (10 being highest)
- Provide clear, procurement-focused explanations
- Consider the business impact and practical applicability
- Rank all results, even if some are not highly relevant"""
    
    def _get_few_shot_examples(self, query_type: QueryType) -> List[Dict[str, Any]]:
        """Get few-shot examples for the query type."""
        return self._few_shot_examples.get(query_type, [])
    
    def _initialize_templates(self) -> Dict[QueryType, PromptTemplate]:
        """Initialize prompt templates for different query types."""
        # This would contain specialized templates for each query type
        # For now, using a general template
        return {query_type: PromptTemplate(
            base_template="",
            few_shot_examples=[],
            scoring_criteria={},
            context_adaptations={}
        ) for query_type in QueryType}
    
    def _initialize_few_shot_examples(self) -> Dict[QueryType, List[Dict[str, Any]]]:
        """Initialize few-shot examples for different query types."""
        
        return {
            QueryType.REQUIREMENTS: [
                {
                    "query": "data processing requirements for financial systems",
                    "top_result": "Technical specification document detailing data processing capabilities, throughput requirements, and security standards",
                    "reasoning": "Directly addresses the query with specific technical requirements and measurable criteria",
                    "scores": {"relevance": 9.0, "specificity": 8.5, "completeness": 8.0, "authority": 9.0}
                }
            ],
            QueryType.COMPLIANCE: [
                {
                    "query": "GDPR compliance requirements for data handling",
                    "top_result": "Official GDPR compliance checklist with specific data handling procedures and audit requirements",
                    "reasoning": "Authoritative source with actionable compliance steps and regulatory accuracy",
                    "scores": {"relevance": 9.5, "specificity": 9.0, "completeness": 8.5, "authority": 10.0}
                }
            ],
            QueryType.PRICING: [
                {
                    "query": "cost breakdown for cloud infrastructure services",
                    "top_result": "Detailed pricing model with per-unit costs, scaling factors, and total cost of ownership analysis",
                    "reasoning": "Comprehensive cost analysis with transparent pricing structure and business value assessment",
                    "scores": {"relevance": 9.0, "specificity": 9.5, "completeness": 8.5, "authority": 8.0}
                }
            ]
        }
    
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize query pattern matching for type detection."""
        
        return {
            "requirements": [
                "requirements", "must have", "shall", "specifications", "criteria",
                "needs", "demands", "mandatory", "essential", "required"
            ],
            "specifications": [
                "specifications", "technical specs", "details", "parameters",
                "configuration", "technical requirements", "implementation"
            ],
            "compliance": [
                "compliance", "regulatory", "standards", "certification",
                "audit", "legal", "regulation", "policy", "governance"
            ],
            "pricing": [
                "cost", "price", "pricing", "budget", "expense", "fee",
                "rate", "charges", "financial", "economic", "value"
            ],
            "vendor": [
                "vendor", "supplier", "provider", "contractor", "company",
                "organization", "firm", "partner", "service provider"
            ],
            "contract": [
                "contract", "agreement", "terms", "conditions", "legal",
                "binding", "obligations", "rights", "responsibilities"
            ],
            "timeline": [
                "timeline", "schedule", "deadline", "delivery", "milestone",
                "duration", "timeframe", "when", "by when", "completion"
            ]
        }


# Singleton instance
_procurement_prompt_generator: Optional[ProcurementPromptGenerator] = None


def get_procurement_prompt_generator() -> ProcurementPromptGenerator:
    """Get the singleton procurement prompt generator instance."""
    global _procurement_prompt_generator
    if _procurement_prompt_generator is None:
        _procurement_prompt_generator = ProcurementPromptGenerator()
    return _procurement_prompt_generator
