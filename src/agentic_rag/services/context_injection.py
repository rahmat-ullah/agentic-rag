"""
Context Injection Service

This module provides intelligent context injection capabilities for procurement queries,
including role-based context, document type context, and dynamic context selection.
"""

import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.query_processor import QueryType, QueryIntent

logger = structlog.get_logger(__name__)


class ContextType(str, Enum):
    """Types of context injection."""
    ROLE_BASED = "role_based"
    DOCUMENT_TYPE = "document_type"
    PROCESS_STAGE = "process_stage"
    DOMAIN_SPECIFIC = "domain_specific"
    TEMPORAL = "temporal"
    RELATIONSHIP = "relationship"


class UserRole(str, Enum):
    """User roles for context injection."""
    PROCUREMENT_MANAGER = "procurement_manager"
    TECHNICAL_EVALUATOR = "technical_evaluator"
    LEGAL_REVIEWER = "legal_reviewer"
    FINANCIAL_ANALYST = "financial_analyst"
    PROJECT_MANAGER = "project_manager"
    VENDOR_MANAGER = "vendor_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    GENERAL_USER = "general_user"


class ProcessStage(str, Enum):
    """Procurement process stages."""
    REQUIREMENTS_GATHERING = "requirements_gathering"
    VENDOR_IDENTIFICATION = "vendor_identification"
    RFQ_PREPARATION = "rfq_preparation"
    PROPOSAL_EVALUATION = "proposal_evaluation"
    NEGOTIATION = "negotiation"
    CONTRACT_FINALIZATION = "contract_finalization"
    IMPLEMENTATION = "implementation"
    MONITORING = "monitoring"


@dataclass
class ContextTemplate:
    """Template for context injection."""
    name: str
    context_type: ContextType
    template: str
    conditions: List[str]
    priority: int
    confidence: float


@dataclass
class InjectedContext:
    """Represents injected context with metadata."""
    content: str
    context_type: ContextType
    source_template: str
    confidence: float
    relevance_score: float


class ContextInjectionConfig(BaseModel):
    """Configuration for context injection."""
    
    # Feature toggles
    enable_role_based_context: bool = Field(True, description="Enable role-based context injection")
    enable_document_type_context: bool = Field(True, description="Enable document type context")
    enable_process_stage_context: bool = Field(True, description="Enable process stage context")
    enable_domain_context: bool = Field(True, description="Enable domain-specific context")
    enable_temporal_context: bool = Field(True, description="Enable temporal context")
    enable_relationship_context: bool = Field(True, description="Enable relationship context")
    
    # Quality settings
    min_context_confidence: float = Field(0.6, ge=0.0, le=1.0, description="Minimum context confidence")
    max_contexts_per_query: int = Field(5, ge=1, le=10, description="Maximum contexts per query")
    context_relevance_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Context relevance threshold")
    
    # User context
    user_role: Optional[UserRole] = Field(None, description="User role for context injection")
    process_stage: Optional[ProcessStage] = Field(None, description="Current process stage")
    
    # Dynamic selection
    enable_dynamic_selection: bool = Field(True, description="Enable dynamic context selection")
    adaptive_weighting: bool = Field(True, description="Enable adaptive context weighting")


class ContextInjectionService:
    """Service for intelligent context injection in procurement queries."""
    
    def __init__(self):
        self._context_templates = self._initialize_context_templates()
        self._role_contexts = self._initialize_role_contexts()
        self._document_type_contexts = self._initialize_document_type_contexts()
        self._process_stage_contexts = self._initialize_process_stage_contexts()
        self._domain_contexts = self._initialize_domain_contexts()
        
        # Context injection statistics
        self._stats = {
            "total_injections": 0,
            "role_based_injections": 0,
            "document_type_injections": 0,
            "process_stage_injections": 0,
            "domain_specific_injections": 0,
            "average_contexts_per_query": 0.0,
            "average_relevance_score": 0.0
        }
        
        logger.info("Context injection service initialized")
    
    def inject_context(
        self,
        query: str,
        query_type: QueryType,
        query_intent: QueryIntent,
        config: ContextInjectionConfig,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[InjectedContext]]:
        """
        Inject relevant context into query based on configuration and analysis.
        
        Args:
            query: Original query string
            query_type: Detected query type
            query_intent: Detected query intent
            config: Context injection configuration
            user_context: Optional user context information
            
        Returns:
            Tuple of (enhanced_query, list_of_injected_contexts)
        """
        
        logger.info(f"Injecting context for query: {query[:100]}...")
        
        # Collect potential contexts
        potential_contexts = []
        
        # 1. Role-based context injection
        if config.enable_role_based_context and config.user_role:
            role_contexts = self._get_role_based_contexts(
                query, query_type, query_intent, config.user_role
            )
            potential_contexts.extend(role_contexts)
        
        # 2. Document type context injection
        if config.enable_document_type_context:
            doc_type_contexts = self._get_document_type_contexts(
                query, query_type, query_intent
            )
            potential_contexts.extend(doc_type_contexts)
        
        # 3. Process stage context injection
        if config.enable_process_stage_context and config.process_stage:
            stage_contexts = self._get_process_stage_contexts(
                query, query_type, query_intent, config.process_stage
            )
            potential_contexts.extend(stage_contexts)
        
        # 4. Domain-specific context injection
        if config.enable_domain_context:
            domain_contexts = self._get_domain_specific_contexts(
                query, query_type, query_intent
            )
            potential_contexts.extend(domain_contexts)
        
        # 5. Temporal context injection
        if config.enable_temporal_context:
            temporal_contexts = self._get_temporal_contexts(
                query, query_type, query_intent, user_context
            )
            potential_contexts.extend(temporal_contexts)
        
        # 6. Relationship context injection
        if config.enable_relationship_context:
            relationship_contexts = self._get_relationship_contexts(
                query, query_type, query_intent, user_context
            )
            potential_contexts.extend(relationship_contexts)
        
        # Filter and select best contexts
        selected_contexts = self._select_optimal_contexts(
            potential_contexts, config
        )
        
        # Build enhanced query
        enhanced_query = self._build_enhanced_query(query, selected_contexts)
        
        # Update statistics
        self._update_statistics(selected_contexts)
        
        logger.info(
            f"Context injection complete",
            original_query_length=len(query),
            enhanced_query_length=len(enhanced_query),
            contexts_injected=len(selected_contexts)
        )
        
        return enhanced_query, selected_contexts
    
    def _get_role_based_contexts(
        self,
        query: str,
        query_type: QueryType,
        query_intent: QueryIntent,
        user_role: UserRole
    ) -> List[InjectedContext]:
        """Get role-based context injections."""
        
        contexts = []
        role_templates = self._role_contexts.get(user_role, [])
        
        for template in role_templates:
            if self._matches_conditions(query, query_type, query_intent, template.conditions):
                relevance = self._calculate_relevance(query, template.template)
                
                context = InjectedContext(
                    content=template.template,
                    context_type=ContextType.ROLE_BASED,
                    source_template=template.name,
                    confidence=template.confidence,
                    relevance_score=relevance
                )
                contexts.append(context)
        
        return contexts
    
    def _get_document_type_contexts(
        self,
        query: str,
        query_type: QueryType,
        query_intent: QueryIntent
    ) -> List[InjectedContext]:
        """Get document type specific contexts."""
        
        contexts = []
        query_lower = query.lower()
        
        # Detect document types mentioned in query
        doc_type_patterns = {
            "rfq": ["rfq", "request for quotation", "quote request"],
            "proposal": ["proposal", "offer", "bid", "quotation"],
            "contract": ["contract", "agreement", "terms"],
            "specification": ["specification", "spec", "requirements"],
            "report": ["report", "analysis", "summary"]
        }
        
        for doc_type, patterns in doc_type_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                doc_contexts = self._document_type_contexts.get(doc_type, [])
                
                for template in doc_contexts:
                    if self._matches_conditions(query, query_type, query_intent, template.conditions):
                        relevance = self._calculate_relevance(query, template.template)
                        
                        context = InjectedContext(
                            content=template.template,
                            context_type=ContextType.DOCUMENT_TYPE,
                            source_template=template.name,
                            confidence=template.confidence,
                            relevance_score=relevance
                        )
                        contexts.append(context)
        
        return contexts
    
    def _get_process_stage_contexts(
        self,
        query: str,
        query_type: QueryType,
        query_intent: QueryIntent,
        process_stage: ProcessStage
    ) -> List[InjectedContext]:
        """Get process stage specific contexts."""
        
        contexts = []
        stage_templates = self._process_stage_contexts.get(process_stage, [])
        
        for template in stage_templates:
            if self._matches_conditions(query, query_type, query_intent, template.conditions):
                relevance = self._calculate_relevance(query, template.template)
                
                context = InjectedContext(
                    content=template.template,
                    context_type=ContextType.PROCESS_STAGE,
                    source_template=template.name,
                    confidence=template.confidence,
                    relevance_score=relevance
                )
                contexts.append(context)
        
        return contexts
    
    def _get_domain_specific_contexts(
        self,
        query: str,
        query_type: QueryType,
        query_intent: QueryIntent
    ) -> List[InjectedContext]:
        """Get domain-specific procurement contexts."""
        
        contexts = []
        query_lower = query.lower()
        
        # Domain detection patterns
        domain_patterns = {
            "technical": ["software", "hardware", "system", "technology", "api", "database"],
            "financial": ["cost", "price", "budget", "payment", "financial", "expense"],
            "legal": ["contract", "compliance", "legal", "regulation", "terms"],
            "quality": ["quality", "testing", "validation", "performance", "standard"],
            "security": ["security", "privacy", "encryption", "authentication", "firewall"]
        }
        
        for domain, patterns in domain_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                domain_contexts = self._domain_contexts.get(domain, [])
                
                for template in domain_contexts:
                    if self._matches_conditions(query, query_type, query_intent, template.conditions):
                        relevance = self._calculate_relevance(query, template.template)
                        
                        context = InjectedContext(
                            content=template.template,
                            context_type=ContextType.DOMAIN_SPECIFIC,
                            source_template=template.name,
                            confidence=template.confidence,
                            relevance_score=relevance
                        )
                        contexts.append(context)
        
        return contexts

    def _get_temporal_contexts(
        self,
        query: str,
        query_type: QueryType,
        query_intent: QueryIntent,
        user_context: Optional[Dict[str, Any]]
    ) -> List[InjectedContext]:
        """Get temporal context based on time-sensitive elements."""

        contexts = []
        query_lower = query.lower()

        # Detect temporal patterns
        temporal_patterns = [
            "urgent", "asap", "deadline", "timeline", "schedule",
            "recent", "latest", "current", "new", "updated"
        ]

        if any(pattern in query_lower for pattern in temporal_patterns):
            temporal_context = "time-sensitive procurement"

            context = InjectedContext(
                content=temporal_context,
                context_type=ContextType.TEMPORAL,
                source_template="temporal_urgency",
                confidence=0.8,
                relevance_score=self._calculate_relevance(query, temporal_context)
            )
            contexts.append(context)

        return contexts

    def _get_relationship_contexts(
        self,
        query: str,
        query_type: QueryType,
        query_intent: QueryIntent,
        user_context: Optional[Dict[str, Any]]
    ) -> List[InjectedContext]:
        """Get relationship context for linked documents."""

        contexts = []
        query_lower = query.lower()

        # Detect relationship patterns
        relationship_patterns = [
            "related", "linked", "connected", "associated",
            "corresponding", "matching", "paired"
        ]

        if any(pattern in query_lower for pattern in relationship_patterns):
            relationship_context = "document relationships"

            context = InjectedContext(
                content=relationship_context,
                context_type=ContextType.RELATIONSHIP,
                source_template="document_relationships",
                confidence=0.7,
                relevance_score=self._calculate_relevance(query, relationship_context)
            )
            contexts.append(context)

        return contexts

    def _matches_conditions(
        self,
        query: str,
        query_type: QueryType,
        query_intent: QueryIntent,
        conditions: List[str]
    ) -> bool:
        """Check if query matches template conditions."""

        if not conditions:
            return True

        query_lower = query.lower()

        for condition in conditions:
            if condition.startswith("query_type:"):
                required_type = condition.split(":")[1]
                if query_type.value != required_type:
                    return False
            elif condition.startswith("intent:"):
                required_intent = condition.split(":")[1]
                if query_intent.value != required_intent:
                    return False
            elif condition.startswith("contains:"):
                required_text = condition.split(":", 1)[1]
                if required_text not in query_lower:
                    return False
            elif condition.startswith("not_contains:"):
                forbidden_text = condition.split(":", 1)[1]
                if forbidden_text in query_lower:
                    return False

        return True

    def _calculate_relevance(self, query: str, context: str) -> float:
        """Calculate relevance score between query and context."""

        query_words = set(query.lower().split())
        context_words = set(context.lower().split())

        if not query_words or not context_words:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(context_words))
        union = len(query_words.union(context_words))

        return intersection / union if union > 0 else 0.0

    def _select_optimal_contexts(
        self,
        potential_contexts: List[InjectedContext],
        config: ContextInjectionConfig
    ) -> List[InjectedContext]:
        """Select optimal contexts based on configuration."""

        # Filter by confidence and relevance thresholds
        filtered_contexts = [
            ctx for ctx in potential_contexts
            if (ctx.confidence >= config.min_context_confidence and
                ctx.relevance_score >= config.context_relevance_threshold)
        ]

        # Sort by combined score (confidence * relevance)
        filtered_contexts.sort(
            key=lambda x: x.confidence * x.relevance_score,
            reverse=True
        )

        # Limit number of contexts
        selected_contexts = filtered_contexts[:config.max_contexts_per_query]

        # Remove duplicate content
        unique_contexts = []
        seen_content = set()

        for context in selected_contexts:
            if context.content not in seen_content:
                unique_contexts.append(context)
                seen_content.add(context.content)

        return unique_contexts

    def _build_enhanced_query(
        self,
        original_query: str,
        contexts: List[InjectedContext]
    ) -> str:
        """Build enhanced query with injected contexts."""

        if not contexts:
            return original_query

        # Collect context content
        context_strings = [ctx.content for ctx in contexts]

        # Build enhanced query
        enhanced_query = f"{original_query} {' '.join(context_strings)}"

        # Clean up extra whitespace
        enhanced_query = re.sub(r'\s+', ' ', enhanced_query.strip())

        return enhanced_query

    def _update_statistics(self, contexts: List[InjectedContext]):
        """Update context injection statistics."""

        self._stats["total_injections"] += len(contexts)

        # Count by context type
        for context in contexts:
            if context.context_type == ContextType.ROLE_BASED:
                self._stats["role_based_injections"] += 1
            elif context.context_type == ContextType.DOCUMENT_TYPE:
                self._stats["document_type_injections"] += 1
            elif context.context_type == ContextType.PROCESS_STAGE:
                self._stats["process_stage_injections"] += 1
            elif context.context_type == ContextType.DOMAIN_SPECIFIC:
                self._stats["domain_specific_injections"] += 1

        # Update averages
        if contexts:
            avg_relevance = sum(ctx.relevance_score for ctx in contexts) / len(contexts)
            current_avg = self._stats["average_relevance_score"]
            total_injections = self._stats["total_injections"]

            # Update running average
            self._stats["average_relevance_score"] = (
                (current_avg * (total_injections - len(contexts))) +
                (avg_relevance * len(contexts))
            ) / total_injections

    def get_statistics(self) -> Dict[str, Any]:
        """Get context injection statistics."""
        return self._stats.copy()

    def _initialize_context_templates(self) -> List[ContextTemplate]:
        """Initialize context templates."""

        return [
            ContextTemplate(
                name="procurement_search",
                context_type=ContextType.DOMAIN_SPECIFIC,
                template="procurement document search",
                conditions=["intent:search"],
                priority=1,
                confidence=0.8
            ),
            ContextTemplate(
                name="vendor_evaluation",
                context_type=ContextType.DOMAIN_SPECIFIC,
                template="vendor evaluation criteria",
                conditions=["contains:vendor", "contains:supplier"],
                priority=2,
                confidence=0.9
            ),
            ContextTemplate(
                name="cost_analysis",
                context_type=ContextType.DOMAIN_SPECIFIC,
                template="cost analysis procurement",
                conditions=["contains:cost", "contains:price", "contains:budget"],
                priority=2,
                confidence=0.85
            )
        ]

    def _initialize_role_contexts(self) -> Dict[UserRole, List[ContextTemplate]]:
        """Initialize role-based context templates."""

        return {
            UserRole.PROCUREMENT_MANAGER: [
                ContextTemplate(
                    name="procurement_management",
                    context_type=ContextType.ROLE_BASED,
                    template="procurement management oversight",
                    conditions=[],
                    priority=1,
                    confidence=0.9
                ),
                ContextTemplate(
                    name="vendor_relationship",
                    context_type=ContextType.ROLE_BASED,
                    template="vendor relationship management",
                    conditions=["contains:vendor", "contains:supplier"],
                    priority=2,
                    confidence=0.85
                )
            ],
            UserRole.TECHNICAL_EVALUATOR: [
                ContextTemplate(
                    name="technical_evaluation",
                    context_type=ContextType.ROLE_BASED,
                    template="technical specification evaluation",
                    conditions=["contains:technical", "contains:specification"],
                    priority=1,
                    confidence=0.9
                ),
                ContextTemplate(
                    name="system_requirements",
                    context_type=ContextType.ROLE_BASED,
                    template="system requirements analysis",
                    conditions=["contains:system", "contains:requirements"],
                    priority=2,
                    confidence=0.85
                )
            ],
            UserRole.LEGAL_REVIEWER: [
                ContextTemplate(
                    name="legal_compliance",
                    context_type=ContextType.ROLE_BASED,
                    template="legal compliance review",
                    conditions=["contains:legal", "contains:compliance"],
                    priority=1,
                    confidence=0.9
                ),
                ContextTemplate(
                    name="contract_terms",
                    context_type=ContextType.ROLE_BASED,
                    template="contract terms analysis",
                    conditions=["contains:contract", "contains:terms"],
                    priority=2,
                    confidence=0.85
                )
            ],
            UserRole.FINANCIAL_ANALYST: [
                ContextTemplate(
                    name="financial_analysis",
                    context_type=ContextType.ROLE_BASED,
                    template="financial impact analysis",
                    conditions=["contains:financial", "contains:cost"],
                    priority=1,
                    confidence=0.9
                ),
                ContextTemplate(
                    name="budget_planning",
                    context_type=ContextType.ROLE_BASED,
                    template="budget planning procurement",
                    conditions=["contains:budget", "contains:planning"],
                    priority=2,
                    confidence=0.85
                )
            ]
        }

    def _initialize_document_type_contexts(self) -> Dict[str, List[ContextTemplate]]:
        """Initialize document type specific contexts."""

        return {
            "rfq": [
                ContextTemplate(
                    name="rfq_processing",
                    context_type=ContextType.DOCUMENT_TYPE,
                    template="request for quotation processing",
                    conditions=[],
                    priority=1,
                    confidence=0.9
                )
            ],
            "proposal": [
                ContextTemplate(
                    name="proposal_evaluation",
                    context_type=ContextType.DOCUMENT_TYPE,
                    template="vendor proposal evaluation",
                    conditions=[],
                    priority=1,
                    confidence=0.9
                )
            ],
            "contract": [
                ContextTemplate(
                    name="contract_review",
                    context_type=ContextType.DOCUMENT_TYPE,
                    template="contract terms review",
                    conditions=[],
                    priority=1,
                    confidence=0.9
                )
            ],
            "specification": [
                ContextTemplate(
                    name="specification_analysis",
                    context_type=ContextType.DOCUMENT_TYPE,
                    template="technical specification analysis",
                    conditions=[],
                    priority=1,
                    confidence=0.9
                )
            ]
        }

    def _initialize_process_stage_contexts(self) -> Dict[ProcessStage, List[ContextTemplate]]:
        """Initialize process stage specific contexts."""

        return {
            ProcessStage.REQUIREMENTS_GATHERING: [
                ContextTemplate(
                    name="requirements_gathering",
                    context_type=ContextType.PROCESS_STAGE,
                    template="requirements gathering phase",
                    conditions=[],
                    priority=1,
                    confidence=0.8
                )
            ],
            ProcessStage.VENDOR_IDENTIFICATION: [
                ContextTemplate(
                    name="vendor_identification",
                    context_type=ContextType.PROCESS_STAGE,
                    template="vendor identification process",
                    conditions=[],
                    priority=1,
                    confidence=0.8
                )
            ],
            ProcessStage.RFQ_PREPARATION: [
                ContextTemplate(
                    name="rfq_preparation",
                    context_type=ContextType.PROCESS_STAGE,
                    template="RFQ preparation phase",
                    conditions=[],
                    priority=1,
                    confidence=0.8
                )
            ],
            ProcessStage.PROPOSAL_EVALUATION: [
                ContextTemplate(
                    name="proposal_evaluation",
                    context_type=ContextType.PROCESS_STAGE,
                    template="proposal evaluation phase",
                    conditions=[],
                    priority=1,
                    confidence=0.8
                )
            ]
        }

    def _initialize_domain_contexts(self) -> Dict[str, List[ContextTemplate]]:
        """Initialize domain-specific contexts."""

        return {
            "technical": [
                ContextTemplate(
                    name="technical_domain",
                    context_type=ContextType.DOMAIN_SPECIFIC,
                    template="technical procurement requirements",
                    conditions=[],
                    priority=1,
                    confidence=0.8
                )
            ],
            "financial": [
                ContextTemplate(
                    name="financial_domain",
                    context_type=ContextType.DOMAIN_SPECIFIC,
                    template="financial procurement analysis",
                    conditions=[],
                    priority=1,
                    confidence=0.8
                )
            ],
            "legal": [
                ContextTemplate(
                    name="legal_domain",
                    context_type=ContextType.DOMAIN_SPECIFIC,
                    template="legal procurement compliance",
                    conditions=[],
                    priority=1,
                    confidence=0.8
                )
            ],
            "quality": [
                ContextTemplate(
                    name="quality_domain",
                    context_type=ContextType.DOMAIN_SPECIFIC,
                    template="quality assurance procurement",
                    conditions=[],
                    priority=1,
                    confidence=0.8
                )
            ],
            "security": [
                ContextTemplate(
                    name="security_domain",
                    context_type=ContextType.DOMAIN_SPECIFIC,
                    template="security requirements procurement",
                    conditions=[],
                    priority=1,
                    confidence=0.8
                )
            ]
        }


# Singleton instance
_context_injection_service: Optional[ContextInjectionService] = None


def get_context_injection_service() -> ContextInjectionService:
    """Get the singleton context injection service instance."""
    global _context_injection_service
    if _context_injection_service is None:
        _context_injection_service = ContextInjectionService()
    return _context_injection_service
