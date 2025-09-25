"""
Enhanced Planner Agent Implementation

This module implements the central planner agent that analyzes queries,
classifies intent, and orchestrates tool selection. It includes advanced query analysis,
intelligent tool selection, workflow planning, and execution plan generation.
"""

import asyncio
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

from .base import Agent, AgentCapability, AgentStatus, Task, TaskPriority, Context, Result
from .registry import get_agent_registry

logger = structlog.get_logger(__name__)


class QueryIntent(str, Enum):
    """Query intent classification."""

    INFORMATION_SEEKING = "information_seeking"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    PRICING_INQUIRY = "pricing_inquiry"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_ASSESSMENT = "risk_assessment"
    DOCUMENT_SEARCH = "document_search"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"


class QueryComplexity(str, Enum):
    """Query complexity levels."""

    SIMPLE = "simple"          # Single-step, direct retrieval
    MODERATE = "moderate"      # 2-3 steps, some analysis
    COMPLEX = "complex"        # Multi-step, multiple agents
    ADVANCED = "advanced"      # Complex workflows, specialized analysis


class TaskRequirement(BaseModel):
    """Individual task requirement within a query."""

    capability: AgentCapability = Field(..., description="Required agent capability")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")
    estimated_duration_ms: int = Field(default=5000, description="Estimated execution time")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    input_requirements: Dict[str, Any] = Field(default_factory=dict, description="Required inputs")
    output_expectations: Dict[str, Any] = Field(default_factory=dict, description="Expected outputs")

    class Config:
        use_enum_values = True


class QueryAnalysis(BaseModel):
    """Enhanced result of query analysis."""

    # Query understanding
    original_query: str = Field(..., description="Original user query")
    processed_query: str = Field(..., description="Processed/normalized query")
    query_type: str = Field(..., description="Type of query")
    intent: QueryIntent = Field(..., description="Classified query intent")
    complexity: QueryComplexity = Field(..., description="Query complexity level")

    # Detailed analysis
    key_entities: List[str] = Field(default_factory=list, description="Extracted key entities")
    domain_context: str = Field(default="general", description="Domain context (procurement, legal, etc.)")
    temporal_context: Optional[str] = Field(None, description="Temporal context if relevant")

    # Task breakdown
    task_requirements: List[TaskRequirement] = Field(default_factory=list, description="Required tasks")
    estimated_total_duration_ms: int = Field(default=10000, description="Total estimated duration")

    # Required capabilities (derived from task requirements)
    required_capabilities: List[AgentCapability] = Field(default_factory=list, description="Required agent capabilities")

    # Context and constraints
    context_requirements: Dict[str, Any] = Field(default_factory=dict, description="Context requirements")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Processing constraints")

    # Quality metrics
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    ambiguity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Query ambiguity level")

    # Optimization hints
    parallel_execution_possible: bool = Field(default=False, description="Can tasks run in parallel")
    caching_opportunities: List[str] = Field(default_factory=list, description="Potential caching points")

    class Config:
        use_enum_values = True


class OptimizationStrategy(BaseModel):
    """Optimization strategy for execution plan."""

    strategy_type: str = Field(..., description="Type of optimization")
    description: str = Field(..., description="Strategy description")
    estimated_improvement_ms: int = Field(default=0, description="Estimated time savings")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Strategy confidence")


class ValidationResult(BaseModel):
    """Result of execution plan validation."""

    is_valid: bool = Field(..., description="Whether plan is valid")
    issues: List[str] = Field(default_factory=list, description="Validation issues")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class ExecutionPlan(BaseModel):
    """Enhanced execution plan generated by the planner."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    query_analysis: QueryAnalysis = Field(..., description="Query analysis results")

    # Workflow definition (as dict to avoid circular imports)
    workflow: Optional[Dict[str, Any]] = Field(None, description="Generated workflow")
    workflow_id: str = Field(..., description="Workflow ID")

    # Resource requirements
    estimated_execution_time_ms: int = Field(..., description="Estimated execution time")
    required_agents: List[str] = Field(..., description="Required agent types")
    parallel_tasks: int = Field(default=1, description="Number of parallel tasks")
    max_concurrent_agents: int = Field(default=5, description="Maximum concurrent agents")

    # Quality expectations
    expected_confidence: float = Field(..., ge=0.0, le=1.0, description="Expected result confidence")
    fallback_strategies: List[str] = Field(default_factory=list, description="Fallback strategies")

    # Optimization
    optimization_strategies: List[OptimizationStrategy] = Field(default_factory=list, description="Applied optimizations")
    is_optimized: bool = Field(default=False, description="Whether plan has been optimized")

    # Validation
    validation_result: Optional[ValidationResult] = Field(None, description="Plan validation result")

    # Performance predictions
    predicted_success_rate: float = Field(default=0.9, ge=0.0, le=1.0, description="Predicted success rate")
    predicted_resource_usage: Dict[str, float] = Field(default_factory=dict, description="Predicted resource usage")

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = Field(default="planner_agent")
    version: str = Field(default="1.0", description="Plan version")

    class Config:
        use_enum_values = True


class PlannerAgent(Agent):
    """Enhanced central planner agent for query analysis and workflow orchestration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="planner_agent",
            name="Enhanced Planner Agent",
            capabilities=[
                AgentCapability.QUERY_ANALYSIS,
                AgentCapability.WORKFLOW_PLANNING,
                AgentCapability.TOOL_SELECTION
            ],
            description="Enhanced central planner for intelligent query analysis and workflow orchestration",
            config=config or {}
        )

        # Configuration
        self.max_workflow_complexity = self.config.get("max_workflow_complexity", 10)
        self.default_timeout_seconds = self.config.get("default_timeout_seconds", 300)
        self.enable_parallel_optimization = self.config.get("enable_parallel_optimization", True)
        self.enable_plan_optimization = self.config.get("enable_plan_optimization", True)
        self.enable_plan_validation = self.config.get("enable_plan_validation", True)
        self.max_optimization_iterations = self.config.get("max_optimization_iterations", 3)

        # Analysis components
        self._query_patterns = self._initialize_query_patterns()
        self._workflow_templates = self._initialize_workflow_templates()
        self._intent_classifiers = self._initialize_intent_classifiers()
        self._entity_extractors = self._initialize_entity_extractors()
        self._optimization_strategies = self._initialize_optimization_strategies()

        # Performance tracking
        self._analysis_cache: Dict[str, QueryAnalysis] = {}
        self._plan_cache: Dict[str, ExecutionPlan] = {}
        self._performance_history: List[Dict[str, Any]] = []
    
    async def initialize(self) -> None:
        """Initialize the planner agent."""
        self._logger.info("Initializing planner agent")
        self.status = AgentStatus.READY
        self._logger.info("Planner agent initialized")
    
    async def execute(self, task: Task, context: Context) -> Result:
        """Execute a planning task."""
        start_time = datetime.now(timezone.utc)
        
        try:
            self.status = AgentStatus.BUSY
            
            if task.capability_required == AgentCapability.QUERY_ANALYSIS:
                result_data = await self._analyze_query(task, context)
            elif task.capability_required == AgentCapability.WORKFLOW_PLANNING:
                result_data = await self._plan_workflow(task, context)
            elif task.capability_required == AgentCapability.TOOL_SELECTION:
                result_data = await self._select_tools(task, context)
            else:
                raise ValueError(f"Unsupported capability: {task.capability_required}")
            
            execution_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            
            result = Result(
                task_id=task.id,
                agent_id=self.agent_id,
                success=True,
                data=result_data,
                execution_time_ms=execution_time,
                confidence_score=result_data.get("confidence_score", 0.8)
            )
            
            self._update_performance_metrics(execution_time, True)
            return result
            
        except Exception as e:
            execution_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            
            self._logger.error(f"Planning task failed: {e}")
            
            result = Result(
                task_id=task.id,
                agent_id=self.agent_id,
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
            
            self._update_performance_metrics(execution_time, False)
            return result
            
        finally:
            self.status = AgentStatus.READY
    
    async def shutdown(self) -> None:
        """Shutdown the planner agent."""
        self._logger.info("Shutting down planner agent")
        self.status = AgentStatus.SHUTTING_DOWN
    
    async def _analyze_query(self, task: Task, context: Context) -> Dict[str, Any]:
        """Enhanced query analysis with sophisticated intent classification and task breakdown."""
        query = task.input_data.get("query", context.original_query)

        # Check cache first
        cache_key = f"{hash(query)}_{context.tenant_id}"
        if cache_key in self._analysis_cache:
            cached_analysis = self._analysis_cache[cache_key]
            return {
                "analysis": cached_analysis.dict(),
                "confidence_score": cached_analysis.confidence_score,
                "cached": True
            }

        # Enhanced query preprocessing
        processed_query = self._preprocess_query(query)

        # Intent classification
        intent = self._classify_intent_advanced(query)

        # Entity extraction
        entities = self._extract_entities(query)

        # Domain context detection
        domain_context = self._detect_domain_context(query, entities)

        # Complexity analysis
        complexity = self._analyze_complexity_advanced(query, intent, entities)

        # Task requirement analysis
        task_requirements = self._analyze_task_requirements(query, intent, complexity, entities)

        # Capability derivation
        required_capabilities = self._derive_capabilities_from_tasks(task_requirements)

        # Temporal context extraction
        temporal_context = self._extract_temporal_context(query)

        # Optimization analysis
        parallel_possible = self._analyze_parallel_potential(task_requirements)
        caching_opportunities = self._identify_caching_opportunities(query, task_requirements)

        # Create comprehensive analysis
        analysis = QueryAnalysis(
            original_query=query,
            processed_query=processed_query,
            query_type=self._classify_query_type(query),
            intent=intent,
            complexity=complexity,
            key_entities=entities,
            domain_context=domain_context,
            temporal_context=temporal_context,
            task_requirements=task_requirements,
            estimated_total_duration_ms=sum(req.estimated_duration_ms for req in task_requirements),
            required_capabilities=required_capabilities,
            context_requirements=self._extract_context_requirements(query, context),
            constraints=self._extract_constraints(query, context),
            confidence_score=self._calculate_analysis_confidence(query, intent, entities),
            ambiguity_score=self._calculate_ambiguity_score(query),
            parallel_execution_possible=parallel_possible,
            caching_opportunities=caching_opportunities
        )

        # Cache the analysis
        self._analysis_cache[cache_key] = analysis

        return {
            "analysis": analysis.dict(),
            "confidence_score": analysis.confidence_score,
            "cached": False
        }
    
    async def _plan_workflow(self, task: Task, context: Context) -> Dict[str, Any]:
        """Enhanced workflow planning with optimization and validation."""
        analysis_data = task.input_data.get("analysis")
        if not analysis_data:
            raise ValueError("Query analysis required for workflow planning")

        analysis = QueryAnalysis(**analysis_data)

        # Check plan cache
        plan_cache_key = f"{hash(str(analysis_data))}_{context.tenant_id}"
        if plan_cache_key in self._plan_cache:
            cached_plan = self._plan_cache[plan_cache_key]
            return {
                "plan": cached_plan.dict(),
                "confidence_score": cached_plan.expected_confidence,
                "cached": True
            }

        # Generate initial workflow
        workflow_data, workflow_id = self._generate_workflow_advanced(analysis, context)

        # Create initial execution plan
        plan = ExecutionPlan(
            query_analysis=analysis,
            workflow=workflow_data,
            workflow_id=workflow_id,
            estimated_execution_time_ms=self._estimate_execution_time(workflow_data),
            required_agents=self._identify_required_agents(workflow_data),
            parallel_tasks=self._count_parallel_tasks(workflow_data),
            max_concurrent_agents=self._calculate_max_concurrent_agents(analysis),
            expected_confidence=self._calculate_expected_confidence(analysis, workflow_data),
            fallback_strategies=self._generate_fallback_strategies(analysis),
            predicted_success_rate=self._predict_success_rate(analysis, workflow_data),
            predicted_resource_usage=self._predict_resource_usage(workflow_data)
        )

        # Apply optimizations if enabled
        if self.enable_plan_optimization:
            plan = await self._optimize_execution_plan(plan, analysis)

        # Validate plan if enabled
        if self.enable_plan_validation:
            validation_result = await self._validate_execution_plan(plan)
            plan.validation_result = validation_result

            if not validation_result.is_valid:
                self._logger.warning(f"Plan validation failed: {validation_result.issues}")
                # Attempt to fix issues
                plan = await self._fix_plan_issues(plan, validation_result)

        # Cache the plan
        self._plan_cache[plan_cache_key] = plan

        return {
            "plan": plan.dict(),
            "confidence_score": plan.expected_confidence,
            "cached": False,
            "optimized": plan.is_optimized,
            "validated": plan.validation_result is not None
        }
    
    async def _select_tools(self, task: Task, context: Context) -> Dict[str, Any]:
        """Select appropriate tools/agents for task execution."""
        required_capability = task.input_data.get("capability")
        if not required_capability:
            raise ValueError("Capability required for tool selection")
        
        # Get agent registry
        registry = await get_agent_registry()
        
        # Find suitable agents
        agents = registry.find_agents_by_capability(
            AgentCapability(required_capability),
            status_filter=[AgentStatus.READY, AgentStatus.BUSY]
        )
        
        # Select best agent
        selected_agent = None
        if agents:
            selected_agent = agents[0]  # Best available agent
        
        return {
            "selected_agent": selected_agent.agent.agent_id if selected_agent else None,
            "available_agents": [a.agent.agent_id for a in agents],
            "confidence_score": 0.9 if selected_agent else 0.0
        }
    
    def _preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing."""
        # Basic cleaning
        processed = query.strip().lower()

        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)

        # Expand common abbreviations
        abbreviations = {
            'rfq': 'request for quotation',
            'rfp': 'request for proposal',
            'po': 'purchase order',
            'sow': 'statement of work',
            'nda': 'non-disclosure agreement'
        }

        for abbr, expansion in abbreviations.items():
            processed = re.sub(rf'\b{abbr}\b', expansion, processed)

        return processed

    def _classify_intent_advanced(self, query: str) -> QueryIntent:
        """Advanced intent classification using pattern matching and keywords."""
        query_lower = query.lower()

        # Intent patterns with weights
        intent_patterns = {
            QueryIntent.COMPARISON: [
                "compare", "difference", "vs", "versus", "better", "best",
                "contrast", "similar", "different", "alternative"
            ],
            QueryIntent.PRICING_INQUIRY: [
                "price", "cost", "pricing", "budget", "expensive", "cheap",
                "quote", "quotation", "estimate", "fee", "rate"
            ],
            QueryIntent.ANALYSIS: [
                "analyze", "analysis", "examine", "evaluate", "assess",
                "review", "study", "investigate", "breakdown"
            ],
            QueryIntent.EXTRACTION: [
                "extract", "get", "obtain", "retrieve", "find", "locate",
                "identify", "list", "show", "display"
            ],
            QueryIntent.SUMMARIZATION: [
                "summarize", "summary", "overview", "brief", "outline",
                "recap", "digest", "abstract", "synopsis"
            ],
            QueryIntent.COMPLIANCE_CHECK: [
                "compliance", "compliant", "regulation", "standard", "requirement",
                "policy", "rule", "guideline", "audit"
            ],
            QueryIntent.RISK_ASSESSMENT: [
                "risk", "risky", "danger", "threat", "vulnerability",
                "security", "safety", "hazard", "concern"
            ],
            QueryIntent.RELATIONSHIP_DISCOVERY: [
                "relationship", "related", "connection", "link", "associated",
                "connected", "tied", "bound", "correlation"
            ]
        }

        # Score each intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score

        # Return highest scoring intent or default
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return QueryIntent.INFORMATION_SEEKING

    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from the query."""
        entities = []

        # Common procurement entities
        entity_patterns = {
            'document_types': r'\b(rfq|rfp|contract|agreement|proposal|quote|invoice|po|purchase order)\b',
            'monetary': r'\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:dollars?|usd|cents?)\b',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            'companies': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Co)\b',
            'products': r'\b(?:software|hardware|service|product|solution|system|platform|tool)\b'
        }

        for entity_type, pattern in entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend([f"{entity_type}:{match}" for match in matches])

        return entities

    def _detect_domain_context(self, query: str, entities: List[str]) -> str:
        """Detect the domain context of the query."""
        query_lower = query.lower()

        domain_keywords = {
            'procurement': ['procurement', 'purchase', 'vendor', 'supplier', 'contract', 'rfq', 'rfp'],
            'legal': ['legal', 'contract', 'agreement', 'compliance', 'regulation', 'law'],
            'financial': ['financial', 'budget', 'cost', 'price', 'payment', 'invoice'],
            'technical': ['technical', 'specification', 'requirement', 'system', 'software'],
            'hr': ['hr', 'human resources', 'employee', 'staff', 'personnel']
        }

        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return "general"

    def _analyze_complexity_advanced(self, query: str, intent: QueryIntent, entities: List[str]) -> QueryComplexity:
        """Advanced complexity analysis."""
        complexity_score = 0

        # Base complexity from query length
        word_count = len(query.split())
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1

        # Intent-based complexity
        intent_complexity = {
            QueryIntent.INFORMATION_SEEKING: 0,
            QueryIntent.EXTRACTION: 1,
            QueryIntent.SUMMARIZATION: 1,
            QueryIntent.COMPARISON: 2,
            QueryIntent.ANALYSIS: 2,
            QueryIntent.PRICING_INQUIRY: 2,
            QueryIntent.COMPLIANCE_CHECK: 3,
            QueryIntent.RISK_ASSESSMENT: 3,
            QueryIntent.RELATIONSHIP_DISCOVERY: 3
        }
        complexity_score += intent_complexity.get(intent, 1)

        # Entity-based complexity
        complexity_score += min(len(entities) // 2, 2)

        # Complex keywords
        complex_keywords = ['analyze', 'compare', 'evaluate', 'assess', 'relationship', 'correlation']
        complexity_score += sum(1 for keyword in complex_keywords if keyword in query.lower())

        # Map to complexity levels
        if complexity_score <= 2:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 4:
            return QueryComplexity.MODERATE
        elif complexity_score <= 6:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.ADVANCED

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
            return "comparison"
        elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
            return "summarization"
        elif any(word in query_lower for word in ["price", "cost", "pricing", "budget"]):
            return "pricing_analysis"
        elif any(word in query_lower for word in ["find", "search", "look for", "retrieve"]):
            return "retrieval"
        else:
            return "general"
    
    def _analyze_task_requirements(self, query: str, intent: QueryIntent, complexity: QueryComplexity, entities: List[str]) -> List[TaskRequirement]:
        """Analyze and break down task requirements."""
        requirements = []

        # Always need document retrieval for most queries
        if intent != QueryIntent.INFORMATION_SEEKING or complexity != QueryComplexity.SIMPLE:
            requirements.append(TaskRequirement(
                capability=AgentCapability.DOCUMENT_RETRIEVAL,
                priority=TaskPriority.HIGH,
                estimated_duration_ms=3000,
                input_requirements={"query": query},
                output_expectations={"documents": "list", "relevance_scores": "float"}
            ))

        # Intent-specific requirements
        if intent == QueryIntent.COMPARISON:
            requirements.append(TaskRequirement(
                capability=AgentCapability.DOCUMENT_COMPARISON,
                priority=TaskPriority.HIGH,
                estimated_duration_ms=5000,
                dependencies=[AgentCapability.DOCUMENT_RETRIEVAL.value],
                input_requirements={"documents": "list"},
                output_expectations={"comparison_matrix": "dict", "differences": "list"}
            ))

        elif intent == QueryIntent.PRICING_INQUIRY:
            requirements.append(TaskRequirement(
                capability=AgentCapability.PRICING_ANALYSIS,
                priority=TaskPriority.HIGH,
                estimated_duration_ms=4000,
                dependencies=[AgentCapability.DOCUMENT_RETRIEVAL.value],
                input_requirements={"documents": "list", "pricing_context": "dict"},
                output_expectations={"pricing_data": "dict", "analysis": "str"}
            ))

        elif intent == QueryIntent.ANALYSIS:
            requirements.append(TaskRequirement(
                capability=AgentCapability.DOCUMENT_COMPARISON,
                priority=TaskPriority.NORMAL,
                estimated_duration_ms=6000,
                dependencies=[AgentCapability.DOCUMENT_RETRIEVAL.value],
                input_requirements={"documents": "list", "analysis_type": "str"},
                output_expectations={"analysis_results": "dict", "insights": "list"}
            ))

        elif intent == QueryIntent.SUMMARIZATION:
            requirements.append(TaskRequirement(
                capability=AgentCapability.SUMMARIZATION,
                priority=TaskPriority.NORMAL,
                estimated_duration_ms=4000,
                dependencies=[AgentCapability.DOCUMENT_RETRIEVAL.value],
                input_requirements={"documents": "list", "summary_length": "str"},
                output_expectations={"summary": "str", "key_points": "list"}
            ))

        elif intent == QueryIntent.COMPLIANCE_CHECK:
            requirements.append(TaskRequirement(
                capability=AgentCapability.COMPLIANCE_CHECKING,
                priority=TaskPriority.HIGH,
                estimated_duration_ms=7000,
                dependencies=[AgentCapability.DOCUMENT_RETRIEVAL.value],
                input_requirements={"documents": "list", "compliance_standards": "list"},
                output_expectations={"compliance_status": "dict", "violations": "list"}
            ))

        elif intent == QueryIntent.RISK_ASSESSMENT:
            requirements.append(TaskRequirement(
                capability=AgentCapability.RISK_ASSESSMENT,
                priority=TaskPriority.HIGH,
                estimated_duration_ms=8000,
                dependencies=[AgentCapability.DOCUMENT_RETRIEVAL.value],
                input_requirements={"documents": "list", "risk_categories": "list"},
                output_expectations={"risk_analysis": "dict", "recommendations": "list"}
            ))

        # Always need answer synthesis for final result
        requirements.append(TaskRequirement(
            capability=AgentCapability.ANSWER_SYNTHESIS,
            priority=TaskPriority.HIGH,
            estimated_duration_ms=3000,
            dependencies=[req.capability.value if hasattr(req.capability, 'value') else str(req.capability) for req in requirements],
            input_requirements={"query": query, "analysis_results": "dict"},
            output_expectations={"final_answer": "str", "citations": "list", "confidence": "float"}
        ))

        # Add redaction if sensitive content might be involved
        if any(keyword in query.lower() for keyword in ['confidential', 'private', 'sensitive', 'personal']):
            requirements.append(TaskRequirement(
                capability=AgentCapability.CONTENT_REDACTION,
                priority=TaskPriority.NORMAL,
                estimated_duration_ms=2000,
                dependencies=[AgentCapability.ANSWER_SYNTHESIS.value],
                input_requirements={"content": "str", "redaction_rules": "list"},
                output_expectations={"redacted_content": "str", "redaction_log": "list"}
            ))

        return requirements

    def _derive_capabilities_from_tasks(self, task_requirements: List[TaskRequirement]) -> List[AgentCapability]:
        """Derive required capabilities from task requirements."""
        return [req.capability for req in task_requirements]

    def _extract_temporal_context(self, query: str) -> Optional[str]:
        """Extract temporal context from query."""
        temporal_patterns = [
            r'\b(last|past|previous)\s+(week|month|year|quarter)\b',
            r'\b(this|current)\s+(week|month|year|quarter)\b',
            r'\b(next|upcoming|future)\s+(week|month|year|quarter)\b',
            r'\b\d{4}\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
        ]

        for pattern in temporal_patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(0)

        return None

    def _analyze_parallel_potential(self, task_requirements: List[TaskRequirement]) -> bool:
        """Analyze if tasks can be executed in parallel."""
        # Check if any tasks have no dependencies
        independent_tasks = [req for req in task_requirements if not req.dependencies]
        return len(independent_tasks) > 1

    def _identify_caching_opportunities(self, query: str, task_requirements: List[TaskRequirement]) -> List[str]:
        """Identify potential caching opportunities."""
        opportunities = []

        # Document retrieval results can often be cached
        if any(req.capability == AgentCapability.DOCUMENT_RETRIEVAL for req in task_requirements):
            opportunities.append("document_retrieval_results")

        # Pricing analysis for similar queries
        if any(req.capability == AgentCapability.PRICING_ANALYSIS for req in task_requirements):
            opportunities.append("pricing_analysis_cache")

        # Entity extraction results
        if len(query.split()) > 10:
            opportunities.append("entity_extraction_cache")

        return opportunities

    def _extract_context_requirements(self, query: str, context: Context) -> Dict[str, Any]:
        """Extract context requirements from query and context."""
        requirements = {
            "tenant_id": str(context.tenant_id),
            "user_id": str(context.user_id),
            "domain": self._detect_domain_context(query, []),
            "security_level": "standard"
        }

        # Check for high-security requirements
        if any(keyword in query.lower() for keyword in ['confidential', 'classified', 'restricted']):
            requirements["security_level"] = "high"

        return requirements

    def _extract_constraints(self, query: str, context: Context) -> Dict[str, Any]:
        """Extract processing constraints."""
        constraints = {
            "max_execution_time_ms": self.default_timeout_seconds * 1000,
            "max_document_count": 100,
            "require_citations": True
        }

        # Extract time constraints from query
        time_patterns = [
            r'\b(urgent|asap|immediately|quickly)\b',
            r'\b(within|in)\s+(\d+)\s+(minutes?|hours?|days?)\b'
        ]

        for pattern in time_patterns:
            if re.search(pattern, query.lower()):
                constraints["priority"] = "high"
                constraints["max_execution_time_ms"] = min(constraints["max_execution_time_ms"], 60000)  # 1 minute max
                break

        return constraints

    def _calculate_analysis_confidence(self, query: str, intent: QueryIntent, entities: List[str]) -> float:
        """Calculate confidence in the analysis."""
        confidence = 0.7  # Base confidence

        # Boost confidence for clear intent indicators
        if intent != QueryIntent.INFORMATION_SEEKING:
            confidence += 0.1

        # Boost confidence for entity recognition
        if entities:
            confidence += min(len(entities) * 0.05, 0.15)

        # Reduce confidence for very short or very long queries
        word_count = len(query.split())
        if word_count < 3 or word_count > 50:
            confidence -= 0.1

        return min(max(confidence, 0.0), 1.0)

    def _calculate_ambiguity_score(self, query: str) -> float:
        """Calculate query ambiguity score."""
        ambiguity = 0.0

        # Check for ambiguous words
        ambiguous_words = ['it', 'this', 'that', 'they', 'them', 'something', 'anything']
        ambiguity += sum(0.1 for word in ambiguous_words if word in query.lower().split())

        # Check for multiple question words
        question_words = ['what', 'where', 'when', 'why', 'how', 'which', 'who']
        question_count = sum(1 for word in question_words if word in query.lower())
        if question_count > 1:
            ambiguity += 0.2

        # Check for vague terms
        vague_terms = ['some', 'any', 'maybe', 'possibly', 'perhaps', 'might']
        ambiguity += sum(0.05 for term in vague_terms if term in query.lower())

        return min(ambiguity, 1.0)
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        # Simple heuristic based on query length and keywords
        complexity_keywords = ["compare", "analyze", "summarize", "extract", "calculate"]
        
        base_complexity = min(len(query.split()) / 20.0, 0.5)
        keyword_complexity = sum(0.1 for word in complexity_keywords if word in query.lower())
        
        return min(base_complexity + keyword_complexity, 1.0)
    
    def _estimate_steps(self, query: str) -> int:
        """Estimate number of steps required."""
        complexity = self._calculate_complexity(query)
        
        if complexity < 0.3:
            return 1
        elif complexity < 0.6:
            return 2
        else:
            return 3
    
    def _identify_required_capabilities(self, query: str) -> List[AgentCapability]:
        """Identify required agent capabilities."""
        capabilities = []
        query_lower = query.lower()
        
        # Always need retrieval for document-based queries
        capabilities.append(AgentCapability.DOCUMENT_RETRIEVAL)
        
        if any(word in query_lower for word in ["price", "cost", "pricing"]):
            capabilities.append(AgentCapability.PRICING_ANALYSIS)
        
        if any(word in query_lower for word in ["compare", "comparison"]):
            capabilities.append(AgentCapability.DOCUMENT_COMPARISON)
        
        if any(word in query_lower for word in ["summarize", "summary"]):
            capabilities.append(AgentCapability.SUMMARIZATION)
        
        # Always need synthesis for final answer
        capabilities.append(AgentCapability.ANSWER_SYNTHESIS)
        
        return capabilities
    
    async def _optimize_execution_plan(self, plan: ExecutionPlan, analysis: QueryAnalysis) -> ExecutionPlan:
        """Optimize execution plan for better performance."""
        optimizations = []

        # Parallel execution optimization
        if analysis.parallel_execution_possible and not plan.is_optimized:
            parallel_optimization = OptimizationStrategy(
                strategy_type="parallel_execution",
                description="Execute independent tasks in parallel",
                estimated_improvement_ms=plan.estimated_execution_time_ms // 3,
                confidence=0.8
            )
            optimizations.append(parallel_optimization)

        # Caching optimization
        if analysis.caching_opportunities:
            cache_optimization = OptimizationStrategy(
                strategy_type="result_caching",
                description="Cache intermediate results for reuse",
                estimated_improvement_ms=2000,
                confidence=0.7
            )
            optimizations.append(cache_optimization)

        # Resource optimization
        if plan.parallel_tasks > 3:
            resource_optimization = OptimizationStrategy(
                strategy_type="resource_optimization",
                description="Optimize resource allocation and scheduling",
                estimated_improvement_ms=1000,
                confidence=0.6
            )
            optimizations.append(resource_optimization)

        # Apply optimizations
        if optimizations:
            total_improvement = sum(opt.estimated_improvement_ms for opt in optimizations)
            plan.estimated_execution_time_ms = max(
                plan.estimated_execution_time_ms - total_improvement,
                plan.estimated_execution_time_ms // 2  # Don't reduce by more than 50%
            )
            plan.optimization_strategies = optimizations
            plan.is_optimized = True

        return plan

    async def _validate_execution_plan(self, plan: ExecutionPlan) -> ValidationResult:
        """Validate execution plan for correctness and feasibility."""
        issues = []
        warnings = []
        suggestions = []

        # Check if required agents are available
        registry = await get_agent_registry()
        for agent_type in plan.required_agents:
            try:
                capability = AgentCapability(agent_type)
                available_agents = registry.find_agents_by_capability(capability)
                if not available_agents:
                    issues.append(f"No agents available for capability: {agent_type}")
            except ValueError:
                issues.append(f"Invalid agent capability: {agent_type}")

        # Check execution time constraints
        if plan.estimated_execution_time_ms > self.default_timeout_seconds * 1000:
            warnings.append(f"Estimated execution time ({plan.estimated_execution_time_ms}ms) exceeds default timeout")

        # Check resource requirements
        if plan.max_concurrent_agents > 10:
            warnings.append(f"High concurrent agent count ({plan.max_concurrent_agents}) may impact performance")

        # Check workflow complexity
        if plan.workflow and len(plan.workflow.get("steps", [])) > self.max_workflow_complexity:
            issues.append(f"Workflow complexity exceeds maximum ({self.max_workflow_complexity} steps)")

        # Suggest improvements
        if plan.predicted_success_rate < 0.8:
            suggestions.append("Consider adding fallback strategies to improve success rate")

        if not plan.is_optimized:
            suggestions.append("Plan could benefit from optimization")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions
        )

    async def _fix_plan_issues(self, plan: ExecutionPlan, validation_result: ValidationResult) -> ExecutionPlan:
        """Attempt to fix issues in execution plan."""
        # For now, just log the issues and return the plan
        # In a full implementation, this would attempt to resolve issues
        self._logger.warning(f"Plan issues detected but not automatically fixable: {validation_result.issues}")
        return plan

    def _generate_workflow_advanced(self, analysis: QueryAnalysis, context: Context) -> tuple[Dict[str, Any], str]:
        """Generate advanced workflow based on analysis."""
        return self._generate_workflow(analysis, context)

    def _count_parallel_tasks(self, workflow_data: Dict[str, Any]) -> int:
        """Count tasks that can run in parallel."""
        steps = workflow_data.get("steps", [])
        independent_steps = [step for step in steps if not step.get("depends_on")]
        return len(independent_steps)

    def _calculate_max_concurrent_agents(self, analysis: QueryAnalysis) -> int:
        """Calculate maximum concurrent agents needed."""
        base_agents = len(analysis.task_requirements)

        if analysis.complexity == QueryComplexity.ADVANCED:
            return min(base_agents + 2, 8)
        elif analysis.complexity == QueryComplexity.COMPLEX:
            return min(base_agents + 1, 6)
        else:
            return min(base_agents, 4)

    def _calculate_expected_confidence(self, analysis: QueryAnalysis, workflow_data: Dict[str, Any]) -> float:
        """Calculate expected confidence of execution results."""
        base_confidence = analysis.confidence_score

        # Adjust based on complexity
        if analysis.complexity == QueryComplexity.SIMPLE:
            base_confidence += 0.1
        elif analysis.complexity == QueryComplexity.ADVANCED:
            base_confidence -= 0.1

        # Adjust based on ambiguity
        base_confidence -= analysis.ambiguity_score * 0.2

        return min(max(base_confidence, 0.0), 1.0)

    def _generate_fallback_strategies(self, analysis: QueryAnalysis) -> List[str]:
        """Generate fallback strategies for the execution plan."""
        strategies = []

        # Always have basic fallback
        strategies.append("fallback_to_simple_search")

        # Complexity-based fallbacks
        if analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.ADVANCED]:
            strategies.append("reduce_complexity_on_failure")
            strategies.append("partial_result_aggregation")

        # Intent-based fallbacks
        if analysis.intent == QueryIntent.COMPARISON:
            strategies.append("fallback_to_individual_analysis")
        elif analysis.intent == QueryIntent.ANALYSIS:
            strategies.append("fallback_to_summarization")

        return strategies

    def _predict_success_rate(self, analysis: QueryAnalysis, workflow_data: Dict[str, Any]) -> float:
        """Predict success rate of execution plan."""
        base_rate = 0.9

        # Adjust based on complexity
        complexity_adjustments = {
            QueryComplexity.SIMPLE: 0.05,
            QueryComplexity.MODERATE: 0.0,
            QueryComplexity.COMPLEX: -0.1,
            QueryComplexity.ADVANCED: -0.2
        }
        base_rate += complexity_adjustments.get(analysis.complexity, 0.0)

        # Adjust based on ambiguity
        base_rate -= analysis.ambiguity_score * 0.3

        # Adjust based on workflow complexity
        step_count = len(workflow_data.get("steps", []))
        if step_count > 5:
            base_rate -= (step_count - 5) * 0.05

        return min(max(base_rate, 0.0), 1.0)

    def _predict_resource_usage(self, workflow_data: Dict[str, Any]) -> Dict[str, float]:
        """Predict resource usage for execution plan."""
        steps = workflow_data.get("steps", [])

        return {
            "cpu_usage": min(len(steps) * 0.1, 1.0),
            "memory_usage": min(len(steps) * 0.05, 0.8),
            "network_usage": min(len(steps) * 0.08, 0.9),
            "storage_usage": min(len(steps) * 0.02, 0.5)
        }

    def _generate_workflow(self, analysis: QueryAnalysis, context: Context) -> tuple[Dict[str, Any], str]:
        """Generate workflow based on analysis."""
        steps = []
        
        # Step 1: Document retrieval
        retrieval_task = Task(
            name="document_retrieval",
            description="Retrieve relevant documents",
            capability_required=AgentCapability.DOCUMENT_RETRIEVAL,
            priority=TaskPriority.HIGH,
            input_data={"query": analysis.processed_query}
        )
        
        retrieval_step_id = str(uuid4())
        retrieval_step = {
            "id": retrieval_step_id,
            "name": "retrieve_documents",
            "type": "task",
            "task": retrieval_task.dict()
        }
        steps.append(retrieval_step)
        
        # Step 2: Specialized analysis (if needed)
        if AgentCapability.PRICING_ANALYSIS in analysis.required_capabilities:
            pricing_task = Task(
                name="pricing_analysis",
                description="Analyze pricing information",
                capability_required=AgentCapability.PRICING_ANALYSIS,
                input_data={"documents": "{{retrieve_documents.result}}"}
            )
            
            pricing_step_id = str(uuid4())
            pricing_step = {
                "id": pricing_step_id,
                "name": "analyze_pricing",
                "type": "task",
                "task": pricing_task.dict(),
                "depends_on": [retrieval_step_id]
            }
            steps.append(pricing_step)
        
        # Step 3: Answer synthesis
        synthesis_task = Task(
            name="answer_synthesis",
            description="Synthesize final answer",
            capability_required=AgentCapability.ANSWER_SYNTHESIS,
            input_data={
                "query": analysis.original_query,
                "documents": "{{retrieve_documents.result}}"
            }
        )
        
        synthesis_step_id = str(uuid4())
        synthesis_step = {
            "id": synthesis_step_id,
            "name": "synthesize_answer",
            "type": "task",
            "task": synthesis_task.dict(),
            "depends_on": [step["id"] for step in steps]
        }
        steps.append(synthesis_step)
        
        workflow_id = str(uuid4())
        workflow_data = {
            "id": workflow_id,
            "name": f"query_processing_{analysis.query_type}",
            "description": f"Process {analysis.query_type} query",
            "steps": steps,
            "timeout_seconds": self.default_timeout_seconds
        }

        return workflow_data, workflow_id
    
    def _estimate_execution_time(self, workflow: Dict[str, Any]) -> int:
        """Estimate workflow execution time."""
        # Simple estimation based on number of steps
        base_time_per_step = 5000  # 5 seconds per step
        return len(workflow.get("steps", [])) * base_time_per_step

    def _identify_required_agents(self, workflow: Dict[str, Any]) -> List[str]:
        """Identify required agent types for workflow."""
        agent_types = set()

        for step_data in workflow.get("steps", []):
            if step_data.get("task") and step_data["task"].get("capability_required"):
                agent_types.add(step_data["task"]["capability_required"])

        return list(agent_types)
    
    def _initialize_query_patterns(self) -> Dict[str, Any]:
        """Initialize enhanced query patterns for analysis."""
        return {
            "comparison_patterns": ["compare", "vs", "versus", "difference", "contrast", "better", "best"],
            "pricing_patterns": ["price", "cost", "pricing", "budget", "expensive", "quote", "estimate"],
            "summarization_patterns": ["summarize", "summary", "overview", "brief", "outline", "recap"],
            "analysis_patterns": ["analyze", "analysis", "examine", "evaluate", "assess", "review"],
            "extraction_patterns": ["extract", "get", "obtain", "retrieve", "find", "locate"],
            "compliance_patterns": ["compliance", "compliant", "regulation", "standard", "requirement"],
            "risk_patterns": ["risk", "risky", "danger", "threat", "vulnerability", "security"],
            "temporal_patterns": ["last", "past", "previous", "current", "next", "upcoming", "future"]
        }

    def _initialize_workflow_templates(self) -> Dict[str, Any]:
        """Initialize enhanced workflow templates."""
        return {
            "simple_retrieval": {
                "steps": ["retrieve", "synthesize"],
                "estimated_time_ms": 8000,
                "complexity": "simple"
            },
            "complex_analysis": {
                "steps": ["retrieve", "analyze", "synthesize"],
                "estimated_time_ms": 15000,
                "complexity": "moderate"
            },
            "comparison": {
                "steps": ["retrieve", "compare", "synthesize"],
                "estimated_time_ms": 12000,
                "complexity": "moderate"
            },
            "pricing_analysis": {
                "steps": ["retrieve", "pricing_analysis", "synthesize"],
                "estimated_time_ms": 14000,
                "complexity": "moderate"
            },
            "compliance_check": {
                "steps": ["retrieve", "compliance_check", "synthesize", "redact"],
                "estimated_time_ms": 20000,
                "complexity": "complex"
            },
            "risk_assessment": {
                "steps": ["retrieve", "risk_assessment", "synthesize"],
                "estimated_time_ms": 18000,
                "complexity": "complex"
            },
            "multi_step_analysis": {
                "steps": ["retrieve", "analyze", "compare", "synthesize"],
                "estimated_time_ms": 25000,
                "complexity": "advanced"
            }
        }

    def _initialize_intent_classifiers(self) -> Dict[str, Any]:
        """Initialize intent classification patterns."""
        return {
            "high_confidence_patterns": {
                QueryIntent.COMPARISON: ["compare", "vs", "versus", "difference between"],
                QueryIntent.PRICING_INQUIRY: ["how much", "price of", "cost of", "pricing for"],
                QueryIntent.SUMMARIZATION: ["summarize", "give me a summary", "overview of"],
                QueryIntent.ANALYSIS: ["analyze", "analysis of", "examine"],
                QueryIntent.EXTRACTION: ["extract", "get me", "find", "retrieve"]
            },
            "medium_confidence_patterns": {
                QueryIntent.COMPLIANCE_CHECK: ["compliant", "meets requirements", "standards"],
                QueryIntent.RISK_ASSESSMENT: ["risks", "dangerous", "safe", "security"],
                QueryIntent.RELATIONSHIP_DISCOVERY: ["related", "connected", "linked"]
            }
        }

    def _initialize_entity_extractors(self) -> Dict[str, Any]:
        """Initialize entity extraction patterns."""
        return {
            "document_types": {
                "pattern": r'\b(rfq|rfp|contract|agreement|proposal|quote|invoice|po|purchase order|sow|nda)\b',
                "confidence": 0.9
            },
            "monetary_values": {
                "pattern": r'\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:dollars?|usd|cents?)\b',
                "confidence": 0.95
            },
            "dates": {
                "pattern": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
                "confidence": 0.85
            },
            "companies": {
                "pattern": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Co)\b',
                "confidence": 0.8
            },
            "products": {
                "pattern": r'\b(?:software|hardware|service|product|solution|system|platform|tool)\b',
                "confidence": 0.7
            }
        }

    def _initialize_optimization_strategies(self) -> Dict[str, Any]:
        """Initialize optimization strategies."""
        return {
            "parallel_execution": {
                "description": "Execute independent tasks in parallel",
                "conditions": ["multiple_independent_tasks", "sufficient_resources"],
                "estimated_improvement": 0.3
            },
            "result_caching": {
                "description": "Cache intermediate results for reuse",
                "conditions": ["repeated_operations", "cacheable_results"],
                "estimated_improvement": 0.2
            },
            "resource_optimization": {
                "description": "Optimize resource allocation",
                "conditions": ["high_resource_usage", "multiple_agents"],
                "estimated_improvement": 0.15
            },
            "workflow_simplification": {
                "description": "Simplify workflow when possible",
                "conditions": ["over_complex_workflow", "simple_query"],
                "estimated_improvement": 0.25
            }
        }


# Global planner agent instance
_planner_agent: Optional[PlannerAgent] = None


async def get_planner_agent(config: Optional[Dict[str, Any]] = None) -> PlannerAgent:
    """Get or create the global planner agent instance."""
    global _planner_agent
    
    if _planner_agent is None:
        _planner_agent = PlannerAgent(config)
        await _planner_agent.initialize()
    
    return _planner_agent


async def close_planner_agent() -> None:
    """Close the global planner agent instance."""
    global _planner_agent
    
    if _planner_agent:
        await _planner_agent.shutdown()
        _planner_agent = None
