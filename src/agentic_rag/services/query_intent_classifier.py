"""
Query Intent Classification Service

This module provides intelligent query intent classification for procurement scenarios
with machine learning-based classification, confidence scoring, and processing pipelines.
"""

import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ProcurementIntent(str, Enum):
    """Procurement-specific intent categories."""
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    VENDOR_EVALUATION = "vendor_evaluation"
    COST_ANALYSIS = "cost_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    CONTRACT_REVIEW = "contract_review"
    TECHNICAL_ASSESSMENT = "technical_assessment"
    TIMELINE_INQUIRY = "timeline_inquiry"
    COMPARISON_REQUEST = "comparison_request"
    STATUS_CHECK = "status_check"
    GENERAL_SEARCH = "general_search"


class ConfidenceLevel(str, Enum):
    """Confidence levels for intent classification."""
    VERY_HIGH = "very_high"  # 0.9+
    HIGH = "high"           # 0.8-0.89
    MEDIUM = "medium"       # 0.6-0.79
    LOW = "low"            # 0.4-0.59
    VERY_LOW = "very_low"  # <0.4


@dataclass
class IntentFeature:
    """Feature for intent classification."""
    name: str
    pattern: str
    weight: float
    intent: ProcurementIntent
    confidence_boost: float


@dataclass
class ClassificationResult:
    """Result of intent classification."""
    intent: ProcurementIntent
    confidence: float
    confidence_level: ConfidenceLevel
    features_matched: List[str]
    alternative_intents: List[Tuple[ProcurementIntent, float]]
    explanation: str


class QueryIntentConfig(BaseModel):
    """Configuration for query intent classification."""
    
    # Classification settings
    min_confidence_threshold: float = Field(0.4, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_alternative_intents: int = Field(3, ge=1, le=5, description="Maximum alternative intents to return")
    feature_weight_multiplier: float = Field(1.0, ge=0.1, le=2.0, description="Feature weight multiplier")
    
    # Processing settings
    enable_pattern_matching: bool = Field(True, description="Enable pattern-based classification")
    enable_keyword_analysis: bool = Field(True, description="Enable keyword analysis")
    enable_context_analysis: bool = Field(True, description="Enable context analysis")
    enable_semantic_analysis: bool = Field(True, description="Enable semantic analysis")
    
    # Quality settings
    require_minimum_features: int = Field(1, ge=1, le=5, description="Minimum features required for classification")
    enable_confidence_calibration: bool = Field(True, description="Enable confidence calibration")
    enable_explanation_generation: bool = Field(True, description="Enable explanation generation")


class QueryIntentClassifier:
    """Service for classifying query intent in procurement scenarios."""
    
    def __init__(self):
        self._intent_features = self._initialize_intent_features()
        self._keyword_patterns = self._initialize_keyword_patterns()
        self._context_patterns = self._initialize_context_patterns()
        self._semantic_patterns = self._initialize_semantic_patterns()
        
        # Classification statistics
        self._stats = {
            "total_classifications": 0,
            "intent_distribution": {intent.value: 0 for intent in ProcurementIntent},
            "confidence_distribution": {level.value: 0 for level in ConfidenceLevel},
            "average_confidence": 0.0,
            "feature_usage": {},
            "classification_accuracy": 0.0
        }
        
        logger.info("Query intent classifier initialized")
    
    def classify_intent(
        self,
        query: str,
        config: QueryIntentConfig,
        context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        """
        Classify the intent of a procurement query.
        
        Args:
            query: Query string to classify
            config: Classification configuration
            context: Optional context information
            
        Returns:
            ClassificationResult with intent and confidence
        """
        
        logger.info(f"Classifying intent for query: {query[:100]}...")
        
        self._stats["total_classifications"] += 1
        
        # Collect classification scores from different methods
        classification_scores = {}
        
        # 1. Pattern-based classification
        if config.enable_pattern_matching:
            pattern_scores = self._classify_by_patterns(query, config)
            self._merge_scores(classification_scores, pattern_scores, weight=0.4)
        
        # 2. Keyword analysis
        if config.enable_keyword_analysis:
            keyword_scores = self._classify_by_keywords(query, config)
            self._merge_scores(classification_scores, keyword_scores, weight=0.3)
        
        # 3. Context analysis
        if config.enable_context_analysis:
            context_scores = self._classify_by_context(query, config, context)
            self._merge_scores(classification_scores, context_scores, weight=0.2)
        
        # 4. Semantic analysis
        if config.enable_semantic_analysis:
            semantic_scores = self._classify_by_semantics(query, config)
            self._merge_scores(classification_scores, semantic_scores, weight=0.1)
        
        # Determine best intent and confidence
        result = self._determine_final_classification(
            query, classification_scores, config
        )
        
        # Update statistics
        self._update_statistics(result)
        
        logger.info(
            f"Intent classification complete",
            intent=result.intent.value,
            confidence=result.confidence,
            features_matched=len(result.features_matched)
        )
        
        return result
    
    def _classify_by_patterns(
        self,
        query: str,
        config: QueryIntentConfig
    ) -> Dict[ProcurementIntent, float]:
        """Classify intent using pattern matching."""
        
        scores = {intent: 0.0 for intent in ProcurementIntent}
        query_lower = query.lower()
        
        for feature in self._intent_features:
            if re.search(feature.pattern, query_lower):
                current_score = scores[feature.intent]
                boost = feature.weight * feature.confidence_boost * config.feature_weight_multiplier
                scores[feature.intent] = min(current_score + boost, 1.0)
                
                # Track feature usage
                if feature.name not in self._stats["feature_usage"]:
                    self._stats["feature_usage"][feature.name] = 0
                self._stats["feature_usage"][feature.name] += 1
        
        return scores
    
    def _classify_by_keywords(
        self,
        query: str,
        config: QueryIntentConfig
    ) -> Dict[ProcurementIntent, float]:
        """Classify intent using keyword analysis."""
        
        scores = {intent: 0.0 for intent in ProcurementIntent}
        query_words = set(query.lower().split())
        
        for intent, keywords in self._keyword_patterns.items():
            matches = len(query_words.intersection(set(keywords)))
            if matches > 0:
                # Score based on keyword density
                score = min(matches / len(keywords), 1.0) * 0.8
                scores[intent] = max(scores[intent], score)
        
        return scores
    
    def _classify_by_context(
        self,
        query: str,
        config: QueryIntentConfig,
        context: Optional[Dict[str, Any]]
    ) -> Dict[ProcurementIntent, float]:
        """Classify intent using context analysis."""
        
        scores = {intent: 0.0 for intent in ProcurementIntent}
        
        if not context:
            return scores
        
        # Analyze context for intent hints
        user_role = context.get("user_role", "")
        document_type = context.get("document_type", "")
        process_stage = context.get("process_stage", "")
        
        # Role-based intent hints
        role_intent_mapping = {
            "procurement_manager": ProcurementIntent.VENDOR_EVALUATION,
            "technical_evaluator": ProcurementIntent.TECHNICAL_ASSESSMENT,
            "legal_reviewer": ProcurementIntent.CONTRACT_REVIEW,
            "financial_analyst": ProcurementIntent.COST_ANALYSIS
        }
        
        if user_role in role_intent_mapping:
            scores[role_intent_mapping[user_role]] += 0.3
        
        # Document type hints
        if "rfq" in document_type.lower():
            scores[ProcurementIntent.REQUIREMENTS_ANALYSIS] += 0.2
        elif "proposal" in document_type.lower():
            scores[ProcurementIntent.VENDOR_EVALUATION] += 0.2
        elif "contract" in document_type.lower():
            scores[ProcurementIntent.CONTRACT_REVIEW] += 0.2
        
        return scores
    
    def _classify_by_semantics(
        self,
        query: str,
        config: QueryIntentConfig
    ) -> Dict[ProcurementIntent, float]:
        """Classify intent using semantic analysis."""
        
        scores = {intent: 0.0 for intent in ProcurementIntent}
        query_lower = query.lower()
        
        # Semantic pattern matching
        for intent, patterns in self._semantic_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                weight = pattern_info["weight"]
                
                if re.search(pattern, query_lower):
                    scores[intent] = max(scores[intent], weight)
        
        return scores
    
    def _merge_scores(
        self,
        target_scores: Dict[ProcurementIntent, float],
        source_scores: Dict[ProcurementIntent, float],
        weight: float
    ):
        """Merge classification scores with weighting."""
        
        for intent, score in source_scores.items():
            if intent not in target_scores:
                target_scores[intent] = 0.0
            target_scores[intent] += score * weight
    
    def _determine_final_classification(
        self,
        query: str,
        scores: Dict[ProcurementIntent, float],
        config: QueryIntentConfig
    ) -> ClassificationResult:
        """Determine final classification result."""
        
        # Sort intents by score
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get best intent and confidence
        best_intent, best_confidence = sorted_intents[0] if sorted_intents else (ProcurementIntent.GENERAL_SEARCH, 0.0)
        
        # Apply confidence calibration
        if config.enable_confidence_calibration:
            best_confidence = self._calibrate_confidence(best_confidence, query)
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(best_confidence)
        
        # Get alternative intents
        alternative_intents = [
            (intent, score) for intent, score in sorted_intents[1:config.max_alternative_intents + 1]
            if score >= config.min_confidence_threshold
        ]
        
        # Generate explanation
        explanation = self._generate_explanation(
            query, best_intent, best_confidence, config
        ) if config.enable_explanation_generation else ""
        
        # Collect matched features
        features_matched = self._get_matched_features(query, best_intent)
        
        return ClassificationResult(
            intent=best_intent,
            confidence=best_confidence,
            confidence_level=confidence_level,
            features_matched=features_matched,
            alternative_intents=alternative_intents,
            explanation=explanation
        )
    
    def _calibrate_confidence(self, confidence: float, query: str) -> float:
        """Calibrate confidence based on query characteristics."""
        
        # Adjust confidence based on query length
        query_length = len(query.split())
        if query_length < 3:
            confidence *= 0.8  # Lower confidence for very short queries
        elif query_length > 20:
            confidence *= 0.9  # Slightly lower confidence for very long queries
        
        # Adjust confidence based on specificity
        specific_terms = ["rfq", "proposal", "contract", "vendor", "cost", "compliance"]
        specific_count = sum(1 for term in specific_terms if term in query.lower())
        if specific_count > 0:
            confidence = min(confidence * (1 + specific_count * 0.1), 1.0)
        
        return confidence
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level."""
        
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_explanation(
        self,
        query: str,
        intent: ProcurementIntent,
        confidence: float,
        config: QueryIntentConfig
    ) -> str:
        """Generate explanation for the classification."""
        
        explanations = {
            ProcurementIntent.REQUIREMENTS_ANALYSIS: "Query focuses on analyzing or defining requirements",
            ProcurementIntent.VENDOR_EVALUATION: "Query involves evaluating or comparing vendors/suppliers",
            ProcurementIntent.COST_ANALYSIS: "Query is related to cost, pricing, or budget analysis",
            ProcurementIntent.COMPLIANCE_CHECK: "Query involves compliance, regulations, or standards",
            ProcurementIntent.CONTRACT_REVIEW: "Query is about contract terms, agreements, or legal aspects",
            ProcurementIntent.TECHNICAL_ASSESSMENT: "Query focuses on technical specifications or capabilities",
            ProcurementIntent.TIMELINE_INQUIRY: "Query is about timelines, deadlines, or scheduling",
            ProcurementIntent.COMPARISON_REQUEST: "Query involves comparing options or alternatives",
            ProcurementIntent.STATUS_CHECK: "Query is asking about status or progress",
            ProcurementIntent.GENERAL_SEARCH: "General search query without specific procurement intent"
        }
        
        base_explanation = explanations.get(intent, "Unknown intent")
        confidence_text = f"with {confidence:.1%} confidence"
        
        return f"{base_explanation} {confidence_text}"
    
    def _get_matched_features(self, query: str, intent: ProcurementIntent) -> List[str]:
        """Get list of features that matched for the given intent."""
        
        matched_features = []
        query_lower = query.lower()
        
        for feature in self._intent_features:
            if feature.intent == intent and re.search(feature.pattern, query_lower):
                matched_features.append(feature.name)
        
        return matched_features

    def _update_statistics(self, result: ClassificationResult):
        """Update classification statistics."""

        # Update intent distribution
        self._stats["intent_distribution"][result.intent.value] += 1

        # Update confidence distribution
        self._stats["confidence_distribution"][result.confidence_level.value] += 1

        # Update average confidence
        total_classifications = self._stats["total_classifications"]
        current_avg = self._stats["average_confidence"]
        new_avg = ((current_avg * (total_classifications - 1)) + result.confidence) / total_classifications
        self._stats["average_confidence"] = new_avg

    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        return self._stats.copy()

    def _initialize_intent_features(self) -> List[IntentFeature]:
        """Initialize intent classification features."""

        return [
            # Requirements Analysis
            IntentFeature(
                name="requirements_keywords",
                pattern=r"\b(requirements?|specifications?|needs?|criteria)\b",
                weight=0.8,
                intent=ProcurementIntent.REQUIREMENTS_ANALYSIS,
                confidence_boost=0.9
            ),
            IntentFeature(
                name="requirements_phrases",
                pattern=r"(what are the|define the|specify the|list the)",
                weight=0.7,
                intent=ProcurementIntent.REQUIREMENTS_ANALYSIS,
                confidence_boost=0.8
            ),

            # Vendor Evaluation
            IntentFeature(
                name="vendor_keywords",
                pattern=r"\b(vendor|supplier|provider|contractor)\b",
                weight=0.8,
                intent=ProcurementIntent.VENDOR_EVALUATION,
                confidence_boost=0.9
            ),
            IntentFeature(
                name="evaluation_phrases",
                pattern=r"(compare vendors|evaluate suppliers|vendor comparison)",
                weight=0.9,
                intent=ProcurementIntent.VENDOR_EVALUATION,
                confidence_boost=0.95
            ),

            # Cost Analysis
            IntentFeature(
                name="cost_keywords",
                pattern=r"\b(cost|price|budget|expense|financial)\b",
                weight=0.8,
                intent=ProcurementIntent.COST_ANALYSIS,
                confidence_boost=0.9
            ),
            IntentFeature(
                name="pricing_phrases",
                pattern=r"(how much|cost analysis|pricing structure|budget for)",
                weight=0.9,
                intent=ProcurementIntent.COST_ANALYSIS,
                confidence_boost=0.95
            ),

            # Compliance Check
            IntentFeature(
                name="compliance_keywords",
                pattern=r"\b(compliance|regulation|standard|certification)\b",
                weight=0.8,
                intent=ProcurementIntent.COMPLIANCE_CHECK,
                confidence_boost=0.9
            ),
            IntentFeature(
                name="compliance_phrases",
                pattern=r"(meets standards|compliant with|regulatory requirements)",
                weight=0.9,
                intent=ProcurementIntent.COMPLIANCE_CHECK,
                confidence_boost=0.95
            ),

            # Contract Review
            IntentFeature(
                name="contract_keywords",
                pattern=r"\b(contract|agreement|terms|conditions)\b",
                weight=0.8,
                intent=ProcurementIntent.CONTRACT_REVIEW,
                confidence_boost=0.9
            ),
            IntentFeature(
                name="legal_phrases",
                pattern=r"(contract terms|legal requirements|agreement details)",
                weight=0.9,
                intent=ProcurementIntent.CONTRACT_REVIEW,
                confidence_boost=0.95
            ),

            # Technical Assessment
            IntentFeature(
                name="technical_keywords",
                pattern=r"\b(technical|technology|system|architecture)\b",
                weight=0.7,
                intent=ProcurementIntent.TECHNICAL_ASSESSMENT,
                confidence_boost=0.8
            ),
            IntentFeature(
                name="technical_phrases",
                pattern=r"(technical specifications|system requirements|technology stack)",
                weight=0.9,
                intent=ProcurementIntent.TECHNICAL_ASSESSMENT,
                confidence_boost=0.95
            ),

            # Timeline Inquiry
            IntentFeature(
                name="timeline_keywords",
                pattern=r"\b(timeline|deadline|schedule|delivery)\b",
                weight=0.8,
                intent=ProcurementIntent.TIMELINE_INQUIRY,
                confidence_boost=0.9
            ),
            IntentFeature(
                name="time_phrases",
                pattern=r"(when will|delivery date|project timeline|completion date)",
                weight=0.9,
                intent=ProcurementIntent.TIMELINE_INQUIRY,
                confidence_boost=0.95
            ),

            # Comparison Request
            IntentFeature(
                name="comparison_keywords",
                pattern=r"\b(compare|comparison|versus|vs|difference)\b",
                weight=0.8,
                intent=ProcurementIntent.COMPARISON_REQUEST,
                confidence_boost=0.9
            ),
            IntentFeature(
                name="comparison_phrases",
                pattern=r"(compare between|which is better|pros and cons)",
                weight=0.9,
                intent=ProcurementIntent.COMPARISON_REQUEST,
                confidence_boost=0.95
            ),

            # Status Check
            IntentFeature(
                name="status_keywords",
                pattern=r"\b(status|progress|update|current)\b",
                weight=0.7,
                intent=ProcurementIntent.STATUS_CHECK,
                confidence_boost=0.8
            ),
            IntentFeature(
                name="status_phrases",
                pattern=r"(what is the status|current progress|latest update)",
                weight=0.9,
                intent=ProcurementIntent.STATUS_CHECK,
                confidence_boost=0.95
            )
        ]

    def _initialize_keyword_patterns(self) -> Dict[ProcurementIntent, List[str]]:
        """Initialize keyword patterns for each intent."""

        return {
            ProcurementIntent.REQUIREMENTS_ANALYSIS: [
                "requirements", "specifications", "needs", "criteria", "define", "specify"
            ],
            ProcurementIntent.VENDOR_EVALUATION: [
                "vendor", "supplier", "provider", "contractor", "evaluate", "assess"
            ],
            ProcurementIntent.COST_ANALYSIS: [
                "cost", "price", "budget", "expense", "financial", "pricing", "money"
            ],
            ProcurementIntent.COMPLIANCE_CHECK: [
                "compliance", "regulation", "standard", "certification", "audit", "policy"
            ],
            ProcurementIntent.CONTRACT_REVIEW: [
                "contract", "agreement", "terms", "conditions", "legal", "clause"
            ],
            ProcurementIntent.TECHNICAL_ASSESSMENT: [
                "technical", "technology", "system", "architecture", "implementation"
            ],
            ProcurementIntent.TIMELINE_INQUIRY: [
                "timeline", "deadline", "schedule", "delivery", "when", "date"
            ],
            ProcurementIntent.COMPARISON_REQUEST: [
                "compare", "comparison", "versus", "difference", "better", "alternative"
            ],
            ProcurementIntent.STATUS_CHECK: [
                "status", "progress", "update", "current", "latest", "state"
            ],
            ProcurementIntent.GENERAL_SEARCH: [
                "search", "find", "look", "show", "list", "information"
            ]
        }

    def _initialize_context_patterns(self) -> Dict[str, Any]:
        """Initialize context-based classification patterns."""

        return {
            "user_roles": {
                "procurement_manager": [
                    ProcurementIntent.VENDOR_EVALUATION,
                    ProcurementIntent.COST_ANALYSIS
                ],
                "technical_evaluator": [
                    ProcurementIntent.TECHNICAL_ASSESSMENT,
                    ProcurementIntent.REQUIREMENTS_ANALYSIS
                ],
                "legal_reviewer": [
                    ProcurementIntent.CONTRACT_REVIEW,
                    ProcurementIntent.COMPLIANCE_CHECK
                ],
                "financial_analyst": [
                    ProcurementIntent.COST_ANALYSIS,
                    ProcurementIntent.COMPARISON_REQUEST
                ]
            },
            "document_types": {
                "rfq": ProcurementIntent.REQUIREMENTS_ANALYSIS,
                "proposal": ProcurementIntent.VENDOR_EVALUATION,
                "contract": ProcurementIntent.CONTRACT_REVIEW,
                "specification": ProcurementIntent.TECHNICAL_ASSESSMENT
            }
        }

    def _initialize_semantic_patterns(self) -> Dict[ProcurementIntent, List[Dict[str, Any]]]:
        """Initialize semantic patterns for intent classification."""

        return {
            ProcurementIntent.REQUIREMENTS_ANALYSIS: [
                {
                    "pattern": r"(what|which|how).*(requirements?|specifications?|needs?)",
                    "weight": 0.9
                },
                {
                    "pattern": r"(define|specify|describe).*(requirements?|criteria)",
                    "weight": 0.85
                }
            ],
            ProcurementIntent.VENDOR_EVALUATION: [
                {
                    "pattern": r"(evaluate|assess|compare).*(vendor|supplier|provider)",
                    "weight": 0.9
                },
                {
                    "pattern": r"(which|what).*(vendor|supplier).*(best|better|recommended)",
                    "weight": 0.85
                }
            ],
            ProcurementIntent.COST_ANALYSIS: [
                {
                    "pattern": r"(how much|what.*cost|price|budget)",
                    "weight": 0.9
                },
                {
                    "pattern": r"(cost.*analysis|pricing.*structure|budget.*breakdown)",
                    "weight": 0.85
                }
            ],
            ProcurementIntent.COMPLIANCE_CHECK: [
                {
                    "pattern": r"(compliant|compliance|meets.*standards?|regulatory)",
                    "weight": 0.9
                },
                {
                    "pattern": r"(audit|certification|standards?|regulations?)",
                    "weight": 0.8
                }
            ],
            ProcurementIntent.CONTRACT_REVIEW: [
                {
                    "pattern": r"(contract.*terms|agreement.*details|legal.*requirements)",
                    "weight": 0.9
                },
                {
                    "pattern": r"(terms.*conditions|contract.*clauses|legal.*obligations)",
                    "weight": 0.85
                }
            ],
            ProcurementIntent.TECHNICAL_ASSESSMENT: [
                {
                    "pattern": r"(technical.*specifications|system.*requirements|technology.*stack)",
                    "weight": 0.9
                },
                {
                    "pattern": r"(architecture|implementation|integration|technical.*details)",
                    "weight": 0.8
                }
            ],
            ProcurementIntent.TIMELINE_INQUIRY: [
                {
                    "pattern": r"(when.*will|delivery.*date|timeline|schedule|deadline)",
                    "weight": 0.9
                },
                {
                    "pattern": r"(completion.*date|project.*timeline|delivery.*schedule)",
                    "weight": 0.85
                }
            ],
            ProcurementIntent.COMPARISON_REQUEST: [
                {
                    "pattern": r"(compare.*between|which.*better|pros.*cons|advantages)",
                    "weight": 0.9
                },
                {
                    "pattern": r"(versus|vs|difference.*between|alternative.*options)",
                    "weight": 0.85
                }
            ],
            ProcurementIntent.STATUS_CHECK: [
                {
                    "pattern": r"(what.*status|current.*progress|latest.*update)",
                    "weight": 0.9
                },
                {
                    "pattern": r"(status.*of|progress.*on|update.*about)",
                    "weight": 0.8
                }
            ]
        }


# Singleton instance
_query_intent_classifier: Optional[QueryIntentClassifier] = None


def get_query_intent_classifier() -> QueryIntentClassifier:
    """Get the singleton query intent classifier instance."""
    global _query_intent_classifier
    if _query_intent_classifier is None:
        _query_intent_classifier = QueryIntentClassifier()
    return _query_intent_classifier
