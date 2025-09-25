"""
Configurable Redaction Rules Service

This module provides a configurable rule engine for redaction with rule definition
language, evaluation engine, priority/conflict resolution, testing/validation,
and performance optimization.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Callable
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, validator

from agentic_rag.config import get_settings
from agentic_rag.models.database import UserRole
from agentic_rag.services.pii_detection import PIIType, RedactionStrategy, PIIMatch
# from agentic_rag.services.redaction_policies import PolicyAction  # Avoid circular import

logger = structlog.get_logger(__name__)


class RuleType(str, Enum):
    """Types of redaction rules."""
    
    PATTERN_BASED = "pattern_based"     # Regex pattern matching
    CONTENT_BASED = "content_based"     # Content analysis based
    CONTEXT_BASED = "context_based"     # Context-aware rules
    METADATA_BASED = "metadata_based"   # Document metadata based
    CONDITIONAL = "conditional"         # Conditional logic rules
    COMPOSITE = "composite"             # Combination of multiple rules


class RuleOperator(str, Enum):
    """Operators for rule conditions."""
    
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"
    NOT_MATCHES = "not_matches"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN = "in"
    NOT_IN = "not_in"


class RuleAction(str, Enum):
    """Actions that rules can take."""
    
    REDACT = "redact"
    ALLOW = "allow"
    DENY = "deny"
    MASK = "mask"
    TRANSFORM = "transform"
    AUDIT = "audit"
    SKIP = "skip"


@dataclass
class RuleCondition:
    """Individual condition within a rule."""
    
    field: str
    operator: RuleOperator
    value: Any
    case_sensitive: bool = False
    negate: bool = False


@dataclass
class RuleMatch:
    """Result of rule matching."""
    
    rule_id: str
    matched: bool
    confidence: float
    action: RuleAction
    redaction_strategy: Optional[RedactionStrategy] = None
    transformation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RedactionRule(BaseModel):
    """Configurable redaction rule."""
    
    # Rule identification
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(default="", description="Rule description")
    
    # Rule type and configuration
    rule_type: RuleType = Field(..., description="Type of rule")
    priority: int = Field(default=100, ge=0, le=1000, description="Rule priority (higher = more important)")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    
    # Rule conditions
    conditions: List[RuleCondition] = Field(
        default_factory=list,
        description="Conditions that must be met"
    )
    condition_logic: str = Field(
        default="AND",
        description="Logic for combining conditions (AND/OR)"
    )
    
    # Pattern-based rules
    pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern for pattern-based rules"
    )
    pattern_flags: int = Field(
        default=0,
        description="Regex flags for pattern matching"
    )
    
    # Actions
    action: RuleAction = Field(..., description="Action to take when rule matches")
    redaction_strategy: Optional[RedactionStrategy] = Field(
        default=None,
        description="Redaction strategy for redact action"
    )
    transformation_template: Optional[str] = Field(
        default=None,
        description="Template for transform action"
    )
    
    # Scope and applicability
    applicable_pii_types: Set[PIIType] = Field(
        default_factory=set,
        description="PII types this rule applies to"
    )
    applicable_roles: Set[UserRole] = Field(
        default_factory=set,
        description="User roles this rule applies to"
    )
    document_types: Set[str] = Field(
        default_factory=set,
        description="Document types this rule applies to"
    )
    
    # Performance settings
    max_execution_time_ms: int = Field(
        default=1000,
        ge=1,
        description="Maximum execution time in milliseconds"
    )
    cache_results: bool = Field(
        default=True,
        description="Whether to cache rule evaluation results"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Rule creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Rule last update timestamp"
    )
    version: int = Field(default=1, description="Rule version")
    tags: Set[str] = Field(default_factory=set, description="Rule tags")


class RuleEvaluationContext(BaseModel):
    """Context for rule evaluation."""
    
    # Content context
    text: str = Field(..., description="Text being evaluated")
    pii_matches: List[PIIMatch] = Field(
        default_factory=list,
        description="Detected PII matches"
    )
    
    # User context
    user_role: UserRole = Field(..., description="User role")
    user_permissions: Set[str] = Field(
        default_factory=set,
        description="User permissions"
    )
    
    # Document context
    document_type: Optional[str] = Field(
        default=None,
        description="Document type"
    )
    document_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    
    # Request context
    request_purpose: Optional[str] = Field(
        default=None,
        description="Purpose of the request"
    )
    client_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Client information"
    )
    
    # Processing context
    processing_stage: str = Field(
        default="evaluation",
        description="Current processing stage"
    )
    previous_actions: List[str] = Field(
        default_factory=list,
        description="Previous actions taken"
    )


class RuleEvaluationResult(BaseModel):
    """Result of rule evaluation."""
    
    # Evaluation results
    matched_rules: List[RuleMatch] = Field(
        default_factory=list,
        description="Rules that matched"
    )
    final_action: RuleAction = Field(
        default=RuleAction.ALLOW,
        description="Final action determined"
    )
    final_strategy: Optional[RedactionStrategy] = Field(
        default=None,
        description="Final redaction strategy"
    )
    
    # Conflict resolution
    conflicts_detected: List[str] = Field(
        default_factory=list,
        description="Conflicts detected between rules"
    )
    conflict_resolution: str = Field(
        default="",
        description="How conflicts were resolved"
    )
    
    # Performance metrics
    evaluation_time_ms: float = Field(
        default=0.0,
        description="Time taken to evaluate rules"
    )
    rules_evaluated: int = Field(
        default=0,
        description="Number of rules evaluated"
    )
    cache_hits: int = Field(
        default=0,
        description="Number of cache hits"
    )
    
    # Explanation
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the decision"
    )
    rule_explanations: List[str] = Field(
        default_factory=list,
        description="Explanations for each matched rule"
    )


class ConfigurableRedactionRulesService:
    """Service for managing and evaluating configurable redaction rules."""
    
    def __init__(self):
        self.settings = get_settings()
        self._rules: Dict[str, RedactionRule] = {}
        self._rule_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self._stats = {
            "total_evaluations": 0,
            "total_rules_evaluated": 0,
            "total_evaluation_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("Configurable redaction rules service initialized")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default redaction rules."""
        
        # High-priority SSN redaction rule
        ssn_rule = RedactionRule(
            rule_id="default_ssn_redaction",
            name="SSN Full Redaction",
            description="Always redact Social Security Numbers",
            rule_type=RuleType.PATTERN_BASED,
            priority=900,
            pattern=r'\b\d{3}-\d{2}-\d{4}\b',
            action=RuleAction.REDACT,
            redaction_strategy=RedactionStrategy.FULL_REDACTION,
            applicable_pii_types={PIIType.SSN},
            applicable_roles={UserRole.VIEWER, UserRole.ANALYST}
        )
        
        # Credit card masking rule
        cc_rule = RedactionRule(
            rule_id="default_cc_masking",
            name="Credit Card Masking",
            description="Mask credit card numbers showing last 4 digits",
            rule_type=RuleType.PATTERN_BASED,
            priority=850,
            pattern=r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            action=RuleAction.REDACT,
            redaction_strategy=RedactionStrategy.MASKING,
            applicable_pii_types={PIIType.CREDIT_CARD},
            applicable_roles={UserRole.VIEWER, UserRole.ANALYST}
        )
        
        # Admin bypass rule
        admin_bypass_rule = RedactionRule(
            rule_id="admin_bypass",
            name="Admin Bypass",
            description="Allow admins to see all content",
            rule_type=RuleType.METADATA_BASED,
            priority=1000,
            conditions=[
                RuleCondition(
                    field="user_role",
                    operator=RuleOperator.EQUALS,
                    value=UserRole.ADMIN.value
                )
            ],
            action=RuleAction.ALLOW,
            applicable_roles={UserRole.ADMIN}
        )
        
        # Add rules
        self.add_rule(ssn_rule)
        self.add_rule(cc_rule)
        self.add_rule(admin_bypass_rule)
        
        logger.info("Default redaction rules initialized")

    def add_rule(self, rule: RedactionRule) -> None:
        """Add or update a redaction rule."""
        rule.updated_at = datetime.now(timezone.utc)
        self._rules[rule.rule_id] = rule
        self._clear_cache()

        logger.info(
            "Redaction rule added/updated",
            rule_id=rule.rule_id,
            name=rule.name,
            priority=rule.priority,
            enabled=rule.enabled
        )

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a redaction rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            self._clear_cache()
            logger.info("Redaction rule removed", rule_id=rule_id)
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[RedactionRule]:
        """Get a specific redaction rule."""
        return self._rules.get(rule_id)

    def list_rules(self, enabled_only: bool = True) -> List[RedactionRule]:
        """List all rules, optionally filtered by enabled status."""
        rules = list(self._rules.values())

        if enabled_only:
            rules = [r for r in rules if r.enabled]

        return sorted(rules, key=lambda r: (-r.priority, r.name))

    def evaluate_rules(self, context: RuleEvaluationContext) -> RuleEvaluationResult:
        """Evaluate redaction rules for the given context."""
        start_time = time.time()

        try:
            # Check cache
            cache_key = self._generate_cache_key(context)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self._stats["cache_hits"] += 1
                return cached_result
            self._stats["cache_misses"] += 1

            # Get applicable rules
            applicable_rules = self._get_applicable_rules(context)

            # Evaluate each rule
            matched_rules = []
            rules_evaluated = 0

            for rule in applicable_rules:
                try:
                    match_result = self._evaluate_single_rule(rule, context)
                    if match_result.matched:
                        matched_rules.append(match_result)
                    rules_evaluated += 1
                except Exception as e:
                    logger.warning(
                        "Rule evaluation failed",
                        rule_id=rule.rule_id,
                        error=str(e)
                    )

            # Resolve conflicts and determine final action
            final_action, final_strategy, conflicts, resolution = self._resolve_conflicts(matched_rules)

            # Calculate evaluation time
            evaluation_time_ms = (time.time() - start_time) * 1000

            # Create result
            result = RuleEvaluationResult(
                matched_rules=matched_rules,
                final_action=final_action,
                final_strategy=final_strategy,
                conflicts_detected=conflicts,
                conflict_resolution=resolution,
                evaluation_time_ms=evaluation_time_ms,
                rules_evaluated=rules_evaluated,
                explanation=self._generate_explanation(final_action, matched_rules, conflicts),
                rule_explanations=[self._explain_rule_match(match) for match in matched_rules]
            )

            # Cache result
            self._cache_result(cache_key, result)

            # Update stats
            self._stats["total_evaluations"] += 1
            self._stats["total_rules_evaluated"] += rules_evaluated
            self._stats["total_evaluation_time_ms"] += evaluation_time_ms
            if conflicts:
                self._stats["conflicts_detected"] += len(conflicts)
                self._stats["conflicts_resolved"] += 1

            logger.debug(
                "Rule evaluation completed",
                matched_rules=len(matched_rules),
                final_action=final_action.value,
                conflicts=len(conflicts),
                evaluation_time_ms=evaluation_time_ms
            )

            return result

        except Exception as e:
            logger.error("Rule evaluation failed", error=str(e))
            # Fail safe - default to redaction
            return RuleEvaluationResult(
                final_action=RuleAction.REDACT,
                final_strategy=RedactionStrategy.FULL_REDACTION,
                explanation=f"Rule evaluation failed: {str(e)}"
            )

    def _get_applicable_rules(self, context: RuleEvaluationContext) -> List[RedactionRule]:
        """Get rules applicable to the given context."""
        applicable = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            # Check role applicability
            if rule.applicable_roles and context.user_role not in rule.applicable_roles:
                continue

            # Check document type applicability
            if (rule.document_types and context.document_type and
                context.document_type not in rule.document_types):
                continue

            # Check PII type applicability
            if rule.applicable_pii_types:
                pii_types_in_context = {match.pii_type for match in context.pii_matches}
                if not rule.applicable_pii_types.intersection(pii_types_in_context):
                    continue

            applicable.append(rule)

        # Sort by priority (highest first)
        return sorted(applicable, key=lambda r: -r.priority)

    def _evaluate_single_rule(self, rule: RedactionRule, context: RuleEvaluationContext) -> RuleMatch:
        """Evaluate a single rule against the context."""

        matched = False
        confidence = 0.0
        metadata = {}

        try:
            if rule.rule_type == RuleType.PATTERN_BASED:
                matched, confidence, metadata = self._evaluate_pattern_rule(rule, context)

            elif rule.rule_type == RuleType.CONTENT_BASED:
                matched, confidence, metadata = self._evaluate_content_rule(rule, context)

            elif rule.rule_type == RuleType.CONTEXT_BASED:
                matched, confidence, metadata = self._evaluate_context_rule(rule, context)

            elif rule.rule_type == RuleType.METADATA_BASED:
                matched, confidence, metadata = self._evaluate_metadata_rule(rule, context)

            elif rule.rule_type == RuleType.CONDITIONAL:
                matched, confidence, metadata = self._evaluate_conditional_rule(rule, context)

            elif rule.rule_type == RuleType.COMPOSITE:
                matched, confidence, metadata = self._evaluate_composite_rule(rule, context)

            else:
                logger.warning("Unknown rule type", rule_type=rule.rule_type, rule_id=rule.rule_id)

        except Exception as e:
            logger.error(
                "Rule evaluation error",
                rule_id=rule.rule_id,
                error=str(e)
            )

        return RuleMatch(
            rule_id=rule.rule_id,
            matched=matched,
            confidence=confidence,
            action=rule.action,
            redaction_strategy=rule.redaction_strategy,
            transformation=rule.transformation_template,
            metadata=metadata
        )

    def _evaluate_pattern_rule(self, rule: RedactionRule, context: RuleEvaluationContext) -> tuple[bool, float, dict]:
        """Evaluate a pattern-based rule."""
        if not rule.pattern:
            return False, 0.0, {}

        try:
            pattern = re.compile(rule.pattern, rule.pattern_flags)
            matches = pattern.findall(context.text)

            if matches:
                confidence = min(1.0, len(matches) * 0.2 + 0.6)  # Base confidence + match count boost
                return True, confidence, {"matches": matches, "match_count": len(matches)}

        except re.error as e:
            logger.warning("Invalid regex pattern", rule_id=rule.rule_id, pattern=rule.pattern, error=str(e))

        return False, 0.0, {}

    def _evaluate_content_rule(self, rule: RedactionRule, context: RuleEvaluationContext) -> tuple[bool, float, dict]:
        """Evaluate a content-based rule."""
        # Content-based rules would analyze the actual content
        # For now, implement basic keyword matching

        if not rule.conditions:
            return False, 0.0, {}

        for condition in rule.conditions:
            if condition.field == "content":
                if self._evaluate_condition(condition, context.text):
                    return True, 0.8, {"condition_matched": condition.field}

        return False, 0.0, {}

    def _evaluate_context_rule(self, rule: RedactionRule, context: RuleEvaluationContext) -> tuple[bool, float, dict]:
        """Evaluate a context-based rule."""
        # Context-based rules consider surrounding context
        # For now, implement basic context checks

        if not rule.conditions:
            return False, 0.0, {}

        # Check if any PII matches have specific context
        for pii_match in context.pii_matches:
            if pii_match.context:
                for condition in rule.conditions:
                    if condition.field == "context":
                        if self._evaluate_condition(condition, pii_match.context):
                            return True, 0.7, {"context_matched": True}

        return False, 0.0, {}

    def _evaluate_metadata_rule(self, rule: RedactionRule, context: RuleEvaluationContext) -> tuple[bool, float, dict]:
        """Evaluate a metadata-based rule."""
        if not rule.conditions:
            return False, 0.0, {}

        # Build evaluation context
        eval_context = {
            "user_role": context.user_role.value,
            "document_type": context.document_type,
            "request_purpose": context.request_purpose,
            **context.document_metadata,
            **context.client_info
        }

        # Evaluate conditions
        condition_results = []
        for condition in rule.conditions:
            result = self._evaluate_condition_on_context(condition, eval_context)
            condition_results.append(result)

        # Apply condition logic
        if rule.condition_logic.upper() == "OR":
            matched = any(condition_results)
        else:  # Default to AND
            matched = all(condition_results)

        if matched:
            return True, 0.9, {"conditions_matched": len([r for r in condition_results if r])}

        return False, 0.0, {}

    def _evaluate_conditional_rule(self, rule: RedactionRule, context: RuleEvaluationContext) -> tuple[bool, float, dict]:
        """Evaluate a conditional rule."""
        # Conditional rules have complex logic
        # For now, treat similar to metadata rules
        return self._evaluate_metadata_rule(rule, context)

    def _evaluate_composite_rule(self, rule: RedactionRule, context: RuleEvaluationContext) -> tuple[bool, float, dict]:
        """Evaluate a composite rule."""
        # Composite rules combine multiple rule types
        # For now, evaluate all available components

        results = []
        metadata = {}

        # Try pattern matching if pattern exists
        if rule.pattern:
            matched, confidence, meta = self._evaluate_pattern_rule(rule, context)
            if matched:
                results.append((matched, confidence))
                metadata.update(meta)

        # Try condition evaluation if conditions exist
        if rule.conditions:
            matched, confidence, meta = self._evaluate_metadata_rule(rule, context)
            if matched:
                results.append((matched, confidence))
                metadata.update(meta)

        if results:
            # Take the highest confidence result
            best_result = max(results, key=lambda x: x[1])
            return best_result[0], best_result[1], metadata

        return False, 0.0, {}

    def _evaluate_condition(self, condition: RuleCondition, value: str) -> bool:
        """Evaluate a single condition against a value."""
        try:
            condition_value = str(condition.value)
            test_value = value if condition.case_sensitive else value.lower()
            condition_value = condition_value if condition.case_sensitive else condition_value.lower()

            if condition.operator == RuleOperator.EQUALS:
                result = test_value == condition_value
            elif condition.operator == RuleOperator.NOT_EQUALS:
                result = test_value != condition_value
            elif condition.operator == RuleOperator.CONTAINS:
                result = condition_value in test_value
            elif condition.operator == RuleOperator.NOT_CONTAINS:
                result = condition_value not in test_value
            elif condition.operator == RuleOperator.STARTS_WITH:
                result = test_value.startswith(condition_value)
            elif condition.operator == RuleOperator.ENDS_WITH:
                result = test_value.endswith(condition_value)
            elif condition.operator == RuleOperator.MATCHES:
                result = bool(re.search(condition_value, test_value))
            elif condition.operator == RuleOperator.NOT_MATCHES:
                result = not bool(re.search(condition_value, test_value))
            else:
                result = False

            return result if not condition.negate else not result

        except Exception as e:
            logger.warning("Condition evaluation failed", condition=condition, error=str(e))
            return False

    def _evaluate_condition_on_context(self, condition: RuleCondition, context: Dict[str, Any]) -> bool:
        """Evaluate a condition against a context dictionary."""
        if condition.field not in context:
            return False

        field_value = context[condition.field]

        try:
            if condition.operator == RuleOperator.EQUALS:
                result = field_value == condition.value
            elif condition.operator == RuleOperator.NOT_EQUALS:
                result = field_value != condition.value
            elif condition.operator == RuleOperator.IN:
                result = field_value in condition.value if isinstance(condition.value, (list, set, tuple)) else False
            elif condition.operator == RuleOperator.NOT_IN:
                result = field_value not in condition.value if isinstance(condition.value, (list, set, tuple)) else True
            elif condition.operator == RuleOperator.GREATER_THAN:
                result = float(field_value) > float(condition.value)
            elif condition.operator == RuleOperator.LESS_THAN:
                result = float(field_value) < float(condition.value)
            else:
                # Fall back to string comparison
                result = self._evaluate_condition(condition, str(field_value))

            return result if not condition.negate else not result

        except Exception as e:
            logger.warning("Context condition evaluation failed", condition=condition, error=str(e))
            return False

    def _resolve_conflicts(self, matched_rules: List[RuleMatch]) -> tuple[RuleAction, Optional[RedactionStrategy], List[str], str]:
        """Resolve conflicts between matched rules."""

        if not matched_rules:
            return RuleAction.ALLOW, None, [], "No rules matched"

        if len(matched_rules) == 1:
            rule = matched_rules[0]
            return rule.action, rule.redaction_strategy, [], f"Single rule {rule.rule_id} applied"

        # Group rules by action
        actions_by_type = {}
        for rule in matched_rules:
            if rule.action not in actions_by_type:
                actions_by_type[rule.action] = []
            actions_by_type[rule.action].append(rule)

        conflicts = []

        # Check for conflicting actions
        if len(actions_by_type) > 1:
            conflict_actions = list(actions_by_type.keys())
            conflicts.append(f"Conflicting actions: {[a.value for a in conflict_actions]}")

        # Resolve conflicts using priority order
        # 1. DENY takes precedence (most restrictive)
        # 2. REDACT takes precedence over ALLOW
        # 3. Within same action, highest priority rule wins

        if RuleAction.DENY in actions_by_type:
            winning_rule = max(actions_by_type[RuleAction.DENY], key=lambda r: r.confidence)
            resolution = f"DENY action from rule {winning_rule.rule_id} takes precedence"
            return RuleAction.DENY, None, conflicts, resolution

        elif RuleAction.REDACT in actions_by_type:
            redact_rules = actions_by_type[RuleAction.REDACT]
            winning_rule = max(redact_rules, key=lambda r: r.confidence)
            resolution = f"REDACT action from rule {winning_rule.rule_id} takes precedence"
            return RuleAction.REDACT, winning_rule.redaction_strategy, conflicts, resolution

        elif RuleAction.MASK in actions_by_type:
            mask_rules = actions_by_type[RuleAction.MASK]
            winning_rule = max(mask_rules, key=lambda r: r.confidence)
            resolution = f"MASK action from rule {winning_rule.rule_id} takes precedence"
            return RuleAction.MASK, winning_rule.redaction_strategy, conflicts, resolution

        elif RuleAction.TRANSFORM in actions_by_type:
            transform_rules = actions_by_type[RuleAction.TRANSFORM]
            winning_rule = max(transform_rules, key=lambda r: r.confidence)
            resolution = f"TRANSFORM action from rule {winning_rule.rule_id} takes precedence"
            return RuleAction.TRANSFORM, winning_rule.redaction_strategy, conflicts, resolution

        elif RuleAction.AUDIT in actions_by_type:
            audit_rules = actions_by_type[RuleAction.AUDIT]
            winning_rule = max(audit_rules, key=lambda r: r.confidence)
            resolution = f"AUDIT action from rule {winning_rule.rule_id} takes precedence"
            return RuleAction.AUDIT, None, conflicts, resolution

        elif RuleAction.ALLOW in actions_by_type:
            allow_rules = actions_by_type[RuleAction.ALLOW]
            winning_rule = max(allow_rules, key=lambda r: r.confidence)
            resolution = f"ALLOW action from rule {winning_rule.rule_id} takes precedence"
            return RuleAction.ALLOW, None, conflicts, resolution

        else:
            # Default fallback
            return RuleAction.ALLOW, None, conflicts, "Default action applied"

    def _generate_explanation(self, final_action: RuleAction, matched_rules: List[RuleMatch], conflicts: List[str]) -> str:
        """Generate human-readable explanation of the decision."""

        if not matched_rules:
            return "No redaction rules matched the content"

        explanation_parts = []

        if len(matched_rules) == 1:
            rule = matched_rules[0]
            explanation_parts.append(f"Rule '{rule.rule_id}' matched and determined action: {final_action.value}")
        else:
            explanation_parts.append(f"{len(matched_rules)} rules matched")

            if conflicts:
                explanation_parts.append(f"Conflicts detected: {'; '.join(conflicts)}")

            explanation_parts.append(f"Final action determined: {final_action.value}")

        return ". ".join(explanation_parts)

    def _explain_rule_match(self, match: RuleMatch) -> str:
        """Generate explanation for a single rule match."""

        explanation = f"Rule {match.rule_id} matched with {match.confidence:.2f} confidence"

        if match.metadata:
            metadata_info = []
            for key, value in match.metadata.items():
                if key == "matches" and isinstance(value, list):
                    metadata_info.append(f"found {len(value)} pattern matches")
                elif key == "conditions_matched":
                    metadata_info.append(f"matched {value} conditions")
                else:
                    metadata_info.append(f"{key}: {value}")

            if metadata_info:
                explanation += f" ({', '.join(metadata_info)})"

        return explanation

    # Utility methods
    def _generate_cache_key(self, context: RuleEvaluationContext) -> str:
        """Generate cache key for rule evaluation context."""
        import hashlib

        # Create a hash of the relevant context elements
        key_elements = [
            context.text,
            context.user_role.value,
            context.document_type or "",
            str(sorted(context.user_permissions)),
            str(len(context.pii_matches)),
            context.processing_stage
        ]

        key_string = "|".join(key_elements)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[RuleEvaluationResult]:
        """Get cached rule evaluation result."""
        if cache_key not in self._rule_cache:
            return None

        cached_item = self._rule_cache[cache_key]
        if time.time() - cached_item["timestamp"] > 300:  # 5 minute cache
            del self._rule_cache[cache_key]
            return None

        return cached_item["result"]

    def _cache_result(self, cache_key: str, result: RuleEvaluationResult) -> None:
        """Cache rule evaluation result."""
        self._rule_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        # Simple cache cleanup
        if len(self._rule_cache) > 1000:
            oldest_key = min(self._rule_cache.keys(),
                           key=lambda k: self._rule_cache[k]["timestamp"])
            del self._rule_cache[oldest_key]

    def _clear_cache(self) -> None:
        """Clear the rule evaluation cache."""
        self._rule_cache.clear()

    def validate_rule(self, rule: RedactionRule) -> List[str]:
        """Validate a redaction rule and return any errors."""
        errors = []

        # Check required fields
        if not rule.rule_id:
            errors.append("Rule ID is required")

        if not rule.name:
            errors.append("Rule name is required")

        # Check for duplicate rule ID
        if rule.rule_id in self._rules:
            existing = self._rules[rule.rule_id]
            if existing.version >= rule.version:
                errors.append(f"Rule ID {rule.rule_id} already exists with same or higher version")

        # Validate pattern if pattern-based
        if rule.rule_type == RuleType.PATTERN_BASED:
            if not rule.pattern:
                errors.append("Pattern is required for pattern-based rules")
            else:
                try:
                    re.compile(rule.pattern, rule.pattern_flags)
                except re.error as e:
                    errors.append(f"Invalid regex pattern: {str(e)}")

        # Validate conditions
        for i, condition in enumerate(rule.conditions):
            if not condition.field:
                errors.append(f"Condition {i}: Field is required")

            if condition.operator in [RuleOperator.IN, RuleOperator.NOT_IN]:
                if not isinstance(condition.value, (list, set, tuple)):
                    errors.append(f"Condition {i}: IN/NOT_IN operators require list/set/tuple value")

        # Check action consistency
        if rule.action == RuleAction.REDACT and not rule.redaction_strategy:
            errors.append("Redaction strategy is required for REDACT action")

        if rule.action == RuleAction.TRANSFORM and not rule.transformation_template:
            errors.append("Transformation template is required for TRANSFORM action")

        return errors

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "total_rules": len(self._rules),
            "enabled_rules": len([r for r in self._rules.values() if r.enabled]),
            "cache_size": len(self._rule_cache),
            "rules_by_type": {
                rule_type.value: len([r for r in self._rules.values() if r.rule_type == rule_type])
                for rule_type in RuleType
            },
            "rules_by_action": {
                action.value: len([r for r in self._rules.values() if r.action == action])
                for action in RuleAction
            }
        }


# Global service instance
_redaction_rules_service: Optional[ConfigurableRedactionRulesService] = None


def get_redaction_rules_service() -> ConfigurableRedactionRulesService:
    """Get or create the global redaction rules service instance."""
    global _redaction_rules_service

    if _redaction_rules_service is None:
        _redaction_rules_service = ConfigurableRedactionRulesService()

    return _redaction_rules_service


def reset_redaction_rules_service() -> None:
    """Reset the global redaction rules service instance."""
    global _redaction_rules_service
    _redaction_rules_service = None
