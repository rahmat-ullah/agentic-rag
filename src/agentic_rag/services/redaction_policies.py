"""
Role-Based Redaction Policies Service

This module provides role-based redaction policies that integrate with the
existing authorization system to control what information is redacted
based on user roles and permissions.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, validator

from agentic_rag.config import get_settings
from agentic_rag.models.database import User, UserRole
from agentic_rag.services.authorization import Permission
from agentic_rag.services.pii_detection import PIIType, RedactionStrategy, PIIDetectionService

logger = structlog.get_logger(__name__)


class RedactionLevel(str, Enum):
    """Levels of redaction intensity."""
    
    NONE = "none"           # No redaction
    MINIMAL = "minimal"     # Only highly sensitive PII
    STANDARD = "standard"   # Standard PII redaction
    STRICT = "strict"       # Aggressive redaction
    MAXIMUM = "maximum"     # Maximum redaction


class PolicyScope(str, Enum):
    """Scope of redaction policy application."""
    
    GLOBAL = "global"           # Apply to all content
    DOCUMENT_TYPE = "document_type"  # Apply to specific document types
    CONTENT_TYPE = "content_type"    # Apply to specific content types
    FIELD_LEVEL = "field_level"      # Apply to specific fields


class PolicyAction(str, Enum):
    """Actions that can be taken by redaction policies."""
    
    ALLOW = "allow"         # Allow access without redaction
    REDACT = "redact"       # Apply redaction
    DENY = "deny"           # Deny access completely
    AUDIT = "audit"         # Allow but log access


@dataclass
class RedactionRule:
    """Individual redaction rule within a policy."""
    
    pii_types: Set[PIIType]
    redaction_strategy: RedactionStrategy
    conditions: Dict[str, Any] = field(default_factory=dict)
    exceptions: List[str] = field(default_factory=list)
    priority: int = 0
    enabled: bool = True


class RedactionPolicy(BaseModel):
    """Role-based redaction policy."""
    
    # Policy identification
    policy_id: str = Field(..., description="Unique policy identifier")
    name: str = Field(..., description="Human-readable policy name")
    description: str = Field(default="", description="Policy description")
    
    # Policy scope
    applicable_roles: Set[UserRole] = Field(
        default_factory=set,
        description="User roles this policy applies to"
    )
    required_permissions: Set[Permission] = Field(
        default_factory=set,
        description="Permissions required to bypass redaction"
    )
    scope: PolicyScope = Field(
        default=PolicyScope.GLOBAL,
        description="Scope of policy application"
    )
    scope_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filters for scoped application"
    )
    
    # Redaction configuration
    default_action: PolicyAction = Field(
        default=PolicyAction.REDACT,
        description="Default action when no specific rule matches"
    )
    redaction_level: RedactionLevel = Field(
        default=RedactionLevel.STANDARD,
        description="Overall redaction level"
    )
    rules: List[RedactionRule] = Field(
        default_factory=list,
        description="Specific redaction rules"
    )
    
    # Policy behavior
    inherit_from_parent: bool = Field(
        default=True,
        description="Whether to inherit from parent policies"
    )
    allow_overrides: bool = Field(
        default=False,
        description="Whether child policies can override this policy"
    )
    strict_mode: bool = Field(
        default=False,
        description="Whether to apply strict interpretation"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Policy creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Policy last update timestamp"
    )
    version: int = Field(default=1, description="Policy version")
    is_active: bool = Field(default=True, description="Whether policy is active")


class PolicyEvaluationContext(BaseModel):
    """Context for policy evaluation."""
    
    # User context
    user: User = Field(..., description="User requesting access")
    user_permissions: Set[Permission] = Field(
        default_factory=set,
        description="User's permissions"
    )
    
    # Content context
    content_type: Optional[str] = Field(
        default=None,
        description="Type of content being accessed"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Type of document being accessed"
    )
    field_name: Optional[str] = Field(
        default=None,
        description="Specific field being accessed"
    )
    
    # Request context
    request_purpose: Optional[str] = Field(
        default=None,
        description="Purpose of the request"
    )
    client_ip: Optional[str] = Field(
        default=None,
        description="Client IP address"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata"
    )


class PolicyEvaluationResult(BaseModel):
    """Result of policy evaluation."""
    
    # Decision
    action: PolicyAction = Field(..., description="Action to take")
    redaction_required: bool = Field(
        default=False,
        description="Whether redaction is required"
    )
    redaction_rules: List[RedactionRule] = Field(
        default_factory=list,
        description="Applicable redaction rules"
    )
    
    # Policy information
    applied_policies: List[str] = Field(
        default_factory=list,
        description="IDs of policies that were applied"
    )
    policy_hierarchy: List[str] = Field(
        default_factory=list,
        description="Policy hierarchy used in evaluation"
    )
    
    # Evaluation metadata
    evaluation_time_ms: float = Field(
        default=0.0,
        description="Time taken to evaluate policies"
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the policy decision"
    )
    
    # Audit information
    audit_required: bool = Field(
        default=False,
        description="Whether this access should be audited"
    )
    audit_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for audit logging"
    )
    
    # Explanation
    decision_reason: str = Field(
        default="",
        description="Human-readable explanation of the decision"
    )
    rule_explanations: List[str] = Field(
        default_factory=list,
        description="Explanations for each applied rule"
    )


class RoleBasedRedactionService:
    """Service for managing and evaluating role-based redaction policies."""
    
    def __init__(self):
        self.settings = get_settings()
        self._policies: Dict[str, RedactionPolicy] = {}
        self._policy_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self._stats = {
            "total_evaluations": 0,
            "policy_hits": 0,
            "policy_misses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_evaluation_time_ms": 0.0
        }
        
        # Initialize default policies
        self._initialize_default_policies()
        
        logger.info("Role-based redaction service initialized")
    
    def _initialize_default_policies(self) -> None:
        """Initialize default redaction policies for each role."""
        
        # Viewer policy - strict redaction
        viewer_policy = RedactionPolicy(
            policy_id="viewer_default",
            name="Viewer Default Policy",
            description="Default redaction policy for viewer role",
            applicable_roles={UserRole.VIEWER},
            redaction_level=RedactionLevel.STRICT,
            rules=[
                RedactionRule(
                    pii_types={PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PHONE, PIIType.EMAIL},
                    redaction_strategy=RedactionStrategy.FULL_REDACTION,
                    priority=100
                ),
                RedactionRule(
                    pii_types={PIIType.PERSON_NAME, PIIType.ADDRESS},
                    redaction_strategy=RedactionStrategy.ANONYMIZATION,
                    priority=90
                )
            ]
        )
        
        # Analyst policy - moderate redaction
        analyst_policy = RedactionPolicy(
            policy_id="analyst_default",
            name="Analyst Default Policy",
            description="Default redaction policy for analyst role",
            applicable_roles={UserRole.ANALYST},
            redaction_level=RedactionLevel.STANDARD,
            rules=[
                RedactionRule(
                    pii_types={PIIType.SSN, PIIType.CREDIT_CARD},
                    redaction_strategy=RedactionStrategy.MASKING,
                    priority=100
                ),
                RedactionRule(
                    pii_types={PIIType.PHONE, PIIType.EMAIL},
                    redaction_strategy=RedactionStrategy.PARTIAL_REDACTION,
                    priority=90
                )
            ]
        )
        
        # Admin policy - minimal redaction
        admin_policy = RedactionPolicy(
            policy_id="admin_default",
            name="Admin Default Policy",
            description="Default redaction policy for admin role",
            applicable_roles={UserRole.ADMIN},
            redaction_level=RedactionLevel.MINIMAL,
            rules=[
                RedactionRule(
                    pii_types={PIIType.SSN, PIIType.CREDIT_CARD},
                    redaction_strategy=RedactionStrategy.PARTIAL_REDACTION,
                    priority=100
                )
            ]
        )
        
        # Add policies
        self.add_policy(viewer_policy)
        self.add_policy(analyst_policy)
        self.add_policy(admin_policy)
        
        logger.info("Default redaction policies initialized")

    def add_policy(self, policy: RedactionPolicy) -> None:
        """Add or update a redaction policy."""
        policy.updated_at = datetime.now(timezone.utc)
        self._policies[policy.policy_id] = policy
        self._clear_policy_cache()

        logger.info(
            "Redaction policy added/updated",
            policy_id=policy.policy_id,
            name=policy.name,
            applicable_roles=[role.value for role in policy.applicable_roles]
        )

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a redaction policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            self._clear_policy_cache()
            logger.info("Redaction policy removed", policy_id=policy_id)
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[RedactionPolicy]:
        """Get a specific redaction policy."""
        return self._policies.get(policy_id)

    def list_policies(self, role: Optional[UserRole] = None) -> List[RedactionPolicy]:
        """List all policies, optionally filtered by role."""
        policies = list(self._policies.values())

        if role:
            policies = [p for p in policies if role in p.applicable_roles]

        return sorted(policies, key=lambda p: p.name)

    def evaluate_policies(self, context: PolicyEvaluationContext) -> PolicyEvaluationResult:
        """Evaluate redaction policies for the given context."""
        start_time = time.time()

        try:
            # Check cache
            cache_key = self._generate_cache_key(context)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self._stats["cache_hits"] += 1
                return cached_result
            self._stats["cache_misses"] += 1

            # Find applicable policies
            applicable_policies = self._find_applicable_policies(context)

            if not applicable_policies:
                # No policies found - default to no redaction for admins, full redaction for others
                default_action = (PolicyAction.ALLOW if context.user.role == UserRole.ADMIN
                                else PolicyAction.REDACT)

                result = PolicyEvaluationResult(
                    action=default_action,
                    redaction_required=(default_action == PolicyAction.REDACT),
                    decision_reason="No applicable policies found, using default behavior"
                )
            else:
                # Evaluate policies in priority order
                result = self._evaluate_policy_chain(applicable_policies, context)

            # Calculate evaluation time
            evaluation_time_ms = (time.time() - start_time) * 1000
            result.evaluation_time_ms = evaluation_time_ms

            # Cache result
            self._cache_result(cache_key, result)

            # Update stats
            self._stats["total_evaluations"] += 1
            self._stats["total_evaluation_time_ms"] += evaluation_time_ms
            if applicable_policies:
                self._stats["policy_hits"] += 1
            else:
                self._stats["policy_misses"] += 1

            logger.debug(
                "Policy evaluation completed",
                user_role=context.user.role.value,
                action=result.action.value,
                redaction_required=result.redaction_required,
                policies_applied=len(result.applied_policies),
                evaluation_time_ms=evaluation_time_ms
            )

            return result

        except Exception as e:
            logger.error("Policy evaluation failed", error=str(e))
            # Fail safe - default to redaction
            return PolicyEvaluationResult(
                action=PolicyAction.REDACT,
                redaction_required=True,
                decision_reason=f"Policy evaluation failed: {str(e)}"
            )

    def _find_applicable_policies(self, context: PolicyEvaluationContext) -> List[RedactionPolicy]:
        """Find policies applicable to the given context."""
        applicable = []

        for policy in self._policies.values():
            if not policy.is_active:
                continue

            # Check role applicability
            if policy.applicable_roles and context.user.role not in policy.applicable_roles:
                continue

            # Check permission requirements
            if policy.required_permissions:
                if not policy.required_permissions.issubset(context.user_permissions):
                    continue

            # Check scope filters
            if not self._check_scope_filters(policy, context):
                continue

            applicable.append(policy)

        # Sort by priority (higher priority first)
        return sorted(applicable, key=lambda p: (-max(r.priority for r in p.rules) if p.rules else 0))

    def _check_scope_filters(self, policy: RedactionPolicy, context: PolicyEvaluationContext) -> bool:
        """Check if policy scope filters match the context."""
        if policy.scope == PolicyScope.GLOBAL:
            return True

        elif policy.scope == PolicyScope.DOCUMENT_TYPE:
            if "document_types" in policy.scope_filters:
                return context.document_type in policy.scope_filters["document_types"]

        elif policy.scope == PolicyScope.CONTENT_TYPE:
            if "content_types" in policy.scope_filters:
                return context.content_type in policy.scope_filters["content_types"]

        elif policy.scope == PolicyScope.FIELD_LEVEL:
            if "field_names" in policy.scope_filters:
                return context.field_name in policy.scope_filters["field_names"]

        return True

    def _evaluate_policy_chain(self, policies: List[RedactionPolicy],
                              context: PolicyEvaluationContext) -> PolicyEvaluationResult:
        """Evaluate a chain of policies and combine results."""

        # Start with most permissive defaults
        final_action = PolicyAction.ALLOW
        redaction_required = False
        applicable_rules = []
        applied_policies = []
        rule_explanations = []
        audit_required = False

        for policy in policies:
            # Evaluate individual policy
            policy_result = self._evaluate_single_policy(policy, context)

            # Combine results (more restrictive wins)
            if policy_result.action == PolicyAction.DENY:
                final_action = PolicyAction.DENY
                redaction_required = False  # Deny overrides redaction
            elif policy_result.action == PolicyAction.REDACT and final_action != PolicyAction.DENY:
                final_action = PolicyAction.REDACT
                redaction_required = True
            elif policy_result.action == PolicyAction.AUDIT:
                audit_required = True

            # Collect applicable rules
            applicable_rules.extend(policy_result.redaction_rules)
            applied_policies.append(policy.policy_id)
            rule_explanations.extend(policy_result.rule_explanations)

        # Remove duplicate rules and sort by priority
        unique_rules = []
        seen_rules = set()
        for rule in sorted(applicable_rules, key=lambda r: -r.priority):
            rule_key = (tuple(sorted(rule.pii_types)), rule.redaction_strategy)
            if rule_key not in seen_rules:
                unique_rules.append(rule)
                seen_rules.add(rule_key)

        return PolicyEvaluationResult(
            action=final_action,
            redaction_required=redaction_required,
            redaction_rules=unique_rules,
            applied_policies=applied_policies,
            audit_required=audit_required,
            decision_reason=self._generate_decision_reason(final_action, applied_policies),
            rule_explanations=rule_explanations
        )

    def _evaluate_single_policy(self, policy: RedactionPolicy,
                               context: PolicyEvaluationContext) -> PolicyEvaluationResult:
        """Evaluate a single policy against the context."""

        applicable_rules = []
        rule_explanations = []

        # Check each rule in the policy
        for rule in policy.rules:
            if not rule.enabled:
                continue

            # Check rule conditions
            if self._check_rule_conditions(rule, context):
                applicable_rules.append(rule)
                rule_explanations.append(
                    f"Rule applied: {rule.redaction_strategy.value} for {', '.join(rule.pii_types)}"
                )

        # Determine action based on rules and policy defaults
        if applicable_rules:
            action = PolicyAction.REDACT
            redaction_required = True
        else:
            action = policy.default_action
            redaction_required = (action == PolicyAction.REDACT)

        return PolicyEvaluationResult(
            action=action,
            redaction_required=redaction_required,
            redaction_rules=applicable_rules,
            applied_policies=[policy.policy_id],
            rule_explanations=rule_explanations
        )

    def _check_rule_conditions(self, rule: RedactionRule,
                              context: PolicyEvaluationContext) -> bool:
        """Check if rule conditions are met for the given context."""

        # Check basic conditions
        if "min_confidence" in rule.conditions:
            # This would be used when integrating with PII detection
            pass

        if "content_types" in rule.conditions:
            if context.content_type not in rule.conditions["content_types"]:
                return False

        if "document_types" in rule.conditions:
            if context.document_type not in rule.conditions["document_types"]:
                return False

        # Check exceptions
        for exception in rule.exceptions:
            if self._check_exception(exception, context):
                return False

        return True

    def _check_exception(self, exception: str, context: PolicyEvaluationContext) -> bool:
        """Check if an exception applies to the context."""

        # Simple exception checking - could be enhanced
        if exception == "admin_override" and context.user.role == UserRole.ADMIN:
            return True

        if exception == "internal_request" and context.client_ip and context.client_ip.startswith("10."):
            return True

        return False

    def _generate_decision_reason(self, action: PolicyAction, applied_policies: List[str]) -> str:
        """Generate human-readable decision reason."""

        if not applied_policies:
            return "No applicable policies found"

        policy_list = ", ".join(applied_policies)

        if action == PolicyAction.ALLOW:
            return f"Access allowed by policies: {policy_list}"
        elif action == PolicyAction.REDACT:
            return f"Redaction required by policies: {policy_list}"
        elif action == PolicyAction.DENY:
            return f"Access denied by policies: {policy_list}"
        elif action == PolicyAction.AUDIT:
            return f"Access audited by policies: {policy_list}"

        return f"Action {action.value} determined by policies: {policy_list}"

    def _generate_cache_key(self, context: PolicyEvaluationContext) -> str:
        """Generate cache key for policy evaluation context."""
        import hashlib

        key_data = (
            str(context.user.id),
            context.user.role.value,
            context.content_type or "",
            context.document_type or "",
            context.field_name or "",
            str(sorted(context.user_permissions))
        )

        key_string = "|".join(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[PolicyEvaluationResult]:
        """Get cached policy evaluation result."""
        if cache_key not in self._policy_cache:
            return None

        cached_item = self._policy_cache[cache_key]
        if time.time() - cached_item["timestamp"] > 300:  # 5 minute cache
            del self._policy_cache[cache_key]
            return None

        return cached_item["result"]

    def _cache_result(self, cache_key: str, result: PolicyEvaluationResult) -> None:
        """Cache policy evaluation result."""
        self._policy_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        # Simple cache cleanup
        if len(self._policy_cache) > 1000:
            oldest_key = min(self._policy_cache.keys(),
                           key=lambda k: self._policy_cache[k]["timestamp"])
            del self._policy_cache[oldest_key]

    def _clear_policy_cache(self) -> None:
        """Clear the policy evaluation cache."""
        self._policy_cache.clear()

    def validate_policy(self, policy: RedactionPolicy) -> List[str]:
        """Validate a redaction policy and return any errors."""
        errors = []

        # Check required fields
        if not policy.policy_id:
            errors.append("Policy ID is required")

        if not policy.name:
            errors.append("Policy name is required")

        # Check for duplicate policy ID
        if policy.policy_id in self._policies:
            existing = self._policies[policy.policy_id]
            if existing.version >= policy.version:
                errors.append(f"Policy ID {policy.policy_id} already exists with same or higher version")

        # Validate rules
        for i, rule in enumerate(policy.rules):
            if not rule.pii_types:
                errors.append(f"Rule {i}: PII types cannot be empty")

            if rule.priority < 0:
                errors.append(f"Rule {i}: Priority must be non-negative")

        # Check role consistency
        if not policy.applicable_roles:
            errors.append("At least one applicable role must be specified")

        return errors

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "total_policies": len(self._policies),
            "active_policies": len([p for p in self._policies.values() if p.is_active]),
            "cache_size": len(self._policy_cache),
            "policies_by_role": {
                role.value: len([p for p in self._policies.values()
                               if role in p.applicable_roles])
                for role in UserRole
            }
        }


# Global service instance
_redaction_policies_service: Optional[RoleBasedRedactionService] = None


def get_redaction_policies_service() -> RoleBasedRedactionService:
    """Get or create the global redaction policies service instance."""
    global _redaction_policies_service

    if _redaction_policies_service is None:
        _redaction_policies_service = RoleBasedRedactionService()

    return _redaction_policies_service


def reset_redaction_policies_service() -> None:
    """Reset the global redaction policies service instance."""
    global _redaction_policies_service
    _redaction_policies_service = None
