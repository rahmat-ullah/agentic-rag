"""
Compliance Checking Service

This module implements comprehensive compliance checking capabilities including:
- Rule definition and management
- Automated compliance assessment
- Gap analysis and reporting
- Compliance scoring and metrics
- Standards validation
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import uuid4
from enum import Enum
from dataclasses import dataclass
import structlog
from pydantic import BaseModel, Field
from datetime import datetime

logger = structlog.get_logger(__name__)


class ComplianceLevel(str, Enum):
    """Compliance assessment levels."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class RuleType(str, Enum):
    """Types of compliance rules."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"
    INFORMATIONAL = "informational"


class AssessmentMethod(str, Enum):
    """Methods for compliance assessment."""
    PATTERN_MATCH = "pattern_match"
    KEYWORD_SEARCH = "keyword_search"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    NUMERICAL_CHECK = "numerical_check"
    DATE_VALIDATION = "date_validation"
    CUSTOM_FUNCTION = "custom_function"


class Severity(str, Enum):
    """Severity levels for compliance issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ComplianceRule:
    """Represents a compliance rule."""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    assessment_method: AssessmentMethod
    pattern: Optional[str]
    keywords: List[str]
    severity: Severity
    weight: float
    category: str
    standard: str
    section: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ComplianceGap:
    """Represents a compliance gap or issue."""
    gap_id: str
    rule_id: str
    rule_name: str
    description: str
    severity: Severity
    location: str
    evidence: str
    recommendation: str
    confidence: float
    metadata: Dict[str, Any]


class ComplianceAssessment(BaseModel):
    """Result of compliance assessment."""
    assessment_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: Optional[str] = None
    content_hash: str
    overall_score: float
    compliance_level: ComplianceLevel
    total_rules: int
    compliant_rules: int
    non_compliant_rules: int
    gaps: List[ComplianceGap]
    rule_results: Dict[str, Dict[str, Any]]
    assessment_date: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float
    metadata: Dict[str, Any]


class ComplianceReport(BaseModel):
    """Comprehensive compliance report."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    assessment_id: str
    executive_summary: str
    compliance_breakdown: Dict[str, Any]
    gap_analysis: Dict[str, Any]
    recommendations: List[str]
    action_items: List[Dict[str, Any]]
    compliance_trends: Dict[str, Any]
    generated_date: datetime = Field(default_factory=datetime.utcnow)


class ComplianceConfig(BaseModel):
    """Configuration for compliance checking."""
    default_standard: str = "ISO_9001"
    min_confidence_threshold: float = 0.7
    enable_semantic_analysis: bool = True
    max_gaps_per_rule: int = 10
    enable_recommendations: bool = True
    cache_assessments: bool = True
    assessment_timeout_seconds: int = 300


class ComplianceCheckingService:
    """Service for comprehensive compliance checking."""
    
    def __init__(self, config: Optional[ComplianceConfig] = None):
        self.config = config or ComplianceConfig()
        
        # Rule registry
        self._rules: Dict[str, ComplianceRule] = {}
        self._rules_by_standard: Dict[str, List[str]] = {}
        self._rules_by_category: Dict[str, List[str]] = {}
        
        # Statistics tracking
        self._stats = {
            "total_assessments": 0,
            "successful_assessments": 0,
            "failed_assessments": 0,
            "total_processing_time_ms": 0.0,
            "average_processing_time_ms": 0.0,
            "total_rules_evaluated": 0,
            "total_gaps_found": 0,
            "average_compliance_score": 0.0
        }
        
        # Cache for repeated assessments
        self._assessment_cache: Dict[str, ComplianceAssessment] = {}
        
        # Load default rules
        self._load_default_rules()
        
        logger.info("Compliance checking service initialized")
    
    def _load_default_rules(self) -> None:
        """Load default compliance rules."""
        
        # ISO 9001 Quality Management System rules
        iso_rules = [
            ComplianceRule(
                rule_id="ISO_9001_4.1",
                name="Context of Organization",
                description="Organization must understand its context and interested parties",
                rule_type=RuleType.MANDATORY,
                assessment_method=AssessmentMethod.KEYWORD_SEARCH,
                pattern=None,
                keywords=["organization", "context", "stakeholder", "interested parties", "scope"],
                severity=Severity.HIGH,
                weight=0.8,
                category="organizational_context",
                standard="ISO_9001",
                section="4.1",
                metadata={}
            ),
            ComplianceRule(
                rule_id="ISO_9001_5.1",
                name="Leadership and Commitment",
                description="Top management must demonstrate leadership and commitment",
                rule_type=RuleType.MANDATORY,
                assessment_method=AssessmentMethod.KEYWORD_SEARCH,
                pattern=None,
                keywords=["leadership", "management", "commitment", "responsibility", "accountability"],
                severity=Severity.CRITICAL,
                weight=1.0,
                category="leadership",
                standard="ISO_9001",
                section="5.1",
                metadata={}
            ),
            ComplianceRule(
                rule_id="ISO_9001_6.1",
                name="Risk Management",
                description="Organization must address risks and opportunities",
                rule_type=RuleType.MANDATORY,
                assessment_method=AssessmentMethod.KEYWORD_SEARCH,
                pattern=None,
                keywords=["risk", "opportunity", "mitigation", "assessment", "management"],
                severity=Severity.HIGH,
                weight=0.9,
                category="risk_management",
                standard="ISO_9001",
                section="6.1",
                metadata={}
            ),
            ComplianceRule(
                rule_id="ISO_9001_8.1",
                name="Operational Planning",
                description="Organization must plan and control operational processes",
                rule_type=RuleType.MANDATORY,
                assessment_method=AssessmentMethod.KEYWORD_SEARCH,
                pattern=None,
                keywords=["planning", "process", "control", "operation", "procedure"],
                severity=Severity.MEDIUM,
                weight=0.7,
                category="operations",
                standard="ISO_9001",
                section="8.1",
                metadata={}
            ),
            ComplianceRule(
                rule_id="ISO_9001_9.1",
                name="Monitoring and Measurement",
                description="Organization must monitor and measure performance",
                rule_type=RuleType.MANDATORY,
                assessment_method=AssessmentMethod.KEYWORD_SEARCH,
                pattern=None,
                keywords=["monitoring", "measurement", "performance", "metrics", "kpi"],
                severity=Severity.MEDIUM,
                weight=0.6,
                category="monitoring",
                standard="ISO_9001",
                section="9.1",
                metadata={}
            )
        ]
        
        # Procurement compliance rules
        procurement_rules = [
            ComplianceRule(
                rule_id="PROC_001",
                name="Vendor Qualification",
                description="All vendors must be properly qualified and assessed",
                rule_type=RuleType.MANDATORY,
                assessment_method=AssessmentMethod.KEYWORD_SEARCH,
                pattern=None,
                keywords=["vendor", "supplier", "qualification", "assessment", "evaluation"],
                severity=Severity.HIGH,
                weight=0.8,
                category="vendor_management",
                standard="PROCUREMENT",
                section="001",
                metadata={}
            ),
            ComplianceRule(
                rule_id="PROC_002",
                name="Cost Justification",
                description="All procurement decisions must include cost justification",
                rule_type=RuleType.MANDATORY,
                assessment_method=AssessmentMethod.KEYWORD_SEARCH,
                pattern=None,
                keywords=["cost", "price", "justification", "budget", "value"],
                severity=Severity.MEDIUM,
                weight=0.7,
                category="cost_management",
                standard="PROCUREMENT",
                section="002",
                metadata={}
            ),
            ComplianceRule(
                rule_id="PROC_003",
                name="Contract Terms",
                description="Contracts must include standard terms and conditions",
                rule_type=RuleType.MANDATORY,
                assessment_method=AssessmentMethod.KEYWORD_SEARCH,
                pattern=None,
                keywords=["contract", "terms", "conditions", "agreement", "clause"],
                severity=Severity.HIGH,
                weight=0.9,
                category="contract_management",
                standard="PROCUREMENT",
                section="003",
                metadata={}
            )
        ]
        
        # Register all rules
        all_rules = iso_rules + procurement_rules
        for rule in all_rules:
            self.register_rule(rule)
    
    def register_rule(self, rule: ComplianceRule) -> None:
        """Register a compliance rule."""
        
        self._rules[rule.rule_id] = rule
        
        # Index by standard
        if rule.standard not in self._rules_by_standard:
            self._rules_by_standard[rule.standard] = []
        self._rules_by_standard[rule.standard].append(rule.rule_id)
        
        # Index by category
        if rule.category not in self._rules_by_category:
            self._rules_by_category[rule.category] = []
        self._rules_by_category[rule.category].append(rule.rule_id)
        
        logger.debug(f"Registered compliance rule: {rule.rule_id}")
    
    async def assess_compliance(self, content: str, 
                              standard: Optional[str] = None,
                              categories: Optional[List[str]] = None) -> ComplianceAssessment:
        """Assess compliance of content against rules."""
        
        start_time = time.time()
        
        try:
            # Use default standard if not specified
            standard = standard or self.config.default_standard
            
            # Generate cache key
            content_hash = str(hash(content))
            cache_key = f"{content_hash}:{standard}:{','.join(categories or [])}"
            
            # Check cache
            if self.config.cache_assessments and cache_key in self._assessment_cache:
                logger.info("Returning cached compliance assessment")
                return self._assessment_cache[cache_key]
            
            # Get applicable rules
            applicable_rules = self._get_applicable_rules(standard, categories)
            
            # Assess each rule
            rule_results = {}
            gaps = []
            compliant_count = 0
            
            for rule_id in applicable_rules:
                rule = self._rules[rule_id]
                result = await self._assess_rule(content, rule)
                rule_results[rule_id] = result
                
                if result['compliance_level'] == ComplianceLevel.COMPLIANT:
                    compliant_count += 1
                elif result['compliance_level'] == ComplianceLevel.NON_COMPLIANT:
                    # Create gap for non-compliant rules
                    gap = ComplianceGap(
                        gap_id=str(uuid4()),
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        description=f"Non-compliance with {rule.name}: {rule.description}",
                        severity=rule.severity,
                        location=result.get('location', 'Unknown'),
                        evidence=result.get('evidence', 'No evidence found'),
                        recommendation=self._generate_recommendation(rule),
                        confidence=result.get('confidence', 0.5),
                        metadata=result.get('metadata', {})
                    )
                    gaps.append(gap)
            
            # Calculate overall compliance
            total_rules = len(applicable_rules)
            compliance_score = compliant_count / total_rules if total_rules > 0 else 0.0
            
            # Determine compliance level
            if compliance_score >= 0.9:
                compliance_level = ComplianceLevel.COMPLIANT
            elif compliance_score >= 0.7:
                compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT
            else:
                compliance_level = ComplianceLevel.NON_COMPLIANT
            
            # Create assessment
            processing_time_ms = (time.time() - start_time) * 1000
            
            assessment = ComplianceAssessment(
                content_hash=content_hash,
                overall_score=compliance_score,
                compliance_level=compliance_level,
                total_rules=total_rules,
                compliant_rules=compliant_count,
                non_compliant_rules=total_rules - compliant_count,
                gaps=gaps,
                rule_results=rule_results,
                processing_time_ms=processing_time_ms,
                metadata={
                    "standard": standard,
                    "categories": categories,
                    "content_length": len(content)
                }
            )
            
            # Cache result
            if self.config.cache_assessments:
                self._assessment_cache[cache_key] = assessment
            
            # Update statistics
            self._update_stats("success", processing_time_ms, total_rules, len(gaps), compliance_score)
            
            logger.info(
                "Compliance assessment completed",
                standard=standard,
                compliance_score=compliance_score,
                gaps_found=len(gaps),
                processing_time_ms=processing_time_ms
            )
            
            return assessment
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats("failure", processing_time_ms, 0, 0, 0)
            
            logger.error(
                "Compliance assessment failed",
                error=str(e),
                standard=standard,
                processing_time_ms=processing_time_ms
            )
            raise

    def _get_applicable_rules(self, standard: str, categories: Optional[List[str]] = None) -> List[str]:
        """Get applicable rules for assessment."""

        applicable_rules = []

        # Get rules by standard
        if standard in self._rules_by_standard:
            applicable_rules.extend(self._rules_by_standard[standard])

        # Filter by categories if specified
        if categories:
            category_rules = []
            for category in categories:
                if category in self._rules_by_category:
                    category_rules.extend(self._rules_by_category[category])

            # Intersection of standard and category rules
            if category_rules:
                applicable_rules = list(set(applicable_rules) & set(category_rules))

        return applicable_rules

    async def _assess_rule(self, content: str, rule: ComplianceRule) -> Dict[str, Any]:
        """Assess a single compliance rule against content."""

        result = {
            'rule_id': rule.rule_id,
            'compliance_level': ComplianceLevel.UNKNOWN,
            'confidence': 0.0,
            'evidence': '',
            'location': '',
            'metadata': {}
        }

        try:
            if rule.assessment_method == AssessmentMethod.KEYWORD_SEARCH:
                result = await self._assess_keyword_rule(content, rule)
            elif rule.assessment_method == AssessmentMethod.PATTERN_MATCH:
                result = await self._assess_pattern_rule(content, rule)
            elif rule.assessment_method == AssessmentMethod.SEMANTIC_ANALYSIS:
                result = await self._assess_semantic_rule(content, rule)
            elif rule.assessment_method == AssessmentMethod.NUMERICAL_CHECK:
                result = await self._assess_numerical_rule(content, rule)
            else:
                # Default to keyword search
                result = await self._assess_keyword_rule(content, rule)

        except Exception as e:
            logger.warning(f"Rule assessment failed for {rule.rule_id}: {e}")
            result['compliance_level'] = ComplianceLevel.UNKNOWN
            result['confidence'] = 0.0

        return result

    async def _assess_keyword_rule(self, content: str, rule: ComplianceRule) -> Dict[str, Any]:
        """Assess rule using keyword search."""

        content_lower = content.lower()
        found_keywords = []
        evidence_snippets = []

        for keyword in rule.keywords:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)

                # Extract evidence snippet
                start_idx = content_lower.find(keyword.lower())
                snippet_start = max(0, start_idx - 50)
                snippet_end = min(len(content), start_idx + len(keyword) + 50)
                snippet = content[snippet_start:snippet_end].strip()
                evidence_snippets.append(snippet)

        # Calculate compliance based on keyword coverage
        keyword_coverage = len(found_keywords) / len(rule.keywords) if rule.keywords else 0

        if keyword_coverage >= 0.8:
            compliance_level = ComplianceLevel.COMPLIANT
            confidence = min(1.0, keyword_coverage)
        elif keyword_coverage >= 0.5:
            compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT
            confidence = keyword_coverage * 0.8
        else:
            compliance_level = ComplianceLevel.NON_COMPLIANT
            confidence = keyword_coverage * 0.6

        return {
            'rule_id': rule.rule_id,
            'compliance_level': compliance_level,
            'confidence': confidence,
            'evidence': '; '.join(evidence_snippets[:3]),  # Limit evidence
            'location': f"Keywords found: {', '.join(found_keywords)}",
            'metadata': {
                'found_keywords': found_keywords,
                'keyword_coverage': keyword_coverage,
                'total_keywords': len(rule.keywords)
            }
        }

    async def _assess_pattern_rule(self, content: str, rule: ComplianceRule) -> Dict[str, Any]:
        """Assess rule using pattern matching."""

        if not rule.pattern:
            return await self._assess_keyword_rule(content, rule)

        try:
            matches = re.finditer(rule.pattern, content, re.IGNORECASE | re.MULTILINE)
            match_list = list(matches)

            if match_list:
                compliance_level = ComplianceLevel.COMPLIANT
                confidence = min(1.0, len(match_list) / 3)  # Normalize by expected matches

                # Extract evidence from first few matches
                evidence_snippets = []
                for match in match_list[:3]:
                    start = max(0, match.start() - 30)
                    end = min(len(content), match.end() + 30)
                    snippet = content[start:end].strip()
                    evidence_snippets.append(snippet)

                evidence = '; '.join(evidence_snippets)
                location = f"Pattern matches at positions: {[m.start() for m in match_list[:5]]}"
            else:
                compliance_level = ComplianceLevel.NON_COMPLIANT
                confidence = 0.0
                evidence = "No pattern matches found"
                location = "No matches"

            return {
                'rule_id': rule.rule_id,
                'compliance_level': compliance_level,
                'confidence': confidence,
                'evidence': evidence,
                'location': location,
                'metadata': {
                    'pattern': rule.pattern,
                    'match_count': len(match_list),
                    'match_positions': [m.start() for m in match_list]
                }
            }

        except re.error as e:
            logger.warning(f"Invalid regex pattern for rule {rule.rule_id}: {e}")
            return await self._assess_keyword_rule(content, rule)

    async def _assess_semantic_rule(self, content: str, rule: ComplianceRule) -> Dict[str, Any]:
        """Assess rule using semantic analysis."""

        # For now, fall back to keyword search
        # In a full implementation, this would use NLP/LLM for semantic understanding
        result = await self._assess_keyword_rule(content, rule)
        result['metadata']['assessment_method'] = 'semantic_fallback'
        return result

    async def _assess_numerical_rule(self, content: str, rule: ComplianceRule) -> Dict[str, Any]:
        """Assess rule using numerical checks."""

        # Extract numbers from content
        numbers = re.findall(r'\d+(?:\.\d+)?', content)

        if numbers:
            compliance_level = ComplianceLevel.COMPLIANT
            confidence = 0.8
            evidence = f"Found {len(numbers)} numerical values"
            location = f"Numbers: {', '.join(numbers[:5])}"
        else:
            compliance_level = ComplianceLevel.NON_COMPLIANT
            confidence = 0.0
            evidence = "No numerical values found"
            location = "No numbers"

        return {
            'rule_id': rule.rule_id,
            'compliance_level': compliance_level,
            'confidence': confidence,
            'evidence': evidence,
            'location': location,
            'metadata': {
                'number_count': len(numbers),
                'numbers_found': numbers[:10]  # Limit to first 10
            }
        }

    def _generate_recommendation(self, rule: ComplianceRule) -> str:
        """Generate recommendation for non-compliant rule."""

        recommendations = {
            "organizational_context": "Clearly define the organization's context, scope, and interested parties in the documentation.",
            "leadership": "Include explicit statements of management commitment and leadership responsibilities.",
            "risk_management": "Add comprehensive risk assessment and mitigation strategies.",
            "operations": "Document operational planning and control procedures.",
            "monitoring": "Implement monitoring and measurement systems with defined metrics.",
            "vendor_management": "Ensure all vendors are properly qualified and assessed before engagement.",
            "cost_management": "Provide detailed cost justification and budget analysis.",
            "contract_management": "Include all required contract terms and conditions."
        }

        return recommendations.get(rule.category, f"Address the requirements for {rule.name} as specified in {rule.standard} section {rule.section}.")

    def _update_stats(self, result: str, processing_time_ms: float,
                     rules_evaluated: int, gaps_found: int, compliance_score: float) -> None:
        """Update service statistics."""

        self._stats["total_assessments"] += 1
        self._stats["total_processing_time_ms"] += processing_time_ms
        self._stats["total_rules_evaluated"] += rules_evaluated
        self._stats["total_gaps_found"] += gaps_found

        if result == "success":
            self._stats["successful_assessments"] += 1

            # Update average compliance score
            current_avg = self._stats.get("average_compliance_score", 0)
            total_successful = self._stats["successful_assessments"]
            self._stats["average_compliance_score"] = (
                (current_avg * (total_successful - 1) + compliance_score) / total_successful
            )
        else:
            self._stats["failed_assessments"] += 1

        # Update averages
        if self._stats["total_assessments"] > 0:
            self._stats["average_processing_time_ms"] = (
                self._stats["total_processing_time_ms"] / self._stats["total_assessments"]
            )

    async def generate_compliance_report(self, assessment: ComplianceAssessment) -> ComplianceReport:
        """Generate comprehensive compliance report."""

        # Executive summary
        executive_summary = self._generate_executive_summary(assessment)

        # Compliance breakdown
        compliance_breakdown = self._generate_compliance_breakdown(assessment)

        # Gap analysis
        gap_analysis = self._generate_gap_analysis(assessment)

        # Recommendations
        recommendations = self._generate_recommendations(assessment)

        # Action items
        action_items = self._generate_action_items(assessment)

        report = ComplianceReport(
            assessment_id=assessment.assessment_id,
            executive_summary=executive_summary,
            compliance_breakdown=compliance_breakdown,
            gap_analysis=gap_analysis,
            recommendations=recommendations,
            action_items=action_items,
            compliance_trends={}  # Would be populated with historical data
        )

        return report

    def _generate_executive_summary(self, assessment: ComplianceAssessment) -> str:
        """Generate executive summary for compliance report."""

        summary = f"""
        Compliance Assessment Summary

        Overall Compliance Score: {assessment.overall_score:.1%}
        Compliance Level: {assessment.compliance_level.value.replace('_', ' ').title()}

        Total Rules Evaluated: {assessment.total_rules}
        Compliant Rules: {assessment.compliant_rules}
        Non-Compliant Rules: {assessment.non_compliant_rules}

        Critical Issues Found: {len([g for g in assessment.gaps if g.severity == Severity.CRITICAL])}
        High Priority Issues: {len([g for g in assessment.gaps if g.severity == Severity.HIGH])}

        Assessment completed in {assessment.processing_time_ms:.0f}ms
        """

        return summary.strip()

    def _generate_compliance_breakdown(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate compliance breakdown by category and severity."""

        breakdown = {
            "by_severity": {},
            "by_category": {},
            "by_standard": {}
        }

        # Breakdown by severity
        for severity in Severity:
            gaps = [g for g in assessment.gaps if g.severity == severity]
            breakdown["by_severity"][severity.value] = {
                "count": len(gaps),
                "percentage": len(gaps) / len(assessment.gaps) if assessment.gaps else 0
            }

        # Breakdown by category (from rule metadata)
        categories = {}
        for rule_id, result in assessment.rule_results.items():
            if rule_id in self._rules:
                rule = self._rules[rule_id]
                category = rule.category
                if category not in categories:
                    categories[category] = {"compliant": 0, "non_compliant": 0}

                if result['compliance_level'] == ComplianceLevel.COMPLIANT:
                    categories[category]["compliant"] += 1
                else:
                    categories[category]["non_compliant"] += 1

        breakdown["by_category"] = categories

        return breakdown

    def _generate_gap_analysis(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate detailed gap analysis."""

        gap_analysis = {
            "total_gaps": len(assessment.gaps),
            "critical_gaps": [g for g in assessment.gaps if g.severity == Severity.CRITICAL],
            "high_priority_gaps": [g for g in assessment.gaps if g.severity == Severity.HIGH],
            "gap_summary": {}
        }

        # Summarize gaps by category
        gap_categories = {}
        for gap in assessment.gaps:
            if gap.rule_id in self._rules:
                rule = self._rules[gap.rule_id]
                category = rule.category
                if category not in gap_categories:
                    gap_categories[category] = []
                gap_categories[category].append(gap)

        gap_analysis["gap_summary"] = gap_categories

        return gap_analysis

    def _generate_recommendations(self, assessment: ComplianceAssessment) -> List[str]:
        """Generate recommendations based on assessment."""

        recommendations = []

        # Priority recommendations for critical gaps
        critical_gaps = [g for g in assessment.gaps if g.severity == Severity.CRITICAL]
        if critical_gaps:
            recommendations.append("Address critical compliance gaps immediately to avoid regulatory issues.")

        # Category-specific recommendations
        gap_categories = {}
        for gap in assessment.gaps:
            if gap.rule_id in self._rules:
                rule = self._rules[gap.rule_id]
                category = rule.category
                if category not in gap_categories:
                    gap_categories[category] = 0
                gap_categories[category] += 1

        for category, count in gap_categories.items():
            if count > 1:
                recommendations.append(f"Focus on improving {category.replace('_', ' ')} compliance ({count} gaps identified).")

        return recommendations

    def _generate_action_items(self, assessment: ComplianceAssessment) -> List[Dict[str, Any]]:
        """Generate action items based on gaps."""

        action_items = []

        for gap in assessment.gaps:
            priority = "High" if gap.severity in [Severity.CRITICAL, Severity.HIGH] else "Medium"

            action_item = {
                "id": gap.gap_id,
                "title": f"Address {gap.rule_name}",
                "description": gap.description,
                "recommendation": gap.recommendation,
                "priority": priority,
                "estimated_effort": "TBD",
                "assigned_to": "TBD",
                "due_date": "TBD"
            }

            action_items.append(action_item)

        return action_items

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""

        return {
            **self._stats,
            "total_rules": len(self._rules),
            "rules_by_standard": {k: len(v) for k, v in self._rules_by_standard.items()},
            "rules_by_category": {k: len(v) for k, v in self._rules_by_category.items()},
            "cache_size": len(self._assessment_cache),
            "config": self.config.dict()
        }

    def clear_cache(self) -> None:
        """Clear assessment cache."""

        self._assessment_cache.clear()
        logger.info("Compliance assessment cache cleared")


# Global service instance
_compliance_checking_service: Optional[ComplianceCheckingService] = None


def get_compliance_checking_service() -> ComplianceCheckingService:
    """Get or create the global compliance checking service instance."""
    global _compliance_checking_service

    if _compliance_checking_service is None:
        _compliance_checking_service = ComplianceCheckingService()

    return _compliance_checking_service


def reset_compliance_checking_service() -> None:
    """Reset the global compliance checking service instance."""
    global _compliance_checking_service
    _compliance_checking_service = None
