"""
Risk Assessment and Identification Service

This module implements comprehensive risk assessment capabilities including:
- Pattern recognition for risk identification
- Risk categorization and classification
- Risk scoring and impact assessment
- Mitigation suggestions and strategies
- Risk reporting and monitoring
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
import math

logger = structlog.get_logger(__name__)


class RiskCategory(str, Enum):
    """Categories of risks."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    LEGAL = "legal"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    VENDOR = "vendor"
    MARKET = "market"
    SCHEDULE = "schedule"
    QUALITY = "quality"
    REPUTATION = "reputation"
    STRATEGIC = "strategic"


class RiskLevel(str, Enum):
    """Risk severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class RiskLikelihood(str, Enum):
    """Risk likelihood levels."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class RiskImpact(str, Enum):
    """Risk impact levels."""
    CATASTROPHIC = "catastrophic"
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"
    NEGLIGIBLE = "negligible"


class RiskStatus(str, Enum):
    """Risk status."""
    IDENTIFIED = "identified"
    ASSESSED = "assessed"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    TRANSFERRED = "transferred"
    AVOIDED = "avoided"


@dataclass
class RiskPattern:
    """Pattern for risk identification."""
    pattern_id: str
    name: str
    description: str
    category: RiskCategory
    keywords: List[str]
    regex_patterns: List[str]
    weight: float
    confidence_threshold: float


@dataclass
class MitigationStrategy:
    """Risk mitigation strategy."""
    strategy_id: str
    name: str
    description: str
    category: str
    effectiveness: float
    cost: str
    timeline: str
    resources_required: List[str]


class IdentifiedRisk(BaseModel):
    """Represents an identified risk."""
    risk_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    category: RiskCategory
    likelihood: RiskLikelihood
    impact: RiskImpact
    risk_level: RiskLevel
    risk_score: float
    confidence: float
    evidence: str
    location: str
    mitigation_strategies: List[MitigationStrategy]
    status: RiskStatus = RiskStatus.IDENTIFIED
    identified_date: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any]


class RiskAssessmentResult(BaseModel):
    """Result of risk assessment."""
    assessment_id: str = Field(default_factory=lambda: str(uuid4()))
    content_hash: str
    overall_risk_score: float
    overall_risk_level: RiskLevel
    total_risks: int
    risks_by_category: Dict[RiskCategory, int]
    risks_by_level: Dict[RiskLevel, int]
    identified_risks: List[IdentifiedRisk]
    assessment_date: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float
    metadata: Dict[str, Any]


class RiskReport(BaseModel):
    """Comprehensive risk report."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    assessment_id: str
    executive_summary: str
    risk_matrix: Dict[str, Any]
    top_risks: List[IdentifiedRisk]
    mitigation_plan: Dict[str, Any]
    recommendations: List[str]
    action_items: List[Dict[str, Any]]
    generated_date: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessmentConfig(BaseModel):
    """Configuration for risk assessment."""
    min_confidence_threshold: float = 0.6
    max_risks_per_category: int = 20
    enable_mitigation_suggestions: bool = True
    risk_score_weights: Dict[str, float] = Field(default_factory=lambda: {
        "likelihood": 0.4,
        "impact": 0.4,
        "confidence": 0.2
    })
    cache_assessments: bool = True


class RiskAssessmentService:
    """Service for comprehensive risk assessment and identification."""
    
    def __init__(self, config: Optional[RiskAssessmentConfig] = None):
        self.config = config or RiskAssessmentConfig()
        
        # Risk patterns registry
        self._risk_patterns: Dict[str, RiskPattern] = {}
        self._patterns_by_category: Dict[RiskCategory, List[str]] = {}
        
        # Mitigation strategies registry
        self._mitigation_strategies: Dict[str, MitigationStrategy] = {}
        
        # Statistics tracking
        self._stats = {
            "total_assessments": 0,
            "successful_assessments": 0,
            "failed_assessments": 0,
            "total_processing_time_ms": 0.0,
            "average_processing_time_ms": 0.0,
            "total_risks_identified": 0,
            "risks_by_category": {},
            "average_risk_score": 0.0
        }
        
        # Cache for repeated assessments
        self._assessment_cache: Dict[str, RiskAssessmentResult] = {}
        
        # Load default patterns and strategies
        self._load_default_patterns()
        self._load_default_mitigation_strategies()
        
        logger.info("Risk assessment service initialized")
    
    def _load_default_patterns(self) -> None:
        """Load default risk identification patterns."""
        
        patterns = [
            # Financial risks
            RiskPattern(
                pattern_id="FIN_001",
                name="Budget Overrun Risk",
                description="Risk of exceeding allocated budget",
                category=RiskCategory.FINANCIAL,
                keywords=["budget", "cost", "overrun", "exceed", "expensive", "price increase"],
                regex_patterns=[r"budget.*exceed", r"cost.*overrun", r"\$[\d,]+.*over"],
                weight=0.8,
                confidence_threshold=0.7
            ),
            RiskPattern(
                pattern_id="FIN_002",
                name="Payment Delay Risk",
                description="Risk of delayed payments or cash flow issues",
                category=RiskCategory.FINANCIAL,
                keywords=["payment", "delay", "cash flow", "invoice", "late payment"],
                regex_patterns=[r"payment.*delay", r"cash.*flow.*issue"],
                weight=0.7,
                confidence_threshold=0.6
            ),
            
            # Operational risks
            RiskPattern(
                pattern_id="OPS_001",
                name="Delivery Delay Risk",
                description="Risk of delayed delivery or schedule slippage",
                category=RiskCategory.OPERATIONAL,
                keywords=["delay", "schedule", "timeline", "delivery", "late", "behind"],
                regex_patterns=[r"delivery.*delay", r"schedule.*slip", r"behind.*schedule"],
                weight=0.8,
                confidence_threshold=0.7
            ),
            RiskPattern(
                pattern_id="OPS_002",
                name="Resource Availability Risk",
                description="Risk of insufficient resources or capacity",
                category=RiskCategory.OPERATIONAL,
                keywords=["resource", "capacity", "shortage", "unavailable", "insufficient"],
                regex_patterns=[r"resource.*shortage", r"capacity.*issue"],
                weight=0.7,
                confidence_threshold=0.6
            ),
            
            # Technical risks
            RiskPattern(
                pattern_id="TECH_001",
                name="Technical Complexity Risk",
                description="Risk from technical complexity or integration challenges",
                category=RiskCategory.TECHNICAL,
                keywords=["complex", "integration", "technical", "compatibility", "system"],
                regex_patterns=[r"technical.*complex", r"integration.*challenge"],
                weight=0.8,
                confidence_threshold=0.7
            ),
            RiskPattern(
                pattern_id="TECH_002",
                name="Technology Obsolescence Risk",
                description="Risk of technology becoming obsolete",
                category=RiskCategory.TECHNICAL,
                keywords=["obsolete", "outdated", "legacy", "deprecated", "end of life"],
                regex_patterns=[r"end.*of.*life", r"legacy.*system"],
                weight=0.6,
                confidence_threshold=0.6
            ),
            
            # Vendor risks
            RiskPattern(
                pattern_id="VEND_001",
                name="Vendor Reliability Risk",
                description="Risk related to vendor reliability or performance",
                category=RiskCategory.VENDOR,
                keywords=["vendor", "supplier", "reliability", "performance", "quality"],
                regex_patterns=[r"vendor.*issue", r"supplier.*problem"],
                weight=0.8,
                confidence_threshold=0.7
            ),
            RiskPattern(
                pattern_id="VEND_002",
                name="Single Source Risk",
                description="Risk from dependency on single vendor",
                category=RiskCategory.VENDOR,
                keywords=["single source", "sole supplier", "dependency", "monopoly"],
                regex_patterns=[r"single.*source", r"sole.*supplier"],
                weight=0.9,
                confidence_threshold=0.8
            ),
            
            # Compliance risks
            RiskPattern(
                pattern_id="COMP_001",
                name="Regulatory Compliance Risk",
                description="Risk of non-compliance with regulations",
                category=RiskCategory.COMPLIANCE,
                keywords=["regulation", "compliance", "legal", "requirement", "standard"],
                regex_patterns=[r"regulatory.*requirement", r"compliance.*issue"],
                weight=0.9,
                confidence_threshold=0.8
            ),
            
            # Security risks
            RiskPattern(
                pattern_id="SEC_001",
                name="Data Security Risk",
                description="Risk to data security and privacy",
                category=RiskCategory.SECURITY,
                keywords=["security", "privacy", "data", "breach", "confidential"],
                regex_patterns=[r"security.*risk", r"data.*breach"],
                weight=0.9,
                confidence_threshold=0.8
            )
        ]
        
        for pattern in patterns:
            self.register_risk_pattern(pattern)
    
    def _load_default_mitigation_strategies(self) -> None:
        """Load default mitigation strategies."""
        
        strategies = [
            MitigationStrategy(
                strategy_id="MIT_001",
                name="Budget Monitoring",
                description="Implement regular budget monitoring and controls",
                category="financial",
                effectiveness=0.8,
                cost="Low",
                timeline="1-2 weeks",
                resources_required=["Finance team", "Monitoring tools"]
            ),
            MitigationStrategy(
                strategy_id="MIT_002",
                name="Vendor Diversification",
                description="Diversify vendor base to reduce single-source dependency",
                category="vendor",
                effectiveness=0.9,
                cost="Medium",
                timeline="3-6 months",
                resources_required=["Procurement team", "Vendor assessment"]
            ),
            MitigationStrategy(
                strategy_id="MIT_003",
                name="Schedule Buffer",
                description="Add buffer time to critical path activities",
                category="operational",
                effectiveness=0.7,
                cost="Low",
                timeline="Immediate",
                resources_required=["Project management"]
            ),
            MitigationStrategy(
                strategy_id="MIT_004",
                name="Technical Review",
                description="Conduct thorough technical review and testing",
                category="technical",
                effectiveness=0.8,
                cost="Medium",
                timeline="2-4 weeks",
                resources_required=["Technical team", "Testing resources"]
            ),
            MitigationStrategy(
                strategy_id="MIT_005",
                name="Compliance Audit",
                description="Regular compliance audits and assessments",
                category="compliance",
                effectiveness=0.9,
                cost="Medium",
                timeline="Ongoing",
                resources_required=["Compliance team", "External auditors"]
            )
        ]
        
        for strategy in strategies:
            self._mitigation_strategies[strategy.strategy_id] = strategy

    def register_risk_pattern(self, pattern: RiskPattern) -> None:
        """Register a risk identification pattern."""

        self._risk_patterns[pattern.pattern_id] = pattern

        # Index by category
        if pattern.category not in self._patterns_by_category:
            self._patterns_by_category[pattern.category] = []
        self._patterns_by_category[pattern.category].append(pattern.pattern_id)

        logger.debug(f"Registered risk pattern: {pattern.pattern_id}")

    async def assess_risks(self, content: str,
                          categories: Optional[List[RiskCategory]] = None) -> RiskAssessmentResult:
        """Assess risks in content."""

        start_time = time.time()

        try:
            # Generate cache key
            content_hash = str(hash(content))
            cache_key = f"{content_hash}:{','.join([c.value for c in categories] if categories else [])}"

            # Check cache
            if self.config.cache_assessments and cache_key in self._assessment_cache:
                logger.info("Returning cached risk assessment")
                return self._assessment_cache[cache_key]

            # Get applicable patterns
            applicable_patterns = self._get_applicable_patterns(categories)

            # Identify risks
            identified_risks = []

            for pattern_id in applicable_patterns:
                pattern = self._risk_patterns[pattern_id]
                risks = await self._identify_risks_by_pattern(content, pattern)
                identified_risks.extend(risks)

            # Remove duplicates and merge similar risks
            identified_risks = self._deduplicate_risks(identified_risks)

            # Calculate overall risk metrics
            overall_risk_score = self._calculate_overall_risk_score(identified_risks)
            overall_risk_level = self._determine_risk_level(overall_risk_score)

            # Categorize risks
            risks_by_category = {}
            risks_by_level = {}

            for risk in identified_risks:
                # By category
                if risk.category not in risks_by_category:
                    risks_by_category[risk.category] = 0
                risks_by_category[risk.category] += 1

                # By level
                if risk.risk_level not in risks_by_level:
                    risks_by_level[risk.risk_level] = 0
                risks_by_level[risk.risk_level] += 1

            # Create assessment result
            processing_time_ms = (time.time() - start_time) * 1000

            result = RiskAssessmentResult(
                content_hash=content_hash,
                overall_risk_score=overall_risk_score,
                overall_risk_level=overall_risk_level,
                total_risks=len(identified_risks),
                risks_by_category=risks_by_category,
                risks_by_level=risks_by_level,
                identified_risks=identified_risks,
                processing_time_ms=processing_time_ms,
                metadata={
                    "content_length": len(content),
                    "patterns_evaluated": len(applicable_patterns),
                    "categories_requested": [c.value for c in categories] if categories else None
                }
            )

            # Cache result
            if self.config.cache_assessments:
                self._assessment_cache[cache_key] = result

            # Update statistics
            self._update_stats("success", processing_time_ms, len(identified_risks), overall_risk_score)

            logger.info(
                "Risk assessment completed",
                risks_identified=len(identified_risks),
                overall_risk_score=overall_risk_score,
                processing_time_ms=processing_time_ms
            )

            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats("failure", processing_time_ms, 0, 0)

            logger.error(
                "Risk assessment failed",
                error=str(e),
                processing_time_ms=processing_time_ms
            )
            raise

    def _get_applicable_patterns(self, categories: Optional[List[RiskCategory]] = None) -> List[str]:
        """Get applicable risk patterns."""

        if not categories:
            # Return all patterns
            return list(self._risk_patterns.keys())

        applicable_patterns = []
        for category in categories:
            if category in self._patterns_by_category:
                applicable_patterns.extend(self._patterns_by_category[category])

        return list(set(applicable_patterns))  # Remove duplicates

    async def _identify_risks_by_pattern(self, content: str, pattern: RiskPattern) -> List[IdentifiedRisk]:
        """Identify risks using a specific pattern."""

        risks = []
        content_lower = content.lower()

        # Keyword-based detection
        keyword_matches = []
        for keyword in pattern.keywords:
            if keyword.lower() in content_lower:
                keyword_matches.append(keyword)

        # Regex-based detection
        regex_matches = []
        for regex_pattern in pattern.regex_patterns:
            try:
                matches = re.finditer(regex_pattern, content, re.IGNORECASE | re.MULTILINE)
                regex_matches.extend(list(matches))
            except re.error:
                logger.warning(f"Invalid regex pattern: {regex_pattern}")

        # Calculate confidence
        keyword_confidence = len(keyword_matches) / len(pattern.keywords) if pattern.keywords else 0
        regex_confidence = min(1.0, len(regex_matches) / 2) if regex_matches else 0
        overall_confidence = (keyword_confidence + regex_confidence) / 2

        # Check if confidence meets threshold
        if overall_confidence >= pattern.confidence_threshold:
            # Extract evidence
            evidence_snippets = []

            # Evidence from keyword matches
            for keyword in keyword_matches[:3]:
                start_idx = content_lower.find(keyword.lower())
                snippet_start = max(0, start_idx - 50)
                snippet_end = min(len(content), start_idx + len(keyword) + 50)
                snippet = content[snippet_start:snippet_end].strip()
                evidence_snippets.append(snippet)

            # Evidence from regex matches
            for match in regex_matches[:2]:
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 30)
                snippet = content[start:end].strip()
                evidence_snippets.append(snippet)

            evidence = '; '.join(evidence_snippets)

            # Determine risk characteristics
            likelihood = self._assess_likelihood(content, pattern, overall_confidence)
            impact = self._assess_impact(content, pattern, overall_confidence)
            risk_score = self._calculate_risk_score(likelihood, impact, overall_confidence)
            risk_level = self._determine_risk_level(risk_score)

            # Get mitigation strategies
            mitigation_strategies = self._get_mitigation_strategies(pattern.category)

            # Create risk
            risk = IdentifiedRisk(
                name=pattern.name,
                description=pattern.description,
                category=pattern.category,
                likelihood=likelihood,
                impact=impact,
                risk_level=risk_level,
                risk_score=risk_score,
                confidence=overall_confidence,
                evidence=evidence,
                location=f"Pattern: {pattern.pattern_id}",
                mitigation_strategies=mitigation_strategies,
                metadata={
                    "pattern_id": pattern.pattern_id,
                    "keyword_matches": keyword_matches,
                    "regex_match_count": len(regex_matches),
                    "keyword_confidence": keyword_confidence,
                    "regex_confidence": regex_confidence
                }
            )

            risks.append(risk)

        return risks

    def _assess_likelihood(self, content: str, pattern: RiskPattern, confidence: float) -> RiskLikelihood:
        """Assess likelihood of risk occurrence."""

        # Simple heuristic based on confidence and pattern weight
        likelihood_score = confidence * pattern.weight

        if likelihood_score >= 0.8:
            return RiskLikelihood.VERY_HIGH
        elif likelihood_score >= 0.6:
            return RiskLikelihood.HIGH
        elif likelihood_score >= 0.4:
            return RiskLikelihood.MEDIUM
        elif likelihood_score >= 0.2:
            return RiskLikelihood.LOW
        else:
            return RiskLikelihood.VERY_LOW

    def _assess_impact(self, content: str, pattern: RiskPattern, confidence: float) -> RiskImpact:
        """Assess impact of risk if it occurs."""

        # Look for impact indicators in content
        impact_keywords = {
            "catastrophic": ["catastrophic", "disaster", "failure", "collapse"],
            "major": ["major", "significant", "severe", "critical"],
            "moderate": ["moderate", "substantial", "considerable"],
            "minor": ["minor", "small", "limited"],
            "negligible": ["negligible", "minimal", "trivial"]
        }

        content_lower = content.lower()
        impact_scores = {}

        for impact_level, keywords in impact_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            impact_scores[impact_level] = score

        # Find highest scoring impact level
        max_score = max(impact_scores.values()) if impact_scores.values() else 0

        if max_score > 0:
            for impact_level, score in impact_scores.items():
                if score == max_score:
                    return RiskImpact(impact_level)

        # Default based on pattern category
        category_impact_map = {
            RiskCategory.FINANCIAL: RiskImpact.MAJOR,
            RiskCategory.SECURITY: RiskImpact.MAJOR,
            RiskCategory.COMPLIANCE: RiskImpact.MAJOR,
            RiskCategory.OPERATIONAL: RiskImpact.MODERATE,
            RiskCategory.TECHNICAL: RiskImpact.MODERATE,
            RiskCategory.VENDOR: RiskImpact.MODERATE,
            RiskCategory.SCHEDULE: RiskImpact.MINOR,
            RiskCategory.QUALITY: RiskImpact.MODERATE
        }

        return category_impact_map.get(pattern.category, RiskImpact.MODERATE)

    def _calculate_risk_score(self, likelihood: RiskLikelihood, impact: RiskImpact, confidence: float) -> float:
        """Calculate numerical risk score."""

        likelihood_values = {
            RiskLikelihood.VERY_HIGH: 5,
            RiskLikelihood.HIGH: 4,
            RiskLikelihood.MEDIUM: 3,
            RiskLikelihood.LOW: 2,
            RiskLikelihood.VERY_LOW: 1
        }

        impact_values = {
            RiskImpact.CATASTROPHIC: 5,
            RiskImpact.MAJOR: 4,
            RiskImpact.MODERATE: 3,
            RiskImpact.MINOR: 2,
            RiskImpact.NEGLIGIBLE: 1
        }

        likelihood_score = likelihood_values.get(likelihood, 3)
        impact_score = impact_values.get(impact, 3)

        # Calculate weighted score
        weights = self.config.risk_score_weights
        score = (
            likelihood_score * weights.get("likelihood", 0.4) +
            impact_score * weights.get("impact", 0.4) +
            confidence * 5 * weights.get("confidence", 0.2)
        )

        return min(5.0, max(1.0, score))

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from numerical score."""

        if risk_score >= 4.5:
            return RiskLevel.CRITICAL
        elif risk_score >= 3.5:
            return RiskLevel.HIGH
        elif risk_score >= 2.5:
            return RiskLevel.MEDIUM
        elif risk_score >= 1.5:
            return RiskLevel.LOW
        else:
            return RiskLevel.NEGLIGIBLE

    def _get_mitigation_strategies(self, category: RiskCategory) -> List[MitigationStrategy]:
        """Get applicable mitigation strategies for a risk category."""

        category_map = {
            RiskCategory.FINANCIAL: ["financial"],
            RiskCategory.OPERATIONAL: ["operational"],
            RiskCategory.TECHNICAL: ["technical"],
            RiskCategory.VENDOR: ["vendor"],
            RiskCategory.COMPLIANCE: ["compliance"],
            RiskCategory.SECURITY: ["security"]
        }

        applicable_categories = category_map.get(category, [])
        strategies = []

        for strategy in self._mitigation_strategies.values():
            if strategy.category in applicable_categories:
                strategies.append(strategy)

        return strategies[:3]  # Return top 3 strategies

    def _deduplicate_risks(self, risks: List[IdentifiedRisk]) -> List[IdentifiedRisk]:
        """Remove duplicate and merge similar risks."""

        if not risks:
            return risks

        # Group risks by category and similarity
        risk_groups = {}

        for risk in risks:
            key = f"{risk.category.value}_{risk.name}"
            if key not in risk_groups:
                risk_groups[key] = []
            risk_groups[key].append(risk)

        # Merge similar risks
        merged_risks = []

        for group in risk_groups.values():
            if len(group) == 1:
                merged_risks.append(group[0])
            else:
                # Merge multiple similar risks
                merged_risk = self._merge_risks(group)
                merged_risks.append(merged_risk)

        return merged_risks

    def _merge_risks(self, risks: List[IdentifiedRisk]) -> IdentifiedRisk:
        """Merge multiple similar risks into one."""

        # Use the risk with highest confidence as base
        base_risk = max(risks, key=lambda r: r.confidence)

        # Combine evidence
        all_evidence = [r.evidence for r in risks if r.evidence]
        combined_evidence = '; '.join(all_evidence[:3])  # Limit evidence

        # Average confidence
        avg_confidence = sum(r.confidence for r in risks) / len(risks)

        # Take highest risk score
        max_risk_score = max(r.risk_score for r in risks)

        # Create merged risk
        merged_risk = IdentifiedRisk(
            name=base_risk.name,
            description=base_risk.description,
            category=base_risk.category,
            likelihood=base_risk.likelihood,
            impact=base_risk.impact,
            risk_level=self._determine_risk_level(max_risk_score),
            risk_score=max_risk_score,
            confidence=avg_confidence,
            evidence=combined_evidence,
            location=f"Merged from {len(risks)} instances",
            mitigation_strategies=base_risk.mitigation_strategies,
            metadata={
                "merged_from": len(risks),
                "original_risk_ids": [r.risk_id for r in risks]
            }
        )

        return merged_risk

    def _calculate_overall_risk_score(self, risks: List[IdentifiedRisk]) -> float:
        """Calculate overall risk score from all identified risks."""

        if not risks:
            return 0.0

        # Weight risks by their individual scores and confidence
        weighted_scores = []

        for risk in risks:
            weight = risk.confidence * (1.0 if risk.risk_level == RiskLevel.CRITICAL else
                                      0.8 if risk.risk_level == RiskLevel.HIGH else
                                      0.6 if risk.risk_level == RiskLevel.MEDIUM else
                                      0.4 if risk.risk_level == RiskLevel.LOW else 0.2)
            weighted_scores.append(risk.risk_score * weight)

        # Calculate weighted average
        total_weight = sum(risk.confidence for risk in risks)
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
        else:
            overall_score = sum(risk.risk_score for risk in risks) / len(risks)

        return min(5.0, max(0.0, overall_score))

    def _update_stats(self, result: str, processing_time_ms: float,
                     risks_identified: int, overall_risk_score: float) -> None:
        """Update service statistics."""

        self._stats["total_assessments"] += 1
        self._stats["total_processing_time_ms"] += processing_time_ms
        self._stats["total_risks_identified"] += risks_identified

        if result == "success":
            self._stats["successful_assessments"] += 1

            # Update average risk score
            current_avg = self._stats.get("average_risk_score", 0)
            total_successful = self._stats["successful_assessments"]
            self._stats["average_risk_score"] = (
                (current_avg * (total_successful - 1) + overall_risk_score) / total_successful
            )
        else:
            self._stats["failed_assessments"] += 1

        # Update averages
        if self._stats["total_assessments"] > 0:
            self._stats["average_processing_time_ms"] = (
                self._stats["total_processing_time_ms"] / self._stats["total_assessments"]
            )

    async def generate_risk_report(self, assessment: RiskAssessmentResult) -> RiskReport:
        """Generate comprehensive risk report."""

        # Executive summary
        executive_summary = self._generate_executive_summary(assessment)

        # Risk matrix
        risk_matrix = self._generate_risk_matrix(assessment)

        # Top risks
        top_risks = sorted(assessment.identified_risks, key=lambda r: r.risk_score, reverse=True)[:10]

        # Mitigation plan
        mitigation_plan = self._generate_mitigation_plan(assessment)

        # Recommendations
        recommendations = self._generate_recommendations(assessment)

        # Action items
        action_items = self._generate_action_items(assessment)

        report = RiskReport(
            assessment_id=assessment.assessment_id,
            executive_summary=executive_summary,
            risk_matrix=risk_matrix,
            top_risks=top_risks,
            mitigation_plan=mitigation_plan,
            recommendations=recommendations,
            action_items=action_items
        )

        return report

    def _generate_executive_summary(self, assessment: RiskAssessmentResult) -> str:
        """Generate executive summary for risk report."""

        critical_risks = len([r for r in assessment.identified_risks if r.risk_level == RiskLevel.CRITICAL])
        high_risks = len([r for r in assessment.identified_risks if r.risk_level == RiskLevel.HIGH])

        summary = f"""
        Risk Assessment Executive Summary

        Overall Risk Score: {assessment.overall_risk_score:.1f}/5.0
        Overall Risk Level: {assessment.overall_risk_level.value.replace('_', ' ').title()}

        Total Risks Identified: {assessment.total_risks}
        Critical Risks: {critical_risks}
        High Risks: {high_risks}

        Top Risk Categories:
        {self._format_top_categories(assessment.risks_by_category)}

        Assessment completed in {assessment.processing_time_ms:.0f}ms
        """

        return summary.strip()

    def _format_top_categories(self, risks_by_category: Dict[RiskCategory, int]) -> str:
        """Format top risk categories for summary."""

        sorted_categories = sorted(risks_by_category.items(), key=lambda x: x[1], reverse=True)
        lines = []

        for category, count in sorted_categories[:5]:
            lines.append(f"- {category.value.replace('_', ' ').title()}: {count} risks")

        return '\n        '.join(lines)

    def _generate_risk_matrix(self, assessment: RiskAssessmentResult) -> Dict[str, Any]:
        """Generate risk matrix visualization data."""

        matrix = {
            "likelihood_impact_matrix": {},
            "category_distribution": assessment.risks_by_category,
            "level_distribution": assessment.risks_by_level
        }

        # Create likelihood-impact matrix
        for risk in assessment.identified_risks:
            key = f"{risk.likelihood.value}_{risk.impact.value}"
            if key not in matrix["likelihood_impact_matrix"]:
                matrix["likelihood_impact_matrix"][key] = []
            matrix["likelihood_impact_matrix"][key].append({
                "name": risk.name,
                "score": risk.risk_score,
                "level": risk.risk_level.value
            })

        return matrix

    def _generate_mitigation_plan(self, assessment: RiskAssessmentResult) -> Dict[str, Any]:
        """Generate mitigation plan."""

        plan = {
            "immediate_actions": [],
            "short_term_actions": [],
            "long_term_actions": [],
            "resource_requirements": {}
        }

        # Categorize actions by priority
        for risk in assessment.identified_risks:
            if risk.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                for strategy in risk.mitigation_strategies:
                    action = {
                        "risk_name": risk.name,
                        "strategy": strategy.name,
                        "description": strategy.description,
                        "timeline": strategy.timeline,
                        "cost": strategy.cost
                    }

                    if "immediate" in strategy.timeline.lower() or "week" in strategy.timeline.lower():
                        plan["immediate_actions"].append(action)
                    elif "month" in strategy.timeline.lower():
                        plan["short_term_actions"].append(action)
                    else:
                        plan["long_term_actions"].append(action)

        return plan

    def _generate_recommendations(self, assessment: RiskAssessmentResult) -> List[str]:
        """Generate recommendations based on risk assessment."""

        recommendations = []

        # Critical risk recommendations
        critical_risks = [r for r in assessment.identified_risks if r.risk_level == RiskLevel.CRITICAL]
        if critical_risks:
            recommendations.append("Address critical risks immediately to prevent potential project failure.")

        # Category-specific recommendations
        category_counts = assessment.risks_by_category

        if category_counts.get(RiskCategory.FINANCIAL, 0) > 2:
            recommendations.append("Implement enhanced financial controls and budget monitoring.")

        if category_counts.get(RiskCategory.VENDOR, 0) > 1:
            recommendations.append("Diversify vendor base and implement vendor risk management.")

        if category_counts.get(RiskCategory.TECHNICAL, 0) > 2:
            recommendations.append("Conduct technical review and implement additional testing.")

        return recommendations

    def _generate_action_items(self, assessment: RiskAssessmentResult) -> List[Dict[str, Any]]:
        """Generate action items based on risks."""

        action_items = []

        # Focus on high and critical risks
        priority_risks = [r for r in assessment.identified_risks
                         if r.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]]

        for risk in priority_risks:
            priority = "Critical" if risk.risk_level == RiskLevel.CRITICAL else "High"

            action_item = {
                "id": risk.risk_id,
                "title": f"Mitigate {risk.name}",
                "description": risk.description,
                "risk_score": risk.risk_score,
                "priority": priority,
                "category": risk.category.value,
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
            "total_patterns": len(self._risk_patterns),
            "patterns_by_category": {k.value: len(v) for k, v in self._patterns_by_category.items()},
            "total_mitigation_strategies": len(self._mitigation_strategies),
            "cache_size": len(self._assessment_cache),
            "config": self.config.dict()
        }

    def clear_cache(self) -> None:
        """Clear assessment cache."""

        self._assessment_cache.clear()
        logger.info("Risk assessment cache cleared")


# Global service instance
_risk_assessment_service: Optional[RiskAssessmentService] = None


def get_risk_assessment_service() -> RiskAssessmentService:
    """Get or create the global risk assessment service instance."""
    global _risk_assessment_service

    if _risk_assessment_service is None:
        _risk_assessment_service = RiskAssessmentService()

    return _risk_assessment_service


def reset_risk_assessment_service() -> None:
    """Reset the global risk assessment service instance."""
    global _risk_assessment_service
    _risk_assessment_service = None
