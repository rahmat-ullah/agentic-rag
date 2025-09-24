"""
RFQ Hint Processing Service

This module provides intelligent RFQ hint processing capabilities to guide
targeted search strategies for procurement scenarios.
"""

import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class HintType(str, Enum):
    """Types of RFQ hints."""
    DOCUMENT_TYPE = "document_type"
    PROCESS_STAGE = "process_stage"
    VENDOR_REFERENCE = "vendor_reference"
    TIMELINE = "timeline"
    BUDGET_RANGE = "budget_range"
    TECHNICAL_REQUIREMENT = "technical_requirement"
    COMPLIANCE_REQUIREMENT = "compliance_requirement"
    GEOGRAPHIC_SCOPE = "geographic_scope"


class HintConfidence(str, Enum):
    """Confidence levels for hints."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RFQHint:
    """Represents an extracted RFQ hint."""
    hint_type: HintType
    content: str
    confidence: HintConfidence
    source_pattern: str
    context: str
    relevance_score: float
    search_strategy_suggestion: str


class RFQHintConfig(BaseModel):
    """Configuration for RFQ hint processing."""
    
    # Feature toggles
    enable_document_type_hints: bool = Field(True, description="Enable document type hint extraction")
    enable_process_stage_hints: bool = Field(True, description="Enable process stage hint extraction")
    enable_vendor_hints: bool = Field(True, description="Enable vendor reference hint extraction")
    enable_timeline_hints: bool = Field(True, description="Enable timeline hint extraction")
    enable_budget_hints: bool = Field(True, description="Enable budget range hint extraction")
    enable_technical_hints: bool = Field(True, description="Enable technical requirement hints")
    enable_compliance_hints: bool = Field(True, description="Enable compliance requirement hints")
    enable_geographic_hints: bool = Field(True, description="Enable geographic scope hints")
    
    # Quality settings
    min_hint_confidence: HintConfidence = Field(HintConfidence.MEDIUM, description="Minimum hint confidence")
    max_hints_per_query: int = Field(10, ge=1, le=20, description="Maximum hints per query")
    relevance_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Relevance threshold for hints")
    
    # Processing settings
    enable_pattern_validation: bool = Field(True, description="Enable pattern validation")
    enable_context_analysis: bool = Field(True, description="Enable context analysis")
    enable_strategy_suggestions: bool = Field(True, description="Enable search strategy suggestions")


class RFQHintProcessor:
    """Service for processing RFQ hints to guide search strategies."""
    
    def __init__(self):
        self._document_type_patterns = self._initialize_document_type_patterns()
        self._process_stage_patterns = self._initialize_process_stage_patterns()
        self._vendor_patterns = self._initialize_vendor_patterns()
        self._timeline_patterns = self._initialize_timeline_patterns()
        self._budget_patterns = self._initialize_budget_patterns()
        self._technical_patterns = self._initialize_technical_patterns()
        self._compliance_patterns = self._initialize_compliance_patterns()
        self._geographic_patterns = self._initialize_geographic_patterns()
        
        # Processing statistics
        self._stats = {
            "total_queries_processed": 0,
            "total_hints_extracted": 0,
            "document_type_hints": 0,
            "process_stage_hints": 0,
            "vendor_hints": 0,
            "timeline_hints": 0,
            "budget_hints": 0,
            "technical_hints": 0,
            "compliance_hints": 0,
            "geographic_hints": 0,
            "average_hints_per_query": 0.0,
            "average_confidence_score": 0.0
        }
        
        logger.info("RFQ hint processor initialized")
    
    def process_rfq_hints(
        self,
        query: str,
        config: RFQHintConfig,
        context: Optional[Dict[str, Any]] = None
    ) -> List[RFQHint]:
        """
        Process query to extract RFQ hints for search strategy guidance.
        
        Args:
            query: Query string to process
            config: RFQ hint processing configuration
            context: Optional processing context
            
        Returns:
            List of extracted RFQ hints
        """
        
        logger.info(f"Processing RFQ hints for query: {query[:100]}...")
        
        self._stats["total_queries_processed"] += 1
        
        # Collect all potential hints
        all_hints = []
        
        # 1. Document type hints
        if config.enable_document_type_hints:
            doc_hints = self._extract_document_type_hints(query, config)
            all_hints.extend(doc_hints)
            self._stats["document_type_hints"] += len(doc_hints)
        
        # 2. Process stage hints
        if config.enable_process_stage_hints:
            stage_hints = self._extract_process_stage_hints(query, config)
            all_hints.extend(stage_hints)
            self._stats["process_stage_hints"] += len(stage_hints)
        
        # 3. Vendor reference hints
        if config.enable_vendor_hints:
            vendor_hints = self._extract_vendor_hints(query, config)
            all_hints.extend(vendor_hints)
            self._stats["vendor_hints"] += len(vendor_hints)
        
        # 4. Timeline hints
        if config.enable_timeline_hints:
            timeline_hints = self._extract_timeline_hints(query, config)
            all_hints.extend(timeline_hints)
            self._stats["timeline_hints"] += len(timeline_hints)
        
        # 5. Budget range hints
        if config.enable_budget_hints:
            budget_hints = self._extract_budget_hints(query, config)
            all_hints.extend(budget_hints)
            self._stats["budget_hints"] += len(budget_hints)
        
        # 6. Technical requirement hints
        if config.enable_technical_hints:
            tech_hints = self._extract_technical_hints(query, config)
            all_hints.extend(tech_hints)
            self._stats["technical_hints"] += len(tech_hints)
        
        # 7. Compliance requirement hints
        if config.enable_compliance_hints:
            compliance_hints = self._extract_compliance_hints(query, config)
            all_hints.extend(compliance_hints)
            self._stats["compliance_hints"] += len(compliance_hints)
        
        # 8. Geographic scope hints
        if config.enable_geographic_hints:
            geo_hints = self._extract_geographic_hints(query, config)
            all_hints.extend(geo_hints)
            self._stats["geographic_hints"] += len(geo_hints)
        
        # Filter and rank hints
        filtered_hints = self._filter_and_rank_hints(all_hints, config)
        
        # Update statistics
        self._update_statistics(filtered_hints)
        
        logger.info(
            f"RFQ hint processing complete",
            total_hints_extracted=len(filtered_hints),
            query_length=len(query)
        )
        
        return filtered_hints
    
    def _extract_document_type_hints(
        self,
        query: str,
        config: RFQHintConfig
    ) -> List[RFQHint]:
        """Extract document type hints from query."""
        
        hints = []
        query_lower = query.lower()
        
        for doc_type, patterns in self._document_type_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]
                strategy = pattern_info["strategy"]
                
                if re.search(pattern, query_lower):
                    relevance = self._calculate_pattern_relevance(query, pattern)
                    
                    hint = RFQHint(
                        hint_type=HintType.DOCUMENT_TYPE,
                        content=f"document_type:{doc_type}",
                        confidence=confidence,
                        source_pattern=pattern,
                        context=f"Detected {doc_type} document reference",
                        relevance_score=relevance,
                        search_strategy_suggestion=strategy
                    )
                    hints.append(hint)
        
        return hints
    
    def _extract_process_stage_hints(
        self,
        query: str,
        config: RFQHintConfig
    ) -> List[RFQHint]:
        """Extract process stage hints from query."""
        
        hints = []
        query_lower = query.lower()
        
        for stage, patterns in self._process_stage_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]
                strategy = pattern_info["strategy"]
                
                if re.search(pattern, query_lower):
                    relevance = self._calculate_pattern_relevance(query, pattern)
                    
                    hint = RFQHint(
                        hint_type=HintType.PROCESS_STAGE,
                        content=f"process_stage:{stage}",
                        confidence=confidence,
                        source_pattern=pattern,
                        context=f"Detected {stage} process stage",
                        relevance_score=relevance,
                        search_strategy_suggestion=strategy
                    )
                    hints.append(hint)
        
        return hints
    
    def _extract_vendor_hints(
        self,
        query: str,
        config: RFQHintConfig
    ) -> List[RFQHint]:
        """Extract vendor reference hints from query."""
        
        hints = []
        query_lower = query.lower()
        
        for vendor_type, patterns in self._vendor_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]
                strategy = pattern_info["strategy"]
                
                if re.search(pattern, query_lower):
                    relevance = self._calculate_pattern_relevance(query, pattern)
                    
                    hint = RFQHint(
                        hint_type=HintType.VENDOR_REFERENCE,
                        content=f"vendor_type:{vendor_type}",
                        confidence=confidence,
                        source_pattern=pattern,
                        context=f"Detected {vendor_type} vendor reference",
                        relevance_score=relevance,
                        search_strategy_suggestion=strategy
                    )
                    hints.append(hint)
        
        return hints
    
    def _extract_timeline_hints(
        self,
        query: str,
        config: RFQHintConfig
    ) -> List[RFQHint]:
        """Extract timeline hints from query."""
        
        hints = []
        query_lower = query.lower()
        
        for timeline_type, patterns in self._timeline_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]
                strategy = pattern_info["strategy"]
                
                matches = re.finditer(pattern, query_lower)
                for match in matches:
                    relevance = self._calculate_pattern_relevance(query, pattern)
                    
                    hint = RFQHint(
                        hint_type=HintType.TIMELINE,
                        content=f"timeline:{timeline_type}:{match.group()}",
                        confidence=confidence,
                        source_pattern=pattern,
                        context=f"Detected {timeline_type} timeline reference",
                        relevance_score=relevance,
                        search_strategy_suggestion=strategy
                    )
                    hints.append(hint)
        
        return hints

    def _extract_budget_hints(
        self,
        query: str,
        config: RFQHintConfig
    ) -> List[RFQHint]:
        """Extract budget range hints from query."""

        hints = []
        query_lower = query.lower()

        for budget_type, patterns in self._budget_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]
                strategy = pattern_info["strategy"]

                matches = re.finditer(pattern, query_lower)
                for match in matches:
                    relevance = self._calculate_pattern_relevance(query, pattern)

                    hint = RFQHint(
                        hint_type=HintType.BUDGET_RANGE,
                        content=f"budget:{budget_type}:{match.group()}",
                        confidence=confidence,
                        source_pattern=pattern,
                        context=f"Detected {budget_type} budget reference",
                        relevance_score=relevance,
                        search_strategy_suggestion=strategy
                    )
                    hints.append(hint)

        return hints

    def _extract_technical_hints(
        self,
        query: str,
        config: RFQHintConfig
    ) -> List[RFQHint]:
        """Extract technical requirement hints from query."""

        hints = []
        query_lower = query.lower()

        for tech_type, patterns in self._technical_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]
                strategy = pattern_info["strategy"]

                if re.search(pattern, query_lower):
                    relevance = self._calculate_pattern_relevance(query, pattern)

                    hint = RFQHint(
                        hint_type=HintType.TECHNICAL_REQUIREMENT,
                        content=f"technical:{tech_type}",
                        confidence=confidence,
                        source_pattern=pattern,
                        context=f"Detected {tech_type} technical requirement",
                        relevance_score=relevance,
                        search_strategy_suggestion=strategy
                    )
                    hints.append(hint)

        return hints

    def _extract_compliance_hints(
        self,
        query: str,
        config: RFQHintConfig
    ) -> List[RFQHint]:
        """Extract compliance requirement hints from query."""

        hints = []
        query_lower = query.lower()

        for compliance_type, patterns in self._compliance_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]
                strategy = pattern_info["strategy"]

                if re.search(pattern, query_lower):
                    relevance = self._calculate_pattern_relevance(query, pattern)

                    hint = RFQHint(
                        hint_type=HintType.COMPLIANCE_REQUIREMENT,
                        content=f"compliance:{compliance_type}",
                        confidence=confidence,
                        source_pattern=pattern,
                        context=f"Detected {compliance_type} compliance requirement",
                        relevance_score=relevance,
                        search_strategy_suggestion=strategy
                    )
                    hints.append(hint)

        return hints

    def _extract_geographic_hints(
        self,
        query: str,
        config: RFQHintConfig
    ) -> List[RFQHint]:
        """Extract geographic scope hints from query."""

        hints = []
        query_lower = query.lower()

        for geo_type, patterns in self._geographic_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]
                strategy = pattern_info["strategy"]

                if re.search(pattern, query_lower):
                    relevance = self._calculate_pattern_relevance(query, pattern)

                    hint = RFQHint(
                        hint_type=HintType.GEOGRAPHIC_SCOPE,
                        content=f"geographic:{geo_type}",
                        confidence=confidence,
                        source_pattern=pattern,
                        context=f"Detected {geo_type} geographic scope",
                        relevance_score=relevance,
                        search_strategy_suggestion=strategy
                    )
                    hints.append(hint)

        return hints

    def _calculate_pattern_relevance(self, query: str, pattern: str) -> float:
        """Calculate relevance score for a pattern match."""

        # Simple relevance calculation based on pattern specificity
        pattern_length = len(pattern.replace(r'\b', '').replace(r'\w+', ''))
        query_length = len(query)

        if query_length == 0:
            return 0.0

        # Base relevance on pattern specificity and query coverage
        specificity_score = min(pattern_length / 20, 1.0)  # Normalize to 0-1
        coverage_score = min(pattern_length / query_length, 1.0)

        return (specificity_score + coverage_score) / 2

    def _filter_and_rank_hints(
        self,
        hints: List[RFQHint],
        config: RFQHintConfig
    ) -> List[RFQHint]:
        """Filter and rank hints based on configuration."""

        # Convert confidence enum to numeric for filtering
        confidence_values = {
            HintConfidence.HIGH: 0.8,
            HintConfidence.MEDIUM: 0.6,
            HintConfidence.LOW: 0.4
        }

        min_confidence_value = confidence_values[config.min_hint_confidence]

        # Filter by confidence and relevance
        filtered_hints = [
            hint for hint in hints
            if (confidence_values[hint.confidence] >= min_confidence_value and
                hint.relevance_score >= config.relevance_threshold)
        ]

        # Sort by combined score (confidence * relevance)
        filtered_hints.sort(
            key=lambda x: confidence_values[x.confidence] * x.relevance_score,
            reverse=True
        )

        # Limit number of hints
        limited_hints = filtered_hints[:config.max_hints_per_query]

        # Remove duplicates based on content
        unique_hints = []
        seen_content = set()

        for hint in limited_hints:
            if hint.content not in seen_content:
                unique_hints.append(hint)
                seen_content.add(hint.content)

        return unique_hints

    def _update_statistics(self, hints: List[RFQHint]):
        """Update processing statistics."""

        self._stats["total_hints_extracted"] += len(hints)

        # Update average hints per query
        total_queries = self._stats["total_queries_processed"]
        total_hints = self._stats["total_hints_extracted"]
        self._stats["average_hints_per_query"] = total_hints / total_queries if total_queries > 0 else 0.0

        # Update average confidence score
        if hints:
            confidence_values = {
                HintConfidence.HIGH: 0.8,
                HintConfidence.MEDIUM: 0.6,
                HintConfidence.LOW: 0.4
            }

            avg_confidence = sum(confidence_values[hint.confidence] for hint in hints) / len(hints)
            current_avg = self._stats["average_confidence_score"]

            # Update running average
            self._stats["average_confidence_score"] = (
                (current_avg * (total_queries - 1)) + avg_confidence
            ) / total_queries if total_queries > 0 else avg_confidence

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()

    def _initialize_document_type_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize document type detection patterns."""

        return {
            "rfq": [
                {
                    "pattern": r"\brfq\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "three_hop_search"
                },
                {
                    "pattern": r"request for quotation",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "three_hop_search"
                },
                {
                    "pattern": r"quote request",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "semantic_search"
                }
            ],
            "proposal": [
                {
                    "pattern": r"\bproposal\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "document_linking"
                },
                {
                    "pattern": r"\boffer\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "semantic_search"
                },
                {
                    "pattern": r"\bbid\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "semantic_search"
                }
            ],
            "contract": [
                {
                    "pattern": r"\bcontract\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "exact_match"
                },
                {
                    "pattern": r"\bagreement\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "semantic_search"
                }
            ],
            "specification": [
                {
                    "pattern": r"\bspecification\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "technical_search"
                },
                {
                    "pattern": r"\brequirements\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "semantic_search"
                }
            ]
        }

    def _initialize_process_stage_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize process stage detection patterns."""

        return {
            "requirements_gathering": [
                {
                    "pattern": r"requirements gathering",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "requirements_focused"
                },
                {
                    "pattern": r"needs assessment",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "semantic_search"
                }
            ],
            "vendor_identification": [
                {
                    "pattern": r"vendor identification",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "vendor_focused"
                },
                {
                    "pattern": r"supplier search",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "vendor_focused"
                }
            ],
            "proposal_evaluation": [
                {
                    "pattern": r"proposal evaluation",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "comparative_analysis"
                },
                {
                    "pattern": r"bid evaluation",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "comparative_analysis"
                }
            ]
        }

    def _initialize_vendor_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize vendor reference patterns."""

        return {
            "specific_vendor": [
                {
                    "pattern": r"\b[A-Z][a-z]+ (Inc|LLC|Corp|Ltd)\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "vendor_specific"
                }
            ],
            "vendor_category": [
                {
                    "pattern": r"\bvendor\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "vendor_focused"
                },
                {
                    "pattern": r"\bsupplier\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "vendor_focused"
                },
                {
                    "pattern": r"\bcontractor\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "vendor_focused"
                }
            ]
        }

    def _initialize_timeline_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize timeline detection patterns."""

        return {
            "urgent": [
                {
                    "pattern": r"\burgent\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "priority_search"
                },
                {
                    "pattern": r"\basap\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "priority_search"
                }
            ],
            "deadline": [
                {
                    "pattern": r"\bdeadline\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "time_sensitive"
                },
                {
                    "pattern": r"due date",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "time_sensitive"
                }
            ],
            "specific_date": [
                {
                    "pattern": r"\b\d{1,2}/\d{1,2}/\d{4}\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "date_specific"
                },
                {
                    "pattern": r"\b\d{4}-\d{2}-\d{2}\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "date_specific"
                }
            ]
        }

    def _initialize_budget_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize budget range detection patterns."""

        return {
            "currency_amount": [
                {
                    "pattern": r"\$[\d,]+(?:\.\d{2})?",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "budget_filtered"
                },
                {
                    "pattern": r"[\d,]+\s*(?:dollars?|USD)",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "budget_filtered"
                }
            ],
            "budget_range": [
                {
                    "pattern": r"budget.*\$[\d,]+",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "budget_focused"
                },
                {
                    "pattern": r"under.*\$[\d,]+",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "budget_filtered"
                }
            ]
        }

    def _initialize_technical_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize technical requirement patterns."""

        return {
            "software": [
                {
                    "pattern": r"\bsoftware\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "technical_focused"
                },
                {
                    "pattern": r"\bapplication\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "technical_focused"
                }
            ],
            "hardware": [
                {
                    "pattern": r"\bhardware\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "technical_focused"
                },
                {
                    "pattern": r"\bserver\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "technical_focused"
                }
            ],
            "integration": [
                {
                    "pattern": r"\bintegration\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "integration_focused"
                },
                {
                    "pattern": r"\bapi\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "technical_focused"
                }
            ]
        }

    def _initialize_compliance_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize compliance requirement patterns."""

        return {
            "security": [
                {
                    "pattern": r"\bsecurity\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "compliance_focused"
                },
                {
                    "pattern": r"\bencryption\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "security_focused"
                }
            ],
            "regulatory": [
                {
                    "pattern": r"\bcompliance\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "compliance_focused"
                },
                {
                    "pattern": r"\bregulation\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "compliance_focused"
                }
            ],
            "standards": [
                {
                    "pattern": r"\bISO\s*\d+\b",
                    "confidence": HintConfidence.HIGH,
                    "strategy": "standards_focused"
                },
                {
                    "pattern": r"\bstandard\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "standards_focused"
                }
            ]
        }

    def _initialize_geographic_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize geographic scope patterns."""

        return {
            "local": [
                {
                    "pattern": r"\blocal\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "geographic_filtered"
                },
                {
                    "pattern": r"\bonsite\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "geographic_filtered"
                }
            ],
            "national": [
                {
                    "pattern": r"\bnational\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "geographic_filtered"
                },
                {
                    "pattern": r"\bcountrywide\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "geographic_filtered"
                }
            ],
            "international": [
                {
                    "pattern": r"\binternational\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "geographic_filtered"
                },
                {
                    "pattern": r"\bglobal\b",
                    "confidence": HintConfidence.MEDIUM,
                    "strategy": "geographic_filtered"
                }
            ]
        }


# Singleton instance
_rfq_hint_processor: Optional[RFQHintProcessor] = None


def get_rfq_hint_processor() -> RFQHintProcessor:
    """Get the singleton RFQ hint processor instance."""
    global _rfq_hint_processor
    if _rfq_hint_processor is None:
        _rfq_hint_processor = RFQHintProcessor()
    return _rfq_hint_processor
