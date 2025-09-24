"""
Procurement Terminology Database and Expansion Service

This module provides procurement-specific terminology expansion, synonym mapping,
acronym expansion, and domain-specific query enhancement capabilities.
"""

import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TerminologyCategory(str, Enum):
    """Categories of procurement terminology."""
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    PROCESS = "process"
    QUALITY = "quality"
    COMPLIANCE = "compliance"
    LOGISTICS = "logistics"
    GENERAL = "general"


class ExpansionType(str, Enum):
    """Types of terminology expansion."""
    SYNONYM = "synonym"
    ACRONYM = "acronym"
    RELATED_TERM = "related_term"
    DOMAIN_SPECIFIC = "domain_specific"
    TECHNICAL_VARIANT = "technical_variant"


@dataclass
class TermExpansion:
    """Represents a term expansion with metadata."""
    original_term: str
    expanded_term: str
    expansion_type: ExpansionType
    category: TerminologyCategory
    confidence: float
    context: Optional[str] = None


class TerminologyExpansionConfig(BaseModel):
    """Configuration for terminology expansion."""
    
    enable_synonym_expansion: bool = Field(True, description="Enable synonym expansion")
    enable_acronym_expansion: bool = Field(True, description="Enable acronym expansion")
    enable_related_terms: bool = Field(True, description="Enable related term expansion")
    enable_domain_specific: bool = Field(True, description="Enable domain-specific expansion")
    
    max_expansions_per_term: int = Field(3, ge=1, le=10, description="Maximum expansions per term")
    min_confidence_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence for expansion")
    preserve_original_terms: bool = Field(True, description="Keep original terms in expanded query")
    
    # Category weights for prioritization
    category_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "technical": 1.0,
            "financial": 0.9,
            "legal": 0.8,
            "process": 0.8,
            "quality": 0.7,
            "compliance": 0.9,
            "logistics": 0.7,
            "general": 0.5
        },
        description="Weights for different terminology categories"
    )


class ProcurementTerminologyService:
    """Service for procurement terminology expansion and enhancement."""
    
    def __init__(self):
        self._terminology_db = self._initialize_terminology_database()
        self._acronym_db = self._initialize_acronym_database()
        self._synonym_db = self._initialize_synonym_database()
        self._related_terms_db = self._initialize_related_terms_database()
        
        # Compiled regex patterns for efficient matching
        self._acronym_patterns = self._compile_acronym_patterns()
        self._technical_patterns = self._compile_technical_patterns()
        
        logger.info("Procurement terminology service initialized")
    
    def expand_query_terms(
        self,
        query: str,
        config: TerminologyExpansionConfig
    ) -> Tuple[str, List[TermExpansion]]:
        """
        Expand query terms with procurement-specific terminology.
        
        Args:
            query: Original query string
            config: Expansion configuration
            
        Returns:
            Tuple of (expanded_query, list_of_expansions)
        """
        
        logger.info(f"Expanding query terms for: {query[:100]}...")
        
        # Extract terms from query
        terms = self._extract_terms(query)
        
        # Find expansions for each term
        all_expansions = []
        for term in terms:
            term_expansions = self._find_term_expansions(term, config)
            all_expansions.extend(term_expansions)
        
        # Filter and rank expansions
        filtered_expansions = self._filter_and_rank_expansions(
            all_expansions, config
        )
        
        # Build expanded query
        expanded_query = self._build_expanded_query(
            query, filtered_expansions, config
        )
        
        logger.info(
            f"Query expansion complete",
            original_terms=len(terms),
            expansions_found=len(filtered_expansions),
            expanded_query_length=len(expanded_query)
        )
        
        return expanded_query, filtered_expansions
    
    def get_term_suggestions(
        self,
        partial_term: str,
        category: Optional[TerminologyCategory] = None,
        limit: int = 10
    ) -> List[str]:
        """Get terminology suggestions for partial terms."""
        
        suggestions = []
        partial_lower = partial_term.lower()
        
        # Search in terminology database
        for term, metadata in self._terminology_db.items():
            if (partial_lower in term.lower() and 
                (category is None or metadata["category"] == category)):
                suggestions.append(term)
        
        # Search in acronym database
        for acronym, expansion in self._acronym_db.items():
            if partial_lower in acronym.lower() or partial_lower in expansion.lower():
                suggestions.append(f"{acronym} ({expansion})")
        
        return sorted(suggestions)[:limit]
    
    def _extract_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query."""
        
        # Basic tokenization with preservation of important patterns
        terms = []
        
        # Extract quoted phrases first
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        terms.extend(quoted_phrases)
        
        # Remove quoted phrases from query for further processing
        query_without_quotes = re.sub(r'"[^"]*"', '', query)
        
        # Extract acronyms (2-6 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', query_without_quotes)
        terms.extend(acronyms)
        
        # Extract technical terms (alphanumeric with hyphens/underscores)
        technical_terms = re.findall(r'\b[a-zA-Z][\w\-]{2,}\b', query_without_quotes)
        terms.extend(technical_terms)
        
        # Extract numbers with units
        number_units = re.findall(r'\b\d+(?:\.\d+)?\s*[a-zA-Z]+\b', query_without_quotes)
        terms.extend(number_units)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in terms:
            term_lower = term.lower()
            if term_lower not in seen and len(term) > 2:
                unique_terms.append(term)
                seen.add(term_lower)
        
        return unique_terms
    
    def _find_term_expansions(
        self,
        term: str,
        config: TerminologyExpansionConfig
    ) -> List[TermExpansion]:
        """Find all possible expansions for a term."""
        
        expansions = []
        term_lower = term.lower()
        
        # 1. Acronym expansion
        if config.enable_acronym_expansion and term.isupper():
            acronym_expansion = self._expand_acronym(term)
            if acronym_expansion:
                expansions.append(acronym_expansion)
        
        # 2. Synonym expansion
        if config.enable_synonym_expansion:
            synonym_expansions = self._expand_synonyms(term)
            expansions.extend(synonym_expansions)
        
        # 3. Related terms
        if config.enable_related_terms:
            related_expansions = self._expand_related_terms(term)
            expansions.extend(related_expansions)
        
        # 4. Domain-specific expansion
        if config.enable_domain_specific:
            domain_expansions = self._expand_domain_specific(term)
            expansions.extend(domain_expansions)
        
        return expansions
    
    def _expand_acronym(self, acronym: str) -> Optional[TermExpansion]:
        """Expand acronym to full form."""
        
        expansion = self._acronym_db.get(acronym.upper())
        if expansion:
            return TermExpansion(
                original_term=acronym,
                expanded_term=expansion,
                expansion_type=ExpansionType.ACRONYM,
                category=TerminologyCategory.TECHNICAL,
                confidence=0.9,
                context=f"Acronym expansion: {acronym} → {expansion}"
            )
        
        return None
    
    def _expand_synonyms(self, term: str) -> List[TermExpansion]:
        """Expand term with synonyms."""
        
        expansions = []
        term_lower = term.lower()
        
        synonyms = self._synonym_db.get(term_lower, [])
        for synonym in synonyms:
            expansion = TermExpansion(
                original_term=term,
                expanded_term=synonym,
                expansion_type=ExpansionType.SYNONYM,
                category=TerminologyCategory.GENERAL,
                confidence=0.8,
                context=f"Synonym: {term} → {synonym}"
            )
            expansions.append(expansion)
        
        return expansions
    
    def _expand_related_terms(self, term: str) -> List[TermExpansion]:
        """Expand with related procurement terms."""
        
        expansions = []
        term_lower = term.lower()
        
        related_terms = self._related_terms_db.get(term_lower, [])
        for related_term in related_terms:
            expansion = TermExpansion(
                original_term=term,
                expanded_term=related_term,
                expansion_type=ExpansionType.RELATED_TERM,
                category=TerminologyCategory.PROCESS,
                confidence=0.7,
                context=f"Related term: {term} → {related_term}"
            )
            expansions.append(expansion)
        
        return expansions
    
    def _expand_domain_specific(self, term: str) -> List[TermExpansion]:
        """Expand with domain-specific procurement terminology."""
        
        expansions = []
        term_lower = term.lower()
        
        # Check if term exists in terminology database
        if term_lower in self._terminology_db:
            term_data = self._terminology_db[term_lower]
            
            # Add technical variants
            for variant in term_data.get("variants", []):
                expansion = TermExpansion(
                    original_term=term,
                    expanded_term=variant,
                    expansion_type=ExpansionType.TECHNICAL_VARIANT,
                    category=TerminologyCategory(term_data["category"]),
                    confidence=0.8,
                    context=f"Technical variant: {term} → {variant}"
                )
                expansions.append(expansion)
        
        return expansions
    
    def _filter_and_rank_expansions(
        self,
        expansions: List[TermExpansion],
        config: TerminologyExpansionConfig
    ) -> List[TermExpansion]:
        """Filter and rank expansions based on configuration."""
        
        # Filter by confidence threshold
        filtered = [
            exp for exp in expansions 
            if exp.confidence >= config.min_confidence_threshold
        ]
        
        # Apply category weights
        for expansion in filtered:
            category_weight = config.category_weights.get(expansion.category.value, 0.5)
            expansion.confidence *= category_weight
        
        # Sort by confidence (descending)
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Group by original term and limit expansions per term
        term_groups = {}
        for expansion in filtered:
            term = expansion.original_term.lower()
            if term not in term_groups:
                term_groups[term] = []
            if len(term_groups[term]) < config.max_expansions_per_term:
                term_groups[term].append(expansion)
        
        # Flatten back to list
        result = []
        for group in term_groups.values():
            result.extend(group)
        
        return result
    
    def _build_expanded_query(
        self,
        original_query: str,
        expansions: List[TermExpansion],
        config: TerminologyExpansionConfig
    ) -> str:
        """Build expanded query with original and expanded terms."""
        
        if not expansions:
            return original_query
        
        # Collect unique expanded terms
        expanded_terms = set()
        for expansion in expansions:
            expanded_terms.add(expansion.expanded_term)
        
        if config.preserve_original_terms:
            # Combine original query with expanded terms
            expanded_query = f"{original_query} {' '.join(expanded_terms)}"
        else:
            # Replace with expanded terms
            expanded_query = ' '.join(expanded_terms)
        
        # Clean up extra whitespace
        expanded_query = re.sub(r'\s+', ' ', expanded_query.strip())
        
        return expanded_query

    def _initialize_terminology_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive procurement terminology database."""

        return {
            # Technical terms
            "api": {
                "category": "technical",
                "variants": ["application programming interface", "interface", "endpoint"],
                "description": "Application Programming Interface"
            },
            "sla": {
                "category": "technical",
                "variants": ["service level agreement", "service agreement", "performance agreement"],
                "description": "Service Level Agreement"
            },
            "database": {
                "category": "technical",
                "variants": ["db", "data store", "repository", "data warehouse"],
                "description": "Data storage system"
            },
            "server": {
                "category": "technical",
                "variants": ["host", "machine", "instance", "node"],
                "description": "Computing server"
            },
            "software": {
                "category": "technical",
                "variants": ["application", "program", "system", "solution"],
                "description": "Software application"
            },

            # Financial terms
            "cost": {
                "category": "financial",
                "variants": ["price", "expense", "fee", "charge", "rate"],
                "description": "Financial cost"
            },
            "budget": {
                "category": "financial",
                "variants": ["allocation", "funding", "financial plan", "expenditure"],
                "description": "Budget allocation"
            },
            "payment": {
                "category": "financial",
                "variants": ["remuneration", "compensation", "settlement", "disbursement"],
                "description": "Payment terms"
            },
            "invoice": {
                "category": "financial",
                "variants": ["bill", "statement", "charge", "billing"],
                "description": "Invoice document"
            },

            # Legal terms
            "contract": {
                "category": "legal",
                "variants": ["agreement", "accord", "pact", "deal"],
                "description": "Legal contract"
            },
            "compliance": {
                "category": "legal",
                "variants": ["conformance", "adherence", "observance", "regulatory compliance"],
                "description": "Regulatory compliance"
            },
            "liability": {
                "category": "legal",
                "variants": ["responsibility", "accountability", "obligation", "duty"],
                "description": "Legal liability"
            },
            "warranty": {
                "category": "legal",
                "variants": ["guarantee", "assurance", "coverage", "protection"],
                "description": "Warranty terms"
            },

            # Process terms
            "procurement": {
                "category": "process",
                "variants": ["purchasing", "acquisition", "sourcing", "buying"],
                "description": "Procurement process"
            },
            "vendor": {
                "category": "process",
                "variants": ["supplier", "provider", "contractor", "partner"],
                "description": "Service vendor"
            },
            "delivery": {
                "category": "process",
                "variants": ["shipment", "fulfillment", "provision", "supply"],
                "description": "Delivery process"
            },
            "timeline": {
                "category": "process",
                "variants": ["schedule", "timeframe", "deadline", "milestone"],
                "description": "Project timeline"
            },

            # Quality terms
            "quality": {
                "category": "quality",
                "variants": ["standard", "grade", "level", "excellence"],
                "description": "Quality standards"
            },
            "testing": {
                "category": "quality",
                "variants": ["validation", "verification", "assessment", "evaluation"],
                "description": "Quality testing"
            },
            "performance": {
                "category": "quality",
                "variants": ["efficiency", "effectiveness", "capability", "throughput"],
                "description": "Performance metrics"
            },

            # Compliance terms
            "security": {
                "category": "compliance",
                "variants": ["protection", "safety", "confidentiality", "privacy"],
                "description": "Security requirements"
            },
            "audit": {
                "category": "compliance",
                "variants": ["review", "inspection", "examination", "assessment"],
                "description": "Audit process"
            },
            "certification": {
                "category": "compliance",
                "variants": ["accreditation", "qualification", "validation", "approval"],
                "description": "Certification requirements"
            }
        }

    def _initialize_acronym_database(self) -> Dict[str, str]:
        """Initialize procurement-specific acronym database."""

        return {
            # Technical acronyms
            "API": "Application Programming Interface",
            "SLA": "Service Level Agreement",
            "SLO": "Service Level Objective",
            "KPI": "Key Performance Indicator",
            "QA": "Quality Assurance",
            "QC": "Quality Control",
            "IT": "Information Technology",
            "UI": "User Interface",
            "UX": "User Experience",
            "DB": "Database",
            "SQL": "Structured Query Language",
            "HTTP": "HyperText Transfer Protocol",
            "HTTPS": "HyperText Transfer Protocol Secure",
            "REST": "Representational State Transfer",
            "JSON": "JavaScript Object Notation",
            "XML": "eXtensible Markup Language",
            "CSV": "Comma Separated Values",
            "PDF": "Portable Document Format",

            # Business acronyms
            "RFQ": "Request for Quotation",
            "RFP": "Request for Proposal",
            "RFI": "Request for Information",
            "SOW": "Statement of Work",
            "MSA": "Master Service Agreement",
            "NDA": "Non-Disclosure Agreement",
            "PO": "Purchase Order",
            "ROI": "Return on Investment",
            "TCO": "Total Cost of Ownership",
            "CAPEX": "Capital Expenditure",
            "OPEX": "Operational Expenditure",
            "SaaS": "Software as a Service",
            "PaaS": "Platform as a Service",
            "IaaS": "Infrastructure as a Service",

            # Compliance acronyms
            "GDPR": "General Data Protection Regulation",
            "HIPAA": "Health Insurance Portability and Accountability Act",
            "SOX": "Sarbanes-Oxley Act",
            "ISO": "International Organization for Standardization",
            "NIST": "National Institute of Standards and Technology",
            "PCI": "Payment Card Industry",
            "DSS": "Data Security Standard",
            "FISMA": "Federal Information Security Management Act",

            # Quality acronyms
            "QMS": "Quality Management System",
            "SPC": "Statistical Process Control",
            "TQM": "Total Quality Management",
            "CMMI": "Capability Maturity Model Integration",
            "ITIL": "Information Technology Infrastructure Library",
            "COBIT": "Control Objectives for Information and Related Technologies"
        }

    def _initialize_synonym_database(self) -> Dict[str, List[str]]:
        """Initialize synonym database for common procurement terms."""

        return {
            "requirement": ["specification", "need", "criteria", "demand"],
            "solution": ["answer", "resolution", "approach", "method"],
            "implementation": ["deployment", "installation", "execution", "rollout"],
            "maintenance": ["support", "upkeep", "servicing", "care"],
            "upgrade": ["enhancement", "improvement", "update", "modernization"],
            "integration": ["connection", "linking", "combination", "merger"],
            "migration": ["transfer", "movement", "transition", "conversion"],
            "optimization": ["improvement", "enhancement", "refinement", "tuning"],
            "scalability": ["expandability", "growth capacity", "flexibility"],
            "reliability": ["dependability", "stability", "consistency", "trustworthiness"],
            "availability": ["uptime", "accessibility", "readiness", "operability"],
            "capacity": ["capability", "volume", "throughput", "bandwidth"],
            "efficiency": ["effectiveness", "productivity", "performance", "optimization"],
            "monitoring": ["tracking", "surveillance", "observation", "oversight"],
            "reporting": ["documentation", "analysis", "summary", "briefing"]
        }

    def _initialize_related_terms_database(self) -> Dict[str, List[str]]:
        """Initialize related terms database for procurement context."""

        return {
            "data": ["information", "records", "files", "content", "dataset"],
            "processing": ["handling", "manipulation", "transformation", "analysis"],
            "storage": ["archiving", "retention", "backup", "repository"],
            "security": ["encryption", "authentication", "authorization", "firewall"],
            "network": ["connectivity", "infrastructure", "bandwidth", "protocol"],
            "cloud": ["hosting", "virtualization", "scalability", "elasticity"],
            "backup": ["recovery", "restoration", "disaster recovery", "continuity"],
            "training": ["education", "instruction", "learning", "development"],
            "documentation": ["manual", "guide", "specification", "handbook"],
            "support": ["assistance", "help", "service", "maintenance"]
        }

    def _compile_acronym_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for acronym detection."""

        patterns = []

        # Standard acronyms (2-6 uppercase letters)
        patterns.append(re.compile(r'\b[A-Z]{2,6}\b'))

        # Acronyms with numbers
        patterns.append(re.compile(r'\b[A-Z]+\d+[A-Z]*\b'))

        # Hyphenated acronyms
        patterns.append(re.compile(r'\b[A-Z]+-[A-Z]+\b'))

        return patterns

    def _compile_technical_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for technical term detection."""

        patterns = []

        # Version numbers
        patterns.append(re.compile(r'\bv?\d+\.\d+(?:\.\d+)?\b', re.IGNORECASE))

        # Technical specifications
        patterns.append(re.compile(r'\b\d+(?:GB|MB|KB|TB|GHz|MHz|CPU|RAM)\b', re.IGNORECASE))

        # File extensions
        patterns.append(re.compile(r'\.\w{2,4}\b'))

        # URLs and domains
        patterns.append(re.compile(r'\b(?:https?://)?[\w\-]+\.[\w\-]+(?:\.[\w\-]+)*\b'))

        return patterns


# Singleton instance
_procurement_terminology_service: Optional[ProcurementTerminologyService] = None


def get_procurement_terminology_service() -> ProcurementTerminologyService:
    """Get the singleton procurement terminology service instance."""
    global _procurement_terminology_service
    if _procurement_terminology_service is None:
        _procurement_terminology_service = ProcurementTerminologyService()
    return _procurement_terminology_service
