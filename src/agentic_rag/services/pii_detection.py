"""
PII Detection and Redaction Service

This module provides comprehensive PII detection and redaction capabilities
with pattern recognition, named entity recognition, context-aware detection,
and configurable redaction strategies.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, validator

from agentic_rag.config import get_settings

logger = structlog.get_logger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    
    # Personal identifiers
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    
    # Names and addresses
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    ZIP_CODE = "zip_code"
    
    # Financial information
    BANK_ACCOUNT = "bank_account"
    ROUTING_NUMBER = "routing_number"
    TAX_ID = "tax_id"
    
    # Medical information
    MEDICAL_RECORD = "medical_record"
    INSURANCE_ID = "insurance_id"
    
    # Government identifiers
    GOVERNMENT_ID = "government_id"
    VOTER_ID = "voter_id"
    
    # Digital identifiers
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    URL = "url"
    
    # Custom patterns
    CUSTOM = "custom"


class RedactionStrategy(str, Enum):
    """Strategies for redacting PII."""
    
    FULL_REDACTION = "full_redaction"          # [REDACTED]
    PARTIAL_REDACTION = "partial_redaction"    # john.***@company.com
    CATEGORY_REPLACEMENT = "category_replacement"  # [EMAIL ADDRESS]
    ANONYMIZATION = "anonymization"            # Person A, Company X
    MASKING = "masking"                       # XXX-XX-1234
    PLACEHOLDER = "placeholder"               # [PERSON_NAME_1]


class ConfidenceLevel(str, Enum):
    """Confidence levels for PII detection."""
    
    VERY_HIGH = "very_high"  # 0.9-1.0
    HIGH = "high"           # 0.7-0.9
    MEDIUM = "medium"       # 0.5-0.7
    LOW = "low"            # 0.3-0.5
    VERY_LOW = "very_low"  # 0.0-0.3


@dataclass
class PIIPattern:
    """Represents a PII detection pattern."""
    
    pii_type: PIIType
    pattern: str
    description: str
    confidence_base: float = 0.8
    context_keywords: List[str] = field(default_factory=list)
    exclusion_patterns: List[str] = field(default_factory=list)
    validation_func: Optional[callable] = None


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    
    pii_type: PIIType
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str = ""
    pattern_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PIIDetectionConfig(BaseModel):
    """Configuration for PII detection."""
    
    # Detection settings
    enabled_pii_types: Set[PIIType] = Field(
        default_factory=lambda: set(PIIType),
        description="PII types to detect"
    )
    min_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detection"
    )
    context_window_size: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Context window size for analysis"
    )
    
    # Pattern settings
    case_sensitive: bool = Field(
        default=False,
        description="Whether pattern matching is case sensitive"
    )
    enable_context_analysis: bool = Field(
        default=True,
        description="Enable context-aware detection"
    )
    enable_validation: bool = Field(
        default=True,
        description="Enable validation functions"
    )
    
    # Performance settings
    max_text_length: int = Field(
        default=1000000,
        ge=1000,
        description="Maximum text length to process"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Cache TTL in seconds"
    )


class RedactionConfig(BaseModel):
    """Configuration for redaction strategies."""
    
    # Default strategies by PII type
    default_strategies: Dict[PIIType, RedactionStrategy] = Field(
        default_factory=lambda: {
            PIIType.SSN: RedactionStrategy.MASKING,
            PIIType.EMAIL: RedactionStrategy.PARTIAL_REDACTION,
            PIIType.PHONE: RedactionStrategy.MASKING,
            PIIType.CREDIT_CARD: RedactionStrategy.MASKING,
            PIIType.PERSON_NAME: RedactionStrategy.ANONYMIZATION,
            PIIType.ADDRESS: RedactionStrategy.CATEGORY_REPLACEMENT,
        },
        description="Default redaction strategies by PII type"
    )
    
    # Redaction settings
    preserve_format: bool = Field(
        default=True,
        description="Preserve original text format when possible"
    )
    preserve_length: bool = Field(
        default=False,
        description="Preserve original text length"
    )
    anonymization_seed: Optional[str] = Field(
        default=None,
        description="Seed for consistent anonymization"
    )
    
    # Custom replacements
    custom_replacements: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom replacement patterns"
    )
    
    # Placeholder settings
    use_numbered_placeholders: bool = Field(
        default=True,
        description="Use numbered placeholders for consistency"
    )
    placeholder_prefix: str = Field(
        default="[",
        description="Placeholder prefix"
    )
    placeholder_suffix: str = Field(
        default="]",
        description="Placeholder suffix"
    )


class PIIDetectionResult(BaseModel):
    """Result of PII detection operation."""
    
    # Detection results
    matches: List[PIIMatch] = Field(
        default_factory=list,
        description="Detected PII matches"
    )
    total_matches: int = Field(
        default=0,
        description="Total number of matches found"
    )
    
    # Statistics by type
    matches_by_type: Dict[PIIType, int] = Field(
        default_factory=dict,
        description="Number of matches by PII type"
    )
    confidence_distribution: Dict[ConfidenceLevel, int] = Field(
        default_factory=dict,
        description="Distribution of confidence levels"
    )
    
    # Processing metadata
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )
    text_length: int = Field(
        default=0,
        description="Length of processed text"
    )
    patterns_used: List[str] = Field(
        default_factory=list,
        description="Patterns used in detection"
    )
    
    # Quality metrics
    coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Coverage score of detection"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence score"
    )


class RedactionResult(BaseModel):
    """Result of redaction operation."""
    
    # Redacted content
    redacted_text: str = Field(
        ...,
        description="Text with PII redacted"
    )
    redaction_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of original to redacted text"
    )
    
    # Redaction statistics
    total_redactions: int = Field(
        default=0,
        description="Total number of redactions performed"
    )
    redactions_by_type: Dict[PIIType, int] = Field(
        default_factory=dict,
        description="Number of redactions by PII type"
    )
    redactions_by_strategy: Dict[RedactionStrategy, int] = Field(
        default_factory=dict,
        description="Number of redactions by strategy"
    )
    
    # Processing metadata
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )
    original_length: int = Field(
        default=0,
        description="Original text length"
    )
    redacted_length: int = Field(
        default=0,
        description="Redacted text length"
    )
    
    # Quality metrics
    redaction_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of detected PII that was redacted"
    )
    text_preservation: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of original text preserved"
    )


class PIIDetectionService:
    """Service for detecting and redacting PII in text."""

    def __init__(self, detection_config: Optional[PIIDetectionConfig] = None,
                 redaction_config: Optional[RedactionConfig] = None):
        self.detection_config = detection_config or PIIDetectionConfig()
        self.redaction_config = redaction_config or RedactionConfig()
        self.settings = get_settings()

        # Initialize patterns
        self._patterns = self._initialize_patterns()

        # Performance tracking
        self._stats = {
            "total_detections": 0,
            "total_redactions": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Cache for results
        self._cache: Dict[str, Any] = {}

        logger.info("PII detection service initialized")

    def _initialize_patterns(self) -> Dict[PIIType, List[PIIPattern]]:
        """Initialize PII detection patterns."""

        patterns = {
            PIIType.SSN: [
                PIIPattern(
                    pii_type=PIIType.SSN,
                    pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                    description="SSN with dashes",
                    confidence_base=0.9,
                    context_keywords=["ssn", "social security", "social security number"],
                    validation_func=self._validate_ssn
                ),
                PIIPattern(
                    pii_type=PIIType.SSN,
                    pattern=r'\b\d{9}\b',
                    description="SSN without dashes",
                    confidence_base=0.7,
                    context_keywords=["ssn", "social security"],
                    validation_func=self._validate_ssn
                )
            ],

            PIIType.EMAIL: [
                PIIPattern(
                    pii_type=PIIType.EMAIL,
                    pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    description="Email address",
                    confidence_base=0.95,
                    context_keywords=["email", "e-mail", "contact"],
                    validation_func=self._validate_email
                )
            ],

            PIIType.PHONE: [
                PIIPattern(
                    pii_type=PIIType.PHONE,
                    pattern=r'\b\d{3}-\d{3}-\d{4}\b',
                    description="Phone with dashes",
                    confidence_base=0.8,
                    context_keywords=["phone", "telephone", "mobile", "cell"],
                    validation_func=self._validate_phone
                ),
                PIIPattern(
                    pii_type=PIIType.PHONE,
                    pattern=r'\(\d{3}\)\s*\d{3}-\d{4}',
                    description="Phone with parentheses",
                    confidence_base=0.85,
                    context_keywords=["phone", "telephone"],
                    validation_func=self._validate_phone
                ),
                PIIPattern(
                    pii_type=PIIType.PHONE,
                    pattern=r'\b\d{10}\b',
                    description="Phone without formatting",
                    confidence_base=0.6,
                    context_keywords=["phone", "telephone", "mobile"],
                    validation_func=self._validate_phone
                )
            ],

            PIIType.CREDIT_CARD: [
                PIIPattern(
                    pii_type=PIIType.CREDIT_CARD,
                    pattern=r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
                    description="Credit card number",
                    confidence_base=0.9,
                    context_keywords=["credit card", "card number", "visa", "mastercard", "amex"],
                    validation_func=self._validate_credit_card
                )
            ],

            PIIType.IP_ADDRESS: [
                PIIPattern(
                    pii_type=PIIType.IP_ADDRESS,
                    pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                    description="IPv4 address",
                    confidence_base=0.8,
                    context_keywords=["ip", "ip address", "server", "host"],
                    validation_func=self._validate_ip_address
                )
            ],

            PIIType.ZIP_CODE: [
                PIIPattern(
                    pii_type=PIIType.ZIP_CODE,
                    pattern=r'\b\d{5}(?:-\d{4})?\b',
                    description="US ZIP code",
                    confidence_base=0.7,
                    context_keywords=["zip", "zip code", "postal code", "address"],
                    validation_func=self._validate_zip_code
                )
            ],

            PIIType.URL: [
                PIIPattern(
                    pii_type=PIIType.URL,
                    pattern=r'https?://[^\s<>"{}|\\^`\[\]]+',
                    description="HTTP/HTTPS URL",
                    confidence_base=0.9,
                    context_keywords=["url", "website", "link", "http"],
                    validation_func=self._validate_url
                )
            ]
        }

        return patterns

    def detect_pii(self, text: str) -> PIIDetectionResult:
        """Detect PII in the given text."""
        start_time = time.time()

        try:
            # Check cache first
            if self.detection_config.enable_caching:
                cache_key = self._generate_cache_key(text, "detection")
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self._stats["cache_hits"] += 1
                    return cached_result
                self._stats["cache_misses"] += 1

            # Validate input
            if len(text) > self.detection_config.max_text_length:
                raise ValueError(f"Text length {len(text)} exceeds maximum {self.detection_config.max_text_length}")

            # Detect PII matches
            all_matches = []
            patterns_used = []

            for pii_type in self.detection_config.enabled_pii_types:
                if pii_type in self._patterns:
                    type_matches = self._detect_pii_type(text, pii_type)
                    all_matches.extend(type_matches)
                    if type_matches:
                        patterns_used.extend([p.pattern for p in self._patterns[pii_type]])

            # Remove overlapping matches (keep highest confidence)
            filtered_matches = self._remove_overlapping_matches(all_matches)

            # Filter by confidence threshold
            final_matches = [
                match for match in filtered_matches
                if match.confidence >= self.detection_config.min_confidence_threshold
            ]

            # Calculate statistics
            processing_time_ms = (time.time() - start_time) * 1000

            result = PIIDetectionResult(
                matches=final_matches,
                total_matches=len(final_matches),
                matches_by_type=self._calculate_matches_by_type(final_matches),
                confidence_distribution=self._calculate_confidence_distribution(final_matches),
                processing_time_ms=processing_time_ms,
                text_length=len(text),
                patterns_used=list(set(patterns_used)),
                coverage_score=self._calculate_coverage_score(text, final_matches),
                confidence_score=self._calculate_average_confidence(final_matches)
            )

            # Cache result
            if self.detection_config.enable_caching:
                self._cache_result(cache_key, result)

            # Update stats
            self._stats["total_detections"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms

            logger.debug(
                "PII detection completed",
                matches_found=len(final_matches),
                processing_time_ms=processing_time_ms,
                text_length=len(text)
            )

            return result

        except Exception as e:
            logger.error("PII detection failed", error=str(e))
            raise

    def _detect_pii_type(self, text: str, pii_type: PIIType) -> List[PIIMatch]:
        """Detect PII of a specific type in text."""
        matches = []

        if pii_type not in self._patterns:
            return matches

        for pattern_def in self._patterns[pii_type]:
            # Compile pattern
            flags = 0 if self.detection_config.case_sensitive else re.IGNORECASE
            pattern = re.compile(pattern_def.pattern, flags)

            # Find all matches
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                matched_text = match.group()

                # Get context
                context = self._extract_context(text, start_pos, end_pos)

                # Calculate confidence
                confidence = self._calculate_confidence(
                    pattern_def, matched_text, context
                )

                # Validate if enabled
                if (self.detection_config.enable_validation and
                    pattern_def.validation_func and
                    not pattern_def.validation_func(matched_text)):
                    continue

                # Create match
                pii_match = PIIMatch(
                    pii_type=pii_type,
                    text=matched_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=confidence,
                    context=context,
                    pattern_used=pattern_def.pattern,
                    metadata={
                        "pattern_description": pattern_def.description,
                        "context_keywords_found": self._find_context_keywords(
                            context, pattern_def.context_keywords
                        )
                    }
                )

                matches.append(pii_match)

        return matches

    def _extract_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extract context around a match."""
        window_size = self.detection_config.context_window_size

        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)

        return text[context_start:context_end]

    def _calculate_confidence(self, pattern_def: PIIPattern,
                            matched_text: str, context: str) -> float:
        """Calculate confidence score for a match."""
        base_confidence = pattern_def.confidence_base

        if not self.detection_config.enable_context_analysis:
            return base_confidence

        # Context keyword boost
        context_boost = 0.0
        if pattern_def.context_keywords:
            found_keywords = self._find_context_keywords(
                context.lower(), pattern_def.context_keywords
            )
            if found_keywords:
                context_boost = min(0.2, len(found_keywords) * 0.05)

        # Length penalty for very short matches
        length_penalty = 0.0
        if len(matched_text) < 3:
            length_penalty = 0.1

        # Calculate final confidence
        confidence = base_confidence + context_boost - length_penalty
        return max(0.0, min(1.0, confidence))

    def _find_context_keywords(self, context: str, keywords: List[str]) -> List[str]:
        """Find context keywords in the given context."""
        found = []
        context_lower = context.lower()

        for keyword in keywords:
            if keyword.lower() in context_lower:
                found.append(keyword)

        return found

    def _remove_overlapping_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping the highest confidence ones."""
        if not matches:
            return matches

        # Sort by position
        sorted_matches = sorted(matches, key=lambda m: (m.start_pos, m.end_pos))

        filtered = []
        for match in sorted_matches:
            # Check for overlap with existing matches
            overlaps = False
            for existing in filtered:
                if (match.start_pos < existing.end_pos and
                    match.end_pos > existing.start_pos):
                    # Overlapping - keep the higher confidence one
                    if match.confidence > existing.confidence:
                        filtered.remove(existing)
                        break
                    else:
                        overlaps = True
                        break

            if not overlaps:
                filtered.append(match)

        return filtered

    def _calculate_matches_by_type(self, matches: List[PIIMatch]) -> Dict[PIIType, int]:
        """Calculate number of matches by PII type."""
        counts = {}
        for match in matches:
            counts[match.pii_type] = counts.get(match.pii_type, 0) + 1
        return counts

    def _calculate_confidence_distribution(self, matches: List[PIIMatch]) -> Dict[ConfidenceLevel, int]:
        """Calculate distribution of confidence levels."""
        distribution = {level: 0 for level in ConfidenceLevel}

        for match in matches:
            level = self._get_confidence_level(match.confidence)
            distribution[level] += 1

        return distribution

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level for a confidence score."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _calculate_coverage_score(self, text: str, matches: List[PIIMatch]) -> float:
        """Calculate coverage score of PII detection."""
        if not text:
            return 0.0

        # Simple heuristic: ratio of detected PII characters to total characters
        pii_chars = sum(len(match.text) for match in matches)
        return min(1.0, pii_chars / len(text))

    def _calculate_average_confidence(self, matches: List[PIIMatch]) -> float:
        """Calculate average confidence score."""
        if not matches:
            return 0.0

        return sum(match.confidence for match in matches) / len(matches)

    # Validation functions
    def _validate_ssn(self, text: str) -> bool:
        """Validate SSN format and check for invalid patterns."""
        # Remove formatting
        digits = re.sub(r'[^\d]', '', text)

        if len(digits) != 9:
            return False

        # Check for invalid patterns
        invalid_patterns = [
            '000000000', '111111111', '222222222', '333333333',
            '444444444', '555555555', '666666666', '777777777',
            '888888888', '999999999', '123456789'
        ]

        return digits not in invalid_patterns

    def _validate_email(self, text: str) -> bool:
        """Validate email format."""
        # Basic validation - more sophisticated validation could be added
        parts = text.split('@')
        if len(parts) != 2:
            return False

        local, domain = parts
        if not local or not domain:
            return False

        # Check for valid domain
        domain_parts = domain.split('.')
        return len(domain_parts) >= 2 and all(part for part in domain_parts)

    def _validate_phone(self, text: str) -> bool:
        """Validate phone number format."""
        # Remove formatting
        digits = re.sub(r'[^\d]', '', text)

        # US phone numbers should be 10 digits
        if len(digits) != 10:
            return False

        # Area code shouldn't start with 0 or 1
        return digits[0] not in ['0', '1']

    def _validate_credit_card(self, text: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        # Remove formatting
        digits = re.sub(r'[^\d]', '', text)

        if len(digits) < 13 or len(digits) > 19:
            return False

        # Luhn algorithm
        def luhn_check(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]

            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10 == 0

        return luhn_check(digits)

    def _validate_ip_address(self, text: str) -> bool:
        """Validate IPv4 address."""
        parts = text.split('.')
        if len(parts) != 4:
            return False

        try:
            for part in parts:
                num = int(part)
                if num < 0 or num > 255:
                    return False
            return True
        except ValueError:
            return False

    def _validate_zip_code(self, text: str) -> bool:
        """Validate US ZIP code."""
        # Remove formatting
        digits = re.sub(r'[^\d]', '', text)

        # Should be 5 or 9 digits
        return len(digits) in [5, 9]

    def _validate_url(self, text: str) -> bool:
        """Validate URL format."""
        # Basic URL validation
        return text.startswith(('http://', 'https://')) and '.' in text

    # Redaction methods
    def redact_text(self, text: str, detection_result: Optional[PIIDetectionResult] = None) -> RedactionResult:
        """Redact PII in text based on detection results."""
        start_time = time.time()

        try:
            # Detect PII if not provided
            if detection_result is None:
                detection_result = self.detect_pii(text)

            # Check cache
            if self.detection_config.enable_caching:
                cache_key = self._generate_cache_key(text, "redaction")
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self._stats["cache_hits"] += 1
                    return cached_result
                self._stats["cache_misses"] += 1

            # Perform redaction
            redacted_text = text
            redaction_map = {}
            redactions_by_type = {}
            redactions_by_strategy = {}

            # Sort matches by position (reverse order to maintain positions)
            sorted_matches = sorted(detection_result.matches,
                                  key=lambda m: m.start_pos, reverse=True)

            for match in sorted_matches:
                # Get redaction strategy
                strategy = self.redaction_config.default_strategies.get(
                    match.pii_type, RedactionStrategy.FULL_REDACTION
                )

                # Apply redaction
                replacement = self._apply_redaction_strategy(match, strategy)

                # Replace in text
                redacted_text = (redacted_text[:match.start_pos] +
                               replacement +
                               redacted_text[match.end_pos:])

                # Update tracking
                redaction_map[match.text] = replacement
                redactions_by_type[match.pii_type] = redactions_by_type.get(match.pii_type, 0) + 1
                redactions_by_strategy[strategy] = redactions_by_strategy.get(strategy, 0) + 1

            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000

            result = RedactionResult(
                redacted_text=redacted_text,
                redaction_map=redaction_map,
                total_redactions=len(detection_result.matches),
                redactions_by_type=redactions_by_type,
                redactions_by_strategy=redactions_by_strategy,
                processing_time_ms=processing_time_ms,
                original_length=len(text),
                redacted_length=len(redacted_text),
                redaction_coverage=len(detection_result.matches) / max(1, len(detection_result.matches)),
                text_preservation=len(redacted_text) / max(1, len(text))
            )

            # Cache result
            if self.detection_config.enable_caching:
                self._cache_result(cache_key, result)

            # Update stats
            self._stats["total_redactions"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms

            logger.debug(
                "Text redaction completed",
                redactions_performed=len(detection_result.matches),
                processing_time_ms=processing_time_ms,
                original_length=len(text),
                redacted_length=len(redacted_text)
            )

            return result

        except Exception as e:
            logger.error("Text redaction failed", error=str(e))
            raise

    def _apply_redaction_strategy(self, match: PIIMatch, strategy: RedactionStrategy) -> str:
        """Apply redaction strategy to a PII match."""

        if strategy == RedactionStrategy.FULL_REDACTION:
            return "[REDACTED]"

        elif strategy == RedactionStrategy.PARTIAL_REDACTION:
            return self._partial_redaction(match)

        elif strategy == RedactionStrategy.CATEGORY_REPLACEMENT:
            return self._category_replacement(match)

        elif strategy == RedactionStrategy.ANONYMIZATION:
            return self._anonymization(match)

        elif strategy == RedactionStrategy.MASKING:
            return self._masking(match)

        elif strategy == RedactionStrategy.PLACEHOLDER:
            return self._placeholder(match)

        else:
            return "[REDACTED]"

    def _partial_redaction(self, match: PIIMatch) -> str:
        """Apply partial redaction strategy."""
        text = match.text

        if match.pii_type == PIIType.EMAIL:
            # Show first character and domain
            if '@' in text:
                local, domain = text.split('@', 1)
                if len(local) > 1:
                    return f"{local[0]}***@{domain}"
            return "***@***.***"

        elif match.pii_type == PIIType.PHONE:
            # Show last 4 digits
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"
            return "***-***-****"

        elif match.pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) >= 4:
                return f"****-****-****-{digits[-4:]}"
            return "****-****-****-****"

        else:
            # Generic partial redaction
            if len(text) <= 3:
                return "***"
            return text[:1] + "*" * (len(text) - 2) + text[-1:]

    def _category_replacement(self, match: PIIMatch) -> str:
        """Apply category replacement strategy."""
        category_map = {
            PIIType.EMAIL: "[EMAIL ADDRESS]",
            PIIType.PHONE: "[PHONE NUMBER]",
            PIIType.SSN: "[SOCIAL SECURITY NUMBER]",
            PIIType.CREDIT_CARD: "[CREDIT CARD NUMBER]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.PERSON_NAME: "[PERSON NAME]",
            PIIType.IP_ADDRESS: "[IP ADDRESS]",
            PIIType.URL: "[URL]",
            PIIType.ZIP_CODE: "[ZIP CODE]"
        }

        return category_map.get(match.pii_type, "[SENSITIVE INFORMATION]")

    def _anonymization(self, match: PIIMatch) -> str:
        """Apply anonymization strategy."""
        # Simple anonymization - could be enhanced with consistent mapping
        if match.pii_type == PIIType.PERSON_NAME:
            return "Person A"
        elif match.pii_type == PIIType.EMAIL:
            return "person.a@company.com"
        elif match.pii_type == PIIType.ADDRESS:
            return "123 Main Street, City, State"
        else:
            return "[ANONYMIZED]"

    def _masking(self, match: PIIMatch) -> str:
        """Apply masking strategy."""
        text = match.text

        if match.pii_type == PIIType.SSN:
            # XXX-XX-1234 format
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) >= 4:
                return f"XXX-XX-{digits[-4:]}"
            return "XXX-XX-XXXX"

        elif match.pii_type == PIIType.CREDIT_CARD:
            # XXXX-XXXX-XXXX-1234 format
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) >= 4:
                return f"XXXX-XXXX-XXXX-{digits[-4:]}"
            return "XXXX-XXXX-XXXX-XXXX"

        elif match.pii_type == PIIType.PHONE:
            # XXX-XXX-1234 format
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) >= 4:
                return f"XXX-XXX-{digits[-4:]}"
            return "XXX-XXX-XXXX"

        else:
            # Generic masking
            return "X" * len(text)

    def _placeholder(self, match: PIIMatch) -> str:
        """Apply placeholder strategy."""
        if not self.redaction_config.use_numbered_placeholders:
            return self._category_replacement(match)

        # Generate numbered placeholder
        pii_type_name = match.pii_type.value.upper().replace("_", "_")
        placeholder_id = hash(match.text) % 1000  # Simple ID generation

        return (f"{self.redaction_config.placeholder_prefix}"
                f"{pii_type_name}_{placeholder_id}"
                f"{self.redaction_config.placeholder_suffix}")

    # Utility methods
    def _generate_cache_key(self, text: str, operation: str) -> str:
        """Generate cache key for text and operation."""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        config_hash = hashlib.md5(
            f"{self.detection_config.json()}{self.redaction_config.json()}".encode()
        ).hexdigest()
        return f"{operation}_{text_hash}_{config_hash}"

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and not expired."""
        if cache_key not in self._cache:
            return None

        cached_item = self._cache[cache_key]
        if time.time() - cached_item["timestamp"] > self.detection_config.cache_ttl_seconds:
            del self._cache[cache_key]
            return None

        return cached_item["result"]

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache result with timestamp."""
        self._cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        # Simple cache cleanup - remove oldest entries if cache is too large
        if len(self._cache) > 1000:
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "enabled_pii_types": list(self.detection_config.enabled_pii_types),
            "patterns_loaded": sum(len(patterns) for patterns in self._patterns.values())
        }

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
        logger.info("PII detection cache cleared")


# Global service instance
_pii_detection_service: Optional[PIIDetectionService] = None


def get_pii_detection_service(
    detection_config: Optional[PIIDetectionConfig] = None,
    redaction_config: Optional[RedactionConfig] = None
) -> PIIDetectionService:
    """Get or create the global PII detection service instance."""
    global _pii_detection_service

    if _pii_detection_service is None:
        _pii_detection_service = PIIDetectionService(detection_config, redaction_config)

    return _pii_detection_service


def reset_pii_detection_service() -> None:
    """Reset the global PII detection service instance."""
    global _pii_detection_service
    _pii_detection_service = None
