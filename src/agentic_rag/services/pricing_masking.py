"""
Pricing Information Masking Service

This module provides specialized masking for pricing and financial information
with pattern detection, currency masking, partial disclosure strategies,
and context preservation for procurement scenarios.
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
from agentic_rag.models.database import UserRole

logger = structlog.get_logger(__name__)


class PricingType(str, Enum):
    """Types of pricing information that can be detected."""
    
    # Basic pricing
    UNIT_PRICE = "unit_price"
    TOTAL_PRICE = "total_price"
    SUBTOTAL = "subtotal"
    TAX_AMOUNT = "tax_amount"
    DISCOUNT = "discount"
    
    # Financial terms
    BUDGET = "budget"
    COST = "cost"
    FEE = "fee"
    RATE = "rate"
    COMMISSION = "commission"
    
    # Procurement specific
    BID_AMOUNT = "bid_amount"
    CONTRACT_VALUE = "contract_value"
    PAYMENT_TERMS = "payment_terms"
    PENALTY_AMOUNT = "penalty_amount"
    
    # Financial metrics
    MARGIN = "margin"
    PROFIT = "profit"
    ROI = "roi"
    NPV = "npv"


class Currency(str, Enum):
    """Supported currencies."""
    
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    CNY = "CNY"


class MaskingStrategy(str, Enum):
    """Strategies for masking pricing information."""
    
    FULL_MASK = "full_mask"           # $XX,XXX
    RANGE_MASK = "range_mask"         # $10K - $50K
    CATEGORY_MASK = "category_mask"   # [HIGH VALUE]
    PERCENTAGE_MASK = "percentage_mask"  # XX% of budget
    RELATIVE_MASK = "relative_mask"   # Above/Below market rate
    PARTIAL_REVEAL = "partial_reveal" # $1X,XXX (partial digits)


class DisclosureLevel(str, Enum):
    """Levels of pricing disclosure."""
    
    NONE = "none"           # No pricing information shown
    CATEGORY = "category"   # High/Medium/Low categories
    RANGE = "range"         # Price ranges
    APPROXIMATE = "approximate"  # Approximate values
    EXACT = "exact"         # Exact values


@dataclass
class PricingPattern:
    """Pattern for detecting pricing information."""
    
    pricing_type: PricingType
    pattern: str
    currency_pattern: Optional[str] = None
    context_keywords: List[str] = field(default_factory=list)
    confidence_base: float = 0.8
    requires_currency: bool = True


@dataclass
class PricingMatch:
    """Detected pricing information match."""
    
    pricing_type: PricingType
    text: str
    amount: Optional[float] = None
    currency: Optional[Currency] = None
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 0.0
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PricingMaskingConfig(BaseModel):
    """Configuration for pricing masking."""
    
    # Detection settings
    enabled_pricing_types: Set[PricingType] = Field(
        default_factory=lambda: set(PricingType),
        description="Pricing types to detect and mask"
    )
    supported_currencies: Set[Currency] = Field(
        default_factory=lambda: {Currency.USD, Currency.EUR, Currency.GBP},
        description="Currencies to recognize"
    )
    min_amount_threshold: float = Field(
        default=0.01,
        ge=0.0,
        description="Minimum amount to consider for masking"
    )
    max_amount_threshold: float = Field(
        default=1000000000.0,
        ge=0.0,
        description="Maximum amount to consider valid"
    )
    
    # Masking strategies by role
    role_strategies: Dict[UserRole, MaskingStrategy] = Field(
        default_factory=lambda: {
            UserRole.VIEWER: MaskingStrategy.FULL_MASK,
            UserRole.ANALYST: MaskingStrategy.RANGE_MASK,
            UserRole.ADMIN: MaskingStrategy.PARTIAL_REVEAL
        },
        description="Masking strategies by user role"
    )
    
    # Disclosure levels by role
    role_disclosure: Dict[UserRole, DisclosureLevel] = Field(
        default_factory=lambda: {
            UserRole.VIEWER: DisclosureLevel.CATEGORY,
            UserRole.ANALYST: DisclosureLevel.RANGE,
            UserRole.ADMIN: DisclosureLevel.EXACT
        },
        description="Disclosure levels by user role"
    )
    
    # Context preservation
    preserve_context: bool = Field(
        default=True,
        description="Whether to preserve pricing context"
    )
    context_window_size: int = Field(
        default=30,
        ge=0,
        le=100,
        description="Context window size for analysis"
    )
    
    # Range settings
    range_categories: Dict[str, Tuple[float, float]] = Field(
        default_factory=lambda: {
            "low": (0, 1000),
            "medium": (1000, 10000),
            "high": (10000, 100000),
            "very_high": (100000, float('inf'))
        },
        description="Price range categories"
    )
    
    # Performance settings
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching"
    )
    cache_ttl_seconds: int = Field(
        default=1800,
        ge=60,
        description="Cache TTL in seconds"
    )


class PricingMaskingResult(BaseModel):
    """Result of pricing masking operation."""
    
    # Masked content
    masked_text: str = Field(..., description="Text with pricing information masked")
    
    # Detection results
    pricing_matches: List[PricingMatch] = Field(
        default_factory=list,
        description="Detected pricing matches"
    )
    total_matches: int = Field(default=0, description="Total pricing matches found")
    
    # Masking statistics
    maskings_by_type: Dict[PricingType, int] = Field(
        default_factory=dict,
        description="Number of maskings by pricing type"
    )
    maskings_by_currency: Dict[Currency, int] = Field(
        default_factory=dict,
        description="Number of maskings by currency"
    )
    
    # Processing metadata
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )
    masking_strategy_used: MaskingStrategy = Field(
        default=MaskingStrategy.FULL_MASK,
        description="Primary masking strategy used"
    )
    disclosure_level_used: DisclosureLevel = Field(
        default=DisclosureLevel.NONE,
        description="Disclosure level applied"
    )
    
    # Quality metrics
    context_preservation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How well context was preserved"
    )
    readability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Readability of masked text"
    )


class PricingMaskingService:
    """Service for detecting and masking pricing information."""
    
    def __init__(self, config: Optional[PricingMaskingConfig] = None):
        self.config = config or PricingMaskingConfig()
        self.settings = get_settings()
        
        # Initialize pricing patterns
        self._patterns = self._initialize_patterns()
        
        # Performance tracking
        self._stats = {
            "total_maskings": 0,
            "total_detections": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Cache for results
        self._cache: Dict[str, Any] = {}
        
        logger.info("Pricing masking service initialized")
    
    def _initialize_patterns(self) -> Dict[PricingType, List[PricingPattern]]:
        """Initialize pricing detection patterns."""
        
        patterns = {
            # Generic price patterns that catch most monetary values
            PricingType.TOTAL_PRICE: [
                PricingPattern(
                    pricing_type=PricingType.TOTAL_PRICE,
                    pattern=r'\$[\d,]+(?:\.\d{2})?',
                    context_keywords=["cost", "price", "total", "amount"],
                    confidence_base=0.8
                ),
                PricingPattern(
                    pricing_type=PricingType.TOTAL_PRICE,
                    pattern=r'(?:€|£|¥)[\d,]+(?:\.\d{2})?',
                    context_keywords=["cost", "price", "total", "amount"],
                    confidence_base=0.8
                )
            ],

            PricingType.UNIT_PRICE: [
                PricingPattern(
                    pricing_type=PricingType.UNIT_PRICE,
                    pattern=r'\$[\d,]+(?:\.\d{2})?\s*(?:per|each|/)',
                    context_keywords=["unit price", "per unit", "each", "price per"],
                    confidence_base=0.9
                ),
                PricingPattern(
                    pricing_type=PricingType.UNIT_PRICE,
                    pattern=r'(?:€|£|¥)[\d,]+(?:\.\d{2})?\s*(?:per|each|/)',
                    context_keywords=["unit price", "per unit", "each"],
                    confidence_base=0.9
                )
            ],

            PricingType.BID_AMOUNT: [
                PricingPattern(
                    pricing_type=PricingType.BID_AMOUNT,
                    pattern=r'(?:bid|proposal)\s*:?\s*\$[\d,]+(?:\.\d{2})?',
                    context_keywords=["bid", "proposal", "offer", "quote"],
                    confidence_base=0.95
                ),
                PricingPattern(
                    pricing_type=PricingType.BID_AMOUNT,
                    pattern=r'\$[\d,]+(?:\.\d{2})?\s*(?:bid|proposal|offer)',
                    context_keywords=["bid", "proposal", "offer"],
                    confidence_base=0.9
                )
            ],

            PricingType.BUDGET: [
                PricingPattern(
                    pricing_type=PricingType.BUDGET,
                    pattern=r'(?:budget|allocated|allocation)\s*:?\s*\$[\d,]+(?:\.\d{2})?',
                    context_keywords=["budget", "allocated", "allocation", "budgeted"],
                    confidence_base=0.9
                )
            ],

            PricingType.DISCOUNT: [
                PricingPattern(
                    pricing_type=PricingType.DISCOUNT,
                    pattern=r'(?:discount|savings?|off)\s*:?\s*\$[\d,]+(?:\.\d{2})?',
                    context_keywords=["discount", "savings", "off", "reduction"],
                    confidence_base=0.85
                ),
                PricingPattern(
                    pricing_type=PricingType.DISCOUNT,
                    pattern=r'\d+%\s*(?:discount|off|savings?)',
                    context_keywords=["discount", "off", "savings"],
                    confidence_base=0.9,
                    requires_currency=False
                )
            ]
        }
        
        return patterns

    def detect_pricing(self, text: str) -> List[PricingMatch]:
        """Detect pricing information in text."""

        matches = []

        for pricing_type in self.config.enabled_pricing_types:
            if pricing_type in self._patterns:
                type_matches = self._detect_pricing_type(text, pricing_type)
                matches.extend(type_matches)

        # Remove overlapping matches
        filtered_matches = self._remove_overlapping_matches(matches)

        return filtered_matches

    def mask_pricing(self, text: str, user_role: UserRole) -> PricingMaskingResult:
        """Mask pricing information in text based on user role."""
        start_time = time.time()

        try:
            # Check cache
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(text, user_role)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self._stats["cache_hits"] += 1
                    return cached_result
                self._stats["cache_misses"] += 1

            # Detect pricing information
            pricing_matches = self.detect_pricing(text)

            # Get masking strategy for role
            strategy = self.config.role_strategies.get(user_role, MaskingStrategy.FULL_MASK)
            disclosure_level = self.config.role_disclosure.get(user_role, DisclosureLevel.NONE)

            # Apply masking
            masked_text = text
            maskings_by_type = {}
            maskings_by_currency = {}

            # Sort matches by position (reverse order to maintain positions)
            sorted_matches = sorted(pricing_matches, key=lambda m: m.start_pos, reverse=True)

            for match in sorted_matches:
                # Apply masking strategy
                masked_value = self._apply_masking_strategy(match, strategy, disclosure_level)

                # Replace in text
                masked_text = (masked_text[:match.start_pos] +
                             masked_value +
                             masked_text[match.end_pos:])

                # Update statistics
                maskings_by_type[match.pricing_type] = maskings_by_type.get(match.pricing_type, 0) + 1
                if match.currency:
                    maskings_by_currency[match.currency] = maskings_by_currency.get(match.currency, 0) + 1

            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000

            result = PricingMaskingResult(
                masked_text=masked_text,
                pricing_matches=pricing_matches,
                total_matches=len(pricing_matches),
                maskings_by_type=maskings_by_type,
                maskings_by_currency=maskings_by_currency,
                processing_time_ms=processing_time_ms,
                masking_strategy_used=strategy,
                disclosure_level_used=disclosure_level,
                context_preservation_score=self._calculate_context_preservation(text, masked_text),
                readability_score=self._calculate_readability_score(masked_text)
            )

            # Cache result
            if self.config.enable_caching:
                self._cache_result(cache_key, result)

            # Update stats
            self._stats["total_maskings"] += 1
            self._stats["total_detections"] += len(pricing_matches)
            self._stats["total_processing_time_ms"] += processing_time_ms

            logger.debug(
                "Pricing masking completed",
                user_role=user_role.value,
                matches_found=len(pricing_matches),
                strategy_used=strategy.value,
                processing_time_ms=processing_time_ms
            )

            return result

        except Exception as e:
            logger.error("Pricing masking failed", error=str(e))
            raise

    def _detect_pricing_type(self, text: str, pricing_type: PricingType) -> List[PricingMatch]:
        """Detect pricing information of a specific type."""

        matches = []

        if pricing_type not in self._patterns:
            return matches

        for pattern_def in self._patterns[pricing_type]:
            # Compile pattern
            pattern = re.compile(pattern_def.pattern, re.IGNORECASE)

            # Find all matches
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                matched_text = match.group()

                # Extract amount and currency
                amount, currency = self._extract_amount_and_currency(matched_text)

                # Validate amount
                if amount is not None:
                    if amount < self.config.min_amount_threshold or amount > self.config.max_amount_threshold:
                        continue

                # Get context
                context = self._extract_context(text, start_pos, end_pos)

                # Calculate confidence
                confidence = self._calculate_confidence(pattern_def, matched_text, context)

                # Create match
                pricing_match = PricingMatch(
                    pricing_type=pricing_type,
                    text=matched_text,
                    amount=amount,
                    currency=currency,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=confidence,
                    context=context,
                    metadata={
                        "pattern_used": pattern_def.pattern,
                        "context_keywords_found": self._find_context_keywords(
                            context, pattern_def.context_keywords
                        )
                    }
                )

                matches.append(pricing_match)

        return matches

    def _extract_amount_and_currency(self, text: str) -> Tuple[Optional[float], Optional[Currency]]:
        """Extract amount and currency from pricing text."""

        # Currency symbols mapping
        currency_symbols = {
            '$': Currency.USD,
            '€': Currency.EUR,
            '£': Currency.GBP,
            '¥': Currency.JPY
        }

        # Find currency
        currency = None
        for symbol, curr in currency_symbols.items():
            if symbol in text:
                currency = curr
                break

        # Extract numeric amount
        amount = None
        # Remove currency symbols and extract numbers
        numeric_text = re.sub(r'[^\d.,]', '', text)
        numeric_text = numeric_text.replace(',', '')

        try:
            if numeric_text:
                amount = float(numeric_text)
        except ValueError:
            pass

        return amount, currency

    def _extract_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extract context around a pricing match."""
        window_size = self.config.context_window_size

        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)

        return text[context_start:context_end]

    def _calculate_confidence(self, pattern_def: PricingPattern,
                            matched_text: str, context: str) -> float:
        """Calculate confidence score for a pricing match."""

        base_confidence = pattern_def.confidence_base

        # Context keyword boost
        context_boost = 0.0
        if pattern_def.context_keywords:
            found_keywords = self._find_context_keywords(
                context.lower(), pattern_def.context_keywords
            )
            if found_keywords:
                context_boost = min(0.2, len(found_keywords) * 0.05)

        # Currency presence boost
        currency_boost = 0.0
        if pattern_def.requires_currency:
            if any(symbol in matched_text for symbol in ['$', '€', '£', '¥']):
                currency_boost = 0.1

        # Calculate final confidence
        confidence = base_confidence + context_boost + currency_boost
        return max(0.0, min(1.0, confidence))

    def _find_context_keywords(self, context: str, keywords: List[str]) -> List[str]:
        """Find context keywords in the given context."""
        found = []
        context_lower = context.lower()

        for keyword in keywords:
            if keyword.lower() in context_lower:
                found.append(keyword)

        return found

    def _remove_overlapping_matches(self, matches: List[PricingMatch]) -> List[PricingMatch]:
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

    def _apply_masking_strategy(self, match: PricingMatch, strategy: MaskingStrategy,
                               disclosure_level: DisclosureLevel) -> str:
        """Apply masking strategy to a pricing match."""

        if strategy == MaskingStrategy.FULL_MASK:
            return self._full_mask(match)

        elif strategy == MaskingStrategy.RANGE_MASK:
            return self._range_mask(match)

        elif strategy == MaskingStrategy.CATEGORY_MASK:
            return self._category_mask(match)

        elif strategy == MaskingStrategy.PERCENTAGE_MASK:
            return self._percentage_mask(match)

        elif strategy == MaskingStrategy.RELATIVE_MASK:
            return self._relative_mask(match)

        elif strategy == MaskingStrategy.PARTIAL_REVEAL:
            return self._partial_reveal(match, disclosure_level)

        else:
            return self._full_mask(match)

    def _full_mask(self, match: PricingMatch) -> str:
        """Apply full masking to pricing information."""

        if match.currency:
            currency_symbol = self._get_currency_symbol(match.currency)
            return f"{currency_symbol}XX,XXX"
        else:
            return "[PRICE REDACTED]"

    def _range_mask(self, match: PricingMatch) -> str:
        """Apply range masking to pricing information."""

        if match.amount is None:
            return "[PRICE RANGE]"

        # Determine range category
        range_category = self._get_range_category(match.amount)

        if match.currency:
            currency_symbol = self._get_currency_symbol(match.currency)

            if range_category == "low":
                return f"{currency_symbol}1K - {currency_symbol}10K"
            elif range_category == "medium":
                return f"{currency_symbol}10K - {currency_symbol}100K"
            elif range_category == "high":
                return f"{currency_symbol}100K - {currency_symbol}1M"
            else:
                return f"{currency_symbol}1M+"
        else:
            return f"[{range_category.upper()} VALUE]"

    def _category_mask(self, match: PricingMatch) -> str:
        """Apply category masking to pricing information."""

        if match.amount is None:
            return "[PRICING INFORMATION]"

        range_category = self._get_range_category(match.amount)

        category_labels = {
            "low": "[LOW VALUE]",
            "medium": "[MEDIUM VALUE]",
            "high": "[HIGH VALUE]",
            "very_high": "[VERY HIGH VALUE]"
        }

        return category_labels.get(range_category, "[PRICING INFORMATION]")

    def _percentage_mask(self, match: PricingMatch) -> str:
        """Apply percentage masking to pricing information."""

        # This would require context about budget or baseline
        # For now, return a generic percentage mask based on amount if available
        if match.amount and match.amount > 0:
            # Simple categorization for percentage representation
            if match.amount < 1000:
                return "< 5% of budget"
            elif match.amount < 10000:
                return "5-15% of budget"
            else:
                return "> 15% of budget"

        return "XX% of budget"

    def _relative_mask(self, match: PricingMatch) -> str:
        """Apply relative masking to pricing information."""

        if match.amount is None:
            return "[MARKET RATE]"

        # Simple relative categorization
        if match.amount < 1000:
            return "Below market rate"
        elif match.amount > 50000:
            return "Above market rate"
        else:
            return "Market rate"

    def _partial_reveal(self, match: PricingMatch, disclosure_level: DisclosureLevel) -> str:
        """Apply partial reveal based on disclosure level."""

        if disclosure_level == DisclosureLevel.NONE:
            return self._full_mask(match)

        elif disclosure_level == DisclosureLevel.CATEGORY:
            return self._category_mask(match)

        elif disclosure_level == DisclosureLevel.RANGE:
            return self._range_mask(match)

        elif disclosure_level == DisclosureLevel.APPROXIMATE:
            return self._approximate_reveal(match)

        elif disclosure_level == DisclosureLevel.EXACT:
            return match.text  # Return original text

        else:
            return self._full_mask(match)

    def _approximate_reveal(self, match: PricingMatch) -> str:
        """Reveal approximate pricing information."""

        if match.amount is None:
            return "[APPROXIMATE PRICE]"

        # Round to nearest significant figure
        if match.amount < 100:
            rounded = round(match.amount, -1)  # Round to nearest 10
        elif match.amount < 1000:
            rounded = round(match.amount, -2)  # Round to nearest 100
        elif match.amount < 10000:
            rounded = round(match.amount, -3)  # Round to nearest 1000
        else:
            rounded = round(match.amount, -4)  # Round to nearest 10000

        if match.currency:
            currency_symbol = self._get_currency_symbol(match.currency)
            return f"~{currency_symbol}{rounded:,.0f}"
        else:
            return f"~{rounded:,.0f}"

    def _get_currency_symbol(self, currency: Currency) -> str:
        """Get currency symbol for display."""

        symbols = {
            Currency.USD: "$",
            Currency.EUR: "€",
            Currency.GBP: "£",
            Currency.JPY: "¥",
            Currency.CAD: "C$",
            Currency.AUD: "A$",
            Currency.CHF: "CHF",
            Currency.CNY: "¥"
        }

        return symbols.get(currency, "$")

    def _get_range_category(self, amount: float) -> str:
        """Get range category for an amount."""

        for category, (min_val, max_val) in self.config.range_categories.items():
            if min_val <= amount < max_val:
                return category

        return "very_high"

    def _calculate_context_preservation(self, original_text: str, masked_text: str) -> float:
        """Calculate how well context was preserved during masking."""

        if not self.config.preserve_context:
            return 0.0

        # Simple metric: ratio of preserved non-pricing words
        original_words = set(original_text.lower().split())
        masked_words = set(masked_text.lower().split())

        # Remove common pricing-related words
        pricing_words = {"price", "cost", "total", "amount", "budget", "bid", "offer"}
        original_words -= pricing_words
        masked_words -= pricing_words

        if not original_words:
            return 1.0

        preserved_words = original_words.intersection(masked_words)
        return len(preserved_words) / len(original_words)

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score of masked text."""

        # Simple readability metric based on sentence structure preservation
        sentences = text.split('.')
        readable_sentences = 0

        for sentence in sentences:
            # Check if sentence has reasonable structure
            words = sentence.strip().split()
            if 3 <= len(words) <= 50:  # Reasonable sentence length
                readable_sentences += 1

        if not sentences:
            return 0.0

        return readable_sentences / len(sentences)

    # Utility methods
    def _generate_cache_key(self, text: str, user_role: UserRole) -> str:
        """Generate cache key for text and user role."""
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()
        role_hash = hashlib.md5(user_role.value.encode()).hexdigest()
        config_hash = hashlib.md5(self.config.model_dump_json().encode()).hexdigest()

        return f"pricing_{text_hash}_{role_hash}_{config_hash}"

    def _get_cached_result(self, cache_key: str) -> Optional[PricingMaskingResult]:
        """Get cached result if available and not expired."""
        if cache_key not in self._cache:
            return None

        cached_item = self._cache[cache_key]
        if time.time() - cached_item["timestamp"] > self.config.cache_ttl_seconds:
            del self._cache[cache_key]
            return None

        return cached_item["result"]

    def _cache_result(self, cache_key: str, result: PricingMaskingResult) -> None:
        """Cache result with timestamp."""
        self._cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        # Simple cache cleanup
        if len(self._cache) > 500:
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "enabled_pricing_types": list(self.config.enabled_pricing_types),
            "supported_currencies": list(self.config.supported_currencies),
            "patterns_loaded": sum(len(patterns) for patterns in self._patterns.values())
        }

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
        logger.info("Pricing masking cache cleared")


# Global service instance
_pricing_masking_service: Optional[PricingMaskingService] = None


def get_pricing_masking_service(config: Optional[PricingMaskingConfig] = None) -> PricingMaskingService:
    """Get or create the global pricing masking service instance."""
    global _pricing_masking_service

    if _pricing_masking_service is None:
        _pricing_masking_service = PricingMaskingService(config)

    return _pricing_masking_service


def reset_pricing_masking_service() -> None:
    """Reset the global pricing masking service instance."""
    global _pricing_masking_service
    _pricing_masking_service = None
