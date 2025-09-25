"""
Pricing Data Extraction Service

This module provides comprehensive pricing extraction algorithms with pattern recognition,
currency detection, context analysis, validation, and confidence scoring for procurement
documents and pricing tables.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, validator

from agentic_rag.config import get_settings
from agentic_rag.models.database import UserRole
from agentic_rag.services.pricing_masking import (
    PricingType, Currency, PricingMatch, PricingPattern
)

logger = structlog.get_logger(__name__)


class ExtractionMethod(str, Enum):
    """Methods for extracting pricing data."""
    
    PATTERN_BASED = "pattern_based"
    TABLE_BASED = "table_based"
    CONTEXT_BASED = "context_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"


class TableStructure(str, Enum):
    """Types of table structures for pricing extraction."""
    
    SIMPLE_PRICING = "simple_pricing"      # Item | Price
    DETAILED_PRICING = "detailed_pricing"  # Item | Qty | Unit Price | Total
    COMPARISON_TABLE = "comparison_table"  # Multiple vendors/offers
    BREAKDOWN_TABLE = "breakdown_table"    # Cost breakdown by category
    SUMMARY_TABLE = "summary_table"        # Summary totals


class ValidationLevel(str, Enum):
    """Levels of validation for extracted pricing data."""
    
    BASIC = "basic"           # Basic format validation
    ENHANCED = "enhanced"     # Enhanced validation with context
    STRICT = "strict"         # Strict validation with cross-checks
    COMPREHENSIVE = "comprehensive"  # Full validation with ML


@dataclass
class TableCell:
    """Individual table cell with pricing information."""
    
    text: str
    row: int
    column: int
    is_header: bool = False
    cell_type: Optional[str] = None  # 'item', 'quantity', 'price', 'total', etc.
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PricingTable:
    """Detected pricing table structure."""
    
    table_id: str = field(default_factory=lambda: str(uuid4()))
    structure_type: TableStructure = TableStructure.SIMPLE_PRICING
    headers: List[str] = field(default_factory=list)
    rows: List[List[TableCell]] = field(default_factory=list)
    confidence: float = 0.0
    extraction_method: ExtractionMethod = ExtractionMethod.TABLE_BASED
    metadata: Dict[str, Any] = field(default_factory=dict)


class PricingItem(BaseModel):
    """Individual pricing item extracted from documents."""
    
    # Item identification
    item_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique item ID")
    item_name: str = Field(..., description="Name/description of the item")
    item_code: Optional[str] = Field(default=None, description="Item/product code")
    category: Optional[str] = Field(default=None, description="Item category")
    
    # Pricing information
    quantity: Optional[float] = Field(default=None, description="Quantity")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")
    unit_price: Optional[Decimal] = Field(default=None, description="Price per unit")
    total_price: Optional[Decimal] = Field(default=None, description="Total price")
    currency: Optional[Currency] = Field(default=None, description="Currency")
    
    # Context information
    vendor: Optional[str] = Field(default=None, description="Vendor/supplier name")
    document_id: Optional[str] = Field(default=None, description="Source document ID")
    page_number: Optional[int] = Field(default=None, description="Page number")
    table_id: Optional[str] = Field(default=None, description="Source table ID")
    
    # Validation and confidence
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Extraction confidence")
    validation_level: ValidationLevel = Field(
        default=ValidationLevel.BASIC,
        description="Level of validation applied"
    )
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.PATTERN_BASED,
        description="Method used for extraction"
    )
    
    # Metadata
    valid_until: Optional[datetime] = Field(default=None, description="Price validity date")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    # Timestamps
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Extraction timestamp"
    )


class PricingExtractionConfig(BaseModel):
    """Configuration for pricing extraction."""
    
    # Extraction methods
    enabled_methods: Set[ExtractionMethod] = Field(
        default_factory=lambda: {
            ExtractionMethod.PATTERN_BASED,
            ExtractionMethod.TABLE_BASED,
            ExtractionMethod.CONTEXT_BASED
        },
        description="Enabled extraction methods"
    )
    
    # Table detection
    table_detection_enabled: bool = Field(
        default=True,
        description="Enable table detection and extraction"
    )
    min_table_rows: int = Field(
        default=2,
        ge=1,
        description="Minimum rows for table detection"
    )
    min_table_columns: int = Field(
        default=2,
        ge=1,
        description="Minimum columns for table detection"
    )
    
    # Currency and amount settings
    supported_currencies: Set[Currency] = Field(
        default_factory=lambda: {
            Currency.USD, Currency.EUR, Currency.GBP, Currency.JPY,
            Currency.CAD, Currency.AUD, Currency.CHF, Currency.CNY
        },
        description="Supported currencies"
    )
    min_amount: Decimal = Field(
        default=Decimal("0.01"),
        ge=0,
        description="Minimum amount to extract"
    )
    max_amount: Decimal = Field(
        default=Decimal("1000000000"),
        ge=0,
        description="Maximum amount to extract"
    )
    
    # Validation settings
    validation_level: ValidationLevel = Field(
        default=ValidationLevel.ENHANCED,
        description="Default validation level"
    )
    require_currency: bool = Field(
        default=False,
        description="Require currency for valid pricing"
    )
    require_context: bool = Field(
        default=True,
        description="Require context for validation"
    )
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_processing_time_seconds: int = Field(
        default=30,
        description="Maximum processing time per document"
    )
    
    # Context analysis
    context_window_size: int = Field(
        default=100,
        ge=10,
        description="Context window size in characters"
    )
    context_keywords: List[str] = Field(
        default_factory=lambda: [
            "price", "cost", "amount", "total", "subtotal", "fee", "charge",
            "rate", "value", "budget", "quote", "bid", "offer", "invoice",
            "payment", "sum", "expense", "revenue", "profit", "loss"
        ],
        description="Keywords that indicate pricing context"
    )


class PricingExtractionResult(BaseModel):
    """Result of pricing extraction operation."""
    
    # Extraction results
    pricing_items: List[PricingItem] = Field(
        default_factory=list,
        description="Extracted pricing items"
    )
    pricing_tables: List[PricingTable] = Field(
        default_factory=list,
        description="Detected pricing tables"
    )
    
    # Statistics
    total_items_extracted: int = Field(default=0, description="Total items extracted")
    total_tables_detected: int = Field(default=0, description="Total tables detected")
    extraction_methods_used: List[ExtractionMethod] = Field(
        default_factory=list,
        description="Extraction methods used"
    )
    
    # Quality metrics
    average_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average extraction confidence"
    )
    validation_passed: bool = Field(
        default=False,
        description="Whether validation passed"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation errors"
    )
    
    # Performance metrics
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )
    memory_usage_mb: float = Field(
        default=0.0,
        description="Memory usage in megabytes"
    )
    
    # Currency analysis
    currencies_detected: Set[Currency] = Field(
        default_factory=set,
        description="Currencies detected in document"
    )
    currency_conversion_needed: bool = Field(
        default=False,
        description="Whether currency conversion is needed"
    )
    
    # Summary statistics
    total_value_by_currency: Dict[Currency, Decimal] = Field(
        default_factory=dict,
        description="Total value by currency"
    )
    items_by_category: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of items by category"
    )
    
    # Metadata
    document_id: Optional[str] = Field(default=None, description="Source document ID")
    extraction_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Extraction timestamp"
    )
    config_used: Optional[PricingExtractionConfig] = Field(
        default=None,
        description="Configuration used for extraction"
    )


class PricingExtractionService:
    """Service for extracting pricing data from documents."""
    
    def __init__(self, config: Optional[PricingExtractionConfig] = None):
        self.config = config or PricingExtractionConfig()
        self.settings = get_settings()
        
        # Initialize extraction patterns
        self._patterns = self._initialize_patterns()
        self._table_patterns = self._initialize_table_patterns()
        
        # Performance tracking
        self._stats = {
            "total_extractions": 0,
            "total_items_extracted": 0,
            "total_tables_detected": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_failures": 0
        }
        
        # Cache for results
        self._cache: Dict[str, Any] = {}
        
        logger.info("Pricing extraction service initialized")
    
    def _initialize_patterns(self) -> Dict[str, List[PricingPattern]]:
        """Initialize pricing extraction patterns."""
        
        patterns = {
            # Enhanced patterns for comprehensive extraction
            "monetary_values": [
                PricingPattern(
                    pricing_type=PricingType.TOTAL_PRICE,
                    pattern=r'[\$€£¥]\s*[\d,]+(?:\.\d{2})?',
                    context_keywords=["price", "cost", "total", "amount"],
                    confidence_base=0.9
                ),
                PricingPattern(
                    pricing_type=PricingType.UNIT_PRICE,
                    pattern=r'[\$€£¥]\s*[\d,]+(?:\.\d{2})?\s*(?:per|/|each)',
                    context_keywords=["unit", "per", "each", "individual"],
                    confidence_base=0.85
                ),
                PricingPattern(
                    pricing_type=PricingType.TOTAL_PRICE,
                    pattern=r'(?:USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY)\s*[\d,]+(?:\.\d{2})?',
                    context_keywords=["price", "cost", "total", "amount"],
                    confidence_base=0.8
                )
            ],

            "table_headers": [
                PricingPattern(
                    pricing_type=PricingType.TOTAL_PRICE,
                    pattern=r'(?i)\b(?:price|cost|amount|total|subtotal|value)\b',
                    context_keywords=["table", "column", "header"],
                    confidence_base=0.7
                )
            ],

            "line_items": [
                PricingPattern(
                    pricing_type=PricingType.TOTAL_PRICE,
                    pattern=r'(?i)(.+?)\s*:\s*[\$€£¥][\d,]+(?:\.\d{2})?',
                    context_keywords=["item", "product", "service"],
                    confidence_base=0.8
                )
            ]
        }
        
        return patterns

    def _initialize_table_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize table detection patterns."""

        return {
            "table_start": re.compile(r'(?i)\|.*\||\+[-+]+\+|┌.*┐'),
            "table_row": re.compile(r'\|[^|]*\|'),
            "table_separator": re.compile(r'\+[-+]+\+|├.*┤|─+'),
            "pricing_column": re.compile(r'(?i)\b(?:price|cost|amount|total|value|rate)\b'),
            "currency_column": re.compile(r'[\$€£¥]|(?i)\b(?:usd|eur|gbp|jpy|cad|aud|chf|cny)\b')
        }

    def extract_pricing_data(self, text: str, document_id: Optional[str] = None) -> PricingExtractionResult:
        """Extract pricing data from text using multiple methods."""

        start_time = time.time()

        try:
            # Check cache
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(text, document_id)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self._stats["cache_hits"] += 1
                    return cached_result
                self._stats["cache_misses"] += 1

            # Initialize result
            result = PricingExtractionResult(
                document_id=document_id,
                config_used=self.config
            )

            # Extract using enabled methods
            all_items = []
            all_tables = []
            methods_used = []

            if ExtractionMethod.TABLE_BASED in self.config.enabled_methods:
                tables, table_items = self._extract_from_tables(text)
                all_tables.extend(tables)
                all_items.extend(table_items)
                if tables:
                    methods_used.append(ExtractionMethod.TABLE_BASED)

            if ExtractionMethod.PATTERN_BASED in self.config.enabled_methods:
                pattern_items = self._extract_from_patterns(text)
                all_items.extend(pattern_items)
                if pattern_items:
                    methods_used.append(ExtractionMethod.PATTERN_BASED)

            if ExtractionMethod.CONTEXT_BASED in self.config.enabled_methods:
                context_items = self._extract_from_context(text)
                all_items.extend(context_items)
                if context_items:
                    methods_used.append(ExtractionMethod.CONTEXT_BASED)

            # Remove duplicates and validate
            unique_items = self._remove_duplicate_items(all_items)
            validated_items = self._validate_items(unique_items)

            # Calculate statistics
            currencies_detected = set()
            total_value_by_currency = {}
            items_by_category = {}

            for item in validated_items:
                if item.currency:
                    currencies_detected.add(item.currency)
                    if item.total_price:
                        total_value_by_currency[item.currency] = (
                            total_value_by_currency.get(item.currency, Decimal('0')) + item.total_price
                        )

                if item.category:
                    items_by_category[item.category] = items_by_category.get(item.category, 0) + 1

            # Calculate quality metrics
            average_confidence = (
                sum(item.confidence for item in validated_items) / len(validated_items)
                if validated_items else 0.0
            )

            # Update result
            result.pricing_items = validated_items
            result.pricing_tables = all_tables
            result.total_items_extracted = len(validated_items)
            result.total_tables_detected = len(all_tables)
            result.extraction_methods_used = methods_used
            result.average_confidence = average_confidence
            result.validation_passed = len(result.validation_errors) == 0
            result.currencies_detected = currencies_detected
            result.currency_conversion_needed = len(currencies_detected) > 1
            result.total_value_by_currency = total_value_by_currency
            result.items_by_category = items_by_category

            # Performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time_ms

            # Update statistics
            self._stats["total_extractions"] += 1
            self._stats["total_items_extracted"] += len(validated_items)
            self._stats["total_tables_detected"] += len(all_tables)
            self._stats["total_processing_time_ms"] += processing_time_ms

            # Cache result
            if self.config.enable_caching:
                self._cache_result(cache_key, result)

            logger.info(
                "Pricing extraction completed",
                document_id=document_id,
                items_extracted=len(validated_items),
                tables_detected=len(all_tables),
                processing_time_ms=processing_time_ms,
                average_confidence=average_confidence
            )

            return result

        except Exception as e:
            logger.error("Pricing extraction failed", error=str(e), document_id=document_id)
            raise

    def _extract_from_tables(self, text: str) -> Tuple[List[PricingTable], List[PricingItem]]:
        """Extract pricing data from table structures."""

        tables = []
        items = []

        try:
            # Detect table structures
            detected_tables = self._detect_tables(text)

            for table in detected_tables:
                # Extract items from table
                table_items = self._extract_items_from_table(table)
                items.extend(table_items)
                tables.append(table)

            logger.debug(f"Table extraction found {len(tables)} tables with {len(items)} items")

        except Exception as e:
            logger.error("Table extraction failed", error=str(e))

        return tables, items

    def _extract_from_patterns(self, text: str) -> List[PricingItem]:
        """Extract pricing data using pattern matching."""

        items = []

        try:
            # Extract using monetary value patterns
            for pattern_group in self._patterns.values():
                for pattern_def in pattern_group:
                    pattern_items = self._extract_with_pattern(text, pattern_def)
                    items.extend(pattern_items)

            logger.debug(f"Pattern extraction found {len(items)} items")

        except Exception as e:
            logger.error("Pattern extraction failed", error=str(e))

        return items

    def _extract_from_context(self, text: str) -> List[PricingItem]:
        """Extract pricing data using context analysis."""

        items = []

        try:
            # Find pricing contexts
            contexts = self._find_pricing_contexts(text)

            for context in contexts:
                context_items = self._extract_from_pricing_context(context)
                items.extend(context_items)

            logger.debug(f"Context extraction found {len(items)} items")

        except Exception as e:
            logger.error("Context extraction failed", error=str(e))

        return items

    def _detect_tables(self, text: str) -> List[PricingTable]:
        """Detect table structures in text."""

        tables = []
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            # Look for table start
            if self._table_patterns["table_start"].search(lines[i]):
                table = self._parse_table_from_line(lines, i)
                if table and self._is_pricing_table(table):
                    tables.append(table)
                    i += len(table.rows) + 2  # Skip processed lines
                else:
                    i += 1
            else:
                i += 1

        return tables

    def _parse_table_from_line(self, lines: List[str], start_idx: int) -> Optional[PricingTable]:
        """Parse table structure starting from a specific line."""

        try:
            table = PricingTable()
            current_line = start_idx

            # Parse header
            if current_line < len(lines):
                header_line = lines[current_line]
                if '|' in header_line:
                    headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
                    table.headers = headers
                    current_line += 1

            # Skip separator line
            if (current_line < len(lines) and
                self._table_patterns["table_separator"].search(lines[current_line])):
                current_line += 1

            # Parse data rows
            while current_line < len(lines):
                line = lines[current_line]
                if self._table_patterns["table_row"].search(line):
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    if len(cells) >= self.config.min_table_columns:
                        table_cells = []
                        for col_idx, cell_text in enumerate(cells):
                            cell = TableCell(
                                text=cell_text,
                                row=len(table.rows),
                                column=col_idx,
                                is_header=False
                            )
                            table_cells.append(cell)
                        table.rows.append(table_cells)
                    current_line += 1
                else:
                    break

            # Validate table
            if len(table.rows) >= self.config.min_table_rows:
                table.confidence = self._calculate_table_confidence(table)
                return table

        except Exception as e:
            logger.error("Table parsing failed", error=str(e))

        return None

    def _is_pricing_table(self, table: PricingTable) -> bool:
        """Determine if a table contains pricing information."""

        # Check headers for pricing indicators
        pricing_indicators = ["price", "cost", "amount", "total", "value", "rate", "$", "€", "£", "¥"]

        for header in table.headers:
            if any(indicator.lower() in header.lower() for indicator in pricing_indicators):
                return True

        # Check cell content for currency symbols
        for row in table.rows[:3]:  # Check first few rows
            for cell in row:
                if any(symbol in cell.text for symbol in ['$', '€', '£', '¥']):
                    return True

        return False

    def _calculate_table_confidence(self, table: PricingTable) -> float:
        """Calculate confidence score for table detection."""

        confidence = 0.5  # Base confidence

        # Header quality boost
        pricing_headers = sum(1 for header in table.headers
                            if any(word in header.lower()
                                 for word in ["price", "cost", "amount", "total", "value"]))
        if pricing_headers > 0:
            confidence += min(0.3, pricing_headers * 0.1)

        # Currency presence boost
        currency_cells = 0
        for row in table.rows:
            for cell in row:
                if any(symbol in cell.text for symbol in ['$', '€', '£', '¥']):
                    currency_cells += 1

        if currency_cells > 0:
            confidence += min(0.2, currency_cells * 0.02)

        return min(1.0, confidence)

    def _extract_items_from_table(self, table: PricingTable) -> List[PricingItem]:
        """Extract pricing items from a table structure."""

        items = []

        try:
            # Identify column types
            column_mapping = self._identify_table_columns(table)

            # Extract items from each row
            for row in table.rows:
                item = self._extract_item_from_table_row(row, column_mapping, table)
                if item:
                    items.append(item)

        except Exception as e:
            logger.error("Table item extraction failed", error=str(e))

        return items

    def _identify_table_columns(self, table: PricingTable) -> Dict[str, int]:
        """Identify the purpose of each table column."""

        column_mapping = {}

        for col_idx, header in enumerate(table.headers):
            header_lower = header.lower()

            if any(word in header_lower for word in ["item", "product", "description", "service"]):
                column_mapping["item_name"] = col_idx
            elif any(word in header_lower for word in ["qty", "quantity", "amount", "count"]):
                column_mapping["quantity"] = col_idx
            elif any(word in header_lower for word in ["unit", "each", "per"]):
                column_mapping["unit_price"] = col_idx
            elif any(word in header_lower for word in ["total", "sum", "subtotal"]):
                column_mapping["total_price"] = col_idx
            elif any(word in header_lower for word in ["price", "cost", "rate", "value"]):
                if "total_price" not in column_mapping:
                    column_mapping["total_price"] = col_idx
                else:
                    column_mapping["unit_price"] = col_idx
            elif any(word in header_lower for word in ["currency", "curr"]):
                column_mapping["currency"] = col_idx
            elif any(word in header_lower for word in ["vendor", "supplier", "company"]):
                column_mapping["vendor"] = col_idx

        return column_mapping

    def _extract_item_from_table_row(self, row: List[TableCell],
                                   column_mapping: Dict[str, int],
                                   table: PricingTable) -> Optional[PricingItem]:
        """Extract a pricing item from a table row."""

        try:
            item = PricingItem(
                table_id=table.table_id,
                extraction_method=ExtractionMethod.TABLE_BASED
            )

            # Extract data based on column mapping
            for field, col_idx in column_mapping.items():
                if col_idx < len(row):
                    cell_text = row[col_idx].text.strip()

                    if field == "item_name" and cell_text:
                        item.item_name = cell_text
                    elif field == "quantity" and cell_text:
                        item.quantity = self._parse_number(cell_text)
                    elif field == "unit_price" and cell_text:
                        amount, currency = self._parse_monetary_value(cell_text)
                        if amount:
                            item.unit_price = amount
                            if currency:
                                item.currency = currency
                    elif field == "total_price" and cell_text:
                        amount, currency = self._parse_monetary_value(cell_text)
                        if amount:
                            item.total_price = amount
                            if currency:
                                item.currency = currency
                    elif field == "currency" and cell_text:
                        currency = self._parse_currency(cell_text)
                        if currency:
                            item.currency = currency
                    elif field == "vendor" and cell_text:
                        item.vendor = cell_text

            # Validate item has minimum required data
            if item.item_name and (item.unit_price or item.total_price):
                item.confidence = self._calculate_item_confidence(item, table)
                return item

        except Exception as e:
            logger.error("Table row extraction failed", error=str(e))

        return None

    def _parse_monetary_value(self, text: str) -> Tuple[Optional[Decimal], Optional[Currency]]:
        """Parse monetary value from text."""

        try:
            # Remove whitespace and normalize
            text = text.strip().replace(',', '').replace(' ', '')

            # Extract currency
            currency = None
            for curr in Currency:
                symbol = self._get_currency_symbol(curr)
                if symbol in text:
                    currency = curr
                    text = text.replace(symbol, '')
                    break

            # Extract amount
            amount_match = re.search(r'[\d.]+', text)
            if amount_match:
                amount_str = amount_match.group()
                amount = Decimal(amount_str)

                # Validate amount range
                if self.config.min_amount <= amount <= self.config.max_amount:
                    return amount, currency

        except (InvalidOperation, ValueError) as e:
            logger.debug(f"Failed to parse monetary value: {text}", error=str(e))

        return None, None

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse numeric value from text."""

        try:
            # Remove non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.]', '', text.strip())
            if cleaned:
                return float(cleaned)
        except ValueError:
            pass

        return None

    def _parse_currency(self, text: str) -> Optional[Currency]:
        """Parse currency from text."""

        text_upper = text.upper().strip()

        # Direct currency code match
        for currency in Currency:
            if currency.value.upper() == text_upper:
                return currency

        # Symbol match
        currency_symbols = {
            '$': Currency.USD,
            '€': Currency.EUR,
            '£': Currency.GBP,
            '¥': Currency.JPY
        }

        for symbol, currency in currency_symbols.items():
            if symbol in text:
                return currency

        return None

    def _get_currency_symbol(self, currency: Currency) -> str:
        """Get currency symbol."""

        symbols = {
            Currency.USD: '$',
            Currency.EUR: '€',
            Currency.GBP: '£',
            Currency.JPY: '¥',
            Currency.CAD: 'C$',
            Currency.AUD: 'A$',
            Currency.CHF: 'CHF',
            Currency.CNY: '¥'
        }

        return symbols.get(currency, currency.value)

    def _extract_with_pattern(self, text: str, pattern_def: PricingPattern) -> List[PricingItem]:
        """Extract pricing items using a specific pattern."""

        items = []

        try:
            pattern = re.compile(pattern_def.pattern, re.IGNORECASE | re.MULTILINE)

            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                matched_text = match.group()

                # Parse monetary value
                amount, currency = self._parse_monetary_value(matched_text)

                # Debug logging
                logger.debug(f"Pattern match: '{matched_text}' -> amount={amount}, currency={currency}")

                if amount:
                    # Extract context
                    context = self._extract_context(text, start_pos, end_pos)

                    # Create item
                    item = PricingItem(
                        item_name=self._extract_item_name_from_context(context, matched_text),
                        total_price=amount,
                        currency=currency,
                        extraction_method=ExtractionMethod.PATTERN_BASED,
                        confidence=self._calculate_pattern_confidence(pattern_def, matched_text, context)
                    )

                    items.append(item)

        except Exception as e:
            logger.error("Pattern extraction failed", error=str(e), pattern=pattern_def.pattern)

        return items

    def _extract_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extract context around a position in text."""

        window_size = self.config.context_window_size
        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)

        return text[context_start:context_end]

    def _extract_item_name_from_context(self, context: str, matched_text: str) -> str:
        """Extract item name from context."""

        # Simple heuristic: take text before the price
        price_pos = context.find(matched_text)
        if price_pos > 0:
            before_price = context[:price_pos].strip()
            # Take the last line or sentence before price
            lines = before_price.split('\n')
            if lines:
                return lines[-1].strip()

        return "Unknown Item"

    def _find_pricing_contexts(self, text: str) -> List[str]:
        """Find contexts that likely contain pricing information."""

        contexts = []

        # Split text into paragraphs
        paragraphs = text.split('\n\n')

        for paragraph in paragraphs:
            # Check if paragraph contains pricing keywords
            if any(keyword.lower() in paragraph.lower()
                   for keyword in self.config.context_keywords):
                contexts.append(paragraph)

        return contexts

    def _extract_from_pricing_context(self, context: str) -> List[PricingItem]:
        """Extract pricing items from a pricing context."""

        items = []

        # Use pattern-based extraction on the context
        for pattern_group in self._patterns.values():
            for pattern_def in pattern_group:
                context_items = self._extract_with_pattern(context, pattern_def)
                items.extend(context_items)

        return items

    def _calculate_pattern_confidence(self, pattern_def: PricingPattern,
                                    matched_text: str, context: str) -> float:
        """Calculate confidence for pattern-based extraction."""

        confidence = pattern_def.confidence_base

        # Context keyword boost
        context_lower = context.lower()
        keyword_matches = sum(1 for keyword in pattern_def.context_keywords
                            if keyword.lower() in context_lower)
        confidence += min(0.2, keyword_matches * 0.05)

        # Currency presence boost
        if any(symbol in matched_text for symbol in ['$', '€', '£', '¥']):
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_item_confidence(self, item: PricingItem, table: Optional[PricingTable] = None) -> float:
        """Calculate confidence score for an extracted item."""

        confidence = 0.5  # Base confidence

        # Required fields boost
        if item.item_name and item.item_name != "Unknown Item":
            confidence += 0.2

        if item.total_price or item.unit_price:
            confidence += 0.2

        if item.currency:
            confidence += 0.1

        # Table context boost
        if table and table.confidence > 0.7:
            confidence += 0.1

        # Validation boost
        if item.validation_level in [ValidationLevel.ENHANCED, ValidationLevel.STRICT]:
            confidence += 0.1

        return min(1.0, confidence)

    def _remove_duplicate_items(self, items: List[PricingItem]) -> List[PricingItem]:
        """Remove duplicate pricing items."""

        unique_items = []
        seen_items = set()

        for item in items:
            # Create a signature for the item
            signature = (
                item.item_name.lower().strip(),
                item.total_price,
                item.unit_price,
                item.currency
            )

            if signature not in seen_items:
                seen_items.add(signature)
                unique_items.append(item)

        return unique_items

    def _validate_items(self, items: List[PricingItem]) -> List[PricingItem]:
        """Validate extracted pricing items."""

        validated_items = []
        validation_errors = []

        for item in items:
            try:
                # Basic validation
                if not item.item_name or item.item_name.strip() == "":
                    validation_errors.append(f"Item {item.item_id}: Missing item name")
                    continue

                if not item.total_price and not item.unit_price:
                    validation_errors.append(f"Item {item.item_id}: Missing price information")
                    continue

                # Enhanced validation
                if self.config.validation_level in [ValidationLevel.ENHANCED, ValidationLevel.STRICT]:
                    if self.config.require_currency and not item.currency:
                        validation_errors.append(f"Item {item.item_id}: Missing currency")
                        continue

                    # Validate price ranges
                    if item.total_price and (item.total_price < self.config.min_amount or
                                           item.total_price > self.config.max_amount):
                        validation_errors.append(f"Item {item.item_id}: Price out of range")
                        continue

                # Set validation level
                item.validation_level = self.config.validation_level
                validated_items.append(item)

            except Exception as e:
                validation_errors.append(f"Item {item.item_id}: Validation error - {str(e)}")

        if validation_errors:
            self._stats["validation_failures"] += len(validation_errors)
            logger.warning(f"Validation failed for {len(validation_errors)} items")

        return validated_items

    def _generate_cache_key(self, text: str, document_id: Optional[str] = None) -> str:
        """Generate cache key for extraction result."""

        import hashlib

        content = f"{text[:1000]}{document_id or ''}{str(self.config.dict())}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[PricingExtractionResult]:
        """Get cached extraction result."""

        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if cached_data.get("expires_at", 0) > time.time():
                return cached_data.get("result")

        return None

    def _cache_result(self, cache_key: str, result: PricingExtractionResult) -> None:
        """Cache extraction result."""

        self._cache[cache_key] = {
            "result": result,
            "expires_at": time.time() + self.config.cache_ttl_seconds
        }

        # Simple cache cleanup
        if len(self._cache) > 1000:
            # Remove oldest entries
            current_time = time.time()
            expired_keys = [
                key for key, data in self._cache.items()
                if data.get("expires_at", 0) <= current_time
            ]
            for key in expired_keys:
                del self._cache[key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""

        return {
            **self._stats,
            "cache_size": len(self._cache),
            "config": self.config.dict(),
            "supported_currencies": [c.value for c in self.config.supported_currencies],
            "enabled_methods": [m.value for m in self.config.enabled_methods]
        }

    def clear_cache(self) -> None:
        """Clear the extraction cache."""

        self._cache.clear()
        logger.info("Pricing extraction cache cleared")


# Global service instance
_pricing_extraction_service: Optional[PricingExtractionService] = None


def get_pricing_extraction_service() -> PricingExtractionService:
    """Get or create the global pricing extraction service instance."""
    global _pricing_extraction_service

    if _pricing_extraction_service is None:
        _pricing_extraction_service = PricingExtractionService()

    return _pricing_extraction_service


def reset_pricing_extraction_service() -> None:
    """Reset the global pricing extraction service instance."""
    global _pricing_extraction_service
    _pricing_extraction_service = None
