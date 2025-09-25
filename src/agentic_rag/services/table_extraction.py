"""
Table Extraction and Analysis Service

This module implements advanced table extraction and analysis capabilities including:
- Table structure recognition
- Data type detection and parsing
- Table relationship analysis
- Table comparison algorithms
- Table data validation
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
import pandas as pd
from datetime import datetime
import numpy as np

logger = structlog.get_logger(__name__)


class DataType(str, Enum):
    """Data types for table cells."""
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    UNKNOWN = "unknown"


class TableStructure(str, Enum):
    """Table structure types."""
    SIMPLE = "simple"           # Basic rows and columns
    HIERARCHICAL = "hierarchical"  # Multi-level headers
    PIVOT = "pivot"            # Pivot table format
    MATRIX = "matrix"          # Matrix format
    LIST = "list"              # List-like structure
    COMPLEX = "complex"        # Complex nested structure


class ValidationLevel(str, Enum):
    """Validation levels for table data."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


@dataclass
class TableCell:
    """Represents a single table cell."""
    row: int
    column: int
    value: Any
    data_type: DataType
    formatted_value: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class TableColumn:
    """Represents a table column."""
    index: int
    name: str
    data_type: DataType
    values: List[Any]
    null_count: int
    unique_count: int
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class TableRow:
    """Represents a table row."""
    index: int
    cells: List[TableCell]
    metadata: Dict[str, Any]


class ExtractedTable(BaseModel):
    """Represents an extracted table."""
    table_id: str = Field(default_factory=lambda: str(uuid4()))
    title: Optional[str] = None
    structure_type: TableStructure
    headers: List[str]
    columns: List[TableColumn]
    rows: List[TableRow]
    row_count: int
    column_count: int
    data_types: Dict[str, DataType]
    relationships: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]


class TableComparisonResult(BaseModel):
    """Result of table comparison."""
    comparison_id: str = Field(default_factory=lambda: str(uuid4()))
    table1_id: str
    table2_id: str
    structure_similarity: float
    data_similarity: float
    schema_differences: List[Dict[str, Any]]
    data_differences: List[Dict[str, Any]]
    added_rows: List[int]
    removed_rows: List[int]
    modified_rows: List[int]
    confidence: float
    processing_time_ms: float


class TableExtractionConfig(BaseModel):
    """Configuration for table extraction."""
    min_rows: int = 2
    min_columns: int = 2
    max_rows: int = 10000
    max_columns: int = 100
    header_detection_threshold: float = 0.7
    data_type_confidence_threshold: float = 0.8
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_relationship_analysis: bool = True
    enable_statistical_analysis: bool = True


class TableExtractionService:
    """Service for advanced table extraction and analysis."""
    
    def __init__(self, config: Optional[TableExtractionConfig] = None):
        self.config = config or TableExtractionConfig()
        
        # Statistics tracking
        self._stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_processing_time_ms": 0.0,
            "average_processing_time_ms": 0.0,
            "total_tables_extracted": 0,
            "total_rows_processed": 0,
            "total_columns_processed": 0
        }
        
        # Cache for repeated extractions
        self._extraction_cache: Dict[str, List[ExtractedTable]] = {}
        
        logger.info("Table extraction service initialized")
    
    async def extract_tables_from_text(self, content: str) -> List[ExtractedTable]:
        """Extract tables from text content."""
        
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = f"text:{hash(content)}"
            
            # Check cache
            if cache_key in self._extraction_cache:
                logger.info("Returning cached table extraction result")
                return self._extraction_cache[cache_key]
            
            # Detect table patterns in text
            table_patterns = self._detect_table_patterns(content)
            
            extracted_tables = []
            
            for i, pattern in enumerate(table_patterns):
                # Extract table structure
                table_data = self._parse_table_pattern(pattern, i)
                
                if table_data:
                    # Analyze structure
                    structure_type = self._analyze_table_structure(table_data)
                    
                    # Detect data types
                    columns = self._analyze_columns(table_data)
                    
                    # Create table rows
                    rows = self._create_table_rows(table_data, columns)
                    
                    # Analyze relationships
                    relationships = []
                    if self.config.enable_relationship_analysis:
                        relationships = self._analyze_relationships(table_data, columns)
                    
                    # Validate data
                    validation_results = self._validate_table_data(table_data, columns)
                    
                    # Calculate confidence
                    confidence = self._calculate_extraction_confidence(table_data, columns, validation_results)
                    
                    # Create extracted table
                    extracted_table = ExtractedTable(
                        title=pattern.get('title'),
                        structure_type=structure_type,
                        headers=[col.name for col in columns],
                        columns=columns,
                        rows=rows,
                        row_count=len(rows),
                        column_count=len(columns),
                        data_types={col.name: col.data_type for col in columns},
                        relationships=relationships,
                        validation_results=validation_results,
                        confidence=confidence,
                        metadata={
                            "source": "text_extraction",
                            "pattern_index": i,
                            "extraction_method": pattern.get('method', 'unknown')
                        }
                    )
                    
                    extracted_tables.append(extracted_table)
            
            # Cache result
            self._extraction_cache[cache_key] = extracted_tables
            
            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats("success", processing_time_ms, len(extracted_tables))
            
            logger.info(
                "Table extraction completed",
                tables_extracted=len(extracted_tables),
                processing_time_ms=processing_time_ms
            )
            
            return extracted_tables
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats("failure", processing_time_ms, 0)
            
            logger.error(
                "Table extraction failed",
                error=str(e),
                processing_time_ms=processing_time_ms
            )
            raise
    
    def _detect_table_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Detect table patterns in text content."""
        
        patterns = []
        lines = content.split('\n')
        
        # Pattern 1: Pipe-separated tables (|col1|col2|col3|)
        pipe_tables = self._detect_pipe_tables(lines)
        patterns.extend(pipe_tables)
        
        # Pattern 2: Tab-separated tables
        tab_tables = self._detect_tab_tables(lines)
        patterns.extend(tab_tables)
        
        # Pattern 3: Space-aligned tables
        space_tables = self._detect_space_tables(lines)
        patterns.extend(space_tables)
        
        # Pattern 4: CSV-like tables
        csv_tables = self._detect_csv_tables(lines)
        patterns.extend(csv_tables)
        
        return patterns
    
    def _detect_pipe_tables(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect pipe-separated tables."""
        
        tables = []
        current_table = []
        in_table = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line looks like a table row
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    in_table = True
                    current_table = []
                
                # Parse row
                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty cells at start/end
                if cells and not cells[0]:
                    cells = cells[1:]
                if cells and not cells[-1]:
                    cells = cells[:-1]
                
                if len(cells) >= self.config.min_columns:
                    current_table.append(cells)
            else:
                if in_table and len(current_table) >= self.config.min_rows:
                    # End of table
                    tables.append({
                        'data': current_table,
                        'method': 'pipe_separated',
                        'start_line': i - len(current_table),
                        'end_line': i - 1
                    })
                in_table = False
                current_table = []
        
        # Check for table at end of content
        if in_table and len(current_table) >= self.config.min_rows:
            tables.append({
                'data': current_table,
                'method': 'pipe_separated',
                'start_line': len(lines) - len(current_table),
                'end_line': len(lines) - 1
            })
        
        return tables

    def _detect_tab_tables(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect tab-separated tables."""

        tables = []
        current_table = []
        in_table = False

        for i, line in enumerate(lines):
            line = line.strip()

            # Check if line has multiple tabs
            if '\t' in line and line.count('\t') >= 1:
                if not in_table:
                    in_table = True
                    current_table = []

                cells = [cell.strip() for cell in line.split('\t')]
                if len(cells) >= self.config.min_columns:
                    current_table.append(cells)
            else:
                if in_table and len(current_table) >= self.config.min_rows:
                    tables.append({
                        'data': current_table,
                        'method': 'tab_separated',
                        'start_line': i - len(current_table),
                        'end_line': i - 1
                    })
                in_table = False
                current_table = []

        if in_table and len(current_table) >= self.config.min_rows:
            tables.append({
                'data': current_table,
                'method': 'tab_separated',
                'start_line': len(lines) - len(current_table),
                'end_line': len(lines) - 1
            })

        return tables

    def _detect_space_tables(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect space-aligned tables."""

        tables = []
        # This is more complex - look for consistent column alignment
        # For now, simplified implementation
        return tables

    def _detect_csv_tables(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect CSV-like tables."""

        tables = []
        current_table = []
        in_table = False

        for i, line in enumerate(lines):
            line = line.strip()

            # Check if line has commas and looks like CSV
            if ',' in line and line.count(',') >= 1:
                if not in_table:
                    in_table = True
                    current_table = []

                # Simple CSV parsing (doesn't handle quoted commas)
                cells = [cell.strip() for cell in line.split(',')]
                if len(cells) >= self.config.min_columns:
                    current_table.append(cells)
            else:
                if in_table and len(current_table) >= self.config.min_rows:
                    tables.append({
                        'data': current_table,
                        'method': 'csv_like',
                        'start_line': i - len(current_table),
                        'end_line': i - 1
                    })
                in_table = False
                current_table = []

        if in_table and len(current_table) >= self.config.min_rows:
            tables.append({
                'data': current_table,
                'method': 'csv_like',
                'start_line': len(lines) - len(current_table),
                'end_line': len(lines) - 1
            })

        return tables

    def _parse_table_pattern(self, pattern: Dict[str, Any], index: int) -> Optional[List[List[str]]]:
        """Parse a detected table pattern into structured data."""

        data = pattern.get('data', [])

        if not data or len(data) < self.config.min_rows:
            return None

        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in data)
        normalized_data = []

        for row in data:
            # Pad short rows with empty strings
            while len(row) < max_cols:
                row.append('')
            # Truncate long rows
            row = row[:max_cols]
            normalized_data.append(row)

        return normalized_data

    def _analyze_table_structure(self, table_data: List[List[str]]) -> TableStructure:
        """Analyze the structure type of a table."""

        if not table_data:
            return TableStructure.SIMPLE

        # Check for hierarchical headers (multiple header rows)
        if len(table_data) > 2:
            first_row = table_data[0]
            second_row = table_data[1]

            # Look for merged cells or sub-headers
            if any(cell == '' for cell in second_row):
                return TableStructure.HIERARCHICAL

        # Check for pivot table structure
        if len(table_data) > 1:
            # Look for numeric data in specific patterns
            numeric_pattern = self._detect_numeric_pattern(table_data)
            if numeric_pattern and numeric_pattern.get('is_pivot'):
                return TableStructure.PIVOT

        # Default to simple structure
        return TableStructure.SIMPLE

    def _detect_numeric_pattern(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """Detect numeric patterns in table data."""

        numeric_cells = 0
        total_cells = 0

        for row in table_data[1:]:  # Skip header
            for cell in row:
                total_cells += 1
                if self._is_numeric(cell):
                    numeric_cells += 1

        numeric_ratio = numeric_cells / total_cells if total_cells > 0 else 0

        return {
            'numeric_ratio': numeric_ratio,
            'is_pivot': numeric_ratio > 0.6,
            'numeric_cells': numeric_cells,
            'total_cells': total_cells
        }

    def _analyze_columns(self, table_data: List[List[str]]) -> List[TableColumn]:
        """Analyze columns and detect data types."""

        if not table_data:
            return []

        columns = []
        headers = table_data[0] if table_data else []
        data_rows = table_data[1:] if len(table_data) > 1 else []

        for col_idx, header in enumerate(headers):
            # Extract column values
            values = []
            for row in data_rows:
                if col_idx < len(row):
                    values.append(row[col_idx])
                else:
                    values.append('')

            # Detect data type
            data_type = self._detect_column_data_type(values)

            # Calculate statistics
            statistics = self._calculate_column_statistics(values, data_type)

            # Count nulls and uniques
            null_count = sum(1 for v in values if not v or v.strip() == '')
            unique_count = len(set(v for v in values if v and v.strip()))

            column = TableColumn(
                index=col_idx,
                name=header or f"Column_{col_idx + 1}",
                data_type=data_type,
                values=values,
                null_count=null_count,
                unique_count=unique_count,
                statistics=statistics,
                metadata={}
            )

            columns.append(column)

        return columns

    def _detect_column_data_type(self, values: List[str]) -> DataType:
        """Detect the data type of a column based on its values."""

        non_empty_values = [v.strip() for v in values if v and v.strip()]

        if not non_empty_values:
            return DataType.UNKNOWN

        # Count different data type patterns
        type_counts = {
            DataType.INTEGER: 0,
            DataType.FLOAT: 0,
            DataType.CURRENCY: 0,
            DataType.PERCENTAGE: 0,
            DataType.DATE: 0,
            DataType.BOOLEAN: 0,
            DataType.EMAIL: 0,
            DataType.URL: 0,
            DataType.PHONE: 0,
            DataType.TEXT: 0
        }

        for value in non_empty_values:
            detected_type = self._detect_value_data_type(value)
            type_counts[detected_type] += 1

        # Find the most common type
        total_values = len(non_empty_values)
        best_type = DataType.TEXT
        best_ratio = 0

        for data_type, count in type_counts.items():
            ratio = count / total_values
            if ratio > best_ratio and ratio >= self.config.data_type_confidence_threshold:
                best_type = data_type
                best_ratio = ratio

        return best_type

    def _detect_value_data_type(self, value: str) -> DataType:
        """Detect the data type of a single value."""

        value = value.strip()

        # Boolean
        if value.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
            return DataType.BOOLEAN

        # Currency
        if re.match(r'^\$?[\d,]+\.?\d*$', value) or re.match(r'^[\d,]+\.?\d*\s*(USD|EUR|GBP|\$)$', value):
            return DataType.CURRENCY

        # Percentage
        if re.match(r'^\d+\.?\d*%$', value):
            return DataType.PERCENTAGE

        # Email
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            return DataType.EMAIL

        # URL
        if re.match(r'^https?://[^\s]+$', value):
            return DataType.URL

        # Phone
        if re.match(r'^[\+]?[\d\s\-\(\)]{10,}$', value):
            return DataType.PHONE

        # Date (simplified patterns)
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
        ]
        if any(re.match(pattern, value) for pattern in date_patterns):
            return DataType.DATE

        # Float
        if re.match(r'^-?\d+\.\d+$', value):
            return DataType.FLOAT

        # Integer
        if re.match(r'^-?\d+$', value):
            return DataType.INTEGER

        # Default to text
        return DataType.TEXT

    def _is_numeric(self, value: str) -> bool:
        """Check if a value is numeric."""

        try:
            float(value.replace(',', '').replace('$', '').replace('%', ''))
            return True
        except (ValueError, AttributeError):
            return False

    def _calculate_column_statistics(self, values: List[str], data_type: DataType) -> Dict[str, Any]:
        """Calculate statistics for a column."""

        stats = {}
        non_empty_values = [v for v in values if v and v.strip()]

        stats['count'] = len(values)
        stats['non_null_count'] = len(non_empty_values)
        stats['null_count'] = len(values) - len(non_empty_values)
        stats['unique_count'] = len(set(non_empty_values))

        if data_type in [DataType.INTEGER, DataType.FLOAT, DataType.CURRENCY]:
            numeric_values = []
            for value in non_empty_values:
                try:
                    # Clean numeric value
                    clean_value = value.replace(',', '').replace('$', '').replace('%', '')
                    numeric_values.append(float(clean_value))
                except (ValueError, AttributeError):
                    continue

            if numeric_values:
                stats['min'] = min(numeric_values)
                stats['max'] = max(numeric_values)
                stats['mean'] = sum(numeric_values) / len(numeric_values)
                stats['median'] = sorted(numeric_values)[len(numeric_values) // 2]

        return stats

    def _create_table_rows(self, table_data: List[List[str]], columns: List[TableColumn]) -> List[TableRow]:
        """Create table rows from data and column definitions."""

        rows = []
        data_rows = table_data[1:] if len(table_data) > 1 else []

        for row_idx, row_data in enumerate(data_rows):
            cells = []

            for col_idx, column in enumerate(columns):
                value = row_data[col_idx] if col_idx < len(row_data) else ''

                # Detect cell data type
                cell_data_type = self._detect_value_data_type(value)

                # Calculate confidence
                confidence = 1.0 if cell_data_type == column.data_type else 0.5

                cell = TableCell(
                    row=row_idx,
                    column=col_idx,
                    value=value,
                    data_type=cell_data_type,
                    formatted_value=value,
                    confidence=confidence,
                    metadata={}
                )

                cells.append(cell)

            row = TableRow(
                index=row_idx,
                cells=cells,
                metadata={}
            )

            rows.append(row)

        return rows

    def _analyze_relationships(self, table_data: List[List[str]], columns: List[TableColumn]) -> List[Dict[str, Any]]:
        """Analyze relationships between columns."""

        relationships = []

        # Look for foreign key relationships
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    # Check if values in col1 appear in col2 (potential foreign key)
                    col1_values = set(col1.values)
                    col2_values = set(col2.values)

                    overlap = len(col1_values & col2_values)
                    if overlap > 0:
                        relationship = {
                            'type': 'potential_foreign_key',
                            'source_column': col1.name,
                            'target_column': col2.name,
                            'overlap_count': overlap,
                            'confidence': overlap / len(col1_values) if col1_values else 0
                        }
                        relationships.append(relationship)

        return relationships

    def _validate_table_data(self, table_data: List[List[str]], columns: List[TableColumn]) -> Dict[str, Any]:
        """Validate table data quality."""

        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 1.0
        }

        # Check for consistent row lengths
        if table_data:
            expected_cols = len(table_data[0])
            for i, row in enumerate(table_data):
                if len(row) != expected_cols:
                    validation_results['warnings'].append(f"Row {i} has {len(row)} columns, expected {expected_cols}")

        # Check for empty headers
        if table_data and table_data[0]:
            for i, header in enumerate(table_data[0]):
                if not header or not header.strip():
                    validation_results['warnings'].append(f"Column {i} has empty header")

        # Check data type consistency
        for column in columns:
            type_consistency = self._check_column_type_consistency(column)
            if type_consistency < 0.8:
                validation_results['warnings'].append(f"Column '{column.name}' has inconsistent data types")

        # Calculate overall quality score
        error_penalty = len(validation_results['errors']) * 0.2
        warning_penalty = len(validation_results['warnings']) * 0.1
        validation_results['quality_score'] = max(0.0, 1.0 - error_penalty - warning_penalty)

        return validation_results

    def _check_column_type_consistency(self, column: TableColumn) -> float:
        """Check consistency of data types in a column."""

        if not column.values:
            return 1.0

        consistent_count = 0
        total_count = 0

        for value in column.values:
            if value and value.strip():
                total_count += 1
                detected_type = self._detect_value_data_type(value)
                if detected_type == column.data_type:
                    consistent_count += 1

        return consistent_count / total_count if total_count > 0 else 1.0

    def _calculate_extraction_confidence(self, table_data: List[List[str]],
                                       columns: List[TableColumn],
                                       validation_results: Dict[str, Any]) -> float:
        """Calculate confidence in the table extraction."""

        if not table_data or not columns:
            return 0.0

        # Base confidence from validation
        base_confidence = validation_results.get('quality_score', 0.5)

        # Adjust for table size
        size_factor = 1.0
        if len(table_data) < 3:  # Very small table
            size_factor = 0.8
        elif len(columns) < 2:  # Too few columns
            size_factor = 0.7

        # Adjust for data type detection confidence
        type_confidence = sum(self._check_column_type_consistency(col) for col in columns) / len(columns)

        # Combine factors
        confidence = base_confidence * size_factor * type_confidence

        return min(1.0, max(0.0, confidence))

    def _update_stats(self, result: str, processing_time_ms: float, tables_extracted: int) -> None:
        """Update service statistics."""

        self._stats["total_extractions"] += 1
        self._stats["total_processing_time_ms"] += processing_time_ms
        self._stats["total_tables_extracted"] += tables_extracted

        if result == "success":
            self._stats["successful_extractions"] += 1
        else:
            self._stats["failed_extractions"] += 1

        # Update averages
        if self._stats["total_extractions"] > 0:
            self._stats["average_processing_time_ms"] = (
                self._stats["total_processing_time_ms"] / self._stats["total_extractions"]
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""

        return {
            **self._stats,
            "cache_size": len(self._extraction_cache),
            "config": self.config.dict()
        }

    def clear_cache(self) -> None:
        """Clear extraction cache."""

        self._extraction_cache.clear()
        logger.info("Table extraction cache cleared")


# Global service instance
_table_extraction_service: Optional[TableExtractionService] = None


def get_table_extraction_service() -> TableExtractionService:
    """Get or create the global table extraction service instance."""
    global _table_extraction_service

    if _table_extraction_service is None:
        _table_extraction_service = TableExtractionService()

    return _table_extraction_service


def reset_table_extraction_service() -> None:
    """Reset the global table extraction service instance."""
    global _table_extraction_service
    _table_extraction_service = None
