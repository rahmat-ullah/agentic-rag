"""
Metadata Validator Service

This module provides validation and consistency checking for chunk metadata
before storage in the vector database.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class MetadataValidationLevel(str, Enum):
    """Validation levels for metadata checking."""
    
    STRICT = "strict"      # All fields must be valid
    MODERATE = "moderate"  # Required fields must be valid, optional can be missing
    LENIENT = "lenient"    # Basic validation only


class MetadataFieldType(str, Enum):
    """Types of metadata fields."""
    
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"


@dataclass
class MetadataFieldSpec:
    """Specification for a metadata field."""
    
    name: str
    field_type: MetadataFieldType
    required: bool = False
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[Set[Any]] = None
    default_value: Optional[Any] = None
    description: str = ""


class MetadataValidationResult(BaseModel):
    """Result of metadata validation."""
    
    is_valid: bool = Field(..., description="Whether metadata is valid")
    validated_metadata: Dict[str, Any] = Field(default_factory=dict, description="Validated and normalized metadata")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    field_count: int = Field(default=0, description="Number of fields validated")
    missing_required: List[str] = Field(default_factory=list, description="Missing required fields")
    invalid_fields: List[str] = Field(default_factory=list, description="Invalid field values")


class ChunkMetadataValidator:
    """Validator for document chunk metadata."""
    
    def __init__(self, validation_level: MetadataValidationLevel = MetadataValidationLevel.MODERATE):
        self.validation_level = validation_level
        self._field_specs = self._initialize_field_specs()
        
        logger.info(f"Metadata validator initialized with {validation_level.value} validation")
    
    def _initialize_field_specs(self) -> Dict[str, MetadataFieldSpec]:
        """Initialize metadata field specifications."""
        return {
            # Core document fields
            "tenant_id": MetadataFieldSpec(
                name="tenant_id",
                field_type=MetadataFieldType.STRING,
                required=True,
                max_length=100,
                description="Tenant identifier"
            ),
            "document_id": MetadataFieldSpec(
                name="document_id",
                field_type=MetadataFieldType.STRING,
                required=True,
                max_length=100,
                description="Document identifier"
            ),
            "chunk_id": MetadataFieldSpec(
                name="chunk_id",
                field_type=MetadataFieldType.STRING,
                required=True,
                max_length=100,
                description="Chunk identifier"
            ),
            
            # Document classification
            "document_kind": MetadataFieldSpec(
                name="document_kind",
                field_type=MetadataFieldType.STRING,
                required=True,
                allowed_values={"RFQ", "OFFER", "UNKNOWN"},
                description="Document kind classification"
            ),
            "document_type": MetadataFieldSpec(
                name="document_type",
                field_type=MetadataFieldType.STRING,
                required=False,
                allowed_values={"RFQ", "RFP", "TENDER", "OFFER_TECH", "OFFER_COMM", "PRICING", "UNKNOWN"},
                description="Document type classification"
            ),
            "document_name": MetadataFieldSpec(
                name="document_name",
                field_type=MetadataFieldType.STRING,
                required=False,
                max_length=500,
                description="Document name"
            ),
            
            # Chunk properties
            "chunk_index": MetadataFieldSpec(
                name="chunk_index",
                field_type=MetadataFieldType.INTEGER,
                required=False,
                min_value=0,
                description="Chunk index in document"
            ),
            "token_count": MetadataFieldSpec(
                name="token_count",
                field_type=MetadataFieldType.INTEGER,
                required=False,
                min_value=0,
                max_value=100000,
                description="Token count in chunk"
            ),
            "quality_score": MetadataFieldSpec(
                name="quality_score",
                field_type=MetadataFieldType.FLOAT,
                required=False,
                min_value=0.0,
                max_value=1.0,
                description="Chunk quality score"
            ),
            
            # Content structure
            "is_table": MetadataFieldSpec(
                name="is_table",
                field_type=MetadataFieldType.BOOLEAN,
                required=False,
                default_value=False,
                description="Whether chunk contains table data"
            ),
            "section_path": MetadataFieldSpec(
                name="section_path",
                field_type=MetadataFieldType.LIST,
                required=False,
                default_value=[],
                description="Section path in document"
            ),
            "page_from": MetadataFieldSpec(
                name="page_from",
                field_type=MetadataFieldType.INTEGER,
                required=False,
                min_value=1,
                description="Starting page number"
            ),
            "page_to": MetadataFieldSpec(
                name="page_to",
                field_type=MetadataFieldType.INTEGER,
                required=False,
                min_value=1,
                description="Ending page number"
            ),
            
            # Deduplication
            "is_duplicate": MetadataFieldSpec(
                name="is_duplicate",
                field_type=MetadataFieldType.BOOLEAN,
                required=False,
                default_value=False,
                description="Whether chunk is a duplicate"
            ),
            "similarity_score": MetadataFieldSpec(
                name="similarity_score",
                field_type=MetadataFieldType.FLOAT,
                required=False,
                min_value=0.0,
                max_value=1.0,
                description="Similarity score for deduplication"
            ),
            
            # Timestamps
            "created_at": MetadataFieldSpec(
                name="created_at",
                field_type=MetadataFieldType.STRING,
                required=False,
                description="Creation timestamp (ISO format)"
            ),
            "updated_at": MetadataFieldSpec(
                name="updated_at",
                field_type=MetadataFieldType.STRING,
                required=False,
                description="Update timestamp (ISO format)"
            ),
            
            # Processing flags
            "reindexing": MetadataFieldSpec(
                name="reindexing",
                field_type=MetadataFieldType.BOOLEAN,
                required=False,
                default_value=False,
                description="Whether this is a reindexing operation"
            )
        }
    
    async def validate_metadata(self, metadata: Dict[str, Any]) -> MetadataValidationResult:
        """
        Validate chunk metadata against field specifications.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            MetadataValidationResult with validation outcome
        """
        result = MetadataValidationResult(
            is_valid=True,
            field_count=len(metadata)
        )
        
        validated_metadata = {}
        
        try:
            # Check required fields
            for field_name, spec in self._field_specs.items():
                if spec.required and field_name not in metadata:
                    result.missing_required.append(field_name)
                    result.errors.append(f"Required field '{field_name}' is missing")
                    result.is_valid = False
            
            # Validate each field in metadata
            for field_name, value in metadata.items():
                validation_result = await self._validate_field(field_name, value)
                
                if validation_result["is_valid"]:
                    validated_metadata[field_name] = validation_result["value"]
                    
                    if validation_result.get("warnings"):
                        result.warnings.extend(validation_result["warnings"])
                else:
                    result.invalid_fields.append(field_name)
                    result.errors.extend(validation_result.get("errors", []))
                    
                    if self.validation_level == MetadataValidationLevel.STRICT:
                        result.is_valid = False
                    elif self.validation_level == MetadataValidationLevel.MODERATE:
                        # Include field with default value if available
                        spec = self._field_specs.get(field_name)
                        if spec and spec.default_value is not None:
                            validated_metadata[field_name] = spec.default_value
                            result.warnings.append(f"Using default value for invalid field '{field_name}'")
            
            # Add default values for missing optional fields
            for field_name, spec in self._field_specs.items():
                if not spec.required and field_name not in validated_metadata and spec.default_value is not None:
                    validated_metadata[field_name] = spec.default_value
            
            # Perform consistency checks
            consistency_result = await self._check_consistency(validated_metadata)
            if not consistency_result["is_valid"]:
                result.errors.extend(consistency_result.get("errors", []))
                result.warnings.extend(consistency_result.get("warnings", []))
                
                if self.validation_level == MetadataValidationLevel.STRICT:
                    result.is_valid = False
            
            result.validated_metadata = validated_metadata
            
            logger.debug(f"Validated metadata: {len(validated_metadata)} fields, valid: {result.is_valid}")
            
        except Exception as e:
            logger.error(f"Metadata validation failed: {e}")
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")
        
        return result
    
    async def _validate_field(self, field_name: str, value: Any) -> Dict[str, Any]:
        """Validate a single metadata field."""
        spec = self._field_specs.get(field_name)
        
        if not spec:
            # Unknown field - allow in lenient mode, warn in others
            if self.validation_level == MetadataValidationLevel.LENIENT:
                return {"is_valid": True, "value": value}
            else:
                return {
                    "is_valid": False,
                    "errors": [f"Unknown field '{field_name}'"],
                    "warnings": [f"Field '{field_name}' is not in specification"]
                }
        
        result = {"is_valid": True, "value": value, "warnings": [], "errors": []}
        
        try:
            # Type validation
            if spec.field_type == MetadataFieldType.STRING:
                if not isinstance(value, str):
                    value = str(value)
                    result["warnings"].append(f"Converted {field_name} to string")
                
                if spec.max_length and len(value) > spec.max_length:
                    result["errors"].append(f"Field '{field_name}' exceeds max length {spec.max_length}")
                    result["is_valid"] = False
            
            elif spec.field_type == MetadataFieldType.INTEGER:
                if not isinstance(value, int):
                    try:
                        value = int(value)
                        result["warnings"].append(f"Converted {field_name} to integer")
                    except (ValueError, TypeError):
                        result["errors"].append(f"Field '{field_name}' cannot be converted to integer")
                        result["is_valid"] = False
                        return result
                
                if spec.min_value is not None and value < spec.min_value:
                    result["errors"].append(f"Field '{field_name}' below minimum value {spec.min_value}")
                    result["is_valid"] = False
                
                if spec.max_value is not None and value > spec.max_value:
                    result["errors"].append(f"Field '{field_name}' above maximum value {spec.max_value}")
                    result["is_valid"] = False
            
            elif spec.field_type == MetadataFieldType.FLOAT:
                if not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                        result["warnings"].append(f"Converted {field_name} to float")
                    except (ValueError, TypeError):
                        result["errors"].append(f"Field '{field_name}' cannot be converted to float")
                        result["is_valid"] = False
                        return result
                
                if spec.min_value is not None and value < spec.min_value:
                    result["errors"].append(f"Field '{field_name}' below minimum value {spec.min_value}")
                    result["is_valid"] = False
                
                if spec.max_value is not None and value > spec.max_value:
                    result["errors"].append(f"Field '{field_name}' above maximum value {spec.max_value}")
                    result["is_valid"] = False
            
            elif spec.field_type == MetadataFieldType.BOOLEAN:
                if not isinstance(value, bool):
                    if isinstance(value, str):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = bool(value)
                    result["warnings"].append(f"Converted {field_name} to boolean")
            
            elif spec.field_type == MetadataFieldType.LIST:
                if not isinstance(value, list):
                    if isinstance(value, str):
                        # Try to parse as comma-separated values
                        value = [item.strip() for item in value.split(',') if item.strip()]
                    else:
                        value = [value]
                    result["warnings"].append(f"Converted {field_name} to list")
            
            elif spec.field_type == MetadataFieldType.DICT:
                if not isinstance(value, dict):
                    result["errors"].append(f"Field '{field_name}' must be a dictionary")
                    result["is_valid"] = False
            
            # Allowed values validation
            if spec.allowed_values and value not in spec.allowed_values:
                result["errors"].append(f"Field '{field_name}' value '{value}' not in allowed values: {spec.allowed_values}")
                result["is_valid"] = False
            
            result["value"] = value
            
        except Exception as e:
            result["errors"].append(f"Field '{field_name}' validation error: {str(e)}")
            result["is_valid"] = False
        
        return result
    
    async def _check_consistency(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency between metadata fields."""
        result = {"is_valid": True, "errors": [], "warnings": []}
        
        try:
            # Check page range consistency
            page_from = metadata.get("page_from")
            page_to = metadata.get("page_to")
            
            if page_from is not None and page_to is not None:
                if page_from > page_to:
                    result["errors"].append(f"page_from ({page_from}) cannot be greater than page_to ({page_to})")
                    result["is_valid"] = False
            
            # Check duplicate consistency
            is_duplicate = metadata.get("is_duplicate", False)
            similarity_score = metadata.get("similarity_score")
            
            if is_duplicate and similarity_score is None:
                result["warnings"].append("is_duplicate is True but similarity_score is missing")
            
            # Check quality score consistency
            quality_score = metadata.get("quality_score")
            if quality_score is not None and quality_score < 0.3:
                result["warnings"].append(f"Low quality score detected: {quality_score}")
            
            # Check timestamp format
            for timestamp_field in ["created_at", "updated_at"]:
                timestamp = metadata.get(timestamp_field)
                if timestamp:
                    try:
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        result["errors"].append(f"Invalid timestamp format for {timestamp_field}: {timestamp}")
                        result["is_valid"] = False
            
        except Exception as e:
            result["errors"].append(f"Consistency check error: {str(e)}")
            result["is_valid"] = False
        
        return result
    
    def get_field_specifications(self) -> Dict[str, MetadataFieldSpec]:
        """Get all field specifications."""
        return self._field_specs.copy()
    
    def add_custom_field_spec(self, spec: MetadataFieldSpec) -> None:
        """Add a custom field specification."""
        self._field_specs[spec.name] = spec
        logger.info(f"Added custom field specification: {spec.name}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation configuration summary."""
        return {
            "validation_level": self.validation_level.value,
            "total_fields": len(self._field_specs),
            "required_fields": [name for name, spec in self._field_specs.items() if spec.required],
            "optional_fields": [name for name, spec in self._field_specs.items() if not spec.required],
            "field_types": {field_type.value: len([spec for spec in self._field_specs.values() if spec.field_type == field_type]) for field_type in MetadataFieldType}
        }


# Global validator instance
_metadata_validator: Optional[ChunkMetadataValidator] = None


async def get_metadata_validator(validation_level: MetadataValidationLevel = MetadataValidationLevel.MODERATE) -> ChunkMetadataValidator:
    """Get or create the global metadata validator instance."""
    global _metadata_validator
    
    if _metadata_validator is None:
        _metadata_validator = ChunkMetadataValidator(validation_level)
    
    return _metadata_validator


async def validate_chunk_metadata(metadata: Dict[str, Any], validation_level: MetadataValidationLevel = MetadataValidationLevel.MODERATE) -> MetadataValidationResult:
    """
    Convenience function to validate chunk metadata.
    
    Args:
        metadata: Metadata dictionary to validate
        validation_level: Validation level to use
        
    Returns:
        MetadataValidationResult with validation outcome
    """
    validator = await get_metadata_validator(validation_level)
    return await validator.validate_metadata(metadata)
