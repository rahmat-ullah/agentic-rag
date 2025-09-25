"""
Source Attribution System Implementation

This module implements detailed source attribution with page numbers, sections,
metadata integration, and provenance tracking for answer synthesis.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)


class AttributionLevel(str, Enum):
    """Attribution detail levels."""
    
    MINIMAL = "minimal"                # Basic document reference
    STANDARD = "standard"              # Document + section + page
    DETAILED = "detailed"              # Full metadata with confidence
    COMPREHENSIVE = "comprehensive"    # Complete provenance chain


class AttributionStyle(str, Enum):
    """Attribution display styles."""
    
    INLINE = "inline"                  # Embedded in text
    SIDEBAR = "sidebar"                # Side panel display
    TOOLTIP = "tooltip"                # Hover information
    FOOTNOTE = "footnote"              # Bottom of page
    POPUP = "popup"                    # Modal display


class ProvenanceType(str, Enum):
    """Types of provenance information."""
    
    ORIGINAL_SOURCE = "original_source"      # Primary document
    DERIVED_SOURCE = "derived_source"        # Processed/extracted content
    AGGREGATED_SOURCE = "aggregated_source"  # Combined from multiple sources
    SYNTHESIZED_SOURCE = "synthesized_source"  # AI-generated content


@dataclass
class LocationInfo:
    """Detailed location information within a document."""
    
    # Page information
    page_number: Optional[int] = None
    page_range: Optional[str] = None
    total_pages: Optional[int] = None
    
    # Section information
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    chapter: Optional[str] = None
    subsection: Optional[str] = None
    
    # Precise location
    paragraph: Optional[int] = None
    line_number: Optional[int] = None
    character_offset: Optional[int] = None
    
    # Content boundaries
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    content_length: Optional[int] = None
    
    # Contextual information
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None


@dataclass
class DocumentMetadata:
    """Comprehensive document metadata."""
    
    # Core identification
    document_id: str
    document_title: str
    document_type: str
    
    # Authorship
    author: Optional[str] = None
    authors: Optional[List[str]] = None
    editor: Optional[str] = None
    
    # Publication details
    publication_date: Optional[datetime] = None
    publisher: Optional[str] = None
    publication_place: Optional[str] = None
    
    # Identifiers
    isbn: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    issn: Optional[str] = None
    
    # Version information
    version: Optional[str] = None
    edition: Optional[str] = None
    revision: Optional[str] = None
    
    # Content details
    language: Optional[str] = None
    subject_tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    
    # Technical metadata
    file_format: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    
    # Access information
    access_date: Optional[datetime] = None
    access_method: Optional[str] = None
    access_restrictions: Optional[str] = None
    
    def __post_init__(self):
        if self.access_date is None:
            self.access_date = datetime.now(timezone.utc)


@dataclass
class ProvenanceChain:
    """Chain of provenance for content transformation."""
    
    chain_id: str = field(default_factory=lambda: str(uuid4()))
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_step(self, step_type: str, description: str, metadata: Optional[Dict] = None):
        """Add a step to the provenance chain."""
        step = {
            "step_id": str(uuid4()),
            "step_type": step_type,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        self.steps.append(step)
    
    def get_summary(self) -> str:
        """Get a summary of the provenance chain."""
        if not self.steps:
            return "No provenance information available"
        
        summary_parts = []
        for step in self.steps:
            summary_parts.append(f"{step['step_type']}: {step['description']}")
        
        return " → ".join(summary_parts)


class SourceAttribution(BaseModel):
    """Comprehensive source attribution information."""
    
    # Core identification
    attribution_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str = Field(..., description="Source chunk identifier")
    
    # Document information
    document_metadata: DocumentMetadata = Field(..., description="Document metadata")
    location_info: LocationInfo = Field(..., description="Location within document")
    
    # Content information
    content_excerpt: str = Field(..., description="Relevant content excerpt")
    content_type: str = Field(default="text", description="Type of content")
    
    # Attribution details
    attribution_level: AttributionLevel = Field(default=AttributionLevel.STANDARD)
    provenance_type: ProvenanceType = Field(default=ProvenanceType.ORIGINAL_SOURCE)
    provenance_chain: ProvenanceChain = Field(default_factory=ProvenanceChain)
    
    # Quality metrics
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Usage tracking
    usage_count: int = Field(default=0)
    first_used: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    related_attributions: List[str] = Field(default_factory=list)
    superseded_by: Optional[str] = Field(None, description="ID of attribution that supersedes this one")
    supersedes: List[str] = Field(default_factory=list, description="IDs of attributions this supersedes")
    
    # Validation
    is_verified: bool = Field(default=False)
    verification_method: Optional[str] = Field(None)
    verification_date: Optional[datetime] = Field(None)
    
    @validator('content_excerpt')
    def validate_content_excerpt(cls, v):
        """Validate content excerpt is not empty."""
        if not v or not v.strip():
            raise ValueError("Content excerpt cannot be empty")
        return v.strip()


class AttributionGroup(BaseModel):
    """Group of related source attributions."""
    
    group_id: str = Field(default_factory=lambda: str(uuid4()))
    attributions: List[SourceAttribution] = Field(..., description="Attributions in group")
    
    # Grouping criteria
    group_type: str = Field(default="document", description="Type of grouping")
    group_criteria: Dict[str, Any] = Field(default_factory=dict)
    
    # Group metadata
    title: Optional[str] = Field(None, description="Group title")
    description: Optional[str] = Field(None, description="Group description")
    
    # Quality metrics
    group_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    group_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @validator('attributions')
    def validate_attributions(cls, v):
        """Validate that attributions are provided."""
        if not v:
            raise ValueError("Attribution group must contain at least one attribution")
        return v
    
    def calculate_group_metrics(self):
        """Calculate group-level quality metrics."""
        if not self.attributions:
            self.group_confidence = 0.0
            self.group_relevance = 0.0
            return
        
        # Calculate average confidence and relevance
        total_confidence = sum(attr.confidence_score for attr in self.attributions)
        total_relevance = sum(attr.relevance_score for attr in self.attributions)
        
        self.group_confidence = total_confidence / len(self.attributions)
        self.group_relevance = total_relevance / len(self.attributions)


class AttributionConfig(BaseModel):
    """Configuration for source attribution system."""
    
    # Attribution settings
    attribution_level: AttributionLevel = Field(default=AttributionLevel.STANDARD)
    attribution_style: AttributionStyle = Field(default=AttributionStyle.INLINE)
    
    # Content settings
    max_excerpt_length: int = Field(default=200, gt=0)
    include_context: bool = Field(default=True)
    context_length: int = Field(default=50, gt=0)
    
    # Display settings
    show_confidence_scores: bool = Field(default=False)
    show_page_numbers: bool = Field(default=True)
    show_section_titles: bool = Field(default=True)
    show_provenance: bool = Field(default=False)
    
    # Grouping settings
    enable_grouping: bool = Field(default=True)
    group_by_document: bool = Field(default=True)
    group_by_section: bool = Field(default=False)
    max_group_size: int = Field(default=10, gt=0)
    
    # Quality settings
    min_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_relevance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    require_verification: bool = Field(default=False)
    
    # Performance settings
    enable_caching: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=24, gt=0)
    max_attributions_per_answer: int = Field(default=20, gt=0)


class SourceAttributionService:
    """Service for managing detailed source attribution."""
    
    def __init__(self, config: Optional[AttributionConfig] = None):
        self.config = config or AttributionConfig()
        
        # Attribution storage
        self._attributions: Dict[str, SourceAttribution] = {}
        self._attribution_groups: Dict[str, AttributionGroup] = {}
        
        # Caching
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._max_cache_size = 1000
        
        # Statistics
        self._stats = {
            "attributions_created": 0,
            "attributions_verified": 0,
            "groups_created": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"Source attribution service initialized with level: {self.config.attribution_level}")
    
    def create_attribution(
        self,
        chunk_id: str,
        content_excerpt: str,
        document_metadata: DocumentMetadata,
        location_info: Optional[LocationInfo] = None,
        confidence_score: float = 0.0,
        relevance_score: float = 0.0
    ) -> SourceAttribution:
        """Create a new source attribution."""
        
        # Validate inputs
        if confidence_score < self.config.min_confidence_threshold:
            logger.warning(f"Attribution confidence {confidence_score} below threshold {self.config.min_confidence_threshold}")
        
        if relevance_score < self.config.min_relevance_threshold:
            logger.warning(f"Attribution relevance {relevance_score} below threshold {self.config.min_relevance_threshold}")
        
        # Truncate excerpt if too long
        if len(content_excerpt) > self.config.max_excerpt_length:
            content_excerpt = content_excerpt[:self.config.max_excerpt_length] + "..."
        
        # Create location info if not provided
        if location_info is None:
            location_info = LocationInfo()
        
        # Create provenance chain
        provenance_chain = ProvenanceChain()
        provenance_chain.add_step(
            "extraction",
            f"Content extracted from {document_metadata.document_title}",
            {"chunk_id": chunk_id, "extraction_method": "automated"}
        )
        
        # Create attribution
        attribution = SourceAttribution(
            chunk_id=chunk_id,
            content_excerpt=content_excerpt,
            document_metadata=document_metadata,
            location_info=location_info,
            attribution_level=self.config.attribution_level,
            confidence_score=confidence_score,
            relevance_score=relevance_score,
            provenance_chain=provenance_chain
        )
        
        # Store attribution
        self._attributions[attribution.attribution_id] = attribution
        self._stats["attributions_created"] += 1
        
        logger.debug(f"Created attribution {attribution.attribution_id} for chunk {chunk_id}")

        return attribution

    def create_attributions_from_sources(
        self,
        sources: List[Dict[str, Any]]
    ) -> List[SourceAttribution]:
        """Create attributions from a list of source data."""

        attributions = []

        for source in sources:
            # Extract required fields
            chunk_id = source.get('chunk_id', str(uuid4()))
            content = source.get('content', '')

            # Create document metadata
            doc_metadata = DocumentMetadata(
                document_id=source.get('document_id', ''),
                document_title=source.get('document_title', 'Unknown Document'),
                document_type=source.get('document_type', 'unknown'),
                author=source.get('author'),
                publication_date=source.get('publication_date'),
                url=source.get('url')
            )

            # Create location info
            location_info = LocationInfo(
                page_number=source.get('page_number'),
                section_title=source.get('section_title'),
                paragraph=source.get('paragraph'),
                start_position=source.get('start_position'),
                end_position=source.get('end_position')
            )

            # Extract quality scores
            confidence_score = source.get('confidence_score', 0.0)
            relevance_score = source.get('relevance_score', 0.0)

            # Create attribution
            attribution = self.create_attribution(
                chunk_id=chunk_id,
                content_excerpt=content,
                document_metadata=doc_metadata,
                location_info=location_info,
                confidence_score=confidence_score,
                relevance_score=relevance_score
            )

            attributions.append(attribution)

        return attributions

    def group_attributions(
        self,
        attributions: List[SourceAttribution],
        group_by: str = "document"
    ) -> List[AttributionGroup]:
        """Group attributions by specified criteria."""

        if not self.config.enable_grouping:
            # Return individual groups
            return [
                AttributionGroup(attributions=[attr], group_type="individual")
                for attr in attributions
            ]

        groups = {}

        for attribution in attributions:
            # Determine grouping key
            if group_by == "document":
                key = attribution.document_metadata.document_id
                title = attribution.document_metadata.document_title
            elif group_by == "section":
                key = f"{attribution.document_metadata.document_id}:{attribution.location_info.section_title}"
                title = f"{attribution.document_metadata.document_title} - {attribution.location_info.section_title}"
            elif group_by == "author":
                key = attribution.document_metadata.author or "Unknown"
                title = f"Works by {key}"
            else:
                key = "all"
                title = "All Sources"

            # Create or add to group
            if key not in groups:
                groups[key] = {
                    "attributions": [],
                    "group_type": group_by,
                    "title": title,
                    "group_criteria": {"group_by": group_by, "key": key}
                }

            # Check group size limit
            if len(groups[key]["attributions"]) < self.config.max_group_size:
                groups[key]["attributions"].append(attribution)
            else:
                # Create overflow group
                overflow_key = f"{key}_overflow_{len(groups)}"
                groups[overflow_key] = {
                    "attributions": [attribution],
                    "group_type": f"{group_by}_overflow",
                    "title": f"{title} (continued)",
                    "group_criteria": {"group_by": group_by, "key": key, "overflow": True}
                }

        # Convert to AttributionGroup objects and calculate group metrics
        group_list = []
        for group_data in groups.values():
            if group_data["attributions"]:  # Only create groups with attributions
                group = AttributionGroup(
                    attributions=group_data["attributions"],
                    group_type=group_data["group_type"],
                    title=group_data["title"],
                    group_criteria=group_data["group_criteria"]
                )
                group.calculate_group_metrics()
                self._attribution_groups[group.group_id] = group
                self._stats["groups_created"] += 1
                group_list.append(group)

        return group_list

    def format_attribution(
        self,
        attribution: SourceAttribution,
        style: Optional[AttributionStyle] = None
    ) -> str:
        """Format attribution for display."""

        style = style or self.config.attribution_style

        if style == AttributionStyle.INLINE:
            return self._format_inline_attribution(attribution)
        elif style == AttributionStyle.FOOTNOTE:
            return self._format_footnote_attribution(attribution)
        elif style == AttributionStyle.TOOLTIP:
            return self._format_tooltip_attribution(attribution)
        else:
            return self._format_standard_attribution(attribution)

    def _format_inline_attribution(self, attribution: SourceAttribution) -> str:
        """Format attribution for inline display."""

        parts = []

        # Document title
        parts.append(attribution.document_metadata.document_title)

        # Author if available
        if attribution.document_metadata.author:
            parts.append(f"by {attribution.document_metadata.author}")

        # Location information
        location_parts = []
        if self.config.show_section_titles and attribution.location_info.section_title:
            location_parts.append(attribution.location_info.section_title)

        if self.config.show_page_numbers and attribution.location_info.page_number:
            location_parts.append(f"p. {attribution.location_info.page_number}")

        if location_parts:
            parts.append(f"({', '.join(location_parts)})")

        # Confidence score if enabled
        if self.config.show_confidence_scores:
            parts.append(f"[confidence: {attribution.confidence_score:.2f}]")

        return " ".join(parts)

    def _format_footnote_attribution(self, attribution: SourceAttribution) -> str:
        """Format attribution for footnote display."""

        parts = []

        # Author
        if attribution.document_metadata.author:
            parts.append(attribution.document_metadata.author)

        # Title
        parts.append(f'"{attribution.document_metadata.document_title}"')

        # Publication info
        if attribution.document_metadata.publication_date:
            parts.append(f"({attribution.document_metadata.publication_date.year})")

        # Location
        if attribution.location_info.page_number:
            parts.append(f"p. {attribution.location_info.page_number}")

        return ", ".join(parts) + "."

    def _format_tooltip_attribution(self, attribution: SourceAttribution) -> str:
        """Format attribution for tooltip display."""

        lines = []

        # Title and author
        title_line = attribution.document_metadata.document_title
        if attribution.document_metadata.author:
            title_line += f" by {attribution.document_metadata.author}"
        lines.append(title_line)

        # Location details
        if attribution.location_info.section_title:
            lines.append(f"Section: {attribution.location_info.section_title}")

        if attribution.location_info.page_number:
            lines.append(f"Page: {attribution.location_info.page_number}")

        # Content excerpt
        excerpt = attribution.content_excerpt
        if len(excerpt) > 100:
            excerpt = excerpt[:100] + "..."
        lines.append(f"Content: {excerpt}")

        # Quality metrics
        if self.config.show_confidence_scores:
            lines.append(f"Confidence: {attribution.confidence_score:.2f}")
            lines.append(f"Relevance: {attribution.relevance_score:.2f}")

        return "\n".join(lines)

    def _format_standard_attribution(self, attribution: SourceAttribution) -> str:
        """Format attribution in standard format."""

        return self._format_inline_attribution(attribution)

    def verify_attribution(
        self,
        attribution_id: str,
        verification_method: str = "manual",
        verified_by: Optional[str] = None
    ) -> bool:
        """Verify an attribution."""

        if attribution_id not in self._attributions:
            logger.error(f"Attribution {attribution_id} not found for verification")
            return False

        attribution = self._attributions[attribution_id]
        attribution.is_verified = True
        attribution.verification_method = verification_method
        attribution.verification_date = datetime.now(timezone.utc)

        # Add verification step to provenance
        attribution.provenance_chain.add_step(
            "verification",
            f"Attribution verified using {verification_method}",
            {"verified_by": verified_by, "verification_date": attribution.verification_date.isoformat()}
        )

        self._stats["attributions_verified"] += 1
        logger.info(f"Attribution {attribution_id} verified using {verification_method}")

        return True

    def link_attributions(
        self,
        primary_attribution_id: str,
        related_attribution_ids: List[str],
        relationship_type: str = "related"
    ) -> bool:
        """Link related attributions."""

        if primary_attribution_id not in self._attributions:
            logger.error(f"Primary attribution {primary_attribution_id} not found")
            return False

        primary_attribution = self._attributions[primary_attribution_id]

        for related_id in related_attribution_ids:
            if related_id not in self._attributions:
                logger.warning(f"Related attribution {related_id} not found")
                continue

            # Add bidirectional relationship
            if related_id not in primary_attribution.related_attributions:
                primary_attribution.related_attributions.append(related_id)

            related_attribution = self._attributions[related_id]
            if primary_attribution_id not in related_attribution.related_attributions:
                related_attribution.related_attributions.append(primary_attribution_id)

        logger.debug(f"Linked attribution {primary_attribution_id} with {len(related_attribution_ids)} related attributions")

        return True

    def generate_attribution_report(
        self,
        attributions: List[SourceAttribution],
        include_provenance: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive attribution report."""

        report = {
            "summary": {
                "total_attributions": len(attributions),
                "verified_attributions": sum(1 for attr in attributions if attr.is_verified),
                "unique_documents": len(set(attr.document_metadata.document_id for attr in attributions)),
                "average_confidence": sum(attr.confidence_score for attr in attributions) / len(attributions) if attributions else 0,
                "average_relevance": sum(attr.relevance_score for attr in attributions) / len(attributions) if attributions else 0
            },
            "attributions": [],
            "document_breakdown": {},
            "quality_analysis": {}
        }

        # Document breakdown
        doc_counts = {}
        for attribution in attributions:
            doc_id = attribution.document_metadata.document_id
            doc_title = attribution.document_metadata.document_title

            if doc_id not in doc_counts:
                doc_counts[doc_id] = {"title": doc_title, "count": 0, "avg_confidence": 0}

            doc_counts[doc_id]["count"] += 1

        # Calculate average confidence per document
        for doc_id in doc_counts:
            doc_attributions = [attr for attr in attributions if attr.document_metadata.document_id == doc_id]
            doc_counts[doc_id]["avg_confidence"] = sum(attr.confidence_score for attr in doc_attributions) / len(doc_attributions)

        report["document_breakdown"] = doc_counts

        # Quality analysis
        confidence_scores = [attr.confidence_score for attr in attributions]
        relevance_scores = [attr.relevance_score for attr in attributions]

        report["quality_analysis"] = {
            "confidence_distribution": {
                "min": min(confidence_scores) if confidence_scores else 0,
                "max": max(confidence_scores) if confidence_scores else 0,
                "median": sorted(confidence_scores)[len(confidence_scores)//2] if confidence_scores else 0
            },
            "relevance_distribution": {
                "min": min(relevance_scores) if relevance_scores else 0,
                "max": max(relevance_scores) if relevance_scores else 0,
                "median": sorted(relevance_scores)[len(relevance_scores)//2] if relevance_scores else 0
            },
            "low_confidence_count": sum(1 for score in confidence_scores if score < 0.5),
            "high_confidence_count": sum(1 for score in confidence_scores if score >= 0.8)
        }

        # Individual attribution details
        for attribution in attributions:
            attr_data = {
                "attribution_id": attribution.attribution_id,
                "chunk_id": attribution.chunk_id,
                "document_title": attribution.document_metadata.document_title,
                "author": attribution.document_metadata.author,
                "page_number": attribution.location_info.page_number,
                "section_title": attribution.location_info.section_title,
                "confidence_score": attribution.confidence_score,
                "relevance_score": attribution.relevance_score,
                "is_verified": attribution.is_verified,
                "usage_count": attribution.usage_count,
                "formatted_attribution": self.format_attribution(attribution)
            }

            if include_provenance:
                attr_data["provenance_summary"] = attribution.provenance_chain.get_summary()
                attr_data["provenance_steps"] = len(attribution.provenance_chain.steps)

            report["attributions"].append(attr_data)

        return report

    def export_attributions(
        self,
        attributions: List[SourceAttribution],
        format: str = "json",
        include_provenance: bool = False
    ) -> str:
        """Export attributions in various formats."""

        if format.lower() == "json":
            return self._export_attributions_json(attributions, include_provenance)
        elif format.lower() == "csv":
            return self._export_attributions_csv(attributions)
        elif format.lower() == "xml":
            return self._export_attributions_xml(attributions, include_provenance)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_attributions_json(self, attributions: List[SourceAttribution], include_provenance: bool) -> str:
        """Export attributions as JSON."""

        export_data = []

        for attribution in attributions:
            data = {
                "attribution_id": attribution.attribution_id,
                "chunk_id": attribution.chunk_id,
                "content_excerpt": attribution.content_excerpt,
                "document_metadata": {
                    "document_id": attribution.document_metadata.document_id,
                    "document_title": attribution.document_metadata.document_title,
                    "document_type": attribution.document_metadata.document_type,
                    "author": attribution.document_metadata.author,
                    "publication_date": attribution.document_metadata.publication_date.isoformat() if attribution.document_metadata.publication_date else None,
                    "url": attribution.document_metadata.url
                },
                "location_info": {
                    "page_number": attribution.location_info.page_number,
                    "section_title": attribution.location_info.section_title,
                    "paragraph": attribution.location_info.paragraph
                },
                "quality_scores": {
                    "confidence_score": attribution.confidence_score,
                    "relevance_score": attribution.relevance_score,
                    "accuracy_score": attribution.accuracy_score
                },
                "verification": {
                    "is_verified": attribution.is_verified,
                    "verification_method": attribution.verification_method,
                    "verification_date": attribution.verification_date.isoformat() if attribution.verification_date else None
                },
                "usage": {
                    "usage_count": attribution.usage_count,
                    "first_used": attribution.first_used.isoformat(),
                    "last_used": attribution.last_used.isoformat()
                }
            }

            if include_provenance:
                data["provenance"] = {
                    "chain_id": attribution.provenance_chain.chain_id,
                    "steps": attribution.provenance_chain.steps,
                    "summary": attribution.provenance_chain.get_summary()
                }

            export_data.append(data)

        return json.dumps(export_data, indent=2, default=str)

    def _export_attributions_csv(self, attributions: List[SourceAttribution]) -> str:
        """Export attributions as CSV."""

        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Attribution ID", "Chunk ID", "Document Title", "Author", "Document Type",
            "Page Number", "Section Title", "Confidence Score", "Relevance Score",
            "Is Verified", "Usage Count", "Content Excerpt"
        ])

        # Data rows
        for attribution in attributions:
            writer.writerow([
                attribution.attribution_id,
                attribution.chunk_id,
                attribution.document_metadata.document_title,
                attribution.document_metadata.author or "",
                attribution.document_metadata.document_type,
                attribution.location_info.page_number or "",
                attribution.location_info.section_title or "",
                attribution.confidence_score,
                attribution.relevance_score,
                attribution.is_verified,
                attribution.usage_count,
                attribution.content_excerpt[:100] + "..." if len(attribution.content_excerpt) > 100 else attribution.content_excerpt
            ])

        return output.getvalue()

    def _export_attributions_xml(self, attributions: List[SourceAttribution], include_provenance: bool) -> str:
        """Export attributions as XML."""

        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<attributions>']

        for attribution in attributions:
            xml_lines.append('  <attribution>')
            xml_lines.append(f'    <attribution_id>{attribution.attribution_id}</attribution_id>')
            xml_lines.append(f'    <chunk_id>{attribution.chunk_id}</chunk_id>')
            xml_lines.append(f'    <content_excerpt><![CDATA[{attribution.content_excerpt}]]></content_excerpt>')

            # Document metadata
            xml_lines.append('    <document_metadata>')
            xml_lines.append(f'      <document_id>{attribution.document_metadata.document_id}</document_id>')
            xml_lines.append(f'      <document_title><![CDATA[{attribution.document_metadata.document_title}]]></document_title>')
            xml_lines.append(f'      <document_type>{attribution.document_metadata.document_type}</document_type>')
            if attribution.document_metadata.author:
                xml_lines.append(f'      <author><![CDATA[{attribution.document_metadata.author}]]></author>')
            xml_lines.append('    </document_metadata>')

            # Location info
            xml_lines.append('    <location_info>')
            if attribution.location_info.page_number:
                xml_lines.append(f'      <page_number>{attribution.location_info.page_number}</page_number>')
            if attribution.location_info.section_title:
                xml_lines.append(f'      <section_title><![CDATA[{attribution.location_info.section_title}]]></section_title>')
            xml_lines.append('    </location_info>')

            # Quality scores
            xml_lines.append('    <quality_scores>')
            xml_lines.append(f'      <confidence_score>{attribution.confidence_score}</confidence_score>')
            xml_lines.append(f'      <relevance_score>{attribution.relevance_score}</relevance_score>')
            xml_lines.append('    </quality_scores>')

            xml_lines.append('  </attribution>')

        xml_lines.append('</attributions>')

        return '\n'.join(xml_lines)

    def get_attribution_statistics(self) -> Dict[str, Any]:
        """Get attribution system statistics."""

        stats = self._stats.copy()
        stats.update({
            "total_stored_attributions": len(self._attributions),
            "total_attribution_groups": len(self._attribution_groups),
            "cache_size": len(self._cache),
            "attribution_level": self.config.attribution_level.value,
            "grouping_enabled": self.config.enable_grouping
        })

        return stats

    def clear_cache(self) -> None:
        """Clear the attribution cache."""
        self._cache.clear()
        logger.info("Attribution cache cleared")

    def get_attribution(self, attribution_id: str) -> Optional[SourceAttribution]:
        """Get attribution by ID."""
        return self._attributions.get(attribution_id)

    def get_attributions_by_document(self, document_id: str) -> List[SourceAttribution]:
        """Get all attributions for a specific document."""
        return [
            attr for attr in self._attributions.values()
            if attr.document_metadata.document_id == document_id
        ]

    def search_attributions(
        self,
        query: str,
        search_fields: List[str] = None
    ) -> List[SourceAttribution]:
        """Search attributions by content."""

        if search_fields is None:
            search_fields = ["content_excerpt", "document_title", "section_title"]

        query_lower = query.lower()
        matching_attributions = []

        for attribution in self._attributions.values():
            # Search in specified fields
            for field in search_fields:
                if field == "content_excerpt" and query_lower in attribution.content_excerpt.lower():
                    matching_attributions.append(attribution)
                    break
                elif field == "document_title" and query_lower in attribution.document_metadata.document_title.lower():
                    matching_attributions.append(attribution)
                    break
                elif field == "section_title" and attribution.location_info.section_title and query_lower in attribution.location_info.section_title.lower():
                    matching_attributions.append(attribution)
                    break

        return matching_attributions


# Global service instance
_source_attribution_service: Optional[SourceAttributionService] = None


def get_source_attribution_service(config: Optional[AttributionConfig] = None) -> SourceAttributionService:
    """Get or create the global source attribution service instance."""
    global _source_attribution_service

    if _source_attribution_service is None:
        _source_attribution_service = SourceAttributionService(config)

    return _source_attribution_service


def reset_source_attribution_service() -> None:
    """Reset the global source attribution service instance."""
    global _source_attribution_service
    _source_attribution_service = None
