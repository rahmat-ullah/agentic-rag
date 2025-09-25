"""
Citation System Implementation

This module implements a comprehensive citation system with academic formatting,
automatic insertion, deduplication, and validation for answer synthesis.
"""

import re
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)


class CitationFormat(str, Enum):
    """Citation format standards."""
    
    APA = "apa"                        # American Psychological Association
    MLA = "mla"                        # Modern Language Association
    CHICAGO = "chicago"                # Chicago Manual of Style
    IEEE = "ieee"                      # Institute of Electrical and Electronics Engineers
    HARVARD = "harvard"                # Harvard referencing
    NUMBERED = "numbered"              # Simple numbered format [1], [2]
    INLINE = "inline"                  # Inline document references
    FOOTNOTE = "footnote"              # Footnote style


class CitationValidationLevel(str, Enum):
    """Citation validation levels."""
    
    BASIC = "basic"                    # Basic format validation
    STANDARD = "standard"              # Standard validation with metadata checks
    STRICT = "strict"                  # Strict validation with completeness checks
    ACADEMIC = "academic"              # Academic-level validation


class CitationPlacement(str, Enum):
    """Citation placement strategies."""
    
    INLINE = "inline"                  # Within the text flow
    END_OF_SENTENCE = "end_of_sentence"  # At the end of sentences
    END_OF_PARAGRAPH = "end_of_paragraph"  # At the end of paragraphs
    FOOTNOTE = "footnote"              # As footnotes
    ENDNOTE = "endnote"               # As endnotes


@dataclass
class CitationMetadata:
    """Extended metadata for citations."""
    
    # Core identification
    chunk_id: str
    document_id: str
    document_title: str
    
    # Document details
    document_type: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    publisher: Optional[str] = None
    isbn: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    
    # Location details
    section_title: Optional[str] = None
    chapter: Optional[str] = None
    page_number: Optional[int] = None
    page_range: Optional[str] = None
    paragraph: Optional[int] = None
    line_number: Optional[int] = None
    
    # Content details
    chunk_index: int = 0
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    content_preview: Optional[str] = None
    
    # Additional metadata
    language: Optional[str] = None
    edition: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    access_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.access_date is None:
            self.access_date = datetime.now(timezone.utc)


class FormattedCitation(BaseModel):
    """A formatted citation with all necessary information."""
    
    id: int = Field(..., description="Citation number/identifier")
    citation_key: str = Field(..., description="Unique citation key")
    formatted_text: str = Field(..., description="Formatted citation text")
    short_form: str = Field(..., description="Short form for inline use")
    
    # Source information
    metadata: CitationMetadata = Field(..., description="Citation metadata")
    
    # Formatting details
    format_style: CitationFormat = Field(..., description="Citation format used")
    placement: CitationPlacement = Field(..., description="Placement strategy")
    
    # Validation
    is_valid: bool = Field(default=True, description="Whether citation is valid")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    # Usage tracking
    usage_count: int = Field(default=0, description="Number of times cited")
    first_used: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CitationGroup(BaseModel):
    """A group of related citations."""
    
    group_id: str = Field(..., description="Group identifier")
    citations: List[FormattedCitation] = Field(..., description="Citations in group")
    group_type: str = Field(default="standard", description="Type of citation group")
    
    # Grouping metadata
    topic: Optional[str] = Field(None, description="Topic or theme")
    document_cluster: Optional[str] = Field(None, description="Document cluster ID")
    
    @validator('citations')
    def validate_citations(cls, v):
        """Validate that citations are provided."""
        if not v:
            raise ValueError("Citation group must contain at least one citation")
        return v


class CitationIndex(BaseModel):
    """Index of all citations for quick lookup and deduplication."""
    
    citations_by_id: Dict[int, FormattedCitation] = Field(default_factory=dict)
    citations_by_key: Dict[str, FormattedCitation] = Field(default_factory=dict)
    citations_by_document: Dict[str, List[FormattedCitation]] = Field(default_factory=dict)
    
    # Deduplication tracking
    duplicate_groups: Dict[str, List[int]] = Field(default_factory=dict)
    
    # Statistics
    total_citations: int = Field(default=0)
    unique_documents: int = Field(default=0)
    duplicate_count: int = Field(default=0)


class CitationConfig(BaseModel):
    """Configuration for citation system."""
    
    # Format settings
    format_style: CitationFormat = Field(default=CitationFormat.NUMBERED)
    placement: CitationPlacement = Field(default=CitationPlacement.INLINE)
    validation_level: CitationValidationLevel = Field(default=CitationValidationLevel.STANDARD)
    
    # Deduplication settings
    enable_deduplication: bool = Field(default=True)
    deduplication_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    merge_similar_citations: bool = Field(default=True)
    
    # Formatting options
    include_page_numbers: bool = Field(default=True)
    include_section_titles: bool = Field(default=True)
    include_access_dates: bool = Field(default=False)
    abbreviate_titles: bool = Field(default=False)
    max_title_length: int = Field(default=100, gt=0)
    
    # Validation options
    require_author: bool = Field(default=False)
    require_publication_date: bool = Field(default=False)
    require_page_numbers: bool = Field(default=False)
    allow_incomplete_citations: bool = Field(default=True)
    
    # Display options
    show_confidence_scores: bool = Field(default=False)
    group_by_document: bool = Field(default=False)
    sort_alphabetically: bool = Field(default=False)
    
    # Advanced options
    auto_generate_keys: bool = Field(default=True)
    key_generation_pattern: str = Field(default="{author_last}{year}")
    max_citations_per_answer: int = Field(default=50, gt=0)


class CitationSystemService:
    """Service for managing citations with formatting, validation, and deduplication."""
    
    def __init__(self, config: Optional[CitationConfig] = None):
        self.config = config or CitationConfig()
        self.citation_index = CitationIndex()
        
        # Format templates
        self._format_templates = self._initialize_format_templates()
        
        # Statistics
        self._stats = {
            "citations_created": 0,
            "citations_deduplicated": 0,
            "validation_errors": 0,
            "format_conversions": 0
        }
        
        logger.info(f"Citation system initialized with format: {self.config.format_style}")
    
    def create_citations(
        self,
        metadata_list: List[CitationMetadata],
        start_id: int = 1
    ) -> List[FormattedCitation]:
        """Create formatted citations from metadata."""
        
        citations = []
        current_id = start_id
        
        for metadata in metadata_list:
            # Check for duplicates if deduplication is enabled
            if self.config.enable_deduplication:
                existing_citation = self._find_duplicate_citation(metadata)
                if existing_citation:
                    existing_citation.usage_count += 1
                    existing_citation.last_used = datetime.now(timezone.utc)
                    citations.append(existing_citation)
                    self._stats["citations_deduplicated"] += 1
                    continue
            
            # Generate citation key
            citation_key = self._generate_citation_key(metadata, current_id)
            
            # Format citation
            formatted_text = self._format_citation(metadata, current_id)
            short_form = self._generate_short_form(metadata, current_id)
            
            # Validate citation
            is_valid, errors, warnings = self._validate_citation(metadata)
            
            # Create formatted citation
            citation = FormattedCitation(
                id=current_id,
                citation_key=citation_key,
                formatted_text=formatted_text,
                short_form=short_form,
                metadata=metadata,
                format_style=self.config.format_style,
                placement=self.config.placement,
                is_valid=is_valid,
                validation_errors=errors,
                validation_warnings=warnings
            )
            
            # Add to index
            self._add_to_index(citation)
            
            citations.append(citation)
            current_id += 1
            self._stats["citations_created"] += 1
        
        return citations
    
    def _find_duplicate_citation(self, metadata: CitationMetadata) -> Optional[FormattedCitation]:
        """Find duplicate citation based on metadata similarity."""
        
        for existing_citation in self.citation_index.citations_by_id.values():
            similarity = self._calculate_citation_similarity(metadata, existing_citation.metadata)
            
            if similarity >= self.config.deduplication_threshold:
                return existing_citation
        
        return None
    
    def _calculate_citation_similarity(self, meta1: CitationMetadata, meta2: CitationMetadata) -> float:
        """Calculate similarity between two citation metadata objects."""
        
        # Check exact matches for key fields
        exact_matches = 0
        total_fields = 0
        
        # Document ID match (highest weight)
        if meta1.document_id == meta2.document_id:
            exact_matches += 3
        total_fields += 3
        
        # Chunk ID match
        if meta1.chunk_id == meta2.chunk_id:
            exact_matches += 2
        total_fields += 2
        
        # Title similarity
        if meta1.document_title and meta2.document_title:
            title_similarity = self._calculate_text_similarity(meta1.document_title, meta2.document_title)
            exact_matches += title_similarity
        total_fields += 1
        
        # Page number match
        if meta1.page_number and meta2.page_number:
            if meta1.page_number == meta2.page_number:
                exact_matches += 1
        total_fields += 1
        
        # Section match
        if meta1.section_title and meta2.section_title:
            section_similarity = self._calculate_text_similarity(meta1.section_title, meta2.section_title)
            exact_matches += section_similarity
        total_fields += 1
        
        return exact_matches / total_fields if total_fields > 0 else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def _generate_citation_key(self, metadata: CitationMetadata, citation_id: int) -> str:
        """Generate a unique citation key."""

        if not self.config.auto_generate_keys:
            return f"cite_{citation_id}"

        # Extract components for key generation
        author_last = ""
        if metadata.author:
            author_parts = metadata.author.split()
            author_last = author_parts[-1] if author_parts else ""

        year = ""
        if metadata.publication_date:
            year = str(metadata.publication_date.year)

        # Generate key based on pattern
        pattern = self.config.key_generation_pattern
        key = pattern.format(
            author_last=author_last,
            year=year,
            id=citation_id,
            document_id=metadata.document_id[:8] if metadata.document_id else "",
            title=metadata.document_title[:10] if metadata.document_title else ""
        )

        # Clean up the key
        key = re.sub(r'[^a-zA-Z0-9_]', '', key)

        # Ensure uniqueness
        base_key = key
        counter = 1
        while key in self.citation_index.citations_by_key:
            key = f"{base_key}_{counter}"
            counter += 1

        return key or f"cite_{citation_id}"

    def _format_citation(self, metadata: CitationMetadata, citation_id: int) -> str:
        """Format citation according to the specified style."""

        formatter = self._format_templates.get(self.config.format_style)
        if not formatter:
            formatter = self._format_templates[CitationFormat.NUMBERED]

        return formatter(metadata, citation_id, self.config)

    def _generate_short_form(self, metadata: CitationMetadata, citation_id: int) -> str:
        """Generate short form for inline citations."""

        if self.config.format_style == CitationFormat.NUMBERED:
            return f"[{citation_id}]"
        elif self.config.format_style == CitationFormat.APA:
            author = metadata.author.split()[0] if metadata.author else "Unknown"
            year = metadata.publication_date.year if metadata.publication_date else "n.d."
            return f"({author}, {year})"
        elif self.config.format_style == CitationFormat.MLA:
            author = metadata.author.split()[0] if metadata.author else "Unknown"
            return f"({author})"
        else:
            return f"[{citation_id}]"

    def _validate_citation(self, metadata: CitationMetadata) -> Tuple[bool, List[str], List[str]]:
        """Validate citation metadata."""

        errors = []
        warnings = []

        # Required field validation based on config
        if self.config.require_author and not metadata.author:
            errors.append("Author is required but missing")

        if self.config.require_publication_date and not metadata.publication_date:
            errors.append("Publication date is required but missing")

        if self.config.require_page_numbers and not metadata.page_number:
            errors.append("Page number is required but missing")

        # Basic validation
        if not metadata.document_title:
            warnings.append("Document title is missing")

        if not metadata.chunk_id:
            errors.append("Chunk ID is required but missing")

        if not metadata.document_id:
            errors.append("Document ID is required but missing")

        # Format-specific validation
        if self.config.validation_level == CitationValidationLevel.ACADEMIC:
            if not metadata.author:
                warnings.append("Author missing for academic citation")

            if not metadata.publication_date:
                warnings.append("Publication date missing for academic citation")

        # URL validation
        if metadata.url and not self._is_valid_url(metadata.url):
            warnings.append("URL format appears invalid")

        is_valid = len(errors) == 0
        if not is_valid:
            self._stats["validation_errors"] += 1

        return is_valid, errors, warnings

    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None

    def _add_to_index(self, citation: FormattedCitation) -> None:
        """Add citation to the index."""

        self.citation_index.citations_by_id[citation.id] = citation
        self.citation_index.citations_by_key[citation.citation_key] = citation

        # Add to document index
        doc_id = citation.metadata.document_id
        if doc_id not in self.citation_index.citations_by_document:
            self.citation_index.citations_by_document[doc_id] = []
        self.citation_index.citations_by_document[doc_id].append(citation)

        # Update statistics
        self.citation_index.total_citations += 1
        self.citation_index.unique_documents = len(self.citation_index.citations_by_document)

    def _initialize_format_templates(self) -> Dict[CitationFormat, callable]:
        """Initialize citation format templates."""

        def format_numbered(metadata: CitationMetadata, citation_id: int, config: CitationConfig) -> str:
            """Format numbered citation."""
            parts = [f"[{citation_id}]", metadata.document_title]

            if config.include_section_titles and metadata.section_title:
                parts.append(metadata.section_title)

            if config.include_page_numbers and metadata.page_number:
                parts.append(f"Page {metadata.page_number}")

            return ", ".join(parts)

        def format_apa(metadata: CitationMetadata, citation_id: int, config: CitationConfig) -> str:
            """Format APA style citation."""
            parts = []

            # Author
            if metadata.author:
                parts.append(metadata.author)

            # Year
            if metadata.publication_date:
                parts.append(f"({metadata.publication_date.year})")

            # Title
            title = metadata.document_title
            if config.abbreviate_titles and len(title) > config.max_title_length:
                title = title[:config.max_title_length] + "..."
            parts.append(f'"{title}"')

            # Page
            if config.include_page_numbers and metadata.page_number:
                parts.append(f"p. {metadata.page_number}")

            return ". ".join(parts) + "."

        def format_mla(metadata: CitationMetadata, citation_id: int, config: CitationConfig) -> str:
            """Format MLA style citation."""
            parts = []

            # Author
            if metadata.author:
                parts.append(metadata.author)

            # Title
            title = metadata.document_title
            if config.abbreviate_titles and len(title) > config.max_title_length:
                title = title[:config.max_title_length] + "..."
            parts.append(f'"{title}"')

            # Publisher and date
            if metadata.publisher:
                parts.append(metadata.publisher)

            if metadata.publication_date:
                parts.append(str(metadata.publication_date.year))

            # Page
            if config.include_page_numbers and metadata.page_number:
                parts.append(f"{metadata.page_number}")

            return ", ".join(parts) + "."

        def format_chicago(metadata: CitationMetadata, citation_id: int, config: CitationConfig) -> str:
            """Format Chicago style citation."""
            parts = []

            # Author
            if metadata.author:
                parts.append(metadata.author)

            # Title
            title = metadata.document_title
            if config.abbreviate_titles and len(title) > config.max_title_length:
                title = title[:config.max_title_length] + "..."
            parts.append(f'"{title}"')

            # Publication info
            if metadata.publisher and metadata.publication_date:
                parts.append(f"{metadata.publisher}, {metadata.publication_date.year}")

            # Page
            if config.include_page_numbers and metadata.page_number:
                parts.append(f"{metadata.page_number}")

            return ", ".join(parts) + "."

        def format_ieee(metadata: CitationMetadata, citation_id: int, config: CitationConfig) -> str:
            """Format IEEE style citation."""
            parts = [f"[{citation_id}]"]

            # Author
            if metadata.author:
                parts.append(metadata.author)

            # Title
            title = metadata.document_title
            if config.abbreviate_titles and len(title) > config.max_title_length:
                title = title[:config.max_title_length] + "..."
            parts.append(f'"{title}"')

            # Year
            if metadata.publication_date:
                parts.append(str(metadata.publication_date.year))

            # Page
            if config.include_page_numbers and metadata.page_number:
                parts.append(f"p. {metadata.page_number}")

            return ", ".join(parts) + "."

        return {
            CitationFormat.NUMBERED: format_numbered,
            CitationFormat.APA: format_apa,
            CitationFormat.MLA: format_mla,
            CitationFormat.CHICAGO: format_chicago,
            CitationFormat.IEEE: format_ieee,
            CitationFormat.HARVARD: format_apa,  # Similar to APA
            CitationFormat.INLINE: format_numbered,
            CitationFormat.FOOTNOTE: format_numbered
        }

    def insert_citations_in_text(self, text: str, citations: List[FormattedCitation]) -> str:
        """Insert citations into text automatically."""

        # Create citation mapping
        citation_map = {cite.metadata.chunk_id: cite for cite in citations}

        # Find potential citation points
        # This is a simplified implementation - in practice, this would be more sophisticated
        sentences = text.split('.')
        result_sentences = []

        for sentence in sentences:
            # Look for content that might need citations
            # This could be enhanced with NLP to identify factual claims
            if len(sentence.strip()) > 20:  # Only cite substantial sentences
                # Add citation at end of sentence if not already present
                if not re.search(r'\[\d+\]', sentence):
                    # Find the most relevant citation (simplified approach)
                    best_citation = self._find_best_citation_for_sentence(sentence, citations)
                    if best_citation:
                        sentence += f" {best_citation.short_form}"

            result_sentences.append(sentence)

        return '.'.join(result_sentences)

    def _find_best_citation_for_sentence(self, sentence: str, citations: List[FormattedCitation]) -> Optional[FormattedCitation]:
        """Find the best citation for a given sentence."""

        sentence_words = set(sentence.lower().split())
        best_citation = None
        best_score = 0.0

        for citation in citations:
            # Calculate relevance based on content overlap
            if citation.metadata.content_preview:
                content_words = set(citation.metadata.content_preview.lower().split())
                overlap = len(sentence_words.intersection(content_words))
                score = overlap / len(sentence_words) if sentence_words else 0

                if score > best_score:
                    best_score = score
                    best_citation = citation

        return best_citation if best_score > 0.1 else None

    def deduplicate_citations(self, citations: List[FormattedCitation]) -> List[FormattedCitation]:
        """Remove duplicate citations and merge similar ones."""

        if not self.config.enable_deduplication:
            return citations

        unique_citations = []
        seen_keys = set()

        for citation in citations:
            # Check for exact duplicates
            if citation.citation_key in seen_keys:
                continue

            # Check for similar citations
            similar_citation = None
            if self.config.merge_similar_citations:
                for existing in unique_citations:
                    similarity = self._calculate_citation_similarity(
                        citation.metadata, existing.metadata
                    )
                    if similarity >= self.config.deduplication_threshold:
                        similar_citation = existing
                        break

            if similar_citation:
                # Merge with existing citation
                similar_citation.usage_count += 1
                self._stats["citations_deduplicated"] += 1
            else:
                # Add as new unique citation
                unique_citations.append(citation)
                seen_keys.add(citation.citation_key)

        return unique_citations

    def generate_bibliography(self, citations: List[FormattedCitation]) -> str:
        """Generate a formatted bibliography."""

        if not citations:
            return ""

        # Sort citations
        sorted_citations = citations.copy()
        if self.config.sort_alphabetically:
            sorted_citations.sort(key=lambda c: c.metadata.document_title.lower())
        else:
            sorted_citations.sort(key=lambda c: c.id)

        # Group by document if configured
        if self.config.group_by_document:
            return self._generate_grouped_bibliography(sorted_citations)
        else:
            return self._generate_standard_bibliography(sorted_citations)

    def _generate_standard_bibliography(self, citations: List[FormattedCitation]) -> str:
        """Generate standard bibliography format."""

        bibliography_lines = ["## References", ""]

        for citation in citations:
            line = citation.formatted_text

            # Add confidence score if configured
            if self.config.show_confidence_scores:
                line += f" (Confidence: {citation.metadata.confidence_score:.2f})"

            bibliography_lines.append(line)

        return "\n".join(bibliography_lines)

    def _generate_grouped_bibliography(self, citations: List[FormattedCitation]) -> str:
        """Generate grouped bibliography by document."""

        # Group citations by document
        document_groups = {}
        for citation in citations:
            doc_id = citation.metadata.document_id
            if doc_id not in document_groups:
                document_groups[doc_id] = []
            document_groups[doc_id].append(citation)

        bibliography_lines = ["## References", ""]

        for doc_id, doc_citations in document_groups.items():
            # Document header
            doc_title = doc_citations[0].metadata.document_title
            bibliography_lines.append(f"### {doc_title}")
            bibliography_lines.append("")

            # Citations for this document
            for citation in doc_citations:
                line = citation.formatted_text
                if self.config.show_confidence_scores:
                    line += f" (Confidence: {citation.metadata.confidence_score:.2f})"
                bibliography_lines.append(line)

            bibliography_lines.append("")

        return "\n".join(bibliography_lines)

    def validate_all_citations(self, citations: List[FormattedCitation]) -> Dict[str, Any]:
        """Validate all citations and return validation report."""

        validation_report = {
            "total_citations": len(citations),
            "valid_citations": 0,
            "invalid_citations": 0,
            "citations_with_warnings": 0,
            "validation_errors": [],
            "validation_warnings": [],
            "recommendations": []
        }

        for citation in citations:
            if citation.is_valid:
                validation_report["valid_citations"] += 1
            else:
                validation_report["invalid_citations"] += 1
                validation_report["validation_errors"].extend(citation.validation_errors)

            if citation.validation_warnings:
                validation_report["citations_with_warnings"] += 1
                validation_report["validation_warnings"].extend(citation.validation_warnings)

        # Generate recommendations
        if validation_report["invalid_citations"] > 0:
            validation_report["recommendations"].append(
                "Review and fix invalid citations before publication"
            )

        if validation_report["citations_with_warnings"] > len(citations) * 0.5:
            validation_report["recommendations"].append(
                "Consider improving citation metadata completeness"
            )

        return validation_report

    def get_citation_statistics(self) -> Dict[str, Any]:
        """Get citation system statistics."""

        stats = self._stats.copy()
        stats.update({
            "total_indexed_citations": self.citation_index.total_citations,
            "unique_documents": self.citation_index.unique_documents,
            "duplicate_groups": len(self.citation_index.duplicate_groups),
            "format_style": self.config.format_style.value,
            "deduplication_enabled": self.config.enable_deduplication,
            "validation_level": self.config.validation_level.value
        })

        return stats

    def export_citations(self, citations: List[FormattedCitation], format: str = "json") -> str:
        """Export citations in various formats."""

        if format.lower() == "json":
            return self._export_json(citations)
        elif format.lower() == "bibtex":
            return self._export_bibtex(citations)
        elif format.lower() == "csv":
            return self._export_csv(citations)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, citations: List[FormattedCitation]) -> str:
        """Export citations as JSON."""

        export_data = []
        for citation in citations:
            data = {
                "id": citation.id,
                "citation_key": citation.citation_key,
                "formatted_text": citation.formatted_text,
                "document_title": citation.metadata.document_title,
                "author": citation.metadata.author,
                "publication_date": citation.metadata.publication_date.isoformat() if citation.metadata.publication_date else None,
                "page_number": citation.metadata.page_number,
                "section_title": citation.metadata.section_title,
                "confidence_score": citation.metadata.confidence_score,
                "usage_count": citation.usage_count
            }
            export_data.append(data)

        return json.dumps(export_data, indent=2)

    def _export_bibtex(self, citations: List[FormattedCitation]) -> str:
        """Export citations as BibTeX."""

        bibtex_entries = []

        for citation in citations:
            entry_type = "article"  # Default type
            if citation.metadata.document_type:
                type_mapping = {
                    "book": "book",
                    "article": "article",
                    "report": "techreport",
                    "thesis": "phdthesis"
                }
                entry_type = type_mapping.get(citation.metadata.document_type.lower(), "misc")

            entry = f"@{entry_type}{{{citation.citation_key},\n"
            entry += f"  title = {{{citation.metadata.document_title}}},\n"

            if citation.metadata.author:
                entry += f"  author = {{{citation.metadata.author}}},\n"

            if citation.metadata.publication_date:
                entry += f"  year = {{{citation.metadata.publication_date.year}}},\n"

            if citation.metadata.page_number:
                entry += f"  pages = {{{citation.metadata.page_number}}},\n"

            entry += "}\n"
            bibtex_entries.append(entry)

        return "\n".join(bibtex_entries)

    def _export_csv(self, citations: List[FormattedCitation]) -> str:
        """Export citations as CSV."""

        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "ID", "Citation Key", "Document Title", "Author", "Publication Date",
            "Page Number", "Section Title", "Confidence Score", "Usage Count"
        ])

        # Data rows
        for citation in citations:
            writer.writerow([
                citation.id,
                citation.citation_key,
                citation.metadata.document_title,
                citation.metadata.author or "",
                citation.metadata.publication_date.isoformat() if citation.metadata.publication_date else "",
                citation.metadata.page_number or "",
                citation.metadata.section_title or "",
                citation.metadata.confidence_score,
                citation.usage_count
            ])

        return output.getvalue()


# Global service instance
_citation_system_service: Optional[CitationSystemService] = None


def get_citation_system_service(config: Optional[CitationConfig] = None) -> CitationSystemService:
    """Get or create the global citation system service instance."""
    global _citation_system_service

    if _citation_system_service is None:
        _citation_system_service = CitationSystemService(config)

    return _citation_system_service


def reset_citation_system_service() -> None:
    """Reset the global citation system service instance."""
    global _citation_system_service
    _citation_system_service = None
