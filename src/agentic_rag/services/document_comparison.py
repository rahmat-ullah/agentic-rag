"""
Document Comparison and Difference Analysis Service

This module implements comprehensive document comparison capabilities including:
- Document alignment algorithms
- Section-by-section comparison
- Change detection and highlighting
- Difference categorization
- Comparison confidence scoring
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from uuid import uuid4
from enum import Enum
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import structlog
from pydantic import BaseModel, Field

from agentic_rag.models.database import Document, DocumentChunk
from agentic_rag.services.document_similarity import DocumentSimilarityService

logger = structlog.get_logger(__name__)


class ChangeType(str, Enum):
    """Types of changes detected in document comparison."""
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    MOVE = "move"
    FORMAT_CHANGE = "format_change"
    NO_CHANGE = "no_change"


class DifferenceCategory(str, Enum):
    """Categories for organizing differences."""
    CONTENT = "content"
    STRUCTURE = "structure"
    FORMATTING = "formatting"
    METADATA = "metadata"
    TECHNICAL = "technical"
    PRICING = "pricing"
    REQUIREMENTS = "requirements"


class ConfidenceLevel(str, Enum):
    """Confidence levels for comparison results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class DocumentSection:
    """Represents a section of a document for comparison."""
    section_id: str
    title: str
    content: str
    level: int
    start_position: int
    end_position: int
    metadata: Dict[str, Any]


@dataclass
class SectionPair:
    """Represents a pair of aligned sections for comparison."""
    section1: Optional[DocumentSection]
    section2: Optional[DocumentSection]
    alignment_confidence: float
    similarity_score: float


@dataclass
class Change:
    """Represents a specific change between document sections."""
    change_id: str
    change_type: ChangeType
    category: DifferenceCategory
    description: str
    section1_id: Optional[str]
    section2_id: Optional[str]
    old_content: Optional[str]
    new_content: Optional[str]
    confidence: float
    impact_level: str  # "low", "medium", "high"
    position: int


class ComparisonResult(BaseModel):
    """Result of document comparison analysis."""
    comparison_id: str = Field(default_factory=lambda: str(uuid4()))
    document1_id: str
    document2_id: str
    overall_similarity: float
    total_differences: int
    major_changes: int
    minor_changes: int
    changes: List[Change]
    section_pairs: List[SectionPair]
    summary: Dict[str, Any]
    recommendations: List[str]
    confidence_level: ConfidenceLevel
    processing_time_ms: float
    created_at: str


class DocumentComparisonConfig(BaseModel):
    """Configuration for document comparison."""
    similarity_threshold: float = 0.7
    section_alignment_threshold: float = 0.6
    change_detection_sensitivity: float = 0.8
    max_section_distance: int = 5
    enable_semantic_comparison: bool = True
    enable_structural_analysis: bool = True
    enable_formatting_detection: bool = True
    confidence_threshold: float = 0.5
    max_processing_time_seconds: int = 300


class DocumentComparisonService:
    """Service for comprehensive document comparison and difference analysis."""
    
    def __init__(self, config: Optional[DocumentComparisonConfig] = None):
        self.config = config or DocumentComparisonConfig()
        self.similarity_service = DocumentSimilarityService()
        
        # Statistics tracking
        self._stats = {
            "total_comparisons": 0,
            "successful_comparisons": 0,
            "failed_comparisons": 0,
            "total_processing_time_ms": 0.0,
            "average_processing_time_ms": 0.0,
            "total_changes_detected": 0,
            "total_sections_aligned": 0
        }
        
        # Cache for repeated comparisons
        self._comparison_cache: Dict[str, ComparisonResult] = {}
        
        logger.info("Document comparison service initialized")
    
    async def compare_documents(self, doc1: Document, doc2: Document) -> ComparisonResult:
        """Compare two documents and identify differences."""
        
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = f"{doc1.id}:{doc2.id}:{hash(doc1.content + doc2.content)}"
            
            # Check cache
            if cache_key in self._comparison_cache:
                logger.info("Returning cached comparison result", 
                          doc1_id=doc1.id, doc2_id=doc2.id)
                return self._comparison_cache[cache_key]
            
            # Extract sections from both documents
            sections1 = await self._extract_sections(doc1)
            sections2 = await self._extract_sections(doc2)
            
            # Align comparable sections
            section_pairs = await self._align_sections(sections1, sections2)
            
            # Detect changes between aligned sections
            changes = await self._detect_changes(section_pairs)
            
            # Calculate overall similarity
            overall_similarity = await self._calculate_overall_similarity(doc1, doc2)
            
            # Categorize and analyze changes
            categorized_changes = self._categorize_changes(changes)
            major_changes = len([c for c in categorized_changes if c.impact_level == "high"])
            minor_changes = len([c for c in categorized_changes if c.impact_level in ["low", "medium"]])
            
            # Generate summary and recommendations
            summary = self._generate_summary(categorized_changes, section_pairs)
            recommendations = self._generate_recommendations(categorized_changes)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(categorized_changes, section_pairs)
            
            # Create result
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = ComparisonResult(
                document1_id=doc1.id,
                document2_id=doc2.id,
                overall_similarity=overall_similarity,
                total_differences=len(categorized_changes),
                major_changes=major_changes,
                minor_changes=minor_changes,
                changes=categorized_changes,
                section_pairs=section_pairs,
                summary=summary,
                recommendations=recommendations,
                confidence_level=confidence_level,
                processing_time_ms=processing_time_ms,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Cache result
            self._comparison_cache[cache_key] = result
            
            # Update statistics
            self._update_stats("success", processing_time_ms, len(categorized_changes), len(section_pairs))
            
            logger.info(
                "Document comparison completed",
                doc1_id=doc1.id,
                doc2_id=doc2.id,
                total_differences=len(categorized_changes),
                major_changes=major_changes,
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats("failure", processing_time_ms, 0, 0)
            
            logger.error(
                "Document comparison failed",
                error=str(e),
                doc1_id=doc1.id,
                doc2_id=doc2.id,
                processing_time_ms=processing_time_ms
            )
            raise
    
    async def _extract_sections(self, document: Document) -> List[DocumentSection]:
        """Extract sections from a document for comparison."""
        
        sections = []
        content = document.content or ""
        
        # Split content into sections based on headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = None
        current_content = []
        position = 0
        
        for i, line in enumerate(lines):
            header_match = re.match(header_pattern, line, re.MULTILINE)
            
            if header_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    current_section.end_position = position
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                current_section = DocumentSection(
                    section_id=str(uuid4()),
                    title=title,
                    content="",
                    level=level,
                    start_position=position,
                    end_position=0,
                    metadata={"line_number": i + 1}
                )
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
            
            position += len(line) + 1  # +1 for newline
        
        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            current_section.end_position = position
            sections.append(current_section)
        
        # If no sections found, treat entire document as one section
        if not sections:
            sections.append(DocumentSection(
                section_id=str(uuid4()),
                title="Document Content",
                content=content,
                level=1,
                start_position=0,
                end_position=len(content),
                metadata={"full_document": True}
            ))
        
        return sections

    async def _align_sections(self, sections1: List[DocumentSection],
                            sections2: List[DocumentSection]) -> List[SectionPair]:
        """Align comparable sections between documents."""

        section_pairs = []
        used_sections2 = set()

        for section1 in sections1:
            best_match = None
            best_score = 0.0
            best_section2 = None

            for section2 in sections2:
                if section2.section_id in used_sections2:
                    continue

                # Calculate similarity score
                similarity = self._calculate_section_similarity(section1, section2)

                if similarity > best_score and similarity >= self.config.section_alignment_threshold:
                    best_score = similarity
                    best_section2 = section2
                    best_match = similarity

            if best_match:
                used_sections2.add(best_section2.section_id)
                section_pairs.append(SectionPair(
                    section1=section1,
                    section2=best_section2,
                    alignment_confidence=best_match,
                    similarity_score=best_score
                ))
            else:
                # Section exists only in document 1
                section_pairs.append(SectionPair(
                    section1=section1,
                    section2=None,
                    alignment_confidence=1.0,
                    similarity_score=0.0
                ))

        # Add sections that exist only in document 2
        for section2 in sections2:
            if section2.section_id not in used_sections2:
                section_pairs.append(SectionPair(
                    section1=None,
                    section2=section2,
                    alignment_confidence=1.0,
                    similarity_score=0.0
                ))

        return section_pairs

    def _calculate_section_similarity(self, section1: DocumentSection,
                                    section2: DocumentSection) -> float:
        """Calculate similarity between two sections."""

        # Title similarity (weighted 40%)
        title_similarity = SequenceMatcher(None, section1.title.lower(),
                                         section2.title.lower()).ratio()

        # Content similarity (weighted 50%)
        content_similarity = SequenceMatcher(None, section1.content.lower(),
                                           section2.content.lower()).ratio()

        # Level similarity (weighted 10%)
        level_similarity = 1.0 if section1.level == section2.level else 0.5

        # Composite similarity
        similarity = (title_similarity * 0.4 +
                     content_similarity * 0.5 +
                     level_similarity * 0.1)

        return similarity

    async def _detect_changes(self, section_pairs: List[SectionPair]) -> List[Change]:
        """Detect specific changes between aligned sections."""

        changes = []

        for pair in section_pairs:
            if pair.section1 and pair.section2:
                # Both sections exist - check for modifications
                section_changes = self._detect_section_changes(pair.section1, pair.section2)
                changes.extend(section_changes)
            elif pair.section1 and not pair.section2:
                # Section deleted
                changes.append(Change(
                    change_id=str(uuid4()),
                    change_type=ChangeType.DELETION,
                    category=DifferenceCategory.CONTENT,
                    description=f"Section '{pair.section1.title}' was deleted",
                    section1_id=pair.section1.section_id,
                    section2_id=None,
                    old_content=pair.section1.content[:200] + "..." if len(pair.section1.content) > 200 else pair.section1.content,
                    new_content=None,
                    confidence=0.9,
                    impact_level="medium",
                    position=pair.section1.start_position
                ))
            elif not pair.section1 and pair.section2:
                # Section added
                changes.append(Change(
                    change_id=str(uuid4()),
                    change_type=ChangeType.ADDITION,
                    category=DifferenceCategory.CONTENT,
                    description=f"Section '{pair.section2.title}' was added",
                    section1_id=None,
                    section2_id=pair.section2.section_id,
                    old_content=None,
                    new_content=pair.section2.content[:200] + "..." if len(pair.section2.content) > 200 else pair.section2.content,
                    confidence=0.9,
                    impact_level="medium",
                    position=pair.section2.start_position
                ))

        return changes

    def _detect_section_changes(self, section1: DocumentSection,
                              section2: DocumentSection) -> List[Change]:
        """Detect changes within a pair of aligned sections."""

        changes = []

        # Title change
        if section1.title != section2.title:
            changes.append(Change(
                change_id=str(uuid4()),
                change_type=ChangeType.MODIFICATION,
                category=DifferenceCategory.STRUCTURE,
                description=f"Section title changed from '{section1.title}' to '{section2.title}'",
                section1_id=section1.section_id,
                section2_id=section2.section_id,
                old_content=section1.title,
                new_content=section2.title,
                confidence=0.95,
                impact_level="low",
                position=section1.start_position
            ))

        # Content changes
        if section1.content != section2.content:
            # Use difflib to find specific changes
            matcher = SequenceMatcher(None, section1.content, section2.content)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    changes.append(Change(
                        change_id=str(uuid4()),
                        change_type=ChangeType.MODIFICATION,
                        category=self._categorize_content_change(section1.content[i1:i2], section2.content[j1:j2]),
                        description=f"Content modified in section '{section1.title}'",
                        section1_id=section1.section_id,
                        section2_id=section2.section_id,
                        old_content=section1.content[i1:i2],
                        new_content=section2.content[j1:j2],
                        confidence=0.8,
                        impact_level=self._assess_change_impact(section1.content[i1:i2], section2.content[j1:j2]),
                        position=section1.start_position + i1
                    ))
                elif tag == 'delete':
                    changes.append(Change(
                        change_id=str(uuid4()),
                        change_type=ChangeType.DELETION,
                        category=DifferenceCategory.CONTENT,
                        description=f"Content deleted from section '{section1.title}'",
                        section1_id=section1.section_id,
                        section2_id=section2.section_id,
                        old_content=section1.content[i1:i2],
                        new_content=None,
                        confidence=0.9,
                        impact_level=self._assess_change_impact(section1.content[i1:i2], ""),
                        position=section1.start_position + i1
                    ))
                elif tag == 'insert':
                    changes.append(Change(
                        change_id=str(uuid4()),
                        change_type=ChangeType.ADDITION,
                        category=DifferenceCategory.CONTENT,
                        description=f"Content added to section '{section1.title}'",
                        section1_id=section1.section_id,
                        section2_id=section2.section_id,
                        old_content=None,
                        new_content=section2.content[j1:j2],
                        confidence=0.9,
                        impact_level=self._assess_change_impact("", section2.content[j1:j2]),
                        position=section1.start_position + i1
                    ))

        return changes

    def _categorize_content_change(self, old_content: str, new_content: str) -> DifferenceCategory:
        """Categorize the type of content change."""

        # Check for pricing-related changes
        pricing_patterns = [r'\$[\d,]+\.?\d*', r'price', r'cost', r'budget', r'fee']
        if any(re.search(pattern, old_content.lower()) or re.search(pattern, new_content.lower())
               for pattern in pricing_patterns):
            return DifferenceCategory.PRICING

        # Check for requirements changes
        requirement_patterns = [r'requirement', r'specification', r'must', r'shall', r'should']
        if any(re.search(pattern, old_content.lower()) or re.search(pattern, new_content.lower())
               for pattern in requirement_patterns):
            return DifferenceCategory.REQUIREMENTS

        # Check for technical changes
        technical_patterns = [r'technical', r'system', r'software', r'hardware', r'API']
        if any(re.search(pattern, old_content.lower()) or re.search(pattern, new_content.lower())
               for pattern in technical_patterns):
            return DifferenceCategory.TECHNICAL

        return DifferenceCategory.CONTENT

    def _assess_change_impact(self, old_content: str, new_content: str) -> str:
        """Assess the impact level of a change."""

        # Calculate change magnitude
        old_len = len(old_content)
        new_len = len(new_content)

        if old_len == 0 and new_len == 0:
            return "low"

        # Large changes are high impact
        if max(old_len, new_len) > 500:
            return "high"

        # Check for critical keywords
        critical_keywords = ['requirement', 'must', 'shall', 'critical', 'mandatory', 'price', 'cost']
        if any(keyword in old_content.lower() or keyword in new_content.lower()
               for keyword in critical_keywords):
            return "high"

        # Medium changes
        if max(old_len, new_len) > 100:
            return "medium"

        return "low"

    async def _calculate_overall_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate overall similarity between two documents."""

        try:
            # Use document similarity service if available
            similarity_result = await self.similarity_service.calculate_similarity(doc1, doc2)
            return similarity_result.similarity_score
        except Exception:
            # Fallback to simple text similarity
            content1 = doc1.content or ""
            content2 = doc2.content or ""
            return SequenceMatcher(None, content1, content2).ratio()

    def _categorize_changes(self, changes: List[Change]) -> List[Change]:
        """Categorize and enhance change analysis."""

        # Changes are already categorized in detection, but we can enhance them here
        for change in changes:
            # Add additional metadata
            change.metadata = {
                "word_count_old": len(change.old_content.split()) if change.old_content else 0,
                "word_count_new": len(change.new_content.split()) if change.new_content else 0,
                "character_count_old": len(change.old_content) if change.old_content else 0,
                "character_count_new": len(change.new_content) if change.new_content else 0
            }

        return changes

    def _generate_summary(self, changes: List[Change], section_pairs: List[SectionPair]) -> Dict[str, Any]:
        """Generate summary of comparison results."""

        # Count changes by type
        change_counts = {}
        for change_type in ChangeType:
            change_counts[change_type.value] = len([c for c in changes if c.change_type == change_type])

        # Count changes by category
        category_counts = {}
        for category in DifferenceCategory:
            category_counts[category.value] = len([c for c in changes if c.category == category])

        # Calculate statistics
        total_sections1 = len([p for p in section_pairs if p.section1])
        total_sections2 = len([p for p in section_pairs if p.section2])
        aligned_sections = len([p for p in section_pairs if p.section1 and p.section2])

        return {
            "change_counts": change_counts,
            "category_counts": category_counts,
            "section_statistics": {
                "total_sections_doc1": total_sections1,
                "total_sections_doc2": total_sections2,
                "aligned_sections": aligned_sections,
                "alignment_rate": aligned_sections / max(total_sections1, total_sections2) if max(total_sections1, total_sections2) > 0 else 0
            },
            "impact_distribution": {
                "high_impact": len([c for c in changes if c.impact_level == "high"]),
                "medium_impact": len([c for c in changes if c.impact_level == "medium"]),
                "low_impact": len([c for c in changes if c.impact_level == "low"])
            }
        }

    def _generate_recommendations(self, changes: List[Change]) -> List[str]:
        """Generate recommendations based on detected changes."""

        recommendations = []

        # High impact changes
        high_impact_changes = [c for c in changes if c.impact_level == "high"]
        if high_impact_changes:
            recommendations.append(f"Review {len(high_impact_changes)} high-impact changes that may affect project scope or budget")

        # Pricing changes
        pricing_changes = [c for c in changes if c.category == DifferenceCategory.PRICING]
        if pricing_changes:
            recommendations.append(f"Analyze {len(pricing_changes)} pricing-related changes for budget impact")

        # Requirements changes
        requirement_changes = [c for c in changes if c.category == DifferenceCategory.REQUIREMENTS]
        if requirement_changes:
            recommendations.append(f"Validate {len(requirement_changes)} requirement changes against project specifications")

        # Technical changes
        technical_changes = [c for c in changes if c.category == DifferenceCategory.TECHNICAL]
        if technical_changes:
            recommendations.append(f"Assess {len(technical_changes)} technical changes for implementation feasibility")

        # Large number of changes
        if len(changes) > 20:
            recommendations.append("Consider detailed review meeting due to significant number of changes")

        # No changes
        if not changes:
            recommendations.append("Documents are identical - no action required")

        return recommendations

    def _determine_confidence_level(self, changes: List[Change],
                                  section_pairs: List[SectionPair]) -> ConfidenceLevel:
        """Determine overall confidence level of the comparison."""

        # Calculate average change confidence
        if changes:
            avg_change_confidence = sum(c.confidence for c in changes) / len(changes)
        else:
            avg_change_confidence = 1.0

        # Calculate average alignment confidence
        aligned_pairs = [p for p in section_pairs if p.section1 and p.section2]
        if aligned_pairs:
            avg_alignment_confidence = sum(p.alignment_confidence for p in aligned_pairs) / len(aligned_pairs)
        else:
            avg_alignment_confidence = 1.0

        # Overall confidence
        overall_confidence = (avg_change_confidence + avg_alignment_confidence) / 2

        if overall_confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif overall_confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif overall_confidence >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def _update_stats(self, result: str, processing_time_ms: float,
                     changes_detected: int, sections_aligned: int) -> None:
        """Update service statistics."""

        self._stats["total_comparisons"] += 1
        self._stats["total_processing_time_ms"] += processing_time_ms
        self._stats["total_changes_detected"] += changes_detected
        self._stats["total_sections_aligned"] += sections_aligned

        if result == "success":
            self._stats["successful_comparisons"] += 1
        else:
            self._stats["failed_comparisons"] += 1

        # Update average processing time
        if self._stats["total_comparisons"] > 0:
            self._stats["average_processing_time_ms"] = (
                self._stats["total_processing_time_ms"] / self._stats["total_comparisons"]
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""

        return {
            **self._stats,
            "cache_size": len(self._comparison_cache),
            "config": self.config.dict()
        }

    def clear_cache(self) -> None:
        """Clear comparison cache."""

        self._comparison_cache.clear()
        logger.info("Comparison cache cleared")

    async def compare_document_chunks(self, chunks1: List[DocumentChunk],
                                    chunks2: List[DocumentChunk]) -> ComparisonResult:
        """Compare document chunks instead of full documents."""

        # Convert chunks to pseudo-documents for comparison
        doc1_content = "\n\n".join([chunk.content for chunk in chunks1])
        doc2_content = "\n\n".join([chunk.content for chunk in chunks2])

        # Create temporary document objects
        temp_doc1 = Document(
            id=f"chunks_{chunks1[0].document_id}" if chunks1 else "empty_1",
            filename="chunks_document_1",
            content=doc1_content,
            tenant_id=chunks1[0].tenant_id if chunks1 else "default"
        )

        temp_doc2 = Document(
            id=f"chunks_{chunks2[0].document_id}" if chunks2 else "empty_2",
            filename="chunks_document_2",
            content=doc2_content,
            tenant_id=chunks2[0].tenant_id if chunks2 else "default"
        )

        return await self.compare_documents(temp_doc1, temp_doc2)

    async def batch_compare_documents(self, document_pairs: List[Tuple[Document, Document]]) -> List[ComparisonResult]:
        """Compare multiple document pairs in batch."""

        results = []

        # Process comparisons in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent comparisons

        async def compare_with_semaphore(doc1: Document, doc2: Document) -> ComparisonResult:
            async with semaphore:
                return await self.compare_documents(doc1, doc2)

        tasks = [compare_with_semaphore(doc1, doc2) for doc1, doc2 in document_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch comparison failed for pair {i}", error=str(result))
            else:
                valid_results.append(result)

        logger.info(f"Batch comparison completed: {len(valid_results)}/{len(document_pairs)} successful")

        return valid_results


# Global service instance
_document_comparison_service: Optional[DocumentComparisonService] = None


def get_document_comparison_service() -> DocumentComparisonService:
    """Get or create the global document comparison service instance."""
    global _document_comparison_service

    if _document_comparison_service is None:
        _document_comparison_service = DocumentComparisonService()

    return _document_comparison_service


def reset_document_comparison_service() -> None:
    """Reset the global document comparison service instance."""
    global _document_comparison_service
    _document_comparison_service = None
