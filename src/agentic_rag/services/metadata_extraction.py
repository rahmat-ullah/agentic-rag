"""
Metadata Extraction Service

This module extracts comprehensive metadata from parsed documents,
including document properties, structural information, content analysis,
and quality metrics.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from agentic_rag.services.content_extraction import ExtractedContent, ContentType, LayoutElement
from agentic_rag.services.docling_client import ParseResponse

logger = logging.getLogger(__name__)


class DocumentProperties(BaseModel):
    """Core document properties and metadata."""
    
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    subject: Optional[str] = Field(None, description="Document subject")
    creator: Optional[str] = Field(None, description="Document creator application")
    producer: Optional[str] = Field(None, description="Document producer")
    creation_date: Optional[datetime] = Field(None, description="Document creation date")
    modification_date: Optional[datetime] = Field(None, description="Last modification date")
    keywords: List[str] = Field(default_factory=list, description="Document keywords")
    language: Optional[str] = Field(None, description="Document language")


class StructuralMetadata(BaseModel):
    """Document structural information."""
    
    page_count: int = Field(..., description="Total number of pages")
    section_count: int = Field(default=0, description="Number of sections/headings")
    paragraph_count: int = Field(default=0, description="Number of paragraphs")
    table_count: int = Field(default=0, description="Number of tables")
    image_count: int = Field(default=0, description="Number of images")
    list_count: int = Field(default=0, description="Number of lists")
    heading_levels: List[int] = Field(default_factory=list, description="Detected heading levels")
    has_toc: bool = Field(default=False, description="Has table of contents")
    has_index: bool = Field(default=False, description="Has index")


class ContentMetrics(BaseModel):
    """Content analysis metrics."""
    
    total_characters: int = Field(..., description="Total character count")
    total_words: int = Field(..., description="Total word count")
    total_sentences: int = Field(..., description="Total sentence count")
    average_words_per_sentence: float = Field(..., description="Average words per sentence")
    reading_time_minutes: float = Field(..., description="Estimated reading time in minutes")
    complexity_score: float = Field(..., description="Text complexity score (0-1)")
    unique_words: int = Field(..., description="Number of unique words")
    vocabulary_richness: float = Field(..., description="Vocabulary richness ratio")


class QualityMetrics(BaseModel):
    """Document quality and extraction metrics."""
    
    extraction_confidence: float = Field(..., description="Overall extraction confidence")
    text_quality_score: float = Field(..., description="Text quality score (0-1)")
    table_quality_score: float = Field(..., description="Table extraction quality score")
    ocr_confidence: Optional[float] = Field(None, description="OCR confidence if applicable")
    parsing_errors: int = Field(default=0, description="Number of parsing errors")
    missing_content_ratio: float = Field(default=0.0, description="Ratio of potentially missing content")


class DocumentClassification(BaseModel):
    """Document type and classification."""
    
    document_type: str = Field(..., description="Primary document type")
    format: str = Field(..., description="File format")
    category: Optional[str] = Field(None, description="Document category")
    domain: Optional[str] = Field(None, description="Subject domain")
    is_scanned: bool = Field(default=False, description="Whether document appears to be scanned")
    is_structured: bool = Field(default=True, description="Whether document has clear structure")
    confidence: float = Field(..., description="Classification confidence")


class EnrichedMetadata(BaseModel):
    """Complete enriched metadata for a document."""
    
    document_id: str = Field(..., description="Document identifier")
    properties: DocumentProperties = Field(..., description="Document properties")
    structural: StructuralMetadata = Field(..., description="Structural metadata")
    content_metrics: ContentMetrics = Field(..., description="Content analysis metrics")
    quality_metrics: QualityMetrics = Field(..., description="Quality metrics")
    classification: DocumentClassification = Field(..., description="Document classification")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metadata extraction timestamp")
    processing_metadata: Dict = Field(default_factory=dict, description="Processing metadata")


class MetadataExtractor:
    """Service for extracting comprehensive metadata from documents."""
    
    def __init__(self):
        # Common words for vocabulary analysis
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from'
        }
        
        # Reading speed (words per minute)
        self.reading_speed_wpm = 200
        
        logger.info("Metadata extractor initialized")
    
    def extract_metadata(
        self,
        extracted_content: ExtractedContent,
        parse_response: ParseResponse,
        original_filename: str
    ) -> EnrichedMetadata:
        """
        Extract comprehensive metadata from extracted content.
        
        Args:
            extracted_content: The extracted content from the document
            parse_response: Original parse response from Granite-Docling
            original_filename: Original filename of the document
            
        Returns:
            EnrichedMetadata: Complete metadata for the document
        """
        logger.info(f"Starting metadata extraction for document {extracted_content.document_id}")
        
        # Extract document properties
        properties = self._extract_document_properties(parse_response, original_filename)
        
        # Extract structural metadata
        structural = self._extract_structural_metadata(extracted_content)
        
        # Calculate content metrics
        content_metrics = self._calculate_content_metrics(extracted_content)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(extracted_content, parse_response)
        
        # Classify document
        classification = self._classify_document(extracted_content, parse_response, original_filename)
        
        # Build enriched metadata
        enriched_metadata = EnrichedMetadata(
            document_id=extracted_content.document_id,
            properties=properties,
            structural=structural,
            content_metrics=content_metrics,
            quality_metrics=quality_metrics,
            classification=classification,
            processing_metadata=extracted_content.processing_metadata
        )
        
        logger.info(
            f"Metadata extraction completed for document {extracted_content.document_id}",
            extra={
                "page_count": structural.page_count,
                "word_count": content_metrics.total_words,
                "table_count": structural.table_count,
                "quality_score": quality_metrics.text_quality_score
            }
        )
        
        return enriched_metadata
    
    def _extract_document_properties(self, parse_response: ParseResponse, filename: str) -> DocumentProperties:
        """Extract basic document properties."""
        # Parse creation date if available
        creation_date = None
        if parse_response.metadata.creation_date:
            try:
                creation_date = datetime.fromisoformat(parse_response.metadata.creation_date.replace('Z', '+00:00'))
            except:
                pass
        
        # Parse modification date if available
        modification_date = None
        if parse_response.metadata.modification_date:
            try:
                modification_date = datetime.fromisoformat(parse_response.metadata.modification_date.replace('Z', '+00:00'))
            except:
                pass
        
        # Extract title (use filename if no title found)
        title = parse_response.metadata.title
        if not title:
            title = self._extract_title_from_filename(filename)
        
        return DocumentProperties(
            title=title,
            author=parse_response.metadata.author,
            subject=parse_response.metadata.subject,
            creator=parse_response.metadata.creator,
            producer=parse_response.metadata.producer,
            creation_date=creation_date,
            modification_date=modification_date,
            language=parse_response.metadata.language,
            keywords=[]  # Could be extracted from content analysis
        )
    
    def _extract_structural_metadata(self, content: ExtractedContent) -> StructuralMetadata:
        """Extract structural information from the document."""
        # Count different element types
        element_counts = {content_type: 0 for content_type in ContentType}
        
        for element in content.elements:
            element_counts[element.type] += 1
        
        # Detect heading levels (simplified)
        heading_levels = []
        for element in content.elements:
            if element.type == ContentType.HEADING:
                # Simple heuristic for heading level based on content
                level = self._detect_heading_level(element.content)
                if level not in heading_levels:
                    heading_levels.append(level)
        
        heading_levels.sort()
        
        # Check for table of contents and index
        has_toc = self._detect_table_of_contents(content.elements)
        has_index = self._detect_index(content.elements)
        
        return StructuralMetadata(
            page_count=content.structure.page_count,
            section_count=element_counts[ContentType.HEADING],
            paragraph_count=element_counts[ContentType.PARAGRAPH],
            table_count=len(content.tables),
            image_count=element_counts[ContentType.IMAGE],
            list_count=element_counts[ContentType.LIST],
            heading_levels=heading_levels,
            has_toc=has_toc,
            has_index=has_index
        )
    
    def _calculate_content_metrics(self, content: ExtractedContent) -> ContentMetrics:
        """Calculate content analysis metrics."""
        text = content.text_content
        
        # Basic counts
        total_characters = len(text)
        words = text.split()
        total_words = len(words)
        
        # Sentence count (simple heuristic)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        total_sentences = len(sentences)
        
        # Calculate averages
        avg_words_per_sentence = total_words / max(total_sentences, 1)
        
        # Reading time (assuming 200 WPM)
        reading_time_minutes = total_words / self.reading_speed_wpm
        
        # Vocabulary analysis
        unique_words = len(set(word.lower() for word in words if word.isalpha()))
        vocabulary_richness = unique_words / max(total_words, 1)
        
        # Complexity score (simplified)
        complexity_score = self._calculate_complexity_score(text, avg_words_per_sentence)
        
        return ContentMetrics(
            total_characters=total_characters,
            total_words=total_words,
            total_sentences=total_sentences,
            average_words_per_sentence=avg_words_per_sentence,
            reading_time_minutes=reading_time_minutes,
            complexity_score=complexity_score,
            unique_words=unique_words,
            vocabulary_richness=vocabulary_richness
        )
    
    def _calculate_quality_metrics(self, content: ExtractedContent, parse_response: ParseResponse) -> QualityMetrics:
        """Calculate document quality and extraction metrics."""
        # Overall extraction confidence (average of element confidences)
        confidences = [elem.confidence for elem in content.elements if elem.confidence is not None]
        extraction_confidence = sum(confidences) / len(confidences) if confidences else 0.8
        
        # Text quality score based on various factors
        text_quality_score = self._calculate_text_quality_score(content.text_content)
        
        # Table quality score
        table_quality_score = self._calculate_table_quality_score(content.tables)
        
        # OCR confidence (if applicable)
        ocr_confidence = None
        if parse_response.document_type in ['image', 'pdf']:
            ocr_confidence = extraction_confidence  # Simplified
        
        return QualityMetrics(
            extraction_confidence=extraction_confidence,
            text_quality_score=text_quality_score,
            table_quality_score=table_quality_score,
            ocr_confidence=ocr_confidence,
            parsing_errors=0,  # Would be tracked during parsing
            missing_content_ratio=0.0  # Would be estimated based on layout analysis
        )
    
    def _classify_document(self, content: ExtractedContent, parse_response: ParseResponse, filename: str) -> DocumentClassification:
        """Classify the document type and characteristics."""
        # Determine if document is scanned (heuristic)
        is_scanned = self._detect_scanned_document(content, parse_response)
        
        # Determine if document is well-structured
        is_structured = len([e for e in content.elements if e.type == ContentType.HEADING]) > 0
        
        # Classify document category (simplified)
        category = self._classify_document_category(content.text_content)
        
        # Determine domain (simplified)
        domain = self._classify_document_domain(content.text_content)
        
        return DocumentClassification(
            document_type=parse_response.document_type,
            format=self._extract_format_from_filename(filename),
            category=category,
            domain=domain,
            is_scanned=is_scanned,
            is_structured=is_structured,
            confidence=0.8  # Simplified confidence score
        )
    
    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract title from filename."""
        # Remove extension and clean up
        title = filename.rsplit('.', 1)[0]
        title = re.sub(r'[_-]', ' ', title)
        title = title.title()
        return title
    
    def _detect_heading_level(self, heading_text: str) -> int:
        """Detect heading level (simplified heuristic)."""
        # Simple heuristic based on text patterns
        if re.match(r'^\d+\.?\s+', heading_text):
            return 1
        elif re.match(r'^\d+\.\d+\.?\s+', heading_text):
            return 2
        elif heading_text.isupper():
            return 1
        else:
            return 2
    
    def _detect_table_of_contents(self, elements: List[LayoutElement]) -> bool:
        """Detect if document has a table of contents."""
        for element in elements:
            if element.type == ContentType.HEADING:
                if 'table of contents' in element.content.lower() or 'contents' in element.content.lower():
                    return True
        return False
    
    def _detect_index(self, elements: List[LayoutElement]) -> bool:
        """Detect if document has an index."""
        for element in elements:
            if element.type == ContentType.HEADING:
                if element.content.lower().strip() == 'index':
                    return True
        return False
    
    def _calculate_complexity_score(self, text: str, avg_words_per_sentence: float) -> float:
        """Calculate text complexity score (0-1)."""
        # Simplified complexity calculation
        # Based on sentence length and vocabulary
        
        words = text.split()
        if not words:
            return 0.0
        
        # Long words factor
        long_words = sum(1 for word in words if len(word) > 6)
        long_word_ratio = long_words / len(words)
        
        # Sentence length factor
        sentence_length_factor = min(avg_words_per_sentence / 20, 1.0)
        
        # Combine factors
        complexity = (long_word_ratio * 0.6) + (sentence_length_factor * 0.4)
        return min(complexity, 1.0)
    
    def _calculate_text_quality_score(self, text: str) -> float:
        """Calculate text quality score based on various factors."""
        if not text.strip():
            return 0.0
        
        score = 1.0
        
        # Penalize for excessive repetition
        words = text.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score *= 0.7
        
        # Penalize for too many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:') / len(text)
        if special_char_ratio > 0.1:
            score *= 0.8
        
        # Penalize for very short text
        if len(text) < 100:
            score *= 0.6
        
        return max(score, 0.0)
    
    def _calculate_table_quality_score(self, tables) -> float:
        """Calculate table extraction quality score."""
        if not tables:
            return 1.0
        
        total_score = 0.0
        for table in tables:
            score = 1.0
            
            # Penalize for too many empty cells
            if hasattr(table, 'metadata') and 'empty_cells' in table.metadata:
                total_cells = table.row_count * table.column_count
                if total_cells > 0:
                    empty_ratio = table.metadata['empty_cells'] / total_cells
                    if empty_ratio > 0.5:
                        score *= 0.5
            
            # Bonus for having headers
            if table.headers:
                score *= 1.1
            
            total_score += score
        
        return min(total_score / len(tables), 1.0)
    
    def _detect_scanned_document(self, content: ExtractedContent, parse_response: ParseResponse) -> bool:
        """Detect if document appears to be scanned."""
        # Simple heuristics for scanned document detection
        if parse_response.document_type == 'image':
            return True
        
        # Check for OCR-like artifacts in text
        text = content.text_content
        if text:
            # Look for common OCR errors
            ocr_indicators = ['rn', 'cl', 'fi', 'fl']  # Common OCR substitutions
            ocr_count = sum(text.count(indicator) for indicator in ocr_indicators)
            if ocr_count > len(text) * 0.01:  # More than 1% OCR indicators
                return True
        
        return False
    
    def _classify_document_category(self, text: str) -> Optional[str]:
        """Classify document category based on content."""
        text_lower = text.lower()
        
        # Simple keyword-based classification
        if any(word in text_lower for word in ['contract', 'agreement', 'terms']):
            return 'legal'
        elif any(word in text_lower for word in ['report', 'analysis', 'findings']):
            return 'report'
        elif any(word in text_lower for word in ['manual', 'guide', 'instructions']):
            return 'documentation'
        elif any(word in text_lower for word in ['proposal', 'rfp', 'bid']):
            return 'proposal'
        
        return None
    
    def _classify_document_domain(self, text: str) -> Optional[str]:
        """Classify document domain based on content."""
        text_lower = text.lower()
        
        # Simple domain classification
        if any(word in text_lower for word in ['procurement', 'vendor', 'supplier']):
            return 'procurement'
        elif any(word in text_lower for word in ['financial', 'budget', 'cost']):
            return 'finance'
        elif any(word in text_lower for word in ['technical', 'specification', 'engineering']):
            return 'technical'
        elif any(word in text_lower for word in ['legal', 'compliance', 'regulation']):
            return 'legal'
        
        return None
    
    def _extract_format_from_filename(self, filename: str) -> str:
        """Extract file format from filename."""
        return filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'unknown'


# Global extractor instance
_metadata_extractor: Optional[MetadataExtractor] = None


def get_metadata_extractor() -> MetadataExtractor:
    """Get or create the global metadata extractor instance."""
    global _metadata_extractor
    
    if _metadata_extractor is None:
        _metadata_extractor = MetadataExtractor()
    
    return _metadata_extractor
