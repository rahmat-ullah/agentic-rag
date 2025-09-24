"""
Query Processing Service

This module provides natural language query processing capabilities including
preprocessing, cleaning, normalization, expansion, and embedding generation
for semantic search operations.
"""

import re
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from agentic_rag.services.embedding_pipeline import get_embedding_pipeline, EmbeddingPipelineRequest
from agentic_rag.services.vector_store import VectorMetadata
from agentic_rag.services.advanced_query_preprocessor import (
    get_advanced_query_preprocessor,
    PreprocessingConfig,
    PreprocessingLevel,
    SpellCheckMode
)

logger = structlog.get_logger(__name__)


class QueryType(str, Enum):
    """Types of search queries."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    QUESTION = "question"
    PHRASE = "phrase"


class QueryIntent(str, Enum):
    """Intent classification for queries."""
    SEARCH = "search"
    QUESTION = "question"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    REQUIREMENT = "requirement"


@dataclass
class QueryAnalysis:
    """Analysis results for a query."""
    original_query: str
    cleaned_query: str
    normalized_query: str
    expanded_query: str
    query_type: QueryType
    intent: QueryIntent
    key_terms: List[str]
    entities: List[str]
    confidence: float
    processing_time_ms: int


class ProcessedQuery(BaseModel):
    """Processed query with embeddings and metadata."""
    
    original_query: str = Field(..., description="Original user query")
    processed_query: str = Field(..., description="Processed and normalized query")
    expanded_query: str = Field(..., description="Query with expansions")
    embedding: List[float] = Field(..., description="Query embedding vector")
    query_type: QueryType = Field(..., description="Detected query type")
    intent: QueryIntent = Field(..., description="Detected query intent")
    key_terms: List[str] = Field(..., description="Extracted key terms")
    entities: List[str] = Field(..., description="Extracted entities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Processing confidence")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")


class QueryProcessor:
    """Service for processing natural language queries."""
    
    def __init__(self, preprocessing_config: Optional[PreprocessingConfig] = None):
        self._embedding_pipeline = None
        self._advanced_preprocessor = None
        self._preprocessing_config = preprocessing_config or PreprocessingConfig(
            level=PreprocessingLevel.STANDARD,
            spell_check_mode=SpellCheckMode.BASIC,
            enable_stop_word_removal=True,
            enable_stemming=False,  # Keep disabled for now to maintain compatibility
            enable_query_expansion=True,
            enable_synonym_expansion=True
        )

        # Statistics tracking
        self._stats = {
            "queries_processed": 0,
            "spell_corrections": 0,
            "query_expansions": 0,
            "average_processing_time_ms": 0.0
        }
        
        # Query processing patterns
        self._stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'these', 'those'
        }
        
        # Question patterns
        self._question_patterns = [
            r'\b(what|how|when|where|why|who|which)\b',
            r'\?$',
            r'\b(explain|describe|define|tell me)\b'
        ]
        
        # Requirement patterns
        self._requirement_patterns = [
            r'\b(requirement|must|shall|should|need|necessary)\b',
            r'\b(specification|criteria|standard)\b'
        ]
        
        # Comparison patterns
        self._comparison_patterns = [
            r'\b(compare|versus|vs|difference|similar|different)\b',
            r'\b(better|worse|best|worst)\b'
        ]
        
        # Entity patterns (simple regex-based)
        self._entity_patterns = {
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'number': r'\b\d+(?:\.\d+)?\b',
            'percentage': r'\b\d+(?:\.\d+)?%\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b'
        }
        
        # Query expansion terms
        self._expansion_terms = {
            'data': ['information', 'dataset', 'records'],
            'process': ['procedure', 'workflow', 'method'],
            'system': ['platform', 'application', 'software'],
            'requirement': ['specification', 'criteria', 'standard'],
            'performance': ['efficiency', 'speed', 'throughput'],
            'security': ['protection', 'safety', 'authentication'],
            'integration': ['connection', 'interface', 'compatibility']
        }
        
        logger.info("Query processor initialized with advanced preprocessing",
                   level=self._preprocessing_config.level.value,
                   spell_check=self._preprocessing_config.spell_check_mode.value)
    
    async def initialize(self) -> None:
        """Initialize the query processor."""
        try:
            self._embedding_pipeline = await get_embedding_pipeline()
            logger.info("Query processor connected to embedding pipeline")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding pipeline: {e}")
            self._embedding_pipeline = None

        try:
            self._advanced_preprocessor = await get_advanced_query_preprocessor(self._preprocessing_config)
            logger.info("Query processor connected to advanced preprocessor")
        except Exception as e:
            logger.warning(f"Failed to initialize advanced preprocessor: {e}")
            self._advanced_preprocessor = None
    
    async def process_query(
        self,
        query: str,
        tenant_id: str,
        expand_query: bool = True,
        generate_embedding: bool = True
    ) -> ProcessedQuery:
        """
        Process a natural language query through the complete pipeline.
        
        Args:
            query: Raw user query
            tenant_id: Tenant identifier
            expand_query: Whether to expand the query with related terms
            generate_embedding: Whether to generate embedding vector
            
        Returns:
            ProcessedQuery with all processing results
        """
        start_time = time.time()
        
        if not self._embedding_pipeline:
            await self.initialize()
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Use advanced preprocessing if available
            if self._advanced_preprocessor:
                preprocessed = await self._advanced_preprocessor.preprocess_query(query)

                # Extract processed components
                cleaned_query = preprocessed.cleaned_query
                normalized_query = preprocessed.normalized_query
                expanded_query = preprocessed.expanded_query if expand_query else normalized_query

                # Update stats
                if preprocessed.spell_corrections:
                    self._stats["spell_corrections"] += len(preprocessed.spell_corrections)
                if preprocessed.expanded_terms:
                    self._stats["query_expansions"] += 1

                # Create analysis from preprocessed data
                analysis = QueryAnalysis(
                    original_query=query,
                    cleaned_query=cleaned_query,
                    normalized_query=normalized_query,
                    expanded_query=expanded_query,
                    query_type=self._detect_query_type(normalized_query),
                    intent=self._detect_intent(normalized_query),
                    key_terms=preprocessed.key_terms,
                    entities=self._extract_entities(normalized_query),
                    confidence=preprocessed.confidence_score,
                    processing_time_ms=preprocessed.processing_time_ms
                )
            else:
                # Fallback to basic processing
                cleaned_query = self._clean_query(query)
                normalized_query = self._normalize_query(cleaned_query)
                analysis = self._analyze_query(normalized_query)

                expanded_query = normalized_query
                if expand_query:
                    expanded_query = self._expand_query(normalized_query, analysis.key_terms)

            # Generate embedding if requested
            embedding = []
            if generate_embedding:
                embedding = await self._generate_query_embedding(expanded_query, tenant_id)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            processed_query = ProcessedQuery(
                original_query=query,
                processed_query=normalized_query,
                expanded_query=expanded_query,
                embedding=embedding,
                query_type=analysis.query_type,
                intent=analysis.intent,
                key_terms=analysis.key_terms,
                entities=analysis.entities,
                confidence=analysis.confidence,
                processing_time_ms=processing_time_ms
            )
            
            logger.info(
                f"Query processed successfully",
                query_type=analysis.query_type.value,
                intent=analysis.intent.value,
                key_terms_count=len(analysis.key_terms),
                processing_time_ms=processing_time_ms
            )
            
            return processed_query
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise
    
    def _clean_query(self, query: str) -> str:
        """Clean and sanitize the query."""
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters but keep basic punctuation
        query = re.sub(r'[^\w\s\-\.\?\!]', ' ', query)
        
        # Remove multiple punctuation
        query = re.sub(r'[\.]{2,}', '.', query)
        query = re.sub(r'[\?]{2,}', '?', query)
        query = re.sub(r'[\!]{2,}', '!', query)
        
        # Normalize whitespace again
        query = re.sub(r'\s+', ' ', query.strip())
        
        return query
    
    def _normalize_query(self, query: str) -> str:
        """Normalize the query for better processing."""
        # Convert to lowercase
        normalized = query.lower()
        
        # Expand common contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        return normalized
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine type, intent, and extract key information."""
        start_time = time.time()
        
        # Detect query type
        query_type = self._detect_query_type(query)
        
        # Detect intent
        intent = self._detect_intent(query)
        
        # Extract key terms
        key_terms = self._extract_key_terms(query)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Calculate confidence based on pattern matches
        confidence = self._calculate_confidence(query, query_type, intent)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return QueryAnalysis(
            original_query=query,
            cleaned_query=query,
            normalized_query=query,
            expanded_query=query,
            query_type=query_type,
            intent=intent,
            key_terms=key_terms,
            entities=entities,
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query."""
        # Check for questions
        for pattern in self._question_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.QUESTION
        
        # Check for phrases (quoted text)
        if '"' in query or "'" in query:
            return QueryType.PHRASE
        
        # Check for boolean operators
        if any(op in query.lower() for op in ['and', 'or', 'not']):
            return QueryType.HYBRID
        
        # Default to semantic search
        return QueryType.SEMANTIC
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect the intent of the query."""
        # Check for questions
        for pattern in self._question_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryIntent.QUESTION
        
        # Check for requirements
        for pattern in self._requirement_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryIntent.REQUIREMENT
        
        # Check for comparisons
        for pattern in self._comparison_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryIntent.COMPARISON
        
        # Check for definitions
        if re.search(r'\b(define|definition|what is|meaning)\b', query, re.IGNORECASE):
            return QueryIntent.DEFINITION
        
        # Check for procedures
        if re.search(r'\b(how to|procedure|steps|process)\b', query, re.IGNORECASE):
            return QueryIntent.PROCEDURE
        
        # Default to search
        return QueryIntent.SEARCH

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from the query."""
        # Split into words
        words = re.findall(r'\b\w+\b', query.lower())

        # Remove stop words
        key_terms = [word for word in words if word not in self._stop_words]

        # Remove very short words
        key_terms = [word for word in key_terms if len(word) > 2]

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from the query using simple patterns."""
        entities = []

        for entity_type, pattern in self._entity_patterns.items():
            matches = re.findall(pattern, query)
            for match in matches:
                entities.append(f"{entity_type}:{match}")

        return entities

    def _calculate_confidence(self, query: str, query_type: QueryType, intent: QueryIntent) -> float:
        """Calculate confidence score for query analysis."""
        confidence = 0.5  # Base confidence

        # Increase confidence for clear patterns
        if query_type == QueryType.QUESTION:
            for pattern in self._question_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    confidence += 0.2
                    break

        if intent == QueryIntent.REQUIREMENT:
            for pattern in self._requirement_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    confidence += 0.2
                    break

        # Increase confidence for longer, more specific queries
        word_count = len(query.split())
        if word_count > 5:
            confidence += 0.1
        if word_count > 10:
            confidence += 0.1

        # Cap at 1.0
        return min(confidence, 1.0)

    def _expand_query(self, query: str, key_terms: List[str]) -> str:
        """Expand query with related terms."""
        expanded_terms = []

        for term in key_terms:
            expanded_terms.append(term)

            # Add expansion terms if available
            if term in self._expansion_terms:
                expanded_terms.extend(self._expansion_terms[term])

        # Remove duplicates
        unique_terms = list(dict.fromkeys(expanded_terms))

        # Create expanded query
        expanded_query = ' '.join(unique_terms)

        return expanded_query

    async def _generate_query_embedding(self, query: str, tenant_id: str) -> List[float]:
        """Generate embedding for the processed query."""
        if not self._embedding_pipeline:
            logger.warning("Embedding pipeline not available, returning empty embedding")
            return []

        try:
            # Create a dummy metadata for the embedding request
            dummy_metadata = VectorMetadata(
                tenant_id=tenant_id,
                document_id="query",
                chunk_id="query",
                document_kind="QUERY"
            )

            # Create embedding request
            embedding_request = EmbeddingPipelineRequest(
                texts=[query],
                metadata_list=[dummy_metadata],
                tenant_id=tenant_id,
                document_id="query",
                store_vectors=False,  # Don't store query embeddings
                validate_quality=False,  # Skip quality validation for queries
                optimize_cost=True  # Use cost optimization
            )

            # Generate embedding
            result = await self._embedding_pipeline.process_embeddings(embedding_request)

            if result.embeddings and len(result.embeddings) > 0:
                return result.embeddings[0]
            else:
                logger.warning("No embedding generated for query")
                return []

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []  # Return empty instead of raising

    async def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a query for safety and appropriateness.

        Args:
            query: Query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check length
        if len(query) > 1000:
            return False, "Query too long (maximum 1000 characters)"

        if len(query.strip()) == 0:
            return False, "Query cannot be empty"

        # Check for potentially malicious patterns
        malicious_patterns = [
            r'<script',
            r'javascript:',
            r'eval\(',
            r'exec\(',
            r'system\(',
            r'__import__'
        ]

        for pattern in malicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains potentially unsafe content"

        return True, None

    async def get_query_suggestions(
        self,
        partial_query: str,
        tenant_id: str,
        limit: int = 5
    ) -> List[str]:
        """
        Generate query suggestions based on partial input.

        Args:
            partial_query: Partial query text
            tenant_id: Tenant identifier
            limit: Maximum number of suggestions

        Returns:
            List of suggested queries
        """
        # For now, implement simple suggestions based on common patterns
        # In a production system, this would use search history and document analysis

        suggestions = []

        # Common query starters
        starters = [
            "what is",
            "how to",
            "requirements for",
            "process for",
            "definition of",
            "examples of",
            "best practices for",
            "guidelines for"
        ]

        # If partial query is short, suggest starters
        if len(partial_query.split()) <= 2:
            for starter in starters:
                if starter.startswith(partial_query.lower()):
                    suggestions.append(starter + " " + partial_query.split()[-1] if len(partial_query.split()) > 1 else starter)

        # Add domain-specific suggestions
        domain_terms = [
            "data processing",
            "security requirements",
            "performance metrics",
            "integration standards",
            "compliance requirements",
            "technical specifications"
        ]

        for term in domain_terms:
            if partial_query.lower() in term or term.startswith(partial_query.lower()):
                suggestions.append(term)

        # Limit results
        return suggestions[:limit]


# Global instance
_query_processor: Optional[QueryProcessor] = None


async def get_query_processor() -> QueryProcessor:
    """Get the global query processor instance."""
    global _query_processor

    if _query_processor is None:
        _query_processor = QueryProcessor()
        await _query_processor.initialize()

    return _query_processor
