"""
Advanced Query Preprocessing Service for Enhanced Search Quality.

This module provides comprehensive query preprocessing capabilities including
spell checking, advanced text cleaning, stop word removal, stemming, and
improved query expansion with synonyms.
"""

import re
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = structlog.get_logger(__name__)


class PreprocessingLevel(Enum):
    """Query preprocessing intensity levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


class SpellCheckMode(Enum):
    """Spell checking modes."""
    DISABLED = "disabled"
    BASIC = "basic"
    ADVANCED = "advanced"


@dataclass
class PreprocessingConfig:
    """Configuration for query preprocessing."""
    level: PreprocessingLevel = PreprocessingLevel.STANDARD
    spell_check_mode: SpellCheckMode = SpellCheckMode.BASIC
    enable_stop_word_removal: bool = True
    enable_stemming: bool = True
    enable_query_expansion: bool = True
    enable_synonym_expansion: bool = True
    max_expansion_terms: int = 10
    min_term_length: int = 2
    preserve_original_terms: bool = True
    custom_stop_words: Optional[Set[str]] = None
    domain_specific_terms: Optional[Dict[str, List[str]]] = None


@dataclass
class PreprocessedQuery:
    """Result of query preprocessing."""
    original_query: str
    cleaned_query: str
    normalized_query: str
    stemmed_query: str
    expanded_query: str
    spell_corrected_query: str
    removed_stop_words: List[str]
    key_terms: List[str]
    expanded_terms: List[str]
    spell_corrections: Dict[str, str]
    processing_time_ms: int
    confidence_score: float


class AdvancedQueryPreprocessor:
    """Advanced query preprocessing service with comprehensive NLP capabilities."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the advanced query preprocessor."""
        self.config = config or PreprocessingConfig()
        self._stemmer = PorterStemmer()
        self._stop_words = self._initialize_stop_words()
        self._spell_dict = self._initialize_spell_dictionary()
        self._synonym_dict = self._initialize_synonym_dictionary()
        self._domain_terms = self._initialize_domain_terms()
        
        logger.info("Advanced query preprocessor initialized", 
                   level=self.config.level.value,
                   spell_check=self.config.spell_check_mode.value)
    
    def _initialize_stop_words(self) -> Set[str]:
        """Initialize stop words set."""
        try:
            stop_words = set(stopwords.words('english'))
            
            # Add custom stop words if provided
            if self.config.custom_stop_words:
                stop_words.update(self.config.custom_stop_words)
            
            # Add domain-specific stop words
            domain_stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'among', 'within'
            }
            stop_words.update(domain_stop_words)
            
            return stop_words
        except Exception as e:
            logger.warning(f"Failed to initialize stop words: {e}")
            return set()
    
    def _initialize_spell_dictionary(self) -> Dict[str, str]:
        """Initialize spell correction dictionary."""
        # Common misspellings in business/technical context
        return {
            'recieve': 'receive',
            'seperate': 'separate',
            'occured': 'occurred',
            'neccessary': 'necessary',
            'accomodate': 'accommodate',
            'definately': 'definitely',
            'managment': 'management',
            'developement': 'development',
            'enviroment': 'environment',
            'requirment': 'requirement',
            'requirments': 'requirements',
            'specifcation': 'specification',
            'specifcations': 'specifications',
            'implmentation': 'implementation',
            'integeration': 'integration',
            'performace': 'performance',
            'availabilty': 'availability',
            'reliabilty': 'reliability',
            'scalabilty': 'scalability',
            'maintainabilty': 'maintainability',
            'compatibilty': 'compatibility'
        }
    
    def _initialize_synonym_dictionary(self) -> Dict[str, List[str]]:
        """Initialize synonym dictionary for query expansion."""
        return {
            # Business terms
            'requirement': ['specification', 'criteria', 'need', 'demand'],
            'requirements': ['specifications', 'criteria', 'needs', 'demands'],
            'proposal': ['offer', 'bid', 'submission', 'tender'],
            'contract': ['agreement', 'deal', 'arrangement', 'accord'],
            'vendor': ['supplier', 'provider', 'contractor', 'partner'],
            'client': ['customer', 'buyer', 'purchaser', 'organization'],
            'project': ['initiative', 'program', 'undertaking', 'effort'],
            'delivery': ['implementation', 'deployment', 'execution', 'completion'],
            'timeline': ['schedule', 'timeframe', 'deadline', 'duration'],
            'budget': ['cost', 'price', 'expense', 'funding'],
            'quality': ['standard', 'grade', 'level', 'excellence'],
            'performance': ['efficiency', 'effectiveness', 'capability', 'output'],
            'security': ['protection', 'safety', 'confidentiality', 'privacy'],
            'compliance': ['adherence', 'conformity', 'regulation', 'standard'],
            'integration': ['connection', 'linking', 'combination', 'merger'],
            'scalability': ['expandability', 'growth', 'flexibility', 'adaptability'],
            'maintenance': ['support', 'upkeep', 'servicing', 'care'],
            'documentation': ['records', 'manuals', 'guides', 'specifications'],
            'testing': ['validation', 'verification', 'evaluation', 'assessment'],
            'training': ['education', 'instruction', 'learning', 'development']
        }
    
    def _initialize_domain_terms(self) -> Dict[str, List[str]]:
        """Initialize domain-specific term mappings."""
        domain_terms = {}
        
        # Add custom domain terms if provided
        if self.config.domain_specific_terms:
            domain_terms.update(self.config.domain_specific_terms)
        
        return domain_terms
    
    async def preprocess_query(self, query: str) -> PreprocessedQuery:
        """
        Perform comprehensive query preprocessing.
        
        Args:
            query: Raw user query
            
        Returns:
            PreprocessedQuery with all preprocessing results
        """
        start_time = time.time()
        
        logger.info(f"Preprocessing query: {query[:100]}...")
        
        try:
            # Step 1: Initial cleaning
            cleaned_query = self._clean_query(query)
            
            # Step 2: Spell correction
            spell_corrected_query, spell_corrections = self._correct_spelling(cleaned_query)
            
            # Step 3: Normalization
            normalized_query = self._normalize_query(spell_corrected_query)
            
            # Step 4: Tokenization and stop word removal
            tokens = self._tokenize_query(normalized_query)
            filtered_tokens, removed_stop_words = self._remove_stop_words(tokens)
            
            # Step 5: Stemming
            stemmed_tokens = self._stem_tokens(filtered_tokens)
            stemmed_query = ' '.join(stemmed_tokens)
            
            # Step 6: Key term extraction
            key_terms = self._extract_key_terms(filtered_tokens)
            
            # Step 7: Query expansion
            expanded_terms = []
            expanded_query = normalized_query
            
            if self.config.enable_query_expansion:
                expanded_terms = self._expand_terms(key_terms)
                expanded_query = self._build_expanded_query(normalized_query, expanded_terms)
            
            # Step 8: Calculate confidence score
            confidence_score = self._calculate_confidence(
                query, cleaned_query, spell_corrections, key_terms
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = PreprocessedQuery(
                original_query=query,
                cleaned_query=cleaned_query,
                normalized_query=normalized_query,
                stemmed_query=stemmed_query,
                expanded_query=expanded_query,
                spell_corrected_query=spell_corrected_query,
                removed_stop_words=removed_stop_words,
                key_terms=key_terms,
                expanded_terms=expanded_terms,
                spell_corrections=spell_corrections,
                processing_time_ms=processing_time_ms,
                confidence_score=confidence_score
            )
            
            logger.info(
                f"Query preprocessing completed",
                processing_time_ms=processing_time_ms,
                key_terms_count=len(key_terms),
                expanded_terms_count=len(expanded_terms),
                spell_corrections_count=len(spell_corrections),
                confidence_score=confidence_score
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}", exc_info=True)
            raise
    
    def _clean_query(self, query: str) -> str:
        """Advanced query cleaning."""
        if self.config.level == PreprocessingLevel.MINIMAL:
            return query.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        if self.config.level == PreprocessingLevel.STANDARD:
            # Remove special characters but keep basic punctuation
            cleaned = re.sub(r'[^\w\s\-\.\?\!\,\:\;]', ' ', cleaned)
        elif self.config.level == PreprocessingLevel.AGGRESSIVE:
            # More aggressive cleaning
            cleaned = re.sub(r'[^\w\s\-\.]', ' ', cleaned)
            # Remove URLs
            cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
            # Remove email addresses
            cleaned = re.sub(r'\S+@\S+', '', cleaned)
        
        # Remove multiple punctuation
        cleaned = re.sub(r'[\.]{2,}', '.', cleaned)
        cleaned = re.sub(r'[\?]{2,}', '?', cleaned)
        cleaned = re.sub(r'[\!]{2,}', '!', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())
        
        return cleaned
    
    def _correct_spelling(self, query: str) -> Tuple[str, Dict[str, str]]:
        """Correct spelling errors in the query."""
        if self.config.spell_check_mode == SpellCheckMode.DISABLED:
            return query, {}
        
        corrections = {}
        words = query.split()
        corrected_words = []
        
        for word in words:
            # Check if word needs correction
            lower_word = word.lower()
            if lower_word in self._spell_dict:
                corrected_word = self._spell_dict[lower_word]
                corrections[word] = corrected_word
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        corrected_query = ' '.join(corrected_words)
        return corrected_query, corrections

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
            "'d": " would",
            "'m": " am",
            "'s": " is"
        }

        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized.strip())

        return normalized

    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize the query into words."""
        try:
            tokens = word_tokenize(query)
            # Filter out very short tokens and punctuation-only tokens
            filtered_tokens = [
                token for token in tokens
                if len(token) >= self.config.min_term_length and token.isalnum()
            ]
            return filtered_tokens
        except Exception as e:
            logger.warning(f"NLTK tokenization failed, using simple split: {e}")
            # Fallback to simple tokenization
            return [word for word in query.split() if len(word) >= self.config.min_term_length]

    def _remove_stop_words(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """Remove stop words from tokens."""
        if not self.config.enable_stop_word_removal:
            return tokens, []

        filtered_tokens = []
        removed_stop_words = []

        for token in tokens:
            if token.lower() in self._stop_words:
                removed_stop_words.append(token)
            else:
                filtered_tokens.append(token)

        return filtered_tokens, removed_stop_words

    def _stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens."""
        if not self.config.enable_stemming:
            return tokens

        try:
            stemmed_tokens = [self._stemmer.stem(token) for token in tokens]
            return stemmed_tokens
        except Exception as e:
            logger.warning(f"Stemming failed: {e}")
            return tokens

    def _extract_key_terms(self, tokens: List[str]) -> List[str]:
        """Extract key terms from filtered tokens."""
        # For now, return all filtered tokens as key terms
        # In the future, this could use TF-IDF or other techniques
        return tokens

    def _expand_terms(self, key_terms: List[str]) -> List[str]:
        """Expand key terms with synonyms and related terms."""
        if not self.config.enable_synonym_expansion:
            return []

        expanded_terms = []

        for term in key_terms:
            term_lower = term.lower()

            # Add synonyms from dictionary
            if term_lower in self._synonym_dict:
                synonyms = self._synonym_dict[term_lower]
                expanded_terms.extend(synonyms[:self.config.max_expansion_terms])

            # Add domain-specific terms
            if term_lower in self._domain_terms:
                domain_terms = self._domain_terms[term_lower]
                expanded_terms.extend(domain_terms[:self.config.max_expansion_terms])

        # Remove duplicates and limit total expansion terms
        unique_expanded = list(dict.fromkeys(expanded_terms))
        return unique_expanded[:self.config.max_expansion_terms]

    def _build_expanded_query(self, original_query: str, expanded_terms: List[str]) -> str:
        """Build expanded query with original and expanded terms."""
        if not expanded_terms:
            return original_query

        if self.config.preserve_original_terms:
            # Combine original query with expanded terms
            expanded_query = f"{original_query} {' '.join(expanded_terms)}"
        else:
            # Replace with expanded terms only
            expanded_query = ' '.join(expanded_terms)

        return expanded_query

    def _calculate_confidence(
        self,
        original_query: str,
        cleaned_query: str,
        spell_corrections: Dict[str, str],
        key_terms: List[str]
    ) -> float:
        """Calculate confidence score for the preprocessing."""
        confidence = 0.5  # Base confidence

        # Boost confidence based on query length
        if len(original_query.split()) >= 3:
            confidence += 0.2

        # Boost confidence if no major cleaning was needed
        if len(original_query) == len(cleaned_query):
            confidence += 0.1

        # Reduce confidence if many spell corrections were made
        if len(spell_corrections) > 0:
            confidence -= min(0.3, len(spell_corrections) * 0.1)

        # Boost confidence based on number of key terms
        if len(key_terms) >= 2:
            confidence += 0.1

        # Cap at 1.0
        return min(confidence, 1.0)


# Singleton instance
_preprocessor_instance: Optional[AdvancedQueryPreprocessor] = None


async def get_advanced_query_preprocessor(
    config: Optional[PreprocessingConfig] = None
) -> AdvancedQueryPreprocessor:
    """Get the advanced query preprocessor instance."""
    global _preprocessor_instance

    if _preprocessor_instance is None:
        _preprocessor_instance = AdvancedQueryPreprocessor(config)

    return _preprocessor_instance


def reset_advanced_query_preprocessor():
    """Reset the preprocessor instance (for testing)."""
    global _preprocessor_instance
    _preprocessor_instance = None
