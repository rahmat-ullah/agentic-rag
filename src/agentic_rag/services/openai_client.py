"""
OpenAI Embeddings Client

This module provides a comprehensive interface to OpenAI's embeddings API
with authentication, error handling, health monitoring, and cost tracking.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field

from agentic_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    """Request for embedding generation."""
    
    texts: List[str] = Field(..., description="List of texts to embed")
    model: str = Field(default="text-embedding-3-large", description="Embedding model to use")
    dimensions: Optional[int] = Field(None, description="Number of dimensions for embedding")
    encoding_format: str = Field(default="float", description="Encoding format for embeddings")
    user: Optional[str] = Field(None, description="User identifier for tracking")


class EmbeddingResponse(BaseModel):
    """Response from embedding generation."""
    
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    processing_time: float = Field(..., description="Time taken to process request")


class EmbeddingUsage(BaseModel):
    """Usage tracking for embeddings."""
    
    total_tokens: int = Field(default=0, description="Total tokens processed")
    total_requests: int = Field(default=0, description="Total API requests made")
    total_embeddings: int = Field(default=0, description="Total embeddings generated")
    estimated_cost: float = Field(default=0.0, description="Estimated cost in USD")
    last_request_time: Optional[datetime] = Field(None, description="Last request timestamp")


class OpenAIEmbeddingsClient:
    """Async OpenAI client for embeddings with comprehensive error handling and monitoring."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[AsyncOpenAI] = None
        
        # API configuration
        self.api_key = settings.ai.openai_api_key
        self.organization = settings.ai.openai_organization
        self.embedding_model = settings.ai.openai_embedding_model
        self.embedding_dimension = settings.retrieval.embedding_dimension
        
        # Rate limiting and retry settings
        self._max_retries = 3
        self._retry_delay = 1.0
        self._rate_limit_delay = 60  # seconds to wait on rate limit
        
        # Usage tracking
        self._usage = EmbeddingUsage()
        
        # Health monitoring
        self._last_health_check = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        
        # Cost tracking (approximate rates for text-embedding-3-large)
        self._cost_per_1k_tokens = 0.00013  # USD per 1K tokens
        
        logger.info(f"OpenAI embeddings client initialized with model: {self.embedding_model}")
    
    async def start(self) -> None:
        """Initialize the OpenAI client."""
        try:
            if not self.api_key:
                raise ValueError("OpenAI API key not configured")
            
            # Initialize OpenAI client
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                max_retries=0,  # We handle retries ourselves
                timeout=60.0
            )
            
            # Test connection
            await self._test_connection()
            
            logger.info("OpenAI embeddings client started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start OpenAI embeddings client: {e}")
            raise
    
    async def stop(self) -> None:
        """Clean up OpenAI client resources."""
        try:
            if self._client:
                await self._client.close()
                self._client = None
            logger.info("OpenAI embeddings client stopped")
        except Exception as e:
            logger.error(f"Error stopping OpenAI embeddings client: {e}")
    
    async def _test_connection(self) -> None:
        """Test OpenAI API connection with a simple request."""
        if not self._client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Test with a simple embedding request
            response = await self._client.embeddings.create(
                input=["test connection"],
                model=self.embedding_model,
                dimensions=self.embedding_dimension
            )
            
            if not response.data:
                raise Exception("Empty response from OpenAI API")
            
            logger.info("OpenAI API connection test successful")
            
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError))
    )
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        dimensions: Optional[int] = None
    ) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use (default: configured model)
            dimensions: Number of dimensions (default: configured dimensions)
            
        Returns:
            EmbeddingResponse with embeddings and usage information
        """
        if not self._client:
            await self.start()
        
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        model = model or self.embedding_model
        dimensions = dimensions or self.embedding_dimension
        
        start_time = time.time()
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {model}")
            
            # Make API request
            response = await self._client.embeddings.create(
                input=texts,
                model=model,
                dimensions=dimensions,
                encoding_format="float"
            )
            
            # Extract embeddings
            embeddings = [embedding.embedding for embedding in response.data]
            
            # Track usage
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            processing_time = time.time() - start_time
            
            # Update usage tracking
            self._update_usage(usage_info["total_tokens"], len(embeddings))
            
            # Reset failure counter on success
            self._consecutive_failures = 0
            
            logger.info(f"Generated {len(embeddings)} embeddings in {processing_time:.2f}s")
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                usage=usage_info,
                request_id=getattr(response, 'id', None),
                processing_time=processing_time
            )
            
        except openai.RateLimitError as e:
            self._consecutive_failures += 1
            logger.warning(f"Rate limit exceeded: {e}")
            await asyncio.sleep(self._rate_limit_delay)
            raise
            
        except openai.AuthenticationError as e:
            self._consecutive_failures += 1
            logger.error(f"Authentication failed: {e}")
            raise
            
        except openai.APIError as e:
            self._consecutive_failures += 1
            logger.error(f"OpenAI API error: {e}")
            raise
            
        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"Unexpected error generating embeddings: {e}")
            raise
    
    def _update_usage(self, tokens: int, embeddings_count: int) -> None:
        """Update usage tracking."""
        self._usage.total_tokens += tokens
        self._usage.total_requests += 1
        self._usage.total_embeddings += embeddings_count
        self._usage.estimated_cost += (tokens / 1000) * self._cost_per_1k_tokens
        self._usage.last_request_time = datetime.utcnow()
    
    async def generate_single_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        dimensions: Optional[int] = None
    ) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            dimensions: Number of dimensions
            
        Returns:
            Embedding vector as list of floats
        """
        response = await self.generate_embeddings([text], model, dimensions)
        return response.embeddings[0]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on OpenAI client."""
        try:
            if not self._client:
                return {
                    "status": "unhealthy",
                    "error": "Client not initialized"
                }
            
            # Check if we have too many consecutive failures
            if self._consecutive_failures >= self._max_consecutive_failures:
                return {
                    "status": "unhealthy",
                    "error": f"Too many consecutive failures: {self._consecutive_failures}"
                }
            
            # Test with a simple request
            start_time = time.time()
            await self._test_connection()
            response_time = time.time() - start_time
            
            self._last_health_check = time.time()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "model": self.embedding_model,
                "dimensions": self.embedding_dimension,
                "consecutive_failures": self._consecutive_failures,
                "usage": self._usage.model_dump(),
                "timestamp": self._last_health_check
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "consecutive_failures": self._consecutive_failures,
                "timestamp": time.time()
            }
    
    def get_usage_stats(self) -> EmbeddingUsage:
        """Get current usage statistics."""
        return self._usage.model_copy()
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._usage = EmbeddingUsage()
        logger.info("Usage statistics reset")
    
    def estimate_cost(self, token_count: int) -> float:
        """Estimate cost for a given number of tokens."""
        return (token_count / 1000) * self._cost_per_1k_tokens
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "model": self.embedding_model,
            "dimensions": self.embedding_dimension,
            "cost_per_1k_tokens": self._cost_per_1k_tokens,
            "max_input_tokens": 8191,  # For text-embedding-3-large
            "max_batch_size": 2048  # OpenAI batch limit
        }


# Global OpenAI client instance
_openai_client: Optional[OpenAIEmbeddingsClient] = None


async def get_openai_client() -> OpenAIEmbeddingsClient:
    """Get or create the global OpenAI embeddings client instance."""
    global _openai_client
    
    if _openai_client is None:
        settings = get_settings()
        _openai_client = OpenAIEmbeddingsClient(settings)
        await _openai_client.start()
    
    return _openai_client


async def close_openai_client() -> None:
    """Close the global OpenAI embeddings client instance."""
    global _openai_client
    
    if _openai_client:
        await _openai_client.stop()
        _openai_client = None
