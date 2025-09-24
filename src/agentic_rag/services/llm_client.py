"""
LLM Client Service

This module provides the OpenAI LLM client for reranking search results.
Includes model selection, authentication, error handling, and health monitoring.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field
import openai
from openai import AsyncOpenAI

from agentic_rag.config import get_settings

logger = structlog.get_logger(__name__)


class LLMModel(str, Enum):
    """Supported LLM models for reranking."""
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class LLMRequestType(str, Enum):
    """Types of LLM requests."""
    RERANKING = "reranking"
    SCORING = "scoring"
    EXPLANATION = "explanation"


@dataclass
class LLMUsageStats:
    """LLM usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class LLMRequest(BaseModel):
    """Request model for LLM operations."""
    
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    model: LLMModel = Field(LLMModel.GPT_4_TURBO, description="Model to use")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum tokens to generate")
    request_type: LLMRequestType = Field(LLMRequestType.RERANKING, description="Type of request")
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: str = Field(..., description="User identifier")


class LLMResponse(BaseModel):
    """Response model for LLM operations."""
    
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    usage: LLMUsageStats = Field(..., description="Token usage statistics")
    response_time_ms: int = Field(..., description="Response time in milliseconds")
    request_id: str = Field(..., description="Request identifier")


class LLMHealthStatus(BaseModel):
    """Health status for LLM service."""
    
    status: str = Field(..., description="Overall health status")
    api_accessible: bool = Field(..., description="Whether API is accessible")
    models_available: List[str] = Field(..., description="Available models")
    last_successful_request: Optional[float] = Field(None, description="Timestamp of last successful request")
    error_rate: float = Field(..., description="Recent error rate percentage")
    average_response_time_ms: float = Field(..., description="Average response time")


class LLMClientService:
    """OpenAI LLM client service for reranking operations."""
    
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        self._settings = get_settings()
        self._initialized = False
        
        # Model pricing (per 1K tokens) - approximate values
        self._model_pricing = {
            LLMModel.GPT_4_TURBO: {"input": 0.01, "output": 0.03},
            LLMModel.GPT_4: {"input": 0.03, "output": 0.06},
            LLMModel.GPT_3_5_TURBO: {"input": 0.0015, "output": 0.002}
        }
        
        # Statistics tracking
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "total_cost_usd": 0.0,
            "average_response_time_ms": 0.0,
            "requests_by_model": {},
            "requests_by_type": {}
        }
        
        # Health monitoring
        self._last_successful_request = None
        self._recent_errors = []
        self._max_error_history = 100
        
        logger.info("LLM client service initialized")
    
    async def initialize(self):
        """Initialize the OpenAI client."""
        if self._initialized:
            return
        
        try:
            # Get OpenAI API key from settings
            api_key = getattr(self._settings, 'OPENAI_API_KEY', None)
            if not api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            
            # Initialize OpenAI client
            self._client = AsyncOpenAI(api_key=api_key)
            
            # Test connection
            await self._test_connection()
            
            self._initialized = True
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def _test_connection(self):
        """Test OpenAI API connection."""
        try:
            # Simple test request
            response = await self._client.chat.completions.create(
                model=LLMModel.GPT_3_5_TURBO,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0
            )
            
            if response and response.choices:
                logger.info("OpenAI API connection test successful")
                return True
            else:
                raise Exception("Invalid response from OpenAI API")
                
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            raise
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate completion using OpenAI API.
        
        Args:
            request: LLM request with messages and configuration
            
        Returns:
            LLM response with generated content and usage stats
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = f"llm_{int(time.time() * 1000)}"
        
        try:
            self._stats["total_requests"] += 1
            self._stats["requests_by_model"][request.model] = self._stats["requests_by_model"].get(request.model, 0) + 1
            self._stats["requests_by_type"][request.request_type] = self._stats["requests_by_type"].get(request.request_type, 0) + 1
            
            logger.info(
                f"Generating LLM completion",
                model=request.model,
                request_type=request.request_type,
                tenant_id=request.tenant_id,
                request_id=request_id
            )
            
            # Make API request
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract content
            content = response.choices[0].message.content if response.choices else ""
            
            # Calculate usage and cost
            usage_stats = self._calculate_usage_stats(response.usage, request.model)
            
            # Update statistics
            self._stats["successful_requests"] += 1
            self._stats["total_tokens_used"] += usage_stats.total_tokens
            self._stats["total_cost_usd"] += usage_stats.cost_usd
            self._update_average_response_time(response_time_ms)
            self._last_successful_request = time.time()
            
            logger.info(
                f"LLM completion generated successfully",
                request_id=request_id,
                response_time_ms=response_time_ms,
                tokens_used=usage_stats.total_tokens,
                cost_usd=usage_stats.cost_usd
            )
            
            return LLMResponse(
                content=content,
                model=request.model,
                usage=usage_stats,
                response_time_ms=response_time_ms,
                request_id=request_id
            )
            
        except Exception as e:
            self._stats["failed_requests"] += 1
            self._record_error(e)
            
            logger.error(
                f"LLM completion failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    def _calculate_usage_stats(self, usage: Any, model: LLMModel) -> LLMUsageStats:
        """Calculate usage statistics and cost."""
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost
        pricing = self._model_pricing.get(model, {"input": 0.01, "output": 0.03})
        cost_usd = (
            (prompt_tokens / 1000) * pricing["input"] +
            (completion_tokens / 1000) * pricing["output"]
        )
        
        return LLMUsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd
        )
    
    def _update_average_response_time(self, response_time_ms: int):
        """Update average response time."""
        current_avg = self._stats["average_response_time_ms"]
        successful_requests = self._stats["successful_requests"]
        
        if successful_requests > 1:
            self._stats["average_response_time_ms"] = (
                (current_avg * (successful_requests - 1) + response_time_ms) / successful_requests
            )
        else:
            self._stats["average_response_time_ms"] = response_time_ms
    
    def _record_error(self, error: Exception):
        """Record error for health monitoring."""
        error_record = {
            "timestamp": time.time(),
            "error": str(error),
            "type": type(error).__name__
        }
        
        self._recent_errors.append(error_record)
        
        # Maintain error history size
        if len(self._recent_errors) > self._max_error_history:
            self._recent_errors = self._recent_errors[-self._max_error_history:]
    
    async def health_check(self) -> LLMHealthStatus:
        """Check LLM service health."""
        try:
            # Test API accessibility
            api_accessible = False
            if self._initialized:
                try:
                    await self._test_connection()
                    api_accessible = True
                except:
                    api_accessible = False
            
            # Calculate error rate (last 100 requests)
            recent_window = time.time() - 3600  # Last hour
            recent_errors = [e for e in self._recent_errors if e["timestamp"] > recent_window]
            total_recent_requests = max(self._stats["total_requests"], 1)
            error_rate = (len(recent_errors) / total_recent_requests) * 100
            
            # Determine overall status
            if api_accessible and error_rate < 5:
                status = "healthy"
            elif api_accessible and error_rate < 20:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return LLMHealthStatus(
                status=status,
                api_accessible=api_accessible,
                models_available=[model.value for model in LLMModel],
                last_successful_request=self._last_successful_request,
                error_rate=error_rate,
                average_response_time_ms=self._stats["average_response_time_ms"]
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return LLMHealthStatus(
                status="unhealthy",
                api_accessible=False,
                models_available=[],
                last_successful_request=self._last_successful_request,
                error_rate=100.0,
                average_response_time_ms=0.0
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM service statistics."""
        return self._stats.copy()


# Singleton instance
_llm_client_service: Optional[LLMClientService] = None


def get_llm_client_service() -> LLMClientService:
    """Get the singleton LLM client service instance."""
    global _llm_client_service
    if _llm_client_service is None:
        _llm_client_service = LLMClientService()
    return _llm_client_service
