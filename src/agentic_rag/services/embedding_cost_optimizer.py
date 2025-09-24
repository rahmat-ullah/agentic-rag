"""
Embedding Cost Optimization and Tracking

This module provides comprehensive cost optimization, usage tracking,
budget management, and caching for embedding operations.
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import redis.asyncio as redis
from pydantic import BaseModel, Field

from agentic_rag.config import get_settings

logger = logging.getLogger(__name__)


class CostPeriod(str, Enum):
    """Time periods for cost tracking."""
    
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AlertLevel(str, Enum):
    """Alert levels for budget monitoring."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class UsageRecord(BaseModel):
    """Record of API usage."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Usage timestamp")
    tenant_id: str = Field(..., description="Tenant identifier")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    tokens_used: int = Field(..., description="Number of tokens consumed")
    embeddings_generated: int = Field(..., description="Number of embeddings generated")
    cost: float = Field(..., description="Cost in USD")
    model: str = Field(..., description="Model used")
    cache_hit: bool = Field(default=False, description="Whether result was cached")


class BudgetAlert(BaseModel):
    """Budget alert notification."""
    
    alert_id: str = Field(..., description="Alert identifier")
    level: AlertLevel = Field(..., description="Alert severity level")
    tenant_id: str = Field(..., description="Tenant identifier")
    period: CostPeriod = Field(..., description="Budget period")
    current_spend: float = Field(..., description="Current spending")
    budget_limit: float = Field(..., description="Budget limit")
    percentage_used: float = Field(..., description="Percentage of budget used")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Alert timestamp")


class CostSummary(BaseModel):
    """Cost summary for a period."""
    
    tenant_id: str = Field(..., description="Tenant identifier")
    period: CostPeriod = Field(..., description="Time period")
    start_time: datetime = Field(..., description="Period start time")
    end_time: datetime = Field(..., description="Period end time")
    total_cost: float = Field(..., description="Total cost in USD")
    total_tokens: int = Field(..., description="Total tokens used")
    total_embeddings: int = Field(..., description="Total embeddings generated")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    cost_per_embedding: float = Field(..., description="Average cost per embedding")
    usage_records: int = Field(..., description="Number of usage records")


class EmbeddingCache:
    """Redis-based cache for embeddings with deduplication."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.cache_prefix = "embedding_cache:"
        self.ttl = 86400 * 7  # 7 days TTL
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("Embedding cache initialized")
    
    async def _ensure_redis(self) -> redis.Redis:
        """Ensure Redis client is available."""
        if not self.redis_client:
            settings = get_settings()
            self.redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password,
                db=settings.redis.db,
                decode_responses=False  # We store binary data
            )
        return self.redis_client
    
    def _generate_cache_key(self, text: str, model: str, tenant_id: str) -> str:
        """Generate cache key for text embedding."""
        # Create hash of text content for deduplication
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"{self.cache_prefix}{tenant_id}:{model}:{text_hash}"
    
    async def get_embedding(self, text: str, model: str, tenant_id: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        try:
            redis_client = await self._ensure_redis()
            cache_key = self._generate_cache_key(text, model, tenant_id)
            
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                # Deserialize embedding
                import pickle
                embedding = pickle.loads(cached_data)
                self.cache_hits += 1
                logger.debug(f"Cache hit for text hash: {cache_key}")
                return embedding
            
            self.cache_misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_misses += 1
            return None
    
    async def store_embedding(
        self,
        text: str,
        model: str,
        tenant_id: str,
        embedding: List[float]
    ) -> bool:
        """Store embedding in cache."""
        try:
            redis_client = await self._ensure_redis()
            cache_key = self._generate_cache_key(text, model, tenant_id)
            
            # Serialize embedding
            import pickle
            cached_data = pickle.dumps(embedding)
            
            await redis_client.setex(cache_key, self.ttl, cached_data)
            logger.debug(f"Cached embedding for text hash: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache store error: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        try:
            redis_client = await self._ensure_redis()
            info = await redis_client.info()
            memory_usage = info.get('used_memory_human', 'unknown')
        except Exception:
            memory_usage = 'unknown'
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "memory_usage": memory_usage,
            "ttl": self.ttl
        }
    
    async def clear_cache(self, tenant_id: Optional[str] = None) -> int:
        """Clear cache entries."""
        try:
            redis_client = await self._ensure_redis()
            
            if tenant_id:
                # Clear only for specific tenant
                pattern = f"{self.cache_prefix}{tenant_id}:*"
            else:
                # Clear all embedding cache
                pattern = f"{self.cache_prefix}*"
            
            keys = await redis_client.keys(pattern)
            if keys:
                deleted = await redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0


class CostTracker:
    """Cost tracking and budget management."""
    
    def __init__(self):
        self.usage_history: List[UsageRecord] = []
        self.budget_limits: Dict[str, Dict[CostPeriod, float]] = defaultdict(dict)
        self.alerts: List[BudgetAlert] = []
        
        # Cost rates (USD per 1K tokens)
        self.cost_rates = {
            "text-embedding-3-large": 0.00013,
            "text-embedding-3-small": 0.00002,
            "text-embedding-ada-002": 0.0001
        }
        
        logger.info("Cost tracker initialized")
    
    def set_budget_limit(self, tenant_id: str, period: CostPeriod, limit: float) -> None:
        """Set budget limit for tenant and period."""
        self.budget_limits[tenant_id][period] = limit
        logger.info(f"Set budget limit for {tenant_id} ({period.value}): ${limit:.2f}")
    
    def get_budget_limit(self, tenant_id: str, period: CostPeriod) -> Optional[float]:
        """Get budget limit for tenant and period."""
        return self.budget_limits.get(tenant_id, {}).get(period)
    
    async def record_usage(
        self,
        tenant_id: str,
        tokens_used: int,
        embeddings_generated: int,
        model: str,
        batch_id: Optional[str] = None,
        cache_hit: bool = False
    ) -> UsageRecord:
        """Record API usage and calculate cost."""
        
        # Calculate cost
        cost_per_1k = self.cost_rates.get(model, 0.0001)  # Default rate
        cost = (tokens_used / 1000) * cost_per_1k
        
        # Create usage record
        record = UsageRecord(
            tenant_id=tenant_id,
            batch_id=batch_id,
            tokens_used=tokens_used,
            embeddings_generated=embeddings_generated,
            cost=cost,
            model=model,
            cache_hit=cache_hit
        )
        
        self.usage_history.append(record)
        
        # Check budget limits
        await self._check_budget_limits(tenant_id, cost)
        
        logger.debug(f"Recorded usage: {tokens_used} tokens, ${cost:.6f} for {tenant_id}")
        
        return record
    
    async def _check_budget_limits(self, tenant_id: str, new_cost: float) -> None:
        """Check if usage exceeds budget limits."""
        tenant_limits = self.budget_limits.get(tenant_id, {})
        
        for period, limit in tenant_limits.items():
            current_spend = await self.get_cost_summary(tenant_id, period)
            total_spend = current_spend.total_cost + new_cost
            percentage_used = (total_spend / limit) * 100
            
            # Generate alerts based on percentage
            alert_level = None
            if percentage_used >= 100:
                alert_level = AlertLevel.EMERGENCY
            elif percentage_used >= 90:
                alert_level = AlertLevel.CRITICAL
            elif percentage_used >= 75:
                alert_level = AlertLevel.WARNING
            elif percentage_used >= 50:
                alert_level = AlertLevel.INFO
            
            if alert_level:
                await self._create_budget_alert(
                    tenant_id, period, total_spend, limit, percentage_used, alert_level
                )
    
    async def _create_budget_alert(
        self,
        tenant_id: str,
        period: CostPeriod,
        current_spend: float,
        budget_limit: float,
        percentage_used: float,
        level: AlertLevel
    ) -> None:
        """Create budget alert."""
        
        alert_id = f"{tenant_id}_{period.value}_{int(time.time())}"
        
        messages = {
            AlertLevel.INFO: f"Budget usage at {percentage_used:.1f}% for {period.value} period",
            AlertLevel.WARNING: f"Budget usage at {percentage_used:.1f}% - approaching limit",
            AlertLevel.CRITICAL: f"Budget usage at {percentage_used:.1f}% - near limit",
            AlertLevel.EMERGENCY: f"Budget exceeded! {percentage_used:.1f}% of limit used"
        }
        
        alert = BudgetAlert(
            alert_id=alert_id,
            level=level,
            tenant_id=tenant_id,
            period=period,
            current_spend=current_spend,
            budget_limit=budget_limit,
            percentage_used=percentage_used,
            message=messages[level]
        )
        
        self.alerts.append(alert)
        
        logger.warning(f"Budget alert ({level.value}): {alert.message}")
    
    async def get_cost_summary(self, tenant_id: str, period: CostPeriod) -> CostSummary:
        """Get cost summary for tenant and period."""
        
        # Calculate period boundaries
        now = datetime.utcnow()
        if period == CostPeriod.HOURLY:
            start_time = now.replace(minute=0, second=0, microsecond=0)
        elif period == CostPeriod.DAILY:
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == CostPeriod.WEEKLY:
            days_since_monday = now.weekday()
            start_time = (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif period == CostPeriod.MONTHLY:
            start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start_time = now - timedelta(days=1)
        
        # Filter records for tenant and period
        period_records = [
            record for record in self.usage_history
            if record.tenant_id == tenant_id and record.timestamp >= start_time
        ]
        
        # Calculate summary
        total_cost = sum(record.cost for record in period_records)
        total_tokens = sum(record.tokens_used for record in period_records)
        total_embeddings = sum(record.embeddings_generated for record in period_records)
        cache_hits = sum(1 for record in period_records if record.cache_hit)
        cache_hit_rate = (cache_hits / len(period_records) * 100) if period_records else 0
        cost_per_embedding = (total_cost / total_embeddings) if total_embeddings > 0 else 0
        
        return CostSummary(
            tenant_id=tenant_id,
            period=period,
            start_time=start_time,
            end_time=now,
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_embeddings=total_embeddings,
            cache_hit_rate=cache_hit_rate,
            cost_per_embedding=cost_per_embedding,
            usage_records=len(period_records)
        )
    
    def get_recent_alerts(self, tenant_id: Optional[str] = None, limit: int = 10) -> List[BudgetAlert]:
        """Get recent budget alerts."""
        alerts = self.alerts
        
        if tenant_id:
            alerts = [alert for alert in alerts if alert.tenant_id == tenant_id]
        
        # Sort by timestamp (most recent first) and limit
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for given tokens and model."""
        cost_per_1k = self.cost_rates.get(model, 0.0001)
        return (tokens / 1000) * cost_per_1k
    
    def get_optimization_recommendations(self, tenant_id: str) -> List[str]:
        """Get cost optimization recommendations."""
        recommendations = []
        
        # Analyze recent usage
        recent_records = [
            record for record in self.usage_history[-1000:]  # Last 1000 records
            if record.tenant_id == tenant_id
        ]
        
        if not recent_records:
            return ["No usage data available for analysis"]
        
        # Check cache hit rate
        cache_hits = sum(1 for record in recent_records if record.cache_hit)
        cache_hit_rate = (cache_hits / len(recent_records)) * 100
        
        if cache_hit_rate < 20:
            recommendations.append("Low cache hit rate - consider implementing better caching strategies")
        
        # Check model usage
        model_usage = defaultdict(int)
        for record in recent_records:
            model_usage[record.model] += record.tokens_used
        
        if "text-embedding-3-large" in model_usage and "text-embedding-3-small" not in model_usage:
            recommendations.append("Consider using text-embedding-3-small for less critical use cases")
        
        # Check batch efficiency
        batch_sizes = []
        for record in recent_records:
            if record.batch_id:
                batch_sizes.append(record.embeddings_generated)
        
        if batch_sizes and np.mean(batch_sizes) < 10:
            recommendations.append("Small batch sizes detected - consider batching requests for better efficiency")
        
        return recommendations or ["Usage patterns look optimized"]


class EmbeddingCostOptimizer:
    """Main cost optimization and tracking service."""
    
    def __init__(self):
        self.cache = EmbeddingCache()
        self.cost_tracker = CostTracker()
        
        logger.info("Embedding cost optimizer initialized")
    
    async def optimize_embedding_request(
        self,
        texts: List[str],
        model: str,
        tenant_id: str,
        batch_id: Optional[str] = None
    ) -> Tuple[List[List[float]], List[bool], float]:
        """
        Optimize embedding request with caching and cost tracking.
        
        Returns:
            Tuple of (embeddings, cache_hits, total_cost)
        """
        embeddings = []
        cache_hits = []
        total_tokens = 0
        
        # Check cache for each text
        for text in texts:
            cached_embedding = await self.cache.get_embedding(text, model, tenant_id)
            if cached_embedding:
                embeddings.append(cached_embedding)
                cache_hits.append(True)
            else:
                # Would need to generate embedding here
                # For now, return placeholder
                embeddings.append([0.0] * 1024)  # Placeholder
                cache_hits.append(False)
                
                # Estimate tokens (rough approximation)
                tokens = len(text.split()) * 1.3  # Rough token estimation
                total_tokens += int(tokens)
                
                # Store in cache
                await self.cache.store_embedding(text, model, tenant_id, embeddings[-1])
        
        # Record usage
        cache_hit_count = sum(cache_hits)
        embeddings_generated = len(embeddings) - cache_hit_count
        
        if embeddings_generated > 0:
            usage_record = await self.cost_tracker.record_usage(
                tenant_id=tenant_id,
                tokens_used=total_tokens,
                embeddings_generated=embeddings_generated,
                model=model,
                batch_id=batch_id,
                cache_hit=False
            )
            total_cost = usage_record.cost
        else:
            total_cost = 0.0
        
        return embeddings, cache_hits, total_cost
    
    async def get_cost_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive cost dashboard for tenant."""
        
        # Get cost summaries for different periods
        daily_summary = await self.cost_tracker.get_cost_summary(tenant_id, CostPeriod.DAILY)
        weekly_summary = await self.cost_tracker.get_cost_summary(tenant_id, CostPeriod.WEEKLY)
        monthly_summary = await self.cost_tracker.get_cost_summary(tenant_id, CostPeriod.MONTHLY)
        
        # Get cache statistics
        cache_stats = await self.cache.get_cache_stats()
        
        # Get recent alerts
        recent_alerts = self.cost_tracker.get_recent_alerts(tenant_id, limit=5)
        
        # Get optimization recommendations
        recommendations = self.cost_tracker.get_optimization_recommendations(tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "cost_summaries": {
                "daily": daily_summary.model_dump(),
                "weekly": weekly_summary.model_dump(),
                "monthly": monthly_summary.model_dump()
            },
            "cache_statistics": cache_stats,
            "recent_alerts": [alert.model_dump() for alert in recent_alerts],
            "optimization_recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cost optimizer."""
        try:
            cache_stats = await self.cache.get_cache_stats()
            
            return {
                "status": "healthy",
                "cache_status": "operational" if cache_stats["total_requests"] >= 0 else "unknown",
                "cache_hit_rate": cache_stats["hit_rate"],
                "total_usage_records": len(self.cost_tracker.usage_history),
                "total_alerts": len(self.cost_tracker.alerts),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Cost optimizer health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Global cost optimizer instance
_cost_optimizer: Optional[EmbeddingCostOptimizer] = None


async def get_cost_optimizer() -> EmbeddingCostOptimizer:
    """Get or create the global cost optimizer instance."""
    global _cost_optimizer
    
    if _cost_optimizer is None:
        _cost_optimizer = EmbeddingCostOptimizer()
    
    return _cost_optimizer


async def close_cost_optimizer() -> None:
    """Close the global cost optimizer instance."""
    global _cost_optimizer
    
    if _cost_optimizer:
        _cost_optimizer = None
