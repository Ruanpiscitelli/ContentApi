"""
Cache utilities shared across all services.
"""

import json
from typing import Any, Optional, Union
from datetime import timedelta
import redis
import orjson
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages Redis cache operations."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        
    def get(self, key: str) -> Optional[Any]:
        """Gets a value from cache."""
        try:
            value = self.redis.get(key)
            if value:
                return orjson.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
            
    def set(
        self,
        key: str,
        value: Any,
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Sets a value in cache."""
        try:
            serialized = orjson.dumps(value)
            return bool(self.redis.set(key, serialized, ex=expire))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """Deletes a value from cache."""
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
            
    def exists(self, key: str) -> bool:
        """Checks if a key exists in cache."""
        try:
            return bool(self.redis.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
            
def cached(
    expire: Optional[Union[int, timedelta]] = None,
    key_prefix: str = ""
):
    """
    Decorator for caching function results.
    
    Args:
        expire: Cache expiration time in seconds or timedelta
        key_prefix: Prefix for cache key
        
    Example:
        @cached(expire=300, key_prefix="user")
        async def get_user(user_id: int):
            return await db.get_user(user_id)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Get cache manager from first arg (self) if available
            cache_manager = getattr(args[0], "cache_manager", None) if args else None
            if not cache_manager:
                logger.warning("No cache manager available for caching")
                return await func(*args, **kwargs)
                
            # Try to get from cache
            cached_value = cache_manager.get(cache_key)
            if cached_value is not None:
                return cached_value
                
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_manager.set(cache_key, result, expire)
            return result
            
        return wrapper
    return decorator 