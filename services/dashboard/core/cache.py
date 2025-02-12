"""
Configuração do Redis para cache.
"""

from typing import Optional, Any
import json
import aioredis
from .config import settings

# Conexão com Redis
redis = aioredis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True
)

async def set_cache(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Armazena um valor no cache.
    
    Args:
        key: Chave para armazenar o valor
        value: Valor a ser armazenado (será convertido para JSON)
        ttl: Tempo de vida em segundos (opcional)
    """
    ttl = ttl or settings.cache_ttl
    await redis.set(key, json.dumps(value), ex=ttl) 