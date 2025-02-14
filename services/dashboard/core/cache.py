"""
Configuração do Redis para cache.
"""

from typing import Optional, Any
import json
import aioredis
from .config import get_settings

settings = get_settings()

# Conexão com Redis
redis = aioredis.from_url(
    settings.REDIS_URL,
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
    ttl = ttl or settings.CACHE_TTL
    await redis.set(key, json.dumps(value), ex=ttl)

async def get_cache(key: str) -> Optional[Any]:
    """
    Recupera um valor do cache.
    
    Args:
        key: Chave do valor armazenado
        
    Returns:
        Valor armazenado ou None se não encontrado
    """
    value = await redis.get(key)
    if value:
        return json.loads(value)
    return None

async def delete_cache(key: str) -> None:
    """
    Remove um valor do cache.
    
    Args:
        key: Chave do valor a ser removido
    """
    await redis.delete(key)

async def clear_cache() -> None:
    """
    Limpa todo o cache.
    """
    await redis.flushall() 