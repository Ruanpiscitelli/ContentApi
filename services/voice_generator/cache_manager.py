"""
Gerenciador de cache para o serviço de voz.
"""
import logging
import json
import hashlib
import time
from typing import Optional, Any, Dict, Union
from functools import wraps
import redis
from config import REDIS_CONFIG, CACHE_CONFIG
from exceptions import CacheError

logger = logging.getLogger(__name__)

class VoiceCache:
    """
    Gerenciador de cache para o serviço de voz.
    Implementa cache em memória e Redis com fallback.
    """
    
    def __init__(self):
        """Inicializa o gerenciador de cache."""
        self.redis_client = None
        if CACHE_CONFIG["enable_cache"]:
            try:
                self.redis_client = redis.from_url(
                    REDIS_CONFIG["url"],
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info("Cache Redis inicializado com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao conectar ao Redis: {e}")
        
        # Cache em memória como fallback
        self._local_cache = {}
        self._local_cache_ttl = {}
    
    def _get_cache_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """
        Gera chave de cache única.
        
        Args:
            prefix: Prefixo da chave
            data: Dados para gerar hash
            
        Returns:
            str: Chave de cache
        """
        # Ordena as chaves para garantir consistência
        sorted_data = {
            k: v for k, v in sorted(data.items())
            if v is not None  # Ignora valores None
        }
        
        # Gera hash dos dados
        data_hash = hashlib.sha256(
            json.dumps(sorted_data).encode()
        ).hexdigest()
        
        return f"{REDIS_CONFIG['prefix']}{prefix}:{data_hash}"
    
    def _get_ttl(self, key_type: str) -> int:
        """Retorna TTL para o tipo de chave."""
        if key_type == "embedding":
            return REDIS_CONFIG["policies"]["embeddings"]["ttl"]
        return REDIS_CONFIG["policies"]["audio"]["ttl"]
    
    async def get(self, key: str) -> Optional[bytes]:
        """
        Obtém valor do cache.
        
        Args:
            key: Chave do cache
            
        Returns:
            Optional[bytes]: Valor do cache ou None se não encontrado
        """
        try:
            # Tenta Redis primeiro
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    logger.debug(f"Cache hit (Redis): {key}")
                    return value
            
            # Fallback para cache local
            if key in self._local_cache:
                # Verifica TTL
                if time.time() < self._local_cache_ttl[key]:
                    logger.debug(f"Cache hit (local): {key}")
                    return self._local_cache[key]
                else:
                    # Remove item expirado
                    del self._local_cache[key]
                    del self._local_cache_ttl[key]
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao acessar cache: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Union[bytes, str],
        ttl: Optional[int] = None,
        key_type: str = "audio"
    ) -> bool:
        """
        Define valor no cache.
        
        Args:
            key: Chave do cache
            value: Valor a armazenar
            ttl: Tempo de vida em segundos
            key_type: Tipo da chave (audio ou embedding)
            
        Returns:
            bool: True se sucesso
        """
        if not ttl:
            ttl = self._get_ttl(key_type)
            
        try:
            # Tenta Redis primeiro
            if self.redis_client:
                self.redis_client.setex(key, ttl, value)
                logger.debug(f"Cache set (Redis): {key}")
                return True
            
            # Fallback para cache local
            self._local_cache[key] = value
            self._local_cache_ttl[key] = time.time() + ttl
            logger.debug(f"Cache set (local): {key}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao definir cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Remove item do cache.
        
        Args:
            key: Chave do cache
            
        Returns:
            bool: True se sucesso
        """
        try:
            # Remove do Redis
            if self.redis_client:
                self.redis_client.delete(key)
            
            # Remove do cache local
            if key in self._local_cache:
                del self._local_cache[key]
                del self._local_cache_ttl[key]
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao remover do cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """
        Limpa todo o cache.
        
        Returns:
            bool: True se sucesso
        """
        try:
            # Limpa Redis
            if self.redis_client:
                self.redis_client.flushdb()
            
            # Limpa cache local
            self._local_cache.clear()
            self._local_cache_ttl.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")
            return False
    
    def cache_result(
        self,
        params: Dict[str, Any],
        result: bytes,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Armazena resultado de geração de voz no cache.
        
        Args:
            params: Parâmetros da requisição
            result: Áudio gerado em bytes
            ttl: Tempo de vida em segundos
            
        Returns:
            bool: True se sucesso
        """
        key = self._get_cache_key("audio", params)
        return self.set(key, result, ttl, "audio")
    
    def cache_embedding(
        self,
        audio_hash: str,
        embedding: bytes,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Armazena embedding de voz no cache.
        
        Args:
            audio_hash: Hash do áudio de referência
            embedding: Embedding em bytes
            ttl: Tempo de vida em segundos
            
        Returns:
            bool: True se sucesso
        """
        key = f"{REDIS_CONFIG['prefix']}embedding:{audio_hash}"
        return self.set(key, embedding, ttl, "embedding")
    
    def get_cached_result(
        self,
        params: Dict[str, Any]
    ) -> Optional[bytes]:
        """
        Obtém resultado do cache.
        
        Args:
            params: Parâmetros da requisição
            
        Returns:
            Optional[bytes]: Áudio em bytes ou None se não encontrado
        """
        key = self._get_cache_key("audio", params)
        return self.get(key)
    
    def get_cached_embedding(
        self,
        audio_hash: str
    ) -> Optional[bytes]:
        """
        Obtém embedding do cache.
        
        Args:
            audio_hash: Hash do áudio de referência
            
        Returns:
            Optional[bytes]: Embedding em bytes ou None se não encontrado
        """
        key = f"{REDIS_CONFIG['prefix']}embedding:{audio_hash}"
        return self.get(key)

def cache_voice(ttl: Optional[int] = None):
    """
    Decorator para cache de resultados de geração de voz.
    
    Args:
        ttl: Tempo de vida em segundos
        
    Returns:
        Callable: Função decorada
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Obtém instância do cache
            cache = getattr(self, "cache", None)
            if not cache:
                return await func(self, *args, **kwargs)
            
            # Gera chave de cache
            params = {**kwargs}
            if args:
                params["args"] = args
            
            # Tenta obter do cache
            cached = await cache.get_cached_result(params)
            if cached:
                logger.info("Usando resultado do cache")
                return cached
            
            # Gera resultado
            result = await func(self, *args, **kwargs)
            
            # Armazena no cache
            await cache.cache_result(params, result, ttl)
            
            return result
        return wrapper
    return decorator

def cache_embedding(ttl: Optional[int] = None):
    """
    Decorator para cache de embeddings.
    
    Args:
        ttl: Tempo de vida em segundos
        
    Returns:
        Callable: Função decorada
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, audio_bytes: bytes, *args, **kwargs):
            # Obtém instância do cache
            cache = getattr(self, "cache", None)
            if not cache:
                return await func(self, audio_bytes, *args, **kwargs)
            
            # Gera hash do áudio
            audio_hash = hashlib.sha256(audio_bytes).hexdigest()
            
            # Tenta obter do cache
            cached = await cache.get_cached_embedding(audio_hash)
            if cached:
                logger.info("Usando embedding do cache")
                return cached
            
            # Gera embedding
            embedding = await func(self, audio_bytes, *args, **kwargs)
            
            # Armazena no cache
            await cache.cache_embedding(audio_hash, embedding, ttl)
            
            return embedding
        return wrapper
    return decorator 