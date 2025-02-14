"""
Sistema de cache hierárquico para o serviço de geração de texto.
"""
import logging
import json
import hashlib
import time
from typing import Optional, Any, Dict, Union
from abc import ABC, abstractmethod
import redis
from pathlib import Path
import aiofiles
import aiofiles.os
import asyncio
from functools import wraps
import msgpack
import lz4.frame
from config import CACHE_CONFIG

logger = logging.getLogger(__name__)

class CacheBackend(ABC):
    """Interface base para backends de cache."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Obtém valor do cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Define valor no cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove valor do cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Limpa todo o cache."""
        pass

class MemoryCache(CacheBackend):
    """Cache em memória (L1)."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = ttl
        self._cache: Dict[str, bytes] = {}
        self._ttls: Dict[str, float] = {}
    
    async def _cleanup(self):
        """Remove itens expirados."""
        now = time.time()
        expired = [
            k for k, t in self._ttls.items()
            if t <= now
        ]
        for k in expired:
            del self._cache[k]
            del self._ttls[k]
    
    async def get(self, key: str) -> Optional[bytes]:
        await self._cleanup()
        
        if key in self._cache:
            if time.time() <= self._ttls[key]:
                return self._cache[key]
            else:
                del self._cache[key]
                del self._ttls[key]
        return None
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        await self._cleanup()
        
        # Remove itens antigos se necessário
        if len(self._cache) >= self.max_size:
            oldest = min(self._ttls.items(), key=lambda x: x[1])[0]
            del self._cache[oldest]
            del self._ttls[oldest]
        
        self._cache[key] = value
        self._ttls[key] = time.time() + (ttl or self.default_ttl)
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            del self._ttls[key]
            return True
        return False
    
    async def clear(self) -> bool:
        self._cache.clear()
        self._ttls.clear()
        return True

class RedisCache(CacheBackend):
    """Cache Redis (L2)."""
    
    def __init__(self, url: str, password: str = "", db: int = 0):
        self.client = redis.from_url(
            url,
            password=password,
            db=db,
            decode_responses=False
        )
    
    async def get(self, key: str) -> Optional[bytes]:
        try:
            value = self.client.get(key)
            if value:
                return value
            return None
        except Exception as e:
            logger.error(f"Erro ao ler do Redis: {e}")
            return None
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        try:
            if ttl:
                return bool(self.client.setex(key, ttl, value))
            return bool(self.client.set(key, value))
        except Exception as e:
            logger.error(f"Erro ao escrever no Redis: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Erro ao deletar do Redis: {e}")
            return False
    
    async def clear(self) -> bool:
        try:
            return bool(self.client.flushdb())
        except Exception as e:
            logger.error(f"Erro ao limpar Redis: {e}")
            return False

class DiskCache(CacheBackend):
    """Cache em disco (L3)."""
    
    def __init__(self, path: Union[str, Path], max_size_gb: int = 100):
        self.path = Path(path)
        self.max_size_gb = max_size_gb
        self.path.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        """Retorna caminho do arquivo de cache."""
        # Usa 2 níveis de diretório para evitar muitos arquivos em um só diretório
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.path / h[:2] / h[2:4] / h
    
    async def _ensure_space(self, size_bytes: int):
        """Garante espaço em disco para novo item."""
        max_bytes = self.max_size_gb * 1024 * 1024 * 1024
        
        # Lista todos os arquivos por data de modificação
        files = []
        for p in self.path.rglob("*"):
            if p.is_file():
                files.append((p, await aiofiles.os.path.getmtime(p)))
        
        files.sort(key=lambda x: x[1])
        
        # Remove arquivos antigos até ter espaço suficiente
        current_size = sum(p.stat().st_size for p, _ in files)
        for file, _ in files:
            if current_size + size_bytes <= max_bytes:
                break
            size = file.stat().st_size
            await aiofiles.os.remove(file)
            current_size -= size
    
    async def get(self, key: str) -> Optional[bytes]:
        try:
            path = self._get_path(key)
            if not path.exists():
                return None
            
            # Verifica TTL
            mtime = await aiofiles.os.path.getmtime(path)
            ttl_file = path.with_suffix(".ttl")
            if ttl_file.exists():
                async with aiofiles.open(ttl_file, "r") as f:
                    ttl = float(await f.read())
                    if time.time() > ttl:
                        await aiofiles.os.remove(path)
                        await aiofiles.os.remove(ttl_file)
                        return None
            
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
                
        except Exception as e:
            logger.error(f"Erro ao ler do disco: {e}")
            return None
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        try:
            path = self._get_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Garante espaço em disco
            await self._ensure_space(len(value))
            
            # Salva valor
            async with aiofiles.open(path, "wb") as f:
                await f.write(value)
            
            # Salva TTL se fornecido
            if ttl:
                ttl_file = path.with_suffix(".ttl")
                async with aiofiles.open(ttl_file, "w") as f:
                    await f.write(str(time.time() + ttl))
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao escrever no disco: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            path = self._get_path(key)
            if path.exists():
                await aiofiles.os.remove(path)
                ttl_file = path.with_suffix(".ttl")
                if ttl_file.exists():
                    await aiofiles.os.remove(ttl_file)
                return True
            return False
        except Exception as e:
            logger.error(f"Erro ao deletar do disco: {e}")
            return False
    
    async def clear(self) -> bool:
        try:
            for p in self.path.rglob("*"):
                if p.is_file():
                    await aiofiles.os.remove(p)
            return True
        except Exception as e:
            logger.error(f"Erro ao limpar cache em disco: {e}")
            return False

class HierarchicalCache:
    """
    Cache hierárquico com três níveis:
    - L1: Memória (rápido, pequeno)
    - L2: Redis (médio, compartilhado)
    - L3: Disco (lento, grande)
    """
    
    def __init__(self):
        """Inicializa o cache hierárquico."""
        # L1 Cache
        self.l1 = MemoryCache(
            max_size=CACHE_CONFIG["l1_cache"]["max_size"],
            ttl=CACHE_CONFIG["l1_cache"]["ttl"]
        )
        
        # L2 Cache
        self.l2 = RedisCache(
            url=CACHE_CONFIG["l2_cache"]["url"],
            password=CACHE_CONFIG["l2_cache"]["password"],
            db=CACHE_CONFIG["l2_cache"]["db"]
        )
        
        # L3 Cache
        self.l3 = DiskCache(
            path=CACHE_CONFIG["l3_cache"]["path"],
            max_size_gb=CACHE_CONFIG["l3_cache"]["max_size_gb"]
        )
    
    async def get(self, key: str) -> Optional[bytes]:
        """
        Obtém valor do cache, tentando cada nível em ordem.
        Se encontrado em um nível inferior, promove para níveis superiores.
        """
        # Tenta L1
        value = await self.l1.get(key)
        if value is not None:
            return value
        
        # Tenta L2
        value = await self.l2.get(key)
        if value is not None:
            # Promove para L1
            await self.l1.set(key, value)
            return value
        
        # Tenta L3
        value = await self.l3.get(key)
        if value is not None:
            # Promove para L1 e L2
            await asyncio.gather(
                self.l1.set(key, value),
                self.l2.set(key, value)
            )
            return value
        
        return None
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Define valor em todos os níveis de cache."""
        try:
            # Comprime dados
            compressed = lz4.frame.compress(value)
            
            # Define em todos os níveis
            results = await asyncio.gather(
                self.l1.set(key, compressed, ttl),
                self.l2.set(key, compressed, ttl),
                self.l3.set(key, compressed, ttl)
            )
            
            return all(results)
            
        except Exception as e:
            logger.error(f"Erro ao definir cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Remove valor de todos os níveis de cache."""
        results = await asyncio.gather(
            self.l1.delete(key),
            self.l2.delete(key),
            self.l3.delete(key)
        )
        return any(results)
    
    async def clear(self) -> bool:
        """Limpa todos os níveis de cache."""
        results = await asyncio.gather(
            self.l1.clear(),
            self.l2.clear(),
            self.l3.clear()
        )
        return all(results)

def cache_result(ttl: Optional[int] = None):
    """
    Decorator para cache de resultados.
    
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
            params = {
                "args": args,
                "kwargs": kwargs,
                "func": func.__name__
            }
            key = hashlib.sha256(
                msgpack.packb(params)
            ).hexdigest()
            
            # Tenta obter do cache
            cached = await cache.get(key)
            if cached:
                logger.debug(f"Cache hit: {key}")
                return msgpack.unpackb(
                    lz4.frame.decompress(cached)
                )
            
            # Executa função
            result = await func(self, *args, **kwargs)
            
            # Armazena no cache
            if result is not None:
                try:
                    # Serializa e comprime
                    value = lz4.frame.compress(
                        msgpack.packb(result)
                    )
                    
                    await cache.set(key, value, ttl)
                    logger.debug(f"Cache set: {key}")
                except Exception as e:
                    logger.error(f"Erro ao definir cache: {e}")
            
            return result
            
        return wrapper
    return decorator 