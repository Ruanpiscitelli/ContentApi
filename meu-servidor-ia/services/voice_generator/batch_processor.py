"""
Processador em batch para otimização de geração de voz
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
import torch
import logging
from dataclasses import dataclass
from config import OPTIMIZATION_CONFIG, CACHE_CONFIG
import redis
import json
import hashlib
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class BatchItem:
    """Item para processamento em batch."""
    text: str
    language: str
    speaker_embedding: Optional[torch.Tensor] = None
    params: Dict[str, Any] = None
    future: asyncio.Future = None

class BatchProcessor:
    """
    Gerencia o processamento em batch de requisições TTS.
    Implementa um sistema de fila e cache de embeddings.
    """
    
    def __init__(self, model, cache_client: redis.Redis):
        self.model = model
        self.cache = cache_client
        self.batch_queue: List[BatchItem] = []
        self.processing = False
        self.batch_size = OPTIMIZATION_CONFIG["batch_processing"]["optimal_batch_size"]
        self.max_wait = OPTIMIZATION_CONFIG["batch_processing"]["max_batch_wait_time"]
        
        # Cache de embeddings
        self.embedding_cache_enabled = CACHE_CONFIG["embedding_cache"]["enabled"]
        self.embedding_cache_ttl = CACHE_CONFIG["embedding_cache"]["ttl"]
    
    async def add_to_batch(
        self,
        text: str,
        language: str = "auto",
        speaker_embedding: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> asyncio.Future:
        """
        Adiciona um item à fila de processamento em batch.
        Retorna uma Future que será completada quando o áudio for gerado.
        """
        future = asyncio.Future()
        item = BatchItem(
            text=text,
            language=language,
            speaker_embedding=speaker_embedding,
            params=params or {},
            future=future
        )
        
        self.batch_queue.append(item)
        
        # Inicia processamento se atingiu tamanho do batch ou tem itens esperando
        if len(self.batch_queue) >= self.batch_size:
            asyncio.create_task(self._process_batch())
        elif not self.processing:
            asyncio.create_task(self._delayed_process())
        
        return future
    
    async def _delayed_process(self):
        """Aguarda um tempo máximo antes de processar batch incompleto."""
        if self.processing:
            return
            
        self.processing = True
        await asyncio.sleep(self.max_wait)
        
        if self.batch_queue:
            await self._process_batch()
        
        self.processing = False
    
    async def _process_batch(self):
        """Processa um batch de requisições."""
        if not self.batch_queue:
            return
            
        try:
            # Prepara batch
            current_batch = self.batch_queue[:self.batch_size]
            self.batch_queue = self.batch_queue[self.batch_size:]
            
            # Processa em batch
            texts = [item.text for item in current_batch]
            languages = [item.language for item in current_batch]
            speaker_embeddings = [item.speaker_embedding for item in current_batch]
            
            # Gera áudios em batch
            with torch.cuda.amp.autocast():
                wavs = await asyncio.to_thread(
                    self.model.tts_batch,
                    texts=texts,
                    languages=languages,
                    speaker_embeddings=speaker_embeddings
                )
            
            # Completa futures
            for item, wav in zip(current_batch, wavs):
                if not item.future.done():
                    item.future.set_result(wav)
                    
        except Exception as e:
            logger.error(f"Erro no processamento em batch: {e}")
            # Falha todas as futures do batch em caso de erro
            for item in current_batch:
                if not item.future.done():
                    item.future.set_exception(e)
    
    def cache_embedding(self, audio_hash: str, embedding: torch.Tensor):
        """Armazena embedding no cache."""
        if not self.embedding_cache_enabled:
            return
            
        try:
            # Serializa o tensor
            embedding_bytes = embedding.cpu().numpy().tobytes()
            
            # Salva no Redis
            self.cache.setex(
                f"emb:{audio_hash}",
                self.embedding_cache_ttl,
                embedding_bytes
            )
        except Exception as e:
            logger.warning(f"Erro ao cachear embedding: {e}")
    
    def get_cached_embedding(self, audio_hash: str) -> Optional[torch.Tensor]:
        """Recupera embedding do cache."""
        if not self.embedding_cache_enabled:
            return None
            
        try:
            # Tenta recuperar do Redis
            embedding_bytes = self.cache.get(f"emb:{audio_hash}")
            if embedding_bytes:
                # Deserializa para tensor
                embedding = torch.from_numpy(
                    np.frombuffer(embedding_bytes, dtype=np.float32)
                ).reshape(-1)
                return embedding.to(self.model.device)
        except Exception as e:
            logger.warning(f"Erro ao recuperar embedding do cache: {e}")
        
        return None
    
    @staticmethod
    def get_audio_hash(audio_bytes: bytes) -> str:
        """Gera hash único para o áudio."""
        return hashlib.sha256(audio_bytes).hexdigest()
    
    def __del__(self):
        """Cleanup ao destruir o objeto."""
        # Cancela futures pendentes
        for item in self.batch_queue:
            if not item.future.done():
                item.future.cancel() 