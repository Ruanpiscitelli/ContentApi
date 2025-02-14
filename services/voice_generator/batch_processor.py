"""
Processador em batch para geração de voz.
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import torch
from config import OPTIMIZATION_CONFIG
from shared.gpu_utils import estimate_max_batch_size

logger = logging.getLogger(__name__)

@dataclass
class BatchItem:
    """Item para processamento em batch."""
    text: str
    language: str
    speaker_embedding: Optional[torch.Tensor]
    params: Dict[str, Any]
    future: asyncio.Future

class BatchProcessor:
    """
    Gerencia o processamento em batch de requisições TTS.
    Implementa um sistema de fila e cache de embeddings.
    """
    
    def __init__(self, backend_manager, cache_client):
        """
        Inicializa o processador em batch.
        
        Args:
            backend_manager: Gerenciador de backends de voz
            cache_client: Cliente de cache Redis
        """
        self.backend_manager = backend_manager
        self.cache = cache_client
        self.batch_queue: List[BatchItem] = []
        self.processing = False
        
        # Configurar tamanho do batch
        model_size_mb = 512  # Tamanho estimado do modelo em MB
        self.batch_size = estimate_max_batch_size(
            model_size_mb,
            safety_factor=OPTIMIZATION_CONFIG.get("batch_safety_factor", 0.8)
        )
        self.max_wait = OPTIMIZATION_CONFIG["batch_processing"]["max_batch_wait_time"]
    
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
        """Aguarda um tempo antes de processar o batch, mesmo que incompleto."""
        if self.processing:
            return
            
        self.processing = True
        try:
            await asyncio.sleep(self.max_wait)
            if self.batch_queue:
                await self._process_batch()
        finally:
            self.processing = False
    
    async def _process_batch(self):
        """Processa os itens em batch."""
        if not self.batch_queue:
            return
            
        # Pega itens do batch atual
        batch = self.batch_queue[:self.batch_size]
        self.batch_queue = self.batch_queue[self.batch_size:]
        
        try:
            # Processa cada item do batch
            for item in batch:
                try:
                    # Tenta gerar áudio
                    audio = await self.backend_manager.generate(
                        text=item.text,
                        language=item.language,
                        speaker_embedding=item.speaker_embedding,
                        **item.params
                    )
                    
                    # Define resultado na Future
                    if not item.future.done():
                        item.future.set_result(audio)
                        
                except Exception as e:
                    logger.error(f"Erro ao processar item do batch: {e}")
                    if not item.future.done():
                        item.future.set_exception(e)
                        
        except Exception as e:
            logger.error(f"Erro ao processar batch: {e}")
            # Em caso de erro, falha todas as Futures não completadas
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)
        
        finally:
            # Se ainda tem itens na fila, agenda próximo processamento
            if self.batch_queue:
                asyncio.create_task(self._process_batch()) 