"""
Módulo de processamento em pipeline para edição de vídeo.
Implementa processamento assíncrono, batch processing e otimizações.
"""

import asyncio
import logging
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import redis
import psutil
import torch
import json
import subprocess

from .config import (
    HARDWARE_CONFIG,
    PIPELINE_CONFIG,
    CACHE_CONFIG,
    CODEC_CONFIG,
    get_ffmpeg_options
)

logger = logging.getLogger(__name__)

@dataclass
class VideoTask:
    """Representa uma tarefa de processamento de vídeo."""
    task_id: str
    input_path: str
    output_path: str
    options: Dict[str, Any]
    status: str = "pending"
    progress: float = 0.0
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class VideoPipeline:
    """
    Pipeline otimizado para processamento de vídeo.
    Implementa processamento em batch, cache e otimizações de hardware.
    """
    
    def __init__(self):
        """Inicializa o pipeline com configurações otimizadas."""
        self.tasks: Dict[str, VideoTask] = {}
        self.queue = asyncio.Queue(maxsize=PIPELINE_CONFIG["max_queue_size"])
        self.executor = ThreadPoolExecutor(max_workers=HARDWARE_CONFIG["cpu_threads"])
        self.processing = False
        
        # Inicializa Redis para cache se habilitado
        if CACHE_CONFIG["enabled"]:
            self.redis = redis.from_url(CACHE_CONFIG["redis_url"])
        else:
            self.redis = None
            
        # Configura GPU se disponível
        self.use_gpu = HARDWARE_CONFIG["use_gpu"] and torch.cuda.is_available()
        if self.use_gpu:
            self.gpu_memory_limit = int(
                torch.cuda.get_device_properties(0).total_memory * 
                HARDWARE_CONFIG["gpu_memory_limit"]
            )
            
        # Inicia workers
        self.start_workers()
        
    def start_workers(self):
        """Inicia os workers de processamento."""
        self.processing = True
        for _ in range(HARDWARE_CONFIG["cpu_threads"]):
            asyncio.create_task(self._process_queue())
            
    async def stop_workers(self):
        """Para os workers de processamento."""
        self.processing = False
        await self.queue.join()
        self.executor.shutdown()
        
    async def add_task(self, task: VideoTask) -> str:
        """
        Adiciona uma tarefa à fila de processamento.
        
        Args:
            task: Tarefa de processamento de vídeo
            
        Returns:
            ID da tarefa
            
        Raises:
            QueueFullError: Se a fila estiver cheia
        """
        if self.queue.full():
            raise QueueFullError("Fila de processamento cheia")
            
        # Verifica cache
        if self.redis and await self._check_cache(task):
            return task.task_id
            
        self.tasks[task.task_id] = task
        await self.queue.put(task)
        return task.task_id
        
    async def get_task_status(self, task_id: str) -> Optional[VideoTask]:
        """Retorna o status de uma tarefa."""
        return self.tasks.get(task_id)
        
    async def _process_queue(self):
        """Worker para processar a fila de tarefas."""
        while self.processing:
            try:
                # Processa em batch quando possível
                batch = await self._get_batch()
                if not batch:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Processa batch
                await self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Erro no processamento: {str(e)}")
                await asyncio.sleep(1)
                
    async def _get_batch(self) -> List[VideoTask]:
        """Obtém um batch de tarefas para processamento."""
        if self.queue.empty():
            return []
            
        batch = []
        try:
            while len(batch) < PIPELINE_CONFIG["max_batch_size"]:
                task = self.queue.get_nowait()
                batch.append(task)
        except asyncio.QueueEmpty:
            pass
            
        return batch
        
    async def _process_batch(self, batch: List[VideoTask]):
        """
        Processa um batch de tarefas.
        Implementa otimizações de hardware e paralelização.
        """
        try:
            # Verifica recursos disponíveis
            await self._check_resources()
            
            # Processa cada tarefa do batch
            for task in batch:
                try:
                    task.start_time = time.time()
                    task.status = "processing"
                    
                    # Aplica otimizações baseadas no hardware
                    options = self._optimize_options(task.options)
                    
                    # Processa o vídeo
                    success = await self._process_video(task, options)
                    
                    if success:
                        task.status = "completed"
                        if self.redis:
                            await self._cache_result(task)
                    else:
                        task.status = "failed"
                        task.error = "Erro no processamento"
                        
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    logger.error(f"Erro processando tarefa {task.task_id}: {str(e)}")
                    
                finally:
                    task.end_time = time.time()
                    self.queue.task_done()
                    
        except Exception as e:
            logger.error(f"Erro processando batch: {str(e)}")
            
    async def _check_resources(self):
        """Verifica disponibilidade de recursos."""
        # Verifica CPU
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > 90:
            raise ResourceError("CPU sobrecarregada")
            
        # Verifica memória
        memory = psutil.virtual_memory()
        if memory.percent > HARDWARE_CONFIG["memory_limit"] * 100:
            raise ResourceError("Memória RAM insuficiente")
            
        # Verifica GPU
        if self.use_gpu:
            gpu_memory = torch.cuda.memory_allocated()
            if gpu_memory > self.gpu_memory_limit:
                raise ResourceError("Memória GPU insuficiente")
                
    def _optimize_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Otimiza opções baseado no hardware disponível."""
        # Aplica otimizações base
        optimized = get_ffmpeg_options(
            quality=options.get("quality", "medium"),
            use_gpu=self.use_gpu
        )
        
        # Adiciona otimizações específicas
        if self.use_gpu:
            optimized.update({
                "hwaccel": "cuda",
                "hwaccel_output_format": "cuda"
            })
            
        # Ajusta threads baseado na CPU
        optimized["threads"] = HARDWARE_CONFIG["cpu_threads"]
        
        return {**options, **optimized}
        
    async def _process_video(self, task: VideoTask, options: Dict[str, Any]) -> bool:
        """
        Processa um vídeo com as opções otimizadas.
        Implementa monitoramento de progresso e recursos.
        """
        try:
            # Constrói comando FFmpeg
            cmd = self._build_ffmpeg_command(task, options)
            
            # Executa em thread separada
            proc = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            )
            
            # Monitora progresso
            while True:
                if proc.poll() is not None:
                    break
                    
                # Atualiza progresso
                progress = self._get_progress(proc)
                task.progress = progress
                
                await asyncio.sleep(0.1)
                
            return proc.returncode == 0
            
        except Exception as e:
            logger.error(f"Erro processando vídeo: {str(e)}")
            return False
            
    async def _check_cache(self, task: VideoTask) -> bool:
        """Verifica se resultado está em cache."""
        if not self.redis:
            return False
            
        cache_key = f"video_task:{task.task_id}"
        cached = await self.redis.get(cache_key)
        
        if cached:
            cached_data = json.loads(cached)
            task.status = "completed"
            task.output_path = cached_data["output_path"]
            return True
            
        return False
        
    async def _cache_result(self, task: VideoTask):
        """Armazena resultado em cache."""
        if not self.redis:
            return
            
        cache_key = f"video_task:{task.task_id}"
        cache_data = {
            "output_path": task.output_path,
            "options": task.options
        }
        
        await self.redis.setex(
            cache_key,
            CACHE_CONFIG["ttl"],
            json.dumps(cache_data)
        )
        
    def _build_ffmpeg_command(self, task: VideoTask, options: Dict[str, Any]) -> List[str]:
        """Constrói comando FFmpeg otimizado."""
        cmd = ["ffmpeg", "-y"]
        
        # Adiciona input
        cmd.extend(["-i", task.input_path])
        
        # Adiciona opções de hardware
        if options.get("hwaccel"):
            cmd.extend(["-hwaccel", options["hwaccel"]])
            if options.get("hwaccel_output_format"):
                cmd.extend(["-hwaccel_output_format", options["hwaccel_output_format"]])
                
        # Adiciona codec de vídeo
        cmd.extend([
            "-c:v", options.get("codec", "libx264"),
            "-preset", options.get("preset", "medium"),
            "-crf", str(options.get("crf", 23))
        ])
        
        # Adiciona codec de áudio
        cmd.extend([
            "-c:a", "aac",
            "-b:a", options.get("audio_bitrate", "192k")
        ])
        
        # Adiciona outras opções
        if options.get("video_bitrate"):
            cmd.extend(["-b:v", options["video_bitrate"]])
            
        if options.get("threads"):
            cmd.extend(["-threads", str(options["threads"])])
            
        # Adiciona output
        cmd.append(task.output_path)
        
        return cmd
        
class QueueFullError(Exception):
    """Erro quando a fila está cheia."""
    pass
    
class ResourceError(Exception):
    """Erro quando recursos são insuficientes."""
    pass 