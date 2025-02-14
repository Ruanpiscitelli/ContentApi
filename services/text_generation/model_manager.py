"""
Gerenciamento de modelos para o serviço de geração de texto.
"""
import os
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

from config import MODEL_CONFIG, VLLM_CONFIG
from metrics import (
    MODEL_INFO,
    GPU_MEMORY_USED,
    GPU_UTILIZATION,
    GPU_TEMPERATURE,
    ERROR_COUNTER
)
from resource_manager import ResourceManager

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Informações sobre modelo carregado."""
    name: str
    path: str
    engine: AsyncLLMEngine
    loaded_at: float
    last_used: float
    total_requests: int
    total_tokens: int
    avg_latency: float

class ModelManager:
    """Gerencia modelos de linguagem."""
    
    def __init__(self, resource_manager: ResourceManager):
        """
        Inicializa o gerenciador de modelos.
        
        Args:
            resource_manager: Gerenciador de recursos
        """
        self.resource_manager = resource_manager
        self.models: Dict[str, ModelInfo] = {}
        self.loading_models: Dict[str, asyncio.Event] = {}
        self._cleanup_task = None
    
    def start(self):
        """Inicia o gerenciador."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Loop de limpeza de modelos não utilizados."""
        while True:
            try:
                current_time = time.time()
                
                for name, info in list(self.models.items()):
                    # Remove modelos não utilizados por muito tempo
                    if (
                        current_time - info.last_used >
                        MODEL_CONFIG["unload_after_seconds"]
                    ):
                        await self.unload_model(name)
                
                await asyncio.sleep(300)  # Executa a cada 5 minutos
                
            except Exception as e:
                logger.error(f"Erro na limpeza de modelos: {e}")
                await asyncio.sleep(300)
    
    async def load_model(self, name: str) -> ModelInfo:
        """
        Carrega um modelo.
        
        Args:
            name: Nome do modelo
            
        Returns:
            Informações do modelo
            
        Raises:
            ValueError: Se o modelo não estiver configurado
        """
        if name not in MODEL_CONFIG["models"]:
            raise ValueError(f"Modelo não configurado: {name}")
        
        # Se já está carregado, atualiza timestamp e retorna
        if name in self.models:
            info = self.models[name]
            info.last_used = time.time()
            return info
        
        # Se está carregando, aguarda
        if name in self.loading_models:
            await self.loading_models[name].wait()
            return self.models[name]
        
        # Marca como carregando
        self.loading_models[name] = asyncio.Event()
        
        try:
            # Configura argumentos do engine
            model_config = MODEL_CONFIG["models"][name]
            engine_args = AsyncEngineArgs(
                model=model_config["path"],
                download_dir=MODEL_CONFIG["download_dir"],
                tensor_parallel_size=VLLM_CONFIG["tensor_parallel_size"],
                gpu_memory_utilization=VLLM_CONFIG["gpu_memory_utilization"],
                max_num_batched_tokens=VLLM_CONFIG["max_num_batched_tokens"],
                max_num_seqs=VLLM_CONFIG["max_num_seqs"],
                quantization=model_config.get("quantization", None),
                dtype=model_config.get("dtype", "auto"),
                trust_remote_code=True
            )
            
            # Cria engine
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Registra informações
            info = ModelInfo(
                name=name,
                path=model_config["path"],
                engine=engine,
                loaded_at=time.time(),
                last_used=time.time(),
                total_requests=0,
                total_tokens=0,
                avg_latency=0.0
            )
            
            self.models[name] = info
            
            # Atualiza métricas
            MODEL_INFO.info({
                "name": name,
                "path": model_config["path"],
                "loaded_at": str(info.loaded_at),
                "quantization": model_config.get("quantization", "none"),
                "dtype": model_config.get("dtype", "auto"),
                "tensor_parallel_size": VLLM_CONFIG["tensor_parallel_size"]
            })
            
            logger.info(f"Modelo {name} carregado com sucesso")
            return info
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {name}: {e}")
            ERROR_COUNTER.labels(
                type="model_load_error",
                model=name
            ).inc()
            raise
            
        finally:
            # Marca como carregado
            event = self.loading_models.pop(name)
            event.set()
    
    async def unload_model(self, name: str):
        """
        Descarrega um modelo.
        
        Args:
            name: Nome do modelo
        """
        if name in self.models:
            try:
                info = self.models[name]
                
                # Remove do dicionário primeiro para evitar uso
                del self.models[name]
                
                # Descarrega engine
                del info.engine
                
                logger.info(f"Modelo {name} descarregado com sucesso")
                
            except Exception as e:
                logger.error(f"Erro ao descarregar modelo {name}: {e}")
                ERROR_COUNTER.labels(
                    type="model_unload_error",
                    model=name
                ).inc()
    
    async def generate(
        self,
        model: str,
        prompt: str,
        sampling_params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Gera texto usando um modelo.
        
        Args:
            model: Nome do modelo
            prompt: Prompt de entrada
            sampling_params: Parâmetros de sampling
            
        Returns:
            Tupla com:
            - str: Texto gerado
            - dict: Informações de uso (tokens, latência, etc)
        """
        # Carrega modelo se necessário
        info = await self.load_model(model)
        
        try:
            # Registra métricas de GPU antes
            if self.resource_manager.gpu_enabled:
                for i in range(self.resource_manager.num_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle,
                        pynvml.NVML_TEMPERATURE_GPU
                    )
                    
                    GPU_MEMORY_USED.labels(device=str(i)).set(memory.used)
                    GPU_UTILIZATION.labels(device=str(i)).set(util.gpu)
                    GPU_TEMPERATURE.labels(device=str(i)).set(temp)
            
            # Configura parâmetros de sampling
            params = SamplingParams(
                n=sampling_params.get("n", 1),
                temperature=sampling_params.get("temperature", 1.0),
                top_p=sampling_params.get("top_p", 1.0),
                max_tokens=sampling_params.get("max_tokens", 100),
                stop=sampling_params.get("stop", None),
                presence_penalty=sampling_params.get("presence_penalty", 0.0),
                frequency_penalty=sampling_params.get("frequency_penalty", 0.0)
            )
            
            # Gera texto
            start_time = time.time()
            outputs = await info.engine.generate(prompt, params)
            duration = time.time() - start_time
            
            # Extrai resultado
            generated_text = outputs[0].outputs[0].text
            prompt_tokens = outputs[0].prompt_token_ids
            completion_tokens = outputs[0].outputs[0].token_ids
            
            # Atualiza estatísticas
            info.total_requests += 1
            info.total_tokens += len(prompt_tokens) + len(completion_tokens)
            info.avg_latency = (
                (info.avg_latency * (info.total_requests - 1) + duration) /
                info.total_requests
            )
            info.last_used = time.time()
            
            usage = {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
                "latency": duration
            }
            
            return generated_text, usage
            
        except Exception as e:
            logger.error(f"Erro na geração com modelo {model}: {e}")
            ERROR_COUNTER.labels(
                type="generation_error",
                model=model
            ).inc()
            raise
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Obtém informações sobre um modelo.
        
        Args:
            name: Nome do modelo
            
        Returns:
            Dicionário com informações ou None se não carregado
        """
        if name not in self.models:
            return None
            
        info = self.models[name]
        return {
            "name": info.name,
            "path": info.path,
            "loaded_at": info.loaded_at,
            "last_used": info.last_used,
            "total_requests": info.total_requests,
            "total_tokens": info.total_tokens,
            "avg_latency": info.avg_latency,
            "status": "loaded"
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos os modelos configurados.
        
        Returns:
            Lista de dicionários com informações dos modelos
        """
        models = []
        
        for name in MODEL_CONFIG["models"]:
            info = self.get_model_info(name)
            
            if info is None:
                info = {
                    "name": name,
                    "path": MODEL_CONFIG["models"][name]["path"],
                    "status": "not_loaded"
                }
            
            models.append(info)
        
        return models 