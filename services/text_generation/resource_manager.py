"""
Gerenciador de recursos e otimizações para o serviço de geração de texto.
"""
import logging
import asyncio
import torch
import psutil
import pynvml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from config import HARDWARE_CONFIG, VLLM_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Informações sobre GPU."""
    index: int
    name: str
    total_memory: int
    free_memory: int
    temperature: int
    utilization: int
    power_usage: float
    power_limit: float

@dataclass
class SystemInfo:
    """Informações do sistema."""
    cpu_percent: float
    memory_percent: float
    swap_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    gpu_info: Optional[List[GPUInfo]] = None

class ResourceManager:
    """Gerenciador de recursos do sistema."""
    
    def __init__(self):
        """Inicializa o gerenciador de recursos."""
        self.gpu_enabled = HARDWARE_CONFIG["gpu"]["enabled"]
        if self.gpu_enabled:
            try:
                pynvml.nvmlInit()
                self.num_gpus = pynvml.nvmlDeviceGetCount()
                logger.info(f"Inicializado com {self.num_gpus} GPU(s)")
            except Exception as e:
                logger.error(f"Erro ao inicializar NVML: {e}")
                self.gpu_enabled = False
    
    async def get_system_info(self) -> SystemInfo:
        """Obtém informações do sistema."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage("/")
        net_io = psutil.net_io_counters()._asdict()
        
        gpu_info = None
        if self.gpu_enabled:
            try:
                gpu_info = []
                for i in range(self.num_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                    
                    gpu_info.append(GPUInfo(
                        index=i,
                        name=name,
                        total_memory=memory.total,
                        free_memory=memory.free,
                        temperature=temp,
                        utilization=util.gpu,
                        power_usage=power,
                        power_limit=power_limit
                    ))
            except Exception as e:
                logger.error(f"Erro ao obter informações da GPU: {e}")
        
        return SystemInfo(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            swap_percent=swap.percent,
            disk_percent=disk.percent,
            network_io=net_io,
            gpu_info=gpu_info
        )
    
    async def optimize_gpu_settings(self):
        """Otimiza configurações de GPU."""
        if not self.gpu_enabled:
            return
        
        try:
            # Configura CUDA
            if HARDWARE_CONFIG["gpu"]["cudnn_benchmark"]:
                torch.backends.cudnn.benchmark = True
            
            if not HARDWARE_CONFIG["gpu"]["deterministic"]:
                torch.backends.cudnn.deterministic = False
            
            # Configura fração de memória
            for i in range(self.num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory = memory.total
                
                # Calcula limite de memória
                memory_limit = int(
                    total_memory * HARDWARE_CONFIG["gpu"]["memory_fraction"]
                )
                
                # Define limite
                torch.cuda.set_per_process_memory_fraction(
                    HARDWARE_CONFIG["gpu"]["memory_fraction"],
                    i
                )
                
                logger.info(
                    f"GPU {i}: Limite de memória definido para "
                    f"{memory_limit / 1024**2:.0f}MB "
                    f"({HARDWARE_CONFIG['gpu']['memory_fraction']*100:.0f}%)"
                )
            
            # Configura tensor parallelism se habilitado
            if HARDWARE_CONFIG["gpu"]["tensor_parallel"]["enabled"]:
                size = HARDWARE_CONFIG["gpu"]["tensor_parallel"]["size"]
                if size > self.num_gpus:
                    logger.warning(
                        f"Tensor parallel size ({size}) maior que número de GPUs "
                        f"({self.num_gpus}). Ajustando para {self.num_gpus}"
                    )
                    size = self.num_gpus
                
                # Define dispositivos para tensor parallelism
                devices = list(range(size))
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
                
                logger.info(f"Tensor parallelism configurado com {size} GPU(s)")
            
        except Exception as e:
            logger.error(f"Erro ao otimizar configurações de GPU: {e}")
    
    async def monitor_resources(self, interval: int = 60):
        """
        Monitora recursos do sistema periodicamente.
        
        Args:
            interval: Intervalo de monitoramento em segundos
        """
        while True:
            try:
                info = await self.get_system_info()
                
                # Verifica limites de recursos
                if info.memory_percent > 95:
                    logger.warning("Uso de memória crítico: %.1f%%", info.memory_percent)
                
                if info.cpu_percent > 95:
                    logger.warning("Uso de CPU crítico: %.1f%%", info.cpu_percent)
                
                if info.gpu_info:
                    for gpu in info.gpu_info:
                        if gpu.temperature > 80:
                            logger.warning(
                                f"Temperatura crítica na GPU {gpu.index}: {gpu.temperature}°C"
                            )
                        
                        memory_used_percent = (
                            (gpu.total_memory - gpu.free_memory) / gpu.total_memory * 100
                        )
                        if memory_used_percent > 95:
                            logger.warning(
                                f"Uso de memória crítico na GPU {gpu.index}: "
                                f"{memory_used_percent:.1f}%"
                            )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Erro no monitoramento de recursos: {e}")
                await asyncio.sleep(interval)

class BatchScheduler:
    """Scheduler para processamento em batch."""
    
    def __init__(self, resource_manager: ResourceManager):
        """
        Inicializa o scheduler.
        
        Args:
            resource_manager: Gerenciador de recursos
        """
        self.resource_manager = resource_manager
        self.batch_size = VLLM_CONFIG["max_num_seqs"]
        self.max_tokens = VLLM_CONFIG["max_num_batched_tokens"]
        self.pending_requests = asyncio.Queue()
        self.processing = False
    
    async def add_request(self, request: Dict[str, Any]) -> asyncio.Future:
        """
        Adiciona requisição ao scheduler.
        
        Args:
            request: Requisição a ser processada
            
        Returns:
            asyncio.Future: Future que será completada com o resultado
        """
        future = asyncio.Future()
        await self.pending_requests.put((request, future))
        
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        return future
    
    def _estimate_tokens(self, request: Dict[str, Any]) -> int:
        """Estima número de tokens em uma requisição."""
        # Implementar estimativa de tokens baseada no modelo
        # Por enquanto usa uma estimativa simples
        if isinstance(request.get("prompt"), str):
            return len(request["prompt"].split())
        return 0
    
    async def _process_batch(self):
        """Processa requisições em batch."""
        if self.processing:
            return
        
        self.processing = True
        try:
            while not self.pending_requests.empty():
                # Coleta requisições até atingir limite do batch
                batch = []
                total_tokens = 0
                
                while len(batch) < self.batch_size:
                    try:
                        request, future = await asyncio.wait_for(
                            self.pending_requests.get(),
                            timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        break
                    
                    tokens = self._estimate_tokens(request)
                    if total_tokens + tokens > self.max_tokens:
                        # Coloca requisição de volta na fila
                        await self.pending_requests.put((request, future))
                        break
                    
                    batch.append((request, future))
                    total_tokens += tokens
                
                if not batch:
                    break
                
                # Processa batch
                try:
                    # Aqui seria chamado o modelo vLLM
                    # Por enquanto só simula o processamento
                    await asyncio.sleep(0.1)
                    
                    for _, future in batch:
                        if not future.done():
                            future.set_result({"status": "success"})
                            
                except Exception as e:
                    logger.error(f"Erro no processamento do batch: {e}")
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(e)
                
        finally:
            self.processing = False
            
            # Se ainda tem requisições, agenda próximo processamento
            if not self.pending_requests.empty():
                asyncio.create_task(self._process_batch()) 