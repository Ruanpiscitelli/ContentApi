"""
Utilitários para otimização de GPU
"""
import os
import torch
import logging
import contextlib
from typing import Optional
import pynvml
import platform

logger = logging.getLogger(__name__)

def optimize_gpu_settings(
    device_id: int = 0,
    memory_fraction: float = 0.9,
    benchmark: bool = True
):
    """
    Otimiza configurações da GPU para melhor performance com Fish Speech.
    
    Args:
        device_id: ID do dispositivo GPU
        memory_fraction: Fração de memória a ser alocada
        benchmark: Ativar benchmark do cuDNN
    """
    if not torch.cuda.is_available():
        if platform.system() == "Darwin":
            if torch.backends.mps.is_available():
                logger.info("MPS disponível para Mac. Usando aceleração Metal.")
                torch.backends.mps.enabled = True
            else:
                logger.warning("GPU não disponível no Mac. Usando CPU.")
        else:
            logger.warning("GPU não disponível. Usando CPU.")
        return
    
    try:
        # Configura dispositivo
        torch.cuda.set_device(device_id)
        
        # Otimizações para precisão e performance
        torch.backends.cuda.matmul.allow_tf32 = True  # Melhor performance em GPUs Ampere+
        torch.backends.cudnn.allow_tf32 = True
        
        # Otimizações cuDNN
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = False
        
        # Otimizações de memória
        if memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(
                memory_fraction,
                device_id
            )
        
        # Habilita alocação assíncrona
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        # Configurar cache para kernels CUDA
        os.environ["CUDA_CACHE_PATH"] = ".cache/cuda"
        os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"  # 2GB
        
        # Otimizações para Fish Speech
        torch.backends.cuda.cufft_plan_cache.max_size = 2048
        torch.backends.cuda.cufft_plan_cache.clear()
        
        if torch.cuda.get_device_capability(device_id)[0] >= 7:
            # Otimizações para Volta+ (>=SM70)
            torch.backends.cuda.preferred_linalg_library = "cusolver"
        
        logger.info(
            f"GPU {device_id} otimizada: "
            f"memoria={memory_fraction:.1%}, benchmark={benchmark}, "
            f"tf32=True, arch={torch.cuda.get_device_capability(device_id)}"
        )
        
    except Exception as e:
        logger.error(f"Erro ao otimizar GPU: {e}")

@contextlib.contextmanager
def cuda_memory_manager(
    device_id: Optional[int] = None,
    optimize: bool = True,
    stream: Optional[torch.cuda.Stream] = None
):
    """
    Gerenciador de contexto otimizado para operações GPU.
    
    Args:
        device_id: ID do dispositivo GPU (opcional)
        optimize: Se True, aplica otimizações
        stream: Stream CUDA customizada (opcional)
    """
    if not torch.cuda.is_available():
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            with torch.mps.device():
                yield
        else:
            yield
        return
    
    try:
        if device_id is not None:
            torch.cuda.set_device(device_id)
        
        if optimize:
            torch.cuda.empty_cache()
        
        stream = stream or torch.cuda.Stream()
        
        with torch.cuda.stream(stream), \
             torch.cuda.amp.autocast(enabled=True), \
             torch.inference_mode():
            yield
            
    finally:
        if optimize and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            if stream:
                stream.synchronize()

def get_gpu_info() -> dict:
    """
    Obtém informações detalhadas sobre as GPUs disponíveis.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "devices": [],
        "mps_available": torch.backends.mps.is_available() if platform.system() == "Darwin" else False
    }
    
    if not info["cuda_available"] and not info["mps_available"]:
        return info
    
    try:
        if info["cuda_available"]:
            pynvml.nvmlInit()
            info["current_device"] = torch.cuda.current_device()
            
            for i in range(info["device_count"]):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_info = {
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "memory": {
                        "total": torch.cuda.get_device_properties(i).total_memory,
                        "allocated": torch.cuda.memory_allocated(i),
                        "cached": torch.cuda.memory_reserved(i)
                    },
                    "temperature": pynvml.nvmlDeviceGetTemperature(
                        handle,
                        pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "power": {
                        "usage": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                        "limit": pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                    },
                    "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    "optimizations": {
                        "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_enabled": torch.backends.cudnn.enabled,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark
                    }
                }
                info["devices"].append(device_info)
            
            pynvml.nvmlShutdown()
        
        elif info["mps_available"]:
            info["current_device"] = "mps"
            info["devices"].append({
                "name": "Apple Silicon GPU",
                "type": "mps",
                "memory": {
                    "total": None,  # Não disponível no MPS
                    "allocated": None
                }
            })
        
    except Exception as e:
        logger.error(f"Erro ao obter informações da GPU: {e}")
    
    return info

def estimate_max_batch_size(
    model_size_mb: float,
    safety_factor: float = 0.8,
    min_batch: int = 4,
    max_batch: int = 32
) -> int:
    """
    Estima o tamanho máximo de batch baseado na memória disponível.
    Otimizado para Fish Speech.
    
    Args:
        model_size_mb: Tamanho do modelo em MB
        safety_factor: Fator de segurança (0-1)
        min_batch: Tamanho mínimo do batch
        max_batch: Tamanho máximo do batch
        
    Returns:
        int: Tamanho máximo estimado do batch
    """
    if not torch.cuda.is_available():
        return min_batch
    
    try:
        # Obtém memória disponível
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        free = total - allocated
        
        # Converte para MB
        free_mb = free / (1024 * 1024)
        
        # Estima tamanho do batch
        # Considera overhead do Fish Speech
        overhead_mb = 512  # 512MB de overhead estimado
        available_mb = (free_mb - overhead_mb) * safety_factor
        
        # Calcula batch size ideal
        ideal_batch = int(available_mb / model_size_mb)
        
        # Limita entre min e max
        batch_size = max(min_batch, min(ideal_batch, max_batch))
        
        # Ajusta para múltiplo de 4 (otimização para Fish Speech)
        batch_size = (batch_size // 4) * 4
        
        logger.info(
            f"Batch size estimado: {batch_size} "
            f"(memória livre: {free_mb:.0f}MB, "
            f"overhead: {overhead_mb}MB)"
        )
        
        return batch_size
        
    except Exception as e:
        logger.error(f"Erro ao estimar batch size: {e}")
        return min_batch 