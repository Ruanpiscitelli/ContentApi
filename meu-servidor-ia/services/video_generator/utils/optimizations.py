"""
Utilitários de otimização para o gerador de vídeos.
Inclui otimizações de GPU, memória e performance.
"""
import torch
import logging
import psutil
import gc
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

OPTIMIZATION_CONFIG = {
    "use_flash_attention": True,
    "use_memory_efficient_attention": True,
    "enable_cuda_graphs": True,
    "use_sdp_attention": True,
    "max_batch_size": 4,
    "prefetch_factor": 2,
    "num_workers": 4,
    "pin_memory": True
}

def optimize_torch_settings():
    """Otimiza configurações globais do PyTorch."""
    if torch.cuda.is_available():
        # Otimizações CUDA
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Otimizações de memória
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.reset_peak_memory_stats()
            
        # Configurar cache para kernels CUDA
        torch.backends.cuda.cufft_plan_cache.max_size = 2048
        
        logger.info("Configurações PyTorch otimizadas para CUDA")
    else:
        logger.warning("GPU não disponível, usando configurações padrão")

@contextmanager
def gpu_memory_manager():
    """Gerenciador de contexto para operações GPU."""
    try:
        if torch.cuda.is_available():
            # Limpa cache antes da operação
            torch.cuda.empty_cache()
            gc.collect()
            
            # Configura stream dedicada
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                yield
                
            # Sincroniza e limpa
            stream.synchronize()
            torch.cuda.empty_cache()
        else:
            yield
    except Exception as e:
        logger.error(f"Erro no gerenciamento de memória GPU: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def get_memory_info() -> Dict[str, Any]:
    """Retorna informações detalhadas sobre uso de memória."""
    info = {
        "ram": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        }
    }
    
    if torch.cuda.is_available():
        info["gpu"] = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated()
        }
        
    return info

def estimate_max_batch_size(
    model_size_mb: float,
    safety_factor: float = 0.8
) -> int:
    """Estima tamanho máximo de batch baseado na memória disponível."""
    if not torch.cuda.is_available():
        return OPTIMIZATION_CONFIG["max_batch_size"]
    
    try:
        # Obtém memória disponível
        gpu_info = get_memory_info()["gpu"]
        available_mb = (gpu_info["reserved"] - gpu_info["allocated"]) / (1024 * 1024)
        
        # Estima tamanho do batch
        max_batch = int((available_mb * safety_factor) / model_size_mb)
        
        # Limita ao configurado
        return min(max_batch, OPTIMIZATION_CONFIG["max_batch_size"])
        
    except Exception as e:
        logger.error(f"Erro ao estimar batch size: {e}")
        return OPTIMIZATION_CONFIG["max_batch_size"]

def clear_gpu_memory():
    """Limpa memória GPU de forma agressiva."""
    if not torch.cuda.is_available():
        return
        
    try:
        # Limpa caches
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reseta estatísticas
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Força sincronização
        torch.cuda.synchronize()
        
    except Exception as e:
        logger.error(f"Erro ao limpar memória GPU: {e}")

def optimize_model_memory(model: torch.nn.Module):
    """Otimiza uso de memória do modelo."""
    if not torch.cuda.is_available():
        return model
        
    try:
        # Usa AMP (Automatic Mixed Precision)
        model = model.half()
        
        # Otimiza buffers
        model.to(memory_format=torch.channels_last)
        
        # Habilita otimizações de atenção
        if hasattr(model, "enable_xformers_memory_efficient_attention"):
            model.enable_xformers_memory_efficient_attention()
        
        return model
        
    except Exception as e:
        logger.error(f"Erro ao otimizar modelo: {e}")
        return model 