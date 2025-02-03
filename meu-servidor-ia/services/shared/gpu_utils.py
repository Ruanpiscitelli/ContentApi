from contextlib import contextmanager
import torch
import logging

logger = logging.getLogger(__name__)

@contextmanager
def cuda_memory_manager():
    """
    Gerenciador de contexto para otimizar uso de memória CUDA.
    Limpa cache antes e depois das operações de GPU.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def optimize_gpu_settings():
    """
    Configura otimizações globais para GPU.
    """
    if torch.cuda.is_available():
        # Habilita TF32 para melhor performance em GPUs Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Otimizações do cuDNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return True
    return False

def get_gpu_memory_info():
    """
    Retorna informações sobre uso de memória GPU.
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        "allocated": torch.cuda.memory_allocated() / (1024**2),  # MB
        "reserved": torch.cuda.memory_reserved() / (1024**2),    # MB
        "max_allocated": torch.cuda.max_memory_allocated() / (1024**2)
    } 