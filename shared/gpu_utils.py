"""
Utilitários para gerenciamento de GPU.
"""
import torch
import contextlib

@contextlib.contextmanager
def cuda_memory_manager():
    """
    Gerencia a memória CUDA, liberando cache após uso.
    """
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def optimize_gpu_settings():
    """
    Otimiza configurações da GPU.
    """
    if torch.cuda.is_available():
        # Configura para usar TF32 se disponível
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Habilita cuDNN benchmark
        torch.backends.cudnn.benchmark = True
        
        # Configura para determinismo se necessário
        # torch.backends.cudnn.deterministic = True
        
        # Limpa cache
        torch.cuda.empty_cache()
