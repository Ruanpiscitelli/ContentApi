"""
Utilitários para otimização de GPU e modelos.
"""
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def setup_gpu_optimizations(
    enable_xformers: bool = True,
    enable_vllm: bool = False,
    device_id: int = 0
):
    """
    Configura otimizações para GPU.
    
    Args:
        enable_xformers: Ativar otimizações xformers
        enable_vllm: Ativar otimizações VLLM
        device_id: ID do dispositivo CUDA
    """
    if torch.cuda.is_available():
        # Configurações básicas CUDA
        torch.cuda.set_device(device_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Otimizações xformers
        if enable_xformers:
            try:
                import xformers
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("xformers ativado com sucesso")
            except ImportError:
                logger.warning("xformers não disponível")
                
        # Otimizações VLLM
        if enable_vllm:
            try:
                import vllm
                logger.info("VLLM ativado com sucesso")
            except ImportError:
                logger.warning("VLLM não disponível")

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
