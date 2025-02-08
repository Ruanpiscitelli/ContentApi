"""
Funções utilitárias para o dashboard.
"""
import psutil
import torch
from datetime import datetime
from typing import Dict, Any, Optional
import humanize
import string
import traceback
import logging

logger = logging.getLogger(__name__)

def get_system_stats() -> Dict[str, Any]:
    """
    Obtém estatísticas do sistema.
    """
    stats = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "gpu_percent": 0
    }
    
    # Adiciona informações da GPU se disponível
    if torch.cuda.is_available():
        try:
            gpu_stats = torch.cuda.mem_get_info()
            stats["gpu_percent"] = (gpu_stats[1] - gpu_stats[0]) / gpu_stats[1] * 100
        except:
            pass
            
    return stats

def format_bytes(size: int) -> str:
    """
    Formata bytes para formato legível.
    """
    return humanize.naturalsize(size)

def format_timestamp(timestamp: datetime) -> str:
    """
    Formata timestamp para exibição.
    """
    return humanize.naturaltime(timestamp)

def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """
    Valida senha de acordo com os requisitos.
    """
    if len(password) < 8:
        return False, "A senha deve ter pelo menos 8 caracteres"
        
    if not any(c.isupper() for c in password):
        return False, "A senha deve ter pelo menos uma letra maiúscula"
        
    if not any(c.islower() for c in password):
        return False, "A senha deve ter pelo menos uma letra minúscula"
        
    if not any(c.isdigit() for c in password):
        return False, "A senha deve ter pelo menos um número"
        
    return True, None

def sanitize_filename(filename: str) -> str:
    """
    Sanitiza nome de arquivo.
    """
    # Remove caracteres inválidos
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = "".join(c for c in filename if c in valid_chars)
    
    # Remove espaços extras
    filename = " ".join(filename.split())
    
    return filename

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Registra erro com contexto.
    """
    logger.error(
        f"Erro: {str(error)}",
        extra={
            "context": context or {},
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc()
        }
    )

def is_valid_api_key(key: str) -> bool:
    """
    Valida formato de chave API.
    """
    if not key:
        return False
        
    # Verifica tamanho
    if len(key) != 32:
        return False
        
    # Verifica caracteres válidos
    valid_chars = set(string.ascii_letters + string.digits)
    if not all(c in valid_chars for c in key):
        return False
        
    return True

def mask_api_key(key: str) -> str:
    """
    Mascara chave API para exibição.
    """
    if not key:
        return ""
        
    # Mantém primeiros e últimos 4 caracteres
    return f"{key[:4]}...{key[-4:]}" 