"""
Funções utilitárias centralizadas para o dashboard.
"""
import psutil
import torch
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import humanize
import string
import traceback
import logging
from .core.exceptions import ValidationError

logger = logging.getLogger(__name__)

class SystemStats:
    """Classe para gerenciar estatísticas do sistema."""
    
    @staticmethod
    def get_stats() -> Dict[str, float]:
        """Obtém estatísticas do sistema."""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "gpu_percent": 0.0
        }
        
        if torch.cuda.is_available():
            try:
                gpu_stats = torch.cuda.mem_get_info()
                stats["gpu_percent"] = (gpu_stats[1] - gpu_stats[0]) / gpu_stats[1] * 100
            except Exception as e:
                logger.warning(f"Erro ao obter estatísticas da GPU: {e}")
                
        return stats

class DataFormatter:
    """Classe para formatação de dados."""
    
    @staticmethod
    def format_bytes(size: int) -> str:
        """Formata bytes para formato legível."""
        return humanize.naturalsize(size)
    
    @staticmethod
    def format_timestamp(timestamp: datetime) -> str:
        """Formata timestamp para exibição."""
        return humanize.naturaltime(timestamp)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitiza nome de arquivo."""
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        filename = "".join(c for c in filename if c in valid_chars)
        return " ".join(filename.split())

class Validator:
    """Classe para validação de dados."""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, Optional[str]]:
        """Valida senha de acordo com os requisitos."""
        if len(password) < 8:
            return False, "A senha deve ter pelo menos 8 caracteres"
            
        if not any(c.isupper() for c in password):
            return False, "A senha deve ter pelo menos uma letra maiúscula"
            
        if not any(c.islower() for c in password):
            return False, "A senha deve ter pelo menos uma letra minúscula"
            
        if not any(c.isdigit() for c in password):
            return False, "A senha deve ter pelo menos um número"
            
        return True, None
    
    @staticmethod
    def validate_api_key(key: str) -> bool:
        """Valida formato de chave API."""
        if not key or len(key) != 32:
            return False
            
        valid_chars = set(string.ascii_letters + string.digits)
        return all(c in valid_chars for c in key)

class Logger:
    """Classe para logging centralizado."""
    
    @staticmethod
    def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Registra erro com contexto."""
        logger.error(
            f"Erro: {str(error)}",
            extra={
                "context": context or {},
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc()
            }
        )

# Funções de conveniência para acesso rápido
get_system_stats = SystemStats.get_stats
format_bytes = DataFormatter.format_bytes
format_timestamp = DataFormatter.format_timestamp
sanitize_filename = DataFormatter.sanitize_filename
validate_password = Validator.validate_password
validate_api_key = Validator.validate_api_key
log_error = Logger.log_error 