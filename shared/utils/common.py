"""
Este módulo contém funções utilitárias compartilhadas entre os serviços.
"""

import os
import logging
from typing import Any, Dict

def setup_logging(service_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configura o logger para o serviço.
    
    Args:
        service_name: Nome do serviço para identificação nos logs
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def validate_file_path(file_path: str) -> bool:
    """
    Valida se um caminho de arquivo é seguro e existe.
    
    Args:
        file_path: Caminho do arquivo a ser validado
    
    Returns:
        bool: True se o arquivo existe e é seguro, False caso contrário
    """
    if not os.path.exists(file_path):
        return False
    
    # Verifica se o caminho é absoluto
    if not os.path.isabs(file_path):
        return False
    
    # Verifica se o arquivo está dentro do diretório permitido
    allowed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    file_dir = os.path.abspath(file_path)
    
    return file_dir.startswith(allowed_dir)

def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    Formata uma exceção para resposta da API.
    
    Args:
        error: Exceção a ser formatada
    
    Returns:
        Dicionário com detalhes do erro formatados
    """
    return {
        "error": True,
        "message": str(error),
        "type": error.__class__.__name__
    } 