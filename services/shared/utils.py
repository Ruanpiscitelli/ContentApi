"""
Utilitários compartilhados entre todos os serviços.
"""

import os
import io
import json
import base64
import asyncio
import logging
import tempfile
from typing import Any, Dict, Optional, Union, BinaryIO
from pathlib import Path
import aiofiles
from PIL import Image
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Conversão de Imagens
def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Converte uma imagem PIL para bytes."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()
    
def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """Converte bytes para uma imagem PIL."""
    return Image.open(io.BytesIO(image_bytes))
    
def base64_to_image(base64_string: str) -> Image.Image:
    """Converte string base64 para imagem PIL."""
    image_bytes = base64.b64decode(base64_string)
    return bytes_to_image(image_bytes)
    
def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Converte imagem PIL para string base64."""
    return base64.b64encode(image_to_bytes(image, format)).decode()
    
# Operações de Arquivo
async def save_upload_file(
    file: BinaryIO,
    directory: Union[str, Path],
    filename: Optional[str] = None
) -> Path:
    """
    Salva um arquivo enviado de forma assíncrona.
    
    Args:
        file: Arquivo para salvar
        directory: Diretório onde salvar
        filename: Nome do arquivo (opcional)
        
    Returns:
        Path: Caminho do arquivo salvo
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    if not filename:
        filename = Path(file.filename).name
        
    file_path = directory / filename
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
        
    return file_path
    
async def create_temp_file(
    content: Union[str, bytes],
    suffix: Optional[str] = None
) -> str:
    """
    Cria um arquivo temporário de forma assíncrona.
    
    Args:
        content: Conteúdo do arquivo
        suffix: Sufixo do arquivo (ex: .txt)
        
    Returns:
        str: Caminho do arquivo temporário
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if isinstance(content, str):
            content = content.encode()
        tmp.write(content)
        return tmp.name
        
# Operações de Tensor
def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converte tensor PyTorch para array NumPy."""
    return tensor.cpu().detach().numpy()
    
def numpy_to_tensor(array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """Converte array NumPy para tensor PyTorch."""
    tensor = torch.from_numpy(array)
    if device:
        tensor = tensor.to(device)
    return tensor
    
# Utilitários de Cache
def generate_cache_key(*args, **kwargs) -> str:
    """
    Gera uma chave de cache única baseada nos argumentos.
    
    Args:
        *args: Argumentos posicionais
        **kwargs: Argumentos nomeados
        
    Returns:
        str: Chave de cache
    """
    key_parts = []
    
    # Adiciona args
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, (list, tuple, dict)):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(hash(arg)))
            
    # Adiciona kwargs ordenados
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (str, int, float, bool)):
            key_parts.append(f"{key}:{value}")
        elif isinstance(value, (list, tuple, dict)):
            key_parts.append(f"{key}:{json.dumps(value, sort_keys=True)}")
        else:
            key_parts.append(f"{key}:{hash(value)}")
            
    return ":".join(key_parts)
    
# Utilitários de Concorrência
async def run_with_timeout(
    coroutine: Any,
    timeout: float,
    fallback: Any = None
) -> Any:
    """
    Executa uma corotina com timeout.
    
    Args:
        coroutine: Corotina para executar
        timeout: Timeout em segundos
        fallback: Valor de fallback se ocorrer timeout
        
    Returns:
        Any: Resultado da corotina ou fallback
    """
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout após {timeout}s")
        return fallback
        
class RateLimiter:
    """Implementa rate limiting usando token bucket."""
    
    def __init__(self, rate: float, burst: int = 1):
        self.rate = rate  # tokens por segundo
        self.burst = burst  # máximo de tokens
        self.tokens = burst  # tokens atuais
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self) -> bool:
        """
        Tenta adquirir um token.
        
        Returns:
            bool: True se token foi adquirido, False caso contrário
        """
        async with self.lock:
            now = time.time()
            # Adiciona tokens baseado no tempo passado
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # Tenta consumir um token
            if self.tokens >= 1:
                self.tokens -= 1
                return True
                
            return False 