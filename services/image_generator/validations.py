"""
Validações e tratamento de erros para o serviço de geração de imagens.
"""
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
import io
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

class ErrorDetail(BaseModel):
    """Modelo para detalhes de erro."""
    code: str
    message: str
    details: Optional[Dict] = None

class ImageGenerationError(HTTPException):
    """Erro base para geração de imagens."""
    def __init__(
        self, 
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict] = None
    ):
        self.error_detail = ErrorDetail(
            code=code,
            message=message,
            details=details
        )
        super().__init__(
            status_code=status_code,
            detail=self.error_detail.dict()
        )

class ValidationError(ImageGenerationError):
    """Erro de validação de parâmetros."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            message=message,
            details=details
        )

class ResourceError(ImageGenerationError):
    """Erro de recursos (memória, GPU, etc)."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="RESOURCE_ERROR",
            message=message,
            details=details
        )

class ProcessingError(ImageGenerationError):
    """Erro durante o processamento da imagem."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="PROCESSING_ERROR",
            message=message,
            details=details
        )

def validate_system_requirements():
    """
    Valida requisitos do sistema conforme documentação Hunyuan.
    
    Raises:
        ResourceError: Se os requisitos não forem atendidos
    """
    if not torch.cuda.is_available():
        raise ResourceError(
            message="GPU NVIDIA com suporte CUDA é necessária",
            details={"cuda_available": False}
        )
        
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Validações baseadas no modelo
    model_requirements = {
        "text-to-image": {
            "min_memory": 8,
            "recommended": 16,
            "max_resolution": (2048, 2048)
        },
        "image-to-image": {
            "min_memory": 12,
            "recommended": 24,
            "max_resolution": (2048, 2048)
        }
    }
    
    model_type = "text-to-image"  # Ou image-to-image
    req = model_requirements[model_type]
    
    if gpu_memory < req["min_memory"]:
        raise ResourceError(
            message=f"GPU com {gpu_memory:.1f}GB detectada. Mínimo de {req['min_memory']}GB necessário",
            details={
                "available_memory": gpu_memory,
                "required_memory": req["min_memory"],
                "model_type": model_type
            }
        )

def validate_batch_size(batch_size: int, model_size_mb: float) -> int:
    """
    Estima e valida tamanho máximo de batch.
    
    Args:
        batch_size: Tamanho do batch desejado
        model_size_mb: Tamanho do modelo em MB
        
    Returns:
        int: Tamanho de batch validado
    """
    if not torch.cuda.is_available():
        return 1
        
    try:
        # Obtém memória disponível
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        free = total - allocated
        
        # Estima batch size
        free_mb = free / (1024 * 1024)
        overhead_mb = 512  # Overhead estimado
        available_mb = (free_mb - overhead_mb) * 0.8  # 80% da memória livre
        
        ideal_batch = int(available_mb / model_size_mb)
        return max(1, min(ideal_batch, batch_size))
        
    except Exception as e:
        logger.error(f"Erro ao estimar batch size: {e}")
        return 1

def validate_model_requirements(model_id: str) -> None:
    """
    Valida requisitos específicos do modelo.
    
    Args:
        model_id: ID do modelo
        
    Raises:
        ResourceError: Se os requisitos não forem atendidos
    """
    # Requisitos específicos por modelo
    model_requirements = {
        "stabilityai/stable-diffusion-xl-base-1.0": {
            "min_gpu_mem": 12,
            "recommended_gpu_mem": 16
        },
        "runwayml/stable-diffusion-v1-5": {
            "min_gpu_mem": 6,
            "recommended_gpu_mem": 8
        }
    }
    
    if model_id in model_requirements:
        req = model_requirements[model_id]
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory < req["min_gpu_mem"]:
            raise ResourceError(
                message=f"Modelo {model_id} requer mínimo de {req['min_gpu_mem']}GB de VRAM. Detectado: {gpu_memory:.1f}GB",
                details={
                    "available_memory": gpu_memory,
                    "required_memory": req["min_gpu_mem"],
                    "model_id": model_id
                }
            )

def validate_model_compatibility(model: str, controlnet_model: Optional[str] = None) -> None:
    """
    Valida compatibilidade entre modelo base e ControlNet.
    
    Args:
        model: ID do modelo base
        controlnet_model: ID do modelo ControlNet (opcional)
        
    Raises:
        ValidationError: Se os modelos forem incompatíveis
    """
    if not controlnet_model:
        return
        
    # Mapeamento de compatibilidade
    compatibility_map = {
        "sdxl": ["lllyasviel/control_v11p_sd15_canny", "lllyasviel/control_v11p_sd15_openpose"],
        "sd15": ["lllyasviel/control_v11p_sd15_canny", "lllyasviel/control_v11p_sd15_openpose"],
        "sd21": ["lllyasviel/control_v11p_sd21_canny", "lllyasviel/control_v11p_sd21_openpose"]
    }
    
    model_base = model.split("/")[-1].lower()
    for base, compatible in compatibility_map.items():
        if base in model_base and controlnet_model not in compatible:
            raise ValidationError(
                message=f"ControlNet {controlnet_model} não é compatível com o modelo {model}",
                details={
                    "model_base": model_base,
                    "controlnet_model": controlnet_model
                }
            )

def validate_lora_weights(loras: List[Dict]) -> None:
    """
    Valida configurações de LoRA.
    
    Args:
        loras: Lista de configurações LoRA
        
    Raises:
        ValidationError: Se as configurações forem inválidas
    """
    for lora in loras:
        if "path" not in lora or "scale" not in lora:
            raise ValidationError(
                "Configuração LoRA deve incluir 'path' e 'scale'"
            )
            
        scale = float(lora["scale"])
        if not 0.0 <= scale <= 2.0:
            raise ValidationError(
                f"Escala LoRA {scale} inválida. Deve estar entre 0.0 e 2.0"
            )
            
        if not Path(lora["path"]).exists():
            raise ValidationError(
                f"Arquivo LoRA não encontrado: {lora['path']}"
            )

def validate_vae_config(vae_id: Optional[str], model: str) -> None:
    """
    Valida configuração do VAE.
    
    Args:
        vae_id: ID ou path do VAE
        model: ID do modelo base
        
    Raises:
        ValidationError: Se a configuração for inválida
    """
    if not vae_id:
        return
        
    # Verifica compatibilidade
    model_base = model.split("/")[-1].lower()
    if "sdxl" in model_base and "sdxl" not in vae_id.lower():
        raise ValidationError(
            f"VAE {vae_id} pode não ser compatível com modelo SDXL"
        )

def validate_scheduler_config(scheduler: str, num_inference_steps: int) -> None:
    """
    Valida configuração do scheduler.
    
    Args:
        scheduler: Nome do scheduler
        num_inference_steps: Número de passos
        
    Raises:
        ValidationError: Se a configuração for inválida
    """
    valid_schedulers = {
        "DDIMScheduler": (20, 100),
        "DPMSolverMultistepScheduler": (15, 100),
        "EulerDiscreteScheduler": (25, 100),
        "EulerAncestralDiscreteScheduler": (25, 100),
        "UniPCMultistepScheduler": (20, 100)
    }
    
    if scheduler not in valid_schedulers:
        raise ValidationError(
            f"Scheduler inválido. Opções: {list(valid_schedulers.keys())}"
        )
        
    min_steps, max_steps = valid_schedulers[scheduler]
    if not min_steps <= num_inference_steps <= max_steps:
        raise ValidationError(
            f"Número de passos {num_inference_steps} inválido para {scheduler}. "
            f"Deve estar entre {min_steps} e {max_steps}"
        )

def validate_memory_requirements(
    height: int,
    width: int,
    batch_size: int = 1,
    use_controlnet: bool = False
) -> None:
    """
    Valida requisitos de memória para geração.
    
    Args:
        height: Altura da imagem
        width: Largura da imagem
        batch_size: Tamanho do batch
        use_controlnet: Se usa ControlNet
        
    Raises:
        ResourceError: Se não houver memória suficiente
    """
    if not torch.cuda.is_available():
        return
        
    try:
        # Estima memória necessária (em GB)
        pixels = height * width
        mem_per_image = pixels * 4 * 4 / (1024 * 1024 * 1024)  # 4 canais, 4 bytes
        total_mem = mem_per_image * batch_size
        
        if use_controlnet:
            total_mem *= 1.5  # 50% extra para ControlNet
            
        # Verifica memória disponível
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
        if total_mem > gpu_mem * 0.8:  # 80% da memória total
            raise ResourceError(
                f"Memória insuficiente para geração. Necessário: {total_mem:.1f}GB, "
                f"Disponível: {gpu_mem:.1f}GB"
            )
            
    except Exception as e:
        logger.error(f"Erro ao validar memória: {e}")
        raise ResourceError(str(e))

def validate_image_data(image_data: bytes, max_size_mb: int = 10) -> Image.Image:
    """
    Valida e pré-processa imagem de entrada.
    
    Args:
        image_data: Bytes da imagem
        max_size_mb: Tamanho máximo em MB
        
    Returns:
        Image.Image: Imagem processada
        
    Raises:
        ValidationError: Se a imagem for inválida
    """
    if len(image_data) > max_size_mb * 1024 * 1024:
        raise ValidationError(
            f"Imagem muito grande. Máximo: {max_size_mb}MB"
        )
        
    try:
        image = Image.open(io.BytesIO(image_data))
        
        # Valida formato
        if image.format not in ["PNG", "JPEG", "WEBP"]:
            raise ValidationError(
                f"Formato {image.format} não suportado. Use PNG, JPEG ou WEBP"
            )
            
        # Valida dimensões
        if max(image.size) > 4096:
            raise ValidationError(
                "Dimensão máxima excedida (4096px)"
            )
            
        # Converte para RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
        
    except Exception as e:
        raise ValidationError(f"Erro ao processar imagem: {str(e)}")

def load_and_validate_template(template_id: str) -> Dict:
    """
    Carrega e valida um template.
    
    Args:
        template_id: ID do template
        
    Returns:
        Dict: Template carregado e validado
        
    Raises:
        ValidationError: Se o template for inválido
    """
    template_path = Path("templates") / f"{template_id}.json"
    
    try:
        if not template_path.exists():
            raise ValidationError(f"Template {template_id} não encontrado")
            
        with open(template_path) as f:
            template = json.load(f)
            
        # Valida campos obrigatórios
        required = ["model", "prompt"]
        missing = [f for f in required if f not in template]
        if missing:
            raise ValidationError(
                f"Campos obrigatórios ausentes no template: {', '.join(missing)}"
            )
            
        # Valida valores
        if "height" in template and "width" in template:
            validate_memory_requirements(
                template["height"],
                template["width"],
                use_controlnet="controlnet" in template
            )
            
        if "scheduler" in template and "num_inference_steps" in template:
            validate_scheduler_config(
                template["scheduler"],
                template["num_inference_steps"]
            )
            
        if "loras" in template:
            validate_lora_weights(template["loras"])
            
        return template
        
    except json.JSONDecodeError:
        raise ValidationError(f"Template {template_id} é um JSON inválido")
    except Exception as e:
        raise ValidationError(f"Erro ao carregar template: {str(e)}")

def estimate_generation_time(
    height: int,
    width: int,
    num_inference_steps: int,
    use_controlnet: bool = False
) -> float:
    """
    Estima tempo de geração em segundos.
    
    Args:
        height: Altura da imagem
        width: Largura da imagem
        num_inference_steps: Número de passos
        use_controlnet: Se usa ControlNet
        
    Returns:
        float: Tempo estimado em segundos
    """
    # Fatores base (ajustar com dados reais)
    pixels = height * width
    base_time = pixels * num_inference_steps * 1e-7
    
    if use_controlnet:
        base_time *= 1.3  # 30% mais lento com ControlNet
        
    if not torch.cuda.is_available():
        base_time *= 10  # 10x mais lento na CPU
        
    return base_time 