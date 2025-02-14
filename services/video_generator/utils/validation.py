"""
Validações e tratamento de erros para o gerador de vídeos.
"""
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class VideoGenerationError(Exception):
    """Erro base para geração de vídeos."""
    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ValidationError(VideoGenerationError):
    """Erro de validação de parâmetros."""
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")

class ResourceError(VideoGenerationError):
    """Erro de recursos (memória, GPU, etc)."""
    def __init__(self, message: str):
        super().__init__(message, "RESOURCE_ERROR")

class ProcessingError(VideoGenerationError):
    """Erro durante o processamento do vídeo."""
    def __init__(self, message: str):
        super().__init__(message, "PROCESSING_ERROR")

def validate_dimensions(width: int, height: int) -> None:
    """
    Valida dimensões do vídeo.
    
    Args:
        width: Largura do vídeo
        height: Altura do vídeo
        
    Raises:
        ValidationError: Se as dimensões forem inválidas
    """
    # Verifica se é múltiplo de 64
    if width % 64 != 0 or height % 64 != 0:
        raise ValidationError(
            "Dimensões devem ser múltiplos de 64"
        )
    
    # Verifica aspect ratio
    if not validate_aspect_ratio(width, height):
        raise ValidationError(
            "Aspect ratio não suportado. Use uma das resoluções suportadas."
        )
    
    # Verifica limites
    if width < 320 or width > 1280 or height < 320 or height > 1280:
        raise ValidationError(
            "Dimensões devem estar entre 320 e 1280 pixels"
        )

def validate_aspect_ratio(width: int, height: int) -> bool:
    """Valida se o aspect ratio é suportado."""
    supported_ratios = {
        "9:16": 0.5625,   # 9/16
        "16:9": 1.7778,   # 16/9
        "4:3": 1.3333,    # 4/3
        "3:4": 0.75,      # 3/4
        "1:1": 1.0        # 1/1
    }
    
    ratio = width / height
    return any(abs(ratio - r) < 0.01 for r in supported_ratios.values())

def validate_duration(frames: int) -> None:
    """
    Valida duração do vídeo em frames.
    
    Args:
        frames: Número de frames
        
    Raises:
        ValidationError: Se a duração for inválida
    """
    if frames % 8 != 0:
        raise ValidationError("Duração deve ser múltiplo de 8 frames")
    
    if frames < 8 or frames > 128:
        raise ValidationError("Duração deve estar entre 8 e 128 frames")

def validate_prompt(prompt: str) -> None:
    """
    Valida prompt de geração.
    
    Args:
        prompt: Texto do prompt
        
    Raises:
        ValidationError: Se o prompt for inválido
    """
    if not prompt or len(prompt.strip()) == 0:
        raise ValidationError("Prompt não pode estar vazio")
    
    if len(prompt) > 1000:
        raise ValidationError("Prompt muito longo (máximo 1000 caracteres)")

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
            
        required_fields = ["prompt", "duration", "height", "width"]
        missing = [f for f in required_fields if f not in template]
        if missing:
            raise ValidationError(
                f"Campos obrigatórios ausentes no template: {', '.join(missing)}"
            )
            
        # Valida campos
        validate_dimensions(template["width"], template["height"])
        validate_duration(template["duration"])
        validate_prompt(template["prompt"])
        
        return template
        
    except json.JSONDecodeError:
        raise ValidationError(f"Template {template_id} é um JSON inválido")
    except Exception as e:
        raise ValidationError(f"Erro ao carregar template: {str(e)}")

def validate_generation_params(params: Dict) -> None:
    """
    Valida parâmetros de geração.
    
    Args:
        params: Dicionário com parâmetros
        
    Raises:
        ValidationError: Se algum parâmetro for inválido
    """
    # Valida guidance scales
    if "guidance_scale" in params:
        if not 1.0 <= params["guidance_scale"] <= 20.0:
            raise ValidationError(
                "guidance_scale deve estar entre 1.0 e 20.0"
            )
            
    if "embedded_cfg_scale" in params:
        if not 1.0 <= params["embedded_cfg_scale"] <= 20.0:
            raise ValidationError(
                "embedded_cfg_scale deve estar entre 1.0 e 20.0"
            )
    
    # Valida motion bucket
    if "motion_bucket_id" in params:
        if not 1 <= params["motion_bucket_id"] <= 255:
            raise ValidationError(
                "motion_bucket_id deve estar entre 1 e 255"
            )
    
    # Valida noise augmentation
    if "noise_aug_strength" in params:
        if not 0.0 <= params["noise_aug_strength"] <= 1.0:
            raise ValidationError(
                "noise_aug_strength deve estar entre 0.0 e 1.0"
            )
    
    # Valida FPS
    if "fps" in params:
        if not 8 <= params["fps"] <= 30:
            raise ValidationError("fps deve estar entre 8 e 30") 