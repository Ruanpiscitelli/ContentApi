"""
Validação e sanitização de entrada para o serviço de geração de texto.
"""
import re
import html
import logging
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, validator, Field
from fastapi import HTTPException

from config import VALIDATION_CONFIG
from metrics import ERROR_COUNTER

logger = logging.getLogger(__name__)

class ValidationError(HTTPException):
    """Erro de validação customizado."""
    
    def __init__(self, detail: str):
        """
        Inicializa o erro.
        
        Args:
            detail: Detalhes do erro
        """
        super().__init__(
            status_code=422,
            detail=detail
        )
        ERROR_COUNTER.labels(
            type="validation_error",
            model="validation"
        ).inc()

class InputSanitizer:
    """Sanitiza entrada de texto."""
    
    def __init__(self):
        """Inicializa o sanitizador."""
        # Regex para detectar XSS
        self.xss_pattern = re.compile(
            r"<[^>]*script|javascript:|data:|vbscript:",
            re.IGNORECASE
        )
        
        # Regex para detectar SQL injection
        self.sql_pattern = re.compile(
            r"\b(union\s+all|union\s+select|insert\s+into|drop\s+table)\b",
            re.IGNORECASE
        )
        
        # Regex para detectar comandos do sistema
        self.command_pattern = re.compile(
            r"\b(exec\s+|system\s*\(|shell_exec\s*\()\b",
            re.IGNORECASE
        )
        
        # Caracteres permitidos
        self.allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?-_()[]{}:;\"'`@#$%^&*+=/<>~"
        )
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitiza texto de entrada.
        
        Args:
            text: Texto a ser sanitizado
            
        Returns:
            Texto sanitizado
            
        Raises:
            ValidationError: Se o texto contiver padrões maliciosos
        """
        # Verifica XSS
        if self.xss_pattern.search(text):
            raise ValidationError("Texto contém padrões XSS suspeitos")
        
        # Verifica SQL injection
        if self.sql_pattern.search(text):
            raise ValidationError("Texto contém padrões SQL injection suspeitos")
        
        # Verifica comandos do sistema
        if self.command_pattern.search(text):
            raise ValidationError("Texto contém padrões de comando suspeitos")
        
        # Remove caracteres não permitidos
        text = "".join(c for c in text if c in self.allowed_chars)
        
        # Escapa HTML
        text = html.escape(text)
        
        return text
    
    def sanitize_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitiza dados JSON.
        
        Args:
            data: Dados a serem sanitizados
            
        Returns:
            Dados sanitizados
        """
        result = {}
        
        for key, value in data.items():
            # Sanitiza chaves
            key = self.sanitize_text(str(key))
            
            # Sanitiza valores recursivamente
            if isinstance(value, dict):
                value = self.sanitize_json(value)
            elif isinstance(value, list):
                value = [
                    self.sanitize_json(v) if isinstance(v, dict)
                    else self.sanitize_text(str(v))
                    for v in value
                ]
            elif value is not None:
                value = self.sanitize_text(str(value))
            
            result[key] = value
        
        return result

class ContentValidator(BaseModel):
    """Validador de conteúdo."""
    
    text: Optional[str] = Field(None, min_length=1, max_length=32768)
    images: Optional[List[str]] = Field(None, max_items=4)
    
    @validator("text")
    def validate_text(cls, v):
        """Valida texto."""
        if v is None:
            return v
            
        # Verifica comprimento
        if len(v) > VALIDATION_CONFIG["max_text_length"]:
            raise ValidationError(
                f"Texto excede tamanho máximo de "
                f"{VALIDATION_CONFIG['max_text_length']} caracteres"
            )
        
        # Verifica se tem conteúdo real
        if not v.strip():
            raise ValidationError("Texto vazio")
        
        return v
    
    @validator("images")
    def validate_images(cls, v):
        """Valida imagens."""
        if v is None:
            return v
            
        # Verifica número de imagens
        if len(v) > VALIDATION_CONFIG["max_images"]:
            raise ValidationError(
                f"Número de imagens excede máximo de "
                f"{VALIDATION_CONFIG['max_images']}"
            )
        
        # Verifica tamanho de cada imagem
        for img in v:
            if len(img) > VALIDATION_CONFIG["max_image_size"]:
                raise ValidationError(
                    f"Tamanho de imagem excede máximo de "
                    f"{VALIDATION_CONFIG['max_image_size']} bytes"
                )
        
        return v

class FunctionValidator(BaseModel):
    """Validador de funções."""
    
    name: str = Field(..., min_length=1, max_length=64)
    description: Optional[str] = Field(None, max_length=1024)
    parameters: Optional[Dict[str, Any]] = None
    
    @validator("name")
    def validate_name(cls, v):
        """Valida nome da função."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValidationError(
                "Nome de função inválido. Use apenas letras, números e _"
            )
        return v
    
    @validator("parameters")
    def validate_parameters(cls, v):
        """Valida parâmetros da função."""
        if v is None:
            return v
            
        # Verifica número de parâmetros
        if len(v) > VALIDATION_CONFIG["max_function_params"]:
            raise ValidationError(
                f"Número de parâmetros excede máximo de "
                f"{VALIDATION_CONFIG['max_function_params']}"
            )
        
        # Valida cada parâmetro
        for name, param in v.items():
            # Valida nome
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
                raise ValidationError(
                    f"Nome de parâmetro inválido: {name}. "
                    "Use apenas letras, números e _"
                )
            
            # Valida tipo
            if not isinstance(param, (str, int, float, bool, list, dict)):
                raise ValidationError(
                    f"Tipo de parâmetro inválido: {type(param)}. "
                    "Use apenas tipos básicos"
                )
        
        return v

class RequestValidator(BaseModel):
    """Validador de requisições."""
    
    model: str = Field(..., min_length=1, max_length=64)
    prompt: Union[str, List[Dict[str, Any]]] = Field(...)
    max_tokens: Optional[int] = Field(None, gt=0, le=32768)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    n: Optional[int] = Field(None, gt=0, le=10)
    stream: Optional[bool] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    
    @validator("model")
    def validate_model(cls, v):
        """Valida modelo."""
        if v not in VALIDATION_CONFIG["allowed_models"]:
            raise ValidationError(f"Modelo não permitido: {v}")
        return v
    
    @validator("prompt")
    def validate_prompt(cls, v):
        """Valida prompt."""
        if isinstance(v, str):
            if len(v) > VALIDATION_CONFIG["max_prompt_length"]:
                raise ValidationError(
                    f"Prompt excede tamanho máximo de "
                    f"{VALIDATION_CONFIG['max_prompt_length']} caracteres"
                )
        else:
            if len(v) > VALIDATION_CONFIG["max_messages"]:
                raise ValidationError(
                    f"Número de mensagens excede máximo de "
                    f"{VALIDATION_CONFIG['max_messages']}"
                )
            
            for msg in v:
                content = ContentValidator(**msg)
                
                if not content.text and not content.images:
                    raise ValidationError(
                        "Mensagem deve ter texto ou imagens"
                    )
        
        return v
    
    @validator("functions")
    def validate_functions(cls, v):
        """Valida funções."""
        if v is None:
            return v
            
        if len(v) > VALIDATION_CONFIG["max_functions"]:
            raise ValidationError(
                f"Número de funções excede máximo de "
                f"{VALIDATION_CONFIG['max_functions']}"
            )
        
        return [FunctionValidator(**func).dict() for func in v]
    
    @validator("function_call")
    def validate_function_call(cls, v):
        """Valida chamada de função."""
        if v is None:
            return v
            
        if isinstance(v, str):
            if v not in ("none", "auto"):
                raise ValidationError(
                    'function_call deve ser "none", "auto" ou um objeto'
                )
        else:
            if "name" not in v:
                raise ValidationError(
                    "function_call deve ter campo 'name'"
                )
        
        return v

# Instância global do sanitizador
input_sanitizer = InputSanitizer() 