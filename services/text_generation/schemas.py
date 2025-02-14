"""
Schemas Pydantic para o serviço de geração de texto.
"""
from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import BaseModel, Field, root_validator, validator
import time
import hashlib
from config import SECURITY_CONFIG

class ImageContent(BaseModel):
    """Conteúdo de imagem para entrada multimodal."""
    type: Literal["image"] = "image"
    image_url: Optional[str] = None
    image_data: Optional[str] = None  # Base64
    resize_mode: Optional[str] = "pad"  # pad, crop, resize
    image_size: Optional[int] = 224

    @validator("image_data")
    def validate_image_data(cls, v):
        if v and len(v) > SECURITY_CONFIG["input_validation"]["max_image_size"]:
            raise ValueError("Imagem muito grande")
        return v

class TextContent(BaseModel):
    """Conteúdo de texto para entrada multimodal."""
    type: Literal["text"] = "text"
    text: str

class FunctionDefinition(BaseModel):
    """Definição de função para function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: Optional[List[str]] = None

class ToolChoice(BaseModel):
    """Escolha de ferramenta para function calling."""
    type: Literal["function"] = "function"
    function: Dict[str, Any]

class Tool(BaseModel):
    """Ferramenta disponível para function calling."""
    type: Literal["function"] = "function"
    function: FunctionDefinition

class FunctionCall(BaseModel):
    """Chamada de função."""
    name: str
    arguments: str
    thoughts: Optional[str] = None

class ChatMessage(BaseModel):
    """Mensagem de chat com suporte a conteúdo multimodal."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Union[TextContent, ImageContent]]]
    name: Optional[str] = None
    tool_calls: Optional[List[FunctionCall]] = None

    @validator("content")
    def validate_content(cls, v):
        if isinstance(v, list):
            # Valida número máximo de imagens
            num_images = sum(1 for item in v if isinstance(item, dict) and item.get("type") == "image")
            if num_images > SECURITY_CONFIG["input_validation"]["max_images"]:
                raise ValueError(f"Máximo de {SECURITY_CONFIG['input_validation']['max_images']} imagens permitido")
        return v

class ChatCompletionRequest(BaseModel):
    """Requisição de chat completion com suporte a multimodal e function calling."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    max_tokens: Optional[int] = Field(default=None)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None

    @root_validator
    def validate_all(cls, values):
        # Valida tamanho total do prompt
        messages = values.get("messages", [])
        total_length = sum(
            len(str(msg.content)) for msg in messages
        )
        if total_length > SECURITY_CONFIG["input_validation"]["max_prompt_length"]:
            raise ValueError(f"Prompt muito longo (máximo: {SECURITY_CONFIG['input_validation']['max_prompt_length']} caracteres)")
        
        # Valida max_tokens
        max_tokens = values.get("max_tokens")
        if max_tokens and max_tokens > SECURITY_CONFIG["input_validation"]["max_output_length"]:
            raise ValueError(f"max_tokens não pode exceder {SECURITY_CONFIG['input_validation']['max_output_length']}")
        
        # Valida número de ferramentas
        tools = values.get("tools", [])
        if tools and len(tools) > 16:
            raise ValueError("Máximo de 16 ferramentas permitido")
        
        return values

class ChatCompletionResponseChoice(BaseModel):
    """Escolha de resposta do chat."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    """Informações de uso de tokens."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    """Resposta de chat completion."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class ErrorResponse(BaseModel):
    """Resposta de erro."""
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int 