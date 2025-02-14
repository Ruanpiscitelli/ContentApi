"""
Schemas Pydantic para validação de dados.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator, constr
from enum import Enum
import mimetypes
from exceptions import ValidationError

class Language(str, Enum):
    """Idiomas suportados."""
    AUTO = "auto"
    EN = "en"
    ZH = "zh"
    JA = "ja"
    KO = "ko"
    FR = "fr"
    DE = "de"
    AR = "ar"
    ES = "es"
    PT = "pt"
    RU = "ru"
    IT = "it"
    NL = "nl"
    PL = "pl"

class Emotion(str, Enum):
    """Emoções suportadas."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISE = "surprise"
    EXCITED = "excited"
    CALM = "calm"

class AudioFormat(str, Enum):
    """Formatos de áudio suportados."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"

class VoiceParameters(BaseModel):
    """Parâmetros para geração de voz."""
    language: Language = Field(
        default=Language.AUTO,
        description="Idioma do texto"
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Velocidade da fala (0.5 a 2.0)"
    )
    pitch: float = Field(
        default=0.0,
        ge=-12.0,
        le=12.0,
        description="Altura da voz (-12 a 12)"
    )
    energy: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Energia da voz (0.5 a 2.0)"
    )
    prompt_text: Optional[constr(max_length=1000)] = Field(
        default=None,
        description="Texto de referência para estilo"
    )
    emotion: Optional[Emotion] = Field(
        default=None,
        description="Emoção desejada"
    )
    speaker: Optional[constr(regex=r'^[a-zA-Z0-9_-]+$')] = Field(
        default=None,
        description="ID do speaker (apenas letras, números, _ e -)"
    )
    output_format: AudioFormat = Field(
        default=AudioFormat.WAV,
        description="Formato do áudio de saída"
    )

    @validator("prompt_text")
    def validate_prompt(cls, v):
        if v is not None:
            v = v.strip()
            if not v:
                return None
            if len(v.split()) > 100:  # Máximo de 100 palavras
                raise ValidationError("Prompt text muito longo (máximo 100 palavras)")
        return v

    @validator("speaker")
    def validate_speaker(cls, v):
        if v is not None and len(v) > 50:
            raise ValidationError("ID do speaker muito longo (máximo 50 caracteres)")
        return v

class VoiceRequest(BaseModel):
    """Requisição de geração de voz."""
    texto: constr(min_length=1, max_length=5000) = Field(
        ...,
        description="Texto para sintetizar"
    )
    template_id: Optional[constr(regex=r'^[a-zA-Z0-9_-]+$')] = Field(
        default=None,
        description="ID do template de voz"
    )
    tempo_max: int = Field(
        default=1200,
        ge=1,
        le=3600,
        description="Duração máxima em segundos"
    )
    parametros: VoiceParameters = Field(
        default_factory=VoiceParameters,
        description="Parâmetros de geração"
    )
    use_voice_clone: bool = Field(
        default=False,
        description="Usar clonagem de voz"
    )
    reference_audio_mime: Optional[str] = Field(
        default=None,
        description="MIME type do áudio de referência"
    )

    @validator("texto")
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValidationError("Texto não pode estar vazio")
        if len(v.split()) > 1000:  # Máximo de 1000 palavras
            raise ValidationError("Texto muito longo (máximo 1000 palavras)")
        return v

    @validator("reference_audio_mime")
    def validate_mime(cls, v, values):
        if values.get("use_voice_clone") and not v:
            raise ValidationError("MIME type obrigatório para clonagem de voz")
        if v and not mimetypes.guess_extension(v):
            raise ValidationError(f"MIME type inválido: {v}")
        return v

class VoiceResponse(BaseModel):
    """Resposta da geração de voz."""
    status: str = Field(..., description="Status da geração")
    job_id: str = Field(..., description="ID do job")
    message: str = Field(..., description="Mensagem descritiva")
    minio_url: str = Field(..., description="URL do áudio no MinIO")
    process_time: float = Field(
        ...,
        description="Tempo de processamento em segundos"
    )
    chunks_processados: int = Field(
        ...,
        ge=0,
        description="Número de chunks processados"
    )
    chunks_total: Optional[int] = Field(
        None,
        ge=0,
        description="Número total de chunks"
    )
    audio_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Informações do áudio gerado"
    )

    @validator("minio_url")
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValidationError("URL inválida")
        return v

class HealthResponse(BaseModel):
    """Resposta do health check."""
    status: str = Field(..., description="Status do serviço")
    version: str = Field(..., description="Versão do serviço")
    dependencies: Dict[str, str] = Field(
        ...,
        description="Versões das dependências"
    )
    gpu_available: bool = Field(
        ...,
        description="Disponibilidade de GPU"
    )
    cuda_version: Optional[str] = Field(
        None,
        description="Versão do CUDA"
    )
    error: Optional[str] = Field(
        None,
        description="Mensagem de erro"
    )
    last_error_time: Optional[float] = Field(
        None,
        description="Timestamp do último erro"
    )
    uptime: float = Field(
        ...,
        ge=0,
        description="Tempo de execução em segundos"
    )

class Template(BaseModel):
    """Template de voz."""
    template_id: constr(regex=r'^[a-zA-Z0-9_-]+$') = Field(
        ...,
        description="ID do template"
    )
    description: Optional[constr(max_length=500)] = Field(
        None,
        description="Descrição do template"
    )
    tempo_max: int = Field(
        default=1200,
        ge=1,
        le=3600,
        description="Duração máxima em segundos"
    )
    parametros: VoiceParameters = Field(
        ...,
        description="Parâmetros do template"
    )
    created_at: float = Field(
        ...,
        description="Timestamp de criação"
    )
    updated_at: Optional[float] = Field(
        None,
        description="Timestamp da última atualização"
    )

    @validator("template_id")
    def validate_id(cls, v):
        if not v.isalnum() and not all(c in "_-" for c in v if not c.isalnum()):
            raise ValidationError("ID deve conter apenas letras, números, _ e -")
        if len(v) > 50:
            raise ValidationError("ID muito longo (máximo 50 caracteres)")
        return v 