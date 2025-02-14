"""
Exceções customizadas para o serviço de voz.
"""

class VoiceGenerationError(Exception):
    """Erro base para geração de voz."""
    pass

class BackendError(VoiceGenerationError):
    """Erro no backend de geração de voz."""
    pass

class APIError(BackendError):
    """Erro na API do Fish Audio."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"API error {status_code}: {message}")

class ModelError(BackendError):
    """Erro no modelo local."""
    pass

class ValidationError(VoiceGenerationError):
    """Erro de validação de parâmetros."""
    pass

class AudioProcessingError(VoiceGenerationError):
    """Erro no processamento de áudio."""
    pass

class CacheError(VoiceGenerationError):
    """Erro no sistema de cache."""
    pass

class WebSocketError(VoiceGenerationError):
    """Erro na conexão WebSocket."""
    pass

class RateLimitError(VoiceGenerationError):
    """Erro de limite de requisições."""
    pass

class AuthenticationError(VoiceGenerationError):
    """Erro de autenticação."""
    pass

class ResourceNotFoundError(VoiceGenerationError):
    """Recurso não encontrado."""
    pass

class ConfigurationError(VoiceGenerationError):
    """Erro de configuração."""
    pass 