"""
Configurações centralizadas do dashboard.
Carrega variáveis de ambiente e fornece configurações para toda a aplicação.
"""

from functools import lru_cache
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configurações da aplicação carregadas de variáveis de ambiente
    """
    # Configurações do servidor
    APP_NAME: str
    DEBUG: bool
    API_V1_STR: str
    
    # Segurança
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    ALGORITHM: str
    
    # Serviços externos
    SERVICES: Dict[str, Dict[str, Any]] = {
        "text": {
            "url": "TEXT_GENERATION_URL",
            "cache_ttl": 300
        },
        "image": {
            "url": "IMAGE_GENERATION_URL", 
            "cache_ttl": 300
        },
        "voice": {
            "url": "VOICE_GENERATION_URL",
            "cache_ttl": 300
        },
        "video": {
            "url": "VIDEO_EDITOR_URL",
            "cache_ttl": 300
        }
    }
    
    # Cache
    REDIS_URL: str
    CACHE_TTL: int
    
    # Banco de dados
    DATABASE_URL: str
    
    class Config:
        case_sensitive = True
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """
    Retorna as configurações da aplicação.
    Usa cache para evitar recarregar o arquivo .env a cada chamada.
    """
    return Settings() 