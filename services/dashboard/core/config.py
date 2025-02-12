"""
Configurações centralizadas do dashboard.
"""
import os
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseSettings

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """Configurações da aplicação"""
    # Servidor
    APP_NAME: str = "Dashboard"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Segurança
    SECRET_KEY: str = os.getenv("DASHBOARD_SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Banco de dados
    DATABASE_URL: str = os.getenv(
        "DASHBOARD_DATABASE_URL",
        f"sqlite+aiosqlite:///{BASE_DIR}/data/dashboard.db"
    )
    
    # Cache
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    CACHE_TTL: int = 300
    
    # Diretórios
    TEMPLATE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "templates")
    STATIC_DIR: str = os.path.join(os.path.dirname(__file__), "..", "static")
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    LOG_DIR: str = os.path.join(BASE_DIR, "logs")
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API
    MAX_API_KEYS_PER_USER: int = 5
    
    # Serviços externos
    SERVICES: Dict[str, Dict[str, Any]] = {
        "text": {
            "url": os.getenv("TEXT_GENERATION_URL", "http://text-generator:8000"),
            "cache_ttl": 300
        },
        "image": {
            "url": os.getenv("IMAGE_GENERATION_URL", "http://image-generator:8000"),
            "cache_ttl": 300
        },
        "voice": {
            "url": os.getenv("VOICE_GENERATION_URL", "http://voice-generator:8000"),
            "cache_ttl": 300
        },
        "video": {
            "url": os.getenv("VIDEO_EDITOR_URL", "http://video-editor:8000"),
            "cache_ttl": 300
        }
    }
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ]
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Instância global das configurações
settings = Settings()

# Criar diretórios necessários
os.makedirs(os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True) 