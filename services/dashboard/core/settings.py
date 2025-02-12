"""
Configurações centralizadas do dashboard usando Dynaconf.
"""
import os
from pathlib import Path
from dynaconf import Dynaconf

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Configuração do Dynaconf
settings = Dynaconf(
    envvar_prefix="DASHBOARD",  # Prefixo para variáveis de ambiente
    settings_files=[           # Arquivos de configuração
        "settings.toml",
        ".secrets.toml",
    ],
    environments=True,        # Habilita ambientes (development, production, etc)
    load_dotenv=True,        # Carrega .env
    default_settings={
        # Servidor
        "app_name": "Dashboard",
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        
        # Segurança
        "secret_key": os.getenv("DASHBOARD_SECRET_KEY", "your-secret-key-here"),
        "algorithm": "HS256",
        "access_token_expire_minutes": 30,
        
        # Banco de dados
        "database_url": os.getenv(
            "DASHBOARD_DATABASE_URL",
            f"sqlite+aiosqlite:///{BASE_DIR}/data/dashboard.db"
        ),
        
        # Cache
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379/0"),
        "cache_ttl": 300,
        
        # Diretórios
        "template_dir": os.path.join(os.path.dirname(__file__), "..", "templates"),
        "static_dir": os.path.join(os.path.dirname(__file__), "..", "static"),
        "upload_dir": os.path.join(BASE_DIR, "uploads"),
        "log_dir": os.path.join(BASE_DIR, "logs"),
        
        # Logging
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        
        # API
        "max_api_keys_per_user": 5,
        
        # Serviços externos
        "services": {
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
        },
        
        # CORS
        "cors_origins": [
            "http://localhost",
            "http://localhost:8000",
            "http://127.0.0.1",
            "http://127.0.0.1:8000",
        ]
    }
)

# Criar diretórios necessários
os.makedirs(os.path.dirname(settings.database_url.replace("sqlite:///", "")), exist_ok=True)
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.log_dir, exist_ok=True) 