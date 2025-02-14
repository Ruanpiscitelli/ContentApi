"""
Este módulo contém as configurações do dashboard.
"""

import os
from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Configurações do servidor
HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
PORT = int(os.getenv("DASHBOARD_PORT", "80"))
DEBUG = os.getenv("DASHBOARD_DEBUG", "False").lower() == "true"

# Configurações de segurança
SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configurações do banco de dados
DATABASE_URL = os.getenv(
    "DASHBOARD_DATABASE_URL",
    f"sqlite+aiosqlite:///{BASE_DIR}/data/dashboard.db"
)

# Configurações de template
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Configurações de API
MAX_API_KEYS_PER_USER = 5
API_KEYS_DIR = os.path.join(os.path.dirname(__file__), "api_keys")

# Configurações de log
LOG_LEVEL = os.getenv("DASHBOARD_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(BASE_DIR, "logs", "dashboard.log")

# Configurações de cache
CACHE_TTL = 300  # 5 minutos em segundos

# Configurações de monitoramento
SYSTEM_STATS_INTERVAL = 5  # segundos
MAX_LOGS_DISPLAY = 100

# Configurações de autenticação
MIN_PASSWORD_LENGTH = 8
PASSWORD_REGEX = r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$"  # Pelo menos 8 caracteres, uma letra e um número

# Configurações de sessão
SESSION_COOKIE_NAME = "dashboard_session"
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"

# Configurações de CORS
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

# Configurações de rate limiting
RATE_LIMIT_REQUESTS = 100  # requisições
RATE_LIMIT_WINDOW = 60  # segundos

# Configurações de upload
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB

# Configurações de email (para futuras implementações)
EMAIL_ENABLED = False
EMAIL_HOST = os.getenv("DASHBOARD_EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("DASHBOARD_EMAIL_PORT", 587))
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.getenv("DASHBOARD_EMAIL_USER", "")
EMAIL_HOST_PASSWORD = os.getenv("DASHBOARD_EMAIL_PASSWORD", "")
DEFAULT_FROM_EMAIL = os.getenv("DASHBOARD_FROM_EMAIL", "noreply@example.com")

# Configurações dos serviços integrados
API_CONFIG = {
    "services": {
        "text_generation": os.getenv("TEXT_GENERATION_URL", "http://localhost:8001"),
        "image_generation": os.getenv("IMAGE_GENERATION_URL", "http://localhost:8002"),
        "voice_generation": os.getenv("VOICE_GENERATION_URL", "http://localhost:8003"),
        "video_editor": os.getenv("VIDEO_EDITOR_URL", "http://localhost:8004")
    },
    "metrics": {
        "enabled": True,
        "collection_interval": 60,  # segundos
        "retention_days": 7
    },
    "cors_origins": CORS_ORIGINS
}

# Criação de diretórios necessários
os.makedirs(os.path.dirname(DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)
os.makedirs(API_KEYS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True) 