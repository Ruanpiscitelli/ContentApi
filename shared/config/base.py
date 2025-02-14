"""
Este módulo contém configurações base compartilhadas entre os serviços.
"""

import os
from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Configurações comuns de servidor
DEFAULT_HOST = "0.0.0.0"
DEFAULT_DEBUG = False

# Configurações de segurança
DEFAULT_SECRET_KEY = "your-secret-key-here"
DEFAULT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configurações de log
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Configurações de upload
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB

# Configurações de CORS padrão
DEFAULT_CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

# Configurações de rate limiting
DEFAULT_RATE_LIMIT_REQUESTS = 100  # requisições
DEFAULT_RATE_LIMIT_WINDOW = 60  # segundos

# Portas padrão dos serviços
SERVICE_PORTS = {
    "dashboard": 8000,
    "text_generation": 8001,
    "image_generation": 8002,
    "voice_generation": 8003,
    "video_editor": 8004
}

# Criação de diretórios necessários
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True) 