"""
Configurações para o serviço de geração de voz usando Fish Speech.
"""
import os
from pathlib import Path

# Diretórios
BASE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints/fish-speech-1.5"
CACHE_DIR = BASE_DIR / "cache"
REFERENCES_DIR = BASE_DIR / "references"

# Configurações do modelo
MODEL_CONFIG = {
    "llama_checkpoint_path": str(CHECKPOINTS_DIR),
    "decoder_checkpoint_path": str(CHECKPOINTS_DIR / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
    "decoder_config_name": "firefly_gan_vq",
    "compile": True,  # Habilita compilação CUDA para melhor performance
}

# Configurações de API
API_TOKEN = os.getenv("API_TOKEN")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))

# Configurações de cache
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hora

# Configurações de MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "audio-files")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Limites e configurações de geração
MAX_AUDIO_LENGTH = int(os.getenv("MAX_AUDIO_LENGTH", "1200"))  # 20 minutos
MAX_REFERENCE_LENGTH = int(os.getenv("MAX_REFERENCE_LENGTH", "30"))  # 30 segundos
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))

# Configurações de GPU
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
TORCH_CUDA_ARCH_LIST = os.getenv("TORCH_CUDA_ARCH_LIST", "7.5") 