"""
Configurações para o serviço de geração de vídeo usando FastHunyuan.
"""
import os
from pathlib import Path

# Diretórios
BASE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints/fasthunyuan"
CACHE_DIR = BASE_DIR / "cache"
REFERENCES_DIR = BASE_DIR / "references"

# Configurações do modelo
MODEL_CONFIG = {
    "model_id": "FastVideo/FastHunyuan-diffusers",
    "use_fp8": True,
    "use_nf4": True,
    "use_cpu_offload": True,
    "scheduler_config": {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear",
        "clip_sample": False
    }
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
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "videos-gerados")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Limites e configurações de geração
MAX_VIDEO_LENGTH = int(os.getenv("MAX_VIDEO_LENGTH", "128"))  # frames
MAX_VIDEO_WIDTH = int(os.getenv("MAX_VIDEO_WIDTH", "1280"))  # pixels
MAX_VIDEO_HEIGHT = int(os.getenv("MAX_VIDEO_HEIGHT", "720"))  # pixels
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))  # Para versão quantizada, manter batch=1

# Configurações de GPU
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
TORCH_CUDA_ARCH_LIST = os.getenv("TORCH_CUDA_ARCH_LIST", "7.5")

# Configurações de otimização de memória
ENABLE_VRAM_OPTIMIZATIONS = True
ENABLE_XFORMERS = True
ENABLE_VAE_SLICING = True 