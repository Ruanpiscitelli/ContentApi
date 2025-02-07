"""
Configurações para o serviço de geração de voz usando Fish Speech.
Inclui todas as configurações necessárias para o modelo e otimizações.
"""
import os
from pathlib import Path
import torch

# Diretórios Base
BASE_DIR = Path(__file__).parent.absolute()
MODEL_DIR = BASE_DIR / "models/fish-speech-1.5"
CACHE_DIR = BASE_DIR / "cache"
REFERENCES_DIR = BASE_DIR / "references"
TEMP_DIR = BASE_DIR / "temp"

# Configurações do Fish Speech
FISH_SPEECH_CONFIG = {
    "model_path": BASE_DIR / "models/fish-speech-1.5",
    "use_fp16": True,
    "batch_size": 16,
    "max_wav_length": 1000000,
    "speaker_embedding_dim": 256,
    "supported_languages": ["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "pt"],
    "cache_embeddings": True,
    "embedding_cache_size": 1000,
    "sample_rate": 16000,
    "max_reference_duration": 30,  # segundos
    "voice_clone": {
        "enabled": True,
        "reference_audio_max_size": 5 * 1024 * 1024,  # 5MB
        "supported_formats": ["wav", "mp3", "ogg", "flac"]
    }
}

# Configurações de API
API_TOKEN = os.getenv("API_TOKEN")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))

# Rate Limiting
RATE_LIMIT = {
    "requests_per_minute": int(os.getenv("RATE_LIMIT_RPM", "60")),
    "burst_size": int(os.getenv("RATE_LIMIT_BURST", "10"))
}

# Cache Redis
REDIS_CONFIG = {
    "url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "password": os.getenv("REDIS_PASSWORD", None),
    "db": int(os.getenv("REDIS_DB", "0"))
}

# MinIO Storage
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    "secure": os.getenv("MINIO_SECURE", "false").lower() == "true",
    "bucket_name": os.getenv("MINIO_BUCKET", "audio-files")
}

# Limites e Configurações de Geração
GENERATION_LIMITS = {
    "max_text_length": 5000,
    "max_audio_duration": 1200,  # segundos
    "max_batch_size": 10
}

# Configurações de GPU
GPU_CONFIG = {
    "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
    "memory_growth": True,
    "per_process_memory_fraction": 0.9,
    "mixed_precision": True,
    "optimize_transformers": True
}

# Configurações de Logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/voice_service.log",
            "formatter": "default",
            "level": "INFO"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO"
    }
}

# Configurações de Monitoramento
MONITORING_CONFIG = {
    "enable_prometheus": True,
    "metrics_port": int(os.getenv("METRICS_PORT", "9090")),
    "enable_tensorboard": True,
    "tensorboard_dir": "logs/tensorboard",
    "profile_inference": True
}

# Configurações de Otimização
OPTIMIZATION_CONFIG = {
    "num_threads": int(os.getenv("NUM_THREADS", "4")),
    "use_mixed_precision": True,
    "pin_memory": True,
    "optimize_cuda_graphs": True,
    "enable_cudnn_benchmark": True,
    "enable_cuda_malloc_async": True,
    "batch_processing": {
        "enabled": True,
        "optimal_batch_size": 16,
        "max_batch_size": 32,
        "min_batch_size": 4,
        "max_batch_wait_time": 0.2
    },
    "memory_optimization": {
        "use_fp16": True,
        "use_tf32": True,
        "use_flash_attention": True,
        "use_memory_efficient_attention": True,
        "optimize_cudnn": True,
        "optimize_cuda_graphs": True
    },
    "performance": {
        "num_workers": 4,
        "prefetch_factor": 2,
        "pin_memory": True,
        "non_blocking": True,
        "use_cuda_events": True
    }
}

# Configurações de Cache
CACHE_CONFIG = {
    "enable_cache": True,
    "ttl": 3600,  # 1 hora
    "max_size": 1000,  # número máximo de itens em cache
    "embedding_cache": {
        "enabled": True,
        "max_size": 500,
        "ttl": 7200  # 2 horas
    }
}

# Configurações de Segurança
SECURITY_CONFIG = {
    "rate_limit_enabled": True,
    "enable_cors": True,
    "allowed_origins": ["*"],
    "allowed_methods": ["*"],
    "allowed_headers": ["*"],
    "allow_credentials": True,
    "max_age": 600,
    "token_expiration": 3600,
    "max_token_uses": 1000
} 