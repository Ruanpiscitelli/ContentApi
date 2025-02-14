"""
Configurações para o serviço de geração de voz com suporte a Fish Audio SDK.
"""
import os
from pathlib import Path
import torch

# Diretórios Base
BASE_DIR = Path(__file__).parent.absolute()
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
TEMP_DIR = BASE_DIR / "temp"

# Criar diretórios necessários
for dir in [LOGS_DIR, MODELS_DIR, CACHE_DIR, TEMP_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Configurações de API
API_TOKEN = os.getenv("API_TOKEN")
FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Configurações Fish Audio
FISH_AUDIO_CONFIG = {
    "base_url": os.getenv("FISH_AUDIO_URL", "https://api.fish.audio"),
    "timeout": 30,
    "max_retries": 3,
    "async_mode": True,
    "api_key": os.getenv("FISH_AUDIO_API_KEY"),
    "models": {
        "default": "fish-speech-v1",
        "available": [
            "fish-speech-v1",
            "fish-speech-v2",
            "fish-speech-multilingual"
        ]
    },
    "reference_audio": {
        "max_duration": 30,  # segundos
        "formats": ["wav", "mp3", "ogg", "flac"],
        "sample_rate": 22050,
        "normalize_audio": True
    }
}

# Configurações FishSpeech Local
FISH_SPEECH_CONFIG = {
    "model_path": str(MODELS_DIR / "fish-speech-1.5" / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
    "use_fp16": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "embedding_cache_size": 1000,
    "sample_rate": 22050,
    "max_wav_length": 30  # segundos
}

# Cache
CACHE_CONFIG = {
    "enable_cache": True,
    "ttl": 3600,  # 1 hora
    "max_size": 1000,  # Máximo de itens no cache
    "embedding_cache": {
        "ttl": 86400,  # 24 horas
        "max_size": 1000  # Máximo de embeddings
    },
    "policies": {
        "eviction": "lru",  # least recently used
        "compression": True,  # Comprime dados antes de armazenar
        "serialization": "pickle"  # Formato de serialização
    },
    "monitoring": {
        "track_hits": True,
        "track_misses": True,
        "track_size": True,
        "track_latency": True
    },
    "cleanup": {
        "interval": 3600,  # Intervalo de limpeza em segundos
        "max_age": 86400  # Idade máxima dos itens em segundos
    }
}

# Redis
REDIS_CONFIG = {
    "url": os.getenv("REDIS_URL", "redis://redis:6379"),
    "db": 0,
    "prefix": "voice_gen:",
    "ttl": 3600,  # 1 hora
    "max_memory": "2gb",
    "policies": {
        "embeddings": {
            "ttl": 86400,  # 24 horas
            "max_size": 1000
        },
        "audio": {
            "ttl": 3600,  # 1 hora
            "max_size": 100
        }
    },
    "pool": {
        "max_connections": 10,
        "timeout": 20,
        "retry_on_timeout": True
    },
    "ssl": {
        "enabled": False,
        "cert_reqs": "required",
        "ca_certs": None
    }
}

# MinIO Storage
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "minio:9000"),
    "access_key": os.getenv("MINIO_ACCESS_KEY"),
    "secret_key": os.getenv("MINIO_SECRET_KEY"),
    "secure": True,
    "bucket_name": "audio-files",
    "url_expiry": 7200,  # 2 horas
    "file_types": {
        "audio": ["wav", "mp3", "ogg", "flac"],
        "models": ["pt", "bin", "onnx"]
    }
}

# Limites e Configurações
GENERATION_LIMITS = {
    "max_text_length": 1000,
    "max_audio_duration": 300,  # 5 minutos
    "max_concurrent_requests": 50,
    "max_batch_size": 16,
    "timeout": 60,
    "rate_limit": {
        "requests_per_minute": 100,
        "burst": 20
    }
}

# Otimizações
OPTIMIZATION_CONFIG = {
    "use_mixed_precision": True,
    "pin_memory": True,
    "num_workers": 4,
    "batch_processing": True,
    "prefetch_factor": 2,
    "cuda_graphs": True
}

# Templates de Voz
VOICE_TEMPLATES = {
    "default": {
        "speed": 1.0,
        "pitch": 0.0,
        "energy": 1.0,
        "language": "auto"
    },
    "fast": {
        "speed": 1.3,
        "pitch": 0.0,
        "energy": 1.2,
        "language": "auto"
    },
    "slow": {
        "speed": 0.8,
        "pitch": -0.2,
        "energy": 0.9,
        "language": "auto"
    }
}

# Logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "voice_service.log"),
            "formatter": "json",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "level": "INFO"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO"
    }
}

# Monitoramento
MONITORING_CONFIG = {
    "enable_prometheus": True,
    "metrics_port": 8001,
    "profile_inference": True,
    "metrics": {
        "latency_window": 100,
        "memory_window": 60,
        "audio_quality_metrics": True,
        "voice_similarity_metrics": True,
        "custom_metrics": {
            "generation_time": True,
            "cache_hit_rate": True,
            "voice_clone_success_rate": True,
            "audio_length_distribution": True
        }
    },
    "alerting": {
        "enable_alerts": True,
        "latency_threshold_ms": 2000,
        "error_rate_threshold": 0.01,
        "memory_threshold": 0.95
    }
}

# Configurações de Backend
BACKEND_CONFIG = {
    "fallback_enabled": True,
    "fallback_timeout": 10,  # segundos
    "retry_attempts": 3,
    "retry_delay": 2,  # segundos
    "preferred_backend": "local",  # "local" ou "api"
    "cache_embeddings": True,
    "cache_results": True,
    "monitoring": {
        "track_latency": True,
        "track_errors": True,
        "track_cache_hits": True
    }
} 