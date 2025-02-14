"""
Configurações para o serviço de geração de texto com suporte multi-backend e otimizações vLLM.
"""
import os
from pathlib import Path
import torch

# Diretórios Base
BASE_DIR = Path(__file__).parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
TEMP_DIR = BASE_DIR / "temp"

# Criar diretórios necessários
for dir in [LOGS_DIR, MODELS_DIR, CACHE_DIR, TEMP_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Configurações de API
API_TOKEN = os.getenv("API_TOKEN")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Configurações vLLM Otimizadas
VLLM_CONFIG = {
    "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTIL", "0.95")),
    "max_num_batched_tokens": int(os.getenv("VLLM_MAX_BATCH_TOKENS", "16384")),
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "256")),
    "block_size": int(os.getenv("VLLM_BLOCK_SIZE", "16")),
    "swap_space": int(os.getenv("VLLM_SWAP_SPACE", "4")),  # GB
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "32768")),
    "quantization": {
        "enabled": bool(os.getenv("VLLM_QUANTIZATION_ENABLED", "true").lower() == "true"),
        "bits": int(os.getenv("VLLM_QUANTIZATION_BITS", "4")),
        "group_size": int(os.getenv("VLLM_QUANTIZATION_GROUP_SIZE", "128")),
        "zero_point": bool(os.getenv("VLLM_QUANTIZATION_ZERO_POINT", "true").lower() == "true")
    },
    "cuda_graphs": {
        "enabled": bool(os.getenv("VLLM_CUDA_GRAPHS_ENABLED", "true").lower() == "true"),
        "min_num_tokens": int(os.getenv("VLLM_CUDA_GRAPHS_MIN_TOKENS", "8")),
        "max_num_tokens": int(os.getenv("VLLM_CUDA_GRAPHS_MAX_TOKENS", "2048"))
    },
    "kv_cache": {
        "enabled": bool(os.getenv("VLLM_KV_CACHE_ENABLED", "true").lower() == "true"),
        "max_size_gb": int(os.getenv("VLLM_KV_CACHE_SIZE", "24")),
        "block_size": int(os.getenv("VLLM_KV_BLOCK_SIZE", "16")),
        "cache_policy": os.getenv("VLLM_KV_CACHE_POLICY", "lru"),
        "prefetch": bool(os.getenv("VLLM_KV_PREFETCH", "true").lower() == "true")
    }
}

# Configurações dos Modelos
MODELS_CONFIG = {
    # Modelos Base
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "type": "vllm",
        "max_tokens": 32768,
        "temperature": 0.7,
        "top_p": 0.95,
        "batch_size": 32,
        "quantization": "awq",
        "tensor_parallel_size": VLLM_CONFIG["tensor_parallel_size"],
        "trust_remote_code": True,
        "revision": "main",
        "dtype": "float16",
        "supports_multimodal": False
    },
    "google/gemma-7b-it": {
        "type": "vllm",
        "max_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.95,
        "batch_size": 32,
        "quantization": "awq",
        "tensor_parallel_size": VLLM_CONFIG["tensor_parallel_size"],
        "trust_remote_code": True,
        "revision": "main",
        "dtype": "float16",
        "supports_multimodal": False
    },
    
    # Modelos Multimodais
    "llava-hf/llava-1.5-7b-hf": {
        "type": "vllm",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.95,
        "batch_size": 16,
        "quantization": "awq",
        "tensor_parallel_size": VLLM_CONFIG["tensor_parallel_size"],
        "trust_remote_code": True,
        "revision": "main",
        "dtype": "float16",
        "supports_multimodal": True,
        "vision_encoder": "openai/clip-vit-large-patch14",
        "vision_tower": True
    }
}

# Cache Hierárquico
CACHE_CONFIG = {
    "l1_cache": {
        "type": "memory",
        "max_size": int(os.getenv("CACHE_L1_SIZE", "1000")),
        "ttl": 300,  # 5 minutos
        "policy": "lru"
    },
    "l2_cache": {
        "type": "redis",
        "url": os.getenv("REDIS_URL", "redis://redis:6379"),
        "password": os.getenv("REDIS_PASSWORD", ""),
        "db": int(os.getenv("REDIS_DB", "0")),
        "max_size": int(os.getenv("CACHE_L2_SIZE", "10000")),
        "ttl": 3600,  # 1 hora
        "policy": "lru"
    },
    "l3_cache": {
        "type": "disk",
        "path": str(CACHE_DIR / "l3_cache"),
        "max_size_gb": int(os.getenv("CACHE_L3_SIZE", "100")),
        "ttl": 86400,  # 24 horas
        "policy": "lru"
    },
    "prefetch": {
        "enabled": True,
        "threshold": 0.8,  # Prefetch quando uso > 80%
        "batch_size": 32
    }
}

# Rate Limiting Avançado
RATE_LIMIT_CONFIG = {
    "global": {
        "requests_per_minute": int(os.getenv("RATE_LIMIT_GLOBAL_RPM", "1000")),
        "burst": int(os.getenv("RATE_LIMIT_GLOBAL_BURST", "50"))
    },
    "per_ip": {
        "requests_per_minute": int(os.getenv("RATE_LIMIT_IP_RPM", "100")),
        "burst": int(os.getenv("RATE_LIMIT_IP_BURST", "10"))
    },
    "per_token": {
        "requests_per_minute": int(os.getenv("RATE_LIMIT_TOKEN_RPM", "500")),
        "burst": int(os.getenv("RATE_LIMIT_TOKEN_BURST", "20"))
    },
    "per_model": {
        "mistralai/Mistral-7B-Instruct-v0.2": {
            "requests_per_minute": 100,
            "burst": 10
        },
        "google/gemma-7b-it": {
            "requests_per_minute": 100,
            "burst": 10
        },
        "llava-hf/llava-1.5-7b-hf": {
            "requests_per_minute": 50,
            "burst": 5
        }
    }
}

# Otimizações de Hardware
HARDWARE_CONFIG = {
    "gpu": {
        "enabled": torch.cuda.is_available(),
        "devices": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        "memory_fraction": float(os.getenv("GPU_MEMORY_FRACTION", "0.95")),
        "mixed_precision": True,
        "cudnn_benchmark": True,
        "deterministic": False,
        "tensor_parallel": {
            "enabled": VLLM_CONFIG["tensor_parallel_size"] > 1,
            "size": VLLM_CONFIG["tensor_parallel_size"],
            "mode": "auto"
        }
    },
    "cpu": {
        "num_threads": int(os.getenv("CPU_NUM_THREADS", str(os.cpu_count()))),
        "numa_aware": True,
        "pin_memory": True
    }
}

# Monitoramento
MONITORING_CONFIG = {
    "metrics": {
        "model": {
            "latency": True,
            "throughput": True,
            "memory_usage": True,
            "cache_stats": True,
            "batch_stats": True,
            "token_stats": True
        },
        "hardware": {
            "gpu_utilization": True,
            "gpu_memory": True,
            "gpu_temperature": True,
            "cpu_utilization": True,
            "memory_usage": True,
            "disk_usage": True
        },
        "requests": {
            "latency": True,
            "error_rate": True,
            "success_rate": True,
            "cache_hit_rate": True,
            "rate_limit_hits": True
        }
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "handlers": ["console", "file"],
        "rotation": {
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5
        }
    },
    "tracing": {
        "enabled": True,
        "exporter": "jaeger",
        "sample_rate": 0.1
    },
    "alerting": {
        "enabled": True,
        "providers": ["slack", "email"],
        "thresholds": {
            "error_rate": 0.01,
            "latency_p99": 2000,
            "memory_usage": 0.95,
            "gpu_temperature": 80
        }
    }
}

# Validação e Segurança
SECURITY_CONFIG = {
    "input_validation": {
        "max_prompt_length": 32768,
        "max_output_length": 4096,
        "max_total_tokens": 32768,
        "max_images": 5,
        "max_image_size": 4194304,  # 4MB
        "allowed_image_types": ["image/jpeg", "image/png", "image/webp"],
        "sanitize_html": True,
        "block_scripts": True
    },
    "rate_limiting": RATE_LIMIT_CONFIG,
    "authentication": {
        "required": True,
        "methods": ["bearer_token", "api_key"],
        "token_expiry": 3600,
        "max_tokens_per_user": 100
    },
    "cors": {
        "enabled": True,
        "origins": ["*"],
        "methods": ["*"],
        "headers": ["*"],
        "max_age": 600
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
            "filename": str(LOGS_DIR / "text_service.log"),
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