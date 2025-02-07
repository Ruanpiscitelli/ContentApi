"""
Configurações para o serviço de geração de texto usando vLLM.
"""
import os
from pathlib import Path

# Diretórios Base
BASE_DIR = Path(__file__).parent.absolute()
LOGS_DIR = BASE_DIR / "logs"

# Criar diretórios se não existirem
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configurações de API
API_TOKEN = os.getenv("API_TOKEN")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))

# Configurações vLLM
VLLM_CONFIG = {
    "default_model": "mistralai/Mistral-7B-v0.1",
    "available_models": [
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "tiiuae/falcon-7b",
        "tiiuae/falcon-40b",
        "mosaicml/mpt-7b",
        "mosaicml/mpt-30b"
    ],
    "model_config": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "trust_remote_code": True,
        "dtype": "float16",
        "tensor_parallel_size": 1
    },
    "server_config": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_version": "v1",
        "served_model": None,  # Será definido em runtime
        "engine_use_ray": False,
        "disable_log_requests": False,
        "max_retries": 3,
        "retry_timeout_seconds": 1.0
    },
    "sampling_config": {
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "best_of": 1,
        "top_k": -1,
        "use_beam_search": False,
        "stop": None,
        "ignore_eos": False,
        "max_tokens_per_step": 16
    }
}

# Rate Limiting
RATE_LIMIT = {
    "requests_per_minute": int(os.getenv("RATE_LIMIT_RPM", "60")),
    "burst_size": int(os.getenv("RATE_LIMIT_BURST", "10")),
    "per_model_limits": {
        "mistralai/Mistral-7B-v0.1": 100,
        "meta-llama/Llama-2-7b-chat-hf": 100,
        "meta-llama/Llama-2-13b-chat-hf": 50,
        "meta-llama/Llama-2-70b-chat-hf": 20,
        "tiiuae/falcon-7b": 100,
        "tiiuae/falcon-40b": 30,
        "mosaicml/mpt-7b": 100,
        "mosaicml/mpt-30b": 40
    }
}

# Cache etcd
CACHE_CONFIG = {
    "type": "etcd",
    "endpoints": [os.getenv("ETCD_ENDPOINT", "http://etcd-cache-service:2379")],
    "prefix": "content-api/",
    "ttl": 3600,  # 1 hora em segundos
    "max_retries": 3,
    "retry_interval": 1.0
}

# Limites e Configurações de Geração
GENERATION_LIMITS = {
    "max_prompt_length": 4096,
    "max_tokens_output": 2048,
    "max_batch_size": 32,
    "timeout": 30,  # 30 segundos
    "max_requests_per_minute": 1000,
    "max_concurrent_requests": 100,
    "per_model_batch_size": {
        "mistralai/Mistral-7B-v0.1": 32,
        "meta-llama/Llama-2-7b-chat-hf": 32,
        "meta-llama/Llama-2-13b-chat-hf": 16,
        "meta-llama/Llama-2-70b-chat-hf": 4,
        "tiiuae/falcon-7b": 32,
        "tiiuae/falcon-40b": 8,
        "mosaicml/mpt-7b": 32,
        "mosaicml/mpt-30b": 8
    }
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
            "filename": str(LOGS_DIR / "text_service.log"),
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
    "profile_inference": True,
    "metrics": {
        "latency_window": 100,  # Últimas N requisições para cálculo de latência
        "memory_window": 60,    # Segundos para média de uso de memória
        "collect_gpu_metrics": True,
        "collect_model_metrics": True,
        "collect_system_metrics": True
    }
} 