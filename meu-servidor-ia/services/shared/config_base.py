"""
Configurações base compartilhadas entre todos os serviços.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import torch
import logging

logger = logging.getLogger(__name__)

class BaseServiceConfig(BaseModel):
    """Configurações base para todos os serviços."""
    
    # Diretórios Base
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    LOGS_DIR: Path = Field(default_factory=lambda: Path("logs"))
    MODELS_DIR: Path = Field(default_factory=lambda: Path("models"))
    CACHE_DIR: Path = Field(default_factory=lambda: Path("cache"))
    TEMP_DIR: Path = Field(default_factory=lambda: Path("temp"))
    
    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_TOKEN: Optional[str] = Field(default=None)
    DEBUG: bool = Field(default=False)
    
    # Hardware
    HARDWARE_CONFIG: Dict[str, Any] = Field(
        default_factory=lambda: {
            "gpu": {
                "enabled": torch.cuda.is_available(),
                "devices": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
                "memory_fraction": float(os.getenv("GPU_MEMORY_FRACTION", "0.95")),
                "mixed_precision": True,
                "cudnn_benchmark": True,
                "deterministic": False
            },
            "cpu": {
                "num_threads": int(os.getenv("CPU_NUM_THREADS", str(os.cpu_count()))),
                "numa_aware": True,
                "pin_memory": True
            }
        }
    )
    
    # Cache
    CACHE_CONFIG: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enable_cache": True,
            "ttl": 3600,  # 1 hora
            "max_size": 1000,  # Máximo de itens no cache
            "policies": {
                "eviction": "lru",
                "compression": True,
                "serialization": "pickle"
            },
            "monitoring": {
                "track_hits": True,
                "track_misses": True,
                "track_size": True,
                "track_latency": True
            }
        }
    )
    
    # Redis
    REDIS_CONFIG: Dict[str, Any] = Field(
        default_factory=lambda: {
            "url": os.getenv("REDIS_URL", "redis://redis:6379"),
            "db": 0,
            "prefix": "service:",
            "ttl": 3600,
            "max_memory": "2gb",
            "pool": {
                "max_connections": 10,
                "timeout": 20,
                "retry_on_timeout": True
            }
        }
    )
    
    # Storage
    STORAGE_CONFIG: Dict[str, Any] = Field(
        default_factory=lambda: {
            "minio": {
                "endpoint": os.getenv("MINIO_ENDPOINT", "minio:9000"),
                "access_key": os.getenv("MINIO_ACCESS_KEY"),
                "secret_key": os.getenv("MINIO_SECRET_KEY"),
                "secure": True,
                "bucket_name": "default",
                "url_expiry": 7200  # 2 horas
            }
        }
    )
    
    # Monitoramento
    MONITORING_CONFIG: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enable_prometheus": True,
            "metrics_port": 8001,
            "profile_inference": True,
            "metrics": {
                "latency_window": 100,
                "memory_window": 60,
                "custom_metrics": {
                    "generation_time": True,
                    "cache_hit_rate": True,
                    "model_load_time": True,
                    "request_queue_size": True
                }
            },
            "alerting": {
                "enable_alerts": True,
                "latency_threshold_ms": 2000,
                "error_rate_threshold": 0.01,
                "memory_threshold": 0.95
            }
        }
    )
    
    # Logging
    LOGGING_CONFIG: Dict[str, Any] = Field(
        default_factory=lambda: {
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
                    "filename": "service.log",
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
    )
    
    # Rate Limiting
    RATE_LIMIT_CONFIG: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "requests_per_minute": 100,
            "burst": 20
        }
    )
    
    def setup_directories(self) -> None:
        """Cria os diretórios necessários."""
        for dir_path in [self.LOGS_DIR, self.MODELS_DIR, self.CACHE_DIR, self.TEMP_DIR]:
            full_path = self.BASE_DIR / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self) -> None:
        """Configura o logging do serviço."""
        import logging.config
        
        # Atualiza o caminho do arquivo de log
        self.LOGGING_CONFIG["handlers"]["file"]["filename"] = str(
            self.LOGS_DIR / "service.log"
        )
        
        # Aplica a configuração
        logging.config.dictConfig(self.LOGGING_CONFIG)
        
    def validate_gpu_requirements(
        self,
        required_memory_gb: float,
        required_compute_capability: Optional[float] = None
    ) -> bool:
        """
        Valida requisitos de GPU.
        
        Args:
            required_memory_gb: Memória GPU necessária em GB
            required_compute_capability: Capacidade de computação mínima necessária
            
        Returns:
            bool: True se os requisitos são atendidos
        """
        if not torch.cuda.is_available():
            logger.warning("GPU não disponível")
            return False
            
        # Verifica memória
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if total_memory < required_memory_gb:
            logger.warning(
                f"Memória GPU insuficiente. Necessário: {required_memory_gb}GB, "
                f"Disponível: {total_memory:.1f}GB"
            )
            return False
            
        # Verifica compute capability
        if required_compute_capability:
            cc = float(f"{torch.cuda.get_device_properties(device).major}."
                      f"{torch.cuda.get_device_properties(device).minor}")
            if cc < required_compute_capability:
                logger.warning(
                    f"Compute capability insuficiente. Necessário: {required_compute_capability}, "
                    f"Disponível: {cc}"
                )
                return False
                
        return True
        
    class Config:
        """Configurações do modelo Pydantic."""
        arbitrary_types_allowed = True 