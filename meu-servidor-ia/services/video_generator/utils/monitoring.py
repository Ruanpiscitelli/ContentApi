"""
Monitoramento e métricas para o gerador de vídeos.
"""
import time
import logging
from typing import Dict, Any, Optional
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import torch

logger = logging.getLogger(__name__)

# Métricas Prometheus
VIDEO_GENERATION_TIME = Histogram(
    'video_generation_seconds',
    'Tempo de geração de vídeo',
    buckets=[1, 5, 10, 30, 60, 120, 300]
)

VIDEO_GENERATION_ERRORS = Counter(
    'video_generation_errors_total',
    'Total de erros na geração de vídeos',
    ['error_type']
)

BATCH_SIZE = Gauge(
    'video_batch_size',
    'Tamanho atual do batch de geração'
)

GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'Uso de memória GPU em bytes'
)

CPU_MEMORY_USAGE = Gauge(
    'process_memory_usage_bytes',
    'Uso de memória do processo em bytes'
)

QUEUE_SIZE = Gauge(
    'video_generation_queue_size',
    'Número de vídeos na fila de geração'
)

GENERATION_SUMMARY = Summary(
    'video_generation_summary',
    'Sumário de métricas de geração de vídeos',
    ['resolution']
)

def track_time(metric: Histogram):
    """
    Decorator para medir tempo de execução.
    
    Args:
        metric: Métrica Prometheus para registrar tempo
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.observe(duration)
        return wrapper
    return decorator

def track_errors(error_type: str):
    """
    Decorator para contar erros.
    
    Args:
        error_type: Tipo do erro para categorização
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                VIDEO_GENERATION_ERRORS.labels(error_type=error_type).inc()
                raise
        return wrapper
    return decorator

def update_resource_metrics():
    """Atualiza métricas de recursos (CPU, GPU, memória)."""
    try:
        # Métricas de CPU
        process = psutil.Process()
        CPU_MEMORY_USAGE.set(process.memory_info().rss)
        
        # Métricas de GPU
        if torch.cuda.is_available():
            GPU_MEMORY_USAGE.set(torch.cuda.memory_allocated())
            
    except Exception as e:
        logger.error(f"Erro ao atualizar métricas de recursos: {e}")

def record_generation_stats(
    resolution: str,
    duration: float,
    success: bool,
    error: Optional[str] = None
):
    """
    Registra estatísticas de uma geração.
    
    Args:
        resolution: Resolução do vídeo (ex: "1280x720")
        duration: Duração da geração em segundos
        success: Se a geração foi bem sucedida
        error: Mensagem de erro se houver falha
    """
    try:
        # Registra tempo
        GENERATION_SUMMARY.labels(resolution=resolution).observe(duration)
        
        # Registra erro se houver
        if not success and error:
            VIDEO_GENERATION_ERRORS.labels(
                error_type=error.split(':')[0]
            ).inc()
            
    except Exception as e:
        logger.error(f"Erro ao registrar estatísticas: {e}")

def get_metrics_snapshot() -> Dict[str, Any]:
    """
    Retorna snapshot atual das métricas.
    
    Returns:
        Dict[str, Any]: Dicionário com métricas atuais
    """
    metrics = {
        "generation": {
            "total": GENERATION_SUMMARY._sum.get(),
            "count": GENERATION_SUMMARY._count.get(),
            "errors": {
                label: counter._value.get()
                for label, counter in VIDEO_GENERATION_ERRORS._metrics.items()
            }
        },
        "resources": {
            "cpu_memory": CPU_MEMORY_USAGE._value.get(),
            "queue_size": QUEUE_SIZE._value.get()
        }
    }
    
    if torch.cuda.is_available():
        metrics["resources"]["gpu_memory"] = GPU_MEMORY_USAGE._value.get()
        
    return metrics

class MetricsContext:
    """Contexto para coleta automática de métricas."""
    
    def __init__(
        self,
        resolution: str,
        track_resources: bool = True
    ):
        self.resolution = resolution
        self.track_resources = track_resources
        self.start_time = None
        self.error = None
        
    def __enter__(self):
        self.start_time = time.time()
        if self.track_resources:
            update_resource_metrics()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        if exc_val:
            self.error = f"{exc_type.__name__}: {str(exc_val)}"
            
        record_generation_stats(
            resolution=self.resolution,
            duration=duration,
            success=success,
            error=self.error
        )
        
        if self.track_resources:
            update_resource_metrics() 