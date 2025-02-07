"""
Monitoramento e métricas para o serviço de geração de imagens.
"""
import time
import logging
import psutil
import torch
from typing import Dict, Any, Optional
from functools import wraps
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, multiprocess, REGISTRY
)
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Registro personalizado para métricas
REGISTRY = CollectorRegistry()

def setup_metrics():
    """
    Configura métricas conforme recomendações Hunyuan e Prometheus.
    """
    return {
        "gpu": {
            "memory_used": Gauge(
                "gpu_memory_used_bytes",
                "Memória GPU utilizada em bytes",
                ["device", "model"],
                registry=REGISTRY
            ),
            "utilization": Gauge(
                "gpu_utilization_percent",
                "Utilização da GPU em porcentagem",
                ["device", "model"],
                registry=REGISTRY
            ),
            "temperature": Gauge(
                "gpu_temperature_celsius",
                "Temperatura da GPU em Celsius",
                ["device"],
                registry=REGISTRY
            ),
            "power": Gauge(
                "gpu_power_watts",
                "Consumo de energia da GPU em watts",
                ["device"],
                registry=REGISTRY
            )
        },
        "inference": {
            "latency": Histogram(
                "inference_latency_seconds",
                "Latência de inferência em segundos",
                ["model", "batch_size", "resolution"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
                registry=REGISTRY
            ),
            "throughput": Counter(
                "inference_throughput_total",
                "Total de inferências realizadas",
                ["model", "status"],
                registry=REGISTRY
            ),
            "errors": Counter(
                "inference_errors_total",
                "Total de erros de inferência",
                ["model", "error_type"],
                registry=REGISTRY
            ),
            "batch_size": Histogram(
                "inference_batch_size",
                "Distribuição de tamanhos de batch",
                ["model"],
                buckets=[1, 2, 4, 8, 16, 32],
                registry=REGISTRY
            )
        },
        "pipeline": {
            "load_time": Histogram(
                "pipeline_load_time_seconds",
                "Tempo de carregamento do pipeline em segundos",
                ["model"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                registry=REGISTRY
            ),
            "optimization_time": Histogram(
                "pipeline_optimization_time_seconds",
                "Tempo de otimização do pipeline em segundos",
                ["model"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
                registry=REGISTRY
            ),
            "cache_hits": Counter(
                "pipeline_cache_hits_total",
                "Total de hits no cache do pipeline",
                ["model"],
                registry=REGISTRY
            ),
            "cache_misses": Counter(
                "pipeline_cache_misses_total",
                "Total de misses no cache do pipeline",
                ["model"],
                registry=REGISTRY
            )
        },
        "resources": {
            "cpu_memory": Gauge(
                "process_memory_bytes",
                "Uso de memória do processo em bytes",
                registry=REGISTRY
            ),
            "cpu_percent": Gauge(
                "process_cpu_percent",
                "Uso de CPU do processo em porcentagem",
                registry=REGISTRY
            ),
            "open_files": Gauge(
                "process_open_files",
                "Número de arquivos abertos pelo processo",
                registry=REGISTRY
            ),
            "threads": Gauge(
                "process_threads",
                "Número de threads do processo",
                registry=REGISTRY
            )
        }
    }

# Inicializa métricas
METRICS = setup_metrics()

@contextmanager
def track_inference_time(model: str, batch_size: int, resolution: str):
    """
    Contexto para tracking de tempo de inferência.
    
    Args:
        model: Nome do modelo
        batch_size: Tamanho do batch
        resolution: Resolução da imagem (ex: "512x512")
    """
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        METRICS["inference"]["latency"].labels(
            model=model,
            batch_size=batch_size,
            resolution=resolution
        ).observe(duration)
        
        METRICS["inference"]["throughput"].labels(
            model=model,
            status="success"
        ).inc(batch_size)
        
        METRICS["inference"]["batch_size"].labels(
            model=model
        ).observe(batch_size)
        
    except Exception as e:
        METRICS["inference"]["throughput"].labels(
            model=model,
            status="error"
        ).inc(batch_size)
        
        METRICS["inference"]["errors"].labels(
            model=model,
            error_type=type(e).__name__
        ).inc()
        raise

def update_gpu_metrics(model: str):
    """
    Atualiza métricas de GPU.
    
    Args:
        model: Nome do modelo sendo executado
    """
    if not torch.cuda.is_available():
        return
        
    try:
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            
            # Memória
            memory_allocated = torch.cuda.memory_allocated(i)
            METRICS["gpu"]["memory_used"].labels(
                device=device,
                model=model
            ).set(memory_allocated)
            
            # Utilização (requer nvidia-smi)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Temperatura
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                METRICS["gpu"]["temperature"].labels(
                    device=device
                ).set(temp)
                
                # Utilização
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                METRICS["gpu"]["utilization"].labels(
                    device=device,
                    model=model
                ).set(util.gpu)
                
                # Energia
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                METRICS["gpu"]["power"].labels(
                    device=device
                ).set(power)
                
            except ImportError:
                logger.warning("pynvml não disponível para métricas detalhadas de GPU")
                
    except Exception as e:
        logger.error(f"Erro ao atualizar métricas GPU: {e}")

def update_resource_metrics():
    """Atualiza métricas de recursos do sistema."""
    try:
        process = psutil.Process()
        
        # Memória
        METRICS["resources"]["cpu_memory"].set(
            process.memory_info().rss
        )
        
        # CPU
        METRICS["resources"]["cpu_percent"].set(
            process.cpu_percent()
        )
        
        # Arquivos abertos
        METRICS["resources"]["open_files"].set(
            len(process.open_files())
        )
        
        # Threads
        METRICS["resources"]["threads"].set(
            process.num_threads()
        )
        
        # GPU
        update_gpu_metrics("stable-diffusion")
        
    except Exception as e:
        logger.error(f"Erro ao atualizar métricas de recursos: {e}")

def get_metrics_snapshot() -> Dict[str, Any]:
    """
    Retorna snapshot atual das métricas.
    
    Returns:
        Dict[str, Any]: Dicionário com métricas atuais
    """
    metrics = {
        "gpu": {},
        "inference": {
            "total": {
                "success": sum(
                    c._value.get() 
                    for c in METRICS["inference"]["throughput"].collect()[0].samples 
                    if c.labels["status"] == "success"
                ),
                "error": sum(
                    c._value.get()
                    for c in METRICS["inference"]["throughput"].collect()[0].samples
                    if c.labels["status"] == "error"
                )
            },
            "errors": {
                label["error_type"]: counter._value.get()
                for label, counter in METRICS["inference"]["errors"]._metrics.items()
            },
            "latency": {
                "avg": METRICS["inference"]["latency"]._sum.get() / 
                      max(METRICS["inference"]["latency"]._count.get(), 1),
                "count": METRICS["inference"]["latency"]._count.get()
            }
        },
        "resources": {
            "cpu_memory": METRICS["resources"]["cpu_memory"]._value.get(),
            "cpu_percent": METRICS["resources"]["cpu_percent"]._value.get(),
            "open_files": METRICS["resources"]["open_files"]._value.get(),
            "threads": METRICS["resources"]["threads"]._value.get()
        }
    }
    
    # Adiciona métricas GPU se disponível
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            metrics["gpu"][device] = {
                "memory_used": METRICS["gpu"]["memory_used"].labels(
                    device=device,
                    model="stable-diffusion"
                )._value.get(),
                "utilization": METRICS["gpu"]["utilization"].labels(
                    device=device,
                    model="stable-diffusion"
                )._value.get(),
                "temperature": METRICS["gpu"]["temperature"].labels(
                    device=device
                )._value.get(),
                "power": METRICS["gpu"]["power"].labels(
                    device=device
                )._value.get()
            }
            
    return metrics 