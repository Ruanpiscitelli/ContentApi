"""
Módulo de monitoramento unificado para todos os serviços.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server, REGISTRY
)
import torch
from functools import wraps

logger = logging.getLogger(__name__)

class ServiceMetrics:
    """Gerenciador de métricas do serviço."""
    
    def __init__(self, service_name: str, port: int = 8001):
        """
        Inicializa as métricas do serviço.
        
        Args:
            service_name: Nome do serviço para namespace das métricas
            port: Porta para expor métricas do Prometheus
        """
        self.service_name = service_name
        self.port = port
        
        # Métricas de Requisições
        self.request_counter = Counter(
            f"{service_name}_requests_total",
            "Total de requisições processadas"
        )
        self.error_counter = Counter(
            f"{service_name}_errors_total",
            "Total de erros",
            ["type"]
        )
        self.request_latency = Histogram(
            f"{service_name}_request_latency_seconds",
            "Latência das requisições",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Métricas de Cache
        self.cache_hits = Counter(
            f"{service_name}_cache_hits_total",
            "Total de hits no cache"
        )
        self.cache_misses = Counter(
            f"{service_name}_cache_misses_total",
            "Total de misses no cache"
        )
        
        # Métricas de GPU
        if torch.cuda.is_available():
            self.gpu_memory_used = Gauge(
                f"{service_name}_gpu_memory_used_bytes",
                "Memória GPU utilizada",
                ["device"]
            )
            self.gpu_utilization = Gauge(
                f"{service_name}_gpu_utilization_percent",
                "Utilização da GPU",
                ["device"]
            )
            
        # Métricas de Sistema
        self.cpu_usage = Gauge(
            f"{service_name}_cpu_usage_percent",
            "Uso de CPU"
        )
        self.memory_usage = Gauge(
            f"{service_name}_memory_usage_bytes",
            "Uso de memória RAM"
        )
        self.open_files = Gauge(
            f"{service_name}_open_files",
            "Número de arquivos abertos"
        )
        
        # Métricas de Modelo
        self.model_load_time = Histogram(
            f"{service_name}_model_load_time_seconds",
            "Tempo de carregamento do modelo",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        self.inference_time = Histogram(
            f"{service_name}_inference_time_seconds",
            "Tempo de inferência",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        # Inicia servidor Prometheus
        try:
            start_http_server(port)
            logger.info(f"Servidor de métricas iniciado na porta {port}")
        except Exception as e:
            logger.error(f"Erro ao iniciar servidor de métricas: {e}")
            
    def track_request(self):
        """Decorator para rastrear métricas de requisição."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                self.request_counter.inc()
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.error_counter.labels(type=type(e).__name__).inc()
                    raise
                finally:
                    self.request_latency.observe(time.time() - start_time)
            return wrapper
        return decorator
        
    def track_cache(self, hit: bool):
        """Registra hit/miss no cache."""
        if hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
            
    def update_gpu_metrics(self):
        """Atualiza métricas de GPU."""
        if not torch.cuda.is_available():
            return
            
        for i in range(torch.cuda.device_count()):
            # Memória
            memory_allocated = torch.cuda.memory_allocated(i)
            self.gpu_memory_used.labels(device=i).set(memory_allocated)
            
            # Utilização (requer pynvml)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.labels(device=i).set(util.gpu)
            except:
                pass
                
    def update_system_metrics(self):
        """Atualiza métricas do sistema."""
        # CPU
        self.cpu_usage.set(psutil.cpu_percent())
        
        # Memória
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)
        
        # Arquivos abertos
        try:
            self.open_files.set(len(psutil.Process().open_files()))
        except:
            pass
            
    def track_model_load(self, duration: float):
        """Registra tempo de carregamento do modelo."""
        self.model_load_time.observe(duration)
        
    def track_inference(self, duration: float):
        """Registra tempo de inferência."""
        self.inference_time.observe(duration)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna todas as métricas atuais."""
        metrics = {}
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                metrics[sample.name] = sample.value
        return metrics 