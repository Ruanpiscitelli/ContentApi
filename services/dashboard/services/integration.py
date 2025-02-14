"""
Serviço de integração com outros microserviços.
"""
import time
import logging
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from ..schemas.integration import ServiceInfo, ServiceStatus, ServiceMetrics
from ..config import API_CONFIG

logger = logging.getLogger(__name__)

class ServiceIntegration:
    """Gerencia integração com outros serviços."""
    
    def __init__(self):
        """Inicializa o serviço de integração."""
        self.services = {
            "text_generation": {
                "url": API_CONFIG["services"]["text_generation"],
                "last_check": 0,
                "cache_ttl": 60,  # 1 minuto
                "info": None
            },
            "image_generation": {
                "url": API_CONFIG["services"]["image_generation"],
                "last_check": 0,
                "cache_ttl": 60,
                "info": None
            },
            "voice_generation": {
                "url": API_CONFIG["services"]["voice_generation"],
                "last_check": 0,
                "cache_ttl": 60,
                "info": None
            },
            "video_editor": {
                "url": API_CONFIG["services"]["video_editor"],
                "last_check": 0,
                "cache_ttl": 60,
                "info": None
            }
        }
        
        # Session para requisições HTTP
        self.session = None
    
    async def start(self):
        """Inicializa o cliente HTTP."""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def stop(self):
        """Fecha o cliente HTTP."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def check_service(self, service_name: str) -> ServiceInfo:
        """
        Verifica status de um serviço específico.
        
        Args:
            service_name: Nome do serviço
            
        Returns:
            Informações atualizadas do serviço
        """
        service = self.services.get(service_name)
        if not service:
            raise ValueError(f"Serviço desconhecido: {service_name}")
            
        # Verifica se pode usar cache
        now = time.time()
        if service["info"] and (now - service["last_check"]) < service["cache_ttl"]:
            return service["info"]
        
        # Faz requisição para o serviço
        try:
            # Health check
            health_info = await self._get_health(service["url"])
            
            # Modelos disponíveis (apenas para serviços relevantes)
            models = None
            if service_name in ["text_generation", "image_generation"]:
                models = await self._get_models(service["url"])
            
            # Métricas do serviço
            metrics = await self._get_metrics(service["url"])
            
            # Cria objeto ServiceInfo
            info = ServiceInfo(
                name=service_name,
                status=ServiceStatus(health_info.get("status", "unhealthy")),
                version=health_info.get("version", "unknown"),
                url=service["url"],
                last_check=now,
                gpu_available=health_info.get("gpu", {}).get("available"),
                gpu_info=health_info.get("gpu"),
                models=models,
                metrics=metrics
            )
            
            # Atualiza cache
            service["info"] = info
            service["last_check"] = now
            
            return info
            
        except Exception as e:
            logger.error(f"Erro ao verificar serviço {service_name}: {str(e)}")
            
            # Retorna status de indisponível
            return ServiceInfo(
                name=service_name,
                status=ServiceStatus.UNAVAILABLE,
                version="unknown",
                url=service["url"],
                last_check=now,
                error=str(e)
            )
    
    async def check_all_services(self) -> Dict[str, ServiceInfo]:
        """
        Verifica status de todos os serviços.
        
        Returns:
            Dicionário com status de cada serviço
        """
        results = {}
        for service_name in self.services:
            results[service_name] = await self.check_service(service_name)
        return results
    
    async def get_service_metrics(self, service_name: str) -> ServiceMetrics:
        """
        Obtém métricas detalhadas de um serviço.
        
        Args:
            service_name: Nome do serviço
            
        Returns:
            Métricas do serviço
        """
        service = self.services.get(service_name)
        if not service:
            raise ValueError(f"Serviço desconhecido: {service_name}")
            
        try:
            metrics = await self._get_metrics(service["url"])
            
            return ServiceMetrics(
                requests_total=metrics.get("requests_total", 0),
                requests_success=metrics.get("requests_success", 0),
                requests_error=metrics.get("requests_error", 0),
                average_latency=metrics.get("average_latency", 0),
                requests_last_hour=metrics.get("requests_last_hour", 0),
                requests_last_day=metrics.get("requests_last_day", 0),
                gpu_utilization=metrics.get("gpu_utilization"),
                memory_utilization=metrics.get("memory_utilization", 0),
                service_uptime=metrics.get("uptime", 0)
            )
            
        except Exception as e:
            logger.error(f"Erro ao obter métricas do serviço {service_name}: {str(e)}")
            raise
    
    async def _get_health(self, base_url: str) -> Dict[str, Any]:
        """Faz requisição para endpoint de health check."""
        if not self.session:
            await self.start()
            
        async with self.session.get(f"{base_url}/health") as response:
            return await response.json()
    
    async def _get_models(self, base_url: str) -> Optional[List[Dict[str, str]]]:
        """Obtém lista de modelos disponíveis."""
        if not self.session:
            await self.start()
            
        try:
            async with self.session.get(f"{base_url}/models") as response:
                data = await response.json()
                return data.get("models")
        except:
            return None
    
    async def _get_metrics(self, base_url: str) -> Dict[str, Any]:
        """Obtém métricas do serviço."""
        if not self.session:
            await self.start()
            
        try:
            async with self.session.get(f"{base_url}/metrics") as response:
                return await response.json()
        except:
            return {} 