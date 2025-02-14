"""
Schemas para integração dos serviços.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum

class ServiceStatus(str, Enum):
    """Status possíveis de um serviço."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"

class ServiceInfo(BaseModel):
    """Informações básicas de um serviço."""
    name: str = Field(..., description="Nome do serviço")
    status: ServiceStatus = Field(..., description="Status atual do serviço")
    version: str = Field(..., description="Versão do serviço")
    url: str = Field(..., description="URL base do serviço")
    last_check: float = Field(..., description="Timestamp da última verificação")
    error: Optional[str] = Field(None, description="Mensagem de erro se houver")
    
    # Informações específicas do serviço
    gpu_available: Optional[bool] = Field(None, description="Se GPU está disponível")
    gpu_info: Optional[Dict[str, Any]] = Field(None, description="Informações da GPU")
    models: Optional[List[Dict[str, str]]] = Field(None, description="Modelos disponíveis")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Métricas do serviço")

class ServicesOverview(BaseModel):
    """Visão geral de todos os serviços."""
    text_generation: ServiceInfo
    image_generation: ServiceInfo
    voice_generation: ServiceInfo
    video_editor: ServiceInfo
    timestamp: float = Field(..., description="Timestamp da coleta")
    
    @property
    def all_healthy(self) -> bool:
        """Verifica se todos os serviços estão saudáveis."""
        return all(
            service.status == ServiceStatus.HEALTHY
            for service in [
                self.text_generation,
                self.image_generation,
                self.voice_generation,
                self.video_editor
            ]
        )

class ServiceMetrics(BaseModel):
    """Métricas de uso dos serviços."""
    requests_total: int = Field(..., description="Total de requisições")
    requests_success: int = Field(..., description="Requisições com sucesso")
    requests_error: int = Field(..., description="Requisições com erro")
    average_latency: float = Field(..., description="Latência média (ms)")
    requests_last_hour: int = Field(..., description="Requisições na última hora")
    requests_last_day: int = Field(..., description="Requisições no último dia")
    gpu_utilization: Optional[float] = Field(None, description="Utilização da GPU (%)")
    memory_utilization: float = Field(..., description="Utilização de memória (%)")
    service_uptime: float = Field(..., description="Tempo online em segundos") 