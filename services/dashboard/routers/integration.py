"""
Router para integração com outros serviços.
"""
import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import User
from ..models.database_config import get_db
from ..security import get_current_user
from ..services.integration import ServiceIntegration
from ..schemas.integration import ServicesOverview, ServiceInfo, ServiceMetrics

router = APIRouter(
    prefix="/integration",
    tags=["integration"]
)

# Instância global do serviço de integração
service_integration = ServiceIntegration()

@router.on_event("startup")
async def startup():
    """Inicializa o serviço de integração."""
    await service_integration.start()

@router.on_event("shutdown")
async def shutdown():
    """Finaliza o serviço de integração."""
    await service_integration.stop()

@router.get("/services/overview")
async def get_services_overview(
    current_user: User = Depends(get_current_user)
) -> ServicesOverview:
    """
    Obtém visão geral de todos os serviços.
    
    Args:
        current_user: Usuário autenticado
        
    Returns:
        Visão geral dos serviços
    """
    try:
        services = await service_integration.check_all_services()
        
        return ServicesOverview(
            text_generation=services["text_generation"],
            image_generation=services["image_generation"],
            voice_generation=services["voice_generation"],
            video_editor=services["video_editor"],
            timestamp=time.time()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter visão geral dos serviços: {str(e)}"
        )

@router.get("/services/{service_name}")
async def get_service_info(
    service_name: str,
    current_user: User = Depends(get_current_user)
) -> ServiceInfo:
    """
    Obtém informações detalhadas de um serviço.
    
    Args:
        service_name: Nome do serviço
        current_user: Usuário autenticado
        
    Returns:
        Informações do serviço
    """
    try:
        return await service_integration.check_service(service_name)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter informações do serviço: {str(e)}"
        )

@router.get("/services/{service_name}/metrics")
async def get_service_metrics(
    service_name: str,
    current_user: User = Depends(get_current_user)
) -> ServiceMetrics:
    """
    Obtém métricas detalhadas de um serviço.
    
    Args:
        service_name: Nome do serviço
        current_user: Usuário autenticado
        
    Returns:
        Métricas do serviço
    """
    try:
        return await service_integration.get_service_metrics(service_name)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter métricas do serviço: {str(e)}"
        ) 