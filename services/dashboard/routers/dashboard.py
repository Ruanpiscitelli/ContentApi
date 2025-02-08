"""
Router para endpoints do dashboard.
"""
from datetime import datetime
import time
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import User, LogEntry
from ..models.database_config import get_db
from ..security import get_current_user
from ..services.integration import ServiceIntegration
from ..utils import get_system_stats, format_timestamp

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def dashboard_home(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Renderiza a página principal do dashboard.
    """
    # Obtém status dos serviços
    service_integration = ServiceIntegration()
    services_overview = await service_integration.check_all_services()
    
    # Obtém métricas do sistema
    system_stats = get_system_stats()
    
    # Obtém logs recentes
    logs_query = await db.execute(
        select(LogEntry)
        .order_by(LogEntry.timestamp.desc())
        .limit(10)
    )
    logs = logs_query.scalars().all()
    
    # Prepara dados para os gráficos
    timestamps = []
    text_requests = []
    image_requests = []
    voice_requests = []
    video_requests = []
    
    # Coleta dados das últimas 24 horas
    for service_name in ["text_generation", "image_generation", "voice_generation", "video_editor"]:
        metrics = await service_integration.get_service_metrics(service_name)
        timestamps.append(metrics.timestamps)
        if service_name == "text_generation":
            text_requests = metrics.requests
        elif service_name == "image_generation":
            image_requests = metrics.requests
        elif service_name == "voice_generation":
            voice_requests = metrics.requests
        else:
            video_requests = metrics.requests
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": current_user,
            "services": [
                {
                    "name": "Text Generation",
                    "status": services_overview.text_generation.status,
                    "requests_total": sum(text_requests),
                    "latency": services_overview.text_generation.metrics.get("average_latency", 0),
                    "gpu_usage": services_overview.text_generation.metrics.get("gpu_utilization", 0)
                },
                {
                    "name": "Image Generation",
                    "status": services_overview.image_generation.status,
                    "requests_total": sum(image_requests),
                    "latency": services_overview.image_generation.metrics.get("average_latency", 0),
                    "gpu_usage": services_overview.image_generation.metrics.get("gpu_utilization", 0)
                },
                {
                    "name": "Voice Generation",
                    "status": services_overview.voice_generation.status,
                    "requests_total": sum(voice_requests),
                    "latency": services_overview.voice_generation.metrics.get("average_latency", 0),
                    "gpu_usage": services_overview.voice_generation.metrics.get("gpu_utilization", 0)
                },
                {
                    "name": "Video Editor",
                    "status": services_overview.video_editor.status,
                    "requests_total": sum(video_requests),
                    "latency": services_overview.video_editor.metrics.get("average_latency", 0),
                    "gpu_usage": services_overview.video_editor.metrics.get("gpu_utilization", 0)
                }
            ],
            "timestamps": timestamps,
            "text_requests": text_requests,
            "image_requests": image_requests,
            "voice_requests": voice_requests,
            "video_requests": video_requests,
            "cpu_usage": system_stats["cpu_percent"],
            "memory_usage": system_stats["memory_percent"],
            "gpu_usage": system_stats["gpu_percent"],
            "logs": [
                {
                    "timestamp": format_timestamp(log.timestamp),
                    "service": log.service,
                    "level": log.level.lower(),
                    "message": log.message
                }
                for log in logs
            ],
            "last_update": format_timestamp(datetime.utcnow())
        }
    ) 