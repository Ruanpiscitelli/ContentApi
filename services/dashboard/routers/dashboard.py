"""
Router principal do dashboard.
"""
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import psutil
import humanize

from ..models.database import User, APIKey, LogEntry
from ..models.database_config import get_db
from ..security import (
    get_current_user,
    authenticate_user,
    create_access_token,
    generate_api_key
)

router = APIRouter()
templates = Jinja2Templates(directory="services/dashboard/templates")

@router.get("/", response_class=HTMLResponse)
async def dashboard_home(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Página inicial do dashboard."""
    # Obtém estatísticas do sistema
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Obtém chaves API do usuário
    result = await db.execute(
        select(APIKey).where(
            APIKey.user_id == current_user.id,
            APIKey.is_active == True
        )
    )
    api_keys = result.scalars().all()
    
    # Obtém logs recentes
    result = await db.execute(
        select(LogEntry)
        .order_by(LogEntry.timestamp.desc())
        .limit(100)
    )
    logs = result.scalars().all()
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": current_user,
            "api_keys": api_keys,
            "logs": logs,
            "stats": {
                "cpu": cpu_percent,
                "memory": {
                    "used": humanize.naturalsize(memory.used),
                    "total": humanize.naturalsize(memory.total),
                    "percent": memory.percent
                },
                "disk": {
                    "used": humanize.naturalsize(disk.used),
                    "total": humanize.naturalsize(disk.total),
                    "percent": disk.percent
                }
            }
        }
    )

@router.post("/api-keys/generate")
async def generate_new_api_key(
    name: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Gera uma nova chave API."""
    # Verifica limite de chaves
    result = await db.execute(
        select(APIKey).where(
            APIKey.user_id == current_user.id,
            APIKey.is_active == True
        )
    )
    active_keys = result.scalars().all()
    
    if len(active_keys) >= 5:
        raise HTTPException(
            status_code=400,
            detail="Limite de 5 chaves API atingido"
        )
    
    # Gera nova chave
    new_key = APIKey(
        key=generate_api_key(),
        name=name,
        user_id=current_user.id,
        expires_at=datetime.utcnow() + timedelta(days=365)
    )
    
    db.add(new_key)
    await db.commit()
    await db.refresh(new_key)
    
    # Salva em arquivo texto
    with open("api_keys.txt", "a") as f:
        f.write(f"{datetime.utcnow()} - {current_user.username} - {name}: {new_key.key}\n")
    
    return {"key": new_key.key}

@router.post("/api-keys/{key_id}/revoke")
async def revoke_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Revoga uma chave API."""
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id
        )
    )
    key = result.scalar_one_or_none()
    
    if not key:
        raise HTTPException(status_code=404, detail="Chave não encontrada")
    
    key.is_active = False
    await db.commit()
    
    return {"status": "success"}

@router.get("/logs/stream")
async def stream_logs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Stream de logs em tempo real."""
    return {"implementation": "TODO"} 