"""
Este módulo implementa o servidor do dashboard para monitoramento e gerenciamento da API.
O dashboard fornece uma interface web para visualizar estatísticas do sistema, gerenciar chaves de API
e monitorar logs em tempo real.
"""

import os
import asyncio
import psutil
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from core.database import get_db, init_db
from core.models import User, APIKey, LogEntry
from core.security import (
    get_current_user,
    authenticate_user,
    create_access_token,
    get_password_hash,
    generate_api_key
)

# Configuração do FastAPI
app = FastAPI(
    title="Dashboard de Monitoramento",
    description="Interface web para monitoramento e gerenciamento da API de IA",
    version="1.0.0"
)

# Configuração dos templates e arquivos estáticos
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurações
ACCESS_TOKEN_EXPIRE_MINUTES = 30
MAX_API_KEYS_PER_USER = 5

@app.on_event("startup")
async def startup_event():
    """Inicializa o banco de dados na inicialização do servidor."""
    await init_db()

@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Renderiza a página principal do dashboard com estatísticas do sistema,
    chaves de API do usuário e logs recentes.
    """
    # Busca as chaves de API do usuário
    api_keys = await db.query(APIKey).filter(
        APIKey.user_id == current_user.id
    ).order_by(APIKey.created_at.desc()).all()
    
    # Busca os logs mais recentes
    logs = await db.query(LogEntry).order_by(
        LogEntry.timestamp.desc()
    ).limit(100).all()
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": current_user,
            "api_keys": api_keys,
            "logs": logs
        }
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Renderiza a página de login."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Processa o login do usuário e cria um token de acesso.
    """
    user = await authenticate_user(form_data.username, form_data.password, db)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Usuário ou senha incorretos"
            },
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    response = Response(status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    response.headers["Location"] = "/"
    return response

@app.get("/logout")
async def logout():
    """
    Realiza o logout do usuário removendo o cookie do token de acesso.
    """
    response = Response(status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    response.headers["Location"] = "/login"
    return response

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Renderiza a página de registro."""
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Processa o registro de um novo usuário.
    """
    # Verifica se o usuário já existe
    existing_user = await db.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first()
    
    if existing_user:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "Usuário ou email já cadastrado"
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )
    
    # Cria o novo usuário
    user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password)
    )
    db.add(user)
    await db.commit()
    
    return Response(
        status_code=status.HTTP_302_FOUND,
        headers={"Location": "/login"}
    )

@app.get("/system-stats")
async def system_stats(current_user: User = Depends(get_current_user)):
    """
    Retorna estatísticas do sistema em tempo real.
    """
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu_percent": cpu_percent,
        "memory_used": memory.used,
        "memory_total": memory.total,
        "disk_used": disk.used,
        "disk_total": disk.total
    }

@app.post("/api-keys/generate")
async def generate_api_key(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Gera uma nova chave de API para o usuário atual.
    """
    # Verifica o limite de chaves
    active_keys_count = await db.query(APIKey).filter(
        APIKey.user_id == current_user.id,
        APIKey.is_active == True
    ).count()
    
    if active_keys_count >= MAX_API_KEYS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limite máximo de chaves ativas atingido"
        )
    
    # Gera e salva a nova chave
    key = generate_api_key()
    api_key = APIKey(
        user_id=current_user.id,
        key=key,
        is_active=True
    )
    db.add(api_key)
    await db.commit()
    
    # Salva a chave em um arquivo
    keys_dir = os.path.join(os.path.dirname(__file__), "api_keys")
    os.makedirs(keys_dir, exist_ok=True)
    
    with open(os.path.join(keys_dir, f"{current_user.username}_keys.txt"), "a") as f:
        f.write(f"{key}\n")
    
    return Response(
        status_code=status.HTTP_302_FOUND,
        headers={"Location": "/"}
    )

@app.post("/api-keys/{key_id}/revoke")
async def revoke_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Revoga uma chave de API específica.
    """
    api_key = await db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == current_user.id
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chave de API não encontrada"
        )
    
    api_key.is_active = False
    await db.commit()
    
    return Response(
        status_code=status.HTTP_302_FOUND,
        headers={"Location": "/"}
    )

@app.get("/logs/stream")
async def stream_logs(current_user: User = Depends(get_current_user)):
    """
    Endpoint para streaming de logs em tempo real usando Server-Sent Events (SSE).
    """
    async def event_generator():
        while True:
            # Simula a busca de novos logs (em uma implementação real,
            # isso seria integrado com um sistema de logs real)
            await asyncio.sleep(1)
            yield {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "Sistema funcionando normalmente"
            }
    
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 