import os
import secrets
import uuid
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sse_starlette.sse import EventSourceResponse  # Certifique-se de instalar o sse-starlette
from typing import List

# Carregue variáveis de ambiente do .env conforme sua configuração (ex: com python-dotenv)
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD")
if not DASHBOARD_PASSWORD:
    raise RuntimeError("DASHBOARD_PASSWORD não configurada no .env.")

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET não configurada no .env.")

API_KEY_FILE = "api_keys.txt"
LOGS = []  # Variável global para armazenar logs

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
templates = Jinja2Templates(directory="dashboard/templates")

def read_api_keys() -> List[str]:
    if not os.path.isfile(API_KEY_FILE):
        return []
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def write_api_keys(keys: List[str]) -> None:
    with open(API_KEY_FILE, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(k + "\n")

def require_login(request: Request):
    if "user" not in request.session:
        raise HTTPException(status_code=401, detail="Não autenticado")
    return request.session["user"]

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("dashboard_login.html", {
        "request": request,
        "error": ""
    })

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, password: str = Form(...)):
    if not secrets.compare_digest(password, DASHBOARD_PASSWORD):
        return templates.TemplateResponse("dashboard_login.html", {
            "request": request,
            "error": "Senha incorreta"
        })
    request.session["user"] = "admin"
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str = Depends(require_login)):
    LOGS.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Dashboard acessado")
    services_status = {
        "image_generator": "OK",
        "voice_generator": "OK",
        "video_generator": "OK",
        "video_editor": "OK"
    }
    metrics = {
        "cpu": "45%",
        "memory": "3.2GB",
        "gpu": {"GPU0": "80%", "GPU1": "75%"}
    }
    current_keys = read_api_keys()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "services_status": services_status,
        "metrics": metrics,
        "api_key": None,
        "all_api_keys": current_keys
    })

@app.post("/generate-api-key", response_class=HTMLResponse)
async def generate_api_key(request: Request, user: str = Depends(require_login)):
    new_api_key = str(uuid.uuid4())
    keys = read_api_keys()
    if len(keys) >= 5:
        keys.pop(0)
    keys.append(new_api_key)
    write_api_keys(keys)
    LOGS.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Nova API key gerada: {new_api_key}")
    services_status = {
        "image_generator": "OK",
        "voice_generator": "OK",
        "video_generator": "OK",
        "video_editor": "OK"
    }
    metrics = {
        "cpu": "45%",
        "memory": "3.2GB",
        "gpu": {"GPU0": "80%", "GPU1": "75%"}
    }
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "services_status": services_status,
        "metrics": metrics,
        "api_key": new_api_key,
        "all_api_keys": keys
    })

@app.get("/api/metrics")
async def api_metrics(user: str = Depends(require_login)):
    return {
        "cpu": 45,  # valor numérico
        "memory": 3.2,
        "gpu": {"GPU0": 80, "GPU1": 75}
    }

# Endpoint SSE para envio dos logs em tempo real
@app.get("/api/logs_stream")
async def logs_stream(request: Request, user: str = Depends(require_login)):
    async def event_generator():
        last_index = 0
        while True:
            if await request.is_disconnected():
                break
            # Se houver novos logs, envie-os
            if len(LOGS) > last_index:
                for log in LOGS[last_index:]:
                    yield {"event": "log", "data": log}
                last_index = len(LOGS)
            await asyncio.sleep(1)
    return EventSourceResponse(event_generator())
