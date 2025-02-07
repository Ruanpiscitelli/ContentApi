import os
import secrets
import uuid
import asyncio
import logging
import psutil
import aiohttp
import json
import time
from datetime import datetime
from fastapi import FastAPI, Request, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from sse_starlette.sse import EventSourceResponse
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from state import dashboard_state, get_dashboard_state
from gpu_monitor import gpu_monitor, get_gpu_monitor
from metrics import MetricsExporter

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dashboard.log')
    ]
)
logger = logging.getLogger(__name__)

# Carregue variáveis de ambiente do .env
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD")
if not DASHBOARD_PASSWORD:
    raise RuntimeError("DASHBOARD_PASSWORD não configurada no .env.")

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET não configurada no .env.")

# Nova variável para definir explicitamente os origins permitidos
# Quando allow_credentials=True, não podemos usar "*" como allow_origins.
# Utilize uma lista de URLs permitidas, separadas por vírgula.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1").split(",")

API_KEY_FILE = "api_keys.txt"

# URLs dos serviços
SERVICES = {
    "text_generator": "http://localhost:8006",
    "image_generator": "http://localhost:8001",
    "voice_generator": "http://localhost:8002",
    "video_generator": "http://localhost:8003",
    "video_editor": "http://localhost:8004"
}

app = FastAPI()

# Configuração do CORS (atualizada para utilizar os origins definidos)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # Utiliza lista específica de origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adiciona suporte a arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Adiciona middleware de sessão
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

templates = Jinja2Templates(directory="templates")

# Adicionar estas constantes no início do arquivo
MAX_DATA_POINTS = 100  # Número máximo de pontos mantidos por gráfico

# Inicializar o exportador de métricas
metrics_exporter = MetricsExporter(port=8001)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logar todas as requisições e respostas com informações sanitizadas."""
    start_time = time.perf_counter()
    
    # Log da requisição com headers sanitizados
    request_id = str(uuid.uuid4())
    sanitized_headers = {
        k: v for k, v in request.headers.items() 
        if k.lower() not in {'authorization', 'cookie', 'x-api-key'}
    }
    
    log_msg = f"""
    Request {request_id}:
    {request.method} {request.url}
    Client: {request.client}
    Headers: {sanitized_headers}
    """
    await dashboard_state.add_log(log_msg)

    try:
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        
        # Log da resposta com informações relevantes
        log_msg = f"""
        Response {request_id}:
        Status: {response.status_code}
        Process Time: {process_time:.4f}s
        """
        await dashboard_state.add_log(log_msg)
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response
    except Exception as e:
        error_msg = f"Error {request_id}: {str(e)}"
        await dashboard_state.add_log(error_msg, level="ERROR")
        raise e

async def check_service_status(url: str) -> bool:
    """Verifica se um serviço está ativo fazendo uma requisição HTTP."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=2) as response:
                is_active = response.status == 200
                await dashboard_state.add_log(f"Verificação de serviço {url}: {'OK' if is_active else 'OFFLINE'}")
                return is_active
    except Exception as e:
        await dashboard_state.add_log(f"Erro ao verificar serviço {url}: {str(e)}")
        return False

async def get_services_status() -> Dict[str, str]:
    """Obtém o status de todos os serviços."""
    services_status = {}
    for service, url in SERVICES.items():
        is_active = await check_service_status(url)
        services_status[service] = "OK" if is_active else "OFFLINE"
        if not is_active:
            await dashboard_state.add_log(f"Serviço {service} está offline")
    return services_status

async def fetch_metrics():
    """Função interna para coletar métricas do sistema (CPU, memória e GPU)."""
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_gb = round(memory.used / (1024 ** 3), 2)
        
        # Cria uma task para coletar métricas da GPU de forma assíncrona
        gpu_task = asyncio.create_task(gpu_monitor.get_metrics())
        gpu_metrics = await gpu_task
        
        return {
            "cpu": cpu_percent,
            "memory": memory_gb,
            "gpu": gpu_metrics
        }
    except Exception as e:
        logger.error(f"Erro ao obter métricas: {e}")
        return {
            "cpu": 0,
            "memory": 0,
            "gpu": {"GPU0": {"status": "ERROR", "error": str(e)}}
        }

def read_api_keys() -> List[str]:
    """Lê as API keys do arquivo de forma segura."""
    try:
        if not os.path.isfile(API_KEY_FILE):
            logger.warning(f"Arquivo de API keys não encontrado: {API_KEY_FILE}")
            return []
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            keys = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Lidas {len(keys)} API keys")
            return keys
    except Exception as e:
        logger.error(f"Erro ao ler API keys: {str(e)}")
        return []

def write_api_keys(keys: List[str]) -> bool:
    """Escreve as API keys no arquivo de forma segura."""
    try:
        with open(API_KEY_FILE, "w", encoding="utf-8") as f:
            for k in keys:
                f.write(k + "\n")
        logger.info(f"Escritas {len(keys)} API keys")
        return True
    except Exception as e:
        logger.error(f"Erro ao escrever API keys: {str(e)}")
        return False

def require_login(request: Request):
    if "user" not in request.session:
        raise HTTPException(status_code=401, detail="Não autenticado")
    return request.session["user"]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    await dashboard_state.add_log("Acessando rota raiz (/)")
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    await dashboard_state.add_log("Acessando página de login")
    return templates.TemplateResponse("dashboard_login.html", {
        "request": request,
        "error": ""
    })

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, username: Optional[str] = Form(None), password: str = Form(...)):
    # Se o campo username não foi enviado, utiliza o valor padrão configurado
    if not username:
        username = DASHBOARD_USERNAME

    if username != DASHBOARD_USERNAME or not secrets.compare_digest(password, DASHBOARD_PASSWORD):
        await dashboard_state.add_log(f"Tentativa de login mal sucedida para usuário: {username}")
        return templates.TemplateResponse("dashboard_login.html", {
            "request": request,
            "error": "Usuário ou senha incorretos"
        })
    request.session["user"] = username
    await dashboard_state.add_log(f"Usuário {username} fez login")
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

@app.get("/logout")
async def logout(request: Request):
    if "user" in request.session:
        username = request.session["user"]
        await dashboard_state.add_log(f"Usuário {username} fez logout")
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str = Depends(require_login)):
    """Rota principal do dashboard."""
    await dashboard_state.add_log(f"Dashboard acessado por {user}")
    
    # Coleta dados de forma assíncrona
    services_task = asyncio.create_task(get_services_status())
    metrics_task = asyncio.create_task(fetch_metrics())
    
    # Aguarda todas as tasks completarem
    services_status = await services_task
    metrics = await metrics_task
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
    """Gera uma nova API key e atualiza o dashboard com métricas e status dos serviços."""
    try:
        new_api_key = str(uuid.uuid4())
        keys = read_api_keys()
        if len(keys) >= 5:
            keys.pop(0)
        keys.append(new_api_key)
        write_api_keys(keys)
        await dashboard_state.add_log(f"Nova API key gerada por {user}: {new_api_key}")
        
        # Coleta dados de forma assíncrona
        services_task = asyncio.create_task(get_services_status())
        metrics_task = asyncio.create_task(fetch_metrics())
        
        # Aguarda todas as tasks completarem
        services_status = await services_task
        metrics = await metrics_task
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "services_status": services_status,
            "metrics": metrics,
            "api_key": new_api_key,
            "all_api_keys": keys
        })
    except Exception as e:
        await dashboard_state.add_log(f"Erro ao gerar API key: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao gerar API key")

@app.get("/api/metrics")
async def api_metrics(user: str = Depends(require_login)):
    """Endpoint para métricas em tempo real"""
    try:
        metrics = await gpu_monitor.get_metrics()
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Coleta métricas do text generator
        text_metrics = {
            "latency": 0,
            "error_rate": 0,
            "cache_hit_rate": 0,
            "tokens_per_minute": 0
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{SERVICES['text_generator']}/metrics") as response:
                    if response.status == 200:
                        text_data = await response.json()
                        text_metrics.update(text_data)
        except Exception as e:
            logger.error(f"Erro ao coletar métricas do text generator: {e}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": cpu_percent,
            "memory": round(memory.used / (1024**3), 2),  # GB
            "gpu": metrics,
            "text_generator": text_metrics
        }
    except Exception as e:
        await dashboard_state.add_log(f"Erro ao coletar métricas: {str(e)}", level="ERROR")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(user: str = Depends(require_login)):
    """Endpoint para obter logs do sistema, protegido por autenticação."""
    try:
        state = await get_dashboard_state()
        return {"logs": state.get_recent_logs(100)}
    except Exception as e:
        logger.error(f"Erro ao obter logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/logs_stream")
async def logs_stream(request: Request, user: str = Depends(require_login)):
    """Stream de logs em tempo real via SSE com controle de duplicação melhorado."""
    async def event_generator():
        try:
            last_processed_logs = set()  # Conjunto para controle de logs já processados
            last_metrics_time = 0
            
            while True:
                if await request.is_disconnected():
                    break

                current_time = time.time()
                
                # Buscar novos logs com controle de duplicação
                logs = await dashboard_state.get_recent_logs(100)
                if logs:
                    for log in logs:
                        log_id = log.get('id')
                        if log_id and log_id not in last_processed_logs:
                            last_processed_logs.add(log_id)
                            # Manter o conjunto com tamanho controlado
                            if len(last_processed_logs) > 1000:
                                last_processed_logs.clear()
                            
                            yield {
                                "event": "log",
                                "id": log_id,
                                "data": json.dumps({
                                    "timestamp": log.get('timestamp'),
                                    "message": log.get('message'),
                                    "level": log.get('level', 'INFO')
                                })
                            }

                # Enviar métricas GPU a cada 5 segundos
                if current_time - last_metrics_time >= 5:
                    try:
                        metrics = await gpu_monitor.get_metrics()
                        if metrics:
                            yield {
                                "event": "metrics",
                                "id": str(uuid.uuid4()),
                                "data": json.dumps(metrics)
                            }
                        last_metrics_time = current_time
                    except Exception as metrics_error:
                        logger.error(f"Erro ao coletar métricas GPU: {metrics_error}")

                await asyncio.sleep(1)
                
        except Exception as e:
            error_msg = f"Erro no stream: {str(e)}"
            logger.error(error_msg)
            await dashboard_state.add_log(error_msg, level="ERROR")
            yield {
                "event": "error",
                "id": str(uuid.uuid4()),
                "data": json.dumps({"error": error_msg})
            }

    return EventSourceResponse(event_generator())

@app.on_event("startup")
async def startup_event():
    """Evento de inicialização do aplicativo com melhor tratamento de erros."""
    try:
        await gpu_monitor.initialize()
        await metrics_exporter.start()
        await dashboard_state.add_log("Aplicativo iniciado com sucesso")
        logger.info("Dashboard iniciado com sucesso")
    except Exception as e:
        error_msg = f"Erro na inicialização do dashboard: {str(e)}"
        logger.error(error_msg)
        await dashboard_state.add_log(error_msg, level="ERROR")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Evento de desligamento do aplicativo com melhor tratamento de erros."""
    try:
        await gpu_monitor.shutdown()
        await dashboard_state.add_log("Aplicativo encerrado com sucesso")
        logger.info("Dashboard encerrado com sucesso")
    except Exception as e:
        error_msg = f"Erro no desligamento do dashboard: {str(e)}"
        logger.error(error_msg)
        # Não propagar o erro no shutdown para garantir encerramento limpo

@app.get("/metrics")
async def prometheus_metrics():
    """Endpoint para o Prometheus coletar métricas"""
    metrics = await gpu_monitor.get_metrics()
    metrics_exporter.update_metrics(metrics)
    return PlainTextResponse("")
