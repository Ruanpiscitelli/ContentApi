import os
import secrets
import uuid
from fastapi import FastAPI, Request, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# Verifica se a senha do dashboard está configurada
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD")
if not DASHBOARD_PASSWORD:
    raise RuntimeError(
        "A variável de ambiente DASHBOARD_PASSWORD não está configurada. "
        "Por favor, configure uma senha segura para o dashboard."
    )

# Verifica se a chave secreta da sessão está configurada
SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError(
        "A variável de ambiente SESSION_SECRET não está configurada. "
        "Por favor, configure uma chave secreta forte e única para as sessões."
    )

app = FastAPI()

# Configuração da sessão com a chave secreta obrigatória
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# Configura o diretório de templates (deve conter os arquivos HTML)
templates = Jinja2Templates(directory="dashboard/templates")

######################################
# Endpoints de Login e Logout
######################################

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    """
    Renderiza o formulário de login.
    """
    return templates.TemplateResponse("dashboard_login.html", {"request": request, "error": ""})

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, password: str = Form(...)):
    """
    Processa o formulário de login.
    Verifica se a senha corresponde à definida na variável de ambiente DASHBOARD_PASSWORD.
    """
    if not secrets.compare_digest(password, DASHBOARD_PASSWORD):
        return templates.TemplateResponse("dashboard_login.html", {"request": request, "error": "Senha incorreta"})
    # Se a senha estiver correta, armazena a flag de usuário na sessão.
    request.session["user"] = "admin"
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

@app.get("/logout")
async def logout(request: Request):
    """
    Encerra a sessão do usuário.
    """
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

def require_login(request: Request):
    """
    Dependência que verifica se o usuário está autenticado.
    """
    if "user" not in request.session:
        raise HTTPException(status_code=401, detail="Não autenticado")
    return request.session["user"]

######################################
# Endpoint do Dashboard e Geração de API Key
######################################

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str = Depends(require_login)):
    """
    Renderiza o Dashboard principal com dados simulados de status e métricas.
    Inicialmente, nenhuma API key é gerada.
    """
    # Dados simulados para status e métricas
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
        "api_key": None
    })

@app.post("/generate-api-key", response_class=HTMLResponse)
async def generate_api_key(request: Request, user: str = Depends(require_login)):
    """
    Gera uma nova API key (Bearer Token) para autenticação das demais APIs.
    A API key é gerada como um UUID (a lógica pode ser aprimorada conforme necessário).
    """
    new_api_key = str(uuid.uuid4())
    # Aqui, você pode persistir a chave em um banco de dados ou outro mecanismo de armazenamento.
    # Para este exemplo, apenas retornamos a chave para exibição no dashboard.
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
        "api_key": new_api_key
    })