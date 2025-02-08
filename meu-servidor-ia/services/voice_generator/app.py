"""
Serviço de geração de voz com suporte a múltiplos backends.
"""
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .routers import voice
from .config import API_CONFIG
from .backends import VoiceBackendManager

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa FastAPI
app = FastAPI(
    title="Serviço de Geração de Voz",
    description="API para geração de voz com suporte a múltiplos backends",
    version="1.0.0"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Handlers de exceção globais
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler para erros de validação."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler para exceções HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para exceções não tratadas."""
    logger.error(f"Erro não tratado: {exc}", exc_info=True)
    return JSONResponse(
            status_code=500,
        content={"detail": "Erro interno do servidor"}
    )

# Inclui routers
app.include_router(voice.router)

# Eventos de lifecycle
@app.on_event("startup")
async def startup():
    """Inicializa recursos na startup."""
    logger.info("Iniciando serviço de voz...")
    await VoiceBackendManager().initialize()

@app.on_event("shutdown")
async def shutdown():
    """Limpa recursos no shutdown."""
    logger.info("Desligando serviço de voz...")
    await VoiceBackendManager().cleanup()

# Health check
@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde."""
        return {
            "status": "healthy",
        "version": "1.0.0"
    }
