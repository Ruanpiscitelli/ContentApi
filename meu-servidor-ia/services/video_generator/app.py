"""
Serviço de geração e edição de vídeos.
"""
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

import routers.video as video_router
from .config import API_CONFIG
from .pipeline import VideoPipeline

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa FastAPI
app = FastAPI(
    title="Video Generator API",
    description="API para geração de vídeos usando modelos de IA",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Incluir routers
app.include_router(video_router.router, prefix="/api/v1", tags=["video"])

# Eventos de lifecycle
@app.on_event("startup")
async def startup():
    """Inicializa recursos na startup."""
    logger.info("Iniciando serviço de vídeo...")
    await VideoPipeline().start_workers()

@app.on_event("shutdown")
async def shutdown():
    """Limpa recursos no shutdown."""
    logger.info("Desligando serviço de vídeo...")
    await VideoPipeline().stop_workers()

# Health check
@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }