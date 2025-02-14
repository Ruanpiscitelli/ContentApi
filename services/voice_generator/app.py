"""
Serviço de geração de voz com suporte a múltiplos backends.
"""
import os
import time
import logging
import asyncio
import torch
import numpy as np
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
from datetime import datetime

# Importa módulos compartilhados
from shared.config_base import BaseServiceConfig
from shared.monitoring import ServiceMetrics
from shared.utils import (
    save_upload_file,
    create_temp_file,
    run_with_timeout,
    RateLimiter
)
from shared.cache import CacheManager
from shared.validation import (
    ResourceValidationError,
    GPUResourceValidator,
    ModelLoadValidator
)

from .routers import voice
from .config import API_CONFIG
from .backends import VoiceBackendManager

# Configurações
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MAX_QUEUE_SIZE = 100
BATCH_SIZE = 4
MODEL_CACHE = {}

class OptimizedVoicePipeline:
    """Pipeline otimizado para geração de voz"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.cache_manager = CacheManager(REDIS_URL)
        self.metrics = ServiceMetrics()
        self.queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.processing = False
    
    async def initialize(self):
        """Inicializa o pipeline de forma assíncrona"""
        try:
            # Carrega modelo e processador
            self.model = await self._load_model()
            self.processor = await self._load_processor()
            
            # Inicia processamento em background
            self.processing = True
            asyncio.create_task(self._process_queue())
            
            logger.info("Pipeline de voz inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro na inicialização do pipeline: {e}")
            raise
    
    async def _load_model(self):
        """Carrega o modelo de forma otimizada"""
        try:
            if torch.cuda.is_available():
                # Configurações de otimização CUDA
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
            
            # Carrega modelo com otimizações
            model = await asyncio.to_thread(
                self._load_model_sync
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _load_model_sync(self):
        """Carregamento síncrono do modelo"""
        from transformers import AutoModelForSpeechSynthesis
        
        model = AutoModelForSpeechSynthesis.from_pretrained(
            "microsoft/speecht5_tts",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.to(self.device)
        
        if torch.cuda.is_available():
            model = torch.compile(model)
        
        return model
    
    async def _load_processor(self):
        """Carrega o processador de forma assíncrona"""
        from transformers import SpeechT5Processor
        return await asyncio.to_thread(
            SpeechT5Processor.from_pretrained,
            "microsoft/speecht5_tts"
        )
    
    async def generate(self, text: str, voice_id: str = "default", language: str = "pt-BR") -> bytes:
        """Gera áudio a partir do texto"""
        try:
            # Verifica cache
            cache_key = f"voice_{voice_id}_{hash(text)}"
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return cached
            
            # Adiciona à fila
            task = {
                "text": text,
                "voice_id": voice_id,
                "language": language,
                "future": asyncio.Future()
            }
            
            await self.queue.put(task)
            
            # Aguarda processamento
            result = await task["future"]
            
            # Cache resultado
            await self.cache_manager.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            raise
    
    async def _process_queue(self):
        """Processa a fila de forma assíncrona"""
        while self.processing:
            try:
                # Coleta batch de tarefas
                batch = []
                for _ in range(BATCH_SIZE):
                    if self.queue.empty():
                        break
                    batch.append(await self.queue.get())
                
                if not batch:
                    await asyncio.sleep(0.1)
                    continue
                
                # Processa batch
                results = await self._process_batch(batch)
                
                # Define resultados
                for task, result in zip(batch, results):
                    task["future"].set_result(result)
                
            except Exception as e:
                logger.error(f"Erro no processamento do batch: {e}")
                for task in batch:
                    if not task["future"].done():
                        task["future"].set_exception(e)
    
    @torch.inference_mode()
    async def _process_batch(self, batch: List[Dict]) -> List[bytes]:
        """Processa um batch de textos"""
        try:
            texts = [task["text"] for task in batch]
            
            # Processa inputs
            inputs = await asyncio.to_thread(
                self.processor,
                text=texts,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Gera áudio
            with torch.cuda.amp.autocast():
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    **inputs
                )
            
            # Converte para áudio
            audio_arrays = []
            for output in outputs:
                audio = await asyncio.to_thread(
                    self.processor.batch_decode,
                    output.cpu().numpy()
                )
                audio_arrays.append(audio)
            
            return audio_arrays
            
        except Exception as e:
            logger.error(f"Erro no processamento do batch: {e}")
            raise

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

# Implementação do middleware de timeout personalizado
class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout=300):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timeout"}
            )

# Adiciona o middleware de timeout
app.add_middleware(TimeoutMiddleware, timeout=300)  # 5 minutos

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
    gpu_ok = torch.cuda.is_available()
    model_ok = voice_pipeline.model is not None
    queue_size = voice_pipeline.queue.qsize()
    
    return {
        "status": "healthy" if all([gpu_ok, model_ok]) else "degraded",
        "version": "1.0.0",
        "gpu": {
            "available": gpu_ok,
            "device": torch.cuda.get_device_name(0) if gpu_ok else None,
            "memory": {
                "allocated": torch.cuda.memory_allocated(0) / 1024**2 if gpu_ok else 0,
                "cached": torch.cuda.memory_reserved(0) / 1024**2 if gpu_ok else 0
            }
        },
        "model": {
            "loaded": model_ok
        },
        "queue": {
            "size": queue_size,
            "max_size": MAX_QUEUE_SIZE
        }
    }

class VoiceRequest(BaseModel):
    """Modelo de requisição para geração de voz"""
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str = Field(default="default", min_length=1)
    language: str = Field(default="pt-BR", regex="^[a-z]{2}-[A-Z]{2}$")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Olá, isso é um teste de geração de voz.",
                "voice_id": "default",
                "language": "pt-BR"
            }
        }

# Pipeline global
voice_pipeline = OptimizedVoicePipeline()

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "Voice Generator Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/generate")
async def generate_voice(request: VoiceRequest, background_tasks: BackgroundTasks):
    """
    Gera áudio a partir do texto fornecido.
    
    Args:
        request: Parâmetros da requisição
        background_tasks: Tarefas em background
        
    Returns:
        Dict com URL do áudio gerado
    """
    try:
        # Valida tamanho da fila
        if voice_pipeline.queue.qsize() >= MAX_QUEUE_SIZE:
            raise HTTPException(
                status_code=429,
                detail=f"Fila cheia ({voice_pipeline.queue.qsize()}/{MAX_QUEUE_SIZE})"
            )
        
        # Gera áudio
        audio_data = await voice_pipeline.generate(
            text=request.text,
            voice_id=request.voice_id,
            language=request.language
        )
        
        # Salva arquivo temporário
        file_name = f"voice_{int(time.time())}_{hash(request.text)}.wav"
        file_path = await create_temp_file(audio_data, file_name)
        
        # Upload em background
        background_tasks.add_task(
            save_upload_file,
            file_path,
            file_name,
            "audio/wav"
        )
        
        return {
            "status": "success",
            "message": "Áudio gerado com sucesso",
            "file_name": file_name
        }
        
    except Exception as e:
        logger.error(f"Erro na geração: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
