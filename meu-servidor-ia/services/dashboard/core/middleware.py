"""
Middlewares para o servidor FastAPI.
"""

import time
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
import json

def setup_middlewares(app: FastAPI) -> None:
    """Configura todos os middlewares da aplicação."""
    
    # Configuração do CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Em produção, especifique os domínios permitidos
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware de logging
    app.add_middleware(LoggingMiddleware)
    
    # Middleware de tempo de resposta
    app.add_middleware(ResponseTimeMiddleware)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging de requisições e respostas."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Log da requisição
        await self._log_request(request)
        
        try:
            # Processa a requisição
            response = await call_next(request)
            
            # Log da resposta
            await self._log_response(response)
            
            return response
        except Exception as e:
            # Log de erro
            logger.error(f"Erro no processamento da requisição: {str(e)}")
            raise
    
    async def _log_request(self, request: Request) -> None:
        """Registra informações da requisição."""
        logger.info(
            "Requisição recebida",
            extra={
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client": request.client.host if request.client else None,
            }
        )
    
    async def _log_response(self, response: Response) -> None:
        """Registra informações da resposta."""
        logger.info(
            "Resposta enviada",
            extra={
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }
        )

class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """Middleware para monitoramento do tempo de resposta."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log do tempo de processamento
        logger.info(
            "Tempo de processamento",
            extra={
                "process_time": process_time,
                "path": request.url.path,
                "method": request.method,
            }
        )
        
        return response 