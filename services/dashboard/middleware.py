"""
Este módulo contém os middlewares utilizados pelo dashboard.
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .utils import logger
from .config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para logging de requisições HTTP.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processa uma requisição HTTP, registrando informações sobre ela.
        
        Args:
            request (Request): Objeto de requisição.
            call_next (Callable): Função para processar a próxima etapa.
        
        Returns:
            Response: Objeto de resposta.
        """
        start_time = time.time()
        
        # Registra informações da requisição
        logger.info(
            f"Requisição iniciada | Método: {request.method} | "
            f"URL: {request.url.path} | Cliente: {request.client.host}"
        )
        
        try:
            response = await call_next(request)
            
            # Calcula o tempo de processamento
            process_time = (time.time() - start_time) * 1000
            
            # Registra informações da resposta
            logger.info(
                f"Requisição finalizada | Status: {response.status_code} | "
                f"Tempo: {process_time:.2f}ms"
            )
            
            # Adiciona headers de tempo de processamento
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            
            return response
            
        except Exception as e:
            # Registra erros não tratados
            logger.error(
                f"Erro na requisição | Método: {request.method} | "
                f"URL: {request.url.path} | Erro: {str(e)}"
            )
            raise

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware para limitação de taxa de requisições.
    """
    
    def __init__(self, app: ASGIApp):
        """
        Inicializa o middleware.
        
        Args:
            app (ASGIApp): Aplicação ASGI.
        """
        super().__init__(app)
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processa uma requisição HTTP, aplicando limitação de taxa.
        
        Args:
            request (Request): Objeto de requisição.
            call_next (Callable): Função para processar a próxima etapa.
        
        Returns:
            Response: Objeto de resposta.
        """
        # Obtém o IP do cliente
        client_ip = request.client.host
        current_time = time.time()
        
        # Limpa registros antigos
        self.cleanup_old_requests(current_time)
        
        # Verifica se o cliente atingiu o limite
        if self.is_rate_limited(client_ip, current_time):
            logger.warning(f"Taxa limite excedida para o IP: {client_ip}")
            return Response(
                content="Taxa limite excedida. Tente novamente mais tarde.",
                status_code=429
            )
        
        # Registra a requisição
        self.record_request(client_ip, current_time)
        
        return await call_next(request)
    
    def cleanup_old_requests(self, current_time: float) -> None:
        """
        Remove registros de requisições antigas.
        
        Args:
            current_time (float): Tempo atual em segundos.
        """
        window_start = current_time - RATE_LIMIT_WINDOW
        
        for ip in list(self.requests.keys()):
            self.requests[ip] = [
                timestamp for timestamp in self.requests[ip]
                if timestamp > window_start
            ]
            
            if not self.requests[ip]:
                del self.requests[ip]
    
    def is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """
        Verifica se um cliente atingiu o limite de requisições.
        
        Args:
            client_ip (str): IP do cliente.
            current_time (float): Tempo atual em segundos.
        
        Returns:
            bool: True se o cliente atingiu o limite, False caso contrário.
        """
        if client_ip not in self.requests:
            return False
        
        window_start = current_time - RATE_LIMIT_WINDOW
        recent_requests = len([
            timestamp for timestamp in self.requests[client_ip]
            if timestamp > window_start
        ])
        
        return recent_requests >= RATE_LIMIT_REQUESTS
    
    def record_request(self, client_ip: str, current_time: float) -> None:
        """
        Registra uma nova requisição para um cliente.
        
        Args:
            client_ip (str): IP do cliente.
            current_time (float): Tempo atual em segundos.
        """
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        self.requests[client_ip].append(current_time)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware para adicionar headers de segurança às respostas.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processa uma requisição HTTP, adicionando headers de segurança.
        
        Args:
            request (Request): Objeto de requisição.
            call_next (Callable): Função para processar a próxima etapa.
        
        Returns:
            Response: Objeto de resposta.
        """
        response = await call_next(request)
        
        # Adiciona headers de segurança
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self' data:; "
            "connect-src 'self'"
        )
        
        return response 