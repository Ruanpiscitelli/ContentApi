"""
Segurança e rate limiting para o serviço de geração de texto.
"""
import time
import jwt
import logging
import asyncio
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Gauge

from config import SECURITY_CONFIG

logger = logging.getLogger(__name__)

# Métricas de rate limiting
RATE_LIMIT_HITS = Counter(
    "text_generation_rate_limit_hits_total",
    "Total de hits no rate limit",
    ["type"]  # global, ip, token
)

RATE_LIMIT_REMAINING = Gauge(
    "text_generation_rate_limit_remaining",
    "Requisições restantes no rate limit",
    ["type", "identifier"]  # type: global, ip, token
)

@dataclass
class RateLimitInfo:
    """Informações de rate limit."""
    requests: int
    reset_time: float
    last_request: float

class RateLimiter:
    """Implementa rate limiting usando algoritmo token bucket."""
    
    def __init__(
        self,
        requests_per_second: float,
        burst_size: int,
        window_size: int = 60
    ):
        """
        Inicializa o rate limiter.
        
        Args:
            requests_per_second: Requisições permitidas por segundo
            burst_size: Número máximo de requisições em burst
            window_size: Tamanho da janela de tempo em segundos
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.window_size = window_size
        self.tokens = {}  # identifier -> RateLimitInfo
        self._cleanup_task = None
    
    def start_cleanup(self):
        """Inicia tarefa de limpeza periódica."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Loop de limpeza de tokens expirados."""
        while True:
            try:
                current_time = time.time()
                expired = []
                
                for identifier, info in self.tokens.items():
                    if current_time - info.last_request > self.window_size:
                        expired.append(identifier)
                
                for identifier in expired:
                    del self.tokens[identifier]
                    
                    # Atualiza métrica
                    RATE_LIMIT_REMAINING.remove(
                        {"type": "token", "identifier": identifier}
                    )
                
                await asyncio.sleep(60)  # Executa a cada minuto
                
            except Exception as e:
                logger.error(f"Erro na limpeza de rate limits: {e}")
                await asyncio.sleep(60)
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit_type: str = "token"
    ) -> Tuple[bool, int, float]:
        """
        Verifica se uma requisição está dentro do rate limit.
        
        Args:
            identifier: Identificador único (token, IP, etc)
            limit_type: Tipo de limite (token, ip, global)
            
        Returns:
            Tupla com:
            - bool: Se a requisição é permitida
            - int: Requisições restantes
            - float: Tempo até reset em segundos
        """
        current_time = time.time()
        
        if identifier not in self.tokens:
            self.tokens[identifier] = RateLimitInfo(
                requests=self.burst_size,
                reset_time=current_time + self.window_size,
                last_request=current_time
            )
        
        info = self.tokens[identifier]
        
        # Calcula tokens disponíveis
        elapsed = current_time - info.last_request
        new_tokens = int(elapsed * self.requests_per_second)
        
        if new_tokens > 0:
            info.requests = min(
                info.requests + new_tokens,
                self.burst_size
            )
            info.last_request = current_time
        
        # Verifica se tem tokens disponíveis
        if info.requests > 0:
            info.requests -= 1
            allowed = True
        else:
            allowed = False
            RATE_LIMIT_HITS.labels(type=limit_type).inc()
        
        # Atualiza métrica
        RATE_LIMIT_REMAINING.labels(
            type=limit_type,
            identifier=identifier
        ).set(info.requests)
        
        return (
            allowed,
            info.requests,
            info.reset_time - current_time
        )

class JWTHandler:
    """Gerencia autenticação JWT."""
    
    def __init__(self):
        """Inicializa o handler JWT."""
        self.secret_key = SECURITY_CONFIG["jwt"]["secret_key"]
        self.algorithm = SECURITY_CONFIG["jwt"]["algorithm"]
        self.access_token_expire = SECURITY_CONFIG["jwt"]["access_token_expire"]
        self.refresh_token_expire = SECURITY_CONFIG["jwt"]["refresh_token_expire"]
        
        # Cache de tokens revogados
        self.revoked_tokens: Dict[str, float] = {}
        self._cleanup_task = None
    
    def start_cleanup(self):
        """Inicia tarefa de limpeza periódica."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Loop de limpeza de tokens revogados expirados."""
        while True:
            try:
                current_time = time.time()
                expired = []
                
                for token, expiry in self.revoked_tokens.items():
                    if current_time > expiry:
                        expired.append(token)
                
                for token in expired:
                    del self.revoked_tokens[token]
                
                await asyncio.sleep(3600)  # Executa a cada hora
                
            except Exception as e:
                logger.error(f"Erro na limpeza de tokens revogados: {e}")
                await asyncio.sleep(3600)
    
    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Cria um token de acesso JWT.
        
        Args:
            data: Dados a serem codificados no token
            expires_delta: Tempo de expiração opcional
            
        Returns:
            Token JWT codificado
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire
            )
        
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
    
    def create_refresh_token(self, data: dict) -> str:
        """
        Cria um token de refresh JWT.
        
        Args:
            data: Dados a serem codificados no token
            
        Returns:
            Token JWT codificado
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire)
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
    
    def decode_token(self, token: str) -> dict:
        """
        Decodifica e valida um token JWT.
        
        Args:
            token: Token JWT a ser decodificado
            
        Returns:
            Dados decodificados do token
            
        Raises:
            HTTPException: Se o token for inválido
        """
        try:
            if token in self.revoked_tokens:
                raise HTTPException(
                    status_code=401,
                    detail="Token revogado"
                )
            
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token expirado"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Token inválido"
            )
    
    def revoke_token(self, token: str):
        """
        Revoga um token JWT.
        
        Args:
            token: Token JWT a ser revogado
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            expiry = payload.get("exp", time.time() + 3600)
            self.revoked_tokens[token] = expiry
            
        except jwt.InvalidTokenError:
            pass  # Ignora tokens inválidos

class SecurityHandler:
    """Handler principal de segurança."""
    
    def __init__(self):
        """Inicializa o handler de segurança."""
        self.bearer_scheme = HTTPBearer()
        self.jwt_handler = JWTHandler()
        
        # Rate limiters
        self.global_limiter = RateLimiter(
            requests_per_second=SECURITY_CONFIG["rate_limit"]["global"]["rps"],
            burst_size=SECURITY_CONFIG["rate_limit"]["global"]["burst"],
            window_size=SECURITY_CONFIG["rate_limit"]["global"]["window"]
        )
        
        self.ip_limiter = RateLimiter(
            requests_per_second=SECURITY_CONFIG["rate_limit"]["ip"]["rps"],
            burst_size=SECURITY_CONFIG["rate_limit"]["ip"]["burst"],
            window_size=SECURITY_CONFIG["rate_limit"]["ip"]["window"]
        )
        
        self.token_limiter = RateLimiter(
            requests_per_second=SECURITY_CONFIG["rate_limit"]["token"]["rps"],
            burst_size=SECURITY_CONFIG["rate_limit"]["token"]["burst"],
            window_size=SECURITY_CONFIG["rate_limit"]["token"]["window"]
        )
    
    def start(self):
        """Inicia handlers de segurança."""
        self.jwt_handler.start_cleanup()
        self.global_limiter.start_cleanup()
        self.ip_limiter.start_cleanup()
        self.token_limiter.start_cleanup()
    
    async def verify_token(
        self,
        credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> dict:
        """
        Verifica token de autenticação.
        
        Args:
            credentials: Credenciais HTTP
            
        Returns:
            Dados do token decodificado
            
        Raises:
            HTTPException: Se o token for inválido
        """
        return self.jwt_handler.decode_token(credentials.credentials)
    
    async def check_rate_limits(
        self,
        token_id: str,
        client_ip: str
    ) -> Tuple[int, float]:
        """
        Verifica todos os rate limits.
        
        Args:
            token_id: ID do token
            client_ip: IP do cliente
            
        Returns:
            Tupla com:
            - int: Requisições restantes (menor entre todos os limites)
            - float: Menor tempo até reset
            
        Raises:
            HTTPException: Se algum rate limit for excedido
        """
        # Verifica limite global
        global_allowed, global_remaining, global_reset = (
            await self.global_limiter.check_rate_limit("global", "global")
        )
        
        if not global_allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit global excedido",
                    "reset_in": global_reset
                }
            )
        
        # Verifica limite por IP
        ip_allowed, ip_remaining, ip_reset = (
            await self.ip_limiter.check_rate_limit(client_ip, "ip")
        )
        
        if not ip_allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit por IP excedido",
                    "reset_in": ip_reset
                }
            )
        
        # Verifica limite por token
        token_allowed, token_remaining, token_reset = (
            await self.token_limiter.check_rate_limit(token_id, "token")
        )
        
        if not token_allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit por token excedido",
                    "reset_in": token_reset
                }
            )
        
        # Retorna menor número de requisições restantes e menor reset
        remaining = min(global_remaining, ip_remaining, token_remaining)
        reset = min(global_reset, ip_reset, token_reset)
        
        return remaining, reset

# Instância global do handler de segurança
security_handler = SecurityHandler() 