"""
Configurações e funções de segurança.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from .config import get_settings
from .exceptions import AuthenticationError, AuthorizationError

settings = get_settings()

# Configuração do contexto de senha
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configuração do OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifica se a senha fornecida corresponde ao hash armazenado.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Gera um hash bcrypt para a senha fornecida.
    """
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Cria um token JWT com os dados fornecidos.
    
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
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt

async def verify_token(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Verifica e decodifica um token JWT.
    
    Args:
        token: Token JWT a ser verificado
        
    Returns:
        Dados decodificados do token
        
    Raises:
        HTTPException: Se o token for inválido ou expirado
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def verify_api_key(request: Request) -> None:
    """Verifica se a chave API é válida."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise AuthenticationError("Chave API não fornecida")
        
    # Aqui você deve implementar a lógica de verificação da chave API
    # Por exemplo, verificar no banco de dados se a chave existe e está ativa
    
    if not is_valid_api_key(api_key):
        raise AuthorizationError("Chave API inválida")

def is_valid_api_key(api_key: str) -> bool:
    """Verifica se a chave API é válida."""
    # Implemente a lógica de validação da chave API
    # Por exemplo, verificar formato, expiração, etc.
    return True  # Placeholder 