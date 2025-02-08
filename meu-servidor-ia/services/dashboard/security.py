"""
Módulo de segurança e autenticação.
"""
from datetime import datetime, timedelta
from typing import Optional
import secrets
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from .models.database import User, APIKey
from .models.database_config import get_db

# Configurações de segurança
SECRET_KEY = "your-secret-key-here"  # Deve ser substituído por variável de ambiente
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Contexto para hash de senhas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme para login
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica se a senha está correta."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Gera hash da senha."""
    return pwd_context.hash(password)

def generate_api_key() -> str:
    """Gera uma nova chave API."""
    return secrets.token_urlsafe(32)

async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """Autentica um usuário."""
    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Cria um token JWT."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Obtém o usuário atual a partir do token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais inválidas",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
        
    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    return user

async def verify_api_key(
    api_key: str,
    db: AsyncSession = Depends(get_db)
) -> Optional[APIKey]:
    """Verifica se uma chave API é válida."""
    result = await db.execute(
        select(APIKey).where(
            APIKey.key == api_key,
            APIKey.is_active == True
        )
    )
    key = result.scalar_one_or_none()
    
    if not key:
        return None
        
    # Atualiza último uso
    key.last_used = datetime.utcnow()
    await db.commit()
    
    return key 