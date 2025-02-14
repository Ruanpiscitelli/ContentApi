"""
Este módulo contém as dependências utilizadas pelo dashboard.
"""

from typing import AsyncGenerator, Optional
from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
from sqlalchemy import select

from .database import get_db
from .models import User
from .exceptions import AuthenticationError
from .config import SECRET_KEY, ALGORITHM
from .utils import logger
from .security import verify_password

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Obtém o usuário atual a partir do token JWT.
    
    Args:
        request (Request): Objeto de requisição.
        db (AsyncSession): Sessão do banco de dados.
    
    Returns:
        User: Objeto do usuário atual.
    
    Raises:
        AuthenticationError: Se o token for inválido ou o usuário não for encontrado.
    """
    # Tenta obter o token do cookie primeiro
    token = request.cookies.get("access_token")
    
    if not token:
        # Se não encontrar no cookie, tenta obter do header Authorization
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
    
    if not token:
        raise AuthenticationError("Token não fornecido")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationError("Token inválido")
    except JWTError as e:
        logger.error(f"Erro ao decodificar token: {str(e)}")
        raise AuthenticationError("Token inválido")
    
    user = await db.query(User).filter(User.username == username).first()
    if user is None:
        raise AuthenticationError("Usuário não encontrado")
    
    return user

async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Tenta obter o usuário atual, mas não falha se não encontrar.
    
    Args:
        request (Request): Objeto de requisição.
        db (AsyncSession): Sessão do banco de dados.
    
    Returns:
        Optional[User]: Objeto do usuário atual ou None se não autenticado.
    """
    try:
        return await get_current_user(request, db)
    except AuthenticationError:
        return None

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Obtém uma sessão do banco de dados.
    
    Yields:
        AsyncSession: Sessão assíncrona do banco de dados.
    """
    async with get_db() as session:
        try:
            yield session
        finally:
            await session.close()

def get_pagination_params(
    page: int = 1,
    per_page: int = 10,
    max_per_page: int = 100
) -> tuple[int, int]:
    """
    Obtém e valida os parâmetros de paginação.
    
    Args:
        page (int): Número da página.
        per_page (int): Itens por página.
        max_per_page (int): Máximo de itens por página permitido.
    
    Returns:
        tuple[int, int]: Tupla contendo página e itens por página validados.
    """
    if page < 1:
        page = 1
    
    if per_page < 1:
        per_page = 10
    elif per_page > max_per_page:
        per_page = max_per_page
    
    return page, per_page

def get_sort_params(
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    allowed_fields: list[str] = []
) -> tuple[Optional[str], bool]:
    """
    Obtém e valida os parâmetros de ordenação.
    
    Args:
        sort_by (Optional[str]): Campo para ordenação.
        sort_order (Optional[str]): Direção da ordenação ('asc' ou 'desc').
        allowed_fields (list[str]): Lista de campos permitidos para ordenação.
    
    Returns:
        tuple[Optional[str], bool]: Tupla contendo campo de ordenação e flag de ordem descendente.
    """
    if not sort_by or sort_by not in allowed_fields:
        return None, False
    
    is_desc = sort_order and sort_order.lower() == 'desc'
    return sort_by, is_desc

def get_filter_params(
    filters: dict,
    allowed_filters: list[str]
) -> dict:
    """
    Filtra e valida os parâmetros de filtro.
    
    Args:
        filters (dict): Dicionário com os filtros.
        allowed_filters (list[str]): Lista de filtros permitidos.
    
    Returns:
        dict: Dicionário com os filtros validados.
    """
    return {
        k: v for k, v in filters.items()
        if k in allowed_filters and v is not None
    }

def get_search_param(
    search: Optional[str] = None,
    min_length: int = 3
) -> Optional[str]:
    """
    Valida o parâmetro de busca.
    
    Args:
        search (Optional[str]): Termo de busca.
        min_length (int): Comprimento mínimo do termo.
    
    Returns:
        Optional[str]: Termo de busca validado ou None.
    """
    if not search or len(search.strip()) < min_length:
        return None
    return search.strip()

async def authenticate_user(username: str, password: str, db: AsyncSession) -> Optional[User]:
    """
    Autentica um usuário verificando suas credenciais.
    """
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user 