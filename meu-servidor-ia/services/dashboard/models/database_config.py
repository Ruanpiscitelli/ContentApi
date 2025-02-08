"""
Configuração do banco de dados.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool

# URL do banco SQLite
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./dashboard.db"

# Cria engine assíncrono
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    poolclass=AsyncAdaptedQueuePool,
    pool_pre_ping=True,
    echo=False
)

# Cria fábrica de sessões assíncronas
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

async def get_db():
    """
    Dependency para obter sessão do banco.
    Deve ser usado com FastAPI Depends.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close() 