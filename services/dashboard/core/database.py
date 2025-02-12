"""
Configuração do banco de dados e sessão SQLAlchemy.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import AsyncAdaptedQueuePool

from .settings import settings

# Cria engine assíncrona
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    poolclass=AsyncAdaptedQueuePool,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

# Configuração da sessão assíncrona
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Base para modelos SQLAlchemy
Base = declarative_base()

async def get_session() -> AsyncSession:
    """
    Dependency para injetar sessões do banco de dados.
    Gerencia o ciclo de vida da sessão automaticamente.
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """
    Inicializa o banco de dados criando todas as tabelas.
    Deve ser chamado na inicialização da aplicação.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        raise Exception(f"Erro ao inicializar banco de dados: {str(e)}")