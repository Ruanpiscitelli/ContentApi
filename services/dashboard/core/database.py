"""
Configuração do banco de dados e sessão SQLAlchemy.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import get_settings

settings = get_settings()

# Cria engine assíncrona
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True
)

# Configuração da sessão assíncrona
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
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
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all) 