"""
Este módulo contém a configuração e funções do banco de dados do dashboard.
"""

import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool
from datetime import datetime

from .config import DATABASE_URL
from .models import Base
from .utils import logger

# Cria o engine do banco de dados
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Define como True para ver as queries SQL no console
    future=True,
    poolclass=AsyncAdaptedQueuePool,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800  # Recicla conexões a cada 30 minutos
)

# Cria a fábrica de sessões
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

async def init_db() -> None:
    """
    Inicializa o banco de dados criando todas as tabelas necessárias.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Banco de dados inicializado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao inicializar o banco de dados: {str(e)}")
        raise

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Obtém uma sessão do banco de dados.
    
    Yields:
        AsyncSession: Sessão assíncrona do banco de dados.
    """
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Erro na sessão do banco de dados: {str(e)}")
            raise
        finally:
            await session.close()

async def cleanup_db() -> None:
    """
    Limpa recursos do banco de dados.
    """
    try:
        await engine.dispose()
        logger.info("Recursos do banco de dados liberados com sucesso")
    except Exception as e:
        logger.error(f"Erro ao limpar recursos do banco de dados: {str(e)}")
        raise

async def check_db_connection() -> bool:
    """
    Verifica se a conexão com o banco de dados está funcionando.
    
    Returns:
        bool: True se a conexão está funcionando, False caso contrário.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Erro ao verificar conexão com o banco de dados: {str(e)}")
        return False

async def backup_db() -> str:
    """
    Cria um backup do banco de dados.
    
    Returns:
        str: Caminho do arquivo de backup.
    """
    try:
        # Obtém o caminho do banco de dados SQLite
        db_path = DATABASE_URL.replace("sqlite+aiosqlite:///", "")
        
        # Cria o diretório de backup se não existir
        backup_dir = os.path.join(os.path.dirname(db_path), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Gera o nome do arquivo de backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"dashboard_backup_{timestamp}.db")
        
        # Copia o arquivo do banco de dados
        import shutil
        shutil.copy2(db_path, backup_path)
        
        logger.info(f"Backup do banco de dados criado em: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Erro ao criar backup do banco de dados: {str(e)}")
        raise

async def restore_db(backup_path: str) -> None:
    """
    Restaura o banco de dados a partir de um backup.
    
    Args:
        backup_path (str): Caminho do arquivo de backup.
    """
    try:
        # Verifica se o arquivo de backup existe
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Arquivo de backup não encontrado: {backup_path}")
        
        # Obtém o caminho do banco de dados atual
        db_path = DATABASE_URL.replace("sqlite+aiosqlite:///", "")
        
        # Fecha todas as conexões
        await engine.dispose()
        
        # Restaura o backup
        import shutil
        shutil.copy2(backup_path, db_path)
        
        # Reinicializa o banco de dados
        await init_db()
        
        logger.info(f"Banco de dados restaurado com sucesso a partir de: {backup_path}")
    except Exception as e:
        logger.error(f"Erro ao restaurar banco de dados: {str(e)}")
        raise

async def vacuum_db() -> None:
    """
    Executa VACUUM no banco de dados para otimizar o espaço em disco.
    """
    try:
        async with engine.begin() as conn:
            await conn.execute("VACUUM")
        logger.info("VACUUM executado com sucesso no banco de dados")
    except Exception as e:
        logger.error(f"Erro ao executar VACUUM no banco de dados: {str(e)}")
        raise