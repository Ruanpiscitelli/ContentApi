"""
Ambiente de execução do Alembic.
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from ..core.config import settings
from ..models.database import Base

# Carrega configurações do alembic.ini
config = context.config

# Configura logging do Alembic
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Adiciona os modelos que serão versionados
target_metadata = Base.metadata

# Obtém URL do banco de dados das configurações
config.set_main_option("sqlalchemy.url", settings.database_url)

def run_migrations_offline() -> None:
    """
    Executa migrações em modo 'offline'.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations() 