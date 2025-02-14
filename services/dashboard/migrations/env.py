"""
Ambiente de execução do Alembic.
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from ..core.config import get_settings
from ..models.database import Base

# Carrega configurações do alembic.ini
config = context.config

# Configura logging do Alembic
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Adiciona os modelos que serão versionados
target_metadata = Base.metadata

# Obtém URL do banco de dados das configurações
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

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

def do_run_migrations(connection):
    """Executa as migrações."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True
    )
    
    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online() -> None:
    """
    Executa migrações em modo 'online'.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = settings.DATABASE_URL
    connectable = AsyncEngine(
        engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            future=True,
        )
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online()) 