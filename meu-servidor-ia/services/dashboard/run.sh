#!/bin/bash

# Configurar variáveis de ambiente
export PYTHONPATH=/app:${PYTHONPATH}

# Aguardar dependências (se necessário)
echo "Aguardando dependências..."
sleep 5

# Executar migrações do banco de dados
echo "Executando migrações do banco de dados..."
cd /app/meu-servidor-ia/services/dashboard
alembic upgrade head

# Iniciar o serviço
echo "Iniciando o serviço de dashboard..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload 