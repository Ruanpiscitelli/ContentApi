#!/bin/bash
set -e

# Configurar variáveis de ambiente
export PYTHONPATH=/app:${PYTHONPATH}

# Verificar e criar diretórios necessários
echo "Verificando diretórios..."
for dir in data logs uploads cache temp; do
    if [ ! -d "/app/$dir" ]; then
        echo "Criando diretório /app/$dir"
        mkdir -p "/app/$dir"
        chmod 755 "/app/$dir"
    fi
done

# Aguardar dependências (se necessário)
echo "Aguardando dependências..."
sleep 5

# Executar migrações do banco de dados
echo "Executando migrações do banco de dados..."
cd /app/meu-servidor-ia/services/dashboard
alembic upgrade head

# Iniciar o serviço
echo "Iniciando o dashboard..."
cd /app/meu-servidor-ia/services/dashboard
if python3 -c "import uvloop" 2>/dev/null; then
    exec uvicorn app:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --loop uvloop \
        --http httptools \
        --timeout-keep-alive 120 \
        --limit-concurrency 1000 \
        --log-level info
else
    exec uvicorn app:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --timeout-keep-alive 120 \
        --limit-concurrency 1000 \
        --log-level info
fi 