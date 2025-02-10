#!/bin/bash
set -e

# Configurar variáveis de ambiente
export PYTHONPATH=/app:${PYTHONPATH}

# Verificar e criar diretórios necessários
echo "Verificando diretórios..."
for dir in models cache/text uploads/text logs; do
    if [ ! -d "/app/$dir" ]; then
        echo "Criando diretório /app/$dir"
        mkdir -p "/app/$dir"
        chmod 755 "/app/$dir"
    fi
done

# Aguardar dependências (se necessário)
echo "Aguardando dependências..."
sleep 5

# Verificar dependências Python
echo "Verificando dependências Python..."
python3 -c "import fastapi" || { echo "FastAPI não encontrado. Instalando dependências..."; pip3 install -r requirements.txt; }

# Iniciar o serviço
echo "Iniciando o serviço de geração de texto..."
cd /app/meu-servidor-ia/services/text_generation
if python3 -c "import uvloop" 2>/dev/null; then
    exec uvicorn app:app \
        --host 0.0.0.0 \
        --port 8001 \
        --workers 1 \
        --loop uvloop \
        --http httptools \
        --timeout-keep-alive 120 \
        --limit-concurrency 1000 \
        --log-level info
else
    exec uvicorn app:app \
        --host 0.0.0.0 \
        --port 8001 \
        --workers 1 \
        --timeout-keep-alive 120 \
        --limit-concurrency 1000 \
        --log-level info
fi