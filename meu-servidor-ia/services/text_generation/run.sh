#!/bin/bash

# Configurar variáveis de ambiente
export PYTHONPATH=/app:${PYTHONPATH}

# Aguardar dependências (se necessário)
echo "Aguardando dependências..."
sleep 5

# Iniciar o serviço
echo "Iniciando o serviço de geração de texto..."
cd /app/meu-servidor-ia/services/text_generation
if python -c "import uvloop" 2>/dev/null; then
    exec uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1 --loop uvloop --http httptools
else
    exec uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
fi