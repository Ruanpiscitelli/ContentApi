#!/bin/bash

# Configurar variáveis de ambiente
export PYTHONPATH=/app:${PYTHONPATH}

# Aguardar dependências (se necessário)
echo "Aguardando dependências..."
sleep 5

# Iniciar o serviço
echo "Iniciando o serviço de geração de imagens..."
cd /app/meu-servidor-ia/services/image_generation
python -m uvicorn app:app --host 0.0.0.0 --port 8002 --reload 