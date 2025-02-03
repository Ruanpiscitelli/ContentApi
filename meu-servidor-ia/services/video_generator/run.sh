#!/bin/bash

# Carrega variáveis de ambiente
set -a
source .env
set +a

# Verifica se os modelos existem
if [ ! -d "checkpoints/fasthunyuan" ]; then
    echo "Baixando modelos do FastHunyuan..."
    python download_models.py
fi

# Verifica se diretórios necessários existem
mkdir -p cache references templates

# Verifica se Redis está instalado
if ! command -v redis-server &> /dev/null; then
    echo "Redis não encontrado. Instalando..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install redis
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y redis-server
    fi
fi

# Inicia o Redis se não estiver rodando
if ! pgrep redis-server > /dev/null; then
    echo "Iniciando Redis..."
    redis-server &
    sleep 2
fi

# Configura otimizações de CUDA
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Habilita otimizações de memória
if [ "${ENABLE_VRAM_OPTIMIZATIONS}" = "true" ]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
fi

# Inicia o serviço
echo "Iniciando serviço FastHunyuan..."
uvicorn app:app \
    --host ${API_HOST:-0.0.0.0} \
    --port ${API_PORT:-8000} \
    --workers 1 \
    --log-level ${LOG_LEVEL:-info} 