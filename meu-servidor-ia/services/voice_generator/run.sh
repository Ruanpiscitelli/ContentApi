#!/bin/bash

# Carrega variáveis de ambiente
set -a
source .env
set +a

# Verifica se os modelos existem
if [ ! -d "checkpoints/fish-speech-1.5" ]; then
    echo "Baixando modelos..."
    python download_models.py
fi

# Verifica se diretórios necessários existem
mkdir -p cache references

# Inicia o Redis se não estiver rodando
if ! pgrep redis-server > /dev/null; then
    echo "Iniciando Redis..."
    redis-server &
    sleep 2
fi

# Inicia o serviço
echo "Iniciando serviço de geração de voz..."
uvicorn app:app --host $API_HOST --port $API_PORT --workers 1 