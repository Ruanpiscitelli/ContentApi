#!/bin/bash
set -e

# Função para verificar dependências
check_dependencies() {
    command -v python3 >/dev/null 2>&1 || { echo "Python3 não encontrado. Por favor, instale o Python3."; exit 1; }
    command -v pip3 >/dev/null 2>&1 || { echo "Pip3 não encontrado. Por favor, instale o Pip3."; exit 1; }
}

# Verifica dependências básicas
check_dependencies

# Carrega variáveis de ambiente se existir
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Verifica se os modelos existem
if [ ! -d "models/fish-speech-1.5" ]; then
    echo "Baixando modelos..."
    python3 download_models.py
    if [ $? -ne 0 ]; then
        echo "Erro ao baixar modelos. Verifique os logs para mais detalhes."
        exit 1
    fi
fi

# Cria diretórios necessários com permissões corretas
for dir in cache temp "logs/tensorboard" references; do
    mkdir -p "$dir"
    chmod 755 "$dir"
done

# Configura variáveis de ambiente para CUDA
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}

echo "Iniciando servidor de geração de voz..."
# Inicia o servidor
if python3 -c "import uvloop" 2>/dev/null; then
    exec uvicorn app:app \
        --host ${API_HOST:-0.0.0.0} \
        --port ${API_PORT:-8003} \
        --workers 1 \
        --loop uvloop \
        --http httptools \
        --timeout-keep-alive 120 \
        --limit-concurrency 1000 \
        --log-level info
else
    exec uvicorn app:app \
        --host ${API_HOST:-0.0.0.0} \
        --port ${API_PORT:-8003} \
        --workers 1 \
        --timeout-keep-alive 120 \
        --limit-concurrency 1000 \
        --log-level info
fi