#!/bin/bash
set -e

# Função para verificar dependências
check_dependencies() {
    command -v python3 >/dev/null 2>&1 || { echo "Python3 não encontrado. Por favor, instale o Python3."; exit 1; }
    command -v pip3 >/dev/null 2>&1 || { echo "Pip3 não encontrado. Por favor, instale o Pip3."; exit 1; }
}

# Função para verificar e criar diretórios
setup_directories() {
    local dirs=("$@")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "Criando diretório $dir"
            mkdir -p "$dir"
            chmod 755 "$dir"
        fi
    done
}

# Verifica dependências básicas
check_dependencies

# Carrega variáveis de ambiente se existir
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Configurar diretórios necessários
DIRS=(
    "/app/models/fish-speech-1.5"
    "/app/cache"
    "/app/temp"
    "/app/logs/tensorboard"
    "/app/references"
    "/app/uploads/voice"
)

echo "Configurando diretórios..."
setup_directories "${DIRS[@]}"

# Verifica se os modelos existem
if [ ! -d "/app/models/fish-speech-1.5" ] && [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then
    echo "Baixando modelos..."
    python3 download_models.py
    if [ $? -ne 0 ]; then
        echo "Erro ao baixar modelos. Verifique os logs para mais detalhes."
        exit 1
    fi
fi

# Verificar dependências Python
echo "Verificando dependências Python..."
python3 -c "import fastapi" || { echo "FastAPI não encontrado. Instalando dependências..."; pip3 install -r requirements.txt; }

# Configura variáveis de ambiente para CUDA
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}

echo "Iniciando servidor de geração de voz..."
# Inicia o servidor
if python3 -c "import uvloop" 2>/dev/null; then
    exec python3 -m uvicorn app:app \
        --host ${API_HOST:-0.0.0.0} \
        --port ${API_PORT:-8003} \
        --workers 1 \
        --loop uvloop \
        --http httptools \
        --timeout-keep-alive 120 \
        --limit-concurrency 1000 \
        --log-level info
else
    exec python3 -m uvicorn app:app \
        --host ${API_HOST:-0.0.0.0} \
        --port ${API_PORT:-8003} \
        --workers 1 \
        --timeout-keep-alive 120 \
        --limit-concurrency 1000 \
        --log-level info
fi