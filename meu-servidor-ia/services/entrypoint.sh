#!/bin/bash
set -e

# Verifica se os modelos necessários existem
check_models() {
    required_models=(
        "/app/models/voz/fish-speech-1.4.3"
    )

    for model in "${required_models[@]}"; do
        if [ ! -d "$model" ]; then
            echo "ERRO: Modelo necessário não encontrado: $model"
            echo "Por favor, baixe os modelos manualmente antes de iniciar o container"
            exit 1
        fi
    done
}

# Verifica modelos apenas se SKIP_MODEL_DOWNLOAD não estiver definido
if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then
    check_models
fi

# Iniciar a aplicação
exec uvicorn services.voice_generator.app:app --host 0.0.0.0 --port 9000 