#!/bin/bash
set -e

# Função para verificar se os modelos já existem
check_models() {
    if [ ! -d "/app/models/voz/fish-speech-1.4.3" ]; then
        echo "Baixando modelos necessários..."
        python3 download_models.py
    else
        echo "Modelos já existem, pulando download..."
    fi
}

# Tentar baixar modelos
check_models

# Iniciar a aplicação
exec uvicorn services.voice_generator.app:app --host 0.0.0.0 --port 9000 