#!/bin/bash
set -e

# Função para verificar se modelo existe
check_model() {
    if [ ! -d "$1" ]; then
        echo "ERRO: Modelo necessário não encontrado: $1"
        if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then
            exit 1
        fi
    fi
}

# Verificar modelos antes de iniciar
for model in /app/models/*; do
    check_model "$model"
done

# Iniciar aplicação
exec "$@" 