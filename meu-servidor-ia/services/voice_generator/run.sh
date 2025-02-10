#!/bin/bash
# Entrypoint para o serviço de geração de voz.
# Este script inicia o servidor uvicorn que executa a aplicação FastAPI.
# A opção 'set -e' garante que o script pare imediatamente se ocorrer algum erro.

set -e

echo "Iniciando o serviço de geração de voz..."
# Inicia o servidor FastAPI usando uvicorn.
# Ajuste o caminho 'meu-servidor-ia.services.voice_generator.main:app' conforme a localização da sua aplicação.
uvicorn meu-servidor-ia.services.voice_generator.main:app --host 0.0.0.0 --port 8000