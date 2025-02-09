#!/bin/bash
set -e

# Iniciar a aplicação
exec uvicorn services.voice_generator.app:app --host 0.0.0.0 --port 9000 