# Copiar código do serviço
COPY meu-servidor-ia/services/video-editor ./meu-servidor-ia/services/video-editor
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Ajustar o script de entrada para o caminho correto
RUN echo '#!/bin/bash\n\
cd /app/meu-servidor-ia/services/video-editor\n\
if python3 -c "import uvloop" 2>/dev/null; then\n\
    exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools\n\
else\n\
    exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1\n\
fi' > /entrypoint.sh && \
    chmod +x /entrypoint.sh 