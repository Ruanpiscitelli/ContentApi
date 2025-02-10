FROM contentapi-base

# Copiar e instalar requirements específicos
COPY --chown=appuser:appuser meu-servidor-ia/services/video-editor/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copiar código
COPY --chown=appuser:appuser meu-servidor-ia/services/video-editor ./meu-servidor-ia/services/video-editor
COPY --chown=appuser:appuser meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretórios necessários
RUN mkdir -p /app/cache/videos && \
    chmod -R 755 /app/cache

CMD ["python3", "-m", "uvicorn", "meu-servidor-ia.services.video_editor.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]