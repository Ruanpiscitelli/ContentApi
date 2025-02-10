FROM contentapi-base

# Copiar e instalar requirements específicos – apenas se houver dependências exclusivas do serviço
COPY --chown=appuser:appuser meu-servidor-ia/services/video-editor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código específico do serviço e código compartilhado
COPY --chown=appuser:appuser meu-servidor-ia/services/video-editor ./meu-servidor-ia/services/video-editor
COPY --chown=appuser:appuser meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretórios específicos, se necessário
RUN mkdir -p /app/cache/videos && chmod -R 755 /app/cache

CMD ["python3", "-m", "uvicorn", "meu-servidor-ia.services.video_editor.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]