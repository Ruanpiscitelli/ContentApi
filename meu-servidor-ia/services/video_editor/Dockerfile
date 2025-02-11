FROM contentapi-base

# Definir variáveis de ambiente
ENV PYTHONPATH=/app
ENV USER_NAME=appuser

# Copiar requirements do serviço
COPY --chown=appuser:appuser services/video_editor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código do serviço e shared
COPY --chown=appuser:appuser services/video_editor ./services/video_editor
COPY --chown=appuser:appuser shared ./shared#Relativo ao contexto ./meu-servidor-ia

# Criar diretórios específicos
RUN mkdir -p /app/cache/videos && \
    chown -R ${USER_NAME}:${USER_NAME} /app/cache

WORKDIR /app/services/video_editor

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]