FROM contentapi-base

# Copiar requirements específicos
COPY --chown=appuser:appuser services/video_generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY --chown=appuser:appuser services/video_generator ./services/video_generator
COPY --chown=appuser:appuser shared ./shared

WORKDIR /app/services/video_generator

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]