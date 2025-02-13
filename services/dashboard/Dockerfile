FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV USER_NAME=appuser

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libcap2-bin && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 ${USER_NAME}

WORKDIR /app

# Install requirements first
COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir \
    pydantic==1.10.12 \
    fastapi==0.95.2 \
    anyio==3.7.1 \
    starlette==0.27.0 \
    -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser ./services/dashboard /app/services/dashboard
COPY --chown=appuser:appuser ./shared /app/shared

# Permissions and ports
RUN mkdir -p /app/logs && \
    chown -R ${USER_NAME}:${USER_NAME} /app && \
    chmod 755 /app/logs && \
    setcap 'cap_net_bind_service=+ep' /usr/local/bin/python3.10

USER ${USER_NAME}

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]