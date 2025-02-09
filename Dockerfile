FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Configurações de ambiente
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar resto do código
COPY . .

# Criar diretórios necessários
RUN mkdir -p /app/models/cache /app/models/hub /app/data /app/logs && \
    chmod -R 777 /app/models /app/data /app/logs

# Configurações HuggingFace
ENV TRANSFORMERS_CACHE=/app/models/cache
ENV HF_HOME=/app/models/hub

# Otimizações CUDA
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_ALLOW_TF32=1

# Porta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para iniciar a aplicação
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]