# Stage 1: Builder
FROM python:3.10-slim as builder

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY meu-servidor-ia/services/text_generation/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Instalar dependências do sistema necessárias em runtime
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    libcudart11.0 \
    libnvinfer8 \
    libnvonnxparsers8 \
    libnvparsers8 \
    libnvinfer-plugin8 \
    && rm -rf /var/lib/apt/lists/*

# Configurar variáveis de ambiente para CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# Copiar pacotes Python instalados do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar código do serviço
COPY meu-servidor-ia/services/text_generation ./meu-servidor-ia/services/text_generation
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretórios necessários
RUN mkdir -p /app/models /app/cache/text /app/uploads/text

# Configurar permissões
RUN chmod -R 755 /app/models /app/cache /app/uploads

# Copiar e configurar script de entrada
COPY meu-servidor-ia/services/text_generation/run.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Configurar health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expor porta do serviço
EXPOSE 8001

# Definir ponto de entrada
ENTRYPOINT ["/entrypoint.sh"]