# Build stage
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel AS builder

WORKDIR /build
COPY services/text_generation/requirements.txt .

# Instalar com flags específicas para CUDA
RUN TORCH_CUDA_ARCH_LIST="8.6" pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV TZ=America/Sao_Paulo
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    python3-dev \
    ninja-build \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar dependências do build stage
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages

# Copiar arquivos do serviço
COPY download_models.py .
COPY services/text_generation ./services/text_generation
COPY shared ./shared

RUN python3 download_models.py

EXPOSE 8000
CMD ["uvicorn", "services.text_generation.app:app", "--host", "0.0.0.0", "--port", "8000"]
