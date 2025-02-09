# Build stage
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel AS builder

WORKDIR /build
COPY services/video_generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
COPY services/video_generator ./services/video_generator
COPY shared ./shared

RUN python3 download_models.py

RUN conda install -y -c conda-forge \
    transformers \
    diffusers \
    accelerate \
    safetensors \
    xformers

EXPOSE 8000
CMD ["uvicorn", "services.video_generator.app:app", "--host", "0.0.0.0", "--port", "8000"]
