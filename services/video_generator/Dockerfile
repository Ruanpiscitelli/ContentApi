FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV TZ=America/Sao_Paulo
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

# Instalar dependências base primeiro
COPY requirements.txt .
RUN pip3 install --no-cache-dir packaging setuptools wheel && \
    pip3 install --no-cache-dir torch==2.2.1 && \
    TORCH_CUDA_ARCH_LIST="8.6" pip3 install --no-cache-dir -r requirements.txt

COPY download_models.py .
COPY services/video_generator ./services/video_generator
COPY shared ./shared

RUN python3 download_models.py

EXPOSE 8000
CMD ["uvicorn", "services.video_generator.app:app", "--host", "0.0.0.0", "--port", "8000"]
