# Utilize uma imagem base com CUDA e ferramentas de desenvolvimento (incluindo nvcc)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Evite prompts durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instale Python 3.10 e dependências essenciais
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    curl \
    libglib2.0-0 \
    git \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && rm -rf /var/lib/apt/lists/*

# Configure o Python 3.10 como padrão
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos de requisitos
COPY requirements-base.txt requirements.txt ./

# Instale primeiro as dependências base (PyTorch e fundamentais)
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-base.txt

# Instale o restante das dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código da aplicação
COPY . .

# Comando para iniciar o serviço utilizando Gunicorn com UvicornWorker (2 workers)
CMD ["gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2"]
