# Build stage
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_CUDA_ARCHITECTURES="89"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    cmake \
    git \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install torch first with CUDA 12.1 support
RUN pip3 install --no-cache-dir torch==2.2.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install triton and xformers for optimizations
RUN pip3 install --no-cache-dir triton==2.1.0 xformers==0.0.23

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/cache/transformers \
    TORCH_HOME=/app/cache/torch \
    HF_HOME=/app/cache/huggingface

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

WORKDIR /app

# Set up app directory and cache
RUN mkdir -p /app/cache/{transformers,torch,huggingface} \
    && mkdir -p /app/{logs,uploads,temp} \
    && chown -R appuser:appuser /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy application code
COPY . /app/services/voice_generator

USER appuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "services.voice_generator.app:app", "--host", "0.0.0.0", "--port", "8000"]