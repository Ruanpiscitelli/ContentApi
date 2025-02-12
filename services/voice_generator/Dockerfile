# Build stage
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up build directory
WORKDIR /build

# Install torch first
RUN pip3 install --no-cache-dir torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Install torchaudio separately
RUN pip3 install --no-cache-dir torchaudio==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Copy and install remaining requirements
COPY ./services/voice_generator/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python and create non-root user
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# Set up app directory
WORKDIR /app
USER appuser

# Copy Python packages and app code from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --chown=appuser:appuser ./services/voice_generator /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 