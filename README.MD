Below is the updated README in English, reflecting the changes (single Docker container deployment and ngrok usage) and optimized for your vast.ai server with 4 RTX 4090 GPUs:

---

```markdown
# My AI Content Generation Server

This project aims to build a robust cloud server that provides multiple APIs for AI-driven content generation. Using FastAPI, the system offers endpoints for:

- **Image Generation** (using models like SDXL/Flux with LoRA injection)
- **Voice Generation/Cloning** (leveraging Fish Speech v2 for synthesis and cloning)
- **Video Generation with Hunyuan** (using JSON templates to define generation parameters)
- **JSON-Based Video Editing** (inspired by [json2video](https://json2video.com/docs/tutorial/))
- **Admin Dashboard** for managing services, uploading models, configuring endpoints, and monitoring logs/metrics

The system is designed to be highly scalable and performance-optimized with asynchronous execution and multi‑GPU support (using `torch.nn.DataParallel` on your 4 RTX 4090 GPUs). The entire application is containerized into a single Docker container for simplified deployment and is exposed externally via ngrok.

---

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation and Configuration](#installation-and-configuration)
  - [Local Development with Docker and Ngrok](#local-development-with-docker-and-ngrok)
- [Endpoint Usage](#endpoint-usage)
- [Templates](#templates)
- [Contribution](#contribution)
- [License](#license)
- [Contact](#contact)
- [Environment Variables](#environment-variables)
- [Security](#security)

---

## Features

- **FastAPI-Based APIs:** Each service is built with FastAPI, ensuring high performance and asynchronous support.
- **Template Support:** Pre-defined JSON templates for configuring image, voice, and video pipelines are stored in the `templates/` directory.
- **Optimized GPU Utilization:** The image generation pipeline is designed for high GPU efficiency, supporting parallel execution across multiple GPUs (leveraging 4 RTX 4090 GPUs via `torch.nn.DataParallel`).
- **Single Docker Container Deployment:** The entire application is packaged into a single Docker container to simplify deployment and management.
- **Ngrok Integration:** Easily expose your API endpoints externally for development, testing, or temporary access.
- **Admin Dashboard:** A web interface for managing services, uploading models, configuring endpoints, and monitoring logs and metrics.

---

## Recent Updates

### New Features
- Support for multiple image generation schedulers (DDIM, DPMSolver++, Euler, etc.)
- Two-level caching system (Redis + VRAM) for performance optimization
- Integration with ControlNet for conditional image generation
- Voice cloning with Fish Speech v2, optimized with CUDA and FP16
- Pre-configured templates for generating shorts and vertical videos

### Performance Optimizations
- Torch Compile for JIT optimization
- VAE Tiling for high-resolution image processing
- Attention Slicing to optimize VRAM usage
- Dynamic GPU resource management with automatic failover
- Distributed caching with Redis for models and audio
- Integration with MinIO/S3 for scalable storage
- Advanced monitoring using Prometheus

### Structural Changes
- New `models/cache` directory for local model storage
- Reorganized templates by category (image/voice/video)
- GPU-hour based rate limiting
- Healthchecks for all services
- Multi-stage Docker builds for optimized container images
- Non-root user configuration for enhanced security

---

## Technologies Used

- **Backend:** Python, FastAPI, Uvicorn, Gunicorn
- **AI Models & Libraries:** vLLM, Torch, Fish Speech v2, Hunyuan, SDXL
- **Cache:** Redis (for API response and data caching)
- **Storage:** Local filesystem or cloud storage (MinIO/S3)
- **Deployment:** Single Docker container
- **Proxy & Exposure:** Ngrok for external access
- **Monitoring:** Prometheus, Grafana, OpenTelemetry
- **Security:** JWT-based authentication

---

## Project Structure

```plaintext
my-ai-content-server/
├── Dockerfile                 # Single Docker container definition for the entire application
├── ngrok.yml                  # Ngrok configuration file (if using a config file)
├── templates/                 # JSON templates for different services
│   ├── image/
│   │   └── image_template1.json
│   ├── voice/
│   │   └── voice_template1.json
│   └── video/
│       ├── video_generator_template1.json
│       └── video_editor_template1.json
├── services/                  # Source code for various services
│   ├── image_generator/
│   │   ├── app.py             # FastAPI app for image generation
│   │   ├── config.py
│   │   └── requirements.txt
│   ├── voice_generator/
│   │   ├── app.py             # FastAPI app for voice generation
│   │   ├── fish_speech_wrapper.py
│   │   ├── config.py
│   │   └── requirements.txt
│   ├── video_generator/
│   │   ├── app.py             # FastAPI app for video generation
│   │   └── requirements.txt
│   └── video_editor/
│       ├── app.py             # FastAPI app for video editing
│       └── requirements.txt
└── dashboard/                 # Admin dashboard for monitoring and management
    ├── app.py
    ├── templates/
    │   ├── dashboard.html
    │   └── dashboard_login.html
    ├── static/
    │   └── js/
    │       └── charts.js
    └── requirements.txt
```

---

## Installation and Configuration

### Local Development with Docker and Ngrok

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/my-ai-content-server.git
   cd my-ai-content-server
   ```

2. **Build and Run the Docker Container:**

   ```bash
   docker build -t ai-content-server .
   docker run -p 8000:8000 ai-content-server
   ```

3. **Expose the API with Ngrok:**

   Once the container is running, expose your FastAPI endpoints using ngrok:

   ```bash
   ngrok http 8000
   ```

   Ngrok will provide a public URL (e.g., `https://abcd1234.ngrok.io`) that you can use to access your APIs externally.

4. **Accessing the Endpoints:**

   Use the ngrok URL for testing and external access. For example, if ngrok returns `https://abcd1234.ngrok.io`, your image generation endpoint will be accessible at `https://abcd1234.ngrok.io/generate-image`.

---

## Endpoint Usage

### Image Generation with ControlNet
```bash
curl -X POST https://your-ngrok-url/generate-image \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "futuristic cityscape",
    "template_id": "image/ultrarealistic",
    "controlnet": {
      "model_id": "lllyasviel/control_v11p_sd15_canny",
      "image_url": "https://example.com/reference.jpg",
      "preprocessor": "canny"
    }
  }'
```

### Voice Cloning with Fish Speech v2
```bash
curl -X POST https://your-ngrok-url/generate-voice \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -F "texto=Hello, this is a voice cloning test" \
  -F "use_voice_clone=true" \
  -F "sample=@/path/to/reference.wav" \
  -F "template_id=voice/youtube"
```

### Video Shorts Generation
```bash
curl -X POST https://your-ngrok-url/edit-video \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "video/editor/shorts",
    "input": "input_video.mp4",
    "format": "vertical",
    "resolution": "1080x1920"
  }'
```

---

## Templates

Templates are JSON files located in the `templates/` directory that define the parameters for each pipeline.

Example `image_template1.json`:
```json
{
  "template_id": "image_template1",
  "model": "sdxl",
  "num_inference_steps": 60,
  "loras": [
    {"path": "loras/lora_face.pt", "scale": 0.8},
    {"path": "loras/lora_style.pt", "scale": 1.0}
  ]
}
```

Templates for voice and video generation/editing follow a similar structure, including all necessary parameters for each workflow.

---

## Contribution

Contributions are welcome! Feel free to open issues or submit pull requests. Please adhere to the project conventions and structure when adding new features.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or suggestions, please reach out at your-email@domain.com.

---

## Environment Variables

```yaml
# API Authentication Token
API_TOKEN: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # JWT token for authentication

# CUDA/GPU Settings
TORCH_COMPILE_MODE: "reduce-overhead"  # JIT optimization setting
ENABLE_VAE_TILING: "true"             # For high-resolution images
GPU_RATE_LIMIT: "100"                 # Requests per GPU-hour
```

---

## Security

This system uses JWT-based token authentication. To generate a new token, run:

```bash
python -c "import jwt; print(jwt.encode({'service': 'ai-content-api', 'iat': 1710954800}, 'your-secret-key', algorithm='HS256'))"
```

Ensure the generated token is added to your `.env` file and included in the `Authorization: Bearer <token>` header for all API requests.
```

---

This README provides a comprehensive overview of the project, covering its features, technologies, structure, and deployment instructions tailored for a single Docker container deployment with ngrok exposure. Enjoy building your AI content generation server!



# 1. Primeiro, parar e remover todos os containers e volumes
docker-compose down -v

# 2. Limpar todo o sistema Docker
docker system prune -af --volumes

# 3. Executar o script de download dos modelos
bash download_models.sh

# 4. Verificar se os modelos foram baixados corretamente
ls -la models/*

# 5. Reconstruir sem cache
docker-compose build --no-cache

# 6. Iniciar os serviços
docker-compose up



./start-services.sh

./setup_services.sh