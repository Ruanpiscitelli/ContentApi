import asyncio
import base64
import json
import os
import time
import uuid
import logging
from io import BytesIO
from pathlib import Path
import concurrent.futures

import torch
import torchvision
from fastapi import FastAPI, HTTPException, Header, Depends, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple

from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import bitsandbytes as bnb
from accelerate import cpu_offload

# Importa o cliente do MinIO e erros
from minio import Minio
from minio.error import S3Error

from shared.gpu_utils import cuda_memory_manager, optimize_gpu_settings

app = FastAPI()

# Configurações do modelo
MODEL_ID = "FastVideo/FastHunyuan-diffusers"  # Modelo otimizado do FastVideo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
USE_NF4 = True  # Usar quantização NF4 para menor uso de VRAM
USE_CPU_OFFLOAD = True  # Habilitar CPU offload para economizar VRAM

# Resoluções suportadas (altura, largura)
SUPPORTED_RESOLUTIONS: Dict[str, List[Tuple[str, str]]] = {
    "9:16": [("544", "960"), ("720", "1280")],  # altura:largura
    "16:9": [("960", "544"), ("1280", "720")],  # largura:altura
    "4:3": [("624", "832"), ("1104", "832")],
    "3:4": [("832", "624"), ("832", "1104")],
    "1:1": [("720", "720"), ("960", "960")]
}

######################################
# 1. Validação de Token
######################################
# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Obtém o token da variável de ambiente
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError(
        "A variável de ambiente API_TOKEN não está configurada. "
        "Por favor, configure um token seguro para autenticação da API."
    )

def verify_bearer_token(authorization: str = Header(None)):
    """
    Valida o header Authorization e extrai o token.
    Implementa validação segura com tratamento de erros adequado.
    
    Args:
        authorization: Header de autorização no formato "Bearer <token>"
        
    Returns:
        str: Token validado
        
    Raises:
        HTTPException: Se o token for inválido ou não autorizado
    """
    if not authorization:
        logger.warning("Tentativa de acesso sem header de autorização")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Header de autorização ausente",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            logger.warning(f"Esquema de autenticação inválido: {scheme}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Esquema de autenticação inválido. Use 'Bearer'",
                headers={"WWW-Authenticate": "Bearer"}
            )
    except ValueError:
        logger.warning("Header de autorização malformado")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Header de autorização malformado",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    if not token:
        logger.warning("Token vazio recebido")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token não fornecido",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    if not token == API_TOKEN:
        logger.warning("Tentativa de acesso com token inválido")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    return token

######################################
# 2. Modelo de Requisição e Templates
######################################
class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(
        default="bad quality, worse quality, low quality",
        description="Prompt negativo para guiar a geração"
    )
    duration: Optional[int] = Field(
        default=16,
        ge=8,
        le=128,
        description="Duração do vídeo em frames (múltiplo de 8)"
    )
    height: Optional[int] = Field(
        default=576,
        ge=320,
        le=1280,
        description="Altura do vídeo (múltiplo de 64)"
    )
    width: Optional[int] = Field(
        default=1024,
        ge=320,
        le=1280,
        description="Largura do vídeo (múltiplo de 64)"
    )
    num_inference_steps: Optional[int] = Field(
        default=50,
        ge=1,
        le=100,
        description="Número de passos de inferência"
    )
    guidance_scale: Optional[float] = Field(
        default=9.0,
        ge=1.0,
        le=20.0,
        description="Escala de guidance do CFG"
    )
    embedded_cfg_scale: Optional[float] = Field(
        default=6.0,
        ge=1.0,
        le=20.0,
        description="Embedded classifier free guidance scale"
    )
    flow_shift: Optional[float] = Field(
        default=7.0,
        description="Flow shift parameter"
    )
    flow_reverse: Optional[bool] = Field(
        default=True,
        description="If reverse, learning/sampling from t=1 -> t=0"
    )
    fps: Optional[int] = Field(
        default=8,
        ge=8,
        le=30,
        description="Frames por segundo"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed para reprodutibilidade"
    )
    use_quantization: Optional[bool] = Field(
        default=True,
        description="Usar quantização NF4/INT8 para menor uso de VRAM"
    )
    motion_bucket_id: Optional[int] = Field(
        default=127,
        ge=1,
        le=255,
        description="Controle de movimento (1-255)"
    )
    noise_aug_strength: Optional[float] = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Força do ruído de augmentação"
    )
    template_id: Optional[str] = None

    def validate_dimensions(self):
        """Valida as dimensões do vídeo"""
        from utils.validation import validate_dimensions, validate_duration
        
        validate_dimensions(self.width, self.height)
        validate_duration(self.duration)
        
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Uma cidade futurista durante o pôr do sol, estilo cyberpunk",
                "negative_prompt": "baixa qualidade, borrado, pixelado",
                "duration": 16,
                "height": 576,
                "width": 1024,
                "fps": 8,
                "motion_bucket_id": 127,
                "noise_aug_strength": 0.02
            }
        }

def load_template(template_id: str) -> dict:
    """
    Carrega um template (arquivo JSON) do diretório 'templates/'.
    Exemplo de template:
    {
      "template_id": "video_generator_template1",
      "prompt": "Uma cidade futurista durante o pôr do sol",
      "duration": 15,
      "height": 1080,
      "width": 1920,
      "fps": 30
    }
    """
    template_path = os.path.join("templates", f"{template_id}.json")
    if not os.path.exists(template_path):
        raise ValueError("Template não encontrado.")
    with open(template_path, "r") as f:
        return json.load(f)

######################################
# 3. Carregamento do Gerador de Vídeo (FastHunyuan)
######################################
video_generator = None

def load_video_generator():
    """Carrega o pipeline do FastHunyuan Video."""
    global video_generator
    try:
        # Configurações de quantização
        quantization_config = None
        if USE_NF4:
            quantization_config = bnb.nn.modules.Linear4bit(
                compress_statistics=True,
                quant_type="nf4",
                use_double_quant=True
            )
        
        # Carrega o pipeline
        video_generator = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True,
            variant="fp16" if DEVICE == "cuda" else None,
            quantization_config=quantization_config if USE_NF4 else None
        )
        
        # Otimizações de memória
        if USE_CPU_OFFLOAD:
            video_generator.enable_model_cpu_offload()
            video_generator.enable_sequential_cpu_offload()
        else:
            video_generator = video_generator.to(DEVICE)
        
        # Otimizações adicionais
        video_generator.enable_vae_slicing()
        video_generator.enable_xformers_memory_efficient_attention()
        video_generator.enable_vae_tiling()
        
        # Configura scheduler
        video_generator.scheduler.set_timesteps(50)
        video_generator.scheduler.config.update({
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "linear",
            "clip_sample": False
        })
        
        return video_generator
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o gerador de vídeo: {str(e)}")

@app.on_event("startup")
async def startup_event():
    optimize_gpu_settings()
    
    # Verificar memória GPU disponível
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 8:  # Mínimo recomendado
            logger.warning(f"GPU com apenas {gpu_memory:.1f}GB detectada.")

    # Verifica CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. GPU NVIDIA é necessária.")
    
    # Verifica memória GPU
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    if gpu_memory < 20:  # Mínimo para versão quantizada
        raise RuntimeError(f"GPU com {gpu_memory:.1f}GB detectada. Mínimo de 20GB necessário.")
    
    # Carrega modelo
    await asyncio.get_event_loop().run_in_executor(None, load_video_generator)

######################################
# 4. Integração com MinIO para Vídeos
######################################
VIDEO_BUCKET_NAME = "videos-gerados"

minio_client = Minio(
    "minio.ruanpiscitelli.com",
    access_key="YYCZ0Fl0gu1nx2LaTORS",
    secret_key="gB0Kl7BWBPolCwLz29OyEPQiOBLnrlAtHqx3cK1Q",
    secure=True
)

def upload_video_to_minio(file_bytes: bytes, file_name: str) -> str:
    """
    Faz o upload dos bytes do arquivo de vídeo para o MinIO e retorna uma URL pré-assinada.
    A URL será válida por 2 dias.
    """
    try:
        if not minio_client.bucket_exists(VIDEO_BUCKET_NAME):
            minio_client.make_bucket(VIDEO_BUCKET_NAME)
        minio_client.put_object(
            VIDEO_BUCKET_NAME,
            file_name,
            data=BytesIO(file_bytes),
            length=len(file_bytes),
            content_type="video/mp4"
        )
        url = minio_client.presigned_get_object(VIDEO_BUCKET_NAME, file_name, expires=2*24*60*60)
        return url
    except S3Error as err:
        raise RuntimeError(f"Erro no upload para o MinIO: {err}")

######################################
# 5. Endpoint de Geração de Vídeo (FastHunyuan) - Assíncrono
######################################
# Executor para operações bloqueantes
thread_pool = concurrent.futures.ThreadPoolExecutor()

def _generate_video_sync(
    generation_params: dict,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Função síncrona auxiliar para gerar vídeo usando o modelo.
    Esta função é executada em uma thread separada para não bloquear o loop de eventos.
    
    Args:
        generation_params: Parâmetros para geração do vídeo
        generator: Gerador opcional para reprodutibilidade
        
    Returns:
        torch.Tensor: Tensor do vídeo gerado
    """
    with gpu_memory_manager():
        output = video_generator(
            **generation_params,
            generator=generator
        )
        return output.videos[0]

@app.post("/generate-video")
@track_time(VIDEO_GENERATION_TIME)
@track_errors("generation")
async def generate_video(
    request: VideoRequest,
    token: str = Depends(verify_bearer_token)
):
    """
    Gera um vídeo a partir do prompt fornecido usando FastHunyuan.
    """
    try:
        # Valida request
        request.validate_dimensions()
        
        # Carrega template se fornecido
        if request.template_id:
            template = load_template(request.template_id)
            # Atualiza request com valores do template
            for key, value in template.items():
                if not getattr(request, key):
                    setattr(request, key, value)
        
        # Prepara parâmetros
        generation_params = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "embedded_cfg_scale": request.embedded_cfg_scale,
            "height": request.height,
            "width": request.width,
            "num_frames": request.duration,
            "fps": request.fps,
            "motion_bucket_id": request.motion_bucket_id,
            "noise_aug_strength": request.noise_aug_strength
        }
        
        # Valida parâmetros
        validate_generation_params(generation_params)
        
        # Configura seed se fornecida
        if request.seed is not None:
            generator = torch.Generator(device=DEVICE)
            generator.manual_seed(request.seed)
        else:
            generator = None
            
        resolution = f"{request.width}x{request.height}"
        
        # Contexto de métricas
        with MetricsContext(resolution=resolution) as metrics:
            # Executa geração de vídeo em uma thread separada
            loop = asyncio.get_running_loop()
            video = await loop.run_in_executor(
                thread_pool,
                _generate_video_sync,
                generation_params,
                generator
            )
            
            # Converte para bytes de forma assíncrona
            video_bytes = await loop.run_in_executor(
                thread_pool,
                tensor_to_video,
                video,
                request.fps
            )
            
            # Upload para MinIO de forma assíncrona
            file_name = f"{uuid.uuid4()}_{int(time.time())}.mp4"
            minio_url = await loop.run_in_executor(
                thread_pool,
                upload_to_minio,
                video_bytes,
                file_name
            )
            
            # Atualiza métricas finais
            BATCH_SIZE.set(1)  # Reset batch size
            update_resource_metrics()
            
            return {
                "status": "sucesso",
                "message": "Vídeo gerado com sucesso",
                "minio_url": minio_url,
                "params": {
                    "resolution": resolution,
                    "duration": request.duration,
                    "fps": request.fps
                },
                "metrics": get_metrics_snapshot()
            }
                
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except ResourceError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except ProcessingError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno do servidor: {str(e)}"
        )

def tensor_to_video(
    video_tensor: torch.Tensor,
    fps: int = 8
) -> bytes:
    """
    Converte tensor do PyTorch para bytes de vídeo MP4.
    
    Args:
        video_tensor: Tensor do vídeo (B,C,T,H,W)
        fps: Frames por segundo
        
    Returns:
        bytes: Vídeo em formato MP4
    """
    try:
        # Remove batch dimension se presente
        if video_tensor.dim() == 5:
            video_tensor = video_tensor.squeeze(0)
            
        # Converte para uint8
        video_tensor = (video_tensor * 255).to(torch.uint8)
        
        # Reorganiza dimensões para (T,H,W,C)
        video_tensor = video_tensor.permute(1, 2, 3, 0)
        
        # Converte para numpy
        video_array = video_tensor.cpu().numpy()
        
        # Salva como MP4
        output_buffer = BytesIO()
        
        # Configura writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_buffer,
            fourcc,
            fps,
            (video_array.shape[2], video_array.shape[1])
        )
        
        # Escreve frames
        for frame in video_array:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
        writer.release()
        
        return output_buffer.getvalue()
        
    except Exception as e:
        raise ProcessingError(f"Erro ao converter vídeo: {str(e)}")

def upload_to_minio(video_bytes: bytes, file_name: str) -> str:
    """
    Faz upload do vídeo para o MinIO.
    
    Args:
        video_bytes: Bytes do vídeo
        file_name: Nome do arquivo
        
    Returns:
        str: URL do vídeo no MinIO
    """
    try:
        # Configura bucket
        bucket_name = os.getenv("MINIO_BUCKET", "videos")
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            
        # Upload
        minio_client.put_object(
            bucket_name,
            file_name,
            BytesIO(video_bytes),
            len(video_bytes),
            content_type="video/mp4"
        )
        
        # Gera URL
        url = minio_client.presigned_get_object(
            bucket_name,
            file_name,
            expires=timedelta(hours=24)
        )
        
        return url
        
    except Exception as e:
        raise ProcessingError(f"Erro no upload para MinIO: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Endpoint para métricas Prometheus."""
    return get_metrics_snapshot()

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde."""
    try:
        # Verifica GPU
        gpu_ok = torch.cuda.is_available()
        
        # Verifica modelo
        model_ok = video_generator is not None
        
        # Verifica MinIO
        minio_ok = check_minio_connection()
        
        status = "healthy" if all([gpu_ok, model_ok, minio_ok]) else "degraded"
        
        return {
            "status": status,
            "gpu": {
                "available": gpu_ok,
                "device": torch.cuda.get_device_name(0) if gpu_ok else None
            },
            "model": {
                "loaded": model_ok,
                "device": str(next(video_generator.parameters()).device) if model_ok else None
            },
            "minio": {
                "connected": minio_ok
            },
            "metrics": get_metrics_snapshot()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

######################################
# 6. Endpoint de Redimensionamento de Vídeo
######################################
class ResizeRequest(BaseModel):
    width: Optional[int] = Field(
        default=1024,
        ge=320,
        le=1280,
        description="Largura desejada (múltiplo de 64)"
    )
    height: Optional[int] = Field(
        default=576,
        ge=320,
        le=720,
        description="Altura desejada (múltiplo de 64)"
    )

    def validate_dimensions(self):
        """Valida as dimensões do vídeo"""
        if self.width % 64 != 0:
            raise HTTPException(
                status_code=400,
                detail="Largura deve ser múltiplo de 64"
            )
        if self.height % 64 != 0:
            raise HTTPException(
                status_code=400,
                detail="Altura deve ser múltiplo de 64"
            )

@app.post("/resize-video")
async def resize_video_endpoint(
    file: UploadFile = File(...),
    params: ResizeRequest = Depends(),
    token: str = Depends(verify_bearer_token)
):
    """
    Redimensiona um vídeo para as dimensões especificadas.
    O vídeo deve estar em um dos formatos: .mp4, .avi, .mov, .mkv
    """
    # Valida dimensões
    params.validate_dimensions()
    
    # Valida extensão do arquivo
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de arquivo não suportado. Use: {', '.join(valid_extensions)}"
        )
    
    try:
        # Cria diretórios temporários se não existirem
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Salva o arquivo temporariamente
        temp_input = temp_dir / f"input_{uuid.uuid4()}{file_ext}"
        temp_output = temp_dir / f"output_{uuid.uuid4()}.mp4"
        
        try:
            # Salva arquivo de entrada
            with open(temp_input, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Redimensiona o vídeo
            from utils.resize_videos import resize_video
            success = resize_video(
                str(temp_input),
                str(temp_output),
                target_width=params.width,
                target_height=params.height
            )
            
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Erro ao redimensionar o vídeo"
                )
            
            # Faz upload do vídeo redimensionado para o MinIO
            with open(temp_output, "rb") as f:
                video_bytes = f.read()
            
            file_name = f"resized_{int(time.time())}_{uuid.uuid4()}.mp4"
            url = upload_video_to_minio(video_bytes, file_name)
            
            return {
                "status": "success",
                "video_url": url,
                "metadata": {
                    "original_filename": file.filename,
                    "dimensions": f"{params.width}x{params.height}",
                    "format": "mp4"
                }
            }
            
        finally:
            # Limpa arquivos temporários
            if temp_input.exists():
                temp_input.unlink()
            if temp_output.exists():
                temp_output.unlink()
                
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento do vídeo: {str(e)}"
        )

async def generate_video_frames(prompt: str, num_frames: int) -> List[Image.Image]:
    with cuda_memory_manager():
        with torch.cuda.amp.autocast():
            try:
                frames = video_generator(
                    prompt=prompt,
                    num_frames=num_frames
                ).images
                return frames
            except Exception as e:
                logger.error(f"Erro na geração de frames: {str(e)}")
                raise