"""
Serviço de geração de imagem usando Stable Diffusion.
"""
import os
import uuid
import time
import torch
import logging
import asyncio
import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler
)

from .validations import (
    ValidationError,
    ResourceError,
    ProcessingError,
    validate_model_compatibility,
    validate_memory_requirements,
    validate_scheduler_config,
    validate_lora_weights,
    validate_vae_config,
    load_and_validate_template
)

from .config import DEVICE, API_CONFIG
from .auth import verify_bearer_token
from .storage import upload_to_minio, check_minio_connection
from .utils import convert_image_to_bytes
from .routers import image
from .pipeline import initialize_models, cleanup_resources

# Configurações de GPU e Otimizações
DEVICE_CONFIG = {
    "min_gpu_memory": 8,  # GB - Mínimo recomendado
    "recommended_gpu_memory": 16,  # GB
    "fp16_enabled": True,  # Usar FP16 quando disponível
    "fp8_enabled": False,  # Suporte a FP8 (experimental)
    "flash_attention": True,  # Recomendado pelo Hunyuan
    "xformers_memory_efficient": True,
    "cuda_graphs_enabled": True,
    "torch_compile": True,
    "parallel_degree": 1  # Grau de paralelismo
}

# Otimizações de memória baseadas no Hunyuan
MEMORY_CONFIG = {
    "cpu_offload": True,  # Recomendado para economizar VRAM
    "attention_slicing": True,
    "vae_slicing": True,
    "model_cpu_offload": True,
    "sequential_cpu_offload": True,
    "enable_vae_tiling": True,
    "gradient_checkpointing": False
}

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adiciona lista de modelos suportados
SUPPORTED_MODELS = {
    "cagliostrolab/animagine-xl-4.0": {
        "type": "sdxl",
        "min_vram": 12,
        "recommended_vram": 16
    },
    "John6666/ultimate-realistic-mix-v2-sdxl": {
        "type": "sdxl",
        "min_vram": 12,
        "recommended_vram": 16
    }
}

def validate_model_support(model_id: str) -> None:
    """
    Valida se o modelo é suportado e se há recursos suficientes.
    
    Args:
        model_id: ID do modelo a ser validado
        
    Raises:
        ValidationError: Se o modelo não for suportado ou não houver recursos suficientes
    """
    if model_id not in SUPPORTED_MODELS:
        raise ValidationError(
            f"Modelo {model_id} não suportado. Modelos suportados: {list(SUPPORTED_MODELS.keys())}"
        )
    
    model_info = SUPPORTED_MODELS[model_id]
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_memory < model_info["min_vram"]:
            raise ValidationError(
                f"Modelo {model_id} requer mínimo de {model_info['min_vram']}GB de VRAM. "
                f"Detectado: {gpu_memory:.1f}GB"
            )

# Gerenciamento de lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o lifecycle da aplicação."""
    # Startup
    logger.info("Inicializando modelos...")
    await initialize_models()
    
    yield
    
    # Shutdown
    logger.info("Limpando recursos...")
    await cleanup_resources()

# Inicializa FastAPI
app = FastAPI(
    title="Image Generator API",
    description="API para geração de imagens usando modelos de IA",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

######################################
# Configurações de Otimização
######################################
# Configurações de CUDA e Torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Otimizações de memória
USE_TORCH_COMPILE = True  # Usar torch.compile() para otimização JIT
USE_ATTENTION_SLICING = True  # Reduz uso de VRAM processando attention em partes
USE_VAE_TILING = True  # Processa VAE em tiles para imagens grandes
USE_TORCH_XLA = False  # Para TPUs (se disponível)

# Configurações de processamento
MAX_QUEUE_SIZE = 200     # Máximo de requisições na fila
MAX_CONCURRENT_TASKS = 128  # Número máximo de tarefas processadas simultaneamente

# Configurações de timeout e retry
MAX_RETRIES = 2         # Número de tentativas
RETRY_DELAY = 3         # Segundos entre retries
TASK_EXPIRY = 3 * 60 * 60  # 3 horas para expirar tarefas antigas

# Otimizações CUDA
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Configurações de cache
os.environ["TRANSFORMERS_CACHE"] = "models/cache"
os.environ["HF_HOME"] = "models/hub"

######################################
# 1. Validação de Token
######################################
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError(
        "A variável de ambiente API_TOKEN não está configurada. "
        "Por favor, configure um token seguro para autenticação da API."
    )

def verify_bearer_token(authorization: str = Header(None)):
    """
    Valida o header Authorization e extrai o token.
    
    Args:
        authorization: Header de autorização no formato "Bearer <token>"
        
    Returns:
        str: Token validado
        
    Raises:
        HTTPException: Se o token for inválido ou não autorizado
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Header de autorização ausente",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Esquema de autenticação inválido. Use 'Bearer'",
                headers={"WWW-Authenticate": "Bearer"}
            )
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Header de autorização malformado",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Token não fornecido",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    if not token == API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Token inválido ou expirado",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    return token

######################################
# 2. Modelo de Requisição e Templates
######################################
class ControlNetConfig(BaseModel):
    """Configuração para ControlNet"""
    model_id: str
    image: str  # Base64 da imagem de condicionamento
    scale: float = Field(default=1.0, ge=0.0, le=1.0)
    
class LoRAConfig(BaseModel):
    """Configuração para LoRA"""
    model_id: str
    scale: float = Field(default=1.0, ge=0.0, le=2.0)
    
class VAEConfig(BaseModel):
    """Configuração para VAE"""
    model_id: str
    tiling: bool = False

class ImageRequest(BaseModel):
    """
    Modelo para requisição de geração de imagem.
    Suporta configurações avançadas como ControlNet, LoRA e VAE.
    """
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = Field(default="", max_length=1000)
    height: int = Field(default=512, ge=256, le=2048, multiple_of=8)
    width: int = Field(default=512, ge=256, le=2048, multiple_of=8)
    num_inference_steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = None
    model: str = Field(default="stabilityai/stable-diffusion-xl-base-1.0")
    scheduler: str = Field(default="euler_a")
    
    # Configurações avançadas
    controlnet: Optional[ControlNetConfig] = None
    loras: Optional[List[LoRAConfig]] = None
    vae: Optional[VAEConfig] = None
    template_id: Optional[str] = None
    
    # Novas configurações SDXL
    refiner_model: Optional[str] = Field(
        default=None,
        description="Modelo refinador SDXL (ex: stabilityai/stable-diffusion-xl-refiner-1.0)"
    )
    refiner_steps: Optional[int] = Field(default=None, ge=1, le=50)
    high_noise_fraction: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    style_preset: Optional[str] = None
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    denoising_strength: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    image_guidance_scale: Optional[float] = Field(default=1.5, ge=0.0, le=10.0)
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Uma paisagem montanhosa ao pôr do sol",
                "negative_prompt": "blur, desfoque, baixa qualidade",
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "seed": 42,
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "scheduler": "euler_a",
                "refiner_model": "stabilityai/stable-diffusion-xl-refiner-1.0",
                "refiner_steps": 20,
                "style_preset": "cinematic",
                "prompt_2": "detalhado, alta resolução",
                "negative_prompt_2": "borrado, pixelado"
            }
        }

def load_template(template_id: str) -> dict:
    """
    Carrega um template (arquivo JSON) do diretório 'templates/'.
    Exemplo de template:
    {
      "template_id": "image_template1",
      "model": "sdxl",
      "prompt": "Um pôr do sol futurista",
      "negative_prompt": "ruído, baixa resolução",
      "height": 768,
      "width": 768,
      "num_inference_steps": 60,
      "loras": [
         {"path": "models/loras/lora_face.pt", "scale": 0.8},
         {"path": "models/loras/lora_style.pt", "scale": 1.0}
      ]
    }
    """
    template_path = os.path.join("templates", f"{template_id}.json")
    if not os.path.exists(template_path):
        raise ValueError("Template não encontrado.")
    with open(template_path, "r") as f:
        return json.load(f)

######################################
# 3. Carregamento do Pipeline e Otimização para GPU
######################################
pipe = None  # Variável global para armazenar o pipeline carregado

# Atualiza o dicionário de schedulers disponíveis
AVAILABLE_SCHEDULERS = {
    "DDIM": "ddim",
    "DPMSolverMultistep": "dpm_solver_multistep",
    "EulerDiscrete": "euler",
    "EulerAncestralDiscrete": "euler_ancestral",
    "UniPCMultistep": "unipc"
}

def process_controlnet_image(image_url: Optional[str] = None, image_base64: Optional[str] = None, preprocessor: str = "canny", params: Dict = {}):
    """
    Processa a imagem para o ControlNet usando o pré-processador especificado.
    
    Preprocessadores suportados:
    - canny: Detecção de bordas Canny
    - depth: Estimativa de profundidade
    - normal: Mapa de normais
    - mlsd: Detecção de linhas
    - hed: Detecção de bordas HED
    - scribble: Conversão para desenho
    - pose: Detecção de pose
    - seg: Segmentação semântica
    - ip2p: Image-to-image com preservação
    - lineart: Conversão para line art
    - softedge: Bordas suaves
    """
    try:
        # Carrega a imagem
        if image_url:
            import requests
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        elif image_base64:
            image = Image.open(BytesIO(base64.b64decode(image_base64)))
        else:
            raise ValueError("É necessário fornecer image_url ou image_base64")

        # Converte para numpy array
        image_np = np.array(image)
        
        # Preprocessadores que usam modelos do HuggingFace
        hf_preprocessors = {
            "depth": "depth-estimation",
            "normal": "normal-estimation",
            "hed": "hed-edge-detection",
            "mlsd": "mlsd-line-detection",
            "pose": "pose-estimation",
            "seg": "semantic-segmentation",
            "lineart": "line-art-detection",
            "softedge": "soft-edge-detection"
        }
        
        if preprocessor in hf_preprocessors:
            from transformers import pipeline
            pipe = pipeline(hf_preprocessors[preprocessor])
            result = pipe(image)
            
            if preprocessor == "depth":
                processed = result["depth"]
                processed = np.array(processed)
                processed = np.stack([processed, processed, processed], axis=-1)
            elif preprocessor == "normal":
                processed = result["prediction"]
            elif preprocessor in ["hed", "mlsd", "lineart", "softedge"]:
                processed = result["edges"]
            elif preprocessor == "pose":
                processed = result["keypoints"]
            elif preprocessor == "seg":
                processed = result["segmentation"]
            else:
                processed = result["prediction"]
                
        elif preprocessor == "canny":
            # Parâmetros do Canny
            low_threshold = params.get("low_threshold", 100)
            high_threshold = params.get("high_threshold", 200)
            
            # Converte para escala de cinza
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Aplica blur se solicitado
            if params.get("blur", True):
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                
            # Detecção de bordas
            processed = cv2.Canny(gray, low_threshold, high_threshold)
            processed = processed[:, :, None]
            processed = np.concatenate([processed, processed, processed], axis=2)
            
        elif preprocessor == "scribble":
            # Converte para escala de cinza
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Aplica threshold adaptativo
            processed = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                params.get("block_size", 9),
                params.get("c", 2)
            )
            
            # Aplica dilatação se solicitado
            if params.get("dilate", True):
                kernel = np.ones((2,2), np.uint8)
                processed = cv2.dilate(processed, kernel, iterations=1)
                
            processed = processed[:, :, None]
            processed = np.concatenate([processed, processed, processed], axis=2)
            
        elif preprocessor == "ip2p":
            # Image-to-image com preservação
            # Aplica uma combinação de bordas e estrutura
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Detecta estruturas com Sobel
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Combina as detecções
            processed = np.sqrt(sobelx**2 + sobely**2)
            processed = (processed * 255 / processed.max()).astype(np.uint8)
            
            # Combina com as bordas
            processed = cv2.addWeighted(edges, 0.5, processed, 0.5, 0)
            processed = processed[:, :, None]
            processed = np.concatenate([processed, processed, processed], axis=2)
            
        else:
            raise ValueError(f"Pré-processador '{preprocessor}' não suportado")

        return Image.fromarray(processed.astype(np.uint8))
        
    except Exception as e:
        logger.error(f"Erro no processamento da imagem: {str(e)}")
        raise ValidationError(f"Erro ao processar imagem: {str(e)}")

######################################
# Otimizações de Memória e VRAM
######################################
def apply_memory_optimizations(pipeline, config: dict = None):
    """
    Aplica otimizações de memória ao pipeline
    
    Args:
        pipeline: Pipeline de difusão
        config: Configurações de otimização
    """
    if config is None:
        config = {}
    
    # Token Merging (ToMe)
    if config.get("tomesd", {}).get("enabled", True):
        try:
            import tomesd
            ratio = config.get("tomesd", {}).get("ratio", 0.4)
            max_downsample = config.get("tomesd", {}).get("max_downsample", 2)
            tomesd.apply_patch(pipeline, ratio=ratio, max_downsample=max_downsample)
            print(f"ToMe aplicado com ratio {ratio} e max_downsample {max_downsample}")
        except ImportError:
            print("tomesd não encontrado. Ignorando otimização ToMe.")
    
    # xFormers
    try:
        import xformers
        pipeline.enable_xformers_memory_efficient_attention()
        print("xFormers habilitado com sucesso")
    except ImportError:
        print("xformers não encontrado. Usando attention padrão.")
    
    # Otimizações de memória padrão
    if config.get("enable_vae_tiling", True):
        pipeline.enable_vae_tiling()
        pipeline.enable_vae_slicing()
        print("VAE tiling e slicing habilitados")
    
    if config.get("enable_attention_slicing", True):
        pipeline.enable_attention_slicing(slice_size="auto")
        print("Attention slicing habilitado")
    
    # Gerenciamento dinâmico de VRAM
    if config.get("enable_sequential_cpu_offload", False):
        pipeline.enable_sequential_cpu_offload()
        print("Sequential CPU offload habilitado")
    
    if config.get("enable_model_cpu_offload", False):
        pipeline.enable_model_cpu_offload()
        print("Model CPU offload habilitado")
    
    return pipeline

def process_prompt(prompt: str, clip_skip: int = None) -> str:
    """
    Processa o prompt aplicando pesos e embeddings personalizados
    
    Args:
        prompt: Prompt original
        clip_skip: Número de camadas CLIP a pular
        
    Returns:
        Prompt processado
    """
    try:
        # Processa pesos entre parênteses: (texto:1.2)
        import re
        prompt = re.sub(r'\(([^:]+):([0-9.]+)\)', lambda m: f"({m.group(1)}){float(m.group(2))}", prompt)
        
        # Processa alternativas entre chaves: {opção1|opção2}
        prompt = re.sub(r'\{([^}]+)\}', lambda m: m.group(1).split('|')[0], prompt)
        
        # Remove espaços extras
        prompt = ' '.join(prompt.split())
        
        return prompt
    except Exception as e:
        print(f"Erro no processamento do prompt: {e}")
        return prompt

def apply_clip_skip(pipeline, clip_skip: int):
    """
    Aplica CLIP skip ao pipeline
    
    Args:
        pipeline: Pipeline de difusão
        clip_skip: Número de camadas a pular
    """
    if clip_skip and clip_skip > 1:
        # Modifica o text_encoder para pular camadas
        pipeline.text_encoder.num_hidden_layers = pipeline.text_encoder.config.num_hidden_layers - (clip_skip - 1)
        print(f"CLIP skip {clip_skip} aplicado")
    return pipeline

def _load_pipeline_sync(model_id: str) -> StableDiffusionPipeline:
    """Função síncrona para carregar o pipeline do modelo"""
    logger.info(f"Carregando modelo {model_id}")
    
    # Verifica se é um caminho local
    local_model_path = os.path.join("models", model_id)
    if os.path.exists(local_model_path):
        logger.info(f"Carregando modelo local de {local_model_path}")
        if "xl" in model_id.lower():
            pipeline = StableDiffusionXLPipeline.from_single_file(
                local_model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        else:
            pipeline = StableDiffusionPipeline.from_single_file(
                local_model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
    else:
        # Carrega do HuggingFace
        if "xl" in model_id.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        
    pipeline.to(DEVICE)
    return pipeline

async def load_pipeline(model_id: str) -> StableDiffusionPipeline:
    """Carrega um pipeline do cache ou baixa se necessário de forma assíncrona"""
    # Valida suporte ao modelo
    validate_model_support(model_id)
    
    if model_id not in MODEL_CACHE:
        # Executa o carregamento pesado em uma thread separada
        pipeline = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _load_pipeline_sync(model_id)
        )
        MODEL_CACHE[model_id] = pipeline
    
    return MODEL_CACHE[model_id]

######################################
# 4. Integração com MinIO
######################################
# Cria o cliente MinIO utilizando as credenciais fornecidas.
minio_client = Minio(
    "minio.ruanpiscitelli.com",  # Host do MinIO
    access_key="YYCZ0Fl0gu1nx2LaTORS",
    secret_key="gB0Kl7BWBPolCwLz29OyEPQiOBLnrlAtHqx3cK1Q",
    secure=True
)
BUCKET_NAME = "imagens-geradas"  # Nome do bucket

def upload_to_minio(file_bytes: bytes, file_name: str) -> str:
    """
    Faz o upload dos bytes do arquivo para o MinIO e retorna uma URL pré-assinada.
    A URL será válida por 2 dias.
    """
    try:
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
        
        minio_client.put_object(
            BUCKET_NAME,
            file_name,
            data=BytesIO(file_bytes),
            length=len(file_bytes),
            content_type="image/png"
        )
        # Gera uma URL pré-assinada válida por 2 dias (2*24*60*60 segundos)
        url = minio_client.presigned_get_object(BUCKET_NAME, file_name, expires=2*24*60*60)
        return url
    except S3Error as err:
        raise RuntimeError(f"Erro no upload para o MinIO: {err}")

######################################
# 5. Placeholder para Injeção de LoRAs
######################################
def apply_loras(pipeline, loras: list):
    """
    Aplica os LoRAs ao pipeline com suporte a múltiplos formatos e validação.
    
    Args:
        pipeline: Pipeline de difusão
        loras: Lista de dicionários com configurações dos LoRAs
        
    Returns:
        Pipeline modificado com os LoRAs aplicados
    """
    try:
        import safetensors
        from safetensors.torch import load_file
        
        for lora in loras:
            lora_path = lora.get("path")
            if not os.path.exists(lora_path):
                raise ValueError(f"LoRA não encontrado: {lora_path}")
                
            scale = lora.get("scale", 1.0)
            is_version = lora.get("is_version", "1.0")
            base_compatibility = lora.get("base_model_compatibility", [])
            
            # Valida compatibilidade
            if base_compatibility and pipeline.model_name not in base_compatibility:
                raise ValueError(f"LoRA {lora_path} não é compatível com o modelo {pipeline.model_name}")
            
            # Carrega LoRA baseado na extensão
            if lora_path.endswith('.safetensors'):
                state_dict = load_file(lora_path)
            elif lora_path.endswith('.pt') or lora_path.endswith('.pth'):
                state_dict = torch.load(lora_path)
            elif lora_path.endswith('.ckpt'):
                state_dict = torch.load(lora_path, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            else:
                raise ValueError(f"Formato de LoRA não suportado: {lora_path}")
            
            # Aplica LoRA ao modelo
            for key in state_dict:
                if 'lora_unet' in key:
                    layer_infos = key.split('.')[0].split('_')
                    curr_layer = pipeline.unet
                    
                    # Navega até a camada correta
                    for layer_info in layer_infos[2:]:
                        if layer_info.isdigit():
                            curr_layer = curr_layer[int(layer_info)]
                        else:
                            curr_layer = getattr(curr_layer, layer_info)
                    
                    # Aplica os pesos do LoRA
                    if hasattr(curr_layer, 'weight'):
                        curr_layer.weight.data += scale * state_dict[key].to(curr_layer.weight.device)
                
                elif 'lora_te' in key:
                    layer_infos = key.split('.')[0].split('_')
                    curr_layer = pipeline.text_encoder
                    
                    # Navega até a camada correta
                    for layer_info in layer_infos[2:]:
                        if layer_info.isdigit():
                            curr_layer = curr_layer[int(layer_info)]
                        else:
                            curr_layer = getattr(curr_layer, layer_info)
                    
                    # Aplica os pesos do LoRA
                    if hasattr(curr_layer, 'weight'):
                        curr_layer.weight.data += scale * state_dict[key].to(curr_layer.weight.device)
            
            print(f"LoRA aplicado com sucesso: {lora_path} (scale: {scale})")
            
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Erro ao aplicar LoRA: {str(e)}")

######################################
# 6. Endpoint de Geração de Imagem (Assíncrono)
######################################
@app.post("/generate-image")
@track_inference(model="stable-diffusion")
async def generate_image(
    request: ImageRequest,
    token: str = Depends(verify_bearer_token)
):
    """
    Endpoint para geração de imagens com otimizações Hunyuan.
    """
    try:
        # Valida requisitos do sistema
        validate_system_requirements()
        
        # Valida requisitos do modelo
        validate_model_requirements(request.model)
        
        # Carrega template se fornecido
        if request.template_id:
            template = load_template(request.template_id)
            request = ImageRequest(**{**template, **request.dict()})
        
        # Carrega e otimiza pipeline
        pipeline = await load_pipeline(request.model)
        pipeline = optimize_pipeline(pipeline)
        
        # Valida e ajusta batch size
        batch_size = validate_batch_size(1, model_size_mb=2048)  # 2GB estimado
        
        # Gera imagem
        with torch.cuda.amp.autocast():
            image = await generate_image_with_pipeline(request, pipeline)
            
        # Converte para bytes
        image_bytes = convert_image_to_bytes(image)
        
        # Upload para MinIO
        file_name = f"image_{int(time.time())}_{uuid.uuid4()}.png"
        url = await upload_to_minio(image_bytes, file_name)
        
        # Atualiza métricas finais
        update_resource_metrics()
        
        return {
            "status": "success",
            "url": url,
            "metadata": {
                "model": request.model,
                "resolution": f"{request.width}x{request.height}",
                "steps": request.num_inference_steps,
                "batch_size": batch_size
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
    except Exception as e:
        logger.error(f"Erro na geração: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde."""
    try:
        # Verifica GPU
        gpu_ok = torch.cuda.is_available()
        
        # Verifica modelo
        model_ok = True  # TODO: Implementar verificação de modelos
        
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
                "loaded": model_ok
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
# Configurações de Fila e Kubernetes
######################################
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import os

# Configurações Kubernetes
NAMESPACE = os.getenv("K8S_NAMESPACE", "default")
CRD_GROUP = "contentapi.ruanpiscitelli.com"
CRD_VERSION = "v1"
CRD_PLURAL = "imagetasks"

class KubernetesQueueManager:
    """Gerenciador de fila de tarefas usando Kubernetes CRDs"""
    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        self.processing_flag = False
        self.custom_api = None
        self.core_api = None
    
    async def initialize(self):
        """Inicializa conexão com Kubernetes e configura CRDs"""
        try:
            # Carrega configuração do Kubernetes
            if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token'):
                config.load_incluster_config()
            else:
                config.load_kube_config()
            
            self.custom_api = client.CustomObjectsApi()
            self.core_api = client.CoreV1Api()
            
            # Define o CRD para tarefas de imagem
            crd_manifest = {
                "apiVersion": "apiextensions.k8s.io/v1",
                "kind": "CustomResourceDefinition",
                "metadata": {
                    "name": f"{CRD_PLURAL}.{CRD_GROUP}"
                },
                "spec": {
                    "group": CRD_GROUP,
                    "versions": [{
                        "name": CRD_VERSION,
                        "served": True,
                        "storage": True,
                        "schema": {
                            "openAPIV3Schema": {
                                "type": "object",
                                "properties": {
                                    "spec": {
                                        "type": "object",
                                        "properties": {
                                            "request": {"type": "object"},
                                            "status": {"type": "object"},
                                            "queuePosition": {"type": "integer"},
                                            "processingNode": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }],
                    "scope": "Namespaced",
                    "names": {
                        "plural": CRD_PLURAL,
                        "singular": "imagetask",
                        "kind": "ImageTask",
                        "shortNames": ["itask"]
                    }
                }
            }
            
            try:
                extensions_api = client.ApiextensionsV1Api()
                extensions_api.create_custom_resource_definition(crd_manifest)
            except ApiException as e:
                if e.status != 409:  # Ignora erro se CRD já existe
                    raise
            
            # Inicia processamento se necessário
            self.processing_flag = True
            asyncio.create_task(self.process_queue())
            
        except Exception as e:
            print(f"Erro ao inicializar Kubernetes: {e}")
            raise
    
    async def add_task(self, task_id: str, request: ImageRequest) -> TaskStatus:
        """Adiciona uma tarefa usando CRD"""
        # Verifica limite da fila
        tasks = await self._list_tasks()
        queue_size = len([t for t in tasks if t['spec']['status']['status'] == 'pending'])
        processing_size = len([t for t in tasks if t['spec']['status']['status'] == 'processing'])
        
        if queue_size + processing_size >= MAX_QUEUE_SIZE:
            raise HTTPException(
                status_code=429,
                detail=f"Fila cheia ({queue_size}/{MAX_QUEUE_SIZE})"
            )
        
        # Cria status da tarefa
        status = TaskStatus(
            task_id=task_id,
            status="pending",
            created_at=datetime.utcnow(),
            queue_position=queue_size + 1,
            estimated_start_time=self._estimate_start_time(queue_size)
        )
        
        # Cria o objeto CRD
        task_manifest = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": "ImageTask",
            "metadata": {
                "name": task_id,
                "namespace": NAMESPACE
            },
            "spec": {
                "request": request.dict(),
                "status": status.dict(),
                "queuePosition": queue_size + 1,
                "processingNode": ""
            }
        }
        
        await self._create_task(task_manifest)
        return status
    
    async def _list_tasks(self):
        """Lista todas as tarefas no namespace"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.custom_api.list_namespaced_custom_object(
                    CRD_GROUP,
                    CRD_VERSION,
                    NAMESPACE,
                    CRD_PLURAL
                )
            )
            return response['items']
        except ApiException as e:
            print(f"Erro ao listar tarefas: {e}")
            return []
    
    async def _create_task(self, task_manifest):
        """Cria uma nova tarefa CRD"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.custom_api.create_namespaced_custom_object(
                    CRD_GROUP,
                    CRD_VERSION,
                    NAMESPACE,
                    CRD_PLURAL,
                    task_manifest
                )
            )
        except ApiException as e:
            print(f"Erro ao criar tarefa: {e}")
            raise HTTPException(status_code=500, detail="Erro ao criar tarefa")
    
    async def process_queue(self):
        """Processa a fila continuamente"""
        while self.processing_flag:
            try:
                # Lista tarefas pendentes
                tasks = await self._list_tasks()
                pending_tasks = [
                    t for t in tasks 
                    if t['spec']['status']['status'] == 'pending'
                ]
                
                # Ordena por posição na fila
                pending_tasks.sort(key=lambda x: x['spec']['queuePosition'])
                
                # Processa em lotes
                batch = pending_tasks[:BATCH_SIZE]
                if not batch:
                    await asyncio.sleep(1)
                    continue
                
                # Marca tarefas como em processamento
                processing_tasks = []
                for task in batch:
                    task_id = task['metadata']['name']
                    if await self._claim_task(task_id):
                        processing_tasks.append(task)
                
                # Processa tarefas
                tasks = []
                for task in processing_tasks:
                    tasks.append(self.process_single_task(task['metadata']['name']))
                
                if tasks:
                    await asyncio.gather(*tasks)
                
            except Exception as e:
                print(f"Erro no processamento do lote: {e}")
                await asyncio.sleep(1)
    
    async def _claim_task(self, task_id: str) -> bool:
        """Tenta reivindicar uma tarefa para processamento"""
        try:
            node_name = os.getenv("K8S_NODE_NAME", "unknown")
            patch = {
                "spec": {
                    "status": {"status": "processing"},
                    "processingNode": node_name
                }
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.custom_api.patch_namespaced_custom_object(
                    CRD_GROUP,
                    CRD_VERSION,
                    NAMESPACE,
                    CRD_PLURAL,
                    task_id,
                    patch
                )
            )
            return True
        except ApiException as e:
            if e.status != 409:  # Ignora conflitos de concorrência
                print(f"Erro ao reivindicar tarefa: {e}")
            return False
    
    async def _get_isolated_pipeline(self, request: ImageRequest):
        """
        Cria um pipeline isolado para a tarefa específica.
        
        Args:
            request: Configuração da requisição de imagem
            
        Returns:
            Pipeline configurado para a tarefa
        """
        try:
            # Executa o carregamento do pipeline em uma thread separada
            pipeline = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: load_pipeline(
                    model_name=request.model,
                    controlnet_config=request.controlnet,
                    vae_id=request.vae
                )
            )
            
            # Aplica configurações específicas da tarefa
            if request.clip_skip:
                pipeline = apply_clip_skip(pipeline, request.clip_skip)
            
            if request.optimization:
                pipeline = apply_memory_optimizations(pipeline, request.optimization)
            
            # Configura o scheduler se especificado
            if request.scheduler and request.scheduler in AVAILABLE_SCHEDULERS:
                sampler_name = AVAILABLE_SCHEDULERS[request.scheduler]
                pipeline._sampler = get_sampler(sampler_name)
            
            return pipeline
            
        except Exception as e:
            raise RuntimeError(f"Erro ao criar pipeline isolado: {str(e)}")

    async def process_single_task(self, task_id: str):
        """Processa uma única tarefa"""
        try:
            # Obtém dados da tarefa
            task = await self._get_task(task_id)
            if not task:
                return
            
            request = ImageRequest(**task['spec']['request'])
            status = TaskStatus(**task['spec']['status'])
            
            try:
                # Gera imagem com retry e isolamento do pipeline
                async with self.semaphore:
                    for attempt in range(MAX_RETRIES):
                        try:
                            # Cria pipeline isolado
                            task_pipe = await self._get_isolated_pipeline(request)
                            result = await generate_image_with_pipeline(request, task_pipe)
                            
                            # Atualiza status
                            await self._update_task_status(
                                task_id,
                                "completed",
                                result["minio_url"]
                            )
                            break
                        except Exception as e:
                            if attempt < MAX_RETRIES - 1:
                                await asyncio.sleep(RETRY_DELAY)
                            else:
                                raise e
            
            except Exception as e:
                await self._update_task_status(
                    task_id,
                    "failed",
                    error=str(e)
                )
            
        finally:
            # Remove claim do nó
            await self._release_task(task_id)
    
    async def _get_task(self, task_id: str):
        """Obtém uma tarefa específica"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.custom_api.get_namespaced_custom_object(
                    CRD_GROUP,
                    CRD_VERSION,
                    NAMESPACE,
                    CRD_PLURAL,
                    task_id
                )
            )
        except ApiException as e:
            if e.status != 404:
                print(f"Erro ao obter tarefa: {e}")
            return None
    
    async def _update_task_status(self, task_id: str, status: str, result_url: str = None, error: str = None):
        """Atualiza o status de uma tarefa"""
        try:
            patch = {
                "spec": {
                    "status": {
                        "status": status,
                        "completed_at": datetime.utcnow().isoformat()
                    }
                }
            }
            
            if result_url:
                patch["spec"]["status"]["result_url"] = result_url
            if error:
                patch["spec"]["status"]["error"] = error
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.custom_api.patch_namespaced_custom_object(
                    CRD_GROUP,
                    CRD_VERSION,
                    NAMESPACE,
                    CRD_PLURAL,
                    task_id,
                    patch
                )
            )
        except ApiException as e:
            print(f"Erro ao atualizar status: {e}")
    
    async def _release_task(self, task_id: str):
        """Remove o claim do nó na tarefa"""
        try:
            patch = {
                "spec": {
                    "processingNode": ""
                }
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.custom_api.patch_namespaced_custom_object(
                    CRD_GROUP,
                    CRD_VERSION,
                    NAMESPACE,
                    CRD_PLURAL,
                    task_id,
                    patch
                )
            )
        except ApiException as e:
            print(f"Erro ao liberar tarefa: {e}")
    
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Obtém o status atual de uma tarefa"""
        task = await self._get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Tarefa não encontrada")
        
        return TaskStatus(**task['spec']['status'])
    
    async def cancel_task(self, task_id: str):
        """Cancela uma tarefa pendente"""
        task = await self._get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Tarefa não encontrada")
        
        if task['spec']['status']['status'] != "pending":
            raise HTTPException(
                status_code=400,
                detail="Apenas tarefas pendentes podem ser canceladas"
            )
        
        await self._update_task_status(
            task_id,
            "failed",
            error="Tarefa cancelada pelo usuário"
        )
    
    async def get_queue_info(self) -> dict:
        """Retorna informações sobre a fila"""
        tasks = await self._list_tasks()
        queue_size = len([t for t in tasks if t['spec']['status']['status'] == 'pending'])
        processing_size = len([t for t in tasks if t['spec']['status']['status'] == 'processing'])
        
        return {
            "queue_size": queue_size,
            "processing": processing_size,
            "total_tasks": queue_size + processing_size,
            "max_queue_size": MAX_QUEUE_SIZE,
            "max_concurrent": MAX_CONCURRENT_TASKS,
            "batch_size": BATCH_SIZE
        }

@contextmanager
def cuda_memory_manager():
    """
    Gerenciador de contexto para otimizar o uso de memória CUDA.
    Limpa cache antes e depois das operações de GPU.
    """
    try:
        torch.cuda.empty_cache()
        yield
    finally:
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

async def generate_image_with_pipeline(request: ImageRequest, pipe):
    """Função isolada para geração de imagem com pipeline específico"""
    try:
        with cuda_memory_manager():
            # Processa imagem do ControlNet se especificado
            controlnet_image = None
            if request.controlnet:
                controlnet_image = process_controlnet_image(
                    image_url=request.controlnet.image_url,
                    image_base64=request.controlnet.image_base64,
                    preprocessor=request.controlnet.preprocessor,
                    params=request.controlnet.preprocessor_params
                )

            # Configura o scheduler
            if request.scheduler and request.scheduler in AVAILABLE_SCHEDULERS:
                sampler_name = AVAILABLE_SCHEDULERS[request.scheduler]
                pipe._sampler = get_sampler(sampler_name)

            # Define seed
            if request.seed is None:
                request.seed = int(time.time())
            generator = torch.Generator(device=DEVICE).manual_seed(request.seed)

            # Aplica LoRAs
            if request.loras:
                pipe = apply_loras(pipe, request.loras)
            
            # Aplica CLIP skip
            if hasattr(request, 'clip_skip') and request.clip_skip:
                pipe = apply_clip_skip(pipe, request.clip_skip)
            
            # Aplica otimizações
            if hasattr(request, 'optimization'):
                pipe = apply_memory_optimizations(pipe, request.optimization)
            
            # Processa prompts
            processed_prompt = process_prompt(request.prompt)
            processed_negative_prompt = process_prompt(request.negative_prompt) if request.negative_prompt else None
            
            # Processa prompts secundários para SDXL
            processed_prompt_2 = process_prompt(request.prompt_2) if request.prompt_2 else None
            processed_negative_prompt_2 = process_prompt(request.negative_prompt_2) if request.negative_prompt_2 else None

            # Prepara argumentos
            generation_args = {
                "prompt": processed_prompt,
                "negative_prompt": processed_negative_prompt,
                "height": request.height,
                "width": request.width,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "generator": generator
            }
            
            # Adiciona argumentos específicos do SDXL
            if processed_prompt_2:
                generation_args["prompt_2"] = processed_prompt_2
            if processed_negative_prompt_2:
                generation_args["negative_prompt_2"] = processed_negative_prompt_2
            if request.style_preset:
                generation_args["style_preset"] = request.style_preset
            if request.denoising_strength:
                generation_args["denoising_strength"] = request.denoising_strength
            if request.image_guidance_scale:
                generation_args["image_guidance_scale"] = request.image_guidance_scale

            if controlnet_image:
                generation_args["image"] = controlnet_image
                generation_args["controlnet_conditioning_scale"] = request.controlnet.scale

            # Gera imagem com otimização de memória
            with torch.cuda.amp.autocast():
                # Geração inicial
                result = pipe(**generation_args)
                image = result.images[0]
                
                # Aplica refinamento se solicitado
                if request.refiner_model:
                    refiner = await load_pipeline(request.refiner_model)
                    refiner_steps = request.refiner_steps or request.num_inference_steps // 2
                    
                    refiner_args = {
                        "prompt": processed_prompt,
                        "negative_prompt": processed_negative_prompt,
                        "image": image,
                        "num_inference_steps": refiner_steps,
                        "guidance_scale": request.guidance_scale,
                        "generator": generator,
                        "high_noise_fraction": request.high_noise_fraction
                    }
                    
                    if processed_prompt_2:
                        refiner_args["prompt_2"] = processed_prompt_2
                    if processed_negative_prompt_2:
                        refiner_args["negative_prompt_2"] = processed_negative_prompt_2
                    
                    result = refiner(**refiner_args)
                    image = result.images[0]

            # Salva e faz upload
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            file_bytes = buffered.getvalue()
            
            file_name = f"{uuid.uuid4()}_{int(time.time())}.png"
            minio_url = upload_to_minio(file_bytes, file_name)
            
            return {
                "status": "sucesso",
                "message": "Imagem gerada com sucesso.",
                "minio_url": minio_url,
                "metadata": {
                    "sampler": pipe._sampler.__class__.__name__,
                    "seed": request.seed,
                    "steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "clip_skip": request.clip_skip if hasattr(request, 'clip_skip') else None,
                    "optimization": request.optimization if hasattr(request, 'optimization') else None,
                    "processed_prompt": processed_prompt,
                    "processed_negative_prompt": processed_negative_prompt,
                    "style_preset": request.style_preset,
                    "refiner_model": request.refiner_model,
                    "refiner_steps": request.refiner_steps,
                    "high_noise_fraction": request.high_noise_fraction
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração da imagem: {str(e)}")

async def generate_image_with_params(
    model_id: str,
    params: Dict[str, Any],
    controlnet_config: Optional[ControlNetConfig] = None,
    lora_configs: Optional[List[LoRAConfig]] = None,
    vae_config: Optional[VAEConfig] = None,
    scheduler_name: str = "euler_a"
) -> Image.Image:
    """
    Gera uma imagem com os parâmetros fornecidos.
    Suporta ControlNet, LoRA e VAE customizado.
    """
    try:
        # Carrega o pipeline base
        pipeline = await load_pipeline(model_id)
        
        # Configura scheduler
        pipeline.scheduler = get_scheduler(scheduler_name, pipeline.scheduler.config)
        
        # Configura VAE se fornecido
        if vae_config:
            vae = await load_vae(vae_config.model_id)
            pipeline.vae = vae
            if vae_config.tiling:
                pipeline.enable_vae_tiling()
        
        # Configura ControlNet se fornecido
        if controlnet_config:
            controlnet = await load_controlnet(controlnet_config.model_id)
            pipeline.controlnet = controlnet
            
            # Processa imagem de condicionamento
            control_image = process_control_image(controlnet_config.image)
            params["image"] = control_image
            params["controlnet_conditioning_scale"] = controlnet_config.scale
        
        # Carrega e aplica LoRAs
        if lora_configs:
            for lora in lora_configs:
                pipeline = await load_and_fuse_lora(
                    pipeline,
                    lora.model_id,
                    lora.scale
                )
        
        # Otimizações
        pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing()
        
        # Gera imagem
        with torch.inference_mode():
            output = pipeline(**params)
            image = output.images[0]
            
        return image
        
    except Exception as e:
        logger.error(f"Erro na geração: {str(e)}")
        raise ProcessingError(f"Falha ao gerar imagem: {str(e)}")

# Atualiza o startup_event para usar o novo gerenciador
@app.on_event("startup")
async def startup_event():
    """Inicializa o serviço e dependências."""
    global queue_manager
    
    # Inicializa o gerenciador de fila
    queue_manager = KubernetesQueueManager()
    await queue_manager.initialize()
    
    # Verifica CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. GPU NVIDIA é necessária.")
    
    # Verifica memória GPU
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    if gpu_memory < 8:  # Mínimo recomendado para SDXL
        raise RuntimeError(f"GPU com {gpu_memory:.1f}GB detectada. Mínimo de 8GB recomendado.")
    
    # Configura diretórios de cache
    os.makedirs("models/cache", exist_ok=True)
    os.makedirs("models/hub", exist_ok=True)
    
    # Carrega o pipeline padrão
    await load_pipeline("sdxl")

# Cache de modelos
MODEL_CACHE = {}
CONTROLNET_CACHE = {}
VAE_CACHE = {}

async def load_controlnet(model_id: str) -> ControlNetModel:
    """Carrega um modelo ControlNet do cache ou baixa se necessário"""
    if model_id not in CONTROLNET_CACHE:
        logger.info(f"Carregando ControlNet {model_id}")
        controlnet = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        controlnet.to(DEVICE)
        CONTROLNET_CACHE[model_id] = controlnet
        
    return CONTROLNET_CACHE[model_id]

async def load_vae(model_id: str) -> AutoencoderKL:
    """Carrega um VAE do cache ou baixa se necessário"""
    if model_id not in VAE_CACHE:
        logger.info(f"Carregando VAE {model_id}")
        vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        vae.to(DEVICE)
        VAE_CACHE[model_id] = vae
        
    return VAE_CACHE[model_id]

async def load_and_fuse_lora(
    pipeline: StableDiffusionPipeline,
    model_id: str,
    scale: float
) -> StableDiffusionPipeline:
    """Carrega e funde um modelo LoRA no pipeline"""
    logger.info(f"Carregando e fundindo LoRA {model_id}")
    pipeline.load_lora_weights(model_id)
    pipeline.fuse_lora(scale)
    return pipeline

def get_scheduler(name: str, config: dict):
    """Retorna o scheduler apropriado baseado no nome"""
    schedulers = {
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpm++": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "ddim": DDIMScheduler,
        "unipc": UniPCMultistepScheduler
    }
    
    if name not in schedulers:
        raise ValidationError(f"Scheduler {name} não suportado")
        
    return schedulers[name].from_config(config)

def process_control_image(image_data: str) -> Image.Image:
    """Processa a imagem de controle do ControlNet"""
    try:
        # Remove cabeçalho do base64 se presente
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
            
        # Decodifica base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Converte para RGB se necessário
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
        
    except Exception as e:
        raise ValidationError(f"Erro ao processar imagem de controle: {str(e)}")

def optimize_pipeline(pipeline):
    """
    Aplica otimizações recomendadas pelo Hunyuan.
    
    Args:
        pipeline: Pipeline de geração
        
    Returns:
        Pipeline otimizado
    """
    if torch.cuda.is_available():
        if DEVICE_CONFIG["fp16_enabled"]:
            pipeline = pipeline.to(torch.float16)
            
        if DEVICE_CONFIG["flash_attention"]:
            pipeline.enable_xformers_memory_efficient_attention()
            
        if MEMORY_CONFIG["cpu_offload"]:
            pipeline.enable_model_cpu_offload()
            
        if MEMORY_CONFIG["vae_slicing"]:
            pipeline.enable_vae_slicing()
            
        # Otimizações de atenção
        if MEMORY_CONFIG["attention_slicing"]:
            pipeline.enable_attention_slicing(1)
        
        # Otimizações de VAE
        if MEMORY_CONFIG["enable_vae_tiling"]:
            pipeline.enable_vae_tiling()
        
        # Otimizações de cache CUDA
        torch.backends.cudnn.benchmark = True
        
        # Compilação do modelo se suportado
        if DEVICE_CONFIG["torch_compile"] and hasattr(torch, 'compile'):
            try:
                pipeline.unet = torch.compile(
                    pipeline.unet,
                    mode="reduce-overhead",
                    fullgraph=True
                )
                logger.info("Pipeline compilado com torch.compile()")
            except Exception as e:
                logger.warning(f"Erro ao compilar pipeline: {e}")
        
    return pipeline

async def initialize_models():
    """Inicializa modelos e recursos."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU com suporte CUDA é necessária")
    
    # Configura diretórios de cache
    os.makedirs("models/cache", exist_ok=True)
    os.makedirs("models/hub", exist_ok=True)
    
    # Carrega o pipeline padrão
    await load_pipeline("sdxl")

async def cleanup_resources():
    """Libera recursos na finalização."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Limpa caches
    MODEL_CACHE.clear()
    CONTROLNET_CACHE.clear()
    VAE_CACHE.clear()

# Handlers de exceção globais
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler para erros de validação."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler para exceções HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para exceções não tratadas."""
    logger.error(f"Erro não tratado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor"}
    )

# Inclui routers
app.include_router(image.router)

class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout=300):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timeout"}
            )