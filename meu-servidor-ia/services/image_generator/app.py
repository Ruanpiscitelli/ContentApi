import asyncio
import base64
import json
import os
from io import BytesIO
import time
import uuid
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Union, Literal, Annotated

from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler
)
import torch
import cv2

# Importa o cliente do MinIO e os erros
from minio import Minio
from minio.error import S3Error

# Importa os samplers
from samplers import get_sampler, SAMPLER_REGISTRY

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
MAX_CONCURRENT_TASKS = 64  # Número máximo de tarefas processadas simultaneamente
BATCH_SIZE = 32          # Tamanho do batch para processamento em lote

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

app = FastAPI()

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
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra='forbid'
    )
    
    model_id: str = Field(description="ID do modelo ControlNet (ex: lllyasviel/control_v11p_sd15_canny)")
    image_url: Optional[str] = Field(
        default=None,
        description="URL da imagem de condicionamento"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Imagem de condicionamento em base64"
    )
    conditioning_scale: Annotated[float, Field(ge=0.0, le=2.0)] = Field(
        default=1.0,
        description="Peso do condicionamento"
    )
    preprocessor: Literal["canny", "depth", "mlsd", "normal", "openpose", "scribble"] = Field(
        default="canny",
        description="Tipo de pré-processamento"
    )
    preprocessor_params: Dict = Field(
        default_factory=dict,
        description="Parâmetros para o pré-processador"
    )

class ImageRequest(BaseModel):
    """Modelo de requisição para geração de imagem"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra='forbid'
    )
    
    prompt: str
    negative_prompt: Optional[str] = None
    model: str = Field(default="sdxl")
    height: Annotated[int, Field(ge=256, le=2048)] = 512
    width: Annotated[int, Field(ge=256, le=2048)] = 512
    num_inference_steps: Annotated[int, Field(ge=1, le=150)] = 50
    template_id: Optional[str] = None
    loras: List[Dict] = Field(default_factory=list)
    controlnet: Optional[ControlNetConfig] = None
    vae: Optional[str] = Field(
        default=None,
        description="ID ou path do VAE personalizado"
    )
    scheduler: str = Field(
        default="DPMSolverMultistepScheduler",
        description="Scheduler para geração"
    )
    guidance_scale: Annotated[float, Field(ge=1.0, le=20.0)] = Field(
        default=7.5,
        description="Escala de guidance do CFG"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed para reprodutibilidade"
    )
    clip_skip: Optional[Annotated[int, Field(ge=1, le=4)]] = Field(
        default=None,
        description="Número de camadas CLIP a pular"
    )
    optimization: Dict = Field(
        default_factory=lambda: {
            "tomesd": {
                "enabled": True,
                "ratio": 0.4,
                "max_downsample": 2
            },
            "enable_vae_tiling": True,
            "enable_vae_slicing": True,
            "enable_attention_slicing": True,
            "enable_sequential_cpu_offload": False,
            "enable_model_cpu_offload": False,
            "torch_compile": True,
            "torch_compile_mode": "reduce-overhead"
        },
        description="Configurações de otimização de memória e performance"
    )

    def validate_dimensions(self):
        """Validações adicionais foram movidas para validators do modelo"""
        pass

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

        # Aplica o pré-processador
        if preprocessor == "canny":
            low_threshold = params.get("low_threshold", 100)
            high_threshold = params.get("high_threshold", 200)
            processed = cv2.Canny(image_np, low_threshold, high_threshold)
            processed = processed[:, :, None]
            processed = np.concatenate([processed, processed, processed], axis=2)
        elif preprocessor == "depth":
            from transformers import pipeline
            depth_estimator = pipeline("depth-estimation")
            processed = depth_estimator(image)["depth"]
            processed = np.array(processed)
            processed = np.stack([processed, processed, processed], axis=-1)
        elif preprocessor == "normal":
            from transformers import pipeline
            normal_estimator = pipeline("depth-estimation")
            processed = normal_estimator(image)["prediction"]
        elif preprocessor == "openpose":
            # Implementar OpenPose
            raise NotImplementedError("OpenPose ainda não implementado")
        elif preprocessor == "scribble":
            # Converte para escala de cinza e aplica detecção de bordas
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            raise ValueError(f"Pré-processador '{preprocessor}' não suportado")

        return Image.fromarray(processed)
    except Exception as e:
        raise RuntimeError(f"Erro no processamento da imagem: {str(e)}")

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

def load_pipeline(model_name: str, checkpoint_path: Optional[str] = None, controlnet_config: Optional[ControlNetConfig] = None, vae_id: Optional[str] = None):
    """
    Carrega o pipeline do modelo com otimizações de memória e performance.
    Suporta ControlNet e VAE personalizado.
    """
    global pipe
    try:
        if model_name.lower() == "flux":
            model_dir = checkpoint_path or "models/flux"
        else:
            model_dir = checkpoint_path or "models/sdxl"

        # Carrega ControlNet se especificado
        if controlnet_config:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_config.model_id,
                torch_dtype=TORCH_DTYPE
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_dir,
                controlnet=controlnet,
                torch_dtype=TORCH_DTYPE,
                variant="fp16" if DEVICE == "cuda" else None,
                use_safetensors=True
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_dir,
                torch_dtype=TORCH_DTYPE,
                variant="fp16" if DEVICE == "cuda" else None,
                use_safetensors=True
            )

        # Carrega VAE personalizado se especificado
        if vae_id:
            pipe.vae = AutoencoderKL.from_pretrained(
                vae_id,
                torch_dtype=TORCH_DTYPE
            )
        
        # Aplica otimizações de memória
        pipe = apply_memory_optimizations(pipe)
        
        # Move para GPU
        pipe = pipe.to(DEVICE)
        
        # Compila o modelo para melhor performance
        if USE_TORCH_COMPILE and DEVICE == "cuda":
            pipe.unet = torch.compile(
                pipe.unet,
                mode="reduce-overhead",
                fullgraph=True
            )
            pipe.vae = torch.compile(
                pipe.vae,
                mode="reduce-overhead",
                fullgraph=True
            )
        
        # Multi-GPU se disponível
        if torch.cuda.device_count() > 1:
            pipe.unet = torch.nn.DataParallel(pipe.unet)
            pipe.vae = torch.nn.DataParallel(pipe.vae)

        # Configura o sampler personalizado
        if hasattr(pipe, 'scheduler'):
            scheduler_name = pipe.scheduler.__class__.__name__
            if scheduler_name in AVAILABLE_SCHEDULERS:
                sampler_name = AVAILABLE_SCHEDULERS[scheduler_name]
                pipe._sampler = get_sampler(sampler_name)
            else:
                pipe._sampler = get_sampler('ddim')  # Sampler padrão
        
        return pipe
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o pipeline: {str(e)}")

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
async def generate_image_endpoint(
    request: ImageRequest,
    token: str = Depends(verify_bearer_token)
):
    """Endpoint para geração de imagem com fila"""
    # Valida as dimensões
    request.validate_dimensions()
    
    # Gera ID único para a tarefa
    task_id = str(uuid.uuid4())
    
    # Adiciona à fila
    status = await queue_manager.add_task(task_id, request)
    
    return {
        "task_id": task_id,
        "status": status.status,
        "message": "Tarefa adicionada à fila com sucesso"
    }

@app.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    token: str = Depends(verify_bearer_token)
):
    """Obtém o status de uma tarefa"""
    return await queue_manager.get_task_status(task_id)

@app.delete("/task/{task_id}")
async def cancel_task(
    task_id: str,
    token: str = Depends(verify_bearer_token)
):
    """Cancela uma tarefa pendente"""
    await queue_manager.cancel_task(task_id)
    return {"message": "Tarefa cancelada com sucesso"}

@app.get("/queue/status")
async def get_queue_status(token: str = Depends(verify_bearer_token)):
    """Obtém status geral da fila"""
    return await queue_manager.get_queue_info()

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

            if controlnet_image:
                generation_args["image"] = controlnet_image
                generation_args["controlnet_conditioning_scale"] = request.controlnet.conditioning_scale

            # Gera imagem com otimização de memória
            with torch.cuda.amp.autocast():
                result = pipe(**generation_args)
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
                    "processed_negative_prompt": processed_negative_prompt
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração da imagem: {str(e)}")

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
    await asyncio.get_event_loop().run_in_executor(None, lambda: load_pipeline("sdxl"))