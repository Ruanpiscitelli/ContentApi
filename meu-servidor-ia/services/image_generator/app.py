import asyncio
import base64
import json
import os
from io import BytesIO
import time
import uuid

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional

from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch

# Importa o cliente do MinIO e os erros
from minio import Minio
from minio.error import S3Error

app = FastAPI()

######################################
# 1. Validação de Token
######################################
def verify_bearer_token(authorization: str = Header(...)):
    """
    Valida o header Authorization e extrai o token.
    Em produção, utilize variáveis de ambiente para gerenciar segredos.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido")
    token = authorization.split(" ")[1]
    if token != "SEU_TOKEN_SEGREDO":  # Substitua por um valor seguro ou utilize variável de ambiente
        raise HTTPException(status_code=401, detail="Não autorizado")
    return token

######################################
# 2. Modelo de Requisição e Templates
######################################
class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model: Optional[str] = "sdxl"       # "sdxl" ou "flux"
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    template_id: Optional[str] = None   # Se informado, carrega os parâmetros do template
    loras: Optional[list] = []         # Lista de LoRAs a serem aplicados

    # Validações para evitar consumo excessivo de recursos
    def validate_dimensions(self):
        """Valida as dimensões da imagem para evitar consumo excessivo de recursos"""
        MAX_WIDTH = 2048
        MAX_HEIGHT = 2048
        MIN_WIDTH = 256
        MIN_HEIGHT = 256
        MAX_STEPS = 150

        if self.width > MAX_WIDTH or self.width < MIN_WIDTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Largura deve estar entre {MIN_WIDTH}px e {MAX_WIDTH}px"
            )
        if self.height > MAX_HEIGHT or self.height < MIN_HEIGHT:
            raise HTTPException(
                status_code=400, 
                detail=f"Altura deve estar entre {MIN_HEIGHT}px e {MAX_HEIGHT}px"
            )
        if self.num_inference_steps > MAX_STEPS:
            raise HTTPException(
                status_code=400, 
                detail=f"Número máximo de passos de inferência é {MAX_STEPS}"
            )

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

def load_pipeline(model_name: str, checkpoint_path: Optional[str] = None):
    """
    Carrega o pipeline do modelo a partir de um caminho local.
    Se model_name for "flux", utiliza "models/flux" (ou checkpoint_path se fornecido);
    caso contrário, utiliza "models/sdxl" como padrão.
    O pipeline é movido para "cuda". Se houver mais de uma GPU disponível, utiliza DataParallel para o UNet.
    """
    global pipe
    try:
        if model_name.lower() == "flux":
            model_dir = checkpoint_path or "models/flux"
        else:
            model_dir = checkpoint_path or "models/sdxl"
            
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir, revision="fp16", torch_dtype=torch.float16
        )
        # Se necessário, carregar componentes adicionais (VAE, CLIP) a partir de diretórios locais
        pipe = pipe.to("cuda")
        if torch.cuda.device_count() > 1:
            pipe.unet = torch.nn.DataParallel(pipe.unet)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o pipeline: {str(e)}")
    return pipe

@app.on_event("startup")
def startup_event():
    # Carrega o pipeline padrão (modelo "sdxl")
    load_pipeline("sdxl")

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
    Aplica os LoRAs ao pipeline.
    Essa função é um placeholder: implemente a lógica real para mesclar os pesos dos LoRAs.
    """
    for lora in loras:
        lora_path = lora.get("path")
        scale = lora.get("scale", 1.0)
        print(f"Aplicando LoRA: {lora_path} com scale {scale}")
    return pipeline

######################################
# 6. Endpoint de Geração de Imagem (Assíncrono)
######################################
@app.post("/generate-image")
async def generate_image(request: ImageRequest, token: str = Depends(verify_bearer_token)):
    global pipe

    # Valida as dimensões antes de prosseguir
    request.validate_dimensions()

    # Se um template for informado, carrega-o e sobrescreve os parâmetros
    if request.template_id:
        try:
            template = load_template(request.template_id)
            request.model = template.get("model", request.model)
            request.prompt = template.get("prompt", request.prompt)
            request.negative_prompt = template.get("negative_prompt", request.negative_prompt)
            request.height = template.get("height", request.height)
            request.width = template.get("width", request.width)
            request.num_inference_steps = template.get("num_inference_steps", request.num_inference_steps)
            # Se o template definir LoRAs, sobrescreve o campo "loras"
            request.loras = template.get("loras", request.loras)
            # Valida novamente após carregar o template
            request.validate_dimensions()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao carregar template: {str(e)}")
    
    # Recarrega o pipeline se necessário (simples verificação; pode ser refinado)
    if pipe is None or request.model.lower() not in ["sdxl", "flux"]:
        load_pipeline(request.model)
    
    try:
        # Aplica os LoRAs se especificados
        if request.loras:
            pipe = apply_loras(pipe, request.loras)
            
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: pipe(
                request.prompt,
                negative_prompt=request.negative_prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps
            )
        )
        image = result.images[0]
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        file_bytes = buffered.getvalue()
        
        # Gera um nome único para o arquivo
        file_name = f"{uuid.uuid4()}_{int(time.time())}.png"
        minio_url = upload_to_minio(file_bytes, file_name)
        
        return {
            "status": "sucesso",
            "message": "Imagem gerada com sucesso.",
            "minio_url": minio_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração da imagem: {str(e)}")