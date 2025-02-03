import asyncio
import base64
import json
import os
import time
import uuid
import logging
from io import BytesIO

from fastapi import FastAPI, HTTPException, Header, Depends, status
from pydantic import BaseModel
from typing import Optional

import torch

# Importa a classe do Hunyuan Video Generator (ajuste conforme o repositório oficial)
from hunyuan import HunyuanVideoGenerator

# Importa o cliente do MinIO e erros
from minio import Minio
from minio.error import S3Error

app = FastAPI()

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
    duration: Optional[int] = 10      # Duração do vídeo em segundos
    height: Optional[int] = 720
    width: Optional[int] = 1280
    fps: Optional[int] = 24           # Frames por segundo (padrão 24)
    template_id: Optional[str] = None   # Se informado, carrega os parâmetros do template

    # Validações para evitar consumo excessivo de recursos
    def validate_dimensions(self):
        """Valida as dimensões do vídeo para evitar consumo excessivo de recursos"""
        MAX_WIDTH = 1920
        MAX_HEIGHT = 1080
        MAX_DURATION = 60  # segundos
        MAX_FPS = 30

        if self.width > MAX_WIDTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Largura máxima permitida é {MAX_WIDTH}px"
            )
        if self.height > MAX_HEIGHT:
            raise HTTPException(
                status_code=400, 
                detail=f"Altura máxima permitida é {MAX_HEIGHT}px"
            )
        if self.duration > MAX_DURATION:
            raise HTTPException(
                status_code=400, 
                detail=f"Duração máxima permitida é {MAX_DURATION} segundos"
            )
        if self.fps > MAX_FPS:
            raise HTTPException(
                status_code=400, 
                detail=f"FPS máximo permitido é {MAX_FPS}"
            )

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
# 3. Carregamento do Gerador de Vídeo (Hunyuan)
######################################
video_generator = None

def load_video_generator(model_path: str = "models/hunyuan_video"):
    """
    Carrega o gerador de vídeo Hunyuan a partir de um checkpoint local.
    Ajuste os parâmetros conforme a documentação oficial do Hunyuan.
    """
    global video_generator
    try:
        video_generator = HunyuanVideoGenerator.from_pretrained(model_path)
        video_generator = video_generator.to("cuda")
        if torch.cuda.device_count() > 1:
            video_generator = torch.nn.DataParallel(video_generator)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o gerador de vídeo: {str(e)}")
    return video_generator

@app.on_event("startup")
def startup_event():
    load_video_generator()

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
# 5. Endpoint de Geração de Vídeo (Hunyuan) - Assíncrono
######################################
@app.post("/generate-video")
async def generate_video(request: VideoRequest, token: str = Depends(verify_bearer_token)):
    """
    Gera vídeo utilizando o modelo Hunyuan.
    - Se um template for informado, os parâmetros (prompt, duration, height, width, fps) são carregados.
    - O vídeo é gerado de forma assíncrona.
    - O arquivo de vídeo final é enviado para o MinIO e uma URL pré-assinada (válida por 2 dias) é retornada.
    """
    # Valida as dimensões antes de prosseguir
    request.validate_dimensions()

    if request.template_id:
        try:
            template = load_template(request.template_id)
            request.prompt = template.get("prompt", request.prompt)
            request.duration = template.get("duration", request.duration)
            request.height = template.get("height", request.height)
            request.width = template.get("width", request.width)
            request.fps = template.get("fps", request.fps)
            # Valida novamente após carregar o template
            request.validate_dimensions()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao carregar template: {str(e)}")
    
    global video_generator
    if video_generator is None:
        load_video_generator()
    
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: video_generator.generate(
                prompt=request.prompt,
                duration=request.duration,
                height=request.height,
                width=request.width,
                fps=request.fps
            )
        )
        video_bytes = result.video_bytes
        
        file_name = f"{uuid.uuid4()}_{int(time.time())}.mp4"
        minio_url = upload_video_to_minio(video_bytes, file_name)
        
        return {
            "status": "sucesso",
            "job_id": f"video_{int(time.time())}",
            "message": "Vídeo gerado com sucesso.",
            "minio_url": minio_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração de vídeo: {str(e)}")