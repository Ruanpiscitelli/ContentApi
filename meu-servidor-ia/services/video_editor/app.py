"""
Módulo principal da API de edição de vídeo.
Implementa endpoints e configurações do FastAPI.
"""

import asyncio
import base64
import json
import os
import subprocess
import time
import uuid
import logging
import re
import requests
from tempfile import TemporaryDirectory
from io import BytesIO
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from minio import Minio
from minio.error import S3Error

from .config import settings
from .security import verify_bearer_token
from .models import Operation, VideoRequest, VideoEditRequest, SilenceCutRequest
from .pipeline import VideoPipeline
from .routers import video

# Configuração do Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa FastAPI com configurações
app = FastAPI(
    title="Serviço de Edição de Vídeo",
    description="API para edição e processamento de vídeos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Adiciona CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente MinIO
minio_client = Minio(**settings["minio"])

# Pipeline de processamento
pipeline = VideoPipeline()

# Inclui routers
app.include_router(video.router)

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
        content={
            "detail": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para exceções gerais."""
    logger.error(f"Erro não tratado: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erro interno do servidor"
        }
    )

@app.on_event("startup")
async def startup():
    """Inicializa recursos na inicialização."""
    # Inicializa pipeline
    pipeline = VideoPipeline()
    await pipeline.start()
    
@app.on_event("shutdown")
async def shutdown():
    """Libera recursos no desligamento."""
    # Para pipeline
    await pipeline.stop()

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

# Configurações do MinIO (use variáveis de ambiente para produção)
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_HOST", "minio.ruanpiscitelli.com"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", "YYCZ0Fl0gu1nx2LaTORS"),
    "secret_key": os.getenv("MINIO_SECRET_KEY", "gB0Kl7BWBPolCwLz29OyEPQiOBLnrlAtHqx3cK1Q"),
    "secure": True
}
minio_client = Minio(**MINIO_CONFIG)

# Token de autenticação para o endpoint de edição
AUTH_TOKEN = os.getenv("VIDEO_EDITOR_AUTH_TOKEN", "PROD_TOKEN_EXEMPLO")

# Modelos Pydantic

class Operation(BaseModel):
    action: str
    start: Optional[float] = None
    end: Optional[float] = None
    text: Optional[str] = None
    font_size: Optional[int] = None
    font_color: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    position: Optional[str] = None
    type: Optional[str] = None
    duration: Optional[float] = None
    image_source: Optional[str] = None

class Element(BaseModel):
    id: int
    type: str  # "image" ou "audio"
    src: str
    duration: float = -1
    zoom: float = 1.0
    width: Optional[int] = None
    height: Optional[int] = None
    position: str = "center-center"

class Scene(BaseModel):
    elements: List[Element]

class VideoRequest(BaseModel):
    id: str
    fps: int = 30
    width: int = 1920
    height: int = 1080
    scenes: List[Scene]
    quality: str = "high"
    settings: Dict[str, Any] = {}  # Pode conter "video_bitrate", "audio_bitrate", "output_format", etc.

# Para edição avançada, definimos modelos mais específicos:
class InputItem(BaseModel):
    source: str
    start: Optional[float] = None
    end: Optional[float] = None

class OutputConfig(BaseModel):
    filename: str
    codec: str = "libx264"
    resolution: str = "1920x1080"
    bitrate: str = "2000k"  # Vídeo bitrate padrão
    format: str = "mp4"

class EditInstructions(BaseModel):
    inputs: List[InputItem]
    operations: List[Operation]
    output: OutputConfig

class VideoEditRequest(BaseModel):
    instructions: EditInstructions

class SilenceCutRequest(BaseModel):
    input_url: str
    silence_threshold: float = Field(default=-35, description="Limiar de silêncio em dB (default: -35)")
    min_silence_duration: float = Field(default=1.0, description="Duração mínima do silêncio em segundos")
    output_format: str = Field(default="mp4", description="Formato do vídeo de saída")

# Funções Auxiliares

def download_from_minio(source_uri: str, local_dir: str) -> str:
    if not source_uri.startswith("minio://"):
        return source_uri
    try:
        bucket, *path = source_uri[8:].split("/")
        object_name = "/".join(path)
        local_path = os.path.join(local_dir, object_name.replace("/", "_"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        minio_client.fget_object(bucket, object_name, local_path)
        logger.info(f"Download do objeto {object_name} do bucket {bucket} para {local_path}")
        return local_path
    except S3Error as e:
        raise RuntimeError(f"Erro ao baixar do MinIO: {e}")

def is_google_drive_url(url: str) -> bool:
    patterns = [
        r"https?://drive\.google\.com/file/d/([^/]+)",
        r"https?://drive\.google\.com/open\?id=([^&]+)",
        r"https?://drive\.google\.com/uc\?id=([^&]+)"
    ]
    return any(re.match(pattern, url) for pattern in patterns)

def is_dropbox_url(url: str) -> bool:
    return "dropbox.com" in url

def is_supabase_url(url: str) -> bool:
    return "supabase.co" in url and "/storage/v1/object" in url

def get_google_drive_direct_url(url: str) -> str:
    file_id = None
    if "/file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
    elif "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
    if not file_id:
        raise ValueError("ID do arquivo do Google Drive não encontrado na URL")
    return f"https://drive.google.com/uc?id={file_id}&export=download"

def get_dropbox_direct_url(url: str) -> str:
    base_url = url.split("?")[0]
    return f"{base_url}?dl=1"

def download_from_url(url: str, local_dir: str) -> str:
    try:
        if is_google_drive_url(url):
            url = get_google_drive_direct_url(url)
            session = requests.Session()
            response = session.get(url, stream=True)
            if "Content-Disposition" in response.headers:
                filename_match = re.findall("filename=(.+)", response.headers["Content-Disposition"])
                filename = filename_match[0].strip('"') if filename_match else f"gdrive_file_{uuid.uuid4()}"
            else:
                filename = f"gdrive_file_{uuid.uuid4()}"
        elif is_dropbox_url(url):
            url = get_dropbox_direct_url(url)
            response = requests.get(url, stream=True)
            filename = url.split("/")[-1].split("?")[0]
        elif is_supabase_url(url):
            response = requests.get(url, stream=True)
            filename = url.split("/")[-1].split("?")[0]
        else:
            response = requests.get(url, stream=True)
            filename = url.split("/")[-1].split("?")[0]
        
        if response.status_code != 200:
            raise RuntimeError(f"Erro ao baixar arquivo: {response.status_code}")
        
        local_path = os.path.join(local_dir, filename)
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Arquivo baixado de {url} para {local_path}")
        return local_path
        
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar arquivo: {str(e)}")

def build_ffmpeg_command_basic(request: VideoRequest, tmpdir: str) -> (List[str], str):
    """
    Constrói o comando FFmpeg para edição básica de vídeo.
    Permite escolher FPS, vídeo bitrate, áudio bitrate e formato de saída através de request.settings.
    """
    cmd = ["ffmpeg", "-y"]
    input_files = []
    filter_complex = []
    scene_elements = []
    
    # Processa cada cena
    for scene_idx, scene in enumerate(request.scenes):
        for elem_idx, elem in enumerate(scene.elements):
            local_path = download_from_url(elem.src, tmpdir)
            input_files.append(local_path)
            input_idx = len(input_files) - 1
            
            if elem.type == "image":
                filter_complex.append(
                    f"[{input_idx}:v]scale={request.width}:{request.height}:force_original_aspect_ratio=decrease,"
                    f"pad={request.width}:{request.height}:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS[v{scene_idx}_{elem_idx}];"
                )
                scene_elements.append(f"[v{scene_idx}_{elem_idx}]")
            elif elem.type == "audio":
                filter_complex.append(
                    f"[{input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,volume=0.8[a{scene_idx}_{elem_idx}];"
                )
                scene_elements.append(f"[a{scene_idx}_{elem_idx}]")
    
    # Adiciona inputs
    for input_file in input_files:
        cmd.extend(["-i", input_file])
    
    # Concatena as cenas
    filter_complex.append(f"{''.join(scene_elements)}concat=n={len(request.scenes)}:v=1:a=1[outv][outa]")
    cmd.extend(["-filter_complex", "".join(filter_complex)])
    
    # Adiciona opção para definir FPS
    cmd.extend(["-r", str(request.fps)])
    
    # Define parâmetros de saída a partir de settings
    output_format = request.settings.get("output_format", "mp4")
    video_bitrate = request.settings.get("video_bitrate", "2000k")
    audio_bitrate = request.settings.get("audio_bitrate", "192k")
    ext = output_format  # Utiliza o mesmo para extensão
    
    output_file = os.path.join(tmpdir, f"output_{uuid.uuid4()}.{ext}")
    cmd.extend([
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "slow" if request.quality == "high" else "medium",
        "-crf", "18" if request.quality == "high" else "23",
        "-b:v", video_bitrate,
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-f", output_format,
        output_file
    ])
    
    return cmd, output_file

# Endpoints

@app.post("/edit-video/basic")
async def video_editor_basic(request: VideoRequest):
    try:
        with TemporaryDirectory() as tmpdir:
            cmd, output_file = build_ffmpeg_command_basic(request, tmpdir)
            
            # Validação dos arquivos de entrada
            for scene in request.scenes:
                for elem in scene.elements:
                    local_file = download_from_url(elem.src, tmpdir)
                    if not (validate_video_file(local_file) or validate_video_format(local_file)):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Arquivo inválido ou formato não suportado: {elem.src}"
                        )
            
            # Executa o comando FFmpeg
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Erro no FFmpeg: {stderr.decode()}")
            
            with open(output_file, "rb") as f:
                video_bytes = f.read()
            
            url = upload_to_minio(video_bytes, "videos-edited", f"{request.id}_{uuid.uuid4()}.mp4")
            return {
                "status": "success",
                "url": url,
                "job_id": f"job_{uuid.uuid4()}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit-video/advanced", dependencies=[Depends(verify_token)])
async def video_editor_advanced(request: VideoEditRequest):
    try:
        with TemporaryDirectory() as tmpdir:
            inputs = []
            has_audio = False
            for inp in request.instructions.inputs:
                local_path = download_from_url(inp.source, tmpdir)
                if not validate_video_file(local_path):
                    raise HTTPException(status_code=400, detail=f"Arquivo inválido ou formato não suportado: {inp.source}")
                if check_audio_stream(local_path):
                    has_audio = True
                inputs.append({
                    "path": local_path,
                    "start": inp.start,
                    "end": inp.end,
                    "has_audio": check_audio_stream(local_path)
                })
            
            cmd = ["ffmpeg", "-y"]
            for item in inputs:
                if item["start"] is not None:
                    cmd += ["-ss", str(item["start"])]
                if item["end"] is not None:
                    cmd += ["-to", str(item["end"])]
                cmd += ["-i", item["path"]]
            
            filter_complex = ""
            if request.instructions.operations:
                filter_complex = build_complex_filter(request.instructions.operations, len(inputs))
                cmd += ["-filter_complex", filter_complex]
                cmd += ["-map", "[outv]"]
                if has_audio:
                    for i, item in enumerate(inputs):
                        if item["has_audio"]:
                            cmd += ["-map", f"{i}:a?"]
                            break
            
            output_conf = request.instructions.output
            output_file = os.path.join(tmpdir, output_conf.filename)
            cmd += [
                "-c:v", output_conf.codec,
                "-s", output_conf.resolution,
                "-b:v", output_conf.bitrate
            ]
            if has_audio:
                cmd += [
                    "-c:a", "aac",
                    "-b:a", "192k"
                ]
            cmd += [
                "-f", output_conf.format,
                output_file
            ]
            
            proc = await asyncio.create_subprocess_exec(*cmd, stderr=subprocess.PIPE)
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
            
            with open(output_file, "rb") as f:
                video_bytes = f.read()
            
            url = upload_to_minio(video_bytes, "videos-edited", f"{uuid.uuid4()}.mp4")
            return {
                "status": "success",
                "url": url,
                "job_id": f"job_{uuid.uuid4()}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cut-silence")
async def cut_silence(request: SilenceCutRequest):
    try:
        with TemporaryDirectory() as tmpdir:
            input_file = download_from_url(request.input_url, tmpdir)
            if not validate_video_file(input_file):
                raise HTTPException(status_code=400, detail="O arquivo fornecido não é um vídeo válido")
            if not validate_video_format(input_file):
                raise HTTPException(status_code=400, detail="Formato de vídeo não suportado")
            
            silences = find_silences(input_file, db=request.silence_threshold, min_duration=request.min_silence_duration)
            if not silences:
                return {
                    "status": "success",
                    "message": "Nenhum silêncio detectado no vídeo",
                    "url": request.input_url
                }
            
            duration = get_video_duration(input_file)
            output_file = os.path.join(tmpdir, f"output_no_silence_{uuid.uuid4()}.{request.output_format}")
            cmd = build_silence_cut_command(input_file, output_file, silences, duration)
            
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Erro no FFmpeg: {stderr.decode()}")
            
            with open(output_file, "rb") as f:
                video_bytes = f.read()
            
            url = upload_to_minio(video_bytes, "videos-edited", f"no_silence_{uuid.uuid4()}.{request.output_format}")
            return {
                "status": "success",
                "url": url,
                "job_id": f"job_{uuid.uuid4()}",
                "silences_removed": len(silences) // 2
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Funções auxiliares de posicionamento e validação

def get_position(pos: str) -> str:
    positions = {
        "top-right": "main_w-overlay_w-10:10",
        "bottom-right": "main_w-overlay_w-10:main_h-overlay_h-10",
        "bottom-left": "10:main_h-overlay_h-10",
        "center": "(main_w-overlay_w)/2:(main_h-overlay_h)/2"
    }
    return positions.get(pos, "10:10")

def get_text_position(pos: str) -> str:
    positions = {
        "bottom-center": "x=(w-text_w)/2:y=h-text_h-10",
        "top-center": "x=(w-text_w)/2:y=10",
        "center": "x=(w-text_w)/2:y=(h-text_h)/2"
    }
    return positions.get(pos, "x=10:y=10")

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido")
    token = authorization.split(" ")[1]
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Acesso negado")
    return token

def validate_video_file(file_path: str) -> bool:
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0 and b"video" in result.stdout
    except Exception:
        return False

def validate_video_format(file_path: str, allowed_formats: List[str] = None) -> bool:
    if not allowed_formats:
        allowed_formats = ["mp4", "avi", "mov", "mkv", "webm"]
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=format_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        format_name = result.stdout.decode().strip()
        return any(fmt in format_name.lower() for fmt in allowed_formats)
    except Exception:
        return False

def check_audio_stream(file_path: str) -> bool:
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0 and b"audio" in result.stdout
    except Exception:
        return False

def find_silences(filename: str, db: float = -35, min_duration: float = 1.0) -> List[float]:
    command = [
        "ffmpeg", "-i", filename,
        "-af", f"silencedetect=n={db}dB:d={min_duration}",
        "-f", "null", "-"
    ]
    output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = output.stderr.decode()
    time_list = []
    for line in stderr.split("\n"):
        if "silence_start" in line:
            time_list.append(float(line.split("silence_start: ")[1].split()[0]))
        elif "silence_end" in line:
            time_list.append(float(line.split("silence_end: ")[1].split()[0]))
    return time_list

def get_video_duration(filename: str) -> float:
    if not os.path.exists(filename):
        raise ValueError(f"O arquivo não existe: {filename}")
    if not validate_video_file(filename):
        raise ValueError(f"O arquivo não é um vídeo válido: {filename}")
    try:
        command = [
            "ffprobe", "-i", filename, "-v", "quiet",
            "-show_entries", "format=duration",
            "-hide_banner", "-of", "default=noprint_wrappers=1:nokey=1"
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Erro ao processar o vídeo: {result.stderr.decode()}")
        duration = result.stdout.decode().strip()
        if not duration:
            raise RuntimeError("Não foi possível determinar a duração do vídeo")
        return float(duration)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout ao processar o vídeo")
    except ValueError as e:
        raise ValueError(f"Erro ao converter duração do vídeo: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Erro inesperado ao processar o vídeo: {str(e)}")

def build_silence_cut_command(input_file: str, output_file: str, silences: List[float], duration: float) -> List[str]:
    segments = [0.0] + silences + [duration]
    select_parts = []
    for i in range(0, len(segments)-1, 2):
        start = segments[i]
        end = segments[i+1]
        select_parts.append(f"between(t,{start},{end})")
    select_expr = "+".join(select_parts)
    return [
        "ffmpeg", "-i", input_file,
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-af", f"aselect='{select_expr}',asetpts=N/SR/TB",
        "-y", output_file
    ]

def upload_to_minio(file_bytes: bytes, bucket: str, file_name: str) -> str:
    try:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
        minio_client.put_object(
            bucket,
            file_name,
            data=BytesIO(file_bytes),
            length=len(file_bytes),
            content_type="video/mp4"
        )
        url = minio_client.presigned_get_object(bucket, file_name, expires=2*24*60*60)
        return url
    except S3Error as err:
        raise RuntimeError(f"Erro no upload para o MinIO: {err}")

@app.get("/")
async def root():
    return {"message": "Video Editor Service"}
