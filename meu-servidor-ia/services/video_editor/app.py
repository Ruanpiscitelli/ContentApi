import asyncio
import base64
import json
import os
import subprocess
import time
import uuid
from io import BytesIO
from typing import List, Optional, Dict, Union, Literal
from tempfile import TemporaryDirectory
import re
import requests
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from minio import Minio
from minio.error import S3Error

app = FastAPI()

# Configurações do MinIO
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_HOST", "minio.ruanpiscitelli.com"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", "YYCZ0Fl0gu1nx2LaTORS"),
    "secret_key": os.getenv("MINIO_SECRET_KEY", "gB0Kl7BWBPolCwLz29OyEPQiOBLnrlAtHqx3cK1Q"),
    "secure": True
}

# Token de autenticação via variável de ambiente
AUTH_TOKEN = os.getenv("VIDEO_EDITOR_AUTH_TOKEN", "PROD_TOKEN_EXEMPLO")

minio_client = Minio(**MINIO_CONFIG)

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
    type: str
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
    settings: Dict = {}

class VideoEditRequest(BaseModel):
    instructions: Dict[str, any]

class SilenceCutRequest(BaseModel):
    """
    Modelo para requisição de corte de silêncio
    """
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
        return local_path
    except S3Error as e:
        raise RuntimeError(f"Erro ao baixar do MinIO: {e}")

def is_google_drive_url(url: str) -> bool:
    """Verifica se a URL é do Google Drive"""
    patterns = [
        r"https?://drive\.google\.com/file/d/([^/]+)",
        r"https?://drive\.google\.com/open\?id=([^/]+)",
        r"https?://drive\.google\.com/uc\?id=([^/]+)"
    ]
    return any(re.match(pattern, url) for pattern in patterns)

def is_dropbox_url(url: str) -> bool:
    """Verifica se a URL é do Dropbox"""
    return "dropbox.com" in url

def is_supabase_url(url: str) -> bool:
    """Verifica se a URL é do Supabase Storage"""
    return "supabase.co" in url and "/storage/v1/object" in url

def get_google_drive_direct_url(url: str) -> str:
    """Converte URL do Google Drive para URL direta de download"""
    file_id = None
    
    # Extrai o ID do arquivo da URL
    if "/file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
    elif "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
    
    if not file_id:
        raise ValueError("ID do arquivo do Google Drive não encontrado na URL")
    
    return f"https://drive.google.com/uc?id={file_id}&export=download"

def get_dropbox_direct_url(url: str) -> str:
    """Converte URL do Dropbox para URL direta de download"""
    # Remove '?dl=0' ou '?dl=1' e adiciona '?dl=1'
    base_url = url.split("?")[0]
    return f"{base_url}?dl=1"

def download_from_url(url: str, local_dir: str) -> str:
    """Função melhorada para download de arquivos de várias fontes"""
    try:
        # Verifica o tipo de URL e converte para URL direta se necessário
        if is_google_drive_url(url):
            url = get_google_drive_direct_url(url)
            # Configuração especial para Google Drive
            session = requests.Session()
            response = session.get(url, stream=True)
            
            # Trata o aviso de download do Google Drive
            if "Content-Disposition" in response.headers:
                filename_match = re.findall("filename=(.+)", response.headers["Content-Disposition"])
                if filename_match:
                    filename = filename_match[0].strip('"')
                else:
                    filename = f"gdrive_file_{uuid.uuid4()}"
            else:
                filename = f"gdrive_file_{uuid.uuid4()}"
                
        elif is_dropbox_url(url):
            url = get_dropbox_direct_url(url)
            response = requests.get(url, stream=True)
            filename = url.split("/")[-1].split("?")[0]
            
        elif is_supabase_url(url):
            # Supabase já fornece URL direta
            response = requests.get(url, stream=True)
            filename = url.split("/")[-1].split("?")[0]
            
        else:
            # Download padrão para outras URLs
            response = requests.get(url, stream=True)
            filename = url.split("/")[-1].split("?")[0]
        
        if response.status_code != 200:
            raise RuntimeError(f"Erro ao baixar arquivo: {response.status_code}")
        
        local_path = os.path.join(local_dir, filename)
        
        # Download em chunks para arquivos grandes
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return local_path
        
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar arquivo: {str(e)}")

def build_ffmpeg_command(request: VideoRequest, tmpdir: str) -> List[str]:
    cmd = ["ffmpeg", "-y"]
    
    # Lista para armazenar os arquivos de entrada
    input_files = []
    filter_complex = []
    
    # Processa cada cena
    for scene_idx, scene in enumerate(request.scenes):
        scene_elements = []
        
        # Processa elementos da cena
        for elem_idx, elem in enumerate(scene.elements):
            # Download do arquivo
            local_path = download_from_url(elem.src, tmpdir)
            input_files.append(local_path)
            input_idx = len(input_files) - 1
            
            if elem.type == "image":
                # Configuração para imagem
                filter_complex.extend([
                    f"[{input_idx}:v]scale={request.width}:{request.height}:force_original_aspect_ratio=decrease,",
                    f"pad={request.width}:{request.height}:(ow-iw)/2:(oh-ih)/2,",
                    f"setpts=PTS-STARTPTS+{scene_idx*5}/TB,",
                    f"zoompan=z='min(zoom+0.002,1.5)':d={elem.duration if elem.duration > 0 else 5}[v{scene_idx}_{elem_idx}];"
                ])
                scene_elements.append(f"[v{scene_idx}_{elem_idx}]")
            
            elif elem.type == "audio":
                # Configuração para áudio
                filter_complex.extend([
                    f"[{input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,",
                    f"volume=0.8[a{scene_idx}_{elem_idx}];"
                ])
                scene_elements.append(f"[a{scene_idx}_{elem_idx}]")
    
    # Adiciona inputs ao comando
    for input_file in input_files:
        cmd.extend(["-i", input_file])
    
    # Concatena os elementos de cada cena
    filter_complex.append(f"{''.join(scene_elements)}concat=n={len(request.scenes)}:v=1:a=1[outv][outa]")
    
    # Adiciona o filtro complexo ao comando
    cmd.extend(["-filter_complex", "".join(filter_complex)])
    
    # Configurações de saída
    output_file = os.path.join(tmpdir, f"output_{uuid.uuid4()}.mp4")
    cmd.extend([
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "slow" if request.quality == "high" else "medium",
        "-crf", "18" if request.quality == "high" else "23",
        "-c:a", "aac",
        "-b:a", "192k",
        output_file
    ])
    
    return cmd, output_file

@app.post("/edit-video/basic")
async def video_editor_basic(request: VideoRequest):
    try:
        with TemporaryDirectory() as tmpdir:
            # Constrói e executa o comando FFmpeg
            cmd, output_file = build_ffmpeg_command(request, tmpdir)
            
            # Valida os arquivos de entrada antes de processar
            for scene in request.scenes:
                for elem in scene.elements:
                    input_file = download_from_url(elem.src, tmpdir)
                    if not validate_video_file(input_file) and not validate_video_format(input_file):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Arquivo inválido ou formato não suportado: {elem.src}"
                        )
            
            # Executa FFmpeg
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                raise RuntimeError(f"Erro no FFmpeg: {stderr.decode()}")
            
            # Upload do resultado
            with open(output_file, "rb") as f:
                video_bytes = f.read()
            
            url = upload_to_minio(
                video_bytes,
                "videos-edited",
                f"{request.id}_{uuid.uuid4()}.mp4"
            )
            
            return {
                "status": "success",
                "url": url,
                "job_id": f"job_{uuid.uuid4()}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Funções auxiliares de posicionamento
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

# Validação de Token
def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido")
    if authorization.split(" ")[1] != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Acesso negado")

# Processamento de Vídeo
def build_complex_filter(operations: List[Operation], inputs_count: int) -> str:
    filters = []
    video_stream = "[0:v]"
    audio_stream = "[0:a]"
    overlay_index = inputs_count

    for i, op in enumerate(operations):
        if op.action == "trim":
            filters.append(f"{video_stream}trim=start={op.start}:end={op.end},setpts=PTS-STARTPTS[v{i}];")
            video_stream = f"[v{i}]"
        
        elif op.action == "overlay":
            filters.append(f"{video_stream}[{overlay_index}:v]scale={op.width or -1}:{op.height or -1}[ov{i}];")
            filters.append(f"[v{i-1}][ov{i}]overlay={get_position(op.position)}:format=auto[v{i}];")
            overlay_index += 1
            video_stream = f"[v{i}]"
        
        elif op.action == "text":
            text = op.text.replace("'", r"\'")
            position = get_text_position(op.position)
            filters.append(
                f"{video_stream}drawtext="
                f"text='{text}':{position}:"
                f"fontsize={op.font_size}:fontcolor={op.font_color}:"
                f"enable='between(t,{op.start_time},{op.end_time or 1e6})'[v{i}];"
            )
            video_stream = f"[v{i}]"
        
        elif op.action == "transition":
            filters.append(
                f"{video_stream}fade=type={op.type}:duration={op.duration}[v{i}];"
            )
            video_stream = f"[v{i}]"

    return "".join(filters).rstrip(";")

# Endpoint Principal
@app.post("/edit-video/advanced", dependencies=[Depends(verify_token)])
async def video_editor_advanced(request: VideoEditRequest):
    try:
        with TemporaryDirectory() as tmpdir:
            # Preparar inputs
            inputs = []
            has_audio = False
            
            for inp in request.instructions.inputs:
                local_path = download_from_minio(inp.source, tmpdir)
                
                # Validar arquivo de entrada
                if not validate_video_file(local_path) or not validate_video_format(local_path):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Arquivo inválido ou formato não suportado: {inp.source}"
                    )
                
                # Verifica se tem áudio
                if check_audio_stream(local_path):
                    has_audio = True
                
                inputs.append({
                    "path": local_path,
                    "start": inp.start,
                    "end": inp.end,
                    "has_audio": check_audio_stream(local_path)
                })

            # Construir comando FFmpeg
            cmd = ["ffmpeg", "-y"]
            
            # Adicionar inputs com trim
            for inp in inputs:
                if inp["start"] is not None:
                    cmd += ["-ss", str(inp["start"])]
                if inp["end"] is not None:
                    cmd += ["-to", str(inp["end"])]
                cmd += ["-i", inp["path"]]

            # Adicionar overlays como inputs adicionais
            for op in request.instructions.operations:
                if isinstance(op, Overlay):
                    cmd += ["-i", download_from_minio(op.image_source, tmpdir)]

            # Construir filtros complexos
            if request.instructions.operations:
                filter_complex = build_complex_filter(
                    request.instructions.operations,
                    len(inputs)
                )
                cmd += ["-filter_complex", filter_complex]
                
                # Mapear vídeo e áudio corretamente
                cmd += ["-map", "[v_final]"]
                if has_audio:
                    # Procura o primeiro input com áudio
                    for i, inp in enumerate(inputs):
                        if inp["has_audio"]:
                            cmd += ["-map", f"{i}:a?"]
                            break

            # Configurações de saída
            output = request.instructions.output
            cmd += [
                "-c:v", output.codec,
                "-s", output.resolution,
                "-b:v", output.bitrate
            ]
            
            # Configurar codec de áudio apenas se houver áudio
            if has_audio:
                cmd += [
                    "-c:a", "aac",
                    "-b:a", "192k"
                ]
            
            cmd += [
                "-f", output.format,
                os.path.join(tmpdir, output.filename)
            ]

            # Executar FFmpeg
            proc = await asyncio.create_subprocess_exec(*cmd, stderr=subprocess.PIPE)
            _, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

            # Upload e resposta
            with open(os.path.join(tmpdir, output.filename), "rb") as f:
                video_bytes = f.read()
            
            url = upload_to_minio(
                video_bytes,
                "videos-edited",
                f"{uuid.uuid4()}.mp4"
            )

            return {
                "status": "success",
                "url": url,
                "job_id": f"job_{uuid.uuid4()}"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_silences(filename: str, db: float = -35, min_duration: float = 1.0) -> List[float]:
    """
    Detecta períodos de silêncio no arquivo de vídeo
    Retorna uma lista onde:
    - índices pares (0,2,4,...) indicam início do silêncio
    - índices ímpares (1,3,5,...) indicam fim do silêncio
    """
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
            time = float(line.split("silence_start: ")[1].split()[0])
            time_list.append(time)
        elif "silence_end" in line:
            time = float(line.split("silence_end: ")[1].split()[0])
            time_list.append(time)
    
    return time_list

def get_video_duration(filename: str) -> float:
    """
    Obtém a duração total do vídeo em segundos
    
    Args:
        filename (str): Caminho do arquivo de vídeo
        
    Returns:
        float: Duração do vídeo em segundos
        
    Raises:
        ValueError: Se o arquivo não existir ou não for um vídeo válido
        RuntimeError: Se houver erro ao processar o arquivo
    """
    # Verifica se o arquivo existe
    if not os.path.exists(filename):
        raise ValueError(f"O arquivo não existe: {filename}")
        
    # Verifica se é um vídeo válido
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
    """Constrói o comando FFmpeg para cortar os silêncios"""
    # Cria a lista de segmentos a manter (entre os silêncios)
    segments = [0.0] + silences + [duration]
    
    # Constrói o filtro de seleção para vídeo e áudio
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

@app.post("/cut-silence")
async def cut_silence(request: SilenceCutRequest):
    """
    Endpoint para remover períodos de silêncio de um vídeo
    """
    try:
        with TemporaryDirectory() as tmpdir:
            # Download do arquivo de entrada
            input_file = download_from_url(request.input_url, tmpdir)
            
            # Validar arquivo de entrada
            if not validate_video_file(input_file):
                raise HTTPException(
                    status_code=400,
                    detail="O arquivo fornecido não é um vídeo válido"
                )
            
            if not validate_video_format(input_file):
                raise HTTPException(
                    status_code=400,
                    detail="Formato de vídeo não suportado"
                )
            
            # Detecta os silêncios
            silences = find_silences(
                input_file,
                db=request.silence_threshold,
                min_duration=request.min_silence_duration
            )
            
            if not silences:
                return {
                    "status": "success",
                    "message": "Nenhum silêncio detectado no vídeo",
                    "url": request.input_url
                }
            
            # Obtém a duração total do vídeo
            duration = get_video_duration(input_file)
            
            # Define o arquivo de saída
            output_file = os.path.join(tmpdir, f"output_no_silence_{uuid.uuid4()}.{request.output_format}")
            
            # Constrói e executa o comando FFmpeg
            cmd = build_silence_cut_command(input_file, output_file, silences, duration)
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                raise RuntimeError(f"Erro no FFmpeg: {stderr.decode()}")
            
            # Upload do resultado para o MinIO
            with open(output_file, "rb") as f:
                video_bytes = f.read()
            
            url = upload_to_minio(
                video_bytes,
                "videos-edited",
                f"no_silence_{uuid.uuid4()}.{request.output_format}"
            )
            
            return {
                "status": "success",
                "url": url,
                "job_id": f"job_{uuid.uuid4()}",
                "silences_removed": len(silences) // 2
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Adicionar após as funções auxiliares existentes e antes dos endpoints
def validate_video_file(file_path: str) -> bool:
    """
    Valida se o arquivo é um vídeo válido usando ffprobe
    Retorna True se for um vídeo válido, False caso contrário
    """
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
    """
    Valida o formato do vídeo
    Retorna True se o formato for permitido, False caso contrário
    """
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
    """
    Verifica se o arquivo tem stream de áudio
    Retorna True se tiver áudio, False caso contrário
    """
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