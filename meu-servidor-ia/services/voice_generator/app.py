import asyncio
import io
import json
import os
import time
import uuid
import logging
import hashlib
from typing import Optional, Dict, Any, List
from functools import lru_cache, partial
import regex as re  # Usar regex Unicode
import queue

from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File, status, Request
from pydantic import BaseModel, Field
from pydub import AudioSegment
from io import BytesIO
from fastapi.middleware import Middleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Importa o cliente do MinIO e seus erros
from minio import Minio
from minio.error import S3Error

# Importações para métricas e monitoramento
from prometheus_client import Counter, Histogram, start_http_server, generate_latest, REGISTRY

# Importa as classes do Fish Speech
# from fish_speech.inference_engine import TTSInferenceEngine
# from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
import torch

# Adicionar imports
import pynvml
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures

# Adicionar contexto de GPU
import contextlib

# Importar o novo módulo de utilidades GPU
from shared.gpu_utils import cuda_memory_manager, optimize_gpu_settings

# Importações atualizadas
# from TTS.api import TTS
# from TTS.utils.synthesizer import Synthesizer
# from TTS.tts.configs.shared_configs import TTSConfig
# from TTS.tts.models import setup_model
# from TTS.utils.audio import AudioProcessor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch.cuda.amp as amp
import onnxruntime as ort
from rich.progress import Progress, SpinnerColumn, TextColumn
from config import (
    FISH_SPEECH_CONFIG,
    RATE_LIMIT,
    REDIS_CONFIG,
    MINIO_CONFIG,
    GENERATION_LIMITS,
    GPU_CONFIG,
    LOGGING_CONFIG,
    MONITORING_CONFIG,
    OPTIMIZATION_CONFIG,
    CACHE_CONFIG,
    SECURITY_CONFIG
)
from fish_speech_wrapper import FishSpeechWrapper

# Adicionar imports necessários
import pkg_resources
import importlib.metadata

# Configurar limiter após criar a app
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Fish Speech API", version="1.4.3")
app.state.limiter = limiter

# Configurar CORS
if SECURITY_CONFIG["enable_cors"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SECURITY_CONFIG["allowed_origins"],
        allow_methods=SECURITY_CONFIG["allowed_methods"],
        allow_headers=SECURITY_CONFIG["allowed_headers"],
        allow_credentials=SECURITY_CONFIG["allow_credentials"],
        max_age=SECURITY_CONFIG["max_age"]
    )

# Configurar rate limiting
if SECURITY_CONFIG["rate_limit_enabled"]:
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{RATE_LIMIT['requests_per_minute']}/minute"]
    )
    app.state.limiter = limiter

# Configurar cache Redis
if CACHE_CONFIG["enable_cache"]:
    import redis
    cache = redis.from_url(
        REDIS_CONFIG["url"],
        encoding="utf-8",
        decode_responses=True
    )

# Configurar MinIO
minio_client = Minio(
    MINIO_CONFIG["endpoint"],
    access_key=MINIO_CONFIG["access_key"],
    secret_key=MINIO_CONFIG["secret_key"],
    secure=MINIO_CONFIG["secure"]
)

# Configurar modelo Fish Speech
fishspeech_model = None
onnx_session = None

# Inicializar BatchProcessor
batch_processor = None

def setup_gpu():
    """Configura ambiente GPU com otimizações."""
    if torch.cuda.is_available():
        # Configurar dispositivos visíveis
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CONFIG["visible_devices"]
        
        # Otimizações CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        if GPU_CONFIG["memory_growth"]:
            for device in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(
                    GPU_CONFIG["per_process_memory_fraction"],
                    device
                )
        
        logger.info(f"GPU disponível: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU não disponível, usando CPU")

def load_fishspeech_model():
    """Carrega e otimiza o modelo Fish Speech."""
    global fishspeech_model
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Carregando modelo...", total=None)
            
            # Carregar modelo usando o wrapper
            model_path = FISH_SPEECH_CONFIG["model_path"]
            fishspeech_model = FishSpeechWrapper(
                model_path=model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_fp16=FISH_SPEECH_CONFIG["use_fp16"]
            )
            
            progress.update(task, completed=True)
            
        logger.info("Modelo Fish Speech carregado e otimizado com sucesso")
        return fishspeech_model
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo Fish Speech: {str(e)}")
        raise RuntimeError(f"Falha ao carregar modelo Fish Speech: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Inicializa recursos na inicialização da aplicação."""
    global fishspeech_model, batch_processor
    
    try:
        # Otimizar configurações de GPU
        optimize_gpu_settings(
            device_id=0,
            memory_fraction=GPU_CONFIG["per_process_memory_fraction"],
            benchmark=True
        )
        
        # Inicializar modelo
        fishspeech_model = FishSpeechWrapper(
            model_path=FISH_SPEECH_CONFIG["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_fp16=FISH_SPEECH_CONFIG["use_fp16"]
        )
        
        # Inicializar BatchProcessor
        from shared.cache_manager import VoiceCache
        cache = VoiceCache(
            embedding_cache_size=FISH_SPEECH_CONFIG["embedding_cache_size"],
            result_cache_size=CACHE_CONFIG["max_size"],
            embedding_ttl=CACHE_CONFIG["embedding_cache"]["ttl"],
            result_ttl=CACHE_CONFIG["ttl"]
        )
        
        batch_processor = BatchProcessor(
            model=fishspeech_model,
            cache_client=cache
        )
        
        logger.info(
            "Serviço inicializado com sucesso "
            f"(GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
        )
        
    except Exception as e:
        logger.error(f"Erro na inicialização: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Limpa recursos ao encerrar a aplicação."""
    global fishspeech_model, batch_processor
    
    if batch_processor:
        batch_processor = None
    
    if fishspeech_model:
        del fishspeech_model
        fishspeech_model = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

######################################
# 1. Configuração de Métricas
######################################
# Métricas Prometheus
VOICE_GENERATION_TIME = Histogram(
    'voice_generation_seconds', 
    'Tempo de geração de voz',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
VOICE_GENERATION_ERRORS = Counter(
    'voice_generation_errors_total',
    'Total de erros na geração de voz'
)
CACHE_HITS = Counter(
    'voice_cache_hits_total',
    'Total de hits no cache de áudio'
)
CACHE_MISSES = Counter(
    'voice_cache_misses_total',
    'Total de misses no cache de áudio'
)

# Adiciona métricas específicas para Fish Speech
VOICE_CLONE_TIME = Histogram(
    'voice_clone_seconds', 
    'Tempo de clonagem de voz',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

VOICE_CLONE_ERRORS = Counter(
    'voice_clone_errors_total',
    'Total de erros na clonagem de voz'
)

MODEL_LOAD_TIME = Histogram(
    'model_load_seconds',
    'Tempo de carregamento do modelo',
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0]
)

######################################
# 2. Configuração de Logging
######################################
logging.config.dictConfig(LOGGING_CONFIG)
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
class VoiceRequest(BaseModel):
    texto: str
    template_id: Optional[str] = None
    tempo_max: Optional[int] = 1200  # Duração máxima em segundos
    parametros: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "language": "auto",  # auto, en, zh, ja, ko, fr, de, ar, es
            "speed": 1.0,       # 0.5 a 2.0
            "pitch": 0.0,       # -12 a 12
            "energy": 1.0,      # 0.5 a 2.0
            "prompt_text": "",  # Texto de referência para estilo (opcional)
            "emotion": None,    # Emoção desejada (opcional)
        }
    )
    use_voice_clone: bool = False  # Se True, usa clonagem de voz com sample

# Atualizar a validação de parâmetros
VALID_EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "excited", "calm"]

def validate_params(params: dict) -> dict:
    validated = params.copy()
    
    # Validação de emoção
    if "emotion" in validated:
        if validated["emotion"].lower() not in VALID_EMOTIONS:
            validated["emotion"] = "neutral"
    
    # Validação de faixas numéricas
    numerical_params = ["speed", "pitch", "energy"]
    for param in numerical_params:
        if param in validated:
            validated[param] = max(0.5, min(2.0, float(validated[param])))
    
    return validated

def load_template(template_id: str) -> dict:
    """
    Carrega um template (arquivo JSON) do diretório 'templates/'.
    Exemplo de template:
    {
      "template_id": "voice_template1",
      "tempo_max": 900,
      "parametros": {"pitch": 1.2, "speed": 0.9}
    }
    """
    template_path = os.path.join("templates", f"{template_id}.json")
    if not os.path.exists(template_path):
        raise ValueError("Template não encontrado.")
    with open(template_path, "r") as f:
        return json.load(f)

######################################
# 3. Integração com FishSpeech
######################################
@lru_cache(maxsize=500)
def get_cached_audio(text: str, params_hash: str) -> Optional[bytes]:
    current_hash = hashlib.sha256(
        f"{text}_{params_hash}".encode()
    ).hexdigest()
    
    # Verificação adicional de consistência
    cached = cache.get(current_hash)
    if cached and validate_audio_integrity(cached):
        return cached
    return None

def validate_audio_integrity(audio_data: bytes) -> bool:
    # Verificação básica do header do arquivo de áudio
    return audio_data[:4] in (b'RIFF', b'OggS', b'fLaC')

def get_params_hash(params: Dict[str, Any]) -> str:
    """
    Gera hash dos parâmetros para uso como chave de cache.
    """
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

######################################
# 4. Pré-processamento de Texto
######################################
def preprocess_text(text: str) -> str:
    """
    Pré-processa o texto para melhor qualidade de síntese.
    """
    # Remove espaços extras
    text = ' '.join(text.split())
    
    # Adiciona pontuação se necessário
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    # TODO: Adicionar mais regras de normalização conforme necessário
    # - Expandir números
    # - Normalizar abreviações
    # - Tratar caracteres especiais
    
    return text

######################################
# 5. Processamento em Batch
######################################
async def process_text_chunks(chunks: List[str], params: Dict[str, Any]) -> List[bytes]:
    """Processa chunks de texto em batch para melhor performance usando ThreadPool"""
    with app.state.executor as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor,
                partial(synthesize_voice, text=text, params=params)
            )
            for text in chunks
        ]
        
        audio_chunks = []
        for future in as_completed(futures):
            try:
                result = await future
                audio_chunks.append(result)
            except Exception as e:
                logger.error(f"Erro no processamento de chunk: {str(e)}")
                continue
    
    return audio_chunks

######################################
# 6. Endpoint de Geração de Voz (Síntese Básica)
######################################
@app.post("/generate-voice")
async def generate_voice(
    request: VoiceRequest,
    token: str = Depends(verify_bearer_token),
    sample: UploadFile = File(None)
):
    """
    Gera áudio a partir do texto fornecido utilizando FishSpeech com batch processing.
    """
    start_time = time.time()
    
    try:
        # Validar template e parâmetros
        if request.template_id:
            try:
                template = load_template(request.template_id)
                request.tempo_max = template.get("tempo_max", request.tempo_max)
                request.parametros = template.get("parametros", request.parametros)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Erro ao carregar template: {str(e)}")
        
        # Pré-processa o texto
        processed_text = preprocess_text(request.texto)
        
        # Verifica cache
        params_hash = get_params_hash(request.parametros)
        cached_audio = get_cached_audio(processed_text, params_hash)
        
        if cached_audio:
            CACHE_HITS.inc()
            logger.info("Áudio encontrado no cache")
            audio_bytes = cached_audio
        else:
            CACHE_MISSES.inc()
            logger.info("Gerando novo áudio com batch processing")
            
            # Divide o texto em chunks
            chunks = split_text_smart(processed_text)
            
            # Processa chunks em batch
            audio_futures = []
            for chunk in chunks:
                future = await batch_processor.add_to_batch(
                    text=chunk,
                    language=request.parametros.get("language", "auto"),
                    params=request.parametros
                )
                audio_futures.append(future)
            
            # Aguarda todos os chunks
            audio_chunks = []
            for future in audio_futures:
                try:
                    wav = await future
                    audio_chunks.append(wav)
                except Exception as e:
                    logger.error(f"Erro no processamento de chunk: {e}")
                    continue
            
            if not audio_chunks:
                raise HTTPException(
                    status_code=500,
                    detail="Falha ao gerar áudio. Todos os chunks falharam."
                )
            
            # Concatena os áudios
            combined = AudioSegment.empty()
            for wav in audio_chunks:
                segment = AudioSegment(
                    wav.cpu().numpy().tobytes(),
                    frame_rate=FISH_SPEECH_CONFIG["sample_rate"],
                    sample_width=2,
                    channels=1
                )
                combined += segment
            
            # Verifica duração
            total_duration_seconds = len(combined) / 1000
            if total_duration_seconds > request.tempo_max:
                raise HTTPException(
                    status_code=400,
                    detail=f"Duração do áudio ({int(total_duration_seconds)}s) excede o máximo permitido ({request.tempo_max}s)."
                )
            
            # Exporta áudio final
            buf = BytesIO()
            combined.export(buf, format="wav")
            buf.seek(0)
            audio_bytes = buf.getvalue()
            
            # Cache do resultado
            if CACHE_CONFIG["enable_cache"]:
                cache_key = f"{processed_text}_{params_hash}"
                batch_processor.cache.cache_result(cache_key, audio_bytes)
        
        # Upload para MinIO
        file_name = f"{uuid.uuid4()}_{int(time.time())}.wav"
        minio_url = upload_to_minio(audio_bytes, file_name)
        
        # Registra métricas
        process_time = time.time() - start_time
        VOICE_GENERATION_TIME.observe(process_time)
        
        return {
            "status": "sucesso",
            "job_id": f"voz_{int(time.time())}",
            "message": f"Áudio gerado com sucesso (duração: {int(len(combined)/1000)}s).",
            "minio_url": minio_url,
            "process_time": f"{process_time:.2f}s",
            "chunks_processados": len(audio_chunks) if 'audio_chunks' in locals() else 1
        }
        
    except Exception as e:
        VOICE_GENERATION_ERRORS.inc()
        logger.error(f"Erro na geração de voz: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na geração de voz: {str(e)}"
        )

######################################
# 7. Endpoint de Clonagem de Voz
######################################
@app.post("/clone-voice")
async def clone_voice_endpoint(
    request: VoiceRequest,
    token: str = Depends(verify_bearer_token),
    sample: UploadFile = File(...)
):
    """
    Clona a voz a partir do texto fornecido e de um sample de áudio.
    - O sample (áudio de referência) é obrigatório para clonagem.
    - Parâmetros adicionais podem ser passados via template ou na requisição.
    - O texto é dividido em chunks e cada chunk é sintetizado via clonagem.
    - Os áudios gerados são concatenados e enviados para o MinIO.
    """
    start_time = time.time()
    
    try:
        # Carrega template se fornecido
        if request.template_id:
            try:
                template = load_template(request.template_id)
                request.tempo_max = template.get("tempo_max", request.tempo_max)
                request.parametros = template.get("parametros", request.parametros)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Erro ao carregar template: {str(e)}")
        
        # Valida e processa o áudio de referência
        if sample is None:
            raise HTTPException(status_code=400, detail="O arquivo de sample é obrigatório para clonagem de voz.")
        
        try:
            sample_bytes = await sample.read()
            processed_sample = validate_audio_sample(sample_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Pré-processa o texto
        processed_text = preprocess_text(request.texto)
        
        # Divide o texto em chunks de forma inteligente
        chunks = split_text_smart(processed_text)
        
        # Processa os chunks em batch
        audio_chunks = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:  # Número ótimo para I/O bound
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(
                    executor,
                    partial(clone_voice, chunk, processed_sample, request.parametros)
                )
                for chunk in chunks
            ]
            
            for future in as_completed(futures):
                try:
                    audio_chunk = await future
                    audio_chunks.append(audio_chunk)
                except Exception as e:
                    logger.error(f"Erro no chunk: {str(e)}")
                    continue
        
        if not audio_chunks:
            raise HTTPException(
                status_code=500,
                detail="Não foi possível gerar nenhum áudio. Todos os chunks falharam."
            )
        
        # Concatena os áudios gerados
        combined = AudioSegment.empty()
        for audio in audio_chunks:
            segment = AudioSegment.from_file(BytesIO(audio), format="wav")
            combined += segment
        
        # Verifica duração máxima
        total_duration_seconds = len(combined) / 1000
        if total_duration_seconds > request.tempo_max:
            raise HTTPException(
                status_code=400,
                detail=f"A duração do áudio gerado ({int(total_duration_seconds)}s) excede o máximo permitido ({request.tempo_max}s)."
            )
        
        # Exporta o áudio final
        buf = BytesIO()
        combined.export(buf, format="wav")
        buf.seek(0)
        file_bytes = buf.getvalue()
        
        # Upload para MinIO
        file_name = f"clone_{uuid.uuid4()}_{int(time.time())}.wav"
        minio_url = upload_to_minio(file_bytes, file_name)
        
        # Registra tempo de processamento
        process_time = time.time() - start_time
        VOICE_CLONE_TIME.observe(process_time)
        
        return {
            "status": "sucesso",
            "job_id": f"clone_{int(time.time())}",
            "message": f"Voz clonada com sucesso (duração: {int(total_duration_seconds)}s).",
            "minio_url": minio_url,
            "process_time": f"{process_time:.2f}s",
            "chunks_processados": len(audio_chunks),
            "chunks_total": len(chunks)
        }
        
    except Exception as e:
        VOICE_CLONE_ERRORS.inc()
        logger.error(f"Erro na clonagem de voz: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na clonagem de voz: {str(e)}"
        )

# Adicionar contexto de GPU
@contextlib.contextmanager
def gpu_context(device_id: int = 0):
    """Contexto otimizado para operações GPU."""
    if not torch.cuda.is_available():
        yield
        return
        
    with torch.cuda.device(device_id), \
         torch.cuda.stream(torch.cuda.Stream()), \
         torch.inference_mode(), \
         amp.autocast(enabled=OPTIMIZATION_CONFIG["use_mixed_precision"]):
        try:
            if OPTIMIZATION_CONFIG["pin_memory"]:
                torch.cuda.empty_cache()
            yield
        finally:
            if OPTIMIZATION_CONFIG["pin_memory"]:
                torch.cuda.empty_cache()

def synthesize_voice(text: str, params: Dict[str, Any]) -> bytes:
    """Gera áudio usando TTS com otimizações."""
    global fishspeech_model
    
    if fishspeech_model is None:
        raise RuntimeError("Modelo não está carregado")
    
    try:
        # Validar parâmetros
        validated_params = validate_params(params)
        
        # Preparar parâmetros
        model_params = {
            "text": text,
            "language": validated_params.get("language", "auto"),
            "speaker": validated_params.get("speaker", None),
            "speed": validated_params.get("speed", 1.0),
        }
        
        # Gerar áudio
        with gpu_context():
            wav = fishspeech_model.tts(**model_params)
        
        # Converter para bytes
        buf = BytesIO()
        AudioSegment(
            wav.cpu().numpy().tobytes(), 
            frame_rate=22050,
            sample_width=2, 
            channels=1
        ).export(buf, format="wav")
        
        return buf.getvalue()
        
    except Exception as e:
        logger.error(f"Erro na síntese de voz: {str(e)}")
        raise RuntimeError(f"Erro na síntese de voz: {str(e)}")

async def process_chunk(chunk: str, processed_sample: AudioSegment, params: dict):
    """Executa clonagem de voz com gerenciamento de contexto"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            partial(clone_voice, chunk, processed_sample, params)
        )

def clone_voice(text: str, sample_bytes: bytes, params: Dict[str, Any]) -> bytes:
    """Clona voz usando TTS com suporte a voice cloning."""
    global fishspeech_model
    
    if fishspeech_model is None:
        raise RuntimeError("Modelo não está carregado")
    
    try:
        # Validar parâmetros
        validated_params = validate_params(params)
        
        # Gerar áudio com clonagem
        with gpu_context():
            wav = fishspeech_model.tts_with_vc(
                text=text,
                speaker_wav=sample_bytes,
                language=validated_params.get("language", "auto")
            )
        
        # Converter para bytes
        buf = BytesIO()
        AudioSegment(
            wav.cpu().numpy().tobytes(), 
            frame_rate=22050,
            sample_width=2, 
            channels=1
        ).export(buf, format="wav")
        
        return buf.getvalue()
        
    except Exception as e:
        logger.error(f"Erro na clonagem de voz: {str(e)}")
        raise RuntimeError(f"Erro na clonagem de voz: {str(e)}")

######################################
# Funções de Processamento de Áudio
######################################
def validate_audio_sample(audio_bytes: bytes, max_duration: int = 30) -> bytes:
    """
    Valida e pré-processa o áudio de referência para clonagem.
    - Verifica duração (máximo 30 segundos por padrão)
    - Normaliza volume
    - Converte para formato adequado (wav, 16kHz, mono)
    """
    try:
        # Verificação de assinatura
        valid_headers = (b'RIFF', b'ID3', b'OggS', b'fLaC')
        if not audio_bytes.startswith(valid_headers):
            raise ValueError("Formato de áudio não suportado")
        
        # Carrega o áudio
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        
        # Verifica duração
        duration_seconds = len(audio) / 1000
        if duration_seconds > max_duration:
            raise ValueError(f"Áudio muito longo ({duration_seconds:.1f}s). Máximo permitido: {max_duration}s")
        
        # Converte para mono se necessário
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Converte sample rate para 16kHz
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        # Normaliza volume
        normalized = audio.normalize()
        
        # Exporta para WAV
        buf = BytesIO()
        normalized.export(buf, format="wav")
        return buf.getvalue()
        
    except Exception as e:
        raise ValueError(f"Erro ao processar áudio de referência: {str(e)}")

def split_text_smart(text: str, max_length: int = 100) -> list:
    # Padrão Unicode para pontuação de sentenças
    sentence_boundary = re.compile(
        r'(\p{Sentence_Terminal}+[\s\u200b]*)',
        re.UNICODE
    )
    
    sentences = []
    current = []
    length = 0
    
    for part in sentence_boundary.split(text):
        part = part.strip()
        if not part:
            continue
            
        if length + len(part) <= max_length:
            current.append(part)
            length += len(part)
        else:
            if current:
                sentences.append(' '.join(current))
            current = [part]
            length = len(part)
    
    if current:
        sentences.append(' '.join(current))
    
    return sentences

def get_package_version(package_name: str) -> str:
    """
    Obtém a versão instalada de um pacote Python.
    Retorna 'unknown' se o pacote não for encontrado.
    """
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde da aplicação que retorna versões dinâmicas."""
    try:
        # Obtém versões dos pacotes dinamicamente
        dependencies = {
            "torch": get_package_version("torch"),
            "transformers": get_package_version("transformers"),
            "fastapi": get_package_version("fastapi"),
            "pydantic": get_package_version("pydantic"),
            "fish_speech": get_package_version("fish_speech")
        }

        # Obtém versão da aplicação do arquivo de configuração ou variável de ambiente
        app_version = os.getenv("APP_VERSION", pkg_resources.get_distribution("voice_generator").version)

        return {
            "status": "healthy",
            "version": app_version,
            "dependencies": dependencies,
            "gpu_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
    except Exception as e:
        logger.error(f"Erro ao verificar saúde da aplicação: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.get("/metrics/voice")
async def voice_metrics():
    """
    Endpoint que retorna métricas específicas do serviço de voz.
    """
    if not fishspeech_model:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está carregado"
        )
    
    try:
        metrics_data = {
            # Coletar via registro oficial
            "total_requests": get_metric_value('voice_generation_seconds_count'),
            "average_time": get_metric_value('voice_generation_seconds_sum') / get_metric_value('voice_generation_seconds_count'),
            # ... outras métricas
        }
        
        if torch.cuda.is_available():
            metrics_data["gpu"] = {
                "memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                "device_utilization": None  # TODO: Implementar medição de utilização da GPU
            }
        
        return metrics_data
        
    except Exception as e:
        logger.error(f"Erro ao coletar métricas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao coletar métricas: {str(e)}"
        )

def get_gpu_metrics() -> dict:
    pynvml.nvmlInit()
    metrics = {}
    
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            metrics[f"gpu_{i}"] = {
                "utilization": {
                    "gpu": util.gpu,
                    "memory": util.memory
                },
                "memory": {
                    "used": memory.used,
                    "total": memory.total
                },
                "temperature": pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            }
    except Exception as e:
        logging.error(f"Erro na coleta de métricas da GPU: {str(e)}")
    
    pynvml.nvmlShutdown()
    return metrics

def check_minio_connection() -> bool:
    try:
        return minio_client.bucket_exists(AUDIO_BUCKET_NAME)
    except S3Error:
        return False

def check_cache_connection() -> bool:
    try:
        return cache.ping()
    except Exception:
        return False
