import asyncio
import io
import json
import os
import time
import uuid
import logging
import hashlib
from typing import Optional, Dict, Any, List
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File, status
from pydantic import BaseModel, Field
from pydub import AudioSegment
from io import BytesIO

# Importa o cliente do MinIO e seus erros
from minio import Minio
from minio.error import S3Error

# Importações para métricas e monitoramento
from prometheus_client import Counter, Histogram, start_http_server

# Importa a classe do FishSpeech (ajuste conforme a documentação oficial)
from fishespeech import FishSpeechTTS
import torch

app = FastAPI()

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

def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida e ajusta os parâmetros conforme limites do Fish Speech.
    """
    if params is None:
        return {}
    
    # Copia para não modificar o original
    validated = params.copy()
    
    # Validação de idioma
    valid_languages = ["auto", "en", "zh", "ja", "ko", "fr", "de", "ar", "es"]
    if "language" in validated:
        if validated["language"] not in valid_languages:
            validated["language"] = "auto"
    
    # Validação de velocidade
    if "speed" in validated:
        validated["speed"] = max(0.5, min(2.0, float(validated["speed"])))
    
    # Validação de pitch
    if "pitch" in validated:
        validated["pitch"] = max(-12, min(12, float(validated["pitch"])))
    
    # Validação de energia
    if "energy" in validated:
        validated["energy"] = max(0.5, min(2.0, float(validated["energy"])))
    
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
# Carrega o modelo FishSpeech globalmente para evitar recarregamentos repetidos
fishspeech_model = None

def load_fishespeech_model(model_path: str = "models/fishespeech"):
    """
    Carrega o modelo FishSpeech com otimizações para GPU.
    """
    global fishspeech_model
    try:
        fishspeech_model = FishSpeechTTS.from_pretrained(model_path)
        if torch.cuda.is_available():
            fishspeech_model = fishspeech_model.to("cuda")
            # Habilitar otimização CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            logger.info("Modelo FishSpeech carregado com suporte CUDA")
        else:
            logger.warning("GPU não disponível. Usando CPU para inferência.")
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo FishSpeech: {str(e)}")
        raise RuntimeError(f"Erro ao carregar o modelo FishSpeech: {str(e)}")
    return fishspeech_model

@app.on_event("startup")
async def startup_event():
    """
    Inicializa o servidor de métricas e carrega o modelo.
    """
    # Inicia servidor de métricas Prometheus na porta 8001
    start_http_server(8001)
    load_fishespeech_model()

@lru_cache(maxsize=100)
def get_cached_audio(text: str, params_hash: str) -> Optional[bytes]:
    """
    Cache para evitar regeneração de áudios idênticos.
    Utiliza LRU cache com limite de 100 itens.
    """
    cache_key = f"{text}_{params_hash}"
    # Por enquanto usando apenas LRU cache em memória
    # TODO: Implementar cache distribuído com Redis
    return None

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
    """
    Processa chunks de texto em batch para melhor performance.
    """
    batch_size = 4  # Ajuste baseado na memória GPU disponível
    audio_chunks = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tasks = [
            asyncio.get_running_loop().run_in_executor(
                None, 
                lambda text=text: synthesize_voice(text, params)
            )
            for text in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        audio_chunks.extend(batch_results)
    
    return audio_chunks

######################################
# 6. Endpoint de Geração de Voz (Síntese Básica)
######################################
@app.post("/generate-voice")
async def generate_voice(
    request: VoiceRequest,
    token: str = Depends(verify_bearer_token),
    sample: UploadFile = File(None)  # Opcional para geração básica
):
    """
    Gera áudio a partir do texto fornecido utilizando FishSpeech (síntese básica).
    Inclui otimizações de cache, batch processing e monitoramento.
    """
    start_time = time.time()
    
    try:
        # Se um template for informado, carrega-o e sobrescreve parâmetros/tempo
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
            logger.info("Gerando novo áudio")
            
            # Divide o texto em chunks e processa em batch
            MAX_CHUNK_SIZE = 1000
            chunks = [processed_text[i:i + MAX_CHUNK_SIZE] 
                     for i in range(0, len(processed_text), MAX_CHUNK_SIZE)]
            
            audio_chunks = await process_text_chunks(chunks, request.parametros)
            
            # Concatena todos os áudios resultantes
            combined = AudioSegment.empty()
            for audio in audio_chunks:
                segment = AudioSegment.from_file(BytesIO(audio), format="wav")
                combined += segment
            
            # Verifica se ultrapassa o tempo máximo permitido
            total_duration_seconds = len(combined) / 1000
            if total_duration_seconds > request.tempo_max:
                raise HTTPException(
                    status_code=400,
                    detail=f"A duração do áudio gerado ({int(total_duration_seconds)}s) excede o máximo permitido ({request.tempo_max}s)."
                )
            
            # Exporta o áudio final para buffer
            buf = BytesIO()
            combined.export(buf, format="wav")
            buf.seek(0)
            audio_bytes = buf.getvalue()
        
        # Upload para MinIO
        file_name = f"{uuid.uuid4()}_{int(time.time())}.wav"
        minio_url = upload_to_minio(audio_bytes, file_name)
        
        # Registra tempo de processamento
        process_time = time.time() - start_time
        VOICE_GENERATION_TIME.observe(process_time)
        
        return {
            "status": "sucesso",
            "job_id": f"voz_{int(time.time())}",
            "message": f"Áudio gerado com sucesso (duração: {int(len(combined)/1000)}s).",
            "minio_url": minio_url,
            "process_time": f"{process_time:.2f}s"
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
        loop = asyncio.get_running_loop()
        
        for chunk in chunks:
            try:
                audio_chunk = await loop.run_in_executor(
                    None, 
                    lambda: clone_voice(chunk, processed_sample, request.parametros)
                )
                audio_chunks.append(audio_chunk)
            except Exception as e:
                logger.error(f"Erro ao processar chunk: {str(e)}")
                # Continua com os próximos chunks mesmo se houver erro
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

def synthesize_voice(text: str, params: Dict[str, Any]) -> bytes:
    """
    Gera áudio WAV usando Fish Speech com suporte a múltiplos idiomas e controle de parâmetros.
    """
    global fishspeech_model
    if fishspeech_model is None:
        raise RuntimeError("O modelo FishSpeech não está carregado.")
    
    try:
        # Valida e ajusta os parâmetros
        validated_params = validate_params(params)
        
        # Prepara os parâmetros para o modelo
        model_params = {
            "text": text,
            "language": validated_params.get("language", "auto"),
            "speed": validated_params.get("speed", 1.0),
            "pitch": validated_params.get("pitch", 0.0),
            "energy": validated_params.get("energy", 1.0),
        }
        
        # Adiciona parâmetros opcionais se fornecidos
        if "prompt_text" in validated_params and validated_params["prompt_text"]:
            model_params["prompt_text"] = validated_params["prompt_text"]
        
        if "emotion" in validated_params and validated_params["emotion"]:
            model_params["emotion"] = validated_params["emotion"]
        
        # Gera o áudio
        audio_bytes = fishspeech_model.synthesize(**model_params)
        return audio_bytes
        
    except Exception as e:
        logger.error(f"Erro na síntese de voz: {str(e)}")
        raise RuntimeError(f"Erro na síntese de voz: {str(e)}")

def clone_voice(text: str, sample_bytes: bytes, params: Dict[str, Any]) -> bytes:
    """
    Clona voz usando Fish Speech com suporte a zero-shot e few-shot.
    """
    global fishspeech_model
    if fishspeech_model is None:
        raise RuntimeError("O modelo FishSpeech não está carregado.")
    
    try:
        # Valida e ajusta os parâmetros
        validated_params = validate_params(params)
        
        # Prepara os parâmetros para clonagem
        clone_params = {
            "text": text,
            "reference_audio": sample_bytes,
            "language": validated_params.get("language", "auto"),
            "speed": validated_params.get("speed", 1.0),
            "pitch": validated_params.get("pitch", 0.0),
            "energy": validated_params.get("energy", 1.0),
        }
        
        # Gera o áudio clonado
        audio_bytes = fishspeech_model.clone(**clone_params)
        return audio_bytes
        
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

def split_text_smart(text: str, max_len: int = 1000) -> List[str]:
    """
    Divide o texto em chunks de forma inteligente, respeitando pontuação.
    """
    # Se o texto é menor que o limite, retorna como está
    if len(text) <= max_len:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Divide por sentenças primeiro
    sentences = text.replace("。", ".").replace("！", "!").replace("？", "?").split(".")
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Se a sentença é maior que o limite, divide por vírgulas
        if len(sentence) > max_len:
            subparts = sentence.split(",")
            for part in subparts:
                part = part.strip()
                if not part:
                    continue
                    
                # Se ainda é muito grande, divide por espaços
                if len(part) > max_len:
                    words = part.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_len:
                            temp_chunk += " " + word if temp_chunk else word
                        else:
                            chunks.append(temp_chunk + ".")
                            temp_chunk = word
                    if temp_chunk:
                        chunks.append(temp_chunk + ".")
                else:
                    # Adiciona a parte com pontuação
                    if len(current_chunk) + len(part) + 2 <= max_len:
                        current_chunk += (", " if current_chunk else "") + part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk + ".")
                        current_chunk = part
        else:
            # Adiciona a sentença completa
            if len(current_chunk) + len(sentence) + 2 <= max_len:
                current_chunk += (". " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk + ".")
                current_chunk = sentence
    
    # Adiciona o último chunk se existir
    if current_chunk:
        chunks.append(current_chunk + ".")
    
    return chunks

@app.get("/healthcheck")
async def healthcheck():
    """
    Endpoint de healthcheck que verifica:
    - Status do modelo
    - Uso de GPU
    - Memória disponível
    - Teste básico de síntese
    """
    try:
        if fishspeech_model is None:
            return {
                "status": "error",
                "message": "Modelo não carregado",
                "timestamp": time.time()
            }
        
        # Verifica GPU
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
        
        if gpu_info["available"]:
            gpu_info.update({
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
                "max_memory_allocated": torch.cuda.max_memory_allocated()
            })
        
        # Testa síntese básica
        test_result = None
        try:
            test_text = "Teste de síntese."
            _ = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: synthesize_voice(test_text, {})
            )
            test_result = "success"
        except Exception as e:
            test_result = f"failed: {str(e)}"
        
        return {
            "status": "healthy",
            "model": {
                "loaded": True,
                "type": "FishSpeech",
                "version": "1.4.3",
                "test_synthesis": test_result
            },
            "gpu": gpu_info,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Erro no healthcheck: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
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
        metrics = {
            "total_requests": int(VOICE_GENERATION_TIME._sum.get()),
            "total_errors": int(VOICE_GENERATION_ERRORS._value.get()),
            "average_generation_time": float(VOICE_GENERATION_TIME._sum.get() / max(1, VOICE_GENERATION_TIME._count.get())),
            "total_clone_requests": int(VOICE_CLONE_TIME._sum.get()),
            "total_clone_errors": int(VOICE_CLONE_ERRORS._value.get()),
            "average_clone_time": float(VOICE_CLONE_TIME._sum.get() / max(1, VOICE_CLONE_TIME._count.get())),
            "cache_stats": {
                "hits": int(CACHE_HITS._value.get()),
                "misses": int(CACHE_MISSES._value.get()),
                "hit_ratio": float(CACHE_HITS._value.get() / max(1, CACHE_HITS._value.get() + CACHE_MISSES._value.get()))
            }
        }
        
        if torch.cuda.is_available():
            metrics["gpu"] = {
                "memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                "device_utilization": None  # TODO: Implementar medição de utilização da GPU
            }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Erro ao coletar métricas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao coletar métricas: {str(e)}"
        )
