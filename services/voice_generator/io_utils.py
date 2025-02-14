"""
Utilitários para manipulação de áudio e I/O.
"""
import io
import logging
import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional, Dict
from pydub import AudioSegment
from exceptions import AudioProcessingError
from schemas import AudioFormat

logger = logging.getLogger(__name__)

SUPPORTED_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000]
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
AUDIO_INFO_KEYS = ["channels", "sample_rate", "duration", "format", "bit_depth"]

def validate_sample_rate(sr: int) -> None:
    """Valida sample rate."""
    if sr not in SUPPORTED_SAMPLE_RATES:
        raise AudioProcessingError(
            f"Sample rate {sr} não suportado. "
            f"Valores permitidos: {SUPPORTED_SAMPLE_RATES}"
        )

def validate_audio_size(size: int) -> None:
    """Valida tamanho do áudio."""
    if size > MAX_AUDIO_SIZE:
        raise AudioProcessingError(
            f"Arquivo muito grande ({size/1024/1024:.1f}MB). "
            f"Máximo permitido: {MAX_AUDIO_SIZE/1024/1024:.1f}MB"
        )

def get_audio_info(audio: AudioSegment) -> Dict[str, Any]:
    """Extrai informações do áudio."""
    return {
        "channels": audio.channels,
        "sample_rate": audio.frame_rate,
        "duration": len(audio) / 1000.0,
        "format": audio.channels == 1 and "mono" or "stereo",
        "bit_depth": audio.sample_width * 8
    }

def load_audio(
    audio_bytes: bytes,
    target_sr: int = 16000,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Carrega e pré-processa áudio a partir de bytes.
    
    Args:
        audio_bytes: Áudio em bytes
        target_sr: Sample rate alvo
        normalize: Se deve normalizar o áudio
        
    Returns:
        Tuple[torch.Tensor, int]: Tensor de áudio e sample rate
        
    Raises:
        AudioProcessingError: Se houver erro no processamento
    """
    try:
        # Valida tamanho
        validate_audio_size(len(audio_bytes))
        
        # Valida sample rate
        validate_sample_rate(target_sr)
        
        # Carrega áudio
        wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
        
        # Valida formato
        if wav.dim() > 2:
            raise AudioProcessingError("Formato de áudio não suportado")
        
        # Converte para mono se necessário
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample se necessário
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
        
        # Normaliza
        if normalize:
            max_val = wav.abs().max()
            if max_val > 0:
                wav = wav / max_val
        
        # Valida resultado
        if torch.isnan(wav).any():
            raise AudioProcessingError("NaN detectado no áudio processado")
        
        return wav, target_sr
        
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        logger.error(f"Erro ao carregar áudio: {e}")
        raise AudioProcessingError(f"Erro ao carregar áudio: {e}")

def save_audio(
    wav: torch.Tensor,
    sample_rate: int,
    format: AudioFormat = AudioFormat.WAV,
    bit_depth: int = 16
) -> bytes:
    """
    Converte tensor de áudio para bytes.
    
    Args:
        wav: Tensor de áudio
        sample_rate: Sample rate
        format: Formato de saída
        bit_depth: Profundidade de bits
        
    Returns:
        bytes: Áudio em bytes
        
    Raises:
        AudioProcessingError: Se houver erro na conversão
    """
    try:
        # Valida entrada
        if torch.isnan(wav).any():
            raise AudioProcessingError("NaN detectado no tensor de entrada")
            
        validate_sample_rate(sample_rate)
        
        if bit_depth not in [8, 16, 24, 32]:
            raise AudioProcessingError(f"Bit depth {bit_depth} não suportado")
        
        # Converte para numpy
        audio_np = wav.cpu().numpy()
        
        # Normaliza se necessário
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val
        
        # Converte para o range correto
        if bit_depth == 8:
            audio_np = (audio_np * 127).astype(np.int8)
        elif bit_depth == 16:
            audio_np = (audio_np * 32767).astype(np.int16)
        elif bit_depth == 24:
            audio_np = (audio_np * 8388607).astype(np.int32)
        else:  # 32
            audio_np = (audio_np * 2147483647).astype(np.int32)
        
        # Cria AudioSegment
        segment = AudioSegment(
            audio_np.tobytes(),
            frame_rate=sample_rate,
            sample_width=bit_depth // 8,
            channels=1
        )
        
        # Exporta no formato desejado
        buf = io.BytesIO()
        segment.export(buf, format=format.value)
        
        # Valida tamanho
        result = buf.getvalue()
        validate_audio_size(len(result))
        
        return result
        
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        logger.error(f"Erro ao salvar áudio: {e}")
        raise AudioProcessingError(f"Erro ao salvar áudio: {e}")

def concatenate_audio(
    chunks: list[bytes],
    format: AudioFormat = AudioFormat.WAV
) -> bytes:
    """
    Concatena múltiplos chunks de áudio.
    
    Args:
        chunks: Lista de chunks de áudio em bytes
        format: Formato dos chunks
        
    Returns:
        bytes: Áudio concatenado em bytes
        
    Raises:
        AudioProcessingError: Se houver erro na concatenação
    """
    try:
        if not chunks:
            raise AudioProcessingError("Lista de chunks vazia")
            
        combined = AudioSegment.empty()
        sample_rate = None
        channels = None
        
        for i, chunk in enumerate(chunks):
            try:
                segment = AudioSegment.from_file(
                    io.BytesIO(chunk),
                    format=format.value
                )
                
                # Valida consistência
                if sample_rate is None:
                    sample_rate = segment.frame_rate
                    channels = segment.channels
                elif segment.frame_rate != sample_rate:
                    raise AudioProcessingError(
                        f"Sample rate inconsistente no chunk {i}"
                    )
                elif segment.channels != channels:
                    raise AudioProcessingError(
                        f"Número de canais inconsistente no chunk {i}"
                    )
                
                combined += segment
                
            except Exception as e:
                raise AudioProcessingError(f"Erro no chunk {i}: {e}")
        
        # Exporta resultado
        buf = io.BytesIO()
        combined.export(buf, format=format.value)
        
        # Valida tamanho
        result = buf.getvalue()
        validate_audio_size(len(result))
        
        return result
        
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        logger.error(f"Erro ao concatenar áudios: {e}")
        raise AudioProcessingError(f"Erro ao concatenar áudios: {e}")

def validate_audio(
    audio_bytes: bytes,
    max_duration: float = 30.0,
    allowed_formats: Optional[list[AudioFormat]] = None
) -> Dict[str, Any]:
    """
    Valida um arquivo de áudio.
    
    Args:
        audio_bytes: Áudio em bytes
        max_duration: Duração máxima em segundos
        allowed_formats: Lista de formatos permitidos
        
    Returns:
        Dict[str, Any]: Informações do áudio
        
    Raises:
        AudioProcessingError: Se o áudio for inválido
    """
    try:
        # Valida tamanho
        validate_audio_size(len(audio_bytes))
        
        # Detecta formato
        format_valid = False
        detected_format = None
        
        for fmt in AudioFormat:
            if audio_bytes.startswith(get_format_header(fmt.value)):
                format_valid = True
                detected_format = fmt
                break
                
        if not format_valid:
            raise AudioProcessingError("Formato de áudio não detectado")
            
        if allowed_formats and detected_format not in allowed_formats:
            raise AudioProcessingError(
                f"Formato {detected_format.value} não permitido. "
                f"Formatos aceitos: {[f.value for f in allowed_formats]}"
            )
        
        # Carrega áudio
        audio = AudioSegment.from_file(
            io.BytesIO(audio_bytes),
            format=detected_format.value
        )
        
        # Valida duração
        duration = len(audio) / 1000  # ms para segundos
        if duration > max_duration:
            raise AudioProcessingError(
                f"Áudio muito longo ({duration:.1f}s). "
                f"Máximo permitido: {max_duration}s"
            )
            
        # Valida sample rate
        validate_sample_rate(audio.frame_rate)
        
        # Extrai informações
        info = get_audio_info(audio)
        info["format"] = detected_format.value
        
        return info
        
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        logger.error(f"Erro ao validar áudio: {e}")
        raise AudioProcessingError(f"Erro ao validar áudio: {e}")

def get_format_header(format: str) -> bytes:
    """Retorna o header para um formato de áudio."""
    headers = {
        AudioFormat.WAV.value: b"RIFF",
        AudioFormat.MP3.value: b"ID3",
        AudioFormat.OGG.value: b"OggS",
        AudioFormat.FLAC.value: b"fLaC"
    }
    return headers.get(format.lower(), b"") 