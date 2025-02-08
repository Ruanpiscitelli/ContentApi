"""
Configurações para o serviço de edição de vídeo.
Inclui otimizações de processamento e gestão de recursos.
"""

import os
from typing import Dict, Any

# Configurações de Hardware
HARDWARE_CONFIG = {
    "use_gpu": True,  # Usar GPU quando disponível
    "gpu_memory_limit": 0.8,  # Limite de uso de memória GPU (80%)
    "cpu_threads": os.cpu_count(),  # Número de threads para processamento
    "memory_limit": 0.7  # Limite de uso de memória RAM (70%)
}

# Configurações de Pipeline
PIPELINE_CONFIG = {
    "max_batch_size": 5,  # Número máximo de vídeos em batch
    "max_queue_size": 100,  # Tamanho máximo da fila
    "processing_timeout": 3600,  # Timeout em segundos
    "cleanup_interval": 3600,  # Intervalo de limpeza em segundos
    "temp_file_ttl": 3600  # Tempo de vida de arquivos temporários
}

# Configurações de Cache
CACHE_CONFIG = {
    "enabled": True,
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "ttl": 24 * 3600,  # 24 horas
    "max_size": 10 * 1024 * 1024 * 1024  # 10GB
}

# Configurações de Codecs
CODEC_CONFIG = {
    "video_encoders": {
        "h264": {
            "codec": "libx264",
            "preset": "medium",
            "crf": 23,
            "gpu_acceleration": True
        },
        "h265": {
            "codec": "libx265",
            "preset": "medium",
            "crf": 28,
            "gpu_acceleration": True
        }
    },
    "audio_encoders": {
        "aac": {
            "codec": "aac",
            "bitrate": "192k"
        },
        "opus": {
            "codec": "libopus",
            "bitrate": "128k"
        }
    }
}

# Configurações de Rate Limiting
RATE_LIMIT_CONFIG = {
    "enabled": True,
    "requests_per_minute": 60,
    "burst_size": 10
}

# Configurações de Validação
VALIDATION_CONFIG = {
    "max_file_size": 1024 * 1024 * 1024,  # 1GB
    "max_duration": 3600,  # 1 hora
    "allowed_formats": ["mp4", "avi", "mov", "mkv", "webm"],
    "allowed_video_codecs": ["h264", "h265", "vp8", "vp9"],
    "allowed_audio_codecs": ["aac", "opus", "mp3"]
}

def get_ffmpeg_options(quality: str = "medium", use_gpu: bool = True) -> Dict[str, Any]:
    """
    Retorna opções otimizadas do FFmpeg baseadas na qualidade desejada.
    
    Args:
        quality: Qualidade desejada ("low", "medium", "high")
        use_gpu: Se deve usar aceleração GPU
        
    Returns:
        Dicionário com opções do FFmpeg
    """
    options = {
        "low": {
            "video_bitrate": "1000k",
            "audio_bitrate": "128k",
            "crf": 28,
            "preset": "veryfast"
        },
        "medium": {
            "video_bitrate": "2000k",
            "audio_bitrate": "192k",
            "crf": 23,
            "preset": "medium"
        },
        "high": {
            "video_bitrate": "4000k",
            "audio_bitrate": "256k",
            "crf": 18,
            "preset": "slow"
        }
    }
    
    base_options = options.get(quality, options["medium"])
    
    if use_gpu and HARDWARE_CONFIG["use_gpu"]:
        base_options.update({
            "hwaccel": "cuda",
            "hwaccel_output_format": "cuda",
            "extra_hw_frames": 2
        })
    
    return base_options 