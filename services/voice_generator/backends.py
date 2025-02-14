"""
Backends para geração de voz com suporte a múltiplos serviços.
"""
from abc import ABC, abstractmethod
import logging
import torch
import torchaudio
from typing import Optional, Dict, Any, Union
from pathlib import Path
import io
from pydub import AudioSegment
import aiohttp
import asyncio
from config import FISH_AUDIO_CONFIG, FISH_SPEECH_CONFIG, BACKEND_CONFIG
from shared.gpu_utils import cuda_memory_manager
from cache_manager import VoiceCache, cache_voice, cache_embedding
import time

logger = logging.getLogger(__name__)

class VoiceBackend(ABC):
    """Interface base para backends de geração de voz."""
    
    def __init__(self):
        """Inicializa o backend."""
        self.cache = VoiceCache()
    
    @abstractmethod
    @cache_voice()
    async def tts(
        self,
        text: str,
        language: str = "auto",
        speaker: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """Gera áudio a partir de texto."""
        pass
    
    @abstractmethod
    @cache_voice()
    async def clone_voice(
        self,
        text: str,
        reference_audio: bytes,
        language: str = "auto",
        **kwargs
    ) -> bytes:
        """Gera áudio com clonagem de voz."""
        pass

class FishAudioAPIBackend(VoiceBackend):
    """Backend que usa a API do Fish Audio."""
    
    def __init__(self, api_key: str):
        """Inicializa o backend."""
        super().__init__()
        self.api_key = api_key
        self.base_url = FISH_AUDIO_CONFIG["base_url"]
        self.timeout = FISH_AUDIO_CONFIG["timeout"]
        self.max_retries = FISH_AUDIO_CONFIG["max_retries"]
        self.track_latency = BACKEND_CONFIG["monitoring"]["track_latency"]
        self.track_errors = BACKEND_CONFIG["monitoring"]["track_errors"]
    
    async def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        files: Optional[Dict[str, bytes]] = None
    ) -> bytes:
        """Faz requisição para a API do Fish Audio."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        start_time = time.time() if self.track_latency else None
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    if files:
                        # Multipart request para upload de arquivo
                        form = aiohttp.FormData()
                        for k, v in data.items():
                            form.add_field(k, str(v))
                        for k, v in files.items():
                            form.add_field(k, v)
                        
                        async with session.post(
                            f"{self.base_url}/{endpoint}",
                            data=form,
                            headers=headers,
                            timeout=self.timeout
                        ) as response:
                            if response.status != 200:
                                raise RuntimeError(
                                    f"API error: {response.status} - {await response.text()}"
                                )
                            result = await response.read()
                    else:
                        # Request JSON normal
                        async with session.post(
                            f"{self.base_url}/{endpoint}",
                            json=data,
                            headers=headers,
                            timeout=self.timeout
                        ) as response:
                            if response.status != 200:
                                raise RuntimeError(
                                    f"API error: {response.status} - {await response.text()}"
                                )
                            result = await response.read()
                    
                    # Registra latência
                    if self.track_latency:
                        latency = time.time() - start_time
                        logger.info(f"API request latency: {latency:.2f}s")
                    
                    return result
                    
            except Exception as e:
                if self.track_errors:
                    logger.error(f"API request error (attempt {attempt + 1}): {e}")
                
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"API request failed after {self.max_retries} attempts: {e}")
                
                await asyncio.sleep(BACKEND_CONFIG["retry_delay"] * (2 ** attempt))
    
    @cache_voice()
    async def tts(
        self,
        text: str,
        language: str = "auto",
        speaker: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """Gera áudio usando a API do Fish Audio."""
        data = {
            "text": text,
            "language": language,
            "speed": speed,
            "model": FISH_AUDIO_CONFIG["models"]["default"]
        }
        if speaker:
            data["speaker"] = speaker
            
        return await self._make_request("tts", data)
    
    @cache_voice()
    async def clone_voice(
        self,
        text: str,
        reference_audio: bytes,
        language: str = "auto",
        **kwargs
    ) -> bytes:
        """Clona voz usando a API do Fish Audio."""
        data = {
            "text": text,
            "language": language,
            "model": FISH_AUDIO_CONFIG["models"]["default"]
        }
        files = {"reference_audio": reference_audio}
        
        return await self._make_request("clone", data, files)

class LocalFishSpeechBackend(VoiceBackend):
    """Backend que usa o modelo Fish Speech localmente."""
    
    def __init__(self):
        """Inicializa o backend local."""
        super().__init__()
        self.model = None
        self.firefly = None
        self.device = FISH_SPEECH_CONFIG["device"]
        self.use_fp16 = FISH_SPEECH_CONFIG["use_fp16"]
        self.sample_rate = FISH_SPEECH_CONFIG["sample_rate"]
        self.track_latency = BACKEND_CONFIG["monitoring"]["track_latency"]
        self.track_errors = BACKEND_CONFIG["monitoring"]["track_errors"]
    
    async def _ensure_model_loaded(self):
        """Garante que o modelo está carregado."""
        if self.model is None:
            try:
                from fish_speech.inference_engine import TTSInferenceEngine
                from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
                
                # Carregar modelo TTS
                self.model = TTSInferenceEngine(
                    model_path=FISH_SPEECH_CONFIG["model_path"],
                    device=self.device,
                    use_fp16=self.use_fp16
                )
                
                # Carregar modelo Firefly para clonagem
                self.firefly = FireflyArchitecture(
                    config_path=str(Path(FISH_SPEECH_CONFIG["model_path"]).parent / "config.json"),
                    device=self.device
                )
                
                if self.use_fp16:
                    self.model.half()
                    self.firefly.half()
                    
                logger.info("Modelo Fish Speech local carregado com sucesso")
                
            except Exception as e:
                if self.track_errors:
                    logger.error(f"Erro ao carregar modelo local: {e}")
                raise RuntimeError(f"Erro ao carregar modelo local: {e}")
    
    def _convert_to_bytes(self, wav: torch.Tensor) -> bytes:
        """Converte tensor de áudio para bytes."""
        buf = io.BytesIO()
        AudioSegment(
            wav.cpu().numpy().tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1
        ).export(buf, format="wav")
        return buf.getvalue()
    
    @cache_voice()
    async def tts(
        self,
        text: str,
        language: str = "auto",
        speaker: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """Gera áudio usando o modelo local."""
        await self._ensure_model_loaded()
        start_time = time.time() if self.track_latency else None
        
        try:
            async with cuda_memory_manager():
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.use_fp16):
                    wav = self.model.tts(
                        text=text,
                        language=language,
                        speaker=speaker,
                        speed=speed,
                        **kwargs
                    )
            
            # Registra latência
            if self.track_latency:
                latency = time.time() - start_time
                logger.info(f"Local TTS latency: {latency:.2f}s")
            
            return self._convert_to_bytes(wav)
            
        except Exception as e:
            if self.track_errors:
                logger.error(f"Erro na síntese local: {e}")
            raise RuntimeError(f"Erro na síntese local: {e}")
    
    @cache_embedding()
    async def _extract_embedding(self, reference_audio: bytes) -> torch.Tensor:
        """Extrai embedding do áudio de referência."""
        # Carregar áudio de referência
        wav, sr = torchaudio.load(io.BytesIO(reference_audio))
        
        # Converter para mono se necessário
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample para 16kHz se necessário
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
        
        # Normalizar
        wav = wav / wav.abs().max()
        wav = wav.to(self.device)
        
        # Extrair embedding
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.use_fp16):
            return self.firefly.extract_speaker(wav)
    
    @cache_voice()
    async def clone_voice(
        self,
        text: str,
        reference_audio: bytes,
        language: str = "auto",
        **kwargs
    ) -> bytes:
        """Clona voz usando o modelo local."""
        await self._ensure_model_loaded()
        start_time = time.time() if self.track_latency else None
        
        try:
            # Extrair embedding (com cache)
            speaker_emb = await self._extract_embedding(reference_audio)
            
            async with cuda_memory_manager():
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.use_fp16):
                    # Gerar áudio com as características
                    wav = self.model.tts(
                        text=text,
                        language=language,
                        speaker_embedding=speaker_emb,
                        **kwargs
                    )
            
            # Registra latência
            if self.track_latency:
                latency = time.time() - start_time
                logger.info(f"Local voice clone latency: {latency:.2f}s")
            
            return self._convert_to_bytes(wav)
            
        except Exception as e:
            if self.track_errors:
                logger.error(f"Erro na clonagem local: {e}")
            raise RuntimeError(f"Erro na clonagem local: {e}")

class VoiceBackendManager:
    """Gerenciador de backends com fallback automático."""
    
    def __init__(self):
        """Inicializa o gerenciador de backends."""
        self.backends = []
        self.track_latency = BACKEND_CONFIG["monitoring"]["track_latency"]
        self.track_errors = BACKEND_CONFIG["monitoring"]["track_errors"]
        self.track_cache_hits = BACKEND_CONFIG["monitoring"]["track_cache_hits"]
        self.cache = VoiceCache()
        self._setup_backends()
    
    def _setup_backends(self):
        """Configura os backends disponíveis."""
        # Determina ordem dos backends
        if BACKEND_CONFIG["preferred_backend"] == "api":
            backend_order = [FishAudioAPIBackend, LocalFishSpeechBackend]
        else:
            backend_order = [LocalFishSpeechBackend, FishAudioAPIBackend]
        
        # Configura backends na ordem preferida
        for backend_class in backend_order:
            try:
                if backend_class == FishAudioAPIBackend:
                    api_key = FISH_AUDIO_CONFIG.get("api_key")
                    if api_key:
                        self.backends.append(FishAudioAPIBackend(api_key))
                        logger.info("Backend Fish Audio API configurado")
                else:
                    self.backends.append(LocalFishSpeechBackend())
                    logger.info("Backend Fish Speech local configurado")
            except Exception as e:
                if self.track_errors:
                    logger.warning(f"Erro ao configurar {backend_class.__name__}: {e}")
        
        if not self.backends:
            raise RuntimeError("Nenhum backend disponível")
    
    @cache_voice()
    async def generate(
        self,
        text: str,
        clone_audio: Optional[bytes] = None,
        **kwargs
    ) -> bytes:
        """
        Gera áudio usando os backends disponíveis com fallback automático.
        
        Args:
            text: Texto para sintetizar
            clone_audio: Áudio de referência para clonagem (opcional)
            **kwargs: Parâmetros adicionais para a geração
            
        Returns:
            bytes: Áudio gerado em formato WAV
            
        Raises:
            RuntimeError: Se todos os backends falharem
        """
        start_time = time.time() if self.track_latency else None
        last_error = None
        
        for backend in self.backends:
            try:
                if clone_audio:
                    result = await backend.clone_voice(text, clone_audio, **kwargs)
                else:
                    result = await backend.tts(text, **kwargs)
                
                # Registra latência total
                if self.track_latency:
                    latency = time.time() - start_time
                    logger.info(
                        f"Total generation latency ({backend.__class__.__name__}): {latency:.2f}s"
                    )
                
                return result
                
            except Exception as e:
                if self.track_errors:
                    logger.warning(f"Backend {backend.__class__.__name__} falhou: {e}")
                last_error = e
                
                # Aguarda antes de tentar próximo backend
                if BACKEND_CONFIG["fallback_enabled"]:
                    await asyncio.sleep(BACKEND_CONFIG["retry_delay"])
                    continue
                else:
                    raise
        
        raise RuntimeError(f"Todos os backends falharam. Último erro: {last_error}") 