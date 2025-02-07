"""
Fish Speech Wrapper - Implementação otimizada para o serviço de voz
"""
import os
import logging
import torch
import torchaudio
import numpy as np
from typing import Optional, Union, Dict, List
from pathlib import Path
import huggingface_hub

logger = logging.getLogger(__name__)

class FishSpeechWrapper:
    """Wrapper para o modelo Fish Speech com otimizações."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        download_if_missing: bool = True
    ):
        """
        Inicializa o wrapper do Fish Speech.
        
        Args:
            model_path: Caminho para os modelos
            device: Dispositivo para inferência ('cuda' ou 'cpu')
            use_fp16: Usar precisão FP16 para otimização
            download_if_missing: Baixar modelos se não existirem
        """
        self.model_path = Path(model_path)
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        
        # Verifica e baixa modelos se necessário
        if download_if_missing:
            self._ensure_models_exist()
        
        # Carrega o modelo
        self._load_model()
        
        logger.info(
            f"Fish Speech inicializado (device: {device}, fp16: {use_fp16})"
        )
    
    def _ensure_models_exist(self):
        """Verifica e baixa os modelos necessários."""
        if not self.model_path.exists():
            logger.info(f"Baixando modelos para {self.model_path}")
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            try:
                huggingface_hub.snapshot_download(
                    repo_id="fishaudio/fish-speech-1.5",
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                raise RuntimeError(f"Erro ao baixar modelos: {e}")
    
    def _load_model(self):
        """Carrega e otimiza o modelo."""
        try:
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
            
            # Configurar modelo
            self.model = TTSInferenceEngine(
                model_path=str(self.model_path),
                device=self.device,
                use_fp16=self.use_fp16
            )
            
            # Carregar arquitetura Firefly para clonagem de voz
            self.firefly = FireflyArchitecture(
                config_path=str(self.model_path / "config.json"),
                device=self.device
            )
            
            if self.use_fp16:
                self.model.half()
                self.firefly.half()
            
            # Otimizações para GPU
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            
        except ImportError as e:
            raise ImportError(
                f"Erro ao importar Fish Speech. Certifique-se de que está instalado: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo: {e}")
    
    def tts(
        self,
        text: str,
        language: str = "auto",
        speaker: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Sintetiza texto em fala.
        
        Args:
            text: Texto para sintetizar
            language: Idioma do texto ('auto', 'en', 'zh', 'ja', etc)
            speaker: ID do speaker (opcional)
            speed: Velocidade da fala (0.5 a 2.0)
            
        Returns:
            torch.Tensor: Áudio gerado
        """
        try:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.use_fp16):
                wav = self.model.tts(
                    text=text,
                    language=language,
                    speaker=speaker,
                    speed=speed,
                    **kwargs
                )
            return wav
            
        except Exception as e:
            logger.error(f"Erro na síntese de voz: {e}")
            raise RuntimeError(f"Erro na síntese de voz: {e}")
    
    def tts_batch(
        self,
        texts: List[str],
        languages: List[str] = None,
        speaker_embeddings: List[Optional[torch.Tensor]] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Sintetiza múltiplos textos em batch.
        
        Args:
            texts: Lista de textos para sintetizar
            languages: Lista de idiomas (opcional)
            speaker_embeddings: Lista de embeddings de speaker (opcional)
            
        Returns:
            List[torch.Tensor]: Lista de áudios gerados
        """
        if not texts:
            return []
            
        try:
            # Prepara parâmetros
            if languages is None:
                languages = ["auto"] * len(texts)
            if speaker_embeddings is None:
                speaker_embeddings = [None] * len(texts)
                
            # Valida tamanhos
            if not (len(texts) == len(languages) == len(speaker_embeddings)):
                raise ValueError("Todas as listas devem ter o mesmo tamanho")
            
            # Processa em batch
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.use_fp16):
                wavs = self.model.tts_batch(
                    texts=texts,
                    languages=languages,
                    speaker_embeddings=speaker_embeddings,
                    **kwargs
                )
            
            return wavs
            
        except Exception as e:
            logger.error(f"Erro na síntese em batch: {e}")
            raise RuntimeError(f"Erro na síntese em batch: {e}")
    
    def tts_with_vc(
        self,
        text: str,
        speaker_wav: bytes,
        language: str = "auto",
        **kwargs
    ) -> torch.Tensor:
        """
        Sintetiza texto em fala com clonagem de voz.
        
        Args:
            text: Texto para sintetizar
            speaker_wav: Áudio de referência em bytes
            language: Idioma do texto
            
        Returns:
            torch.Tensor: Áudio gerado
        """
        try:
            # Carregar áudio de referência
            wav_tensor = self._load_reference_audio(speaker_wav)
            
            # Extrair características do speaker
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.use_fp16):
                speaker_emb = self.firefly.extract_speaker(wav_tensor)
                
                # Gerar áudio com as características do speaker
                wav = self.model.tts(
                    text=text,
                    language=language,
                    speaker_embedding=speaker_emb,
                    **kwargs
                )
            
            return wav
            
        except Exception as e:
            logger.error(f"Erro na clonagem de voz: {e}")
            raise RuntimeError(f"Erro na clonagem de voz: {e}")
    
    def _load_reference_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """Carrega e pré-processa áudio de referência."""
        import io
        
        try:
            # Carregar áudio
            wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
            
            # Converter para mono se necessário
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Resample para 16kHz se necessário
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
            
            # Normalizar
            wav = wav / wav.abs().max()
            
            return wav.to(self.device)
            
        except Exception as e:
            raise ValueError(f"Erro ao processar áudio de referência: {e}")
    
    def __del__(self):
        """Cleanup ao destruir o objeto."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "firefly"):
            del self.firefly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()