"""
Backend WebSocket para streaming de áudio em tempo real.
"""
import asyncio
import logging
import json
from typing import Optional, Dict, Any, AsyncGenerator
import aiohttp
from contextlib import asynccontextmanager
from config import FISH_AUDIO_CONFIG, BACKEND_CONFIG
from exceptions import WebSocketError

logger = logging.getLogger(__name__)

class WebSocketSession:
    """Sessão WebSocket para streaming de áudio."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = FISH_AUDIO_CONFIG["base_url"].replace("http", "ws")
        self.track_latency = BACKEND_CONFIG["monitoring"]["track_latency"]
        self.track_errors = BACKEND_CONFIG["monitoring"]["track_errors"]
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._active_connections = 0
        self._max_connections = BACKEND_CONFIG.get("max_ws_connections", 100)
    
    async def __aenter__(self):
        """Suporte para uso com context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup ao sair do context manager."""
        await self.close()
    
    @asynccontextmanager
    async def _get_connection(self):
        """Gerencia conexão WebSocket com cleanup automático."""
        if self._active_connections >= self._max_connections:
            raise WebSocketError("Número máximo de conexões atingido")
            
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            self._ws = await self._session.ws_connect(
                f"{self.base_url}/ws/tts",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            self._active_connections += 1
            
            yield self._ws
            
        except Exception as e:
            raise WebSocketError(f"Erro na conexão WebSocket: {e}")
            
        finally:
            if self._ws:
                await self._ws.close()
                self._ws = None
                self._active_connections -= 1
    
    async def close(self):
        """Fecha a sessão e limpa recursos."""
        if self._ws:
            await self._ws.close()
            self._ws = None
            
        if self._session:
            await self._session.close()
            self._session = None
            
        self._active_connections = 0
    
    async def tts_stream(
        self,
        text: str,
        language: str = "auto",
        speaker: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Gera áudio via streaming usando WebSocket.
        
        Args:
            text: Texto para sintetizar
            language: Idioma do texto
            speaker: ID do speaker (opcional)
            speed: Velocidade da fala
            **kwargs: Parâmetros adicionais
            
        Yields:
            bytes: Chunks de áudio WAV
            
        Raises:
            WebSocketError: Se houver erro na conexão ou streaming
        """
        async with self._get_connection() as ws:
            try:
                # Envia parâmetros
                await ws.send_json({
                    "text": text,
                    "language": language,
                    "speed": speed,
                    "model": FISH_AUDIO_CONFIG["models"]["default"],
                    **({"speaker": speaker} if speaker else {}),
                    **kwargs
                })
                
                # Recebe chunks de áudio
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        yield msg.data
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("event") == "error":
                            raise WebSocketError(f"Erro no streaming: {data.get('reason')}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise WebSocketError(f"Erro no WebSocket: {ws.exception()}")
                        
            except Exception as e:
                if self.track_errors:
                    logger.error(f"Erro no streaming TTS: {e}")
                raise WebSocketError(f"Erro no streaming TTS: {e}")
    
    async def clone_voice_stream(
        self,
        text: str,
        reference_audio: bytes,
        language: str = "auto",
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Clona voz via streaming usando WebSocket.
        
        Args:
            text: Texto para sintetizar
            reference_audio: Áudio de referência em bytes
            language: Idioma do texto
            **kwargs: Parâmetros adicionais
            
        Yields:
            bytes: Chunks de áudio WAV
            
        Raises:
            WebSocketError: Se houver erro na conexão ou streaming
        """
        async with self._get_connection() as ws:
            try:
                # Envia parâmetros e áudio de referência
                await ws.send_json({
                    "text": text,
                    "language": language,
                    "model": FISH_AUDIO_CONFIG["models"]["default"],
                    **kwargs
                })
                await ws.send_bytes(reference_audio)
                
                # Recebe chunks de áudio
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        yield msg.data
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("event") == "error":
                            raise WebSocketError(f"Erro no streaming: {data.get('reason')}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise WebSocketError(f"Erro no WebSocket: {ws.exception()}")
                        
            except Exception as e:
                if self.track_errors:
                    logger.error(f"Erro no streaming de clonagem: {e}")
                raise WebSocketError(f"Erro no streaming de clonagem: {e}")