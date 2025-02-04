"""
Módulo responsável pelo gerenciamento de estado do dashboard.
Implementa um sistema de logs com buffer circular e gerenciamento de estado das GPUs.
"""

import asyncio
import collections
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from pydantic import BaseModel

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogEntry(BaseModel):
    """Modelo para entrada de logs"""
    id: str
    timestamp: str
    message: str
    source: str = "system"
    level: str = "INFO"

class DashboardState:
    """
    Classe para gerenciar o estado global do dashboard.
    Implementa um buffer circular para logs e mantém o estado das GPUs.
    """
    
    def __init__(self):
        self._logs = deque(maxlen=1000)  # Otimizado para O(1) nas operações
        self._gpu_states: Dict[str, Dict[str, Any]] = {}
        self._connection_state = False
        self._connection_error: Optional[str] = None
        self._lock = asyncio.Lock()
        
    async def add_log(self, message: str, level: str = "INFO"):
        """Adiciona log com validação"""
        log_entry = LogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            message=message,
            level=level
        )
        async with self._lock:
            self._logs.append(log_entry.dict(exclude_unset=True))
            
    def get_recent_logs(self, count: int = 100) -> List[dict]:
        """Retorna os logs mais recentes como dicionários"""
        return list(self._logs)[-count:]
            
    async def update_gpu_state(self, gpu_id: str, state: Dict[str, Any]) -> None:
        """Valida o estado da GPU antes de atualizar"""
        try:
            validated = GPUState(**state)
            async with self._lock:
                self._gpu_states[gpu_id] = validated.dict()
        except ValidationError as e:
            error_msg = f"Estado inválido da GPU {gpu_id}: {str(e)}"
            await self.add_log(error_msg)
            async with self._lock:
                self._gpu_states[gpu_id] = {
                    "status": "ERROR",
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            
    async def get_gpu_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna o estado atual de todas as GPUs.
        
        Returns:
            Dict[str, Dict[str, Any]]: Estado atual das GPUs
        """
        async with self._lock:
            return self._gpu_states.copy()
            
    async def set_connection_state(self, connected: bool, error: Optional[str] = None) -> None:
        """Atualiza o estado da conexão."""
        async with self._lock:
            self._connection_state = connected
            self._connection_error = error
            
            if not connected and error:
                await self.add_log(f"Erro de conexão: {error}")
            
    def get_connection_state(self) -> Dict[str, any]:
        """Retorna o estado atual da conexão."""
        return {
            "connected": self._connection_state,
            "error": self._connection_error
        }

# Singleton para estado global
_dashboard_state = DashboardState()

def get_dashboard_state() -> DashboardState:
    """Retorna a instância global do estado do dashboard."""
    return _dashboard_state

dashboard_state = _dashboard_state 