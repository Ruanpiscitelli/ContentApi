"""
Módulo responsável pelo monitoramento das GPUs NVIDIA.
Utiliza a biblioteca NVML para coletar métricas em tempo real.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from functools import partial
from pydantic import BaseModel, ValidationError
from circuitbreaker import circuit  # Adicionar dependência

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml não está instalado. O monitoramento de GPU será limitado.")

from state import dashboard_state, get_dashboard_state

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUState(BaseModel):
    """Modelo de validação para o estado da GPU"""
    name: str
    status: str
    utilization: float
    memory: dict
    temperature: float
    power_usage: float
    sm_clock: int
    processes: int
    timestamp: str
    error: Optional[str] = None

class GPUMonitor:
    """
    Classe para monitorar GPUs NVIDIA usando NVML.
    Coleta métricas como utilização, memória, temperatura e clock.
    """
    
    def __init__(self):
        """Inicializa o monitor de GPU."""
        self._initialized = False
        self._error_count = 0
        self._max_errors = 3
        self._last_error: Optional[str] = None
        self._lock = asyncio.Lock()
        self._handles = {}  # Dicionário para armazenar handles
        
    async def initialize(self) -> None:
        """
        Inicializa a biblioteca NVML.
        Deve ser chamado antes de começar a coletar métricas.
        """
        async with self._lock:
            if not NVML_AVAILABLE:
                await dashboard_state.add_log("NVML não disponível. Monitoramento de GPU desabilitado.")
                return
            
            try:
                if not self._initialized:
                    pynvml.nvmlInit()
                    self._initialized = True
                    await dashboard_state.add_log("NVML inicializado com sucesso")
                    await dashboard_state.set_connection_state(True)
                    
                    # Pré-alocar handles
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        self._handles[f"GPU{i}"] = handle
            except Exception as e:
                self._last_error = str(e)
                await dashboard_state.add_log(f"Erro ao inicializar NVML: {e}")
                await dashboard_state.set_connection_state(False, str(e))
            
    async def shutdown(self) -> None:
        """
        Desliga a biblioteca NVML.
        Deve ser chamado ao encerrar o aplicativo.
        """
        async with self._lock:
            if self._initialized and NVML_AVAILABLE:
                try:
                    pynvml.nvmlShutdown()
                    self._handles.clear()
                    self._initialized = False
                    await dashboard_state.add_log("NVML desligado com sucesso")
                except Exception as e:
                    await dashboard_state.add_log(f"Erro ao desligar NVML: {e}")
            
    def _get_gpu_metrics_sync(self, handle: Any, gpu_index: int) -> Dict[str, Any]:
        """
        Coleta métricas detalhadas de uma GPU específica de forma síncrona.
        
        Args:
            handle: Handle da GPU obtido via NVML
            gpu_index (int): Índice da GPU
            
        Returns:
            Dict[str, Any]: Métricas da GPU
        """
        try:
            # Informações básicas
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Informações de energia e clock
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Converte para watts
            except pynvml.NVMLError:
                power = 0
                
            try:
                clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except pynvml.NVMLError:
                clock = 0
                
            # Nome da GPU
            try:
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            except:
                name = f"GPU {gpu_index}"
                
            # Processos usando a GPU
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                process_count = len(processes)
            except:
                process_count = 0
                
            return {
                "name": name,
                "status": "OK",
                "utilization": util.gpu,  # Porcentagem
                "memory": {
                    "used": round(info.used / (1024**2), 2),  # MB
                    "total": round(info.total / (1024**2), 2),  # MB
                    "percent": round((info.used / info.total) * 100, 2)
                },
                "temperature": temp,  # Celsius
                "power_usage": round(power, 2),  # Watts
                "sm_clock": clock,  # MHz
                "processes": process_count,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Erro ao coletar métricas da GPU {gpu_index}: {e}")
            return {
                "name": f"GPU {gpu_index}",
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    @circuit(failure_threshold=3, recovery_timeout=30)
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Coleta métricas de todas as GPUs disponíveis.
        
        Returns:
            Dict[str, Any]: Métricas de todas as GPUs
        """
        async with self._lock:
            if not self._initialized or not NVML_AVAILABLE:
                return {"GPU0": {"status": "UNAVAILABLE", "error": "NVML não inicializado"}}
                
            try:
                metrics = {}
                for gpu_id, handle in self._handles.items():
                    try:
                        raw_metrics = await asyncio.get_event_loop().run_in_executor(
                            None, self._get_gpu_metrics_sync, handle, gpu_id
                        )
                        validated = GPUState(**raw_metrics)
                        metrics[gpu_id] = validated.dict()
                        
                        await asyncio.get_event_loop().run_in_executor(
                            None, pynvml.nvmlDeviceReleaseHandle, handle
                        )
                    except ValidationError as ve:
                        error_msg = f"Erro de validação na {gpu_id}: {str(ve)}"
                        await dashboard_state.add_log(error_msg)
                        metrics[gpu_id] = {
                            "status": "ERROR",
                            "error": error_msg,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                self._error_count = 0
                await dashboard_state.set_connection_state(True)
                return metrics
                
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                error_msg = f"Erro geral na coleta de métricas: {str(e)}"
                await dashboard_state.add_log(error_msg)
                
                if self._error_count >= self._max_errors:
                    await dashboard_state.set_connection_state(False, error_msg)
                    
                return {"GPU0": {
                    "status": "ERROR",
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                }}

# Instância global do monitor de GPU
gpu_monitor = GPUMonitor()

def get_gpu_monitor() -> GPUMonitor:
    """
    Retorna a instância global do monitor de GPU.
    
    Returns:
        GPUMonitor: Instância global do monitor
    """
    return gpu_monitor 