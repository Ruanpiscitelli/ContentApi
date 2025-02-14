"""
GPU utilities shared across all services.
"""

import os
import torch
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU resources and optimizations."""
    
    @staticmethod
    def get_available_devices() -> List[int]:
        """Returns list of available GPU devices."""
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    
    @staticmethod
    def get_device(device_id: Optional[int] = None) -> torch.device:
        """Gets the specified or best available device."""
        if not torch.cuda.is_available():
            return torch.device('cpu')
            
        if device_id is not None and device_id < torch.cuda.device_count():
            return torch.device(f'cuda:{device_id}')
            
        return torch.device('cuda')
    
    @staticmethod
    def optimize_gpu_memory():
        """Applies memory optimizations for GPU usage."""
        if not torch.cuda.is_available():
            return
            
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        
        # Enable gradients AMP
        torch.set_float32_matmul_precision('high')
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Returns information about GPU usage."""
        if not torch.cuda.is_available():
            return {"available": False}
            
        info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "devices": []
        }
        
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "name": device.name,
                "total_memory": device.total_memory / 1024**3,  # Convert to GB
                "major": device.major,
                "minor": device.minor,
                "multi_processor_count": device.multi_processor_count
            })
            
        return info
    
    @staticmethod
    def clear_gpu_memory():
        """Clears unused memory on GPUs."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# Initialize GPU optimizations on module import
GPUManager.optimize_gpu_memory() 