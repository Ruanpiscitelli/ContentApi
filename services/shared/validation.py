"""
Validation utilities shared across all services.
"""

from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status
import torch
import logging

logger = logging.getLogger(__name__)

class ResourceValidationError(Exception):
    """Exception for resource validation errors."""
    pass

class GPUResourceValidator:
    """Validates GPU resource requirements."""
    
    @staticmethod
    def validate_gpu_availability():
        """Validates if GPU is available."""
        if not torch.cuda.is_available():
            raise ResourceValidationError("No GPU available")
            
    @staticmethod
    def validate_memory_requirements(required_memory_gb: float):
        """Validates if enough GPU memory is available."""
        try:
            if not torch.cuda.is_available():
                raise ResourceValidationError("No GPU available")
                
            # Get total and free memory
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
            free_memory = total_memory - (torch.cuda.memory_allocated() / (1024**3))  # GB
            
            if free_memory < required_memory_gb:
                raise ResourceValidationError(
                    f"Not enough GPU memory. Required: {required_memory_gb}GB, Available: {free_memory:.2f}GB"
                )
        except Exception as e:
            logger.error(f"Error validating GPU memory: {e}")
            raise ResourceValidationError(str(e))

class BaseRequestValidator(BaseModel):
    """Base class for request validation."""
    
    @validator("*", pre=True)
    def empty_str_to_none(cls, v):
        if v == "":
            return None
        return v
        
    class Config:
        arbitrary_types_allowed = True
        
class PaginationParams(BaseModel):
    """Common pagination parameters."""
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=10, ge=1, le=100)
    
class ModelLoadValidator:
    """Validates model loading requirements."""
    
    @staticmethod
    def validate_model_path(model_path: str) -> bool:
        """Validates if model file exists and is accessible."""
        try:
            import os
            if not os.path.exists(model_path):
                raise ResourceValidationError(f"Model not found at {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error validating model path: {e}")
            raise ResourceValidationError(str(e))
            
    @staticmethod
    def validate_model_requirements(
        model_path: str,
        required_memory_gb: Optional[float] = None,
        required_compute_capability: Optional[float] = None
    ):
        """Validates if system meets model requirements."""
        try:
            # Validate model file
            ModelLoadValidator.validate_model_path(model_path)
            
            # Validate GPU availability
            GPUResourceValidator.validate_gpu_availability()
            
            # Validate memory if specified
            if required_memory_gb:
                GPUResourceValidator.validate_memory_requirements(required_memory_gb)
                
            # Validate compute capability if specified
            if required_compute_capability and torch.cuda.is_available():
                device = torch.cuda.current_device()
                cc = float(f"{torch.cuda.get_device_properties(device).major}.{torch.cuda.get_device_properties(device).minor}")
                if cc < required_compute_capability:
                    raise ResourceValidationError(
                        f"GPU compute capability {cc} is lower than required {required_compute_capability}"
                    )
                    
            return True
        except Exception as e:
            logger.error(f"Error validating model requirements: {e}")
            raise ResourceValidationError(str(e)) 