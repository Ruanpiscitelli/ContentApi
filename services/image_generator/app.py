"""
Serviço de geração de imagem usando Stable Diffusion.
"""
import os
import uuid
import time
import torch
import logging
import asyncio
import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler
)

from services.image_generator.validations import (
    ValidationError,
    ResourceError,
    ProcessingError,
    validate_model_compatibility,
    validate_memory_requirements,
    validate_scheduler_config,
    validate_lora_weights,
    validate_vae_config,
    load_and_validate_template
)

from services.image_generator.config import DEVICE, API_CONFIG
from services.image_generator.auth import verify_bearer_token
from services.image_generator.storage import upload_to_minio, check_minio_connection
from services.image_generator.utils import convert_image_to_bytes
from services.image_generator.routers import image
from services.image_generator.pipeline import initialize_models, cleanup_resources

# ... resto do código ... 