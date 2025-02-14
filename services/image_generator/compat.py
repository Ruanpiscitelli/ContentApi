# compat.py - Camada de compatibilidade para migração Pydantic v1 -> v2
from typing import Any, Dict, Optional, Union, List, Literal
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, ConfigDict

class BaseModel(PydanticBaseModel):
    """Classe base para modelos com configurações compatíveis v1/v2"""
    model_config = ConfigDict(
        populate_by_name=True,  # Equivalente ao allow_population_by_field_name v1
        validate_assignment=True,  # Equivalente ao validate_assignment v1
        extra='forbid',  # Equivalente ao extra='forbid' v1
        str_strip_whitespace=True,  # Remove espaços em branco de strings
        validate_default=True  # Valida valores default
    ) 