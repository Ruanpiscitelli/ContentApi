"""
Funções de chamada para o serviço de geração de texto.
"""
import json
import logging
import inspect
from typing import Dict, Any, List, Optional, Callable, Type
from pydantic import BaseModel, create_model, ValidationError

from metrics import ERROR_COUNTER

logger = logging.getLogger(__name__)

class FunctionRegistry:
    """Registro de funções disponíveis para chamada."""
    
    def __init__(self):
        """Inicializa o registro."""
        self.functions: Dict[str, Callable] = {}
        self.schemas: Dict[str, Type[BaseModel]] = {}
    
    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Callable:
        """
        Decorator para registrar uma função.
        
        Args:
            name: Nome opcional da função
            description: Descrição opcional da função
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            # Obtém nome da função
            func_name = name or func.__name__
            
            # Obtém docstring
            func_description = description or inspect.getdoc(func) or ""
            
            # Obtém assinatura
            sig = inspect.signature(func)
            
            # Cria schema dos parâmetros
            params = {}
            for param_name, param in sig.parameters.items():
                if param.annotation == inspect.Parameter.empty:
                    # Se não tem tipo anotado, assume str
                    param_type = str
                else:
                    param_type = param.annotation
                
                # Verifica se é opcional
                if param.default != inspect.Parameter.empty:
                    params[param_name] = (Optional[param_type], param.default)
                else:
                    params[param_name] = (param_type, ...)
            
            # Cria modelo Pydantic
            schema = create_model(
                f"{func_name}_params",
                **params
            )
            
            # Registra função e schema
            self.functions[func_name] = func
            self.schemas[func_name] = schema
            
            # Adiciona metadados
            func._function_name = func_name
            func._function_description = func_description
            func._function_schema = schema
            
            return func
            
        return decorator
    
    def get_function(self, name: str) -> Optional[Callable]:
        """
        Obtém uma função registrada.
        
        Args:
            name: Nome da função
            
        Returns:
            Função registrada ou None se não encontrada
        """
        return self.functions.get(name)
    
    def get_schema(self, name: str) -> Optional[Type[BaseModel]]:
        """
        Obtém schema de uma função.
        
        Args:
            name: Nome da função
            
        Returns:
            Schema da função ou None se não encontrada
        """
        return self.schemas.get(name)
    
    def list_functions(self) -> List[Dict[str, Any]]:
        """
        Lista todas as funções registradas.
        
        Returns:
            Lista de dicionários com informações das funções
        """
        functions = []
        
        for name, func in self.functions.items():
            schema = self.schemas[name]
            
            # Obtém schema OpenAPI
            openapi_schema = schema.schema()
            
            functions.append({
                "name": name,
                "description": func._function_description,
                "parameters": {
                    "type": "object",
                    "properties": openapi_schema["properties"],
                    "required": openapi_schema.get("required", [])
                }
            })
        
        return functions

class FunctionCaller:
    """Executa chamadas de função."""
    
    def __init__(self, registry: FunctionRegistry):
        """
        Inicializa o caller.
        
        Args:
            registry: Registro de funções
        """
        self.registry = registry
    
    async def call_function(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Chama uma função.
        
        Args:
            name: Nome da função
            arguments: Argumentos da função
            
        Returns:
            Resultado da função
            
        Raises:
            ValueError: Se a função não existe
            ValidationError: Se os argumentos são inválidos
        """
        # Obtém função
        func = self.registry.get_function(name)
        if func is None:
            raise ValueError(f"Função não encontrada: {name}")
        
        try:
            # Valida argumentos
            schema = self.registry.get_schema(name)
            validated_args = schema(**arguments)
            
            # Chama função
            result = await func(**validated_args.dict())
            
            # Converte resultado para dict
            if isinstance(result, BaseModel):
                result = result.dict()
            elif not isinstance(result, dict):
                result = {"result": result}
            
            return result
            
        except ValidationError as e:
            logger.error(f"Erro de validação na função {name}: {e}")
            ERROR_COUNTER.labels(
                type="function_validation_error",
                model="function_calling"
            ).inc()
            raise
            
        except Exception as e:
            logger.error(f"Erro na execução da função {name}: {e}")
            ERROR_COUNTER.labels(
                type="function_execution_error",
                model="function_calling"
            ).inc()
            raise

# Instância global do registro
function_registry = FunctionRegistry()

# Exemplo de função registrada
@function_registry.register(
    description="Obtém informações sobre o clima de uma cidade"
)
async def get_weather(
    city: str,
    country: Optional[str] = None,
    units: str = "metric"
) -> Dict[str, Any]:
    """
    Obtém informações sobre o clima de uma cidade.
    
    Args:
        city: Nome da cidade
        country: Código do país (opcional)
        units: Unidade de medida (metric/imperial)
        
    Returns:
        Dicionário com informações do clima
    """
    # Aqui seria feita uma chamada real à API de clima
    # Por enquanto retorna dados fake
    return {
        "temperature": 25.0,
        "humidity": 65,
        "description": "Céu limpo",
        "units": units
    }

@function_registry.register(
    description="Traduz um texto para outro idioma"
)
async def translate_text(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None
) -> str:
    """
    Traduz um texto para outro idioma.
    
    Args:
        text: Texto a ser traduzido
        target_lang: Código do idioma alvo
        source_lang: Código do idioma fonte (opcional)
        
    Returns:
        Texto traduzido
    """
    # Aqui seria feita uma chamada real à API de tradução
    # Por enquanto retorna o texto original
    return text

@function_registry.register(
    description="Busca informações sobre um produto"
)
async def search_product(
    query: str,
    category: Optional[str] = None,
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Busca informações sobre produtos.
    
    Args:
        query: Termo de busca
        category: Categoria do produto (opcional)
        max_results: Número máximo de resultados
        
    Returns:
        Lista de produtos encontrados
    """
    # Aqui seria feita uma busca real em uma base de produtos
    # Por enquanto retorna dados fake
    return [
        {
            "name": f"Produto {i}",
            "price": 99.99,
            "category": category or "Geral"
        }
        for i in range(max_results)
    ]

# Instância global do caller
function_caller = FunctionCaller(function_registry) 