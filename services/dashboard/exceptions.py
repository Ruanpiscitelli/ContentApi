"""
Este módulo contém as exceções personalizadas utilizadas pelo dashboard.
"""

from typing import Any, Dict, Optional

class DashboardException(Exception):
    """
    Exceção base para todas as exceções do dashboard.
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa uma nova exceção do dashboard.
        
        Args:
            message (str): Mensagem de erro.
            status_code (int): Código de status HTTP.
            error_code (str): Código de erro interno.
            details (Optional[Dict[str, Any]]): Detalhes adicionais do erro.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}

class AuthenticationError(DashboardException):
    """
    Exceção para erros de autenticação.
    """
    
    def __init__(
        self,
        message: str = "Erro de autenticação",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa uma nova exceção de autenticação.
        
        Args:
            message (str): Mensagem de erro.
            details (Optional[Dict[str, Any]]): Detalhes adicionais do erro.
        """
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )

class AuthorizationError(DashboardException):
    """
    Exceção para erros de autorização.
    """
    
    def __init__(
        self,
        message: str = "Acesso não autorizado",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa uma nova exceção de autorização.
        
        Args:
            message (str): Mensagem de erro.
            details (Optional[Dict[str, Any]]): Detalhes adicionais do erro.
        """
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            details=details
        )

class ValidationError(DashboardException):
    """
    Exceção para erros de validação.
    """
    
    def __init__(
        self,
        message: str = "Erro de validação",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa uma nova exceção de validação.
        
        Args:
            message (str): Mensagem de erro.
            details (Optional[Dict[str, Any]]): Detalhes adicionais do erro.
        """
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )

class ResourceNotFoundError(DashboardException):
    """
    Exceção para recursos não encontrados.
    """
    
    def __init__(
        self,
        message: str = "Recurso não encontrado",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa uma nova exceção de recurso não encontrado.
        
        Args:
            message (str): Mensagem de erro.
            details (Optional[Dict[str, Any]]): Detalhes adicionais do erro.
        """
        super().__init__(
            message=message,
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            details=details
        )

class RateLimitError(DashboardException):
    """
    Exceção para limite de taxa excedido.
    """
    
    def __init__(
        self,
        message: str = "Limite de taxa excedido",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa uma nova exceção de limite de taxa.
        
        Args:
            message (str): Mensagem de erro.
            details (Optional[Dict[str, Any]]): Detalhes adicionais do erro.
        """
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )

class DatabaseError(DashboardException):
    """
    Exceção para erros de banco de dados.
    """
    
    def __init__(
        self,
        message: str = "Erro no banco de dados",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa uma nova exceção de banco de dados.
        
        Args:
            message (str): Mensagem de erro.
            details (Optional[Dict[str, Any]]): Detalhes adicionais do erro.
        """
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details=details
        )

class ConfigurationError(DashboardException):
    """
    Exceção para erros de configuração.
    """
    
    def __init__(
        self,
        message: str = "Erro de configuração",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa uma nova exceção de configuração.
        
        Args:
            message (str): Mensagem de erro.
            details (Optional[Dict[str, Any]]): Detalhes adicionais do erro.
        """
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details=details
        ) 