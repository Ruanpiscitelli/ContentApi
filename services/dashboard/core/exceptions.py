"""
Exceções personalizadas e handlers globais.
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

class DashboardException(HTTPException):
    """
    Exceção base para todas as exceções do dashboard.
    """
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)

class AuthenticationError(DashboardException):
    """
    Erro de autenticação (credenciais inválidas, token expirado, etc).
    """
    def __init__(self, detail: str = "Erro de autenticação"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

class AuthorizationError(DashboardException):
    """
    Erro de autorização (permissões insuficientes).
    """
    def __init__(self, detail: str = "Acesso não autorizado"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

class ValidationError(DashboardException):
    """
    Erro de validação de dados.
    """
    def __init__(self, detail: str = "Erro de validação"):
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

class ResourceNotFoundError(DashboardException):
    """
    Recurso não encontrado.
    """
    def __init__(self, detail: str = "Recurso não encontrado"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

async def dashboard_exception_handler(request: Request, exc: DashboardException):
    """
    Handler global para exceções do dashboard.
    Formata a resposta de erro de forma consistente.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": exc.__class__.__name__
            }
        }
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handler global para exceções HTTP.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "HTTPException"
            }
        }
    ) 