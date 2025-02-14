"""
Router de autenticação.
"""
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..models.database import User
from ..models.database_config import get_db
from ..security import (
    authenticate_user,
    create_access_token,
    get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()
templates = Jinja2Templates(directory="services/dashboard/templates")

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Página de login."""
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Processa login."""
    user = await authenticate_user(db, username, password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Usuário ou senha incorretos"
            },
            status_code=400
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True
    )
    return response

@router.get("/logout")
async def logout():
    """Realiza logout."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="access_token")
    return response

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Página de registro."""
    return templates.TemplateResponse(
        "register.html",
        {"request": request}
    )

@router.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Processa registro."""
    # Verifica se usuário já existe
    result = await db.execute(
        select(User).where(User.username == username)
    )
    if result.scalar_one_or_none():
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "Usuário já existe"
            },
            status_code=400
        )
    
    # Verifica se email já existe
    result = await db.execute(
        select(User).where(User.email == email)
    )
    if result.scalar_one_or_none():
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "Email já cadastrado"
            },
            status_code=400
        )
    
    # Cria novo usuário
    user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password)
    )
    db.add(user)
    await db.commit()
    
    # Redireciona para login
    return RedirectResponse(url="/login", status_code=303) 