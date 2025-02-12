from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./dashboard.db"
    # Restante das configurações... 