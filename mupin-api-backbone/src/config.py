import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://mupin:mupin@postgres:5432/mupin"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
BACKBONE_PORT = int(os.environ.get("BACKBONE_PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")
