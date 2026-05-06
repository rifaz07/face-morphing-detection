from typing import Generator

from sqlalchemy.orm import Session

from app.core.database import get_db

# Re-export get_db so all endpoints import from a single location.
__all__ = ["get_db"]
