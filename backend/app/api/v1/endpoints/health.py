from fastapi import APIRouter

from app.core.config import settings
from app.core.database import check_db_connection
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the API status and verifies database connectivity.",
    tags=["Health"],
)
def health_check() -> HealthResponse:
    """
    Lightweight health probe used by Docker, load balancers, and monitoring tools.

    - **status**: `ok` when all subsystems are healthy, `degraded` otherwise.
    - **database**: result of a `SELECT 1` against PostgreSQL.
    - **version**: current API version from settings.
    - **environment**: active deployment environment.
    """
    db_ok = check_db_connection()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database="connected" if db_ok else "disconnected",
        version=settings.VERSION,
        environment=settings.ENV,
    )
