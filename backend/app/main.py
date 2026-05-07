from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import check_db_connection
from app.core.logging import setup_logging
from app.ml.clustering.kmeans_classifier import KMeansClassifier
from app.ml.evaluation.evaluator import ModelEvaluator

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENV}]")
    if check_db_connection():
        logger.info("Database connection: OK")
    else:
        logger.warning("Database connection: FAILED — running without DB")

    # Module 7 — initialise K-Means classifier (auto-loads saved model or trains on synthetic data).
    classifier = KMeansClassifier()
    app.state.classifier = classifier
    info = classifier.get_model_info()
    if info.trained_on_synthetic:
        logger.info("K-Means model trained on synthetic data | inertia={}", info.inertia)
    else:
        logger.info("K-Means model loaded from disk | inertia={}", info.inertia)

    # Module 8 — run startup evaluation and cache the report.
    evaluator = ModelEvaluator(classifier)
    app.state.evaluator = evaluator
    try:
        report = evaluator.generate_report()
        app.state.evaluation_report = report
        m = report.metrics
        logger.info(
            "Startup evaluation complete | acc={:.3f} FAR={:.3f} FRR={:.3f} F1={:.3f}",
            m.accuracy, m.far, m.frr, m.f1_score,
        )
    except Exception as exc:
        logger.warning("Startup evaluation failed (non-fatal): {}", exc)
        app.state.evaluation_report = None

    yield
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-grade face morphing detection API. "
        "Uses LBP + DCT feature extraction with K-Means clustering to "
        "classify uploaded face images as REAL or MORPHED."
    ),
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "Service health and readiness checks."},
        {"name": "Detection", "description": "Face morphing detection endpoints."},
    ],
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# ---------------------------------------------------------------------------
# Custom error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "The requested resource was not found."},
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse({"message": f"Welcome to {settings.PROJECT_NAME}. Visit /docs for API reference."})
