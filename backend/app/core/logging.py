import sys

from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    """Configure loguru with environment-appropriate formatting."""
    logger.remove()  # Remove default stderr handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=log_format,
        level=settings.LOG_LEVEL.upper(),
        colorize=not settings.is_production,
        backtrace=not settings.is_production,
        diagnose=not settings.is_production,
    )

    if settings.is_production:
        logger.add(
            "logs/app.log",
            format=log_format,
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            colorize=False,
        )
