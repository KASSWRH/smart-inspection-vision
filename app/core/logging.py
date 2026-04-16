"""
Logging Configuration
=====================
Structured logging with Loguru. Supports JSON format in production
and colored console output in development.
"""

import json
import sys
from pathlib import Path

from loguru import logger

from app.core.config import get_settings

settings = get_settings()


def _json_sink(message: "loguru.Message") -> None:  # type: ignore[name-defined]
    """Custom sink that writes structured JSON logs."""
    record = message.record
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "message": record["message"],
        "function": record["function"],
        "line": record["line"],
    }
    if record["exception"]:
        log_entry["exception"] = str(record["exception"])
    if record.get("extra"):
        log_entry["extra"] = record["extra"]
    print(json.dumps(log_entry, ensure_ascii=False), file=sys.stderr)


def setup_logging() -> None:
    """Configure application logging based on environment."""
    logger.remove()  # Remove default handler

    log_level = settings.LOG_LEVEL.upper()

    if settings.is_production:
        # JSON structured logs for production (parseable by ELK/Grafana)
        logger.add(_json_sink, level=log_level, enqueue=True)
    else:
        # Human-readable colored logs for development
        logger.add(
            sys.stdout,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

    # Always write errors to file
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/error.log",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
    )

    logger.info(
        "Logging initialized",
        environment=settings.ENVIRONMENT,
        level=log_level,
    )


def get_logger(name: str) -> "loguru.Logger":  # type: ignore[name-defined]
    """Return a named logger bound to a specific module."""
    return logger.bind(name=name)
