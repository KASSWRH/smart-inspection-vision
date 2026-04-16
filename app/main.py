"""
Smart Inspection Vision System — Application Entry Point
=========================================================
FastAPI app with lifespan management, CORS, Prometheus metrics,
and versioned API routing.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import inspection, system
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.detection_service import get_detection_service

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm up model and resources on startup; clean up on shutdown."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    detector = get_detection_service()
    try:
        detector.load_model()
        logger.info("Model warm-up complete")
    except Exception as e:
        logger.warning(f"Model pre-load failed (will retry on first request): {e}")

    yield

    logger.info("Shutting down — releasing resources")


# ── App Instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready AI inspection system for compliance detection, "
        "before/after comparison, and edge deployment. "
        "Supports Arabic and English bilingual responses."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Attach X-Process-Time header to every response."""
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.perf_counter() - start) * 1000:.2f}ms"
    return response


# ── Prometheus Metrics ────────────────────────────────────────────────────────

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ── Routers ───────────────────────────────────────────────────────────────────

prefix = settings.API_V1_PREFIX
app.include_router(inspection.router, prefix=prefix)
app.include_router(system.router, prefix=prefix)


# ── Global Exception Handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }
