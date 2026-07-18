"""Main FastAPI application entry point.

This module initializes the FastAPI app, configures middlewares, setting routes,
and manages startup/shutdown lifespans.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, status


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown events."""
    # Startup tasks (e.g. loading model weights, database connection setup)
    yield
    # Shutdown tasks (e.g. closing database connections, flushing loggers)


# Initialize FastAPI app
app = FastAPI(
    title="FraudLens AI - API Service",
    description=(
        "REST API service for real-time transaction fraud detection "
        "scoring and explainability."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check() -> dict[str, str]:
    """Performs a simple health check.

    Returns:
        dict: A status check indicating whether the service is healthy.
    """
    return {"status": "healthy", "service": "fraudlens-backend"}
