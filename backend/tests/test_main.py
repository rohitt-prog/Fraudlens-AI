"""Tests for backend API entry points and health checks."""

from fastapi import status
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    """Verifies that the API healthcheck returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "status": "healthy",
        "service": "fraudlens-backend",
    }
