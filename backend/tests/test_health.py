from unittest.mock import patch

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    """Health endpoint returns 200 when DB is reachable."""
    with patch("app.api.v1.endpoints.health.check_db_connection", return_value=True):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"
    assert "version" in body
    assert "environment" in body


def test_health_degraded(client: TestClient) -> None:
    """Health endpoint returns 200 with degraded status when DB is unreachable."""
    with patch("app.api.v1.endpoints.health.check_db_connection", return_value=False):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "disconnected"


def test_root(client: TestClient) -> None:
    """Root endpoint returns a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_docs_accessible(client: TestClient) -> None:
    """Swagger UI is served at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200
