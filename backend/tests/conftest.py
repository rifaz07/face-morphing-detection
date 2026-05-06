import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Synchronous TestClient — no real DB required for unit tests."""
    with TestClient(app) as c:
        yield c
