"""
Integration tests for the Detection API endpoints.

These tests use FastAPI's TestClient (WSGI, synchronous) and hit the real
validator + detector without mocking, so they also serve as integration smoke
tests for Modules 1 and 2.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_upload(data: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
    return ("file", (filename, io.BytesIO(data), content_type))


# ---------------------------------------------------------------------------
# POST /api/v1/detection/validate
# ---------------------------------------------------------------------------


def test_validate_endpoint_returns_200_for_valid(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/validate",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_valid"] is True
    assert body["errors"] == []
    assert "width" in body["metadata"]


def test_validate_endpoint_returns_400_for_invalid_format(
    client: TestClient,
    invalid_format_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/validate",
        files=[_to_upload(invalid_format_bytes, "file.txt", "text/plain")],
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "detail" in body


def test_validate_endpoint_returns_400_for_tiny_image(
    client: TestClient,
    tiny_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/validate",
        files=[_to_upload(tiny_image_bytes, "tiny.jpg")],
    )
    assert resp.status_code == 400


def test_validate_endpoint_returns_warnings_for_grayscale(
    client: TestClient,
    grayscale_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/validate",
        files=[_to_upload(grayscale_image_bytes, "gray.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_valid"] is True
    assert len(body["warnings"]) > 0


# ---------------------------------------------------------------------------
# POST /api/v1/detection/detect-face
# ---------------------------------------------------------------------------


def test_detect_face_endpoint_returns_200_with_face(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/detect-face",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "face_count" in body
    assert "faces" in body
    assert "cropped_faces_b64" in body
    assert "processing_time_ms" in body
    assert body["face_count"] == len(body["faces"])
    assert body["face_count"] == len(body["cropped_faces_b64"])


def test_detect_face_endpoint_returns_200_with_no_face_count_zero(
    client: TestClient,
    random_noise_image_bytes: bytes,
) -> None:
    """No face in image must return 200 with face_count=0, not a 4xx error."""
    resp = client.post(
        "/api/v1/detection/detect-face",
        files=[_to_upload(random_noise_image_bytes, "noise.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["face_count"] == 0
    assert body["faces"] == []
    assert body["largest_face"] is None


def test_detect_face_endpoint_returns_400_for_invalid_image(
    client: TestClient,
    invalid_format_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/detect-face",
        files=[_to_upload(invalid_format_bytes, "bad.txt", "text/plain")],
    )
    assert resp.status_code == 400


def test_detect_face_endpoint_structure(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    """Response must include all documented fields."""
    resp = client.post(
        "/api/v1/detection/detect-face",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    required_fields = {
        "face_count", "faces", "largest_face",
        "cropped_faces_b64", "processing_time_ms", "image_dimensions",
    }
    assert required_fields.issubset(body.keys())
    assert body["image_dimensions"]["width"] > 0
    assert body["image_dimensions"]["height"] > 0
