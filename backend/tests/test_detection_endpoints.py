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
    """Response must include all documented fields including preprocessed_faces."""
    resp = client.post(
        "/api/v1/detection/detect-face",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    required_fields = {
        "face_count", "faces", "largest_face",
        "cropped_faces_b64", "processing_time_ms", "image_dimensions",
        "preprocessed_faces",
    }
    assert required_fields.issubset(body.keys())
    assert body["image_dimensions"]["width"] > 0
    assert body["image_dimensions"]["height"] > 0
    assert isinstance(body["preprocessed_faces"], list)
    assert len(body["preprocessed_faces"]) == body["face_count"]


def test_detect_face_preprocessed_faces_structure(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    """Each preprocessed face must have the expected Module 3 fields."""
    resp = client.post(
        "/api/v1/detection/detect-face",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    for pf in body["preprocessed_faces"]:
        assert pf["numpy_array_shape"] == [128, 128]
        assert pf["steps_applied"] == [
            "resize_128x128", "grayscale",
            "histogram_equalization", "normalize_0_1",
        ]
        assert 0.0 <= pf["normalized_stats"]["min"]
        assert pf["normalized_stats"]["max"] <= 1.0
        assert "preprocessed_b64" in pf


# ---------------------------------------------------------------------------
# POST /api/v1/detection/preprocess
# ---------------------------------------------------------------------------


def test_preprocess_endpoint_returns_200(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/preprocess",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "detection" in body
    assert "preprocessing" in body


def test_preprocess_endpoint_structure(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    """When a face is found, preprocessing must not be null."""
    resp = client.post(
        "/api/v1/detection/preprocess",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    det = body["detection"]
    assert "face_count" in det
    assert "faces" in det
    # If a face was detected, preprocessing should be populated
    if det["face_count"] > 0:
        assert body["preprocessing"] is not None
        pp = body["preprocessing"]
        assert pp["numpy_array_shape"] == [128, 128]
        assert "preprocessed_b64" in pp


def test_preprocess_endpoint_no_face_returns_null_preprocessing(
    client: TestClient,
    random_noise_image_bytes: bytes,
) -> None:
    """No face in image → preprocessing field must be null, not an error."""
    resp = client.post(
        "/api/v1/detection/preprocess",
        files=[_to_upload(random_noise_image_bytes, "noise.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection"]["face_count"] == 0
    assert body["preprocessing"] is None


def test_preprocess_endpoint_returns_400_for_invalid(
    client: TestClient,
    invalid_format_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/preprocess",
        files=[_to_upload(invalid_format_bytes, "bad.txt", "text/plain")],
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/v1/detection/extract-lbp
# ---------------------------------------------------------------------------


def test_extract_lbp_endpoint_returns_200(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/extract-lbp",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "detection" in body
    assert "lbp" in body


def test_extract_lbp_endpoint_feature_vector_structure(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    """When a face is found, lbp field must contain a 59-element normalised vector."""
    resp = client.post(
        "/api/v1/detection/extract-lbp",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    if body["detection"]["face_count"] > 0:
        lbp = body["lbp"]
        assert lbp is not None
        assert lbp["feature_vector_length"] == 59
        assert len(lbp["feature_vector"]) == 59
        assert abs(sum(lbp["feature_vector"]) - 1.0) < 1e-5
        assert lbp["method"] == "uniform"
        assert "lbp_image_b64" in lbp
        assert "histogram_stats" in lbp


def test_extract_lbp_no_face_returns_null_lbp(
    client: TestClient,
    random_noise_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/extract-lbp",
        files=[_to_upload(random_noise_image_bytes, "noise.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection"]["face_count"] == 0
    assert body["lbp"] is None


def test_extract_lbp_returns_400_for_invalid_image(
    client: TestClient,
    invalid_format_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/extract-lbp",
        files=[_to_upload(invalid_format_bytes, "bad.txt", "text/plain")],
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/v1/detection/extract-dct
# ---------------------------------------------------------------------------


def test_extract_dct_endpoint_returns_200(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/extract-dct",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "detection" in body
    assert "dct" in body


def test_extract_dct_feature_vector_structure(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    """When a face is found, dct must contain a 1024-element log-compressed vector."""
    resp = client.post(
        "/api/v1/detection/extract-dct",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    if body["detection"]["face_count"] > 0:
        dct = body["dct"]
        assert dct is not None
        assert dct["feature_vector_length"] == 1024
        assert len(dct["feature_vector"]) == 1024
        assert dct["dct_size"] == 32
        assert dct["normalization"] == "log_compression"
        assert "dct_image_b64" in dct
        assert "dct_block_stats" in dct


def test_extract_dct_no_face_returns_null(
    client: TestClient,
    random_noise_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/extract-dct",
        files=[_to_upload(random_noise_image_bytes, "noise.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection"]["face_count"] == 0
    assert body["dct"] is None


def test_extract_dct_returns_400_for_invalid_image(
    client: TestClient,
    invalid_format_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/extract-dct",
        files=[_to_upload(invalid_format_bytes, "bad.txt", "text/plain")],
    )
    assert resp.status_code == 400
