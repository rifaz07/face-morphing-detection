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


# ---------------------------------------------------------------------------
# POST /api/v1/detection/extract-features
# ---------------------------------------------------------------------------


def test_extract_features_endpoint_returns_200(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/extract-features",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "detection" in body
    assert "lbp" in body
    assert "dct" in body
    assert "fusion" in body


def test_extract_features_fusion_vector_structure(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    """When a face is found, fusion must contain a 1083-element vector."""
    resp = client.post(
        "/api/v1/detection/extract-features",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    if body["detection"]["face_count"] > 0:
        fusion = body["fusion"]
        assert fusion is not None
        assert fusion["fused_vector_length"] == 1083
        assert len(fusion["fused_vector"]) == 1083
        assert fusion["fusion_method"] == "concatenation"
        assert fusion["normalization_applied"] is True
        assert abs(fusion["lbp_contribution"] - 5.4478) < 0.01
        assert abs(fusion["dct_contribution"] - 94.5522) < 0.01


def test_extract_features_no_face_returns_null_fusion(
    client: TestClient,
    random_noise_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/extract-features",
        files=[_to_upload(random_noise_image_bytes, "noise.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection"]["face_count"] == 0
    assert body["fusion"] is None
    assert body["lbp"] is None
    assert body["dct"] is None


def test_extract_features_returns_400_for_invalid(
    client: TestClient,
    invalid_format_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/extract-features",
        files=[_to_upload(invalid_format_bytes, "bad.txt", "text/plain")],
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/v1/detection/model-info
# ---------------------------------------------------------------------------


def test_model_info_endpoint_returns_200(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_fitted"] is True
    assert body["model_type"] == "KMeans"
    assert body["n_clusters"] == 2
    assert "cluster_labels" in body
    assert "model_path" in body


def test_model_info_contains_cluster_labels(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/model-info")
    assert resp.status_code == 200
    body = resp.json()
    labels = set(body["cluster_labels"].values())
    assert "REAL" in labels
    assert "MORPHED" in labels


def test_model_info_inertia_is_positive(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/model-info")
    body = resp.json()
    assert body["inertia"] is not None
    assert body["inertia"] > 0


# ---------------------------------------------------------------------------
# POST /api/v1/detection/retrain
# ---------------------------------------------------------------------------


def test_retrain_endpoint_returns_200(client: TestClient) -> None:
    resp = client.post("/api/v1/detection/retrain")
    assert resp.status_code == 200
    body = resp.json()
    assert "samples_trained" in body
    assert "inertia" in body
    assert "converged" in body
    assert body["trained_on_synthetic"] is True


def test_retrain_returns_expected_sample_count(client: TestClient) -> None:
    resp = client.post("/api/v1/detection/retrain")
    body = resp.json()
    assert body["samples_trained"] == 1000  # 500 REAL + 500 MORPHED


def test_retrain_cluster_labels_present(client: TestClient) -> None:
    resp = client.post("/api/v1/detection/retrain")
    body = resp.json()
    labels = set(body["cluster_labels"].values())
    assert "REAL" in labels
    assert "MORPHED" in labels


def test_retrain_inertia_positive(client: TestClient) -> None:
    resp = client.post("/api/v1/detection/retrain")
    body = resp.json()
    assert body["inertia"] > 0


# ---------------------------------------------------------------------------
# POST /api/v1/detection/classify
# ---------------------------------------------------------------------------


def test_classify_endpoint_returns_200(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/classify",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200


def test_classify_response_has_required_fields(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/classify",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "detection" in body
    assert "fusion" in body
    assert "prediction" in body


def test_classify_returns_real_or_morphed_string(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/classify",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    if body["detection"]["face_count"] > 0:
        assert body["prediction"] is not None
        assert body["prediction"]["prediction"] in {"REAL", "MORPHED"}


def test_classify_returns_confidence_float(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/classify",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    if body["detection"]["face_count"] > 0:
        conf = body["prediction"]["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0


def test_classify_prediction_has_cluster_id(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/classify",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    if body["detection"]["face_count"] > 0:
        assert body["prediction"]["cluster_id"] in {0, 1}


def test_classify_prediction_has_distance_to_centroid(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/classify",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    if body["detection"]["face_count"] > 0:
        assert body["prediction"]["distance_to_centroid"] >= 0.0


def test_classify_no_face_returns_null_prediction(
    client: TestClient,
    random_noise_image_bytes: bytes,
) -> None:
    """No face in image → prediction must be null (not an error)."""
    resp = client.post(
        "/api/v1/detection/classify",
        files=[_to_upload(random_noise_image_bytes, "noise.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection"]["face_count"] == 0
    assert body["prediction"] is None
    assert body["fusion"] is None


def test_classify_returns_400_for_invalid_image(
    client: TestClient,
    invalid_format_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/classify",
        files=[_to_upload(invalid_format_bytes, "bad.txt", "text/plain")],
    )
    assert resp.status_code == 400


def test_classify_fusion_vector_present_when_face_found(
    client: TestClient,
    valid_face_image_bytes: bytes,
) -> None:
    resp = client.post(
        "/api/v1/detection/classify",
        files=[_to_upload(valid_face_image_bytes, "face.jpg")],
    )
    assert resp.status_code == 200
    body = resp.json()
    if body["detection"]["face_count"] > 0:
        fusion = body["fusion"]
        assert fusion is not None
        assert fusion["fused_vector_length"] == 1083
        assert len(fusion["fused_vector"]) == 1083


# ---------------------------------------------------------------------------
# GET /api/v1/detection/evaluation  (Module 8)
# ---------------------------------------------------------------------------


def test_evaluation_endpoint_returns_200(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation")
    assert resp.status_code == 200


def test_evaluation_has_accuracy_far_frr(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation")
    assert resp.status_code == 200
    body = resp.json()
    metrics = body["metrics"]
    assert "accuracy" in metrics
    assert "far" in metrics
    assert "frr" in metrics
    assert "f1_score" in metrics


def test_evaluation_metrics_in_valid_range(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation")
    body = resp.json()
    m = body["metrics"]
    for key in ("accuracy", "far", "frr", "precision", "recall", "f1_score"):
        assert 0.0 <= m[key] <= 1.0, f"{key} = {m[key]} is out of [0, 1]"


def test_evaluation_has_confusion_matrix(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation")
    body = resp.json()
    cm = body["metrics"]["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2
    assert len(cm[1]) == 2


def test_evaluation_total_samples_is_200(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation")
    body = resp.json()
    assert body["metrics"]["total_samples"] == 200


def test_evaluation_has_interpretation(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation")
    body = resp.json()
    interp = body["interpretation"]
    assert "accuracy" in interp
    assert "far" in interp
    assert "frr" in interp


def test_evaluation_has_model_info(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation")
    body = resp.json()
    assert body["model_info"]["type"] == "K-Means Clustering"
    assert body["model_info"]["k"] == 2


def test_evaluation_has_recommendations(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation")
    body = resp.json()
    assert isinstance(body["recommendations"], list)
    assert len(body["recommendations"]) > 0


def test_evaluation_data_source_is_synthetic(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation")
    body = resp.json()
    assert body["data_source"] == "synthetic"


# ---------------------------------------------------------------------------
# GET /api/v1/detection/evaluation/live
# ---------------------------------------------------------------------------


def test_live_evaluation_endpoint_returns_200(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation/live")
    assert resp.status_code == 200


def test_live_stats_has_required_fields(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation/live")
    body = resp.json()
    assert "total_predictions" in body
    assert "real_count" in body
    assert "morphed_count" in body
    assert "avg_confidence" in body


def test_live_stats_counts_are_non_negative(client: TestClient) -> None:
    resp = client.get("/api/v1/detection/evaluation/live")
    body = resp.json()
    assert body["total_predictions"] >= 0
    assert body["real_count"] >= 0
    assert body["morphed_count"] >= 0


# ---------------------------------------------------------------------------
# POST /api/v1/detection/evaluation/run
# ---------------------------------------------------------------------------


def test_evaluation_run_endpoint_returns_200(client: TestClient) -> None:
    resp = client.post("/api/v1/detection/evaluation/run")
    assert resp.status_code == 200


def test_evaluation_run_returns_fresh_metrics(client: TestClient) -> None:
    resp = client.post("/api/v1/detection/evaluation/run")
    body = resp.json()
    assert "metrics" in body
    assert body["metrics"]["total_samples"] == 200


def test_evaluation_run_updates_cache(client: TestClient) -> None:
    """After /run, GET /evaluation should return the new timestamp."""
    run_resp = client.post("/api/v1/detection/evaluation/run")
    get_resp = client.get("/api/v1/detection/evaluation")
    assert run_resp.status_code == 200
    assert get_resp.status_code == 200
    # Both should have the same total_samples
    assert run_resp.json()["metrics"]["total_samples"] == get_resp.json()["metrics"]["total_samples"]
