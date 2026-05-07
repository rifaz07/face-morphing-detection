"""
Unit tests for Module 4 — LBPExtractor.

Tests verify the feature vector properties, LBP image output, and error
handling.  We use the session-scoped face image fixture and derive the
preprocessed float32 array inline so these tests are self-contained.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.ml.exceptions import ImageProcessingError
from app.ml.feature_extractors.lbp_extractor import LBPExtractor, _N_BINS
from app.ml.preprocessors.image_preprocessor import ImagePreprocessor


@pytest.fixture(scope="module")
def extractor() -> LBPExtractor:
    return LBPExtractor()


@pytest.fixture(scope="module")
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor()


@pytest.fixture(scope="module")
def face_array(valid_face_image_bytes: bytes, preprocessor: ImagePreprocessor) -> np.ndarray:
    """Float32 128×128 preprocessed face array."""
    return preprocessor.get_normalized_array(valid_face_image_bytes)


@pytest.fixture(scope="module")
def noise_array(random_noise_image_bytes: bytes, preprocessor: ImagePreprocessor) -> np.ndarray:
    """Float32 128×128 preprocessed noise array (no face)."""
    return preprocessor.get_normalized_array(random_noise_image_bytes)


# ---------------------------------------------------------------------------
# Feature vector properties
# ---------------------------------------------------------------------------


def test_feature_vector_length_is_59(extractor: LBPExtractor, face_array: np.ndarray) -> None:
    result = extractor.extract(face_array)
    assert result.feature_vector_length == 59
    assert len(result.feature_vector) == 59
    assert _N_BINS == 59


def test_feature_vector_sums_to_1(extractor: LBPExtractor, face_array: np.ndarray) -> None:
    """Normalised histogram must sum to 1.0 (within floating-point tolerance)."""
    result = extractor.extract(face_array)
    total = sum(result.feature_vector)
    assert abs(total - 1.0) < 1e-6, f"Feature vector sums to {total}, expected 1.0"


def test_feature_vector_all_non_negative(
    extractor: LBPExtractor, face_array: np.ndarray
) -> None:
    result = extractor.extract(face_array)
    assert all(v >= 0.0 for v in result.feature_vector)


# ---------------------------------------------------------------------------
# LBP image
# ---------------------------------------------------------------------------


def test_lbp_image_same_shape_as_input(
    extractor: LBPExtractor, face_array: np.ndarray
) -> None:
    """The LBP visualisation image should be 128×128 (matching input shape)."""
    result = extractor.extract(face_array)
    raw = base64.b64decode(result.lbp_image_b64)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (128, 128), f"Expected (128, 128), got {img.size}"


def test_lbp_b64_is_valid_jpeg(extractor: LBPExtractor, face_array: np.ndarray) -> None:
    result = extractor.extract(face_array)
    raw = base64.b64decode(result.lbp_image_b64)
    assert raw[:3] == b"\xff\xd8\xff", "lbp_image_b64 is not a valid JPEG"


# ---------------------------------------------------------------------------
# Discriminative power
# ---------------------------------------------------------------------------


def test_different_images_give_different_vectors(
    extractor: LBPExtractor,
    face_array: np.ndarray,
    noise_array: np.ndarray,
) -> None:
    """
    A face image and random noise must produce meaningfully different LBP
    histograms.  We measure the L1 distance between the two vectors and
    require it to be non-trivial (> 0.01 out of a max of 2.0).
    """
    face_result = extractor.extract(face_array)
    noise_result = extractor.extract(noise_array)

    fv_face = np.array(face_result.feature_vector)
    fv_noise = np.array(noise_result.feature_vector)
    l1_distance = float(np.sum(np.abs(fv_face - fv_noise)))

    assert l1_distance > 0.01, (
        f"Expected L1 distance > 0.01 between face and noise vectors, got {l1_distance:.4f}"
    )


# ---------------------------------------------------------------------------
# Histogram stats
# ---------------------------------------------------------------------------


def test_histogram_stats_are_correct(
    extractor: LBPExtractor, face_array: np.ndarray
) -> None:
    result = extractor.extract(face_array)
    stats = result.histogram_stats
    for key in ("mean", "std", "max_bin", "min_bin", "max_value", "min_value"):
        assert key in stats, f"Missing key '{key}' in histogram_stats"
    assert 0 <= stats["max_bin"] < 59
    assert 0 <= stats["min_bin"] < 59
    assert stats["max_value"] >= stats["min_value"]
    assert stats["mean"] > 0


# ---------------------------------------------------------------------------
# Extract from preprocessed array
# ---------------------------------------------------------------------------


def test_extract_from_preprocessed_array_works(
    extractor: LBPExtractor, face_array: np.ndarray
) -> None:
    result = extractor.extract(face_array)
    assert result.method == "uniform"
    assert result.radius == 1
    assert result.n_points == 8
    assert result.processing_time_ms >= 0


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_result_method_and_params(
    extractor: LBPExtractor, face_array: np.ndarray
) -> None:
    result = extractor.extract(face_array)
    assert result.method == "uniform"
    assert result.radius == 1
    assert result.n_points == 8


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_wrong_dimensions_raises_error(extractor: LBPExtractor) -> None:
    """3-D array (colour image) should be rejected."""
    bad = np.zeros((128, 128, 3), dtype=np.float32)
    with pytest.raises(ImageProcessingError):
        extractor.extract(bad)


def test_wrong_dtype_raises_error(extractor: LBPExtractor) -> None:
    """uint8 array without normalisation should be rejected."""
    bad = np.zeros((128, 128), dtype=np.uint8)
    with pytest.raises(ImageProcessingError):
        extractor.extract(bad)
