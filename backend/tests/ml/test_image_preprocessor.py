"""
Unit tests for Module 3 — ImagePreprocessor.

Tests cover all four pipeline steps (resize, grayscale, histogram
equalisation, normalisation) plus both input methods (bytes and base64).
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.ml.exceptions import ImageProcessingError
from app.ml.preprocessors.image_preprocessor import ImagePreprocessor


@pytest.fixture(scope="module")
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor()


@pytest.fixture(scope="module")
def face_bytes(valid_face_image_bytes: bytes) -> bytes:
    """Re-expose the session fixture under a shorter name."""
    return valid_face_image_bytes


@pytest.fixture(scope="module")
def face_b64(valid_face_image_bytes: bytes) -> str:
    return base64.b64encode(valid_face_image_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# Output shape / size
# ---------------------------------------------------------------------------


def test_output_is_128x128(preprocessor: ImagePreprocessor, face_bytes: bytes) -> None:
    result = preprocessor.preprocess_bytes(face_bytes)
    assert result.numpy_array_shape == [128, 128], (
        f"Expected [128, 128], got {result.numpy_array_shape}"
    )


def test_output_is_grayscale(preprocessor: ImagePreprocessor, face_bytes: bytes) -> None:
    """Shape must be 2-D (no channel dimension) — confirming single-channel grayscale."""
    result = preprocessor.preprocess_bytes(face_bytes)
    assert len(result.numpy_array_shape) == 2, (
        "Expected 2-D shape for grayscale, got shape with "
        f"{len(result.numpy_array_shape)} dimensions"
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_pixel_values_normalized_0_to_1(
    preprocessor: ImagePreprocessor, face_bytes: bytes
) -> None:
    """normalised_stats must report values within [0.0, 1.0]."""
    result = preprocessor.preprocess_bytes(face_bytes)
    assert result.normalized_stats["min"] >= 0.0
    assert result.normalized_stats["max"] <= 1.0


def test_normalized_stats_populated(
    preprocessor: ImagePreprocessor, face_bytes: bytes
) -> None:
    result = preprocessor.preprocess_bytes(face_bytes)
    stats = result.normalized_stats
    for key in ("min", "max", "mean", "std"):
        assert key in stats
        assert isinstance(stats[key], float)


# ---------------------------------------------------------------------------
# Histogram equalisation
# ---------------------------------------------------------------------------


def test_histogram_equalization_changes_pixel_distribution(
    preprocessor: ImagePreprocessor,
) -> None:
    """
    Verify that histogram equalisation actually shifts the pixel distribution.

    Strategy: build a 200×200 image whose pixels are random noise in [0, 50]
    (dark, but non-constant — important: a flat constant image is a degenerate
    case for equalizeHist and won't spread the histogram).  After equalisation
    the values should be stretched to span [0, 255], pushing the mean to near
    0.5 normalised.  We use a lower bound of 0.35 to avoid flakiness from
    JPEG compression artifacts while still proving the shift occurred.
    """
    rng = np.random.default_rng(seed=7)
    # Dark image: pixel values uniformly distributed in [0, 50]
    dark_arr = rng.integers(0, 51, (200, 200, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(dark_arr, "RGB").save(buf, format="JPEG", quality=95)
    dark_bytes = buf.getvalue()

    result = preprocessor.preprocess_bytes(dark_bytes)
    # Before equalisation mean ≈ 25/255 ≈ 0.098.
    # After equalisation the histogram is spread to [0, 255] → mean ≈ 0.5.
    assert result.normalized_stats["mean"] > 0.35, (
        f"Expected mean > 0.35 after histogram equalization of a dark image, "
        f"got {result.normalized_stats['mean']}"
    )


# ---------------------------------------------------------------------------
# Steps applied
# ---------------------------------------------------------------------------


def test_steps_applied_list_is_correct(
    preprocessor: ImagePreprocessor, face_bytes: bytes
) -> None:
    result = preprocessor.preprocess_bytes(face_bytes)
    expected = ["resize_128x128", "grayscale", "histogram_equalization", "normalize_0_1"]
    assert result.steps_applied == expected


# ---------------------------------------------------------------------------
# Both input methods
# ---------------------------------------------------------------------------


def test_preprocess_from_bytes_works(
    preprocessor: ImagePreprocessor, face_bytes: bytes
) -> None:
    result = preprocessor.preprocess_bytes(face_bytes)
    assert result.numpy_array_shape == [128, 128]
    assert result.processing_time_ms >= 0


def test_preprocess_from_b64_works(
    preprocessor: ImagePreprocessor, face_b64: str
) -> None:
    result = preprocessor.preprocess(face_b64)
    assert result.numpy_array_shape == [128, 128]
    assert result.processing_time_ms >= 0


# ---------------------------------------------------------------------------
# Output JPEG is valid
# ---------------------------------------------------------------------------


def test_preprocessed_b64_is_valid_jpeg(
    preprocessor: ImagePreprocessor, face_bytes: bytes
) -> None:
    result = preprocessor.preprocess_bytes(face_bytes)
    raw = base64.b64decode(result.preprocessed_b64)
    assert raw[:3] == b"\xff\xd8\xff", "preprocessed_b64 is not a valid JPEG"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_b64_raises_error(preprocessor: ImagePreprocessor) -> None:
    with pytest.raises(ImageProcessingError):
        preprocessor.preprocess("this-is-not-valid-base64!!!")


def test_invalid_bytes_raises_error(preprocessor: ImagePreprocessor) -> None:
    with pytest.raises(ImageProcessingError):
        preprocessor.preprocess_bytes(b"not an image")
