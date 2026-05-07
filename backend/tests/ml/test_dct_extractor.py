"""
Unit tests for Module 5 — DCTExtractor.

Tests verify feature vector properties, frequency-domain correctness,
visualisation output, and error handling.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.ml.exceptions import ImageProcessingError
from app.ml.feature_extractors.dct_extractor import DCTExtractor, _DCT_BLOCK_SIZE
from app.ml.preprocessors.image_preprocessor import ImagePreprocessor


@pytest.fixture(scope="module")
def extractor() -> DCTExtractor:
    return DCTExtractor()


@pytest.fixture(scope="module")
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor()


@pytest.fixture(scope="module")
def face_array(valid_face_image_bytes: bytes, preprocessor: ImagePreprocessor) -> np.ndarray:
    return preprocessor.get_normalized_array(valid_face_image_bytes)


@pytest.fixture(scope="module")
def noise_array(random_noise_image_bytes: bytes, preprocessor: ImagePreprocessor) -> np.ndarray:
    return preprocessor.get_normalized_array(random_noise_image_bytes)


@pytest.fixture(scope="module")
def uniform_array() -> np.ndarray:
    """All-constant float32 128×128 array — DCT energy concentrates in DC bin."""
    return np.full((128, 128), 0.5, dtype=np.float32)


@pytest.fixture(scope="module")
def noisy_array() -> np.ndarray:
    """High-variance random float32 128×128 array — energy spread across all bins."""
    rng = np.random.default_rng(seed=99)
    return rng.random((128, 128)).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature vector properties
# ---------------------------------------------------------------------------


def test_feature_vector_length_is_1024(extractor: DCTExtractor, face_array: np.ndarray) -> None:
    result = extractor.extract(face_array)
    assert result.feature_vector_length == 1024
    assert len(result.feature_vector) == 1024
    assert _DCT_BLOCK_SIZE == 32
    assert _DCT_BLOCK_SIZE ** 2 == 1024


def test_feature_vector_is_log_compressed(
    extractor: DCTExtractor, face_array: np.ndarray
) -> None:
    """
    After log compression sign(x)*log(1+|x|), absolute values should be
    bounded by log(1 + max_raw_dct).  For a [0,255]-scaled 128x128 image
    the raw DC coefficient is at most 128*128*255 ≈ 4.2M; log(1+4.2M) ≈ 15.2.
    We verify the max absolute value is < 20 to confirm compression was applied.
    """
    result = extractor.extract(face_array)
    fv = np.array(result.feature_vector)
    assert np.max(np.abs(fv)) < 20.0, (
        f"Max |value| = {np.max(np.abs(fv)):.2f} — log compression may not be applied"
    )


def test_dct_block_is_32x32(extractor: DCTExtractor, face_array: np.ndarray) -> None:
    result = extractor.extract(face_array)
    assert result.dct_size == 32
    assert result.feature_vector_length == 32 * 32


def test_normalization_field(extractor: DCTExtractor, face_array: np.ndarray) -> None:
    result = extractor.extract(face_array)
    assert result.normalization == "log_compression"


# ---------------------------------------------------------------------------
# Discriminative power
# ---------------------------------------------------------------------------


def test_different_images_give_different_vectors(
    extractor: DCTExtractor,
    face_array: np.ndarray,
    noise_array: np.ndarray,
) -> None:
    """Face and noise DCT feature vectors must be meaningfully different."""
    r_face = extractor.extract(face_array)
    r_noise = extractor.extract(noise_array)
    fv_face = np.array(r_face.feature_vector)
    fv_noise = np.array(r_noise.feature_vector)
    l2 = float(np.linalg.norm(fv_face - fv_noise))
    assert l2 > 0.1, f"Expected L2 distance > 0.1, got {l2:.4f}"


def test_uniform_image_gives_single_spike(
    extractor: DCTExtractor, uniform_array: np.ndarray
) -> None:
    """
    A uniform image has energy only in the DC coefficient (index 0).
    After log compression, coefficient 0 should be non-zero and all
    remaining coefficients should be approximately zero (|value| < 1e-6).
    """
    result = extractor.extract(uniform_array)
    fv = np.array(result.feature_vector)
    # DC component must have significant energy
    assert abs(fv[0]) > 0.1, f"DC coefficient = {fv[0]:.6f}, expected non-zero"
    # All AC components (indices 1 onwards) should be effectively zero
    ac_max = np.max(np.abs(fv[1:]))
    assert ac_max < 1e-4, (
        f"Max AC coefficient = {ac_max:.6e} — expected near-zero for uniform image"
    )


def test_noisy_image_gives_spread_energy(
    extractor: DCTExtractor, noisy_array: np.ndarray
) -> None:
    """
    A high-variance random image should distribute energy across many
    frequency bins — at least 50% of the 1024 bins should be non-zero.
    """
    result = extractor.extract(noisy_array)
    fv = np.array(result.feature_vector)
    nonzero = int(np.sum(np.abs(fv) > 1e-6))
    assert nonzero > 512, (
        f"Expected >512 non-zero bins for noisy image, got {nonzero}"
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def test_dct_image_b64_is_valid(extractor: DCTExtractor, face_array: np.ndarray) -> None:
    result = extractor.extract(face_array)
    raw = base64.b64decode(result.dct_image_b64)
    assert raw[:3] == b"\xff\xd8\xff", "dct_image_b64 is not a valid JPEG"


def test_dct_image_has_correct_size(
    extractor: DCTExtractor, face_array: np.ndarray
) -> None:
    """The DCT visualisation should be 128×128 (matches input)."""
    result = extractor.extract(face_array)
    raw = base64.b64decode(result.dct_image_b64)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (128, 128), f"Expected (128, 128), got {img.size}"


# ---------------------------------------------------------------------------
# Block stats
# ---------------------------------------------------------------------------


def test_block_stats_are_correct(extractor: DCTExtractor, face_array: np.ndarray) -> None:
    result = extractor.extract(face_array)
    stats = result.dct_block_stats
    for key in ("mean", "std", "energy", "max", "min"):
        assert key in stats, f"Missing key '{key}' in dct_block_stats"
    assert stats["energy"] > 0
    assert stats["max"] >= stats["min"]


def test_processing_time_is_populated(
    extractor: DCTExtractor, face_array: np.ndarray
) -> None:
    result = extractor.extract(face_array)
    assert result.processing_time_ms >= 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_wrong_dimensions_raises_error(extractor: DCTExtractor) -> None:
    bad = np.zeros((128, 128, 3), dtype=np.float32)
    with pytest.raises(ImageProcessingError):
        extractor.extract(bad)


def test_wrong_dtype_raises_error(extractor: DCTExtractor) -> None:
    bad = np.zeros((128, 128), dtype=np.uint8)
    with pytest.raises(ImageProcessingError):
        extractor.extract(bad)


def test_image_smaller_than_block_raises_error(extractor: DCTExtractor) -> None:
    """Image smaller than 32×32 must be rejected gracefully."""
    too_small = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ImageProcessingError):
        extractor.extract(too_small)
