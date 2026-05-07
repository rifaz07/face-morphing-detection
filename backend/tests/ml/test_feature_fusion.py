"""
Unit tests for Module 6 — FeatureFusion.

Tests verify fusion correctness, contribution percentages, normalisation,
and that the full-pipeline convenience method works end-to-end.
"""

import numpy as np
import pytest

from app.ml.exceptions import ImageProcessingError
from app.ml.feature_extractors.feature_fusion import (
    FeatureFusion,
    _DCT_SIZE,
    _FUSED_SIZE,
    _LBP_SIZE,
)
from app.ml.feature_extractors.dct_extractor import DCTExtractor
from app.ml.feature_extractors.lbp_extractor import LBPExtractor
from app.ml.preprocessors.image_preprocessor import ImagePreprocessor


@pytest.fixture(scope="module")
def fusion() -> FeatureFusion:
    return FeatureFusion()


@pytest.fixture(scope="module")
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor()


@pytest.fixture(scope="module")
def lbp_extractor() -> LBPExtractor:
    return LBPExtractor()


@pytest.fixture(scope="module")
def dct_extractor() -> DCTExtractor:
    return DCTExtractor()


@pytest.fixture(scope="module")
def face_array(valid_face_image_bytes: bytes, preprocessor: ImagePreprocessor) -> np.ndarray:
    return preprocessor.get_normalized_array(valid_face_image_bytes)


@pytest.fixture(scope="module")
def noise_array(random_noise_image_bytes: bytes, preprocessor: ImagePreprocessor) -> np.ndarray:
    return preprocessor.get_normalized_array(random_noise_image_bytes)


@pytest.fixture(scope="module")
def face_lbp(face_array: np.ndarray, lbp_extractor: LBPExtractor) -> list[float]:
    return lbp_extractor.extract(face_array).feature_vector


@pytest.fixture(scope="module")
def face_dct(face_array: np.ndarray, dct_extractor: DCTExtractor) -> list[float]:
    return dct_extractor.extract(face_array).feature_vector


@pytest.fixture(scope="module")
def noise_lbp(noise_array: np.ndarray, lbp_extractor: LBPExtractor) -> list[float]:
    return lbp_extractor.extract(noise_array).feature_vector


@pytest.fixture(scope="module")
def noise_dct(noise_array: np.ndarray, dct_extractor: DCTExtractor) -> list[float]:
    return dct_extractor.extract(noise_array).feature_vector


# ---------------------------------------------------------------------------
# Vector size and composition
# ---------------------------------------------------------------------------


def test_fused_vector_length_is_1083(
    fusion: FeatureFusion, face_lbp: list[float], face_dct: list[float]
) -> None:
    result = fusion.fuse(face_lbp, face_dct)
    assert result.fused_vector_length == 1083
    assert len(result.fused_vector) == 1083
    assert _FUSED_SIZE == 1083
    assert _LBP_SIZE + _DCT_SIZE == 1083


def test_lbp_contribution_percentage_correct(
    fusion: FeatureFusion, face_lbp: list[float], face_dct: list[float]
) -> None:
    result = fusion.fuse(face_lbp, face_dct)
    expected = round(59 / 1083 * 100, 4)
    assert abs(result.lbp_contribution - expected) < 0.001
    assert abs(result.lbp_contribution - 5.4478) < 0.01


def test_dct_contribution_percentage_correct(
    fusion: FeatureFusion, face_lbp: list[float], face_dct: list[float]
) -> None:
    result = fusion.fuse(face_lbp, face_dct)
    expected = round(1024 / 1083 * 100, 4)
    assert abs(result.dct_contribution - expected) < 0.001
    assert abs(result.dct_contribution - 94.5522) < 0.01


def test_contributions_sum_to_100(
    fusion: FeatureFusion, face_lbp: list[float], face_dct: list[float]
) -> None:
    result = fusion.fuse(face_lbp, face_dct)
    assert abs(result.lbp_contribution + result.dct_contribution - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_dct_normalised_before_fusion(
    fusion: FeatureFusion, face_lbp: list[float], face_dct: list[float]
) -> None:
    """
    The DCT portion of the fused vector (indices 59–1082) should be in [0, 1]
    because MinMax normalisation was applied before concatenation.
    """
    result = fusion.fuse(face_lbp, face_dct)
    dct_portion = result.fused_vector[_LBP_SIZE:]
    assert min(dct_portion) >= -1e-9, "DCT min is below 0 after normalisation"
    assert max(dct_portion) <= 1.0 + 1e-9, "DCT max exceeds 1 after normalisation"


def test_lbp_portion_unchanged(
    fusion: FeatureFusion, face_lbp: list[float], face_dct: list[float]
) -> None:
    """LBP is already in [0,1]; first 59 values should match the input exactly."""
    result = fusion.fuse(face_lbp, face_dct)
    lbp_portion = result.fused_vector[:_LBP_SIZE]
    for orig, fused in zip(face_lbp, lbp_portion):
        assert abs(orig - fused) < 1e-9


def test_normalization_applied_flag(
    fusion: FeatureFusion, face_lbp: list[float], face_dct: list[float]
) -> None:
    result = fusion.fuse(face_lbp, face_dct)
    assert result.normalization_applied is True
    assert result.fusion_method == "concatenation"


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


def test_fused_vector_all_values_finite(
    fusion: FeatureFusion, face_lbp: list[float], face_dct: list[float]
) -> None:
    result = fusion.fuse(face_lbp, face_dct)
    fv = np.array(result.fused_vector)
    assert np.all(np.isfinite(fv)), "Fused vector contains NaN or Inf values"


def test_fusion_stats_are_correct(
    fusion: FeatureFusion, face_lbp: list[float], face_dct: list[float]
) -> None:
    result = fusion.fuse(face_lbp, face_dct)
    for section in (result.lbp_stats, result.dct_stats, result.fused_stats):
        for key in ("min", "max", "mean", "std"):
            assert key in section
            assert isinstance(section[key], float)
    assert result.fused_stats["max"] >= result.fused_stats["min"]
    assert 0.0 <= result.fused_stats["min"]
    assert result.fused_stats["max"] <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Discriminative power
# ---------------------------------------------------------------------------


def test_different_images_give_different_fused_vectors(
    fusion: FeatureFusion,
    face_lbp: list[float],
    face_dct: list[float],
    noise_lbp: list[float],
    noise_dct: list[float],
) -> None:
    r_face = fusion.fuse(face_lbp, face_dct)
    r_noise = fusion.fuse(noise_lbp, noise_dct)
    fv_face = np.array(r_face.fused_vector)
    fv_noise = np.array(r_noise.fused_vector)
    l2 = float(np.linalg.norm(fv_face - fv_noise))
    assert l2 > 0.01, f"Expected L2 > 0.01, got {l2:.4f}"


# ---------------------------------------------------------------------------
# Full pipeline convenience method
# ---------------------------------------------------------------------------


def test_full_pipeline_method_works(
    fusion: FeatureFusion, face_array: np.ndarray
) -> None:
    result = fusion.extract_full_pipeline(face_array)
    assert result.fused_vector_length == 1083
    assert len(result.fused_vector) == 1083
    assert result.normalization_applied is True
    fv = np.array(result.fused_vector)
    assert np.all(np.isfinite(fv))


def test_full_pipeline_matches_manual_fuse(
    fusion: FeatureFusion,
    face_array: np.ndarray,
    face_lbp: list[float],
    face_dct: list[float],
) -> None:
    """extract_full_pipeline must produce the same result as fuse() called manually."""
    r_pipeline = fusion.extract_full_pipeline(face_array)
    r_manual = fusion.fuse(face_lbp, face_dct)
    fv_p = np.array(r_pipeline.fused_vector)
    fv_m = np.array(r_manual.fused_vector)
    assert np.allclose(fv_p, fv_m, atol=1e-9), "Pipeline and manual fusion differ"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_wrong_lbp_length_raises_error(
    fusion: FeatureFusion, face_dct: list[float]
) -> None:
    bad_lbp = [0.0] * 50  # wrong length
    with pytest.raises(ImageProcessingError):
        fusion.fuse(bad_lbp, face_dct)


def test_wrong_dct_length_raises_error(
    fusion: FeatureFusion, face_lbp: list[float]
) -> None:
    bad_dct = [0.0] * 512  # wrong length
    with pytest.raises(ImageProcessingError):
        fusion.fuse(face_lbp, bad_dct)
