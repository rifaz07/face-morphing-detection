"""
Unit tests for Module 1 — ImageValidator.

Each test targets a single validation rule so failures are easy to diagnose.
"""

import io
import struct

import pytest
from PIL import Image

from app.ml.validators.image_validator import ImageValidator


@pytest.fixture(scope="module")
def validator() -> ImageValidator:
    return ImageValidator()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_jpeg_passes(validator: ImageValidator, valid_face_image_bytes: bytes) -> None:
    result = validator.validate(valid_face_image_bytes, "face.jpg")
    assert result.is_valid is True
    assert result.errors == []
    assert result.metadata["format"] in ("JPEG", "unknown")  # Pillow sets format on load


def test_valid_png_passes(validator: ImageValidator, valid_face_image_png_bytes: bytes) -> None:
    result = validator.validate(valid_face_image_png_bytes, "face.png")
    assert result.is_valid is True
    assert result.errors == []


def test_metadata_populated(validator: ImageValidator, valid_face_image_bytes: bytes) -> None:
    result = validator.validate(valid_face_image_bytes, "face.jpg")
    assert result.is_valid is True
    meta = result.metadata
    assert "width" in meta and meta["width"] > 0
    assert "height" in meta and meta["height"] > 0
    assert "size_kb" in meta and meta["size_kb"] > 0
    assert "aspect_ratio" in meta


# ---------------------------------------------------------------------------
# File-size check
# ---------------------------------------------------------------------------


def test_oversized_file_rejected(validator: ImageValidator, oversized_image_bytes: bytes) -> None:
    result = validator.validate(oversized_image_bytes, "big.jpg")
    assert result.is_valid is False
    assert any("MB" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Format / magic bytes check
# ---------------------------------------------------------------------------


def test_unsupported_format_rejected(
    validator: ImageValidator, invalid_format_bytes: bytes
) -> None:
    result = validator.validate(invalid_format_bytes, "file.txt")
    assert result.is_valid is False
    # Should catch either the bad extension OR the magic bytes mismatch
    assert len(result.errors) > 0


def test_jpeg_extension_with_png_content_rejected(
    validator: ImageValidator, valid_face_image_png_bytes: bytes
) -> None:
    """Magic bytes say PNG but extension says JPG — should fail."""
    result = validator.validate(valid_face_image_png_bytes, "sneaky.jpg")
    assert result.is_valid is False
    assert any("content" in e.lower() or "disguised" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Dimension checks
# ---------------------------------------------------------------------------


def test_too_small_dimensions_rejected(
    validator: ImageValidator, tiny_image_bytes: bytes
) -> None:
    result = validator.validate(tiny_image_bytes, "tiny.jpg")
    assert result.is_valid is False
    assert any("minimum" in e.lower() or "100" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Corruption check
# ---------------------------------------------------------------------------


def test_corrupted_image_rejected(validator: ImageValidator) -> None:
    # Truncate a valid JPEG magic header to force Pillow to fail
    bad_data = b"\xff\xd8\xff\xe0" + b"\x00" * 50  # valid header, garbage body
    result = validator.validate(bad_data, "corrupted.jpg")
    assert result.is_valid is False


# ---------------------------------------------------------------------------
# Colour mode check (warn, not fail)
# ---------------------------------------------------------------------------


def test_grayscale_warns_but_allows(
    validator: ImageValidator, grayscale_image_bytes: bytes
) -> None:
    result = validator.validate(grayscale_image_bytes, "gray.jpg")
    # Grayscale is allowed — pipeline will convert to RGB
    assert result.is_valid is True
    assert any("grayscale" in w.lower() or "mode=L" in w for w in result.warnings)
