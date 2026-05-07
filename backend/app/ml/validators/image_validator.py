"""
Module 1 — Image Validation

Purpose: Guard the pipeline entrance. Every image must pass these checks
before any ML processing occurs, preventing waste of GPU/CPU resources and
blocking malicious or malformed uploads.

Why validate both extension AND magic bytes?
    A malicious actor could rename a .exe to .jpg. Magic bytes are the first
    few bytes of a file that identify its true format regardless of filename.
    Checking both layers prevents disguised-file attacks.
"""

import io
import struct
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.ml.exceptions import InvalidImageError


# Magic byte signatures for supported formats.
# These are the canonical first bytes that identify each format.
_MAGIC_BYTES: dict[str, list[bytes]] = {
    "JPEG": [b"\xff\xd8\xff"],
    "PNG": [b"\x89PNG\r\n\x1a\n"],
    "WEBP": [b"RIFF"],  # Further validated by bytes 8-12 == b"WEBP"
}


@dataclass
class ValidationResult:
    """
    Outcome of image validation.

    Attributes:
        is_valid:  True when the image can proceed to the ML pipeline.
        errors:    Hard failures — at least one error means is_valid=False.
        warnings:  Soft issues (e.g. grayscale input) that are auto-corrected.
        metadata:  Extracted image properties for downstream modules.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _detect_format_from_magic(data: bytes) -> str | None:
    """
    Identify image format by inspecting raw bytes, not file extension.

    Returns the canonical Pillow format string ("JPEG", "PNG", "WEBP")
    or None if the file does not match any supported signature.
    """
    if len(data) < 12:
        return None

    for fmt, signatures in _MAGIC_BYTES.items():
        for sig in signatures:
            if data[: len(sig)] == sig:
                # Extra check for WebP: bytes 8-12 must spell "WEBP"
                if fmt == "WEBP":
                    if data[8:12] == b"WEBP":
                        return "WEBP"
                    continue
                return fmt
    return None


class ImageValidator:
    """
    Validates an uploaded image before it enters the ML pipeline.

    Checks (in order):
    1. File size ≤ MAX_IMAGE_SIZE_MB
    2. Format matches allowed list (JPEG / PNG / WebP)
    3. Magic bytes confirm the declared format (prevents disguised files)
    4. Dimensions are within [MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION]
    5. Image is not corrupted (Pillow verify + OpenCV decode)
    6. Colour mode is RGB (grayscale / RGBA auto-conversion — warning only)
    """

    def __init__(self) -> None:
        self._max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        self._min_dim = settings.MIN_IMAGE_DIMENSION
        self._max_dim = settings.MAX_IMAGE_DIMENSION
        self._allowed_formats: list[str] = [
            f.upper() for f in settings.ALLOWED_IMAGE_FORMATS
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, file_bytes: bytes, filename: str) -> ValidationResult:
        """
        Run all validation checks against raw image bytes.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename:   Original filename (used for extension check and errors).

        Returns:
            ValidationResult with is_valid, errors, warnings, and metadata.

        Raises:
            InvalidImageError: When a hard failure occurs that cannot be
                               expressed as a soft validation result (e.g.
                               completely unreadable data).
        """
        result = ValidationResult()

        # 1. File size
        self._check_file_size(file_bytes, result)

        # 2 & 3. Format: extension + magic bytes
        detected_format = self._check_format(file_bytes, filename, result)

        # 4–6 require a decodable image — skip if already invalid to avoid
        # cascading errors from unreadable data.
        if result.is_valid and detected_format:
            pil_image = self._try_open_pillow(file_bytes, filename, result)
            if pil_image is not None:
                self._check_dimensions(pil_image, result)
                self._check_color_mode(pil_image, result)
                self._check_opencv_decodable(file_bytes, result)
                self._populate_metadata(pil_image, file_bytes, result)

        return result

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_file_size(self, data: bytes, result: ValidationResult) -> None:
        """Reject files that exceed the configured size limit."""
        size_bytes = len(data)
        if size_bytes > self._max_bytes:
            size_mb = size_bytes / (1024 * 1024)
            result.is_valid = False
            result.errors.append(
                f"File size {size_mb:.1f} MB exceeds the {settings.MAX_IMAGE_SIZE_MB} MB limit."
            )

    def _check_format(
        self, data: bytes, filename: str, result: ValidationResult
    ) -> str | None:
        """
        Verify format via both file extension and magic bytes.

        Why both?  Extension is user-controlled and trivially spoofed.
        Magic bytes are intrinsic to the file content. Requiring both
        ensures consistency and blocks disguised uploads.
        """
        ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else ""
        ext_map = {"JPG": "JPEG", "JPEG": "JPEG", "PNG": "PNG", "WEBP": "WEBP"}
        ext_format = ext_map.get(ext)

        if ext_format not in self._allowed_formats:
            result.is_valid = False
            result.errors.append(
                f"Extension '.{ext.lower()}' is not allowed. "
                f"Accepted: {', '.join(self._allowed_formats)}."
            )
            return None

        magic_format = _detect_format_from_magic(data)
        if magic_format is None:
            result.is_valid = False
            result.errors.append(
                "File header does not match any supported image format. "
                "The file may be corrupted or disguised."
            )
            return None

        if magic_format != ext_format:
            result.is_valid = False
            result.errors.append(
                f"File extension claims '{ext_format}' but file content "
                f"is '{magic_format}'. Possible disguised file."
            )
            return None

        return magic_format

    def _try_open_pillow(
        self, data: bytes, filename: str, result: ValidationResult
    ) -> Image.Image | None:
        """
        Attempt to open the image with Pillow.

        Pillow's verify() detects truncated files and certain corruption
        patterns. We open twice: once to verify, once to actually use,
        because verify() leaves the file pointer in an unusable state.
        """
        try:
            # First pass: verify integrity
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except (UnidentifiedImageError, Exception) as exc:
            result.is_valid = False
            result.errors.append(f"Image is corrupted or unreadable: {exc}")
            return None

        try:
            # Second pass: open for actual use (verify() exhausts the stream)
            img = Image.open(io.BytesIO(data))
            img.load()  # Force full decode to catch truncated images
            return img
        except Exception as exc:
            result.is_valid = False
            result.errors.append(f"Failed to decode image data: {exc}")
            return None

    def _check_dimensions(
        self, img: Image.Image, result: ValidationResult
    ) -> None:
        """Enforce minimum and maximum pixel dimension constraints."""
        w, h = img.size
        if w < self._min_dim or h < self._min_dim:
            result.is_valid = False
            result.errors.append(
                f"Image dimensions {w}×{h} are below the minimum "
                f"{self._min_dim}×{self._min_dim} px required for reliable detection."
            )
        if w > self._max_dim or h > self._max_dim:
            result.is_valid = False
            result.errors.append(
                f"Image dimensions {w}×{h} exceed the maximum "
                f"{self._max_dim}×{self._max_dim} px."
            )

    def _check_color_mode(
        self, img: Image.Image, result: ValidationResult
    ) -> None:
        """
        Warn (but allow) non-RGB images.

        Grayscale images lose colour information; RGBA images have
        transparency. Both can still be processed after conversion to RGB,
        but the user should know their image was modified.
        """
        mode = img.mode
        if mode == "L":
            result.warnings.append(
                "Image is grayscale (mode=L). It will be converted to RGB "
                "before processing; colour features will be absent."
            )
        elif mode == "RGBA":
            result.warnings.append(
                "Image has an alpha channel (mode=RGBA). The transparency "
                "layer will be discarded and the image converted to RGB."
            )
        elif mode not in ("RGB",):
            result.warnings.append(
                f"Unusual colour mode '{mode}'. The image will be converted "
                "to RGB; results may vary."
            )

    def _check_opencv_decodable(
        self, data: bytes, result: ValidationResult
    ) -> None:
        """
        Verify OpenCV can decode the image.

        Pillow and OpenCV use different decoders. An image that Pillow
        accepts may still fail cv2.imdecode (e.g. certain progressive JPEGs
        or exotic sub-formats). We confirm here so downstream modules don't
        hit unexpected decode errors.
        """
        arr = np.frombuffer(data, np.uint8)
        img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_cv is None:
            result.is_valid = False
            result.errors.append(
                "OpenCV could not decode the image. "
                "The file may use an unsupported sub-format."
            )

    def _populate_metadata(
        self,
        img: Image.Image,
        data: bytes,
        result: ValidationResult,
    ) -> None:
        """Attach extracted image properties to the result metadata dict."""
        w, h = img.size
        result.metadata = {
            "width": w,
            "height": h,
            "format": img.format or "unknown",
            "mode": img.mode,
            "size_kb": round(len(data) / 1024, 2),
            "aspect_ratio": round(w / h, 4) if h else 0,
        }
