"""
Module 3 — Image Preprocessing

Purpose: Transform a raw face crop into a normalised, fixed-size grayscale
array ready for feature extraction (Modules 4 LBP and 5 DCT).

Why each step matters (viva-ready):

1. Resize to 128×128
   All downstream feature extractors expect a fixed input size.
   128×128 is the standard for face recognition benchmarks — large enough
   to preserve texture detail, small enough to keep computation fast.
   Using a fixed size also means all feature vectors have identical
   dimensionality, which is a requirement for clustering (Module 7).

2. Convert to grayscale
   LBP (Local Binary Pattern) captures texture differences between
   neighbouring pixels.  DCT (Discrete Cosine Transform) captures
   frequency content.  Both operate on intensity, not colour.
   Dropping the two chroma channels also reduces the feature space
   by 3× without losing texture or frequency information.

3. Histogram equalization
   Real-world face images vary wildly in brightness: passport photos are
   well-lit studio shots; morphed images may come from outdoor selfies or
   low-quality scans.  equalizeHist remaps the pixel distribution so that
   all intensity levels are used equally, effectively normalising for
   lighting.  This makes the same face look consistent regardless of
   whether it was photographed in sunlight or under fluorescent light.

4. Normalise to [0.0, 1.0]
   ML algorithms (K-Means in Module 7, scikit-learn estimators) are
   sensitive to feature scale.  Dividing by 255.0 maps the uint8 range
   into floating-point [0, 1], preventing large raw pixel values from
   numerically dominating smaller feature components and improving the
   stability of the distance calculations used in clustering.
"""

import base64
import io
import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel

from app.ml.exceptions import ImageProcessingError


class PreprocessingResult(BaseModel):
    """
    Full output of a single preprocessing pass.

    Attributes:
        preprocessed_b64:   Base64-encoded JPEG of the final 128×128 grayscale
                            image (uint8, before normalisation) — suitable for
                            display in the frontend "before/after" panel.
        numpy_array_shape:  Shape of the normalised float32 array, e.g. [128, 128].
        steps_applied:      Ordered list of processing steps that were executed.
        processing_time_ms: Wall-clock time for the full preprocessing pass.
        original_size:      Width and height of the input image before resize.
        normalized_stats:   Descriptive statistics of the final normalised array
                            (min, max, mean, std) — useful for debugging and for
                            confirming normalisation was applied correctly.
    """

    preprocessed_b64: str
    numpy_array_shape: list[int]
    steps_applied: list[str]
    processing_time_ms: float
    original_size: dict
    normalized_stats: dict


class ImagePreprocessor:
    """
    Applies a fixed four-step preprocessing pipeline to a face crop.

    The pipeline is deterministic and stateless — no model weights or
    learned parameters are involved.  The same input always produces
    the same output, making it safe to call from multiple threads.
    """

    TARGET_SIZE: tuple[int, int] = (128, 128)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, face_b64: str) -> PreprocessingResult:
        """
        Preprocess a base64-encoded face crop (Module 2 output format).

        Args:
            face_b64: Base64-encoded JPEG string as produced by FaceDetector.

        Returns:
            PreprocessingResult with the processed array and metadata.

        Raises:
            ImageProcessingError: When base64 decoding or image decode fails.
        """
        try:
            raw_bytes = base64.b64decode(face_b64)
        except Exception as exc:
            raise ImageProcessingError(
                f"Invalid base64 input: {exc}",
                details={"input_length": len(face_b64)},
            )
        return self.preprocess_bytes(raw_bytes)

    def preprocess_bytes(self, image_bytes: bytes) -> PreprocessingResult:
        """
        Preprocess raw image bytes (JPEG / PNG / WebP face crop).

        Args:
            image_bytes: Raw bytes of a face crop image.

        Returns:
            PreprocessingResult with the processed array and metadata.

        Raises:
            ImageProcessingError: When OpenCV cannot decode the bytes.
        """
        t_start = time.perf_counter()
        steps: list[str] = []

        # Decode
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ImageProcessingError(
                "cv2.imdecode returned None — face crop bytes could not be decoded.",
                details={"bytes_length": len(image_bytes)},
            )

        original_h, original_w = bgr.shape[:2]

        # Step 1: Resize to 128×128
        resized = cv2.resize(bgr, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
        steps.append(f"resize_{self.TARGET_SIZE[0]}x{self.TARGET_SIZE[1]}")

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        steps.append("grayscale")

        # Step 3: Histogram equalisation — normalise contrast / lighting
        equalized = cv2.equalizeHist(gray)
        steps.append("histogram_equalization")

        # Step 4: Normalise to [0.0, 1.0]
        normalized: np.ndarray = equalized.astype(np.float32) / 255.0
        steps.append("normalize_0_1")

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Build a JPEG preview from the uint8 equalized image (before float
        # conversion) so the frontend can render it without extra decoding.
        _, buf = cv2.imencode(".jpg", equalized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        preprocessed_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        stats = {
            "min": float(np.min(normalized)),
            "max": float(np.max(normalized)),
            "mean": round(float(np.mean(normalized)), 6),
            "std": round(float(np.std(normalized)), 6),
        }

        logger.debug(
            "Preprocessing complete | {}×{} → {} | steps={} | {:.2f} ms",
            original_w,
            original_h,
            self.TARGET_SIZE,
            steps,
            elapsed_ms,
        )

        return PreprocessingResult(
            preprocessed_b64=preprocessed_b64,
            numpy_array_shape=list(normalized.shape),
            steps_applied=steps,
            processing_time_ms=round(elapsed_ms, 3),
            original_size={"width": original_w, "height": original_h},
            normalized_stats=stats,
        )

    # ------------------------------------------------------------------
    # Internal helper (used by tests that need the raw array)
    # ------------------------------------------------------------------

    def get_normalized_array(self, image_bytes: bytes) -> np.ndarray:
        """
        Return just the normalised float32 numpy array without building
        the full PreprocessingResult.  Used by downstream modules (4, 5)
        that need the raw array, not the API-facing model.
        """
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ImageProcessingError(
                "cv2.imdecode failed in get_normalized_array.",
                details={"bytes_length": len(image_bytes)},
            )
        resized = cv2.resize(bgr, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return equalized.astype(np.float32) / 255.0
