"""
Module 5 — DCT (Discrete Cosine Transform) Feature Extraction

Background — What is DCT?
    The Discrete Cosine Transform converts an image from the spatial domain
    (pixel intensities) into the frequency domain (cosine wave amplitudes).
    Each coefficient in the DCT output represents how much of a particular
    spatial frequency is present in the image.

    The 2D DCT of an N×N image produces an N×N matrix of coefficients:
    • DC coefficient (0,0)          — average brightness of the whole image
    • Low-frequency coefficients    — coarse structure, faces, large shapes
      (small row/col indices)
    • Mid-frequency coefficients    — edges, skin texture, hair boundaries
    • High-frequency coefficients   — fine details, noise, compression
      (large row/col indices)         artefacts

    This is exactly the principle behind JPEG compression: JPEG discards
    high-frequency coefficients because they are less visually important.

Why DCT detects morphing artefacts?
    Face morphing creates artificial images by blending two real faces.
    The blending process — whether pixel-averaging, warping, or GAN
    synthesis — disturbs the natural frequency distribution of the face:

    • Pixel blending produces interpolated values that add energy in
      mid-frequency bands not present in either source image.
    • Geometric warping introduces re-sampling artefacts (ringing,
      blurring) visible as anomalous DCT coefficients in mid-to-high
      frequency bands.
    • GAN synthesis creates textures that look natural spatially but
      differ from real photography in frequency-domain statistics.

    By extracting the top-left 32×32 block (DC + low/mid frequencies)
    we capture the range where morphing artefacts are most visible while
    discarding the high-frequency noise that varies with compression level
    and camera sensor.

Why the 32×32 block?
    Using the full 128×128 = 16 384 DCT coefficients would make the
    feature vector too large for K-Means clustering (curse of dimensionality)
    and most of those coefficients represent high-frequency noise.
    The 32×32 = 1 024 top-left block is an established trade-off in the
    face forensics literature: it retains enough information to distinguish
    real from morphed while keeping computation tractable.

Why log compression?
    Raw DCT coefficients have an enormous dynamic range — the DC coefficient
    (top-left) can be orders of magnitude larger than mid-frequency values.
    Feeding raw values into distance-based algorithms (K-Means) would cause
    the DC component to dominate all others.  Log compression:

        compressed(x) = sign(x) × log(1 + |x|)

    preserves the sign, retains relative ordering, and compresses the
    range so that all 1 024 coefficients contribute meaningfully to the
    Euclidean distance used in clustering.

LBP vs DCT — complementary perspectives:
    LBP (Module 4) captures *spatial texture* — the micro-pattern of pixel
    relationships within a small neighbourhood.  It is sensitive to changes
    in skin texture caused by morphing.

    DCT (Module 5) captures *global frequency content* — how much of each
    spatial frequency is present across the whole image.  It is sensitive to
    the spectral artefacts introduced by blending and re-sampling.

    Together (Module 6 Feature Fusion) they give the classifier a richer
    description of the face than either could provide alone.
"""

import base64
import time

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel
from scipy.fft import dctn

from app.ml.exceptions import ImageProcessingError


_DCT_BLOCK_SIZE = 32  # rows/cols of top-left DCT block to keep


class DCTResult(BaseModel):
    """
    Output of a single DCT feature extraction pass.

    Attributes:
        feature_vector:        Log-compressed 32×32 DCT coefficients flattened
                               to 1 024 float values.
        feature_vector_length: Always 1024.
        dct_image_b64:         Base64-encoded JPEG of the full log-scaled DCT
                               coefficient map — useful for visualisation.
        dct_block_stats:       Descriptive statistics of the 32×32 block:
                               mean, std, energy (sum of squares), max, min.
        processing_time_ms:    Wall-clock time for this extraction pass.
        dct_size:              Side length of the extracted block (32).
        normalization:         Always "log_compression".
    """

    feature_vector: list[float]
    feature_vector_length: int
    dct_image_b64: str
    dct_block_stats: dict
    processing_time_ms: float
    dct_size: int
    normalization: str


class DCTExtractor:
    """
    Extracts a log-compressed DCT frequency feature vector from a preprocessed
    face image.

    Expects a float32 numpy array of shape (128, 128) with values in [0, 1]
    as produced by Module 3 (ImagePreprocessor).
    """

    def __init__(self, block_size: int = _DCT_BLOCK_SIZE) -> None:
        self._block_size = block_size
        logger.info(
            "DCTExtractor initialised | block_size={}x{} | feature_len={}",
            block_size,
            block_size,
            block_size * block_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, preprocessed_array: np.ndarray) -> DCTResult:
        """
        Compute the log-compressed DCT feature vector for a preprocessed face.

        Pipeline:
        1. Validate input shape and dtype.
        2. Scale float [0,1] → float64 [0,255] for numerical stability.
        3. Apply 2D DCT (scipy dctn, type-II, orthonormal normalisation).
        4. Extract top-left block_size×block_size coefficients.
        5. Apply log compression: sign(x) * log(1 + |x|).
        6. Flatten to 1-D feature vector.
        7. Encode the full DCT coefficient map as a JPEG for visualisation.

        Args:
            preprocessed_array: Float32 numpy array of shape (128, 128) with
                                 pixel values in [0.0, 1.0].

        Returns:
            DCTResult with the 1024-element feature vector and metadata.

        Raises:
            ImageProcessingError: When the input has an unexpected shape/dtype.
        """
        self._validate_input(preprocessed_array)
        t_start = time.perf_counter()

        # Scale to [0, 255] float64 — improves numerical conditioning of DCT.
        scaled = preprocessed_array.astype(np.float64) * 255.0

        # 2D DCT (type II, scipy default).  norm="ortho" makes the transform
        # orthonormal so the total energy is preserved — important for the
        # energy statistic in dct_block_stats.
        dct_full = dctn(scaled, type=2, norm="ortho")

        # Extract top-left block (low + mid frequencies).
        block = dct_full[: self._block_size, : self._block_size].copy()

        # Log compression — stabilises large dynamic range.
        compressed = np.sign(block) * np.log1p(np.abs(block))

        feature_vector = compressed.ravel().tolist()

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Build visualisation from the full DCT map (log-scaled for contrast).
        vis_b64 = self._encode_dct_image(dct_full)

        stats = self._compute_stats(block, compressed)

        logger.debug(
            "DCT extraction complete | block={}x{} | energy={:.2f} | {:.2f} ms",
            self._block_size,
            self._block_size,
            stats["energy"],
            elapsed_ms,
        )

        return DCTResult(
            feature_vector=feature_vector,
            feature_vector_length=len(feature_vector),
            dct_image_b64=vis_b64,
            dct_block_stats=stats,
            processing_time_ms=round(elapsed_ms, 3),
            dct_size=self._block_size,
            normalization="log_compression",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, arr: np.ndarray) -> None:
        if arr.ndim != 2:
            raise ImageProcessingError(
                f"DCTExtractor expects a 2-D (grayscale) array, got shape {arr.shape}.",
                details={"shape": list(arr.shape)},
            )
        if arr.dtype not in (np.float32, np.float64):
            raise ImageProcessingError(
                f"DCTExtractor expects a float array, got dtype={arr.dtype}.",
                details={"dtype": str(arr.dtype)},
            )
        h, w = arr.shape
        if h < self._block_size or w < self._block_size:
            raise ImageProcessingError(
                f"Image ({h}×{w}) is smaller than DCT block size ({self._block_size}).",
                details={"shape": [h, w], "block_size": self._block_size},
            )

    def _encode_dct_image(self, dct_full: np.ndarray) -> str:
        """
        Encode the full DCT coefficient map as a base64 JPEG.

        The raw DCT values have a huge dynamic range, so we apply a log
        transform before mapping to uint8 so that mid-frequency detail
        is visible — otherwise the image would look almost all-black.
        """
        log_map = np.log1p(np.abs(dct_full))
        max_val = log_map.max()
        if max_val > 0:
            vis = (log_map / max_val * 255).astype(np.uint8)
        else:
            vis = np.zeros_like(dct_full, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _compute_stats(raw_block: np.ndarray, compressed_block: np.ndarray) -> dict:
        """Descriptive statistics on the extracted 32×32 block (raw values)."""
        flat = raw_block.ravel()
        return {
            "mean": round(float(flat.mean()), 4),
            "std": round(float(flat.std()), 4),
            "energy": round(float(np.sum(flat ** 2)), 4),
            "max": round(float(flat.max()), 4),
            "min": round(float(flat.min()), 4),
        }
