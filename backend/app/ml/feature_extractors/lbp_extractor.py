"""
Module 4 — LBP (Local Binary Pattern) Feature Extraction

Background — What is LBP?
    Local Binary Pattern is a texture descriptor introduced by Ojala et al.
    (1996, extended 2002).  For each pixel in a grayscale image, it compares
    the pixel's intensity with N evenly-spaced neighbours on a circle of
    radius R.  Each comparison produces a 1-bit result (neighbour ≥ centre →
    1, else 0).  Reading the N bits clockwise gives an N-bit binary code;
    converting that code to decimal gives the LBP value for that pixel.

    Example (R=1, N=8 — the most common setting):
        Neighbours:  45 | 78 | 82
                     23 | 60 | 91    centre = 60
                     10 | 55 | 38

        Comparisons (clockwise from top-left):
            45 <  60 → 0
            78 >= 60 → 1
            82 >= 60 → 1
            91 >= 60 → 1
            38 <  60 → 0
            55 <  60 → 0
            10 <  60 → 0
            23 <  60 → 0

        Binary code: 01110000 → decimal 112 → LBP(pixel) = 112

    The result is an LBP image of the same size as the input, where each
    pixel holds its texture code (0–255 for N=8).  Computing a histogram
    over all LBP codes gives the texture descriptor (feature vector) that
    represents the image's overall texture distribution.

Why uniform patterns?
    Most real-world texture patterns produce LBP codes with at most two
    0→1 or 1→0 transitions around the bit string (e.g. 00011100).  These
    are called "uniform".  Non-uniform codes (three or more transitions) are
    rare in natural images and typically caused by noise.

    With N=8 neighbours there are 58 uniform patterns + 1 catch-all
    non-uniform bin = 59 total bins.  Using uniform patterns:
    • Reduces the feature vector from 256 → 59 values
    • Improves robustness to noise (non-uniform outliers collapse to one bin)
    • Retains the discriminative power of the uniform codes

Why LBP detects morphing artifacts?
    Face morphing blends two images using pixel-level mixing, warping, or
    GAN-based synthesis.  These operations alter the microstructure of the
    skin texture in ways that are invisible to the human eye but measurable
    with LBP:
    • Blending creates interpolated pixel values that disrupt natural skin
      texture patterns (pores, fine lines, hair follicles).
    • Warping introduces re-sampling artifacts (bilinear/bicubic blurring)
      that change the high-frequency texture content.
    • GAN synthesis produces statistically different texture distributions
      compared to real photographs.

    A histogram of LBP codes acts as a texture fingerprint.  Real face
    images cluster together in LBP feature space; morphed images land in
    different (or mixed) regions.  K-Means clustering (Module 7) exploits
    this separation to assign a Real / Morphed label.
"""

import base64
import time

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel
from skimage.feature import local_binary_pattern

from app.ml.exceptions import ImageProcessingError


# Number of LBP bins for uniform method with 8 neighbours:
# 58 uniform patterns + 1 non-uniform catch-all = 59
_N_POINTS = 8
_RADIUS = 1
_N_BINS = _N_POINTS * (_N_POINTS - 1) + 3  # = 59


class LBPResult(BaseModel):
    """
    Output of a single LBP feature extraction pass.

    Attributes:
        feature_vector:        Normalised histogram of LBP codes — 59 float
                               values that sum to 1.0.  This is the texture
                               fingerprint fed into the clustering module.
        feature_vector_length: Always 59 for uniform LBP with N=8, R=1.
        lbp_image_b64:         Base64-encoded JPEG of the LBP-transformed
                               image — useful for viva visualisation.
        histogram_stats:       Descriptive statistics of the feature vector:
                               mean, std, max_bin, min_bin (bin index),
                               max_value, min_value.
        processing_time_ms:    Wall-clock time for this extraction pass.
        method:                LBP variant used — always "uniform".
        radius:                Neighbourhood radius — always 1.
        n_points:              Number of neighbours — always 8.
    """

    feature_vector: list[float]
    feature_vector_length: int
    lbp_image_b64: str
    histogram_stats: dict
    processing_time_ms: float
    method: str
    radius: int
    n_points: int


class LBPExtractor:
    """
    Extracts a uniform LBP texture feature vector from a preprocessed face.

    Expects a float32 numpy array of shape (128, 128) with values in [0, 1]
    as produced by Module 3 (ImagePreprocessor).
    """

    def __init__(
        self,
        radius: int = _RADIUS,
        n_points: int = _N_POINTS,
        method: str = "uniform",
    ) -> None:
        self._radius = radius
        self._n_points = n_points
        self._method = method
        # Uniform LBP bin count: n_points*(n_points-1)+3
        self._n_bins = n_points * (n_points - 1) + 3
        logger.info(
            "LBPExtractor initialised | radius={} n_points={} method={} n_bins={}",
            radius,
            n_points,
            method,
            self._n_bins,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, preprocessed_array: np.ndarray) -> LBPResult:
        """
        Compute the uniform LBP feature vector for a preprocessed face image.

        Args:
            preprocessed_array: Float32 numpy array of shape (128, 128) with
                                 pixel values normalised to [0.0, 1.0].

        Returns:
            LBPResult containing the 59-element feature vector, the LBP
            image as base64, and diagnostic statistics.

        Raises:
            ImageProcessingError: When the input array has an unexpected
                                  shape or dtype.
        """
        self._validate_input(preprocessed_array)
        t_start = time.perf_counter()

        # Convert float [0,1] → uint8 [0,255] for skimage's LBP function.
        # skimage accepts floats but uint8 is the canonical input for face
        # texture analysis and avoids floating-point precision issues in the
        # neighbour comparisons.
        uint8_image = (preprocessed_array * 255).astype(np.uint8)

        # Compute the LBP image.
        # local_binary_pattern returns a float array of shape (H, W) where
        # each pixel holds its LBP code (0 to n_bins-1 for uniform method).
        lbp_image = local_binary_pattern(
            uint8_image,
            P=self._n_points,
            R=self._radius,
            method=self._method,
        )

        # Build the normalised histogram (feature vector).
        feature_vector = self._compute_histogram(lbp_image)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Build a uint8 LBP image for visualisation.  Scale the LBP codes
        # (0 to n_bins-1 = 0 to 58) to 0-255 for JPEG encoding.
        scale = 255.0 / (self._n_bins - 1)
        lbp_visual = np.clip(lbp_image * scale, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", lbp_visual, [cv2.IMWRITE_JPEG_QUALITY, 90])
        lbp_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        stats = self._compute_stats(feature_vector)

        logger.debug(
            "LBP extraction complete | vector_len={} | mean={:.4f} | {:.2f} ms",
            len(feature_vector),
            stats["mean"],
            elapsed_ms,
        )

        return LBPResult(
            feature_vector=feature_vector,
            feature_vector_length=len(feature_vector),
            lbp_image_b64=lbp_b64,
            histogram_stats=stats,
            processing_time_ms=round(elapsed_ms, 3),
            method=self._method,
            radius=self._radius,
            n_points=self._n_points,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, arr: np.ndarray) -> None:
        """Enforce shape and dtype constraints on the preprocessed array."""
        if arr.ndim != 2:
            raise ImageProcessingError(
                f"LBPExtractor expects a 2-D (grayscale) array, got shape {arr.shape}.",
                details={"shape": list(arr.shape)},
            )
        if arr.dtype not in (np.float32, np.float64):
            raise ImageProcessingError(
                f"LBPExtractor expects a float array, got dtype={arr.dtype}.",
                details={"dtype": str(arr.dtype)},
            )

    def _compute_histogram(self, lbp_image: np.ndarray) -> list[float]:
        """
        Build a normalised histogram of LBP codes.

        np.histogram bins the LBP codes into self._n_bins equally-spaced
        buckets ranging from 0 to n_bins.  Dividing by the total pixel
        count makes the histogram invariant to image size.

        Returns:
            List of self._n_bins float values summing to 1.0.
        """
        hist, _ = np.histogram(
            lbp_image.ravel(),
            bins=self._n_bins,
            range=(0, self._n_bins),
        )
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist.tolist()

    @staticmethod
    def _compute_stats(feature_vector: list[float]) -> dict:
        """Return descriptive statistics of the feature vector."""
        arr = np.array(feature_vector)
        max_idx = int(np.argmax(arr))
        min_idx = int(np.argmin(arr))
        return {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "max_bin": max_idx,
            "min_bin": min_idx,
            "max_value": round(float(arr[max_idx]), 6),
            "min_value": round(float(arr[min_idx]), 6),
        }
