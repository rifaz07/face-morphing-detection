"""
Module 6 — Feature Fusion

Purpose: Combine the LBP texture feature vector (59 values) and the DCT
frequency feature vector (1024 values) into a single 1083-dimensional
representation that describes a face from two complementary perspectives.

Why combine LBP + DCT?
    Each descriptor captures a different aspect of the face:

    • LBP (Module 4) encodes *local texture microstructure* — the pattern
      of intensity differences between neighbouring pixels.  It is sensitive
      to the skin texture anomalies introduced by morphing blending.

    • DCT (Module 5) encodes *global frequency content* — how much energy
      is present at each spatial frequency across the entire 128×128 image.
      It captures the spectral artefacts of re-sampling and blending.

    No single descriptor is sufficient.  A face morphed by simple alpha-
    blending may fool LBP (the skin texture remains realistic in small
    patches) but leave a clear signature in the DCT domain (the blending
    introduces mid-frequency energy not present in real faces).  The reverse
    is also true for GAN-based morphing.  Fusing both gives K-Means a richer
    feature space in which real and morphed faces are more linearly separable.

Why normalise DCT before fusion?
    The LBP histogram values are already in [0.0, 1.0] (it is a normalised
    probability distribution).  Raw DCT coefficients after log compression
    still span several units (roughly –20 to +20 for the block extracted
    from a 128×128 face scaled to [0, 255]).  If we concatenate without
    normalisation, the DCT values would be ~10–20× larger than the LBP
    values, effectively reducing the LBP contribution to noise in any
    Euclidean distance calculation — including K-Means centroid updates.

    We apply MinMaxScaler to the DCT vector independently, mapping it to
    [0.0, 1.0] so that both components contribute on equal footing.

    Note: the scaler is fitted on each individual vector, not on a training
    corpus.  This is appropriate here because we are building a per-image
    feature descriptor, not training a normalisation model.  The assumption
    is that the min/max of a 1024-element vector from a single face is a
    reasonable proxy for the global range — the DCT block values from
    different faces of the same class tend to cluster in similar ranges.

What does the 1083-dimensional vector represent?
    The fused vector is the complete face signature fed into K-Means
    (Module 7).  Each of the 1083 dimensions encodes one independent aspect
    of the face's appearance:
    • Dimensions 0–58:    LBP uniform histogram bins (texture distribution)
    • Dimensions 59–1082: DCT top-left 32×32 block coefficients (frequency)

    K-Means will partition the feature space into two clusters.  After
    clustering, the cluster whose centroid is farther from the "real face"
    distribution in this high-dimensional space is labelled MORPHED.
"""

import time

import numpy as np
from loguru import logger
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

from app.ml.exceptions import ImageProcessingError
from app.ml.feature_extractors.dct_extractor import DCTExtractor
from app.ml.feature_extractors.lbp_extractor import LBPExtractor


# Expected input sizes — used for validation and contribution percentages.
_LBP_SIZE = 59
_DCT_SIZE = 1024
_FUSED_SIZE = _LBP_SIZE + _DCT_SIZE  # 1083


class FusionResult(BaseModel):
    """
    Output of a single feature fusion pass.

    Attributes:
        fused_vector:           Concatenated [LBP | DCT_normalised] vector,
                                1083 float values all in [0.0, 1.0].
        fused_vector_length:    Always 1083.
        lbp_contribution:       Fraction of the vector from LBP (59/1083 ≈ 5.45 %).
        dct_contribution:       Fraction of the vector from DCT (1024/1083 ≈ 94.55 %).
        lbp_stats:              Descriptive stats of the LBP portion (already [0,1]).
        dct_stats:              Descriptive stats of the DCT portion *after* MinMax
                                normalisation to [0, 1].
        fused_stats:            Descriptive stats of the full 1083-element vector.
        processing_time_ms:     Wall-clock time for this fusion pass.
        fusion_method:          Always "concatenation".
        normalization_applied:  Always True — DCT was MinMax-normalised before fusion.
    """

    fused_vector: list[float]
    fused_vector_length: int
    lbp_contribution: float
    dct_contribution: float
    lbp_stats: dict
    dct_stats: dict
    fused_stats: dict
    processing_time_ms: float
    fusion_method: str
    normalization_applied: bool


class FeatureFusion:
    """
    Fuses LBP and DCT feature vectors into a single 1083-dimensional descriptor.

    Can be used in two ways:
    1. ``fuse(lbp_vector, dct_vector)`` — when you already have both vectors.
    2. ``extract_full_pipeline(preprocessed_array)`` — derives both vectors
       internally from the preprocessed float32 array and fuses them.
       This is the entry point called by Module 7 (K-Means clustering).
    """

    def __init__(self) -> None:
        self._lbp_extractor = LBPExtractor()
        self._dct_extractor = DCTExtractor()
        logger.info(
            "FeatureFusion initialised | lbp_size={} dct_size={} fused_size={}",
            _LBP_SIZE,
            _DCT_SIZE,
            _FUSED_SIZE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        lbp_vector: list[float],
        dct_vector: list[float],
    ) -> FusionResult:
        """
        Fuse pre-computed LBP and DCT feature vectors.

        Args:
            lbp_vector: 59-element normalised LBP histogram (values in [0, 1]).
            dct_vector: 1024-element log-compressed DCT coefficients.

        Returns:
            FusionResult with the 1083-element fused vector and metadata.

        Raises:
            ImageProcessingError: When input vectors have unexpected lengths.
        """
        self._validate_inputs(lbp_vector, dct_vector)
        t_start = time.perf_counter()

        lbp_arr = np.array(lbp_vector, dtype=np.float64)
        dct_arr = np.array(dct_vector, dtype=np.float64)

        # Normalise DCT to [0, 1] so it matches the LBP scale.
        dct_normalised = self._minmax_normalise(dct_arr)

        # Concatenate: [LBP(59) | DCT_normalised(1024)]
        fused = np.concatenate([lbp_arr, dct_normalised])

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        lbp_stats = self._stats(lbp_arr)
        dct_stats = self._stats(dct_normalised)
        fused_stats = self._stats(fused)

        logger.debug(
            "Feature fusion complete | len={} | mean={:.4f} | {:.3f} ms",
            len(fused),
            fused_stats["mean"],
            elapsed_ms,
        )

        return FusionResult(
            fused_vector=fused.tolist(),
            fused_vector_length=len(fused),
            lbp_contribution=round(_LBP_SIZE / _FUSED_SIZE * 100, 4),
            dct_contribution=round(_DCT_SIZE / _FUSED_SIZE * 100, 4),
            lbp_stats=lbp_stats,
            dct_stats=dct_stats,
            fused_stats=fused_stats,
            processing_time_ms=round(elapsed_ms, 3),
            fusion_method="concatenation",
            normalization_applied=True,
        )

    def extract_full_pipeline(self, preprocessed_array: np.ndarray) -> FusionResult:
        """
        Run LBP + DCT extraction then fuse — the single call used by Module 7.

        Args:
            preprocessed_array: Float32 numpy array of shape (128, 128) with
                                 values in [0.0, 1.0] from Module 3.

        Returns:
            FusionResult with the complete 1083-element feature vector.
        """
        lbp_result = self._lbp_extractor.extract(preprocessed_array)
        dct_result = self._dct_extractor.extract(preprocessed_array)
        return self.fuse(lbp_result.feature_vector, dct_result.feature_vector)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self, lbp_vector: list[float], dct_vector: list[float]
    ) -> None:
        if len(lbp_vector) != _LBP_SIZE:
            raise ImageProcessingError(
                f"LBP vector must have {_LBP_SIZE} elements, got {len(lbp_vector)}.",
                details={"received_length": len(lbp_vector), "expected": _LBP_SIZE},
            )
        if len(dct_vector) != _DCT_SIZE:
            raise ImageProcessingError(
                f"DCT vector must have {_DCT_SIZE} elements, got {len(dct_vector)}.",
                details={"received_length": len(dct_vector), "expected": _DCT_SIZE},
            )

    @staticmethod
    def _minmax_normalise(arr: np.ndarray) -> np.ndarray:
        """
        Scale array to [0, 1] using per-vector min/max.

        When all values are identical (degenerate case, e.g. a uniform image)
        the range is zero; we return an all-zeros array rather than dividing
        by zero.
        """
        min_val = arr.min()
        max_val = arr.max()
        rng = max_val - min_val
        if rng == 0.0:
            return np.zeros_like(arr)
        return (arr - min_val) / rng

    @staticmethod
    def _stats(arr: np.ndarray) -> dict:
        return {
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
        }
