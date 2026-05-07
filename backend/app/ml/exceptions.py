"""
Custom exceptions for the ML pipeline.

Each exception carries an optional ``details`` dict so callers can attach
structured context (e.g. filename, measured size) without string formatting.
"""


class MLPipelineError(Exception):
    """Base class for all ML pipeline failures."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class InvalidImageError(MLPipelineError):
    """
    Raised when an uploaded image fails validation.

    Covers format mismatches, dimension violations, size limits, and
    corruption — anything that prevents the pipeline from proceeding.
    """


class NoFaceDetectedError(MLPipelineError):
    """
    Raised in strict mode when no face is found in a valid image.

    In non-strict (default) mode the detector returns an empty list instead
    of raising, so callers can decide whether the absence of a face is fatal.
    """


class ImageProcessingError(MLPipelineError):
    """
    Raised when a technically valid image cannot be processed by OpenCV.

    Examples: unexpected colour space, corrupt pixel data that passes
    Pillow verification but fails cv2.imdecode, etc.
    """
