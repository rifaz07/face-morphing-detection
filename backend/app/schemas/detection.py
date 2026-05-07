"""
Pydantic v2 schemas for the Detection API endpoints.

These are the wire-format models — separate from the internal ML dataclasses
so the API contract can evolve independently of the ML internals.
"""

from typing import Optional

from pydantic import BaseModel, Field, computed_field


class ImageValidationResponse(BaseModel):
    """Response schema for POST /api/v1/detection/validate."""

    is_valid: bool = Field(
        description="True when the image passes all validation checks.",
        examples=[True],
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Hard failures that prevented validation (non-empty means is_valid=False).",
        examples=[[]],
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Soft issues that were auto-corrected (e.g. grayscale input).",
        examples=[[]],
    )
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Extracted image properties: width, height, format, mode, "
            "size_kb, aspect_ratio."
        ),
        examples=[
            {
                "width": 640,
                "height": 480,
                "format": "JPEG",
                "mode": "RGB",
                "size_kb": 102.4,
                "aspect_ratio": 1.3333,
            }
        ],
    )


class FaceBoundingBox(BaseModel):
    """Pixel coordinates of a single detected face."""

    x: int = Field(description="Left edge of the bounding box (pixels).", examples=[120])
    y: int = Field(description="Top edge of the bounding box (pixels).", examples=[80])
    width: int = Field(description="Width of the bounding box (pixels).", examples=[200])
    height: int = Field(description="Height of the bounding box (pixels).", examples=[200])

    @computed_field  # type: ignore[misc]
    @property
    def area(self) -> int:
        """Pixel area — used by clients to rank faces by prominence."""
        return self.width * self.height


class FaceDetectionResponse(BaseModel):
    """Response schema for POST /api/v1/detection/detect-face."""

    face_count: int = Field(
        description="Total number of faces detected.",
        examples=[1],
    )
    faces: list[FaceBoundingBox] = Field(
        description="Bounding box for every detected face.",
        examples=[[{"x": 120, "y": 80, "width": 200, "height": 200, "area": 40000}]],
    )
    largest_face: Optional[FaceBoundingBox] = Field(
        default=None,
        description="The face with the greatest pixel area, or null when no face found.",
    )
    cropped_faces_b64: list[str] = Field(
        description=(
            "Base64-encoded JPEG thumbnail for each detected face "
            "(same order as `faces`)."
        ),
        examples=[["<base64-string>"]],
    )
    processing_time_ms: float = Field(
        description="Wall-clock time for the detection step in milliseconds.",
        examples=[42.5],
    )
    image_dimensions: dict = Field(
        description="Width and height of the input image.",
        examples=[{"width": 640, "height": 480}],
    )
    preprocessed_faces: list["PreprocessingResult"] = Field(
        default_factory=list,
        description=(
            "Module 3 preprocessing result for each detected face crop "
            "(same order as `faces`). Empty list when no faces were found."
        ),
    )


class PreprocessingResult(BaseModel):
    """Result of the Module 3 preprocessing pipeline for a single face."""

    preprocessed_b64: str = Field(
        description=(
            "Base64-encoded JPEG of the 128×128 histogram-equalised grayscale "
            "face — suitable for rendering a before/after comparison."
        ),
        examples=["<base64-string>"],
    )
    numpy_array_shape: list[int] = Field(
        description="Shape of the normalised float32 array, e.g. [128, 128].",
        examples=[[128, 128]],
    )
    steps_applied: list[str] = Field(
        description="Ordered list of preprocessing steps executed.",
        examples=[["resize_128x128", "grayscale", "histogram_equalization", "normalize_0_1"]],
    )
    processing_time_ms: float = Field(
        description="Wall-clock time for the preprocessing pass in milliseconds.",
        examples=[3.2],
    )
    original_size: dict = Field(
        description="Width and height of the face crop before resizing.",
        examples=[{"width": 200, "height": 220}],
    )
    normalized_stats: dict = Field(
        description="Descriptive statistics of the normalised [0,1] array: min, max, mean, std.",
        examples=[{"min": 0.0, "max": 1.0, "mean": 0.512, "std": 0.241}],
    )


class PreprocessingResponse(BaseModel):
    """Response schema for POST /api/v1/detection/preprocess."""

    detection: FaceDetectionResponse = Field(
        description="Full face detection result (Modules 1+2).",
    )
    preprocessing: Optional[PreprocessingResult] = Field(
        default=None,
        description=(
            "Preprocessing result for the largest detected face. "
            "Null when no face was found in the image."
        ),
    )


class LBPResult(BaseModel):
    """Result of the Module 4 LBP feature extraction pass."""

    feature_vector: list[float] = Field(
        description=(
            "Normalised histogram of LBP codes — 59 float values summing to 1.0. "
            "This is the texture fingerprint used by the clustering module."
        ),
        examples=[[0.012, 0.034, 0.001]],
    )
    feature_vector_length: int = Field(
        description="Length of the feature vector (59 for uniform LBP with N=8, R=1).",
        examples=[59],
    )
    lbp_image_b64: str = Field(
        description="Base64-encoded JPEG of the LBP-transformed image — useful for visualisation.",
        examples=["<base64-string>"],
    )
    histogram_stats: dict = Field(
        description="Descriptive statistics of the feature vector: mean, std, max_bin, min_bin, max_value, min_value.",
        examples=[{"mean": 0.017, "std": 0.031, "max_bin": 58, "min_bin": 3, "max_value": 0.21, "min_value": 0.0}],
    )
    processing_time_ms: float = Field(
        description="Wall-clock time for the extraction pass in milliseconds.",
        examples=[5.1],
    )
    method: str = Field(
        description="LBP variant — always 'uniform'.",
        examples=["uniform"],
    )
    radius: int = Field(description="Neighbourhood radius.", examples=[1])
    n_points: int = Field(description="Number of neighbour points.", examples=[8])


class LBPResponse(BaseModel):
    """Response schema for POST /api/v1/detection/extract-lbp."""

    detection: FaceDetectionResponse = Field(
        description="Full face detection result (Modules 1+2+3).",
    )
    lbp: Optional[LBPResult] = Field(
        default=None,
        description=(
            "LBP feature extraction result for the largest detected face. "
            "Null when no face was found."
        ),
    )


class DetectionErrorResponse(BaseModel):
    """Returned on 400 / 422 errors from detection endpoints."""

    error_code: str = Field(
        description="Machine-readable error identifier.",
        examples=["INVALID_IMAGE"],
    )
    message: str = Field(
        description="Human-readable description of the error.",
        examples=["File size 12.3 MB exceeds the 10 MB limit."],
    )
    details: dict = Field(
        default_factory=dict,
        description="Additional structured context (validation errors, etc.).",
        examples=[{"errors": ["File size 12.3 MB exceeds the 10 MB limit."]}],
    )
