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
