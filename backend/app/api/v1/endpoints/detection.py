"""
Detection API endpoints — Modules 1 & 2.

POST /api/v1/detection/validate     → Image validation pre-flight
POST /api/v1/detection/detect-face  → Full validate + face detection pipeline
"""

from fastapi import APIRouter, HTTPException, UploadFile, status
from loguru import logger

from app.ml.detectors.face_detector import FaceDetector
from app.ml.exceptions import ImageProcessingError, InvalidImageError
from app.ml.validators.image_validator import ImageValidator
from app.schemas.detection import (
    DetectionErrorResponse,
    FaceDetectionResponse,
    FaceBoundingBox,
    ImageValidationResponse,
)

router = APIRouter()

# Module-level singletons — the Haar Cascade XML is loaded once at startup.
_validator = ImageValidator()
_detector = FaceDetector()


# ---------------------------------------------------------------------------
# POST /validate
# ---------------------------------------------------------------------------


@router.post(
    "/validate",
    response_model=ImageValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate an uploaded image",
    description=(
        "Pre-flight check for the detection pipeline. Verifies file size, "
        "format (JPEG / PNG / WebP), magic bytes, pixel dimensions, colour "
        "mode, and decodability. Use this endpoint to give instant feedback "
        "before committing to a full detection pass."
    ),
    responses={
        200: {"description": "Validation result (may contain warnings even when valid)"},
        400: {
            "model": DetectionErrorResponse,
            "description": "Hard validation failure — image cannot be processed",
        },
    },
    tags=["Detection"],
)
async def validate_image(file: UploadFile) -> ImageValidationResponse:
    """
    Validate an uploaded image without running face detection.

    Returns a :class:`ImageValidationResponse` with:
    - **is_valid** — whether the image can proceed to the ML pipeline
    - **errors** — list of hard failures (non-empty → is_valid=False)
    - **warnings** — list of soft issues that were auto-corrected
    - **metadata** — extracted image properties
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"

    try:
        result = _validator.validate(file_bytes, filename)
    except InvalidImageError as exc:
        logger.warning("Validation raised InvalidImageError: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=DetectionErrorResponse(
                error_code="INVALID_IMAGE",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    if not result.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=DetectionErrorResponse(
                error_code="INVALID_IMAGE",
                message="Image failed one or more validation checks.",
                details={"errors": result.errors, "warnings": result.warnings},
            ).model_dump(),
        )

    return ImageValidationResponse(
        is_valid=result.is_valid,
        errors=result.errors,
        warnings=result.warnings,
        metadata=result.metadata,
    )


# ---------------------------------------------------------------------------
# POST /detect-face
# ---------------------------------------------------------------------------


@router.post(
    "/detect-face",
    response_model=FaceDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect faces in an uploaded image",
    description=(
        "Runs the full Modules 1 → 2 pipeline: validates the image then "
        "applies OpenCV Haar Cascade face detection. Returns bounding boxes, "
        "the largest detected face, base64-encoded face crops, and timing "
        "information. Returns face_count=0 (not an error) when no face is "
        "found in a valid image."
    ),
    responses={
        200: {"description": "Detection results — face_count=0 means no face found"},
        400: {
            "model": DetectionErrorResponse,
            "description": "Image failed validation — correct the image and retry",
        },
        422: {
            "model": DetectionErrorResponse,
            "description": "Valid image but ML processing failed unexpectedly",
        },
    },
    tags=["Detection"],
)
async def detect_face(file: UploadFile) -> FaceDetectionResponse:
    """
    Validate an image then detect all frontal faces using Haar Cascade.

    Pipeline:
    1. Read uploaded bytes.
    2. Validate (Module 1) — 400 on failure.
    3. Detect faces (Module 2) — 422 on processing error.
    4. Return :class:`FaceDetectionResponse` (face_count=0 is valid, not an error).
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"

    # --- Module 1: Validate ---
    try:
        validation = _validator.validate(file_bytes, filename)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=DetectionErrorResponse(
                error_code="INVALID_IMAGE",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    if not validation.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=DetectionErrorResponse(
                error_code="INVALID_IMAGE",
                message="Image failed validation.",
                details={
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                },
            ).model_dump(),
        )

    # --- Module 2: Detect faces ---
    try:
        detection = _detector.detect(file_bytes, strict=False)
    except ImageProcessingError as exc:
        logger.error("ImageProcessingError during detection: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=DetectionErrorResponse(
                error_code="PROCESSING_ERROR",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    return FaceDetectionResponse(
        face_count=detection.face_count,
        faces=[
            FaceBoundingBox(
                x=f.x, y=f.y, width=f.width, height=f.height
            )
            for f in detection.faces
        ],
        largest_face=(
            FaceBoundingBox(
                x=detection.largest_face.x,
                y=detection.largest_face.y,
                width=detection.largest_face.width,
                height=detection.largest_face.height,
            )
            if detection.largest_face
            else None
        ),
        cropped_faces_b64=detection.cropped_faces_b64,
        processing_time_ms=detection.processing_time_ms,
        image_dimensions=detection.image_dimensions,
    )
