"""
Detection API endpoints — Modules 1, 2, 3 & 4.

POST /api/v1/detection/validate     → Image validation pre-flight (Module 1)
POST /api/v1/detection/detect-face  → Validate + detect + preprocess (Modules 1+2+3)
POST /api/v1/detection/preprocess   → Full pipeline, returns preprocessing detail (Modules 1+2+3)
POST /api/v1/detection/extract-lbp  → Full pipeline + LBP texture features (Modules 1+2+3+4)
"""

from fastapi import APIRouter, HTTPException, UploadFile, status
from loguru import logger

from app.ml.detectors.face_detector import FaceDetector
from app.ml.exceptions import ImageProcessingError, InvalidImageError
from app.ml.feature_extractors.lbp_extractor import LBPExtractor
from app.ml.preprocessors.image_preprocessor import ImagePreprocessor
from app.ml.validators.image_validator import ImageValidator
from app.schemas.detection import (
    DetectionErrorResponse,
    FaceBoundingBox,
    FaceDetectionResponse,
    ImageValidationResponse,
    LBPResponse,
    LBPResult,
    PreprocessingResponse,
    PreprocessingResult,
)

router = APIRouter()

# Module-level singletons — loaded once at startup.
_validator = ImageValidator()
_detector = FaceDetector()
_preprocessor = ImagePreprocessor()
_lbp_extractor = LBPExtractor()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _read_validate(file_bytes: bytes, filename: str) -> None:
    """Run Module 1 validation, raising HTTP 400 on failure."""
    try:
        result = _validator.validate(file_bytes, filename)
    except InvalidImageError as exc:
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
                message="Image failed validation.",
                details={"errors": result.errors, "warnings": result.warnings},
            ).model_dump(),
        )


def _run_detection(file_bytes: bytes):
    """Run Module 2 detection, raising HTTP 422 on processing error."""
    try:
        return _detector.detect(file_bytes, strict=False)
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


def _preprocess_crops(crops_b64: list[str]) -> list[PreprocessingResult]:
    """Run Module 3 on each base64 face crop, raising HTTP 422 on failure."""
    results: list[PreprocessingResult] = []
    for b64 in crops_b64:
        try:
            pr = _preprocessor.preprocess(b64)
        except ImageProcessingError as exc:
            logger.error("ImageProcessingError during preprocessing: {}", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=DetectionErrorResponse(
                    error_code="PREPROCESSING_ERROR",
                    message=str(exc),
                    details=exc.details,
                ).model_dump(),
            )
        results.append(
            PreprocessingResult(
                preprocessed_b64=pr.preprocessed_b64,
                numpy_array_shape=pr.numpy_array_shape,
                steps_applied=pr.steps_applied,
                processing_time_ms=pr.processing_time_ms,
                original_size=pr.original_size,
                normalized_stats=pr.normalized_stats,
            )
        )
    return results


def _build_detection_response(detection, preprocessed: list[PreprocessingResult]) -> FaceDetectionResponse:
    return FaceDetectionResponse(
        face_count=detection.face_count,
        faces=[
            FaceBoundingBox(x=f.x, y=f.y, width=f.width, height=f.height)
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
        preprocessed_faces=preprocessed,
    )


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
# POST /detect-face  (Modules 1 + 2 + 3)
# ---------------------------------------------------------------------------


@router.post(
    "/detect-face",
    response_model=FaceDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect and preprocess faces in an uploaded image",
    description=(
        "Runs the full Modules 1 → 2 → 3 pipeline: validates the image, "
        "detects faces with Haar Cascade, then preprocesses each face crop "
        "(resize 128×128, grayscale, histogram equalisation, normalise 0–1). "
        "Returns bounding boxes, base64 crops, preprocessing results, and "
        "timing information. Returns face_count=0 (not an error) when no "
        "face is found in a valid image."
    ),
    responses={
        200: {"description": "Detection + preprocessing results — face_count=0 means no face found"},
        400: {
            "model": DetectionErrorResponse,
            "description": "Image failed validation",
        },
        422: {
            "model": DetectionErrorResponse,
            "description": "Valid image but ML processing failed",
        },
    },
    tags=["Detection"],
)
async def detect_face(file: UploadFile) -> FaceDetectionResponse:
    """
    Validate → detect faces → preprocess each crop (Modules 1+2+3).

    Returns :class:`FaceDetectionResponse` including a
    ``preprocessed_faces`` list with one :class:`PreprocessingResult`
    per detected face.
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"

    _read_validate(file_bytes, filename)
    detection = _run_detection(file_bytes)
    preprocessed = _preprocess_crops(detection.cropped_faces_b64)

    return _build_detection_response(detection, preprocessed)


# ---------------------------------------------------------------------------
# POST /preprocess  (Modules 1 + 2 + 3, largest face only — debug/demo)
# ---------------------------------------------------------------------------


@router.post(
    "/preprocess",
    response_model=PreprocessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Preprocess the largest detected face",
    description=(
        "Debug / demo endpoint. Runs the full Modules 1 → 2 → 3 pipeline "
        "and returns the preprocessing result for the *largest* detected "
        "face only, alongside the complete detection result. "
        "Ideal for viva demonstrations: shows the original image, the "
        "cropped face, and the preprocessed (equalised, normalised) face "
        "side by side."
    ),
    responses={
        200: {"description": "Detection result + preprocessing detail for the largest face"},
        400: {
            "model": DetectionErrorResponse,
            "description": "Image failed validation",
        },
        422: {
            "model": DetectionErrorResponse,
            "description": "Valid image but ML processing failed",
        },
    },
    tags=["Detection"],
)
async def preprocess_face(file: UploadFile) -> PreprocessingResponse:
    """
    Validate → detect faces → preprocess largest face (Modules 1+2+3).

    Returns :class:`PreprocessingResponse` with:
    - **detection** — full face detection result (all faces)
    - **preprocessing** — preprocessing detail for the largest face,
      or null if no face was detected
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"

    _read_validate(file_bytes, filename)
    detection = _run_detection(file_bytes)

    # Preprocess all crops for the detect field, but surface only the
    # largest face in the dedicated preprocessing field.
    preprocessed_all = _preprocess_crops(detection.cropped_faces_b64)
    detection_response = _build_detection_response(detection, preprocessed_all)

    largest_preprocessing: PreprocessingResult | None = None
    if detection.largest_face and detection.cropped_faces_b64:
        # Find the crop index that corresponds to the largest face.
        largest_area = detection.largest_face.area
        for i, box in enumerate(detection.faces):
            if box.area == largest_area:
                largest_preprocessing = preprocessed_all[i]
                break

    return PreprocessingResponse(
        detection=detection_response,
        preprocessing=largest_preprocessing,
    )


# ---------------------------------------------------------------------------
# POST /extract-lbp  (Modules 1 + 2 + 3 + 4)
# ---------------------------------------------------------------------------


@router.post(
    "/extract-lbp",
    response_model=LBPResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract LBP texture features from the largest detected face",
    description=(
        "Runs the full Modules 1 → 2 → 3 → 4 pipeline: validates the image, "
        "detects faces, preprocesses, then extracts a 59-element uniform LBP "
        "histogram from the largest face. The LBP feature vector is the primary "
        "texture descriptor used by the morphing detection classifier. "
        "Returns null lbp field when no face is found."
    ),
    responses={
        200: {"description": "Detection + LBP feature vector for the largest face"},
        400: {
            "model": DetectionErrorResponse,
            "description": "Image failed validation",
        },
        422: {
            "model": DetectionErrorResponse,
            "description": "Valid image but ML processing failed",
        },
    },
    tags=["Detection"],
)
async def extract_lbp(file: UploadFile) -> LBPResponse:
    """
    Full pipeline: validate → detect → preprocess → extract LBP (Modules 1+2+3+4).

    Returns :class:`LBPResponse` with:
    - **detection** — face detection result including preprocessed crops
    - **lbp** — 59-element uniform LBP feature vector for the largest face,
      or null if no face detected
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"

    # Modules 1 + 2
    _read_validate(file_bytes, filename)
    detection = _run_detection(file_bytes)

    # Module 3 — preprocess all crops
    preprocessed_all = _preprocess_crops(detection.cropped_faces_b64)
    detection_response = _build_detection_response(detection, preprocessed_all)

    if not detection.largest_face or not detection.cropped_faces_b64:
        return LBPResponse(detection=detection_response, lbp=None)

    # Identify the preprocessed result for the largest face
    largest_area = detection.largest_face.area
    largest_index = 0
    for i, box in enumerate(detection.faces):
        if box.area == largest_area:
            largest_index = i
            break

    largest_preprocessed = preprocessed_all[largest_index]

    # Module 4 — extract LBP from the preprocessed array
    # Re-derive the float32 array from the preprocessed bytes to avoid
    # carrying raw numpy data through the Pydantic response chain.
    try:
        largest_crop_bytes = __import__("base64").b64decode(
            detection.cropped_faces_b64[largest_index]
        )
        normalized_array = _preprocessor.get_normalized_array(largest_crop_bytes)
        lbp_result_raw = _lbp_extractor.extract(normalized_array)
    except ImageProcessingError as exc:
        logger.error("LBP extraction failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=DetectionErrorResponse(
                error_code="LBP_EXTRACTION_ERROR",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    lbp_result = LBPResult(
        feature_vector=lbp_result_raw.feature_vector,
        feature_vector_length=lbp_result_raw.feature_vector_length,
        lbp_image_b64=lbp_result_raw.lbp_image_b64,
        histogram_stats=lbp_result_raw.histogram_stats,
        processing_time_ms=lbp_result_raw.processing_time_ms,
        method=lbp_result_raw.method,
        radius=lbp_result_raw.radius,
        n_points=lbp_result_raw.n_points,
    )

    return LBPResponse(detection=detection_response, lbp=lbp_result)
