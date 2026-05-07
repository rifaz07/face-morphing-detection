"""
Detection API endpoints — Modules 1, 2, 3, 4, 5, 6 & 7.

POST /api/v1/detection/validate          → Image validation pre-flight (Module 1)
POST /api/v1/detection/detect-face       → Validate + detect + preprocess (Modules 1+2+3)
POST /api/v1/detection/preprocess        → Full pipeline, preprocessing detail (Modules 1+2+3)
POST /api/v1/detection/extract-lbp       → Full pipeline + LBP texture features (Modules 1+2+3+4)
POST /api/v1/detection/extract-dct       → Full pipeline + DCT frequency features (Modules 1+2+3+5)
POST /api/v1/detection/extract-features  → Complete feature extraction pipeline (Modules 1–6)
POST /api/v1/detection/classify          → MAIN: full pipeline + K-Means prediction (Modules 1–7)
GET  /api/v1/detection/model-info        → K-Means model status and metadata
POST /api/v1/detection/retrain           → Retrain K-Means on synthetic data
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from loguru import logger

from app.ml.detectors.face_detector import FaceDetector
from app.ml.exceptions import ImageProcessingError, InvalidImageError
from app.ml.feature_extractors.dct_extractor import DCTExtractor
from app.ml.feature_extractors.feature_fusion import FeatureFusion
from app.ml.feature_extractors.lbp_extractor import LBPExtractor
from app.ml.preprocessors.image_preprocessor import ImagePreprocessor
from app.ml.validators.image_validator import ImageValidator
from app.schemas.detection import (
    ClassifyResponse,
    DCTResponse,
    DCTResult,
    DetectionErrorResponse,
    FaceBoundingBox,
    FaceDetectionResponse,
    FusionResponse,
    FusionResult,
    ImageValidationResponse,
    KMeansModelInfo,
    KMeansPredictionResult,
    KMeansTrainingResult,
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
_dct_extractor = DCTExtractor()
_feature_fusion = FeatureFusion()


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


# ---------------------------------------------------------------------------
# POST /extract-dct  (Modules 1 + 2 + 3 + 5)
# ---------------------------------------------------------------------------


@router.post(
    "/extract-dct",
    response_model=DCTResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract DCT frequency features from the largest detected face",
    description=(
        "Runs the full Modules 1 → 2 → 3 → 5 pipeline: validates the image, "
        "detects faces, preprocesses, then applies a 2D DCT and extracts the "
        "top-left 32×32 block (1024 coefficients) with log compression. "
        "DCT frequency analysis complements LBP texture analysis — morphing "
        "artefacts appear in the mid-frequency range captured by this block. "
        "Returns null dct field when no face is found."
    ),
    responses={
        200: {"description": "Detection result + DCT feature vector for the largest face"},
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
async def extract_dct(file: UploadFile) -> DCTResponse:
    """
    Full pipeline: validate → detect → preprocess → extract DCT (Modules 1+2+3+5).

    Returns :class:`DCTResponse` with:
    - **detection** — face detection result including preprocessed crops
    - **dct** — 1024-element log-compressed DCT feature vector for the
      largest face, or null if no face detected
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
        return DCTResponse(detection=detection_response, dct=None)

    # Identify the crop for the largest face
    largest_area = detection.largest_face.area
    largest_index = 0
    for i, box in enumerate(detection.faces):
        if box.area == largest_area:
            largest_index = i
            break

    # Module 5 — extract DCT from the preprocessed float32 array
    try:
        largest_crop_bytes = __import__("base64").b64decode(
            detection.cropped_faces_b64[largest_index]
        )
        normalized_array = _preprocessor.get_normalized_array(largest_crop_bytes)
        dct_result_raw = _dct_extractor.extract(normalized_array)
    except ImageProcessingError as exc:
        logger.error("DCT extraction failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=DetectionErrorResponse(
                error_code="DCT_EXTRACTION_ERROR",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    dct_result = DCTResult(
        feature_vector=dct_result_raw.feature_vector,
        feature_vector_length=dct_result_raw.feature_vector_length,
        dct_image_b64=dct_result_raw.dct_image_b64,
        dct_block_stats=dct_result_raw.dct_block_stats,
        processing_time_ms=dct_result_raw.processing_time_ms,
        dct_size=dct_result_raw.dct_size,
        normalization=dct_result_raw.normalization,
    )

    return DCTResponse(detection=detection_response, dct=dct_result)


# ---------------------------------------------------------------------------
# POST /extract-features  (Modules 1 + 2 + 3 + 4 + 5 + 6)
# ---------------------------------------------------------------------------


@router.post(
    "/extract-features",
    response_model=FusionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract full fused feature vector (LBP + DCT) from the largest face",
    description=(
        "Complete feature extraction pipeline — Modules 1 → 2 → 3 → 4 → 5 → 6. "
        "Validates the image, detects faces, preprocesses, extracts LBP texture "
        "features (59 values) and DCT frequency features (1024 values), then fuses "
        "them into a single 1083-dimensional vector with DCT MinMax-normalised to "
        "[0,1] before concatenation. This fused vector is the direct input to the "
        "K-Means clustering module. Returns null fusion field when no face found."
    ),
    responses={
        200: {"description": "Fused 1083-dim feature vector for the largest detected face"},
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
async def extract_features(file: UploadFile) -> FusionResponse:
    """
    Full pipeline: validate → detect → preprocess → LBP → DCT → fuse
    (Modules 1+2+3+4+5+6).

    Returns :class:`FusionResponse` with:
    - **detection** — face detection + preprocessing results
    - **lbp** — 59-element LBP feature vector for the largest face
    - **dct** — 1024-element DCT feature vector for the largest face
    - **fusion** — 1083-element fused vector (LBP + DCT_normalised), or null
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
        return FusionResponse(
            detection=detection_response, lbp=None, dct=None, fusion=None
        )

    # Identify the largest-face crop index
    largest_area = detection.largest_face.area
    largest_index = 0
    for i, box in enumerate(detection.faces):
        if box.area == largest_area:
            largest_index = i
            break

    try:
        import base64 as _base64
        largest_crop_bytes = _base64.b64decode(
            detection.cropped_faces_b64[largest_index]
        )
        normalized_array = _preprocessor.get_normalized_array(largest_crop_bytes)

        # Modules 4 + 5
        lbp_raw = _lbp_extractor.extract(normalized_array)
        dct_raw = _dct_extractor.extract(normalized_array)

        # Module 6 — fuse
        fusion_raw = _feature_fusion.fuse(
            lbp_raw.feature_vector, dct_raw.feature_vector
        )
    except ImageProcessingError as exc:
        logger.error("Feature extraction/fusion failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=DetectionErrorResponse(
                error_code="FEATURE_EXTRACTION_ERROR",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    lbp_result = LBPResult(
        feature_vector=lbp_raw.feature_vector,
        feature_vector_length=lbp_raw.feature_vector_length,
        lbp_image_b64=lbp_raw.lbp_image_b64,
        histogram_stats=lbp_raw.histogram_stats,
        processing_time_ms=lbp_raw.processing_time_ms,
        method=lbp_raw.method,
        radius=lbp_raw.radius,
        n_points=lbp_raw.n_points,
    )
    dct_result = DCTResult(
        feature_vector=dct_raw.feature_vector,
        feature_vector_length=dct_raw.feature_vector_length,
        dct_image_b64=dct_raw.dct_image_b64,
        dct_block_stats=dct_raw.dct_block_stats,
        processing_time_ms=dct_raw.processing_time_ms,
        dct_size=dct_raw.dct_size,
        normalization=dct_raw.normalization,
    )
    fusion_result = FusionResult(
        fused_vector=fusion_raw.fused_vector,
        fused_vector_length=fusion_raw.fused_vector_length,
        lbp_contribution=fusion_raw.lbp_contribution,
        dct_contribution=fusion_raw.dct_contribution,
        lbp_stats=fusion_raw.lbp_stats,
        dct_stats=fusion_raw.dct_stats,
        fused_stats=fusion_raw.fused_stats,
        processing_time_ms=fusion_raw.processing_time_ms,
        fusion_method=fusion_raw.fusion_method,
        normalization_applied=fusion_raw.normalization_applied,
    )

    return FusionResponse(
        detection=detection_response,
        lbp=lbp_result,
        dct=dct_result,
        fusion=fusion_result,
    )


# ---------------------------------------------------------------------------
# POST /classify  (Modules 1 → 7 — MAIN ENDPOINT)
# ---------------------------------------------------------------------------


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a face image as REAL or MORPHED (complete pipeline, Modules 1–7)",
    description=(
        "The primary detection endpoint. Runs the complete pipeline: "
        "validates the image (Module 1) → detects faces with Haar Cascade "
        "(Module 2) → preprocesses the largest face (Module 3) → extracts "
        "LBP texture features (Module 4) → extracts DCT frequency features "
        "(Module 5) → fuses into a 1083-dim vector (Module 6) → classifies "
        "with K-Means (Module 7). "
        "Returns REAL or MORPHED with a confidence score in [0, 1]. "
        "Always returns HTTP 200 — a MORPHED result is a valid detection, "
        "not an error. Returns null prediction when no face is detected."
    ),
    responses={
        200: {"description": "Classification result — REAL or MORPHED with confidence"},
        400: {"model": DetectionErrorResponse, "description": "Image failed validation"},
        422: {"model": DetectionErrorResponse, "description": "ML processing error"},
    },
    tags=["Detection"],
)
async def classify_image(request: Request, file: UploadFile) -> ClassifyResponse:
    """
    Complete pipeline Modules 1–7: validate → detect → preprocess → LBP →
    DCT → fuse → K-Means classify.

    Returns :class:`ClassifyResponse` with:
    - **detection** — face detection result (bounding boxes, preprocessing)
    - **fusion**    — 1083-dim fused feature vector for the largest face
    - **prediction** — REAL / MORPHED label + confidence score, or null
    """
    import base64 as _base64

    file_bytes = await file.read()
    filename = file.filename or "upload"

    # Modules 1 + 2
    _read_validate(file_bytes, filename)
    detection = _run_detection(file_bytes)

    # Module 3
    preprocessed_all = _preprocess_crops(detection.cropped_faces_b64)
    detection_response = _build_detection_response(detection, preprocessed_all)

    if not detection.largest_face or not detection.cropped_faces_b64:
        return ClassifyResponse(detection=detection_response, fusion=None, prediction=None)

    # Identify the largest-face crop index
    largest_area = detection.largest_face.area
    largest_index = 0
    for i, box in enumerate(detection.faces):
        if box.area == largest_area:
            largest_index = i
            break

    try:
        largest_crop_bytes = _base64.b64decode(detection.cropped_faces_b64[largest_index])
        normalized_array = _preprocessor.get_normalized_array(largest_crop_bytes)

        # Modules 4 + 5
        lbp_raw = _lbp_extractor.extract(normalized_array)
        dct_raw = _dct_extractor.extract(normalized_array)

        # Module 6 — fuse
        fusion_raw = _feature_fusion.fuse(lbp_raw.feature_vector, dct_raw.feature_vector)
    except ImageProcessingError as exc:
        logger.error("Feature extraction failed during classify: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=DetectionErrorResponse(
                error_code="FEATURE_EXTRACTION_ERROR",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    fusion_result = FusionResult(
        fused_vector=fusion_raw.fused_vector,
        fused_vector_length=fusion_raw.fused_vector_length,
        lbp_contribution=fusion_raw.lbp_contribution,
        dct_contribution=fusion_raw.dct_contribution,
        lbp_stats=fusion_raw.lbp_stats,
        dct_stats=fusion_raw.dct_stats,
        fused_stats=fusion_raw.fused_stats,
        processing_time_ms=fusion_raw.processing_time_ms,
        fusion_method=fusion_raw.fusion_method,
        normalization_applied=fusion_raw.normalization_applied,
    )

    # Module 7 — K-Means classify
    classifier = request.app.state.classifier
    try:
        pred_raw = classifier.predict(fusion_raw.fused_vector)
    except ImageProcessingError as exc:
        logger.error("K-Means prediction failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=DetectionErrorResponse(
                error_code="CLASSIFICATION_ERROR",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    prediction_result = KMeansPredictionResult(
        prediction=pred_raw.prediction,
        confidence=pred_raw.confidence,
        cluster_id=pred_raw.cluster_id,
        distance_to_centroid=pred_raw.distance_to_centroid,
        processing_time_ms=pred_raw.processing_time_ms,
    )

    logger.info(
        "classify | file={} prediction={} confidence={:.3f}",
        filename, prediction_result.prediction, prediction_result.confidence,
    )

    return ClassifyResponse(
        detection=detection_response,
        fusion=fusion_result,
        prediction=prediction_result,
    )


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------


@router.get(
    "/model-info",
    response_model=KMeansModelInfo,
    status_code=status.HTTP_200_OK,
    summary="Get K-Means model metadata",
    description=(
        "Returns the current state of the K-Means classifier: whether it is "
        "fitted, the cluster-label mapping, inertia, training sample count, "
        "and whether it was trained on synthetic or real data."
    ),
    tags=["Detection"],
)
async def model_info(request: Request) -> KMeansModelInfo:
    """Return metadata about the loaded K-Means model."""
    classifier = request.app.state.classifier
    info = classifier.get_model_info()
    return KMeansModelInfo(
        is_fitted=info.is_fitted,
        model_type=info.model_type,
        n_clusters=info.n_clusters,
        cluster_labels=info.cluster_labels,
        training_samples=info.training_samples,
        inertia=info.inertia,
        trained_on_synthetic=info.trained_on_synthetic,
        model_path=info.model_path,
    )


# ---------------------------------------------------------------------------
# POST /retrain
# ---------------------------------------------------------------------------


@router.post(
    "/retrain",
    response_model=KMeansTrainingResult,
    status_code=status.HTTP_200_OK,
    summary="Retrain the K-Means model on synthetic data",
    description=(
        "Triggers a fresh K-Means training run using synthetic feature vectors "
        "(500 REAL + 500 MORPHED).  The new model immediately replaces the "
        "current one in memory and is saved to disk.  In production, replace "
        "the synthetic data with a real labelled face dataset."
    ),
    tags=["Detection"],
)
async def retrain_model(request: Request) -> KMeansTrainingResult:
    """Retrain the K-Means classifier on synthetic data and return training metrics."""
    classifier = request.app.state.classifier
    try:
        result = classifier._train_on_synthetic_data()
    except ImageProcessingError as exc:
        logger.error("Retrain failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=DetectionErrorResponse(
                error_code="TRAINING_ERROR",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    logger.info(
        "K-Means retrained | inertia={:.2f} acc={}",
        result.inertia,
        result.accuracy,
    )
    return KMeansTrainingResult(
        samples_trained=result.samples_trained,
        inertia=result.inertia,
        iterations=result.iterations,
        converged=result.converged,
        cluster_labels=result.cluster_labels,
        silhouette_score=result.silhouette_score,
        accuracy=result.accuracy,
        trained_on_synthetic=result.trained_on_synthetic,
        processing_time_ms=result.processing_time_ms,
    )
