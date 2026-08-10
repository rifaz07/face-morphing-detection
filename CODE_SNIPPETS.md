## Module 1: image_validator.py

```python
"""
Module 1 — Image Validation

Purpose: Guard the pipeline entrance. Every image must pass these checks
before any ML processing occurs, preventing waste of GPU/CPU resources and
blocking malicious or malformed uploads.

Why validate both extension AND magic bytes?
    A malicious actor could rename a .exe to .jpg. Magic bytes are the first
    few bytes of a file that identify its true format regardless of filename.
    Checking both layers prevents disguised-file attacks.
"""

import io
import struct
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.ml.exceptions import InvalidImageError


# Magic byte signatures for supported formats.
# These are the canonical first bytes that identify each format.
_MAGIC_BYTES: dict[str, list[bytes]] = {
    "JPEG": [b"\xff\xd8\xff"],
    "PNG": [b"\x89PNG\r\n\x1a\n"],
    "WEBP": [b"RIFF"],  # Further validated by bytes 8-12 == b"WEBP"
}


@dataclass
class ValidationResult:
    """
    Outcome of image validation.

    Attributes:
        is_valid:  True when the image can proceed to the ML pipeline.
        errors:    Hard failures — at least one error means is_valid=False.
        warnings:  Soft issues (e.g. grayscale input) that are auto-corrected.
        metadata:  Extracted image properties for downstream modules.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _detect_format_from_magic(data: bytes) -> str | None:
    """
    Identify image format by inspecting raw bytes, not file extension.

    Returns the canonical Pillow format string ("JPEG", "PNG", "WEBP")
    or None if the file does not match any supported signature.
    """
    if len(data) < 12:
        return None

    for fmt, signatures in _MAGIC_BYTES.items():
        for sig in signatures:
            if data[: len(sig)] == sig:
                # Extra check for WebP: bytes 8-12 must spell "WEBP"
                if fmt == "WEBP":
                    if data[8:12] == b"WEBP":
                        return "WEBP"
                    continue
                return fmt
    return None


class ImageValidator:
    """
    Validates an uploaded image before it enters the ML pipeline.

    Checks (in order):
    1. File size ≤ MAX_IMAGE_SIZE_MB
    2. Format matches allowed list (JPEG / PNG / WebP)
    3. Magic bytes confirm the declared format (prevents disguised files)
    4. Dimensions are within [MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION]
    5. Image is not corrupted (Pillow verify + OpenCV decode)
    6. Colour mode is RGB (grayscale / RGBA auto-conversion — warning only)
    """

    def __init__(self) -> None:
        self._max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        self._min_dim = settings.MIN_IMAGE_DIMENSION
        self._max_dim = settings.MAX_IMAGE_DIMENSION
        self._allowed_formats: list[str] = [
            f.upper() for f in settings.ALLOWED_IMAGE_FORMATS
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, file_bytes: bytes, filename: str) -> ValidationResult:
        """
        Run all validation checks against raw image bytes.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename:   Original filename (used for extension check and errors).

        Returns:
            ValidationResult with is_valid, errors, warnings, and metadata.

        Raises:
            InvalidImageError: When a hard failure occurs that cannot be
                               expressed as a soft validation result (e.g.
                               completely unreadable data).
        """
        result = ValidationResult()

        # 1. File size
        self._check_file_size(file_bytes, result)

        # 2 & 3. Format: extension + magic bytes
        detected_format = self._check_format(file_bytes, filename, result)

        # 4–6 require a decodable image — skip if already invalid to avoid
        # cascading errors from unreadable data.
        if result.is_valid and detected_format:
            pil_image = self._try_open_pillow(file_bytes, filename, result)
            if pil_image is not None:
                self._check_dimensions(pil_image, result)
                self._check_color_mode(pil_image, result)
                self._check_opencv_decodable(file_bytes, result)
                self._populate_metadata(pil_image, file_bytes, result)

        return result

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_file_size(self, data: bytes, result: ValidationResult) -> None:
        """Reject files that exceed the configured size limit."""
        size_bytes = len(data)
        if size_bytes > self._max_bytes:
            size_mb = size_bytes / (1024 * 1024)
            result.is_valid = False
            result.errors.append(
                f"File size {size_mb:.1f} MB exceeds the {settings.MAX_IMAGE_SIZE_MB} MB limit."
            )

    def _check_format(
        self, data: bytes, filename: str, result: ValidationResult
    ) -> str | None:
        """
        Verify format via both file extension and magic bytes.

        Why both?  Extension is user-controlled and trivially spoofed.
        Magic bytes are intrinsic to the file content. Requiring both
        ensures consistency and blocks disguised uploads.
        """
        ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else ""
        ext_map = {"JPG": "JPEG", "JPEG": "JPEG", "PNG": "PNG", "WEBP": "WEBP"}
        ext_format = ext_map.get(ext)

        if ext_format not in self._allowed_formats:
            result.is_valid = False
            result.errors.append(
                f"Extension '.{ext.lower()}' is not allowed. "
                f"Accepted: {', '.join(self._allowed_formats)}."
            )
            return None

        magic_format = _detect_format_from_magic(data)
        if magic_format is None:
            result.is_valid = False
            result.errors.append(
                "File header does not match any supported image format. "
                "The file may be corrupted or disguised."
            )
            return None

        if magic_format != ext_format:
            result.is_valid = False
            result.errors.append(
                f"File extension claims '{ext_format}' but file content "
                f"is '{magic_format}'. Possible disguised file."
            )
            return None

        return magic_format

    def _try_open_pillow(
        self, data: bytes, filename: str, result: ValidationResult
    ) -> Image.Image | None:
        """
        Attempt to open the image with Pillow.

        Pillow's verify() detects truncated files and certain corruption
        patterns. We open twice: once to verify, once to actually use,
        because verify() leaves the file pointer in an unusable state.
        """
        try:
            # First pass: verify integrity
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except (UnidentifiedImageError, Exception) as exc:
            result.is_valid = False
            result.errors.append(f"Image is corrupted or unreadable: {exc}")
            return None

        try:
            # Second pass: open for actual use (verify() exhausts the stream)
            img = Image.open(io.BytesIO(data))
            img.load()  # Force full decode to catch truncated images
            return img
        except Exception as exc:
            result.is_valid = False
            result.errors.append(f"Failed to decode image data: {exc}")
            return None

    def _check_dimensions(
        self, img: Image.Image, result: ValidationResult
    ) -> None:
        """Enforce minimum and maximum pixel dimension constraints."""
        w, h = img.size
        if w < self._min_dim or h < self._min_dim:
            result.is_valid = False
            result.errors.append(
                f"Image dimensions {w}×{h} are below the minimum "
                f"{self._min_dim}×{self._min_dim} px required for reliable detection."
            )
        if w > self._max_dim or h > self._max_dim:
            result.is_valid = False
            result.errors.append(
                f"Image dimensions {w}×{h} exceed the maximum "
                f"{self._max_dim}×{self._max_dim} px."
            )

    def _check_color_mode(
        self, img: Image.Image, result: ValidationResult
    ) -> None:
        """
        Warn (but allow) non-RGB images.

        Grayscale images lose colour information; RGBA images have
        transparency. Both can still be processed after conversion to RGB,
        but the user should know their image was modified.
        """
        mode = img.mode
        if mode == "L":
            result.warnings.append(
                "Image is grayscale (mode=L). It will be converted to RGB "
                "before processing; colour features will be absent."
            )
        elif mode == "RGBA":
            result.warnings.append(
                "Image has an alpha channel (mode=RGBA). The transparency "
                "layer will be discarded and the image converted to RGB."
            )
        elif mode not in ("RGB",):
            result.warnings.append(
                f"Unusual colour mode '{mode}'. The image will be converted "
                "to RGB; results may vary."
            )

    def _check_opencv_decodable(
        self, data: bytes, result: ValidationResult
    ) -> None:
        """
        Verify OpenCV can decode the image.

        Pillow and OpenCV use different decoders. An image that Pillow
        accepts may still fail cv2.imdecode (e.g. certain progressive JPEGs
        or exotic sub-formats). We confirm here so downstream modules don't
        hit unexpected decode errors.
        """
        arr = np.frombuffer(data, np.uint8)
        img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_cv is None:
            result.is_valid = False
            result.errors.append(
                "OpenCV could not decode the image. "
                "The file may use an unsupported sub-format."
            )

    def _populate_metadata(
        self,
        img: Image.Image,
        data: bytes,
        result: ValidationResult,
    ) -> None:
        """Attach extracted image properties to the result metadata dict."""
        w, h = img.size
        result.metadata = {
            "width": w,
            "height": h,
            "format": img.format or "unknown",
            "mode": img.mode,
            "size_kb": round(len(data) / 1024, 2),
            "aspect_ratio": round(w / h, 4) if h else 0,
        }
```

## Module 2: face_detector.py

```python
"""
Module 2 — Face Detection (Haar Cascade)

Background — how Haar Cascades work:
    A Haar Cascade is a machine-learned classifier trained using Viola-Jones
    (2001). It scans an image at multiple scales using a sliding window.
    At each position it applies hundreds of "weak" rectangle-based feature
    tests (Haar features) in a cascade: early stages quickly reject background
    regions; only promising windows reach later, more expensive stages.
    The result is fast (~10 ms on CPU), lightweight detection suitable for
    real-time use — ideal as a pipeline pre-stage before heavier ML work.

Key parameters:
    scale_factor  — how much the image is shrunk at each pyramid level.
                    Smaller values (e.g. 1.05) = more scales = more detections
                    but slower.  1.1 is the standard default.
    min_neighbors — how many overlapping detections must agree before a
                    region is accepted as a face.  Higher = fewer false
                    positives but may miss small or partially occluded faces.
    min_size      — smallest bounding box accepted (pixels).  Faces smaller
                    than this are likely noise or background textures.
"""

import base64
import io
import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel, computed_field

from app.core.config import settings
from app.ml.exceptions import ImageProcessingError, NoFaceDetectedError


class FaceBoundingBox(BaseModel):
    """
    Pixel coordinates of a detected face region.

    (x, y) is the top-left corner; width and height define the extent.
    """

    x: int
    y: int
    width: int
    height: int

    @computed_field  # type: ignore[misc]
    @property
    def area(self) -> int:
        """Pixel area of the bounding box — used to rank faces by size."""
        return self.width * self.height


class FaceDetectionResult(BaseModel):
    """
    Full output of a single face-detection pass.

    Attributes:
        face_count:          Total faces found.
        faces:               Bounding box for every detected face.
        largest_face:        The face with the greatest pixel area, or None.
        cropped_faces_b64:   Base64-encoded JPEG of each face crop (in
                             detection order).  Suitable for direct embedding
                             in JSON responses or <img src="data:..."> tags.
        processing_time_ms:  Wall-clock time for the detection step alone.
        image_dimensions:    Width and height of the input image.
    """

    face_count: int
    faces: list[FaceBoundingBox]
    largest_face: Optional[FaceBoundingBox]
    cropped_faces_b64: list[str]
    processing_time_ms: float
    image_dimensions: dict


class FaceDetector:
    """
    Wraps OpenCV's frontal-face Haar Cascade classifier.

    The cascade XML is bundled with opencv-python-headless and loaded once
    at construction time so every subsequent call skips the disk read.
    """

    def __init__(
        self,
        scale_factor: float | None = None,
        min_neighbors: int | None = None,
        min_face_size: int | None = None,
    ) -> None:
        """
        Load the Haar Cascade and confirm it initialised successfully.

        Args:
            scale_factor:   Image pyramid scale step (default from settings).
            min_neighbors:  Minimum overlapping detections required (default
                            from settings).
            min_face_size:  Minimum face side length in pixels (default from
                            settings).
        """
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        )
        self._cascade = cv2.CascadeClassifier(cascade_path)

        if self._cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from '{cascade_path}'. "
                "Ensure opencv-python-headless is installed correctly."
            )

        self._scale_factor = scale_factor or settings.HAAR_SCALE_FACTOR
        self._min_neighbors = min_neighbors or settings.HAAR_MIN_NEIGHBORS
        side = min_face_size or settings.HAAR_MIN_FACE_SIZE
        self._min_size = (side, side)

        logger.info(
            "FaceDetector initialised | scale_factor={} min_neighbors={} "
            "min_size={}",
            self._scale_factor,
            self._min_neighbors,
            self._min_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, image_bytes: bytes, strict: bool = False
    ) -> FaceDetectionResult:
        """
        Detect all frontal faces in the provided image.

        Pipeline:
        1. Decode bytes → numpy array via OpenCV.
        2. Convert BGR → grayscale (Haar operates on single-channel images).
        3. Run detectMultiScale with configured parameters.
        4. Crop each detected region from the *colour* image (better UX).
        5. Base64-encode crops as JPEG for JSON transport.

        Args:
            image_bytes: Raw image bytes (JPEG / PNG / WebP).
            strict:      If True, raise NoFaceDetectedError when no face is
                         found.  Default False — returns empty list instead.

        Returns:
            FaceDetectionResult containing all detection data.

        Raises:
            ImageProcessingError:  When OpenCV cannot decode the input.
            NoFaceDetectedError:   Only when strict=True and no face found.
        """
        t_start = time.perf_counter()

        bgr_image = self._decode_image(image_bytes)
        h, w = bgr_image.shape[:2]

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        detections = self._run_cascade(gray)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if len(detections) == 0:
            logger.warning("No face detected in image ({}×{}).", w, h)
            if strict:
                raise NoFaceDetectedError(
                    "No face detected in the image.",
                    details={"width": w, "height": h},
                )
            return FaceDetectionResult(
                face_count=0,
                faces=[],
                largest_face=None,
                cropped_faces_b64=[],
                processing_time_ms=round(elapsed_ms, 2),
                image_dimensions={"width": w, "height": h},
            )

        boxes = [
            FaceBoundingBox(x=int(x), y=int(y), width=int(bw), height=int(bh))
            for (x, y, bw, bh) in detections
        ]
        largest = max(boxes, key=lambda b: b.area)
        crops_b64 = [self._crop_and_encode(bgr_image, b) for b in boxes]

        logger.info(
            "Detected {} face(s) in {}×{} image in {:.1f} ms.",
            len(boxes),
            w,
            h,
            elapsed_ms,
        )

        return FaceDetectionResult(
            face_count=len(boxes),
            faces=boxes,
            largest_face=largest,
            cropped_faces_b64=crops_b64,
            processing_time_ms=round(elapsed_ms, 2),
            image_dimensions={"width": w, "height": h},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_image(self, data: bytes) -> np.ndarray:
        """
        Decode raw bytes to a BGR numpy array.

        OpenCV's imdecode returns None on failure rather than raising, so we
        check explicitly and raise a domain exception with context.
        """
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageProcessingError(
                "cv2.imdecode returned None — image bytes could not be decoded.",
                details={"bytes_length": len(data)},
            )
        return img

    def _run_cascade(self, gray: np.ndarray) -> np.ndarray:
        """
        Execute the Haar Cascade on a grayscale image.

        equalizeHist normalises contrast (e.g. overexposed or underlit
        faces), improving detection recall across varied lighting conditions.
        """
        equalized = cv2.equalizeHist(gray)
        detections = self._cascade.detectMultiScale(
            equalized,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        # detectMultiScale returns an empty tuple when nothing is found
        if not isinstance(detections, np.ndarray):
            return np.empty((0, 4), dtype=int)
        return detections

    def _crop_and_encode(
        self, bgr_image: np.ndarray, box: FaceBoundingBox
    ) -> str:
        """
        Crop a face region from the BGR image and return as base64 JPEG.

        The crop uses the colour (BGR) image so the encoded thumbnail looks
        natural to end users.  We add a small padding (10 %) around the
        bounding box to include forehead and chin, which aids downstream
        modules that rely on face geometry.
        """
        h, w = bgr_image.shape[:2]
        pad_x = int(box.width * 0.1)
        pad_y = int(box.height * 0.1)

        x1 = max(0, box.x - pad_x)
        y1 = max(0, box.y - pad_y)
        x2 = min(w, box.x + box.width + pad_x)
        y2 = min(h, box.y + box.height + pad_y)

        crop = bgr_image[y1:y2, x1:x2]
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("ascii")
```

## Module 3: image_preprocessor.py

```python
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
```

## Module 4: lbp_extractor.py

```python
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
```

## Module 5: dct_extractor.py

```python
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
```

## Module 6: feature_fusion.py

```python
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
```

## Module 7: kmeans_classifier.py

```python
"""
Module 7 — K-Means Clustering Classifier

Purpose: Cluster 1083-dimensional fused feature vectors (LBP + DCT) into two
groups representing REAL and MORPHED face images.

Why unsupervised (K-Means)?
    Labeled face morphing datasets are expensive to collect and annotate.
    K-Means can discover natural groupings in the feature space without ground
    truth labels — the assumption is that real and morphed faces produce
    statistically different LBP+DCT signatures and will cluster separately.

Confidence score:
    Distance-to-centroid is converted to a [0, 1] confidence value using:

        confidence = 1 / (1 + d / μ_cluster)

    where d is the sample's distance to its assigned centroid and μ_cluster is
    the mean distance of all *training* samples in that cluster.  This anchors
    "50% confidence" at the average cluster radius and provides an intuitive
    interpretation: samples sitting on the centroid score 1.0; samples at the
    fringe of the cluster score ~0.5; outliers score near 0.

Label assignment:
    After K-Means fit, cluster labels are assigned by majority vote using the
    ground-truth labels supplied at training time.  For synthetic training the
    correct labels are known; for production training a labelled validation set
    should be supplied.

    If no labels are provided, clusters are named "CLUSTER_0" / "CLUSTER_1".

NOTE — Synthetic data:
    When no saved model is found on startup, the classifier auto-trains on
    synthetic feature vectors (500 REAL + 500 MORPHED) so the system is
    immediately usable.  Replace with a real labelled dataset for production.
"""

import time
from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from pydantic import BaseModel
from sklearn.cluster import KMeans

from app.ml.exceptions import ImageProcessingError


_MODELS_DIR = Path(__file__).parent.parent / "models"
_MODEL_FILENAME = "kmeans_model.joblib"
_META_FILENAME = "kmeans_meta.joblib"

_N_CLUSTERS = 2
_RANDOM_STATE = 42
_FUSED_VECTOR_SIZE = 1083

# Synthetic training parameters
_N_SYNTHETIC_REAL = 500
_N_SYNTHETIC_MORPHED = 500
_REAL_STD = 0.05    # tight cluster — real faces have consistent texture
_MORPHED_STD = 0.15  # loose cluster — morphed faces vary more


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------


class PredictionResult(BaseModel):
    """Result of classifying a single 1083-dim feature vector."""

    prediction: str
    confidence: float
    cluster_id: int
    distance_to_centroid: float
    processing_time_ms: float


class TrainingResult(BaseModel):
    """Result of a K-Means training run."""

    samples_trained: int
    inertia: float
    iterations: int
    converged: bool
    cluster_labels: dict
    silhouette_score: float | None
    accuracy: float | None
    trained_on_synthetic: bool
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Current state of the K-Means model."""

    is_fitted: bool
    model_type: str
    n_clusters: int
    cluster_labels: dict
    training_samples: int | None
    inertia: float | None
    trained_on_synthetic: bool
    model_path: str


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class KMeansClassifier:
    """
    K-Means classifier for face morphing detection (Module 7).

    Usage
    -----
    The class is designed as a singleton (see main.py lifespan).  On
    construction it attempts to load a previously saved model from disk.
    If none exists it auto-trains on synthetic data so the ``/classify``
    endpoint is always available immediately.

    Public methods
    --------------
    train(feature_vectors, labels) → TrainingResult
        Fit a new K-Means model.  Saves model to disk on success.
    predict(feature_vector)        → PredictionResult
        Classify one 1083-dim vector.
    save_model()                   → bool
    load_model()                   → bool
    get_model_info()               → ModelInfo
    """

    def __init__(
        self,
        n_clusters: int = _N_CLUSTERS,
        random_state: int = _RANDOM_STATE,
        models_dir: Path | None = None,
    ) -> None:
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._models_dir: Path = models_dir or _MODELS_DIR

        self._kmeans: KMeans | None = None
        self._cluster_labels: dict[int, str] = {}
        self._mean_cluster_distances: dict[int, float] = {}
        self._training_samples: int | None = None
        self._trained_on_synthetic: bool = False
        self._prediction_history: list[dict] = []  # capped at 1000 entries

        self._models_dir.mkdir(parents=True, exist_ok=True)

        if not self.load_model():
            logger.info("No saved K-Means model found — training on synthetic data.")
            self._train_on_synthetic_data()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        feature_vectors: list[list[float]],
        labels: list[str] | None = None,
        trained_on_synthetic: bool = False,
    ) -> TrainingResult:
        """
        Fit K-Means on a list of 1083-dimensional fused feature vectors.

        Args:
            feature_vectors:      List of 1083-element fused vectors (from Module 6).
            labels:               Optional ground-truth labels ("REAL" or "MORPHED")
                                  for majority-vote cluster-label assignment.
            trained_on_synthetic: Set True when using generated data (logged/reported).

        Returns:
            TrainingResult with inertia, iterations, cluster labels, and accuracy.

        Raises:
            ImageProcessingError: When fewer than n_clusters samples provided, or a
                                  vector has the wrong length.
        """
        if len(feature_vectors) < self._n_clusters:
            raise ImageProcessingError(
                f"Need at least {self._n_clusters} samples to train K-Means, "
                f"got {len(feature_vectors)}.",
                details={"received": len(feature_vectors), "required": self._n_clusters},
            )
        for i, vec in enumerate(feature_vectors):
            if len(vec) != _FUSED_VECTOR_SIZE:
                raise ImageProcessingError(
                    f"Feature vector at index {i} has length {len(vec)}, "
                    f"expected {_FUSED_VECTOR_SIZE}.",
                    details={"index": i, "received_length": len(vec)},
                )

        t_start = time.perf_counter()
        X = np.array(feature_vectors, dtype=np.float64)

        self._kmeans = KMeans(
            n_clusters=self._n_clusters,
            random_state=self._random_state,
            n_init=10,
            max_iter=300,
        )
        assignments = self._kmeans.fit_predict(X)
        converged = self._kmeans.n_iter_ < self._kmeans.max_iter

        # Per-cluster mean distance to centroid (used for confidence at inference).
        self._mean_cluster_distances = {}
        for c in range(self._n_clusters):
            mask = assignments == c
            if mask.any():
                dists = np.linalg.norm(X[mask] - self._kmeans.cluster_centers_[c], axis=1)
                self._mean_cluster_distances[c] = float(dists.mean())
            else:
                self._mean_cluster_distances[c] = 1.0  # degenerate edge case

        # Silhouette score (requires at least 2 populated clusters with > 1 sample).
        silhouette: float | None = None
        unique_assigned = np.unique(assignments)
        if len(unique_assigned) >= 2 and len(feature_vectors) > self._n_clusters:
            try:
                from sklearn.metrics import silhouette_score as _sil
                silhouette = round(float(_sil(X, assignments)), 4)
            except Exception:
                pass

        # Label assignment by majority vote.
        self._cluster_labels = {}
        accuracy: float | None = None
        if labels is not None:
            if len(labels) != len(feature_vectors):
                raise ImageProcessingError(
                    "Number of labels must match number of feature vectors.",
                    details={"n_vectors": len(feature_vectors), "n_labels": len(labels)},
                )
            normalised = [lb.upper() for lb in labels]
            for c in range(self._n_clusters):
                mask = assignments == c
                cluster_gt = [normalised[i] for i, m in enumerate(mask) if m]
                real_count = cluster_gt.count("REAL")
                morphed_count = cluster_gt.count("MORPHED")
                self._cluster_labels[c] = "REAL" if real_count >= morphed_count else "MORPHED"
            predictions = [self._cluster_labels[int(a)] for a in assignments]
            correct = sum(p == gt for p, gt in zip(predictions, normalised))
            accuracy = round(correct / len(normalised), 4)
        else:
            for c in range(self._n_clusters):
                self._cluster_labels[c] = f"CLUSTER_{c}"

        self._training_samples = len(feature_vectors)
        self._trained_on_synthetic = trained_on_synthetic

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.save_model()

        logger.info(
            "K-Means trained | n={} k={} inertia={:.2f} iters={} sil={} acc={} synthetic={}",
            len(feature_vectors),
            self._n_clusters,
            self._kmeans.inertia_,
            self._kmeans.n_iter_,
            f"{silhouette:.4f}" if silhouette is not None else "N/A",
            f"{accuracy:.4f}" if accuracy is not None else "N/A",
            trained_on_synthetic,
        )

        return TrainingResult(
            samples_trained=self._training_samples,
            inertia=round(float(self._kmeans.inertia_), 4),
            iterations=int(self._kmeans.n_iter_),
            converged=converged,
            cluster_labels=self._cluster_labels,
            silhouette_score=silhouette,
            accuracy=accuracy,
            trained_on_synthetic=trained_on_synthetic,
            processing_time_ms=round(elapsed_ms, 3),
        )

    def predict(self, feature_vector: list[float]) -> PredictionResult:
        """
        Classify a single 1083-element fused feature vector.

        Args:
            feature_vector: 1083-element vector from FeatureFusion (Module 6).

        Returns:
            PredictionResult with prediction label, confidence, and metadata.

        Raises:
            ImageProcessingError: When the model is not fitted or vector length
                                  does not match expected 1083.
        """
        if self._kmeans is None:
            raise ImageProcessingError(
                "K-Means model is not fitted.  Call train() first or ensure a "
                "saved model exists at startup.",
                details={"model_path": str(self._models_dir / _MODEL_FILENAME)},
            )
        if len(feature_vector) != _FUSED_VECTOR_SIZE:
            raise ImageProcessingError(
                f"Feature vector has length {len(feature_vector)}, "
                f"expected {_FUSED_VECTOR_SIZE}.",
                details={"received_length": len(feature_vector), "expected": _FUSED_VECTOR_SIZE},
            )

        t_start = time.perf_counter()
        X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)

        cluster_id = int(self._kmeans.predict(X)[0])
        centroid = self._kmeans.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(X[0] - centroid))

        mean_dist = self._mean_cluster_distances.get(cluster_id, 1.0)
        confidence = 1.0 / (1.0 + distance / max(mean_dist, 1e-9))
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        label = self._cluster_labels.get(cluster_id, f"CLUSTER_{cluster_id}")
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        logger.debug(
            "K-Means predict | cluster={} label={} dist={:.4f} conf={:.3f} {:.2f}ms",
            cluster_id, label, distance, confidence, elapsed_ms,
        )

        result = PredictionResult(
            prediction=label,
            confidence=confidence,
            cluster_id=cluster_id,
            distance_to_centroid=round(distance, 4),
            processing_time_ms=round(elapsed_ms, 3),
        )
        self._record_prediction(result)
        return result

    def get_session_stats(self) -> dict:
        """Return live statistics from predictions made in the current session."""
        history = self._prediction_history
        total = len(history)
        if total == 0:
            return {
                "total_predictions": 0,
                "real_count": 0,
                "morphed_count": 0,
                "real_percentage": 0.0,
                "morphed_percentage": 0.0,
                "avg_confidence": 0.0,
            }
        real = sum(1 for p in history if p["prediction"] == "REAL")
        morphed = total - real
        avg_conf = sum(p["confidence"] for p in history) / total
        return {
            "total_predictions": total,
            "real_count": real,
            "morphed_count": morphed,
            "real_percentage": round(real / total * 100, 2),
            "morphed_percentage": round(morphed / total * 100, 2),
            "avg_confidence": round(avg_conf, 4),
        }

    def save_model(self) -> bool:
        """Persist the fitted K-Means model and metadata to disk."""
        if self._kmeans is None:
            return False
        try:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._kmeans, self._models_dir / _MODEL_FILENAME)
            meta = {
                "cluster_labels": self._cluster_labels,
                "mean_cluster_distances": self._mean_cluster_distances,
                "training_samples": self._training_samples,
                "trained_on_synthetic": self._trained_on_synthetic,
            }
            joblib.dump(meta, self._models_dir / _META_FILENAME)
            logger.info("K-Means model saved → {}", self._models_dir / _MODEL_FILENAME)
            return True
        except Exception as exc:
            logger.warning("Failed to save K-Means model: {}", exc)
            return False

    def load_model(self) -> bool:
        """Load K-Means model and metadata from disk if available."""
        model_path = self._models_dir / _MODEL_FILENAME
        meta_path = self._models_dir / _META_FILENAME
        if not model_path.exists():
            return False
        try:
            self._kmeans = joblib.load(model_path)
            if meta_path.exists():
                meta = joblib.load(meta_path)
                self._cluster_labels = meta.get("cluster_labels", {})
                self._mean_cluster_distances = meta.get("mean_cluster_distances", {})
                self._training_samples = meta.get("training_samples")
                self._trained_on_synthetic = meta.get("trained_on_synthetic", False)
            logger.info(
                "K-Means model loaded ← {} | labels={} synthetic={}",
                model_path,
                self._cluster_labels,
                self._trained_on_synthetic,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load K-Means model: {}", exc)
            self._kmeans = None
            return False

    def get_model_info(self) -> ModelInfo:
        """Return a snapshot of the current model state."""
        return ModelInfo(
            is_fitted=self._kmeans is not None,
            model_type="KMeans",
            n_clusters=self._n_clusters,
            cluster_labels=self._cluster_labels,
            training_samples=self._training_samples,
            inertia=round(float(self._kmeans.inertia_), 4) if self._kmeans else None,
            trained_on_synthetic=self._trained_on_synthetic,
            model_path=str(self._models_dir / _MODEL_FILENAME),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_prediction(self, result: PredictionResult) -> None:
        """Append to in-memory prediction history, capping at 1000 entries."""
        self._prediction_history.append({
            "prediction": result.prediction,
            "confidence": result.confidence,
        })
        if len(self._prediction_history) > 1000:
            self._prediction_history.pop(0)

    def _train_on_synthetic_data(self) -> TrainingResult:
        """
        Generate synthetic REAL and MORPHED feature vectors and train K-Means.

        Real vectors form a tight cluster (std=0.05) around a centroid in the
        middle of the [0,1] feature space.  Morphed vectors form a wider,
        overlapping cluster (std=0.15) around a centroid on the opposite side
        of the space.  This contrast gives K-Means a clear separation signal.

        NOTE: Replace this with a real labelled face dataset for production use.
        """
        rng = np.random.default_rng(_RANDOM_STATE)

        # Real centroid: values in [0.35, 0.65] — middle of feature space.
        real_centroid = rng.uniform(0.35, 0.65, size=_FUSED_VECTOR_SIZE)

        # Morphed centroid: mirrored so clusters are well-separated.
        morphed_centroid = np.clip(1.0 - real_centroid + rng.normal(0, 0.05, size=_FUSED_VECTOR_SIZE), 0.0, 1.0)

        # Generate samples and clip to valid [0, 1] range.
        real_X = np.clip(
            real_centroid + rng.normal(0, _REAL_STD, size=(_N_SYNTHETIC_REAL, _FUSED_VECTOR_SIZE)),
            0.0,
            1.0,
        )
        morphed_X = np.clip(
            morphed_centroid + rng.normal(0, _MORPHED_STD, size=(_N_SYNTHETIC_MORPHED, _FUSED_VECTOR_SIZE)),
            0.0,
            1.0,
        )

        X = np.vstack([real_X, morphed_X])
        labels = ["REAL"] * _N_SYNTHETIC_REAL + ["MORPHED"] * _N_SYNTHETIC_MORPHED

        result = self.train(X.tolist(), labels=labels, trained_on_synthetic=True)
        logger.info(
            "Synthetic K-Means training complete | inertia={:.2f} labels={}",
            result.inertia,
            self._cluster_labels,
        )
        return result
```

## Module 8: evaluator.py

```python
"""
Module 8 — Classification Evaluation

Purpose: Compute standard binary-classification metrics for the face morphing
detection system and produce a viva-ready evaluation report.

Label convention used throughout this module
─────────────────────────────────────────────
  0 = REAL   (the class we want to *accept*)
  1 = MORPHED (the class we want to *reject*)

Confusion-matrix layout
───────────────────────
                   Predicted REAL  Predicted MORPHED
  Actual MORPHED      FP               TN
  Actual REAL         TP               FN

  → [[TN, FP],
     [FN, TP]]

This layout keeps the security-domain definitions consistent:
  • FP  = a MORPHED face incorrectly *accepted* as REAL   ← "False Acceptance"
  • FN  = a REAL face incorrectly *rejected* as MORPHED   ← "False Rejection"

Metric formulas
───────────────
  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  FAR       = FP / (FP + TN)   — False Acceptance Rate (morphed slips through)
  FRR       = FN / (FN + TP)   — False Rejection Rate  (real user blocked)
  Precision = TP / (TP + FP)   — of all "REAL" predictions, how many are correct
  Recall    = TP / (TP + FN)   — of all actual REAL faces, how many we accepted
  F1 Score  = 2 × Precision × Recall / (Precision + Recall)

Real-world stakes
─────────────────
  FAR matters in border control, banking, or document verification — a high FAR
  means attackers slip through.  FRR matters for user experience — a high FRR
  means genuine users are blocked.  FAR and FRR trade off: tightening the
  classifier to reduce FAR typically raises FRR.  The F1 score balances both.

Synthetic evaluation note
─────────────────────────
  Because no labelled real-world face-morphing dataset is bundled with this
  project, ``evaluate_on_synthetic()`` generates 200 test vectors drawn from
  the same distribution used during K-Means training (100 REAL tight cluster,
  100 MORPHED loose cluster) but with a different random seed.  Replace with a
  real holdout dataset for production use.
"""

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from app.ml.clustering.kmeans_classifier import KMeansClassifier


_FUSED_VECTOR_SIZE = 1083
_N_SYNTHETIC_TEST_REAL = 100
_N_SYNTHETIC_TEST_MORPHED = 100
_TEST_REAL_STD = 0.05
_TEST_MORPHED_STD = 0.15
_TEST_RANDOM_SEED = 99  # different from training seed (42)


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """All binary-classification metrics for one evaluation run."""

    accuracy: float
    far: float
    frr: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]
    total_samples: int
    correct_predictions: int
    tp: int
    tn: int
    fp: int
    fn: int
    evaluation_time_ms: float


class EvaluationReport(BaseModel):
    """Full viva-ready report wrapping metrics with interpretation and context."""

    metrics: EvaluationResult
    model_info: dict
    interpretation: dict
    recommendations: list[str]
    timestamp: str
    data_source: str


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """
    Module 8 — evaluates the K-Means classifier and produces an EvaluationReport.

    Usage
    -----
    Constructed with a fitted KMeansClassifier.  Call ``generate_report()``
    once at startup to cache the evaluation results.

    Public methods
    --------------
    evaluate(y_true, y_pred) → EvaluationResult
        Compute all metrics from pre-computed label arrays.
    evaluate_on_synthetic()  → EvaluationResult
        Generate synthetic test vectors, classify them, then evaluate.
    generate_report()        → EvaluationReport
        Full viva-ready report (calls evaluate_on_synthetic internally).
    """

    def __init__(self, classifier: "KMeansClassifier") -> None:
        self._classifier = classifier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: list[int],
        y_pred: list[int],
    ) -> EvaluationResult:
        """
        Compute all metrics from ground-truth and predicted label arrays.

        Args:
            y_true: Ground-truth labels — 0=REAL, 1=MORPHED.
            y_pred: Predicted labels   — 0=REAL, 1=MORPHED.

        Returns:
            EvaluationResult with accuracy, FAR, FRR, precision, recall,
            F1, confusion matrix, and raw TP/TN/FP/FN counts.

        Raises:
            ValueError: When y_true and y_pred have different lengths or are empty.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length "
                f"(got {len(y_true)} and {len(y_pred)})."
            )
        if not y_true:
            raise ValueError("y_true must not be empty.")

        t_start = time.perf_counter()

        # --- raw counts ---------------------------------------------------
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)  # REAL → REAL
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)  # MORPHED → MORPHED
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)  # MORPHED → REAL (false accept)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)  # REAL → MORPHED (false reject)

        n = tp + tn + fp + fn

        # --- metrics ------------------------------------------------------
        accuracy  = (tp + tn) / n if n > 0 else 0.0
        far       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "Evaluation | n={} acc={:.3f} FAR={:.3f} FRR={:.3f} F1={:.3f} ({:.1f}ms)",
            n, accuracy, far, frr, f1, elapsed_ms,
        )

        return EvaluationResult(
            accuracy=round(accuracy, 4),
            far=round(far, 4),
            frr=round(frr, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            confusion_matrix=[[tn, fp], [fn, tp]],
            total_samples=n,
            correct_predictions=tp + tn,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            evaluation_time_ms=round(elapsed_ms, 3),
        )

    def evaluate_on_synthetic(self) -> EvaluationResult:
        """
        Generate 200 synthetic test vectors, classify each with K-Means,
        and compute evaluation metrics.

        The test vectors are drawn from the *same* Gaussian distributions
        used in training (tight for REAL, loose for MORPHED) but with a
        different random seed (99 vs 42) so there is no data leakage.

        Returns:
            EvaluationResult computed against the synthetic test labels.
        """
        kmeans = self._classifier._kmeans
        if kmeans is None:
            raise RuntimeError("KMeansClassifier is not fitted — cannot evaluate.")

        rng = np.random.default_rng(_TEST_RANDOM_SEED)

        real_cluster_id = self._label_to_cluster_id("REAL")
        morphed_cluster_id = self._label_to_cluster_id("MORPHED")

        real_centroid = kmeans.cluster_centers_[real_cluster_id]
        morphed_centroid = kmeans.cluster_centers_[morphed_cluster_id]

        # Test vectors — same spread as training but independent seed
        real_test = np.clip(
            real_centroid + rng.normal(0, _TEST_REAL_STD, (_N_SYNTHETIC_TEST_REAL, _FUSED_VECTOR_SIZE)),
            0.0, 1.0,
        )
        morphed_test = np.clip(
            morphed_centroid + rng.normal(0, _TEST_MORPHED_STD, (_N_SYNTHETIC_TEST_MORPHED, _FUSED_VECTOR_SIZE)),
            0.0, 1.0,
        )

        y_true: list[int] = [0] * _N_SYNTHETIC_TEST_REAL + [1] * _N_SYNTHETIC_TEST_MORPHED
        y_pred: list[int] = []

        for vec in real_test:
            pred = self._classifier.predict(vec.tolist())
            y_pred.append(0 if pred.prediction == "REAL" else 1)

        for vec in morphed_test:
            pred = self._classifier.predict(vec.tolist())
            y_pred.append(0 if pred.prediction == "REAL" else 1)

        return self.evaluate(y_true, y_pred)

    def generate_report(self) -> EvaluationReport:
        """
        Produce a complete viva-ready evaluation report.

        Runs ``evaluate_on_synthetic()`` internally, then wraps the metrics
        with plain-English interpretation, model context, and recommendations.

        Returns:
            EvaluationReport ready to serve from the /evaluation endpoint.
        """
        metrics = self.evaluate_on_synthetic()
        info = self._classifier.get_model_info()

        model_info = {
            "type": "K-Means Clustering",
            "k": info.n_clusters,
            "features": f"LBP (59) + DCT (1024) = {_FUSED_VECTOR_SIZE} dimensions",
            "training_samples": info.training_samples,
            "training_data": (
                f"Synthetic ({info.training_samples // 2} REAL + "
                f"{info.training_samples // 2} MORPHED samples)"
                if info.trained_on_synthetic
                else "Real labelled face images"
            ),
            "cluster_labels": info.cluster_labels,
            "inertia": info.inertia,
        }

        interpretation = self._build_interpretation(metrics)
        recommendations = self._build_recommendations(metrics)

        return EvaluationReport(
            metrics=metrics,
            model_info=model_info,
            interpretation=interpretation,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data_source="synthetic",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label_to_cluster_id(self, label: str) -> int:
        for cid, lbl in self._classifier._cluster_labels.items():
            if lbl == label:
                return cid
        raise RuntimeError(
            f"Label '{label}' not found in cluster labels: {self._classifier._cluster_labels}"
        )

    @staticmethod
    def _build_interpretation(m: EvaluationResult) -> dict:
        far_1_in_n = f"1 in {int(round(1 / m.far))}" if m.far > 0 else "none"
        frr_1_in_n = f"1 in {int(round(1 / m.frr))}" if m.frr > 0 else "none"

        f1_quality = (
            "Excellent balance" if m.f1_score >= 0.90
            else "Good balance" if m.f1_score >= 0.80
            else "Moderate balance" if m.f1_score >= 0.70
            else "Poor balance — model needs retraining"
        )

        return {
            "accuracy": (
                f"{m.accuracy * 100:.1f}% of test images correctly classified "
                f"({m.correct_predictions}/{m.total_samples})"
            ),
            "far": (
                f"{m.far * 100:.1f}% of morphed faces incorrectly accepted as real "
                f"({far_1_in_n} morphed images slips through). "
                "In a border-control context this is the primary security metric."
            ),
            "frr": (
                f"{m.frr * 100:.1f}% of real faces incorrectly rejected as morphed "
                f"({frr_1_in_n} genuine users blocked). "
                "A high FRR harms user experience."
            ),
            "precision": (
                f"{m.precision * 100:.1f}% of images predicted as REAL are actually real. "
                "High precision = few false alarms."
            ),
            "recall": (
                f"{m.recall * 100:.1f}% of all real faces are correctly accepted. "
                "High recall = few real users blocked."
            ),
            "f1_score": (
                f"{f1_quality} between precision and recall "
                f"(F1 = {m.f1_score:.3f}). "
                "F1 is the harmonic mean — useful when classes are imbalanced."
            ),
            "confusion_matrix": (
                f"[[TN={m.tn}, FP={m.fp}], [FN={m.fn}, TP={m.tp}]] — "
                f"rows = actual label, columns = predicted label. "
                f"FP (top-right) drives FAR; FN (bottom-left) drives FRR."
            ),
        }

    @staticmethod
    def _build_recommendations(m: EvaluationResult) -> list[str]:
        recs: list[str] = []

        if m.far > 0.20:
            recs.append(
                "FAR is high (> 20%). Consider increasing Haar min_neighbors in face "
                "detection to only process high-confidence face crops, or collect more "
                "diverse morphed training examples."
            )
        if m.frr > 0.20:
            recs.append(
                "FRR is high (> 20%). The REAL cluster may be too tight. Consider "
                "increasing the std of synthetic REAL training vectors or adding "
                "data augmentation."
            )
        if m.accuracy < 0.70:
            recs.append(
                "Accuracy is below 70%. Replace synthetic training data with a real "
                "labelled face-morphing dataset (e.g., MorGAN or SMDD benchmarks) for "
                "significant performance gains."
            )
        if m.f1_score >= 0.80:
            recs.append(
                "F1 score is good. For further improvement, tune the DCT block size "
                "(currently 32×32) or experiment with additional feature descriptors "
                "such as HOG or SIFT."
            )
        if not recs:
            recs.append(
                "Metrics look healthy. For production deployment, validate on a real "
                "holdout dataset and monitor FAR/FRR in live traffic."
            )

        return recs
```
