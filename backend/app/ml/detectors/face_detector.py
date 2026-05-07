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
