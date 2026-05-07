import io

import numpy as np
import pytest
import urllib.request
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Synchronous TestClient — no real DB required for unit tests."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Image fixture helpers (shared by ml/ tests and endpoint tests)
# ---------------------------------------------------------------------------


def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _make_synthetic_face(width: int = 300, height: int = 300) -> Image.Image:
    arr = np.full((height, width, 3), 200, dtype=np.uint8)
    fy, fh = int(height * 0.1), int(height * 0.8)
    fx, fw = int(width * 0.15), int(width * 0.7)
    arr[fy : fy + fh, fx : fx + fw] = [220, 190, 160]
    ey, eh = int(height * 0.3), int(height * 0.1)
    ex, ew = int(width * 0.25), int(width * 0.18)
    arr[ey : ey + eh, ex : ex + ew] = [30, 30, 30]
    ex2 = int(width * 0.57)
    arr[ey : ey + eh, ex2 : ex2 + ew] = [30, 30, 30]
    ny, nh = int(height * 0.44), int(height * 0.15)
    nx, nw = int(width * 0.43), int(width * 0.14)
    arr[ny : ny + nh, nx : nx + nw] = [230, 200, 175]
    my, mh = int(height * 0.63), int(height * 0.07)
    mx, mw = int(width * 0.33), int(width * 0.34)
    arr[my : my + mh, mx : mx + mw] = [80, 40, 40]
    return Image.fromarray(arr, mode="RGB")


def _download_face_image() -> bytes | None:
    url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pytest-fixture"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read()
    except Exception:
        return None


@pytest.fixture(scope="session")
def valid_face_image_bytes() -> bytes:
    downloaded = _download_face_image()
    if downloaded:
        img = Image.open(io.BytesIO(downloaded))
        return _pil_to_jpeg_bytes(img)
    return _pil_to_jpeg_bytes(_make_synthetic_face(300, 300))


@pytest.fixture(scope="session")
def valid_face_image_png_bytes() -> bytes:
    downloaded = _download_face_image()
    if downloaded:
        img = Image.open(io.BytesIO(downloaded))
        return _pil_to_png_bytes(img)
    return _pil_to_png_bytes(_make_synthetic_face(300, 300))


@pytest.fixture(scope="session")
def random_noise_image_bytes() -> bytes:
    rng = np.random.default_rng(seed=42)
    noise = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    return _pil_to_jpeg_bytes(Image.fromarray(noise, mode="RGB"))


@pytest.fixture(scope="session")
def oversized_image_bytes() -> bytes:
    arr = np.zeros((3000, 3000, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100, subsampling=0)
    data = buf.getvalue()
    if len(data) < 10 * 1024 * 1024:
        data = data * (10 * 1024 * 1024 // len(data) + 2)
    return data


@pytest.fixture(scope="session")
def invalid_format_bytes() -> bytes:
    return b"This is not an image file at all."


@pytest.fixture(scope="session")
def tiny_image_bytes() -> bytes:
    img = Image.new("RGB", (50, 50), color=(128, 128, 128))
    return _pil_to_jpeg_bytes(img)


@pytest.fixture(scope="session")
def grayscale_image_bytes() -> bytes:
    arr = np.full((200, 200), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def two_face_image_bytes() -> bytes:
    canvas = np.full((400, 800, 3), 200, dtype=np.uint8)

    def _draw_face(arr: np.ndarray, ox: int, oy: int, w: int, h: int) -> None:
        arr[oy : oy + h, ox : ox + w] = [220, 190, 160]
        ew, eh = w // 5, h // 10
        arr[oy + h // 3 : oy + h // 3 + eh, ox + w // 4 : ox + w // 4 + ew] = [30, 30, 30]
        arr[oy + h // 3 : oy + h // 3 + eh, ox + w // 2 : ox + w // 2 + ew] = [30, 30, 30]
        mh = h // 10
        arr[
            oy + int(h * 0.65) : oy + int(h * 0.65) + mh,
            ox + w // 3 : ox + w // 3 + w // 3,
        ] = [80, 40, 40]

    _draw_face(canvas, 20, 50, 200, 300)
    _draw_face(canvas, 480, 25, 280, 350)
    return _pil_to_jpeg_bytes(Image.fromarray(canvas, mode="RGB"))
