# FaceGuard Backend — FastAPI ML Detection API

Production-grade FastAPI backend for the Face Morphing Detection system.

## Stack

- **FastAPI** — async web framework
- **SQLAlchemy 2.0** — ORM with PostgreSQL
- **Alembic** — database migrations
- **Pydantic v2** — data validation and settings
- **Loguru** — structured logging
- **Uvicorn** — ASGI server
- **OpenCV + scikit-learn + scikit-image** — ML pipeline (modules added later)

---

## Running Locally (with Docker — recommended)

```bash
# From the repo root
docker compose up -d --build backend

# View logs
docker compose logs -f backend

# Run tests inside the container
docker exec fmd-backend pytest
```

API available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/api/v1/health

---

## Running Locally (without Docker)

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements-dev.txt

# Copy env file and set DATABASE_URL to your local postgres
cp .env.example .env
# Edit .env — change DATABASE_URL host to localhost and port to 5435

# Start server with hot-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Folder Structure

```
app/
├── core/
│   ├── config.py       → Pydantic settings (reads env vars)
│   ├── database.py     → SQLAlchemy engine + session dependency
│   └── logging.py      → Loguru configuration
├── api/
│   ├── deps.py         → Shared FastAPI dependencies
│   └── v1/
│       ├── router.py   → Aggregated v1 router
│       └── endpoints/
│           └── health.py  → GET /api/v1/health
├── models/
│   └── base.py         → SQLAlchemy declarative base
├── schemas/
│   └── health.py       → Pydantic response schemas
├── services/           → Business logic layer
└── ml/                 → ML pipeline modules (added in later tasks)
```

---

## ML Pipeline Modules

| # | Module | Status | Endpoint |
|---|--------|--------|---------|
| 1 | Image Validation | ✅ Complete | `POST /api/v1/detection/validate` |
| 2 | Face Detection — Haar Cascade | ✅ Complete | `POST /api/v1/detection/detect-face` |
| 3 | Preprocessing (resize / grayscale / normalise) | ✅ Complete | `POST /api/v1/detection/preprocess` |
| 4 | LBP Feature Extraction | ✅ Complete | `POST /api/v1/detection/extract-lbp` |
| 5 | DCT Feature Extraction | ✅ Complete | `POST /api/v1/detection/extract-dct` |
| 6 | Feature Fusion (LBP + DCT) | ✅ Complete | `POST /api/v1/detection/extract-features` |
| 7 | K-Means Clustering | Pending | — |
| 8 | Classification & Evaluation | Pending | — |

### Feature Fusion (Module 6)

Combines the LBP vector (59 values, already in [0,1]) and the DCT vector
(1024 values, log-compressed) into a single 1083-dimensional descriptor:

| Component | Size | Share |
|---|---|---|
| LBP (texture) | 59 | 5.45 % |
| DCT (frequency, MinMax-normalised) | 1024 | 94.55 % |
| **Fused vector** | **1083** | **100 %** |

DCT is MinMax-normalised to [0,1] before concatenation so neither component
dominates the Euclidean distance used in K-Means clustering.

**Extract full features (Modules 1–6):**
```bash
curl -X POST http://localhost:8000/api/v1/detection/extract-features \
  -F "file=@/path/to/face.jpg"
```

### DCT Feature Extraction

DCT (Discrete Cosine Transform) converts the face image from the spatial
domain (pixels) into the frequency domain (cosine wave amplitudes).  The
top-left 32×32 block (1024 coefficients) captures low-to-mid frequencies
where morphing artefacts are most prominent — pixel blending introduces
energy in mid-frequency bands not present in real faces.  Log compression
(`sign(x) × log(1 + |x|)`) normalises the huge dynamic range of raw DCT
values.  DCT complements LBP: LBP measures local texture patterns; DCT
measures global frequency content.

### LBP Feature Extraction

LBP (Local Binary Pattern) compares each pixel with its 8 neighbours on a
circle of radius 1.  Each comparison yields one bit; the 8-bit code for a
pixel represents its local texture.  Collecting a normalised histogram of
all LBP codes (59 bins for the uniform variant) gives the **texture
fingerprint** of the face.  Morphed images produce statistically different
fingerprints from real images, which is what the classifier exploits.

### Example curl commands

**Validate image:**
```bash
curl -X POST http://localhost:8000/api/v1/detection/validate \
  -F "file=@/path/to/face.jpg"
```

**Detect faces + preprocess:**
```bash
curl -X POST http://localhost:8000/api/v1/detection/detect-face \
  -F "file=@/path/to/face.jpg"
```

**Preprocess largest face (debug/demo):**
```bash
curl -X POST http://localhost:8000/api/v1/detection/preprocess \
  -F "file=@/path/to/face.jpg"
```

**Extract LBP features:**
```bash
curl -X POST http://localhost:8000/api/v1/detection/extract-lbp \
  -F "file=@/path/to/face.jpg"
```

**Extract DCT features:**
```bash
curl -X POST http://localhost:8000/api/v1/detection/extract-dct \
  -F "file=@/path/to/face.jpg"
```

---

## Adding a New Endpoint

1. Create `app/api/v1/endpoints/your_endpoint.py`
2. Define an `APIRouter` and add route functions
3. Import and include the router in `app/api/v1/router.py`

---

## Database Migrations (Alembic)

```bash
# Inside Docker
docker exec fmd-backend alembic revision --autogenerate -m "your message"
docker exec fmd-backend alembic upgrade head

# Check current revision
docker exec fmd-backend alembic current

# Downgrade one step
docker exec fmd-backend alembic downgrade -1
```

---

## Testing

```bash
# Inside Docker
docker exec fmd-backend pytest

# With coverage report
docker exec fmd-backend pytest --cov=app --cov-report=term-missing

# Locally (activate venv first)
pytest
```

---

## Code Quality

```bash
# Format
black app tests

# Lint
ruff check app tests

# Sort imports
isort app tests

# Type check
mypy app
```
