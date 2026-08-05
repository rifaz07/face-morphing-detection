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
| 7 | K-Means Clustering | ✅ Complete | `POST /api/v1/detection/classify` |
| 8 | Classification & Evaluation | ✅ Complete | `GET /api/v1/detection/evaluation` |

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

### Evaluation Metrics (Module 8)

`GET /api/v1/detection/evaluation` is computed on 200 synthetic test samples
(100 REAL + 100 MORPHED, different seed to training) — it evaluates whatever
model is currently loaded, synthetic or real. For metrics from the real-data
training run, see [Training on Real Data](#training-on-real-data) below or
`ml/checkpoints/real_evaluation_report.json`.

| Metric | Formula | What it means |
|--------|---------|---------------|
| **Accuracy** | (TP+TN) / N | Overall correct classification rate |
| **FAR** | FP / (FP+TN) | Morphed faces incorrectly accepted as real — primary security metric |
| **FRR** | FN / (FN+TP) | Real faces incorrectly rejected as morphed — affects user experience |
| **Precision** | TP / (TP+FP) | Of all REAL predictions, fraction actually real |
| **Recall** | TP / (TP+FN) | Of all real faces, fraction correctly accepted |
| **F1 Score** | 2·P·R / (P+R) | Harmonic mean of precision and recall |

Confusion matrix layout: `[[TN, FP], [FN, TP]]` — FP drives FAR, FN drives FRR.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/detection/evaluation` | Cached report (Accuracy, FAR, FRR, F1, confusion matrix + interpretation) |
| `GET /api/v1/detection/evaluation/live` | Live session stats (predictions since startup) |
| `POST /api/v1/detection/evaluation/run` | Trigger fresh evaluation, update cache |

**Get evaluation report:**
```bash
curl http://localhost:8000/api/v1/detection/evaluation
```

**Run fresh evaluation:**
```bash
curl -X POST http://localhost:8000/api/v1/detection/evaluation/run
```

### K-Means Clustering (Module 7)

K-Means (k=2) partitions the 1083-dimensional fused feature space into two
clusters — one for **REAL** faces and one for **MORPHED** faces.  After
fitting, cluster labels are assigned by majority vote against ground-truth
labels.  At inference, the sample's Euclidean distance to its assigned
centroid is converted to a confidence score:

```
confidence = 1 / (1 + distance / mean_cluster_distance)
```

`mean_cluster_distance` is the average distance of training samples to their
centroid, so confidence 1.0 means "on the centroid" and ~0.5 means "at the
edge of the cluster".

**Current training data:** trained on a 4,000-image sample (2,000 REAL + 2,000
MORPHED) from the real Kaggle SSMD face-morphing dataset — see
[Training on Real Data](#training-on-real-data) below. Falls back to 500
synthetic REAL + 500 synthetic MORPHED vectors only when no saved model
exists on disk.

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/detection/classify` | **Main endpoint** — full pipeline → REAL/MORPHED + confidence |
| `GET  /api/v1/detection/model-info` | K-Means model state & metadata |
| `POST /api/v1/detection/retrain` | Retrain on synthetic data |

### Training on Real Data

`backend/scripts/train_on_real_dataset.py` replaces the synthetic bootstrap
model with one trained on the real Kaggle SSMD dataset. Every sampled image
is pushed through the full Modules 1–6 pipeline (validate → detect →
preprocess → LBP → DCT → fuse) to build a 1083-dim feature vector, exactly
as `/classify` does at inference time. Images that fail any step (no face
detected, corrupt file, validation failure) are logged and skipped — they
never abort the run.

**Prerequisites:**
1. Dataset present on the host at `ml/dataset/real/*.png` and
   `ml/dataset/morphed/*.png` (gitignored — not committed to the repo).
2. `docker-compose.yml` mounts `./ml` read-only into the backend container
   at `/app/ml_data`, plus a read-write overlay at
   `/app/ml_data/checkpoints` so the script can save its outputs. Rebuild
   after pulling this change: `docker-compose up -d --build backend`.

**Sample run (validation — 4,000 images, ~5 min):**
```bash
docker exec fmd-backend python scripts/train_on_real_dataset.py --sample-per-class 2000
```

**Full run (all ~40,000 images, once the sample run is validated):**
```bash
docker exec fmd-backend python scripts/train_on_real_dataset.py --full
```

`--sample-per-class N` samples N images from **each** of `real/` and
`morphed/` using `random.sample(seed=42)` for reproducibility; `--full`
ignores the sample size and uses every image in both folders.

**What the script does:**
1. Lists and samples files from `real/` and `morphed/`.
2. Runs each sampled image through the ML pipeline, logging progress every
   100 images with a running success/fail count and ETA.
3. Saves the extracted feature vectors to
   `ml/checkpoints/features_sample.npz` (`X`, `y`, `failed_files`,
   `sample_size`).
4. Splits 80/20 (stratified, `random_state=42`), trains `KMeansClassifier`
   on the training split (overwriting `kmeans_model.joblib` /
   `kmeans_meta.joblib`), and evaluates on the held-out real test split.
5. Writes `ml/checkpoints/real_evaluation_report.json` with the full metric
   set and `"data_source": "real_kaggle_ssmd_set_sample"`.

**After training, restart the backend to load the new model:**
```bash
docker-compose restart backend
curl http://localhost:8000/api/v1/detection/model-info
```

**Result of the first 4,000-image validation run** (2,000 REAL + 2,000
MORPHED sampled, 3,989 processed successfully, 11 skipped — no face
detected):

| Metric | Value |
|---|---|
| Accuracy | 71.18% |
| FAR (morphed accepted as real) | 50.25% |
| FRR (real rejected as morphed) | 7.29% |
| Precision | 64.74% |
| Recall | 92.71% |
| F1 Score | 0.7624 |

The high FAR shows that raw LBP+DCT features with unsupervised K-Means do
not cleanly separate this dataset's morphed faces from real ones — a
realistic result, unlike the synthetic data which was constructed to be
trivially separable. This is expected for a first pass on real data and is
the reason the task plan calls for scaling up to the full dataset next.

### Example curl commands

**Classify a face image (MAIN endpoint — Modules 1–7):**
```bash
curl -X POST http://localhost:8000/api/v1/detection/classify \
  -F "file=@/path/to/face.jpg"
```

**Get model info:**
```bash
curl http://localhost:8000/api/v1/detection/model-info
```

**Retrain model:**
```bash
curl -X POST http://localhost:8000/api/v1/detection/retrain
```

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
