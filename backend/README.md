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
