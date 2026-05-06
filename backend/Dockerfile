# ============================================================
# Stage 1 — builder: install dependencies into a virtual env
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools needed by C extensions (psycopg2)
# opencv-python-headless needs libglib2.0-0 at runtime but no GL libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Create venv inside builder stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# Stage 2 — runtime: lean production image
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY . .

# Run as non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
