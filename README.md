# Clustering Based Face Morphing Detection System

> A production-grade web application for detecting morphed face images using LBP + DCT feature extraction and K-Means clustering.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/rifaz07/face-morphing-detection/ci.yml?branch=main&label=build)](https://github.com/rifaz07/face-morphing-detection/actions)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)

---

## About

This system leverages a classical machine learning pipeline to identify morphed facial images — a critical tool in preventing biometric fraud. Users upload a face image and receive a real-time verdict (REAL or MORPHED) along with a confidence score derived from the clustering model.

**Key capabilities:**
- Automated face detection using OpenCV Haar Cascades
- Texture analysis via Local Binary Patterns (LBP)
- Frequency-domain analysis via Discrete Cosine Transform (DCT)
- Unsupervised K-Means clustering for classification
- Full-stack SaaS interface with authentication, upload history, and analytics

---

## Tech Stack

| Layer          | Technology                                          |
|----------------|-----------------------------------------------------|
| Frontend       | Next.js 15, Tailwind CSS v4, shadcn/ui, Framer Motion |
| Backend        | FastAPI, Python 3.11, OpenCV, scikit-learn          |
| ML Pipeline    | LBP, DCT, K-Means Clustering                        |
| Database       | PostgreSQL (Supabase), Prisma ORM, SQLAlchemy       |
| Storage        | Cloudinary                                          |
| Auth           | Clerk (Google + Email)                              |
| Deployment     | Vercel (frontend), Render (backend), Supabase (DB)  |
| DevOps         | Docker, GitHub Actions, Sentry, Husky               |

---

## Folder Structure

```
face-morphing-detection/
├── frontend/              # Next.js 15 web application
│   ├── app/               # App Router pages and layouts
│   ├── components/        # Reusable UI components
│   ├── lib/               # Utilities, API clients, helpers
│   └── public/            # Static assets
├── backend/               # FastAPI ML API
│   ├── app/               # Route handlers and core logic
│   ├── ml/                # ML pipeline modules
│   └── Dockerfile         # Container definition for Render
├── ml/                    # Training scripts and notebooks
│   ├── notebooks/         # Jupyter exploration notebooks
│   ├── scripts/           # Standalone training scripts
│   └── artifacts/         # Saved model files (.pkl, .joblib)
├── .github/
│   └── workflows/         # GitHub Actions CI/CD pipelines
├── docker-compose.yml     # Local multi-service orchestration
├── CLAUDE.md              # Full project context for Claude Code
├── .env.example           # Environment variable template
├── .gitignore
└── README.md
```

---

## ML Pipeline

The detection system runs an 8-stage pipeline:

```
Image Upload
    │
    ▼
[1] Input Validation      → Check format, size, dimensions
    │
    ▼
[2] Face Detection        → OpenCV Haar Cascade
    │
    ▼
[3] Preprocessing         → Resize 128×128, grayscale, normalize
    │
    ▼
[4] LBP Extraction        → Local Binary Pattern texture features
    │
    ▼
[5] DCT Extraction        → Frequency-domain feature vector
    │
    ▼
[6] Feature Fusion        → Concatenate LBP + DCT vectors
    │
    ▼
[7] K-Means Clustering    → Assign to Real / Morphed cluster
    │
    ▼
[8] Classification        → Verdict + Confidence + FAR/FRR metrics
```

---

## Setup Instructions

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | 4.x+ | Runs PostgreSQL + pgAdmin locally |
| [Node.js](https://nodejs.org/) | 20+ | Frontend development |
| [Python](https://www.python.org/) | 3.11+ | Backend / ML API |
| [Git](https://git-scm.com/) | any | Version control |

### Step 1 — Clone the repo

```bash
git clone https://github.com/rifaz07/face-morphing-detection.git
cd face-morphing-detection
```

### Step 2 — Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your values (passwords, API keys, etc.)
```

### Step 3 — Start the database

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 16** on `localhost:5432` (container: `fmd-postgres`)
- **pgAdmin 4** on `localhost:5050` (container: `fmd-pgadmin`)

### Step 4 — Verify both containers are healthy

```bash
docker ps
```

You should see both containers with status `healthy` or `Up`.

### Step 5 — Access pgAdmin

Open **http://localhost:5050** in your browser.

- **Email:** value of `PGADMIN_DEFAULT_EMAIL` in your `.env`
- **Password:** value of `PGADMIN_DEFAULT_PASSWORD` in your `.env`

To add the database server in pgAdmin:
1. Right-click **Servers** → **Register → Server**
2. **Name:** `FMD Local`
3. **Connection tab:** Host = `fmd-postgres`, Port = `5432`, Username/DB from your `.env`

### Step 6 — Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### Step 7 — Start the backend (coming soon)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# → http://localhost:8000
```

> See [docs/database.md](docs/database.md) for full database management guide.

---

## Roadmap

| Phase | Description                                        | Status        |
|-------|----------------------------------------------------|---------------|
| 1     | Project setup, structure, CI/CD scaffolding        | ✅ Complete   |
| 2     | Next.js frontend (Landing, Auth, Dashboard)        | 🔄 In Progress |
| 3     | FastAPI backend + ML pipeline (8 modules)          | Planned       |
| 4     | Database integration, history, analytics           | Planned       |
| 5     | Docker, deployment (Vercel + Render), Sentry       | Planned       |

---

## Contributing

This is a final-year academic project. Branching strategy follows [Gitflow](https://nvie.com/posts/a-successful-git-branching-model/). All commits follow [Conventional Commits](https://www.conventionalcommits.org/).

---

## License

[MIT](LICENSE) © 2025 rifaz shaikh razak
