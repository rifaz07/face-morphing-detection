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

> Full setup guide coming after scaffolding is complete.

### Prerequisites

- Node.js 20+
- Python 3.11+
- Docker + Docker Compose
- PostgreSQL (or Supabase account)
- Cloudinary account
- Clerk account

### Quick Start (Local Dev)

```bash
# 1. Clone the repo
git clone https://github.com/rifaz07/face-morphing-detection.git
cd face-morphing-detection

# 2. Copy env template
cp .env.example .env.local

# 3. Start all services with Docker
docker-compose up --build

# OR run individually:

# Frontend
cd frontend && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
```

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
