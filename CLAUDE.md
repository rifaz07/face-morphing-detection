# CLAUDE.md — Face Morphing Detection System

This file is the canonical project context for all Claude Code sessions. Read it fully before taking any action.

---

## PROJECT OVERVIEW

**Name:** Clustering Based Face Morphing Detection System
**Goal:** Production-grade web application that detects whether an uploaded face image is REAL or MORPHED using a clustering-based ML pipeline (LBP + DCT features + K-Means clustering).
**Type:** Final-year project built to SaaS production standards — not a college demo.

---

## TECH STACK (STRICTLY FOLLOW — DO NOT SUBSTITUTE)

### Frontend
- Next.js 15 (App Router, **JavaScript — NOT TypeScript**)
- Tailwind CSS v4
- shadcn/ui components
- React Dropzone (image uploads)
- Recharts (metrics visualization)
- Axios (API calls)
- React Hook Form + Zod (form validation)
- Framer Motion (animations)
- next-themes (dark/light mode)
- Sonner (toast notifications)
- Lucide React (icons)
- @vercel/analytics

### Backend (ML API)
- FastAPI (Python 3.11+)
- OpenCV, NumPy, scikit-learn, scikit-image
- Pillow, Joblib
- Pydantic v2
- Uvicorn server

### Database
- **Local dev:** PostgreSQL 16 running in Docker via `docker-compose` (container: `fmd-postgres`)
- **Production:** Supabase (same PostgreSQL 16 engine — identical SQL, zero migration surprises)
- Prisma ORM (Next.js side)
- SQLAlchemy (FastAPI side)

### Storage
- Cloudinary (image uploads + CDN delivery)

### Authentication
- Clerk (Google + email login)

### Deployment
- Frontend → Vercel
- Backend → Render (Dockerized)
- Database → Supabase

### DevOps
- Docker + docker-compose
- GitHub Actions (CI/CD)
- Sentry (error tracking)
- ESLint + Prettier + Husky (pre-commit hooks)
- Conventional Commits

---

## PROJECT STRUCTURE

```
face-morphing-detection/
├── frontend/              → Next.js 15 app
├── backend/               → FastAPI ML API
│   ├── app/
│   │   ├── main.py        → FastAPI entry point, CORS, lifespan
│   │   ├── core/
│   │   │   ├── config.py  → Pydantic Settings (env vars)
│   │   │   ├── database.py→ SQLAlchemy engine + get_db dependency
│   │   │   └── logging.py → Loguru setup
│   │   ├── api/
│   │   │   ├── deps.py    → Shared FastAPI dependencies
│   │   │   └── v1/
│   │   │       ├── router.py         → Aggregated v1 router
│   │   │       └── endpoints/
│   │   │           └── health.py     → GET /api/v1/health
│   │   ├── models/base.py → SQLAlchemy DeclarativeBase
│   │   ├── schemas/       → Pydantic v2 response schemas
│   │   ├── services/      → Business logic layer
│   │   └── ml/            → ML pipeline modules (added per task)
│   ├── tests/             → pytest test suite
│   ├── alembic/           → DB migrations
│   ├── Dockerfile         → Multi-stage production image
│   ├── requirements.txt
│   └── requirements-dev.txt
├── ml/                    → Training scripts, notebooks, model artifacts
├── .github/workflows/     → CI/CD pipelines
├── docker-compose.yml     → postgres + pgadmin + backend services
├── CLAUDE.md              → Project context for Claude
├── README.md
└── .gitignore
```

### API URL Conventions
- **Local (host machine):** http://localhost:8000
- **Inside Docker network:** http://fmd-backend:8000
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health endpoint:** http://localhost:8000/api/v1/health
- **DATABASE_URL inside Docker:** `postgresql://fmd_user:<pw>@fmd-postgres:5432/face_morphing_db`
  - Host = container name `fmd-postgres`, port = `5432` (NOT the host-mapped `5435`)

---

## GIT BRANCHING STRATEGY (MANDATORY — ENFORCE ON EVERY TASK)

### Main Branches
| Branch    | Purpose                                  |
|-----------|------------------------------------------|
| `main`    | Production-ready code (auto-deploys)     |
| `develop` | Integration branch                        |

### Supporting Branch Prefixes
| Prefix         | Use case                              |
|----------------|---------------------------------------|
| `feature/<name>` | New features                        |
| `fix/<name>`     | Bug fixes                           |
| `hotfix/<name>`  | Urgent production patches           |
| `chore/<name>`   | Configs, deps, refactors            |
| `docs/<name>`    | Documentation only                  |

### Rules (Follow Strictly for Every Task)
1. For every new feature/module requested, create a **new branch from `develop`**.
2. Branch naming: `feature/<short-kebab-name>` (e.g., `feature/landing-page`)
3. After completing the feature:
   - Stage and commit with a Conventional Commit message
   - Push the branch to remote
   - **DO NOT merge to develop or main automatically**
   - Report the branch name and ask for approval before merging
4. Use `fix/<name>` for bug fixes, `chore/<name>` for configs, `docs/<name>` for docs.
5. Always run `git status` before committing to confirm changes.
6. **NEVER force push. NEVER commit directly to main.**
7. Only the FIRST setup task may commit directly to develop. After that, ALL work goes through feature branches.

### Commit Message Format (Conventional Commits)
```
feat: add LBP feature extraction module
fix: resolve CORS error in upload endpoint
chore: configure husky pre-commit hooks
docs: update API documentation
refactor: simplify K-means clustering logic
style: format code with prettier
test: add unit tests for face detection
```

---

## ML PIPELINE (8 MODULES — BUILD ONE AT A TIME)

| # | Module                        | Description                                              |
|---|-------------------------------|----------------------------------------------------------|
| 1 | Image Input & Validation ✅   | Accept image, validate format/size/dimensions            |
| 2 | Face Detection ✅             | OpenCV Haar Cascade to detect and crop face region       |
| 3 | Preprocessing ✅              | Resize to 128×128, grayscale, histogram eq, normalize    |
| 4 | LBP Feature Extraction ✅     | Local Binary Pattern texture features (59-element vector)|
| 5 | DCT Feature Extraction ✅     | DCT frequency features, 32×32 block, log-compressed (1024)|
| 6 | Feature Fusion ✅             | Concatenate LBP(59) + DCT_norm(1024) = 1083-dim vector   |
| 7 | K-Means Clustering ✅         | K-Means (k=2) on 1083-dim vectors → REAL / MORPHED label|
| 8 | Classification & Evaluation ✅| Accuracy, FAR, FRR, F1, confusion matrix + viva report   |

**Synthetic-data limitation — partially resolved:** K-Means (Module 7) was
originally bootstrapped on synthetic feature vectors only. As of the
`feature/train-real-dataset` branch, `backend/scripts/train_on_real_dataset.py`
trains it on a 4,000-image sample (2,000 REAL + 2,000 MORPHED, random_state=42)
drawn from the real Kaggle SSMD dataset (`ml/dataset/real/` — 25,000 images,
`ml/dataset/morphed/` — 15,000 images). Real-data evaluation on this sample:
71.18% accuracy, FAR 50.25%, FRR 7.29%, F1 0.7624 — see
`ml/checkpoints/real_evaluation_report.json` and the backend README's
"Training on Real Data" section. Full 30,000+ image training (`--full` flag)
is pending as a follow-up run.

---

## WEBSITE PAGES

### Public Pages
| Route         | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `/`           | Landing — Hero, How It Works, Features, Stats, FAQ, Footer         |
| `/about`      | Project background, methodology, team                              |
| `/how-it-works` | Visual pipeline of 8 ML modules with animations                  |
| `/pricing`    | Tiered plans (Free / Pro / Enterprise — SaaS look)                 |
| `/contact`    | Contact form                                                       |

### Authenticated Pages
| Route         | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `/dashboard`  | User overview, quick stats, recent uploads                         |
| `/detect`     | Drag & drop upload + result display + confidence score             |
| `/history`    | Past uploads with filters (date, result type)                      |
| `/analytics`  | Charts (Accuracy, FAR, FRR) — Admin only                          |
| `/settings`   | Profile management                                                  |

### System Pages
- Custom 404 and 500 error pages
- Loading skeletons everywhere
- Maintenance mode page

---

## DESIGN REQUIREMENTS

- Modern SaaS aesthetic (Vercel / Linear / Stripe inspired)
- Hero section with animated gradient + bold typography
- Glassmorphism touches on cards
- Smooth scroll animations (Framer Motion)
- Dark / Light mode toggle (next-themes)
- Geist or Inter font
- Gradient accents (purple/blue palette)
- Fully responsive (mobile-first)
- Accessibility compliant (ARIA labels, keyboard navigation)
- SEO optimized (metadata, OG images, sitemap.xml, robots.txt)
- Image optimization via next/image
- Vercel Analytics integrated
- Toast notifications via Sonner
- Lucide React for all icons
- Loading skeletons (shadcn Skeleton component)
- Empty states with illustrations

---

## WEB APP FEATURES

- Clerk auth (Google + email)
- Image upload via drag & drop (React Dropzone)
- Cloudinary integration for image storage + CDN
- Real-time prediction result display with confidence score
- Animated result card (Real ✅ / Morphed ❌)
- User dashboard with upload history
- Admin dashboard with model metrics (Recharts)
- Download detection report (PDF) — future feature
- Responsive design (mobile + desktop)
- Loading states, error handling, toast notifications

---

## CODE QUALITY RULES

- Write clean, modular, production-grade code
- Add JSDoc / docstrings to every function
- Handle errors gracefully (try/catch, proper HTTP status codes)
- Use environment variables for all secrets (.env.local, .env)
- **Never hardcode API keys or secrets**
- Add input validation everywhere (Zod on frontend, Pydantic on backend)
- Write viva-ready code — a student should be able to explain every line
- Add comments only where logic is non-obvious
- **Component names:** PascalCase
- **File names:** kebab-case for utilities, PascalCase for components
- Follow ESLint + Prettier rules

---

## WORKFLOW FOR EVERY TASK (MANDATORY)

1. Confirm understanding of the task in 1–2 lines
2. Create a new git branch from `develop` (feature/fix/chore as appropriate)
3. List the files to be created or modified **before** writing them
4. Implement the changes
5. Run any relevant checks (lint, type-check, test)
6. Stage and commit with a Conventional Commit message
7. Push the branch to remote
8. Report: branch name, files changed, commit hash, what was built
9. Ask: **"Ready to merge into develop?"** — wait for approval before merging

---

## ENVIRONMENT VARIABLES

All secrets live in environment files. Never commit real values.

| Variable                      | Used By       | Purpose                            |
|-------------------------------|---------------|------------------------------------|
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Frontend | Clerk auth public key             |
| `CLERK_SECRET_KEY`            | Frontend      | Clerk auth secret key              |
| `NEXT_PUBLIC_CLERK_SIGN_IN_URL` | Frontend    | Clerk sign-in redirect URL         |
| `NEXT_PUBLIC_CLERK_SIGN_UP_URL` | Frontend    | Clerk sign-up redirect URL         |
| `DATABASE_URL`                | Frontend      | Prisma → Supabase connection string|
| `NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME` | Frontend | Cloudinary cloud name            |
| `CLOUDINARY_API_KEY`          | Frontend      | Cloudinary API key                 |
| `CLOUDINARY_API_SECRET`       | Frontend      | Cloudinary API secret              |
| `NEXT_PUBLIC_API_URL`         | Frontend      | FastAPI backend URL                |
| `NEXT_PUBLIC_SENTRY_DSN`      | Frontend      | Sentry error tracking DSN          |
| `DATABASE_URL`                | Backend       | SQLAlchemy → Supabase connection   |
| `CLOUDINARY_CLOUD_NAME`       | Backend       | Cloudinary for backend uploads     |
| `CLOUDINARY_API_KEY`          | Backend       | Cloudinary API key (backend)       |
| `CLOUDINARY_API_SECRET`       | Backend       | Cloudinary API secret (backend)    |
| `SENTRY_DSN`                  | Backend       | Sentry DSN for FastAPI             |
| `ALLOWED_ORIGINS`             | Backend       | CORS allowed origins               |

---

## LOCAL DEVELOPMENT SETUP

### Prerequisites
- Docker Desktop 4.x+ (running)
- Node.js 20+
- Python 3.11+
- Git

### Database (Docker)

```bash
# Copy env template and fill in values
cp .env.example .env

# Start PostgreSQL + pgAdmin in background
docker compose up -d

# Verify both containers are running and postgres is healthy
docker ps

# View postgres logs (follow)
docker compose logs -f postgres

# Stop containers (data is preserved in named volumes)
docker compose down

# Stop AND delete all data (full reset)
docker compose down -v
```

### Access pgAdmin GUI
- URL: http://localhost:5050
- Login: `PGADMIN_DEFAULT_EMAIL` / `PGADMIN_DEFAULT_PASSWORD` from `.env`
- Add server: host = `fmd-postgres`, port = `5432`, user/db from `.env`

### Direct CLI access
```bash
docker exec -it fmd-postgres psql -U fmd_user -d face_morphing_db
```

---

## DEPLOYMENT NOTES

- **Frontend:** Deploy to Vercel. Connect GitHub repo, set env vars in Vercel dashboard.
- **Backend:** Dockerized FastAPI deployed to Render. Uses `Dockerfile` in `backend/`.
- **Database:** Supabase PostgreSQL. Connection string goes into `DATABASE_URL`.
- **CI/CD:** GitHub Actions workflows in `.github/workflows/`. Run lint + tests on PR.

---

## REPOSITORY INFO

- **Remote:** https://github.com/rifaz07/face-morphing-detection.git
- **Default branch:** main
- **Integration branch:** develop
- **Git user:** rifaz shaikh razak
