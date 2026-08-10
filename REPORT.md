# Clustering Based Face Morphing Detection System
## Technical Project Report

**Author:** Rifaz Shaikh Razak  
**Date:** August 2026  
**Repository:** https://github.com/rifaz07/face-morphing-detection  
**Branch:** develop

---

## 1. Abstract

This report documents the design and implementation of a production-grade, web-based face morphing detection system. The system classifies uploaded face images as REAL or MORPHED using a three-stage feature engineering pipeline — Local Binary Pattern (LBP) texture extraction, Discrete Cosine Transform (DCT) frequency extraction, and feature fusion — followed by unsupervised K-Means clustering. The complete eight-module ML pipeline is served through a FastAPI backend (12 detection endpoints) and integrated with a Next.js 15 frontend supporting Clerk authentication and Cloudinary image storage. A test suite of 179 test functions covers all ML modules and API endpoints. The current classifier is trained exclusively on synthetic data; metrics reported in this document carry that caveat explicitly.

---

## 2. Introduction

### 2.1 Problem Statement

Face morphing is a biometric attack in which two or more face images are blended to produce a composite that can be used to spoof identity verification systems. A morphed passport photograph, for instance, may be accepted by automated face-recognition gates for two different individuals. Detection of such attacks is therefore a prerequisite for trustworthy biometric security.

### 2.2 Motivation

Existing commercial biometric systems often lack a standalone morphing-detection stage. Open-source approaches vary widely in reproducibility. This project addresses the gap by building a complete, end-to-end, deployable detection pipeline — from raw image upload to a confidence-scored REAL / MORPHED verdict — with full API documentation and a web interface accessible to non-technical operators.

### 2.3 Objectives

1. Implement an eight-module ML pipeline that processes a raw face photograph into a binary classification.
2. Expose the pipeline through a documented REST API with Swagger UI.
3. Provide a multi-page web frontend with authentication, upload, and result display.
4. Define standard biometric evaluation metrics (Accuracy, FAR, FRR, F1) and compute them in an automated evaluation report.
5. Build to production engineering standards: typed schemas, structured logging, Docker containerisation, database persistence.

---

## 3. Background

### 3.1 Face Morphing Attacks

A morphed face image is created by combining pixel information from two source faces using one of several techniques:

- **Alpha blending:** weighted pixel-level averaging of two face images.
- **Geometric warping:** landmark-based alignment followed by blending to reduce structural misalignment.
- **GAN synthesis:** a generative adversarial network trained to produce composite faces.

All three techniques alter the microstructure of the resulting image in ways that are largely invisible to human observers but measurable through signal processing.

### 3.2 Why LBP?

Local Binary Pattern (LBP), introduced by Ojala et al. (1996, extended 2002), is a texture descriptor that encodes the relationship between each pixel and its circular neighbourhood. Face morphing operations — particularly blending and resampling — disrupt the natural skin texture microstructure (pores, fine lines, hair follicles). LBP captures these disruptions as changes in the histogram of texture codes. Real face images cluster tightly in LBP feature space; morphed images occupy different or mixed regions, providing a discriminative signal for clustering.

### 3.3 Why DCT?

The 2D Discrete Cosine Transform decomposes an image into spatial frequency components. Morphing artefacts appear in the mid-frequency range of the DCT spectrum:

- Alpha blending introduces interpolated values that add energy in mid-frequency bands absent in natural faces.
- Geometric resampling introduces ringing artefacts visible as anomalous coefficients.
- GAN synthesis produces textures that look natural spatially but differ from real photography in frequency-domain statistics.

By restricting attention to the top-left 32×32 block of DCT coefficients (DC + low-to-mid frequencies), the system targets the frequency range most affected by morphing while discarding high-frequency noise.

### 3.4 Why K-Means?

Labelled face morphing datasets are expensive to collect and annotate. K-Means clustering offers an unsupervised alternative: given that real and morphed faces produce statistically different LBP+DCT signatures, K-Means can discover the natural groupings in feature space without ground-truth labels. The cluster whose centroid lies farther from the "real face" distribution is assigned the MORPHED label by majority vote at training time.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT BROWSER                          │
│  Next.js 15 (App Router, JavaScript)  │  Tailwind v4 + shadcn   │
│  Clerk Auth  │  React Dropzone  │  Framer Motion  │  Recharts    │
└────────────────────────────┬────────────────────────────────────┘
                             │  HTTPS / REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND  (Python 3.11)               │
│  POST /api/v1/detection/classify  ─── Main detection endpoint    │
│  + 11 additional detection endpoints + 1 health endpoint         │
│  Pydantic v2 schemas │ Loguru logging │ CORS middleware           │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    ML PIPELINE (8 Modules)               │    │
│  │  1.Validate → 2.FaceDetect → 3.Preprocess               │    │
│  │  → 4.LBP → 5.DCT → 6.Fuse → 7.K-Means → 8.Evaluate    │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────┬─────────────────────────────────┬───────────────────┘
            │ SQLAlchemy / Prisma              │ cloudinary SDK
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────────┐
│  PostgreSQL 16         │         │  Cloudinary CDN            │
│  Docker (local): 5432  │         │  Image storage + delivery  │
│  host-mapped port 5435 │         └───────────────────────────┘
│  Supabase (production) │
└───────────────────────┘
```

**Deployment targets (planned):**

| Component  | Platform   | Notes                                  |
|------------|------------|----------------------------------------|
| Frontend   | Vercel     | Auto-deploys from main branch          |
| Backend    | Render     | Dockerised FastAPI via `Dockerfile`    |
| Database   | Supabase   | PostgreSQL 16, same engine as Docker   |

**CI/CD status:** No GitHub Actions workflows have been committed to the repository at the time of writing. The `.github/workflows/` directory does not exist.

---

## 5. Methodology — The Eight ML Modules

### Module 1: Image Input and Validation (`image_validator.py`)

Every image passes six sequential checks before any ML processing occurs:

1. **File size** — must not exceed 10 MB (`MAX_IMAGE_SIZE_MB`).
2. **Extension check** — must be `.jpg`, `.jpeg`, `.png`, or `.webp`.
3. **Magic-byte check** — the first raw bytes of the file must match the declared format's canonical signature (JPEG: `\xff\xd8\xff`; PNG: `\x89PNG\r\n\x1a\n`; WebP: `RIFF...WEBP`). This prevents disguised-file uploads where, e.g., an executable is renamed to `.jpg`.
4. **Dimension check** — width and height must be in [100 px, 4096 px] (`MIN_IMAGE_DIMENSION`, `MAX_IMAGE_DIMENSION`).
5. **Pillow integrity check** — Pillow's `verify()` detects truncated or structurally corrupt files. The file is opened twice because `verify()` exhausts the stream.
6. **OpenCV decodability check** — `cv2.imdecode` confirms the file can be decoded by the downstream ML modules, which use OpenCV exclusively.

Grayscale (mode=L) and RGBA images produce warnings rather than errors; they are convertible to RGB.

### Module 2: Face Detection (`face_detector.py`)

Face detection uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) with the Viola-Jones framework. The configured parameters are:

| Parameter       | Value | Effect                                                    |
|-----------------|-------|-----------------------------------------------------------|
| `scaleFactor`   | 1.1   | Image pyramid step size — standard default                |
| `minNeighbors`  | 5     | Minimum overlapping detections required to accept a face  |
| `minSize`       | 30 px | Minimum bounding box dimension                            |

The detector returns all detected face bounding boxes. The largest face by pixel area is forwarded to subsequent modules. A face count of zero is a valid result (HTTP 200), not an error. Face crops are JPEG-encoded and base64-transmitted to downstream modules.

### Module 3: Preprocessing (`image_preprocessor.py`)

A deterministic four-step pipeline transforms each face crop:

| Step | Operation                   | Implementation                             | Rationale                                                      |
|------|-----------------------------|--------------------------------------------|----------------------------------------------------------------|
| 1    | Resize to 128×128           | `cv2.resize(..., cv2.INTER_AREA)`          | Fixed size required for identical feature vector dimensions    |
| 2    | Convert to grayscale        | `cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)`    | LBP and DCT operate on intensity; drops chroma channels        |
| 3    | Histogram equalisation      | `cv2.equalizeHist(gray)`                   | Normalises contrast/lighting across varied photographic conditions |
| 4    | Normalise to [0.0, 1.0]     | `equalized.astype(np.float32) / 255.0`     | Prevents large pixel values dominating Euclidean distance      |

> **Note on equalisation:** The implementation uses `cv2.equalizeHist` (global histogram equalisation), not CLAHE.  
> **Note on interpolation:** The resize step uses `cv2.INTER_AREA`, not bilinear interpolation.

### Module 4: LBP Feature Extraction (`lbp_extractor.py`)

**Algorithm:** `skimage.feature.local_binary_pattern` with parameters `P=8, R=1, method="uniform"`.

**Computation:**
For each pixel at position (x, y), the algorithm compares the pixel's intensity with 8 evenly spaced neighbours on a circle of radius 1. Each comparison produces one bit (neighbour ≥ centre → 1, else 0). The 8-bit code is the LBP value for that pixel.

**Uniform patterns:** A pattern is "uniform" if it has at most two 0→1 or 1→0 transitions in its bit string. With N=8 neighbours, there are 58 uniform patterns plus 1 non-uniform catch-all bin = **59 bins total**.

**Feature vector construction:**
```
hist[bin] = count of pixels whose LBP code falls in that bin
hist /= hist.sum()          # sum-normalisation → values in [0.0, 1.0]
```
The output is a 59-element float vector summing to 1.0.

> **Note on normalisation:** The LBP histogram uses sum-normalisation (divide by total pixel count), not L2-normalisation.

### Module 5: DCT Feature Extraction (`dct_extractor.py`)

**Algorithm:** `scipy.fft.dctn` (2D DCT Type II, `norm="ortho"`).

**Computation pipeline:**

1. Scale the float32 [0,1] array to float64 [0,255] for numerical conditioning.
2. Apply 2D orthonormal DCT: `dctn(scaled, type=2, norm="ortho")`.
3. Extract the top-left 32×32 block of the 128×128 DCT coefficient matrix.
4. Apply log compression to stabilise the large dynamic range:

   ```
   compressed(x) = sign(x) × log(1 + |x|)
   ```

5. Flatten the 32×32 block to a 1024-element vector.

The 32×32 block captures DC, low-frequency, and mid-frequency content — the range where morphing artefacts are most detectable — while discarding the high-frequency noise that varies with compression and sensor.

### Module 6: Feature Fusion (`feature_fusion.py`)

The 59-element LBP vector and the 1024-element DCT vector are combined into a single 1083-dimensional descriptor.

**Normalisation before fusion:**  
The LBP histogram is already in [0.0, 1.0]. Raw log-compressed DCT values span approximately −20 to +20, which would cause the DCT dimensions to numerically dominate Euclidean distance calculations in K-Means. Before concatenation, the DCT vector is MinMax-normalised per vector:

```
dct_normalised = (dct - min(dct)) / (max(dct) - min(dct))
```

When all DCT values are identical (degenerate case), the output is an all-zeros vector.

**Concatenation:**
```
fused_vector = [LBP(59) | DCT_normalised(1024)]   → 1083 dimensions
```

Dimension breakdown:
- Dims 0–58 (5.45%): LBP uniform histogram bins — texture distribution
- Dims 59–1082 (94.55%): DCT top-left 32×32 block coefficients — frequency content

> **Note on DCT normalisation scope:** MinMax is applied per individual inference vector, not fitted on a training corpus, because the goal is to scale each image's own DCT range uniformly.

### Module 7: K-Means Clustering (`kmeans_classifier.py`)

**Model:** `sklearn.cluster.KMeans` with parameters:

| Parameter      | Value | Note                                 |
|----------------|-------|--------------------------------------|
| `n_clusters`   | 2     | REAL and MORPHED                     |
| `random_state` | 42    | Reproducibility                      |
| `n_init`       | 10    | Number of independent restarts       |
| `max_iter`     | 300   | Maximum EM iterations per restart    |

**Label assignment:** After fitting, cluster labels are assigned by majority vote using the ground-truth labels supplied at training time. For synthetic training, the REAL and MORPHED labels are known.

**Confidence score:** Distance to the assigned centroid is converted to a [0, 1] confidence score:

```
confidence = 1 / (1 + d / μ_cluster)
```

where `d` is the Euclidean distance to the assigned centroid and `μ_cluster` is the mean distance of all training samples in that cluster. A sample at the centroid scores 1.0; a sample at the average cluster boundary scores approximately 0.5.

**Persistence:** The fitted model is saved to disk via `joblib`. On startup, the application attempts to load the saved model; if none is found, it trains on synthetic data automatically.

**Synthetic training data (current implementation):**  
In the absence of a labelled real-world dataset, the classifier is initialised on programmatically generated feature vectors:

- 500 REAL samples: drawn from a Gaussian with std=0.05 around a centroid in [0.35, 0.65] of feature space.
- 500 MORPHED samples: drawn from a Gaussian with std=0.15 around a mirrored centroid.

**This is a known limitation.** See Section 9.

### Module 8: Classification and Evaluation (`evaluator.py`)

**Metric definitions** (label convention: 0=REAL, 1=MORPHED):

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
FAR       = FP / (FP + TN)     [False Acceptance Rate — morphed accepted as REAL]
FRR       = FN / (FN + TP)     [False Rejection Rate  — real rejected as MORPHED]
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1 Score  = 2 × Precision × Recall / (Precision + Recall)
```

Confusion matrix layout:

```
                   Predicted REAL   Predicted MORPHED
  Actual MORPHED       FP               TN
  Actual REAL          TP               FN

  → [[TN, FP],
     [FN, TP]]
```

A FP (false positive in this layout) represents a MORPHED face accepted as REAL — the primary security failure mode. FP drives FAR; FN drives FRR.

**Evaluation dataset:** 200 synthetic test vectors (100 REAL + 100 MORPHED) generated using random seed 99 (training used seed 42, ensuring no data leakage from the same Gaussian distributions).

---

## 6. Implementation

### 6.1 Backend — FastAPI

**Language:** Python 3.11+

**Key dependencies** (from `requirements.txt`):

| Package                    | Version   | Purpose                              |
|----------------------------|-----------|--------------------------------------|
| fastapi                    | ≥0.115.0  | REST API framework                   |
| uvicorn[standard]          | ≥0.32.0   | ASGI server                          |
| pydantic                   | ≥2.9.0    | Data validation and serialisation    |
| pydantic-settings          | ≥2.6.0    | Environment-variable config          |
| sqlalchemy                 | ≥2.0.36   | ORM for PostgreSQL                   |
| psycopg2-binary            | ≥2.9.10   | PostgreSQL driver                    |
| alembic                    | ≥1.14.0   | Database migration tool (configured) |
| loguru                     | ≥0.7.2    | Structured logging                   |
| opencv-python-headless     | ≥4.10.0   | Image decoding, face detection       |
| numpy                      | ≥2.1.0    | Numerical arrays                     |
| scikit-learn               | ≥1.5.0    | K-Means clustering, MinMaxScaler     |
| scikit-image               | ≥0.24.0   | LBP feature extraction               |
| scipy                      | ≥1.14.0   | 2D DCT (dctn)                        |
| pillow                     | ≥11.0.0   | Image integrity verification         |
| joblib                     | ≥1.4.2    | Model persistence                    |

**Application structure:**

```
backend/app/
├── main.py                         → FastAPI app, CORS, lifespan hooks
├── core/
│   ├── config.py                   → Pydantic Settings (all env vars)
│   ├── database.py                 → SQLAlchemy engine, get_db dependency
│   └── logging.py                  → Loguru setup
├── api/v1/
│   ├── router.py                   → Aggregates health and detection routers
│   └── endpoints/
│       ├── health.py               → GET /api/v1/health
│       └── detection.py            → 12 detection endpoints
├── ml/
│   ├── validators/image_validator.py
│   ├── detectors/face_detector.py
│   ├── preprocessors/image_preprocessor.py
│   ├── feature_extractors/
│   │   ├── lbp_extractor.py
│   │   ├── dct_extractor.py
│   │   └── feature_fusion.py
│   ├── clustering/kmeans_classifier.py
│   └── evaluation/evaluator.py
├── models/base.py                  → SQLAlchemy DeclarativeBase
└── schemas/detection.py            → Pydantic v2 API response schemas
```

### 6.2 API Endpoints

**Base URL:** `http://localhost:8000/api/v1`

**Health endpoint (1):**

| Method | Path            | Description                         |
|--------|-----------------|-------------------------------------|
| GET    | /health         | Service liveness and readiness check |

**Detection endpoints (12):**

| Method | Path                        | ML Modules | Description                                         |
|--------|-----------------------------|------------|-----------------------------------------------------|
| POST   | /detection/validate         | 1          | Image pre-flight validation only                    |
| POST   | /detection/detect-face      | 1+2+3      | Validate → detect → preprocess all faces            |
| POST   | /detection/preprocess       | 1+2+3      | Validate → detect → preprocess largest face (debug) |
| POST   | /detection/extract-lbp      | 1+2+3+4    | Pipeline through LBP extraction                     |
| POST   | /detection/extract-dct      | 1+2+3+5    | Pipeline through DCT extraction                     |
| POST   | /detection/extract-features | 1–6        | Full feature extraction, fused vector               |
| POST   | /detection/classify         | 1–7        | **Main endpoint** — REAL/MORPHED verdict            |
| GET    | /detection/model-info       | —          | K-Means model metadata                              |
| POST   | /detection/retrain          | —          | Retrain on synthetic data                           |
| GET    | /detection/evaluation       | 8          | Cached evaluation report (startup)                  |
| GET    | /detection/evaluation/live  | —          | Live session prediction statistics                  |
| POST   | /detection/evaluation/run   | 8          | Trigger a fresh evaluation run                      |

Interactive documentation is served at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

### 6.3 Frontend — Next.js

**Framework:** Next.js 15.5.15 (App Router, JavaScript — not TypeScript)  
**React:** 19.1.0

**Key dependencies** (from `package.json`):

| Package              | Version   | Purpose                          |
|----------------------|-----------|----------------------------------|
| @clerk/nextjs        | ^7.3.3    | Authentication (Google + email)  |
| @vercel/analytics    | ^2.0.1    | Usage analytics                  |
| axios                | ^1.16.0   | HTTP client for API calls        |
| framer-motion        | ^12.38.0  | Page and component animations    |
| lucide-react         | ^1.14.0   | Icon library                     |
| next-themes          | ^0.4.6    | Dark/light mode toggle           |
| react-dropzone       | ^15.0.0   | Drag-and-drop image upload       |
| react-hook-form      | ^7.75.0   | Form state management            |
| recharts             | ^3.8.1    | Metrics visualisation charts     |
| sonner               | ^2.0.7    | Toast notifications              |
| zod                  | ^4.4.3    | Form schema validation           |
| cloudinary           | ^2.10.0   | Image upload SDK                 |
| tailwindcss          | ^4        | Utility-first CSS                |

**shadcn/ui components installed (17):**

accordion, avatar, badge, button, card, dialog, dropdown-menu, form, input, label, select, separator, sheet, skeleton, sonner, tabs, textarea

**shadcn configuration:** style=`new-york`, baseColor=`slate`, RSC=true, TSX=false (JavaScript), iconLibrary=`lucide`.

**Frontend directory structure (`src/`):**

```
src/
├── app/        → Next.js App Router pages and layouts
├── components/ → Shared React components (including shadcn/ui)
├── constants/  → Shared constants
├── hooks/      → Custom React hooks
├── lib/        → Utility functions
└── middleware.js → Clerk authentication middleware
```

### 6.4 Docker and Local Development

The `docker-compose.yml` defines three services on an isolated bridge network (`fmd-network`):

| Service     | Container name | Image                  | Internal port | Host-mapped port         |
|-------------|----------------|------------------------|---------------|--------------------------|
| postgres    | fmd-postgres   | postgres:16-alpine     | 5432          | `${POSTGRES_PORT:-5432}` (documented as 5435 in CLAUDE.md) |
| backend     | fmd-backend    | Built from `./backend` | 8000          | 8000                     |
| pgadmin     | fmd-pgadmin    | dpage/pgadmin4:latest  | 80            | `${PGADMIN_PORT:-5050}`  |

The PostgreSQL container uses a named volume (`fmd-postgres-data`) so data persists across `docker-compose down` restarts. A `pg_isready` health check gates the backend and pgAdmin startup.

The `DATABASE_URL` inside the Docker network uses the container name as host:  
`postgresql://fmd_user:<pw>@fmd-postgres:5432/face_morphing_db`

---

### 6.5 Frontend Pages

All pages use the Next.js App Router. Protected pages are wrapped by the `(dashboard)` route group which applies a shared sidebar layout.

**Public Pages**

| Route          | File                                     | What It Displays                                                                                      | Key Components / Hooks                         |
|----------------|------------------------------------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------|
| `/`            | `app/page.js`                            | Landing page: hero, how-it-works overview, features, stats, FAQ, footer. Title: "Face Morphing Detection — Detect Morphed Faces with AI" | `<LandingPage>` from `components/pages/landing-page` |
| `/about`       | `app/about/page.js`                      | Project background, methodology, team. Description references the "clustering-based methodology"      | `<AboutPage>` from `components/pages/about-page` |
| `/how-it-works`| `app/how-it-works/page.js`               | Visual deep-dive into all 8 ML modules                                                                | `<HowItWorksPage>` from `components/pages/how-it-works-page` |
| `/pricing`     | `app/pricing/page.js`                    | Tiered plans: Free / Pro / Enterprise (SaaS-style). Brand name used: "FaceGuard"                     | `<PricingPage>` from `components/pages/pricing-page` |
| `/contact`     | `app/contact/page.js`                    | Contact form for bug reports, features, partnerships                                                  | `<ContactPage>` from `components/pages/contact-page` |
| `/sign-in`     | `app/sign-in/[[...sign-in]]/page.js`     | Clerk-rendered sign-in UI. Heading: "Welcome back to FaceGuard". Custom purple/blue gradient theme    | `<SignIn>` (Clerk), `ScanFace` icon (Lucide)   |
| `/sign-up`     | `app/sign-up/[[...sign-up]]/page.js`     | Clerk-rendered sign-up UI. Heading: "Create your FaceGuard account"                                  | `<SignUp>` (Clerk), `ScanFace` icon (Lucide)   |

Both auth pages apply a custom `appearance` object setting `colorPrimary: '#7c3aed'` and `borderRadius: '0.75rem'` to match the brand palette.

**Protected Pages** (all under the `(dashboard)` route group)

The dashboard layout (`app/(dashboard)/layout.js`) calls the `useSyncUser()` hook on every load, which triggers `POST /api/auth/sync-user` to upsert the authenticated Clerk user into the database. It renders a `<Sidebar>` beside the page content.

| Route          | File                                     | What It Displays                                                                                                                                                                  | Key Components / Hooks                                                               |
|----------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `/dashboard`   | `app/(dashboard)/dashboard/page.js`      | Personalised greeting using `user.firstName`. Four stat cards (Total Detections, Real Faces, Morphed Faces, Avg Confidence). Quick-detect call-to-action card. Table of 5 most recent predictions with thumbnail, filename, date, confidence, and REAL/MORPHED badge. Skeleton loaders while fetching. | `Card`, `Skeleton`, `Badge`, `Button` (shadcn); `Image`, `Link` (Next.js); `useUser` (Clerk); fetches `/api/stats` and `/api/predictions` |
| `/detect`      | `app/(dashboard)/detect/page.js`         | Drag-and-drop upload zone (React Dropzone; accepts JPEG/PNG/WebP, max 10 MB). Image preview after selection. Four animated pipeline progress steps shown during analysis. Animated result card with REAL/MORPHED verdict, animated confidence count-up display, face count, processing time, and image dimensions. Error state with descriptive messages for NO_FACE, INVALID_IMAGE, SERVICE_UNAVAILABLE. | `Card`, `Button` (shadcn); `useDropzone` (react-dropzone); `motion`, `AnimatePresence` (Framer Motion); `toast` (Sonner); calls `POST /api/detect` |
| `/history`     | `app/(dashboard)/history/page.js`        | Paginated list of the user's past predictions. Filter buttons: All / Real / Morphed. Expandable rows showing confidence, face count, processing time, and image thumbnail. Skeleton loaders. Empty state with call-to-action. | `Card`, `Badge`, `Button`, `Skeleton` (shadcn); `Image` (Next.js); fetches `/api/predictions` |
| `/analytics`   | `app/(dashboard)/analytics/page.js`      | Three metric cards: Accuracy, FAR, FRR (with percentage values). Recharts `BarChart` visualising the three metrics side by side using colour coding (purple=Accuracy, red=FAR, orange=FRR). Additional model info section showing feature dimensions and cluster count via icons. Skeleton loaders. | `Card`, `Skeleton` (shadcn); `BarChart`, `Bar`, `XAxis`, `YAxis`, `CartesianGrid`, `Tooltip`, `ResponsiveContainer`, `Cell` (Recharts); `BarChart2`, `Cpu`, `Layers` icons (Lucide); fetches `/api/metrics` |
| `/settings`    | `app/(dashboard)/settings/page.js`       | Embeds the Clerk `<UserProfile>` component with `routing="hash"` to allow profile management, connected accounts, and security settings without a page navigation. Custom purple theme applied via `appearance` prop. | `UserProfile` (Clerk) |

**System Pages**

| File              | Route trigger                        | What It Displays                                                                                                   | Key Components           |
|-------------------|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------|
| `not-found.jsx`   | Any unmatched URL                    | Animated 404 with floating ambient gradient shapes. Large gradient "404" text, "Page not found" heading, humorous tagline. "Back to Home" and "Go Back" buttons. | `Button`, `Link`; `motion` (Framer Motion); `Home`, `ArrowLeft` icons |
| `error.jsx`       | Unhandled React rendering error      | "Something went wrong" screen with error digest code. "Try Again" (calls `reset()`) and "Back to Home" buttons. Logs the error to `console.error`. | `Button`, `Link`; `motion` (Framer Motion); `AlertTriangle`, `RefreshCw`, `Home` icons |

---

### 6.6 Next.js API Routes

All route handlers live under `frontend/src/app/api/`. They are invoked server-side and have access to Prisma and environment secrets.

| Route                        | Methods       | Auth Required | What It Does                                                                                                                          | Response Shape                                                                                     |
|------------------------------|---------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `/api/detect`                | POST          | Yes (Clerk)   | Full detection pipeline: authenticates user → uploads image to Cloudinary → calls FastAPI `/api/v1/detection/classify` → writes Prediction row to DB. Returns the verdict. | `{ prediction, confidence, faceCount, processingTimeMs, imageUrl, imageDimensions, predictionId }` |
| `/api/upload`                | POST          | No            | Standalone Cloudinary upload. Accepts `FormData` with `file` field; uploads to folder `faceguard/detections`. Returns Cloudinary metadata. | `{ url, publicId, width, height, format, bytes }` |
| `/api/predictions`           | GET           | Yes (Clerk)   | Returns up to 50 predictions for the authenticated user, ordered by `createdAt DESC`. Includes nested `user` fields.                  | `{ predictions: [...], count: N }` |
| `/api/predictions`           | POST          | Yes (Clerk)   | Direct prediction create (fallback path; normal flow uses `/api/detect`). Requires `imageUrl`, `imageName`, `imageSize` in request body. | `{ prediction }` with HTTP 201 |
| `/api/predictions/[id]`      | GET           | No            | Fetches a single `Prediction` record by CUID. Returns 404 if not found.                                                               | `{ prediction }` |
| `/api/predictions/[id]`      | DELETE        | No            | Deletes a `Prediction` record by CUID. Returns 404 if not found.                                                                      | `{ message: "Prediction deleted", id }` |
| `/api/stats`                 | GET           | Yes (Clerk)   | Aggregates all predictions for the current user: counts by result, accuracy (real/classified), and average confidence score.          | `{ total, real, morphed, pending, errors, accuracy, avgConfidence }` |
| `/api/metrics`               | GET           | No            | Returns the most recent `ModelMetrics` row from the database, ordered by `recordedAt DESC`. Returns 404 when no seed data exists.     | `{ metrics }` |
| `/api/auth/sync-user`        | POST          | Yes (Clerk)   | Upserts the authenticated Clerk user into the `User` table. Updates email, name, and imageUrl on every call; creates the row if it does not exist. | `{ user }` |

---

### 6.7 Authentication

**Middleware** (`frontend/src/middleware.js`)

Clerk middleware is applied to all routes via the following matcher configuration (quoted exactly from source):

```js
export const config = {
  matcher: [
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    '/(api|trpc)(.*)',
  ],
}
```

The middleware inspects every matched request. It enforces authentication (`auth.protect()`) only on the following five route patterns:

```js
const isProtectedRoute = createRouteMatcher([
  '/dashboard(.*)',
  '/detect(.*)',
  '/history(.*)',
  '/analytics(.*)',
  '/settings(.*)',
])
```

All other routes (public pages, API routes) are not blocked by the middleware itself; individual API routes perform their own `auth()` call where needed.

**JWT Verification in API Routes**

API routes that require authentication call `auth()` from `@clerk/nextjs/server` at the top of the handler:

```js
const { userId } = await auth()
if (!userId) {
  return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
}
```

Clerk validates the session JWT from the request cookie server-side; no manual token parsing is required in application code.

**User Sync Flow**

Clerk manages authentication state (sessions, OAuth providers, email/password). The application maintains its own `User` record in PostgreSQL to store predictions and role information. The sync flow works as follows:

1. User signs in via Clerk (Google or email).
2. On every protected page load, the `(dashboard)` layout calls the `useSyncUser()` hook (`src/hooks/use-sync-user.js`).
3. The hook fires `POST /api/auth/sync-user` whenever `isLoaded && isSignedIn` is true.
4. The route handler calls `currentUser()` (Clerk server SDK) to fetch the full Clerk user object, then executes a Prisma `upsert` on the `User` table keyed by `clerkId`:
   - **On create:** writes `clerkId`, `email`, `name`, `imageUrl`, `role: 'USER'`.
   - **On update:** refreshes `email`, `name`, and `imageUrl` to reflect any Clerk profile changes.
5. The same `getOrCreateDbUser` helper is duplicated inline in `/api/detect`, `/api/predictions`, and `/api/stats` as a safety net: if the sync-user call was missed, those routes create the database record on first use.

---

### 6.8 End-to-End Detection Flow

The main detection flow is orchestrated entirely by `POST /api/detect` (`frontend/src/app/api/detect/route.js`). The exact sequence is:

```
Client (detect page)
  │
  │  POST /api/detect  (FormData: { file })
  ▼
Step 1 — Auth check
  Calls auth() from @clerk/nextjs/server.
  Returns HTTP 401 if no valid session.

Step 2 — Parse file
  Reads file from FormData; returns HTTP 400 (NO_FILE) if absent.

Step 3 — Find or create DB User
  Calls getOrCreateDbUser(userId):
    prisma.user.findUnique({ where: { clerkId } })
    → if not found: currentUser() + prisma.user.upsert(create)

Step 4 — Upload to Cloudinary
  cloudinary.uploader.upload_stream(buffer, {
    folder: 'faceguard/detections',
    resource_type: 'image',
    public_id: `${Date.now()}_${filename_without_extension}`,
  })
  Returns cloudinaryResult.secure_url (CDN URL) and cloudinaryResult.public_id.
  Returns HTTP 500 (UPLOAD_FAILED) on Cloudinary error.

Step 5 — Call FastAPI /classify
  Rebuilds FormData with the original buffer as a Blob.
  fetch(`${NEXT_PUBLIC_API_URL}/api/v1/detection/classify`, { method: 'POST', body: fastapiForm })
  Returns HTTP 503 (SERVICE_UNAVAILABLE) if FastAPI is unreachable.
  Returns HTTP 400 (INVALID_IMAGE) if FastAPI returns 400.

Step 6 — No-face branch
  If mlResult.prediction is null (no face detected):
    prisma.prediction.create({ result: 'ERROR', errorMessage: 'No face detected', faceCount: 0, metadata: {...} })
    Returns HTTP 422 (NO_FACE).

Step 7 — Save Prediction to DB
  Computes total processing time:
    totalMs = detection.processing_time_ms + fusion.processing_time_ms + prediction.processing_time_ms

  prisma.prediction.create({
    userId: dbUser.id,
    imageUrl: cloudinaryResult.secure_url,
    imageName: file.name,
    imageSize: file.size,
    result: mlResult.prediction.prediction,        // "REAL" or "MORPHED"
    confidence: mlResult.prediction.confidence,    // float [0, 1]
    faceCount: mlResult.detection.face_count,
    processingTimeMs: Math.round(totalMs * 10) / 10,
    fusedVector: mlResult.fusion.fused_vector,     // 1083-element array stored as JSON
    metadata: {
      width, height,                               // from detection.image_dimensions
      cloudinaryPublicId: cloudinaryResult.public_id,
      cloudinaryFormat: cloudinaryResult.format,
      clusterId: mlResult.prediction.cluster_id,
      distanceToCentroid: mlResult.prediction.distance_to_centroid,
    },
  })

Step 8 — Return to client
  { prediction, confidence, faceCount, processingTimeMs, imageUrl, imageDimensions, predictionId }
```

**Fields persisted to the database:**

| Field              | Source                                          |
|--------------------|-------------------------------------------------|
| `imageUrl`         | `cloudinaryResult.secure_url`                   |
| `imageName`        | Original `file.name` from FormData              |
| `imageSize`        | `file.size` in bytes                            |
| `result`           | `mlResult.prediction.prediction` ("REAL"/"MORPHED") |
| `confidence`       | `mlResult.prediction.confidence` (float [0,1])  |
| `faceCount`        | `mlResult.detection.face_count`                 |
| `processingTimeMs` | Sum of all three ML stage timings               |
| `fusedVector`      | `mlResult.fusion.fused_vector` (1083 floats, JSON) |
| `metadata.cloudinaryPublicId` | `cloudinaryResult.public_id`       |
| `metadata.clusterId`          | `mlResult.prediction.cluster_id`   |
| `metadata.distanceToCentroid` | `mlResult.prediction.distance_to_centroid` |

Note: `lbpVector` and `dctVector` fields exist in the Prisma schema but are not populated by `/api/detect`; only `fusedVector` is saved. This is intentional — the 1083-element fused vector subsumes both.

**Cloudinary folder structure:** All uploads go to the `faceguard/detections` folder. The `public_id` is `{Unix_timestamp_ms}_{filename_without_extension}`, ensuring no collisions between concurrent uploads.

---

### 6.9 Version Control and Development Workflow

**Repository statistics:**

| Metric               | Value                                              |
|----------------------|----------------------------------------------------|
| Total commits        | 31 (across all branches)                          |
| Current branch       | `develop`                                          |
| Main branch          | `main`                                             |
| Feature branches     | 14 (all pushed to remote)                          |
| Commit format        | Conventional Commits                               |

**Commit message format (Conventional Commits):**

```
feat: description          — new feature
feat(ml): description      — ML-scoped feature
fix(api): description      — scoped bug fix
chore: description         — project setup
```

**Branching strategy:**

All features are developed on `feature/<name>` branches cut from `develop`. After completion, the feature branch is committed and pushed. A merge commit is then made into `develop` (not a rebase or squash). No work is committed directly to `main` beyond the initial project scaffolding.

**Feature branches and what each delivered:**

| Branch                              | Delivered                                                                                      |
|-------------------------------------|------------------------------------------------------------------------------------------------|
| `feature/frontend-scaffold`         | Next.js 15 project with Tailwind v4, shadcn/ui, base layout, and global styles                |
| `feature/docker-postgres-setup`     | `docker-compose.yml` with PostgreSQL 16 + pgAdmin, named volumes, health check                |
| `feature/backend-scaffold`          | FastAPI application structure, SQLAlchemy, Alembic, health endpoint, Docker multi-stage build |
| `feature/public-pages`              | All five public marketing pages (landing, about, how-it-works, pricing, contact) + error pages |
| `feature/database-schema`           | Prisma schema (User, Prediction, ModelMetrics), Prisma migrations, seed script, API routes     |
| `feature/clerk-auth`                | Clerk integration, `middleware.js`, protected route layout, sidebar, `useSyncUser` hook        |
| `feature/ml-validation-and-face-detection` | Modules 1 (ImageValidator) + 2 (FaceDetector) + detection API endpoints              |
| `feature/ml-preprocessing`          | Module 3 (ImagePreprocessor): resize, grayscale, equaliseHist, normalise                      |
| `feature/ml-lbp-extraction`         | Module 4 (LBPExtractor): uniform LBP histogram, 59-element vector                             |
| `feature/ml-dct-extraction`         | Module 5 (DCTExtractor): DCT block, log compression, 1024-element vector                      |
| `feature/ml-feature-fusion`         | Module 6 (FeatureFusion): MinMax normalisation + concatenation → 1083-dimensional vector       |
| `feature/ml-kmeans-clustering`      | Module 7 (KMeansClassifier): K-Means k=2, synthetic training, confidence scoring, persistence |
| `feature/ml-classification-evaluation` | Module 8 (ModelEvaluator): Accuracy, FAR, FRR, F1, confusion matrix, evaluation report   |
| `feature/detection-integration`     | End-to-end `/api/detect` handler: Clerk auth → Cloudinary → FastAPI → Prisma; `/detect` page UI |

One bug-fix commit (`fix(api): remove duplicate Detection tag from router include`) was made directly on the feature branch during development of the validation/face-detection module.

---

## 7. Database Design

### 7.1 Prisma Schema (PostgreSQL 16)

The frontend uses Prisma ORM (v5.22.0). Three models are defined in `prisma/schema.prisma`:

**Enums:**

```
Role   → USER | ADMIN
Result → PENDING | REAL | MORPHED | ERROR
```

**Model: User**

| Field      | Type     | Constraint       | Description                        |
|------------|----------|------------------|------------------------------------|
| id         | String   | PK, cuid()       | Internal primary key               |
| clerkId    | String   | UNIQUE           | Clerk-issued user identifier       |
| email      | String   | UNIQUE           | User email address                 |
| name       | String?  | nullable         | Display name                       |
| imageUrl   | String?  | nullable         | Profile picture URL                |
| role       | Role     | DEFAULT USER     | Access level                       |
| predictions| Prediction[] | relation    | One-to-many: all user predictions  |
| createdAt  | DateTime | default now()    | Account creation timestamp         |
| updatedAt  | DateTime | @updatedAt       | Auto-updated on mutation           |

**Model: Prediction**

| Field            | Type     | Constraint          | Description                              |
|------------------|----------|---------------------|------------------------------------------|
| id               | String   | PK, cuid()          | Internal primary key                     |
| userId           | String   | FK → User.id        | Owner of this prediction                 |
| imageUrl         | String   |                     | Cloudinary CDN URL of the uploaded image |
| imageName        | String   |                     | Original filename                        |
| imageSize        | Int      |                     | File size in bytes                       |
| result           | Result   | DEFAULT PENDING     | Classification outcome                   |
| confidence       | Float?   | nullable, [0.0–1.0] | K-Means confidence score                 |
| faceCount        | Int?     | nullable            | Number of faces detected                 |
| processingTimeMs | Float?   | nullable            | End-to-end latency                       |
| lbpVector        | Json?    | nullable            | 59-element LBP feature vector            |
| dctVector        | Json?    | nullable            | 1024-element DCT feature vector          |
| fusedVector      | Json?    | nullable            | 1083-element fused feature vector        |
| errorMessage     | String?  | nullable            | Error detail if result=ERROR             |
| metadata         | Json?    | nullable            | Image dimensions, format, etc.           |
| createdAt        | DateTime | default now()       | Prediction timestamp                     |

**Model: ModelMetrics**

| Field        | Type     | Description                                         |
|--------------|----------|-----------------------------------------------------|
| id           | String   | PK, cuid()                                          |
| accuracy     | Float    | (TP+TN)/N                                           |
| far          | Float    | False Acceptance Rate                               |
| frr          | Float    | False Rejection Rate                                |
| f1Score      | Float    | Harmonic mean of precision and recall               |
| precision    | Float    | TP/(TP+FP)                                          |
| recall       | Float    | TP/(TP+FN)                                          |
| totalSamples | Int      | Number of test samples used                         |
| dataSource   | String   | "synthetic" or "real"                               |
| recordedAt   | DateTime | Snapshot timestamp                                  |

### 7.2 Alembic Status

Alembic is installed and configured in the backend (`alembic>=1.14.0`). However, the `backend/alembic/versions/` directory contains no migration files at the time of writing. Database schema changes are currently managed through Prisma migrations on the frontend side.

---

## 8. Results

> **Critical caveat:** All metrics in this section are computed on **synthetic test data** generated from the same Gaussian distributions used during training (tight cluster for REAL, wide cluster for MORPHED), using a different random seed (99 vs 42). These numbers reflect the model's ability to separate two artificial Gaussian blobs in 1083-dimensional space — not its performance on real face morphing datasets. They should not be interpreted as real-world detection capability.

### 8.1 Evaluation Configuration

| Parameter                | Value                                |
|--------------------------|--------------------------------------|
| Test samples             | 200 (100 REAL + 100 MORPHED)         |
| Test random seed         | 99                                   |
| Training random seed     | 42                                   |
| REAL test std            | 0.05 (same as training distribution) |
| MORPHED test std         | 0.15 (same as training distribution) |
| Evaluation endpoint      | GET /api/v1/detection/evaluation     |

### 8.2 Metric Definitions

| Metric    | Formula                             | Interpretation                                              |
|-----------|-------------------------------------|-------------------------------------------------------------|
| Accuracy  | (TP+TN)/N                           | Overall classification rate                                 |
| FAR       | FP/(FP+TN)                          | Fraction of morphed faces incorrectly accepted — primary security metric |
| FRR       | FN/(FN+TP)                          | Fraction of real faces incorrectly rejected — user experience metric |
| Precision | TP/(TP+FP)                          | Of all "REAL" predictions, how many are correct             |
| Recall    | TP/(TP+FN)                          | Of all actual real faces, how many are accepted             |
| F1 Score  | 2·P·R/(P+R)                         | Harmonic mean of precision and recall                       |

### 8.3 Test Suite

179 test functions have been written across the following test files:

| Test file                     | Scope                                                     |
|-------------------------------|-----------------------------------------------------------|
| `test_image_validator.py`     | Module 1 — all validation checks, edge cases              |
| `test_face_detector.py`       | Module 2 — Haar detection, bounding box logic             |
| `test_image_preprocessor.py`  | Module 3 — resize, grayscale, equalisation, normalisation |
| `test_lbp_extractor.py`       | Module 4 — LBP codes, histogram, 59-element output        |
| `test_dct_extractor.py`       | Module 5 — DCT block, log compression, 1024-element output|
| `test_feature_fusion.py`      | Module 6 — MinMax normalisation, concatenation, 1083 dims |
| `test_kmeans_classifier.py`   | Module 7 — training, prediction, confidence, persistence  |
| `test_evaluator.py`           | Module 8 — metric formulas, boundary conditions, report   |
| `test_detection_endpoints.py` | Integration tests for all 12 API endpoints                |
| `test_health.py`              | Health endpoint                                           |

**These 179 test functions have been written but have not been executed in a passing state at the time of this report.** The Python environment on the development machine did not permit `pytest` to run during report generation. Test execution should be verified before submission.

---

## 9. Limitations and Future Work

### 9.1 Synthetic Training Data

The most significant limitation of the current system is that the K-Means classifier is trained and evaluated on programmatically generated Gaussian blobs, not real face photographs. This means:

- The reported accuracy, FAR, FRR, and F1 scores reflect the model's ability to separate artificial distributions, not to detect real morphing attacks.
- The feature space characteristics of real morphed faces (GAN artefacts, compression signatures, sensor noise patterns) are not captured by the synthetic data.

**Remediation:** Replace the `_train_on_synthetic_data()` call with a real labelled dataset such as MorGAN, SMDD (Synthetic Morphing Detection Database), or FERET-based morphing collections.

### 9.2 Empty Alembic Versions

The Alembic migration framework is installed but no migration scripts exist in `backend/alembic/versions/`. Backend database schema changes cannot be versioned or rolled back without these files. Database schema management currently relies on Prisma (frontend-only).

### 9.3 No CI/CD Pipeline

No GitHub Actions workflows exist in the repository (`.github/workflows/` directory is absent). Automated lint, type-check, and test execution on pull requests have not been configured.

### 9.4 No Sentry Integration

Sentry error tracking is documented as a planned feature in project documentation but has not been implemented. Neither `sentry-sdk` nor any Sentry DSN configuration appears in `requirements.txt`, `requirements-dev.txt`, or any application module.

### 9.5 Unsupervised Limitations

K-Means is a centroid-based method sensitive to initialisation and the "curse of dimensionality" in 1083-dimensional space. The `n_init=10` setting mitigates initialisation variance, but the model has no mechanism to adapt online to new morphing techniques. Supervised alternatives (SVM, Random Forest, or a fine-tuned CNN) trained on a real dataset would be expected to significantly outperform K-Means on real-world data.

### 9.6 Future Work

1. **Real dataset integration:** Collect or license a labelled face morphing dataset; replace the synthetic training path.
2. **Alembic migrations:** Write initial migration scripts to bring the backend schema under version control.
3. **CI/CD:** Add GitHub Actions workflows for lint (`eslint`, `ruff`), type checking, and `pytest` on all pull requests.
4. **Sentry integration:** Add `sentry-sdk` to `requirements.txt` and initialise in `main.py`.
5. **Feature expansion:** Experiment with HOG (Histogram of Oriented Gradients) or SIFT as additional feature descriptors alongside LBP+DCT.
6. **DCT block size tuning:** The 32×32 block is an empirical choice from the face forensics literature. A grid search over block sizes {16, 32, 64} with a real dataset may improve F1.
7. **PDF report download:** Implement the planned `/detect` page feature to export a per-prediction PDF report.
8. **Admin analytics page:** Complete the `/analytics` route for displaying ModelMetrics over time using Recharts.

---

## 10. Conclusion

This project delivers a complete, production-structured face morphing detection system with an eight-module ML pipeline, a 12-endpoint FastAPI REST API, a PostgreSQL-backed Next.js 15 frontend with Clerk authentication and Cloudinary image storage, and 179 written test functions. All major architectural components — image validation, Haar cascade face detection, preprocessing (INTER_AREA resize, equalizeHist, [0,1] normalisation), LBP uniform histogram, DCT log-compression, MinMax feature fusion, K-Means clustering, and a formal biometric evaluation module — are implemented and documented in source code.

The primary outstanding limitation is the reliance on synthetic training and test data. The evaluation metrics reported by the running system should be interpreted only as validation that the pipeline functions correctly as a software artifact, not as evidence of real-world detection performance. Replacing the synthetic data with a real labelled dataset is the single highest-impact next step.

---

## 11. References

1. Ojala, T., Pietikäinen, M., & Harwood, D. (1996). A comparative study of texture measures with classification based on feature distributions. *Pattern Recognition*, 29(1), 51–59.
2. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(7), 971–987.
3. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *Proceedings of IEEE CVPR*, 1, 511–518.
4. Ahmed, N., Natarajan, T., & Rao, K. R. (1974). Discrete cosine transform. *IEEE Transactions on Computers*, C-23(1), 90–93.
5. Damer, N., Saladié, A. M., Braun, A., & Kuijper, A. (2018). MorGAN: Recognition vulnerability and attack detectability of face morphing attacks created by generative adversarial network. *Proceedings of BTAS*.
6. Scherhag, U., Rathgeb, C., Merkle, J., Breithaupt, R., & Busch, C. (2019). Face recognition systems under morphing attacks: A survey. *IEEE Access*, 7, 23012–23026.
7. FastAPI documentation. https://fastapi.tiangolo.com/
8. scikit-image: Image processing in Python. https://scikit-image.org/
9. scikit-learn: Machine learning in Python. https://scikit-learn.org/
10. Prisma ORM documentation. https://www.prisma.io/docs/
11. Clerk authentication documentation. https://clerk.com/docs/
