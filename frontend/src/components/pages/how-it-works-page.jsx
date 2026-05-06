"use client";

import { motion } from "framer-motion";
import {
  ImagePlus, Scan, Layers, Fingerprint, Activity,
  GitMerge, GitFork, ShieldCheck,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const fadeUp = { hidden: { opacity: 0, y: 28 }, show: { opacity: 1, y: 0 } };

const MODULES = [
  {
    icon: ImagePlus,
    title: "Image Input & Validation",
    summary: "Accept and validate the uploaded face image before processing begins.",
    body: [
      "The pipeline begins by receiving the uploaded image and performing strict validation. Accepted formats are JPEG, PNG, and WebP, with a maximum file size of 10 MB. This prevents malformed or corrupt files from entering the processing chain.",
      "We also check minimum resolution (at least 64×64 pixels) because face detection requires sufficient pixel density to work reliably. If any check fails, the user receives a descriptive error message rather than a silent failure.",
    ],
    snippet: `# FastAPI endpoint validates before processing
@app.post("/detect")
async def detect(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Unsupported file type")
    data = await file.read()
    if len(data) > MAX_SIZE_BYTES:
        raise HTTPException(413, "File too large")`,
    color: "from-violet-500/20 to-violet-500/5",
    iconColor: "text-violet-500",
  },
  {
    icon: Scan,
    title: "Face Detection",
    summary: "Locate and crop the face region using OpenCV Haar Cascade.",
    body: [
      "We use OpenCV's pre-trained Haar Cascade classifier (haarcascade_frontalface_default.xml) to detect the face bounding box within the uploaded image. Haar Cascades use Integral Images for fast feature evaluation across sliding windows.",
      "If no face is detected, the request is rejected immediately. If multiple faces are found, the largest bounding box is selected as the primary subject. The cropped face region is then forwarded to the preprocessing stage.",
    ],
    snippet: `face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5
)`,
    color: "from-blue-500/20 to-blue-500/5",
    iconColor: "text-blue-500",
  },
  {
    icon: Layers,
    title: "Preprocessing",
    summary: "Standardise the face crop for consistent feature extraction.",
    body: [
      "The detected face region is resized to a fixed 128×128 pixel resolution. This ensures every feature vector has the same dimensionality regardless of the original image size, which is a prerequisite for K-Means clustering.",
      "The resized crop is then converted to grayscale (single channel) and normalised so pixel intensities fall in the [0, 1] range. Normalisation prevents features derived from very bright or very dark images from being artificially amplified.",
    ],
    snippet: `face_crop = img[y:y+h, x:x+w]
face_resized = cv2.resize(face_crop, (128, 128))
gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
# Normalise to [0, 1]
norm = gray.astype(np.float32) / 255.0`,
    color: "from-teal-500/20 to-teal-500/5",
    iconColor: "text-teal-500",
  },
  {
    icon: Fingerprint,
    title: "LBP Feature Extraction",
    summary: "Encode micro-texture patterns that betray morphing boundaries.",
    body: [
      "Local Binary Patterns (LBP) describe the texture of each pixel by comparing it with its 8 circular neighbours at radius R=1. Each comparison (greater/lesser) produces a binary bit; the 8 bits form a code in [0, 255]. A histogram of all codes across the image becomes the feature vector.",
      "Morphed images contain texture inconsistencies at blend boundaries — regions where two different skin textures meet. LBP histograms of morphed faces show characteristic peaks and troughs in specific texture-code bins that genuine faces do not.",
    ],
    snippet: `from skimage.feature import local_binary_pattern

# P=8 neighbours, R=1 radius, 'uniform' variant
lbp = local_binary_pattern(
    gray_uint8, P=8, R=1, method='uniform'
)
# Histogram of 59 uniform patterns
hist, _ = np.histogram(
    lbp.ravel(), bins=59, range=(0, 59), density=True
)`,
    color: "from-purple-500/20 to-purple-500/5",
    iconColor: "text-purple-500",
  },
  {
    icon: Activity,
    title: "DCT Feature Extraction",
    summary: "Reveal frequency-domain artifacts left by the morphing process.",
    body: [
      "The Discrete Cosine Transform converts the spatial image into the frequency domain. High-frequency components represent fine detail and edges; low-frequency components represent broad structure. Morphing tools introduce characteristic compression artifacts when blending two images.",
      "We divide the face into non-overlapping 8×8 blocks (matching the JPEG compression block size), apply 2-D DCT to each, and retain the top-left 8×8 coefficients (the lowest frequencies) as the feature contribution for that block. These are concatenated into the DCT feature vector.",
    ],
    snippet: `from scipy.fft import dct

blocks = []
for i in range(0, 128, 8):
    for j in range(0, 128, 8):
        block = norm[i:i+8, j:j+8]
        dct_block = dct(dct(block.T).T)
        # Retain top-left 8x8 (low-freq)
        blocks.append(dct_block[:8, :8].flatten())
dct_features = np.concatenate(blocks)`,
    color: "from-cyan-500/20 to-cyan-500/5",
    iconColor: "text-cyan-500",
  },
  {
    icon: GitMerge,
    title: "Feature Fusion",
    summary: "Concatenate LBP and DCT vectors into a unified representation.",
    body: [
      "Feature fusion is the process of combining two complementary feature vectors — the LBP texture histogram and the DCT frequency coefficients — into a single, higher-dimensional vector. This fused representation captures information from both the spatial and frequency domains simultaneously.",
      "Simple concatenation is used after L2-normalising each sub-vector independently. Normalisation ensures neither feature type dominates the distance metric used by K-Means. The resulting fused vector typically has 512–1024 dimensions depending on DCT block sampling density.",
    ],
    snippet: `from sklearn.preprocessing import normalize

lbp_norm = normalize(lbp_hist.reshape(1, -1))[0]
dct_norm = normalize(dct_features.reshape(1, -1))[0]

# Fused feature vector
fused = np.concatenate([lbp_norm, dct_norm])
# Shape: (59 + 512,) = (571,)`,
    color: "from-orange-500/20 to-orange-500/5",
    iconColor: "text-orange-500",
  },
  {
    icon: GitFork,
    title: "K-Means Clustering",
    summary: "Assign the feature vector to the real or morphed cluster.",
    body: [
      "K-Means with K=2 is trained offline on a labelled dataset of genuine and morphed face feature vectors. The two cluster centroids converge to represent the average feature profile of each class. At inference time, a new feature vector is assigned to whichever centroid it is nearest to (using Euclidean distance).",
      "The confidence score is derived from the relative distances to each centroid — a point very close to one centroid and far from the other produces a high-confidence prediction, while a point equidistant from both produces a low-confidence result near the 50% boundary.",
    ],
    snippet: `from sklearn.cluster import KMeans
import joblib

# Training (offline)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(training_features)
joblib.dump(kmeans, "models/kmeans.joblib")

# Inference
model = joblib.load("models/kmeans.joblib")
label = model.predict(fused.reshape(1, -1))[0]
distances = model.transform(fused.reshape(1, -1))[0]
confidence = 1 - (distances.min() / distances.sum())`,
    color: "from-amber-500/20 to-amber-500/5",
    iconColor: "text-amber-500",
  },
  {
    icon: ShieldCheck,
    title: "Classification & Evaluation",
    summary: "Return the verdict, confidence score, and FAR/FRR metrics.",
    body: [
      "The cluster label is mapped to REAL (label 0) or MORPHED (label 1) based on the centroid that showed lower intra-cluster variance during training. The confidence score (0–100%) and the binarised prediction are returned to the client alongside the raw distance values for transparency.",
      "The system also reports False Acceptance Rate (FAR) — the proportion of morphed images incorrectly classified as real — and False Rejection Rate (FRR) — genuine images wrongly flagged as morphed. Both are evaluated on a held-out test set and updated with each model retrain.",
    ],
    snippet: `LABEL_MAP = {0: "REAL", 1: "MORPHED"}
verdict = LABEL_MAP[label]
confidence = round(float(confidence * 100), 2)

return {
    "verdict": verdict,
    "confidence": confidence,
    "is_morphed": label == 1,
    "far": model_metrics["FAR"],
    "frr": model_metrics["FRR"],
    "accuracy": model_metrics["accuracy"],
}`,
    color: "from-emerald-500/20 to-emerald-500/5",
    iconColor: "text-emerald-500",
  },
];

export function HowItWorksPage() {
  return (
    <main className="pb-24">
      {/* Hero */}
      <section className="py-16 sm:py-24 text-center px-4">
        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
          className="mx-auto max-w-3xl"
        >
          <Badge variant="outline" className="mb-4">ML Pipeline Deep-Dive</Badge>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight gradient-brand-text">
            How Our ML Pipeline Works
          </h1>
          <p className="mt-5 text-muted-foreground text-lg leading-relaxed">
            Eight discrete modules from raw image upload to final verdict.
            Every step is explained in plain language with the code that runs it.
          </p>
        </motion.div>
      </section>

      {/* Modules — alternating left/right */}
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 space-y-20 sm:space-y-28">
        {MODULES.map(({ icon: Icon, title, summary, body, snippet, color, iconColor }, i) => {
          const isEven = i % 2 === 0;
          return (
            <motion.article
              key={title}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.55 }}
              className={cn(
                "grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-start",
                !isEven && "lg:[direction:rtl]"
              )}
            >
              {/* Text side */}
              <div className={cn("flex flex-col gap-4", !isEven && "lg:[direction:ltr]")}>
                <div className="flex items-center gap-3">
                  <span className="text-5xl font-black gradient-brand-text opacity-30 leading-none select-none">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <div className={cn("w-10 h-10 rounded-xl bg-gradient-to-br flex items-center justify-center shrink-0", color)}>
                    <Icon className={cn("size-5", iconColor)} />
                  </div>
                </div>
                <div>
                  <h2 className="text-xl sm:text-2xl font-bold">{title}</h2>
                  <p className="text-sm font-medium text-muted-foreground mt-1">{summary}</p>
                </div>
                {body.map((para, j) => (
                  <p key={j} className="text-muted-foreground text-sm sm:text-base leading-relaxed">
                    {para}
                  </p>
                ))}
              </div>

              {/* Code side */}
              <div className={cn(!isEven && "lg:[direction:ltr]")}>
                <div className="rounded-xl overflow-hidden border bg-[oklch(0.15_0_0)] dark:bg-[oklch(0.12_0_0)]">
                  <div className="flex items-center gap-1.5 px-4 py-3 border-b border-white/10">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500/70" />
                    <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/70" />
                    <span className="w-2.5 h-2.5 rounded-full bg-green-500/70" />
                    <span className="ml-2 text-xs text-white/40 font-mono">pipeline_module_{String(i + 1).padStart(2, "0")}.py</span>
                  </div>
                  <pre className="text-xs sm:text-sm text-white/85 font-mono p-5 overflow-x-auto leading-relaxed">
                    <code>{snippet}</code>
                  </pre>
                </div>
              </div>
            </motion.article>
          );
        })}
      </div>
    </main>
  );
}
