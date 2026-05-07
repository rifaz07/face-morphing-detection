"""
Module 7 — K-Means Clustering Classifier

Purpose: Cluster 1083-dimensional fused feature vectors (LBP + DCT) into two
groups representing REAL and MORPHED face images.

Why unsupervised (K-Means)?
    Labeled face morphing datasets are expensive to collect and annotate.
    K-Means can discover natural groupings in the feature space without ground
    truth labels — the assumption is that real and morphed faces produce
    statistically different LBP+DCT signatures and will cluster separately.

Confidence score:
    Distance-to-centroid is converted to a [0, 1] confidence value using:

        confidence = 1 / (1 + d / μ_cluster)

    where d is the sample's distance to its assigned centroid and μ_cluster is
    the mean distance of all *training* samples in that cluster.  This anchors
    "50% confidence" at the average cluster radius and provides an intuitive
    interpretation: samples sitting on the centroid score 1.0; samples at the
    fringe of the cluster score ~0.5; outliers score near 0.

Label assignment:
    After K-Means fit, cluster labels are assigned by majority vote using the
    ground-truth labels supplied at training time.  For synthetic training the
    correct labels are known; for production training a labelled validation set
    should be supplied.

    If no labels are provided, clusters are named "CLUSTER_0" / "CLUSTER_1".

NOTE — Synthetic data:
    When no saved model is found on startup, the classifier auto-trains on
    synthetic feature vectors (500 REAL + 500 MORPHED) so the system is
    immediately usable.  Replace with a real labelled dataset for production.
"""

import time
from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from pydantic import BaseModel
from sklearn.cluster import KMeans

from app.ml.exceptions import ImageProcessingError


_MODELS_DIR = Path(__file__).parent.parent / "models"
_MODEL_FILENAME = "kmeans_model.joblib"
_META_FILENAME = "kmeans_meta.joblib"

_N_CLUSTERS = 2
_RANDOM_STATE = 42
_FUSED_VECTOR_SIZE = 1083

# Synthetic training parameters
_N_SYNTHETIC_REAL = 500
_N_SYNTHETIC_MORPHED = 500
_REAL_STD = 0.05    # tight cluster — real faces have consistent texture
_MORPHED_STD = 0.15  # loose cluster — morphed faces vary more


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------


class PredictionResult(BaseModel):
    """Result of classifying a single 1083-dim feature vector."""

    prediction: str
    confidence: float
    cluster_id: int
    distance_to_centroid: float
    processing_time_ms: float


class TrainingResult(BaseModel):
    """Result of a K-Means training run."""

    samples_trained: int
    inertia: float
    iterations: int
    converged: bool
    cluster_labels: dict
    silhouette_score: float | None
    accuracy: float | None
    trained_on_synthetic: bool
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Current state of the K-Means model."""

    is_fitted: bool
    model_type: str
    n_clusters: int
    cluster_labels: dict
    training_samples: int | None
    inertia: float | None
    trained_on_synthetic: bool
    model_path: str


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class KMeansClassifier:
    """
    K-Means classifier for face morphing detection (Module 7).

    Usage
    -----
    The class is designed as a singleton (see main.py lifespan).  On
    construction it attempts to load a previously saved model from disk.
    If none exists it auto-trains on synthetic data so the ``/classify``
    endpoint is always available immediately.

    Public methods
    --------------
    train(feature_vectors, labels) → TrainingResult
        Fit a new K-Means model.  Saves model to disk on success.
    predict(feature_vector)        → PredictionResult
        Classify one 1083-dim vector.
    save_model()                   → bool
    load_model()                   → bool
    get_model_info()               → ModelInfo
    """

    def __init__(
        self,
        n_clusters: int = _N_CLUSTERS,
        random_state: int = _RANDOM_STATE,
        models_dir: Path | None = None,
    ) -> None:
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._models_dir: Path = models_dir or _MODELS_DIR

        self._kmeans: KMeans | None = None
        self._cluster_labels: dict[int, str] = {}
        self._mean_cluster_distances: dict[int, float] = {}
        self._training_samples: int | None = None
        self._trained_on_synthetic: bool = False

        self._models_dir.mkdir(parents=True, exist_ok=True)

        if not self.load_model():
            logger.info("No saved K-Means model found — training on synthetic data.")
            self._train_on_synthetic_data()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        feature_vectors: list[list[float]],
        labels: list[str] | None = None,
        trained_on_synthetic: bool = False,
    ) -> TrainingResult:
        """
        Fit K-Means on a list of 1083-dimensional fused feature vectors.

        Args:
            feature_vectors:      List of 1083-element fused vectors (from Module 6).
            labels:               Optional ground-truth labels ("REAL" or "MORPHED")
                                  for majority-vote cluster-label assignment.
            trained_on_synthetic: Set True when using generated data (logged/reported).

        Returns:
            TrainingResult with inertia, iterations, cluster labels, and accuracy.

        Raises:
            ImageProcessingError: When fewer than n_clusters samples provided, or a
                                  vector has the wrong length.
        """
        if len(feature_vectors) < self._n_clusters:
            raise ImageProcessingError(
                f"Need at least {self._n_clusters} samples to train K-Means, "
                f"got {len(feature_vectors)}.",
                details={"received": len(feature_vectors), "required": self._n_clusters},
            )
        for i, vec in enumerate(feature_vectors):
            if len(vec) != _FUSED_VECTOR_SIZE:
                raise ImageProcessingError(
                    f"Feature vector at index {i} has length {len(vec)}, "
                    f"expected {_FUSED_VECTOR_SIZE}.",
                    details={"index": i, "received_length": len(vec)},
                )

        t_start = time.perf_counter()
        X = np.array(feature_vectors, dtype=np.float64)

        self._kmeans = KMeans(
            n_clusters=self._n_clusters,
            random_state=self._random_state,
            n_init=10,
            max_iter=300,
        )
        assignments = self._kmeans.fit_predict(X)
        converged = self._kmeans.n_iter_ < self._kmeans.max_iter

        # Per-cluster mean distance to centroid (used for confidence at inference).
        self._mean_cluster_distances = {}
        for c in range(self._n_clusters):
            mask = assignments == c
            if mask.any():
                dists = np.linalg.norm(X[mask] - self._kmeans.cluster_centers_[c], axis=1)
                self._mean_cluster_distances[c] = float(dists.mean())
            else:
                self._mean_cluster_distances[c] = 1.0  # degenerate edge case

        # Silhouette score (requires at least 2 populated clusters with > 1 sample).
        silhouette: float | None = None
        unique_assigned = np.unique(assignments)
        if len(unique_assigned) >= 2 and len(feature_vectors) > self._n_clusters:
            try:
                from sklearn.metrics import silhouette_score as _sil
                silhouette = round(float(_sil(X, assignments)), 4)
            except Exception:
                pass

        # Label assignment by majority vote.
        self._cluster_labels = {}
        accuracy: float | None = None
        if labels is not None:
            if len(labels) != len(feature_vectors):
                raise ImageProcessingError(
                    "Number of labels must match number of feature vectors.",
                    details={"n_vectors": len(feature_vectors), "n_labels": len(labels)},
                )
            normalised = [lb.upper() for lb in labels]
            for c in range(self._n_clusters):
                mask = assignments == c
                cluster_gt = [normalised[i] for i, m in enumerate(mask) if m]
                real_count = cluster_gt.count("REAL")
                morphed_count = cluster_gt.count("MORPHED")
                self._cluster_labels[c] = "REAL" if real_count >= morphed_count else "MORPHED"
            predictions = [self._cluster_labels[int(a)] for a in assignments]
            correct = sum(p == gt for p, gt in zip(predictions, normalised))
            accuracy = round(correct / len(normalised), 4)
        else:
            for c in range(self._n_clusters):
                self._cluster_labels[c] = f"CLUSTER_{c}"

        self._training_samples = len(feature_vectors)
        self._trained_on_synthetic = trained_on_synthetic

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.save_model()

        logger.info(
            "K-Means trained | n={} k={} inertia={:.2f} iters={} sil={} acc={} synthetic={}",
            len(feature_vectors),
            self._n_clusters,
            self._kmeans.inertia_,
            self._kmeans.n_iter_,
            f"{silhouette:.4f}" if silhouette is not None else "N/A",
            f"{accuracy:.4f}" if accuracy is not None else "N/A",
            trained_on_synthetic,
        )

        return TrainingResult(
            samples_trained=self._training_samples,
            inertia=round(float(self._kmeans.inertia_), 4),
            iterations=int(self._kmeans.n_iter_),
            converged=converged,
            cluster_labels=self._cluster_labels,
            silhouette_score=silhouette,
            accuracy=accuracy,
            trained_on_synthetic=trained_on_synthetic,
            processing_time_ms=round(elapsed_ms, 3),
        )

    def predict(self, feature_vector: list[float]) -> PredictionResult:
        """
        Classify a single 1083-element fused feature vector.

        Args:
            feature_vector: 1083-element vector from FeatureFusion (Module 6).

        Returns:
            PredictionResult with prediction label, confidence, and metadata.

        Raises:
            ImageProcessingError: When the model is not fitted or vector length
                                  does not match expected 1083.
        """
        if self._kmeans is None:
            raise ImageProcessingError(
                "K-Means model is not fitted.  Call train() first or ensure a "
                "saved model exists at startup.",
                details={"model_path": str(self._models_dir / _MODEL_FILENAME)},
            )
        if len(feature_vector) != _FUSED_VECTOR_SIZE:
            raise ImageProcessingError(
                f"Feature vector has length {len(feature_vector)}, "
                f"expected {_FUSED_VECTOR_SIZE}.",
                details={"received_length": len(feature_vector), "expected": _FUSED_VECTOR_SIZE},
            )

        t_start = time.perf_counter()
        X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)

        cluster_id = int(self._kmeans.predict(X)[0])
        centroid = self._kmeans.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(X[0] - centroid))

        mean_dist = self._mean_cluster_distances.get(cluster_id, 1.0)
        confidence = 1.0 / (1.0 + distance / max(mean_dist, 1e-9))
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        label = self._cluster_labels.get(cluster_id, f"CLUSTER_{cluster_id}")
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        logger.debug(
            "K-Means predict | cluster={} label={} dist={:.4f} conf={:.3f} {:.2f}ms",
            cluster_id, label, distance, confidence, elapsed_ms,
        )

        return PredictionResult(
            prediction=label,
            confidence=confidence,
            cluster_id=cluster_id,
            distance_to_centroid=round(distance, 4),
            processing_time_ms=round(elapsed_ms, 3),
        )

    def save_model(self) -> bool:
        """Persist the fitted K-Means model and metadata to disk."""
        if self._kmeans is None:
            return False
        try:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._kmeans, self._models_dir / _MODEL_FILENAME)
            meta = {
                "cluster_labels": self._cluster_labels,
                "mean_cluster_distances": self._mean_cluster_distances,
                "training_samples": self._training_samples,
                "trained_on_synthetic": self._trained_on_synthetic,
            }
            joblib.dump(meta, self._models_dir / _META_FILENAME)
            logger.info("K-Means model saved → {}", self._models_dir / _MODEL_FILENAME)
            return True
        except Exception as exc:
            logger.warning("Failed to save K-Means model: {}", exc)
            return False

    def load_model(self) -> bool:
        """Load K-Means model and metadata from disk if available."""
        model_path = self._models_dir / _MODEL_FILENAME
        meta_path = self._models_dir / _META_FILENAME
        if not model_path.exists():
            return False
        try:
            self._kmeans = joblib.load(model_path)
            if meta_path.exists():
                meta = joblib.load(meta_path)
                self._cluster_labels = meta.get("cluster_labels", {})
                self._mean_cluster_distances = meta.get("mean_cluster_distances", {})
                self._training_samples = meta.get("training_samples")
                self._trained_on_synthetic = meta.get("trained_on_synthetic", False)
            logger.info(
                "K-Means model loaded ← {} | labels={} synthetic={}",
                model_path,
                self._cluster_labels,
                self._trained_on_synthetic,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load K-Means model: {}", exc)
            self._kmeans = None
            return False

    def get_model_info(self) -> ModelInfo:
        """Return a snapshot of the current model state."""
        return ModelInfo(
            is_fitted=self._kmeans is not None,
            model_type="KMeans",
            n_clusters=self._n_clusters,
            cluster_labels=self._cluster_labels,
            training_samples=self._training_samples,
            inertia=round(float(self._kmeans.inertia_), 4) if self._kmeans else None,
            trained_on_synthetic=self._trained_on_synthetic,
            model_path=str(self._models_dir / _MODEL_FILENAME),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_on_synthetic_data(self) -> TrainingResult:
        """
        Generate synthetic REAL and MORPHED feature vectors and train K-Means.

        Real vectors form a tight cluster (std=0.05) around a centroid in the
        middle of the [0,1] feature space.  Morphed vectors form a wider,
        overlapping cluster (std=0.15) around a centroid on the opposite side
        of the space.  This contrast gives K-Means a clear separation signal.

        NOTE: Replace this with a real labelled face dataset for production use.
        """
        rng = np.random.default_rng(_RANDOM_STATE)

        # Real centroid: values in [0.35, 0.65] — middle of feature space.
        real_centroid = rng.uniform(0.35, 0.65, size=_FUSED_VECTOR_SIZE)

        # Morphed centroid: mirrored so clusters are well-separated.
        morphed_centroid = np.clip(1.0 - real_centroid + rng.normal(0, 0.05, size=_FUSED_VECTOR_SIZE), 0.0, 1.0)

        # Generate samples and clip to valid [0, 1] range.
        real_X = np.clip(
            real_centroid + rng.normal(0, _REAL_STD, size=(_N_SYNTHETIC_REAL, _FUSED_VECTOR_SIZE)),
            0.0,
            1.0,
        )
        morphed_X = np.clip(
            morphed_centroid + rng.normal(0, _MORPHED_STD, size=(_N_SYNTHETIC_MORPHED, _FUSED_VECTOR_SIZE)),
            0.0,
            1.0,
        )

        X = np.vstack([real_X, morphed_X])
        labels = ["REAL"] * _N_SYNTHETIC_REAL + ["MORPHED"] * _N_SYNTHETIC_MORPHED

        result = self.train(X.tolist(), labels=labels, trained_on_synthetic=True)
        logger.info(
            "Synthetic K-Means training complete | inertia={:.2f} labels={}",
            result.inertia,
            self._cluster_labels,
        )
        return result
