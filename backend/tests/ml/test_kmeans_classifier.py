"""
Unit tests for Module 7 — KMeansClassifier.

Tests verify synthetic training, prediction correctness, confidence bounds,
model persistence (save/load cycle), and cluster separation properties.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.ml.clustering.kmeans_classifier import (
    KMeansClassifier,
    _FUSED_VECTOR_SIZE,
    _N_CLUSTERS,
    _N_SYNTHETIC_MORPHED,
    _N_SYNTHETIC_REAL,
)
from app.ml.exceptions import ImageProcessingError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmp_models_dir(tmp_path_factory) -> Path:
    """Isolated temp directory so tests don't touch the real models/ folder."""
    return tmp_path_factory.mktemp("kmeans_models")


@pytest.fixture(scope="module")
def classifier(tmp_models_dir: Path) -> KMeansClassifier:
    """Classifier auto-trained on synthetic data using a temp dir."""
    return KMeansClassifier(models_dir=tmp_models_dir)


@pytest.fixture(scope="module")
def real_vector(classifier: KMeansClassifier) -> list[float]:
    """A sample vector that should sit close to the REAL centroid."""
    rng = np.random.default_rng(0)
    # Pull the real centroid from the fitted model and add tiny noise.
    real_cluster_id = _cluster_id_for_label(classifier, "REAL")
    centroid = classifier._kmeans.cluster_centers_[real_cluster_id]
    vec = np.clip(centroid + rng.normal(0, 0.01, size=_FUSED_VECTOR_SIZE), 0.0, 1.0)
    return vec.tolist()


@pytest.fixture(scope="module")
def morphed_vector(classifier: KMeansClassifier) -> list[float]:
    """A sample vector that should sit close to the MORPHED centroid."""
    rng = np.random.default_rng(1)
    morphed_cluster_id = _cluster_id_for_label(classifier, "MORPHED")
    centroid = classifier._kmeans.cluster_centers_[morphed_cluster_id]
    vec = np.clip(centroid + rng.normal(0, 0.01, size=_FUSED_VECTOR_SIZE), 0.0, 1.0)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cluster_id_for_label(clf: KMeansClassifier, label: str) -> int:
    """Return the cluster ID assigned to a given label string."""
    for cid, lbl in clf._cluster_labels.items():
        if lbl == label:
            return cid
    raise ValueError(f"Label '{label}' not found in cluster_labels: {clf._cluster_labels}")


def _random_vector(seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=_FUSED_VECTOR_SIZE).tolist()


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


def test_model_trains_successfully(classifier: KMeansClassifier) -> None:
    """After auto-training on synthetic data the model should be fitted."""
    assert classifier._kmeans is not None
    assert classifier.get_model_info().is_fitted is True


def test_training_result_samples_count(classifier: KMeansClassifier) -> None:
    assert classifier._training_samples == _N_SYNTHETIC_REAL + _N_SYNTHETIC_MORPHED


def test_training_result_converged(tmp_models_dir: Path) -> None:
    clf = KMeansClassifier(models_dir=tmp_models_dir)
    info = clf.get_model_info()
    assert info.is_fitted
    # Check via direct retraining
    result = clf._train_on_synthetic_data()
    assert result.converged is True


def test_cluster_labels_assigned(classifier: KMeansClassifier) -> None:
    labels = set(classifier._cluster_labels.values())
    assert "REAL" in labels
    assert "MORPHED" in labels


def test_trained_on_synthetic_flag(classifier: KMeansClassifier) -> None:
    assert classifier._trained_on_synthetic is True
    assert classifier.get_model_info().trained_on_synthetic is True


def test_model_info_fields(classifier: KMeansClassifier) -> None:
    info = classifier.get_model_info()
    assert info.model_type == "KMeans"
    assert info.n_clusters == _N_CLUSTERS
    assert info.inertia is not None
    assert info.inertia > 0
    assert info.training_samples == _N_SYNTHETIC_REAL + _N_SYNTHETIC_MORPHED


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


def test_prediction_returns_real_or_morphed(
    classifier: KMeansClassifier, real_vector: list[float]
) -> None:
    result = classifier.predict(real_vector)
    assert result.prediction in {"REAL", "MORPHED"}


def test_confidence_between_0_and_1(
    classifier: KMeansClassifier, real_vector: list[float]
) -> None:
    result = classifier.predict(real_vector)
    assert 0.0 <= result.confidence <= 1.0


def test_predict_returns_cluster_id(
    classifier: KMeansClassifier, real_vector: list[float]
) -> None:
    result = classifier.predict(real_vector)
    assert result.cluster_id in {0, 1}


def test_predict_distance_to_centroid_is_non_negative(
    classifier: KMeansClassifier, real_vector: list[float]
) -> None:
    result = classifier.predict(real_vector)
    assert result.distance_to_centroid >= 0.0


def test_predict_processing_time_is_positive(
    classifier: KMeansClassifier, real_vector: list[float]
) -> None:
    result = classifier.predict(real_vector)
    assert result.processing_time_ms >= 0.0


def test_real_vector_classified_as_real(
    classifier: KMeansClassifier, real_vector: list[float]
) -> None:
    """Vector sitting on the REAL centroid should be classified as REAL."""
    result = classifier.predict(real_vector)
    assert result.prediction == "REAL"


def test_morphed_vector_classified_as_morphed(
    classifier: KMeansClassifier, morphed_vector: list[float]
) -> None:
    """Vector sitting on the MORPHED centroid should be classified as MORPHED."""
    result = classifier.predict(morphed_vector)
    assert result.prediction == "MORPHED"


def test_centroid_vector_has_high_confidence(
    classifier: KMeansClassifier, real_vector: list[float]
) -> None:
    """A vector very close to the centroid should score > 0.8 confidence."""
    result = classifier.predict(real_vector)
    assert result.confidence > 0.8


def test_different_vectors_give_different_predictions(
    classifier: KMeansClassifier,
    real_vector: list[float],
    morphed_vector: list[float],
) -> None:
    """Real and morphed centroid vectors must land in different clusters."""
    r = classifier.predict(real_vector)
    m = classifier.predict(morphed_vector)
    assert r.cluster_id != m.cluster_id
    assert r.prediction != m.prediction


def test_random_vectors_produce_valid_predictions(classifier: KMeansClassifier) -> None:
    """Any valid 1083-dim vector should return a well-formed result."""
    for seed in range(5):
        vec = _random_vector(seed)
        result = classifier.predict(vec)
        assert result.prediction in {"REAL", "MORPHED"}
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Cluster separation tests
# ---------------------------------------------------------------------------


def test_synthetic_real_cluster_tighter_than_morphed(classifier: KMeansClassifier) -> None:
    """REAL cluster should have a smaller mean distance to centroid than MORPHED."""
    real_id = _cluster_id_for_label(classifier, "REAL")
    morphed_id = _cluster_id_for_label(classifier, "MORPHED")
    real_mean_dist = classifier._mean_cluster_distances[real_id]
    morphed_mean_dist = classifier._mean_cluster_distances[morphed_id]
    # REAL uses std=0.05, MORPHED uses std=0.15 → REAL cluster is tighter.
    assert real_mean_dist < morphed_mean_dist


def test_inertia_is_finite_and_positive(classifier: KMeansClassifier) -> None:
    import math
    inertia = classifier._kmeans.inertia_
    assert inertia > 0
    assert math.isfinite(inertia)


# ---------------------------------------------------------------------------
# Save / load cycle
# ---------------------------------------------------------------------------


def test_model_saves_and_loads(tmp_path: Path) -> None:
    """Train → save → delete runtime state → load → predict must still work."""
    clf = KMeansClassifier(models_dir=tmp_path)
    assert clf._kmeans is not None

    # Verify model files exist after auto-training.
    model_file = tmp_path / "kmeans_model.joblib"
    meta_file = tmp_path / "kmeans_meta.joblib"
    assert model_file.exists()
    assert meta_file.exists()

    # Wipe runtime state.
    saved_labels = clf._cluster_labels.copy()
    clf._kmeans = None
    clf._cluster_labels = {}

    # Load from disk.
    assert clf.load_model() is True
    assert clf._kmeans is not None
    assert clf._cluster_labels == saved_labels

    # Prediction should still work after reload.
    vec = _random_vector(99)
    result = clf.predict(vec)
    assert result.prediction in {"REAL", "MORPHED"}


def test_load_model_returns_false_when_no_file(tmp_path: Path) -> None:
    clf = KMeansClassifier.__new__(KMeansClassifier)
    clf._n_clusters = 2
    clf._random_state = 42
    clf._models_dir = tmp_path / "nonexistent"
    clf._kmeans = None
    clf._cluster_labels = {}
    clf._mean_cluster_distances = {}
    clf._training_samples = None
    clf._trained_on_synthetic = False
    assert clf.load_model() is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_predict_raises_when_not_fitted() -> None:
    clf = KMeansClassifier.__new__(KMeansClassifier)
    clf._kmeans = None
    clf._cluster_labels = {}
    clf._mean_cluster_distances = {}
    clf._models_dir = Path("/tmp")
    with pytest.raises(ImageProcessingError, match="not fitted"):
        clf.predict(_random_vector())


def test_predict_raises_for_wrong_vector_length(classifier: KMeansClassifier) -> None:
    short_vec = [0.5] * 100
    with pytest.raises(ImageProcessingError, match="length 100"):
        classifier.predict(short_vec)


def test_train_raises_for_too_few_samples(tmp_models_dir: Path) -> None:
    clf = KMeansClassifier(models_dir=tmp_models_dir)
    with pytest.raises(ImageProcessingError, match="at least 2 samples"):
        clf.train([[0.5] * _FUSED_VECTOR_SIZE])


def test_train_raises_for_wrong_vector_length(tmp_models_dir: Path) -> None:
    clf = KMeansClassifier(models_dir=tmp_models_dir)
    bad_vectors = [[0.5] * 100, [0.3] * 100]
    with pytest.raises(ImageProcessingError, match="length 100"):
        clf.train(bad_vectors)
