"""
Module 8 — Classification Evaluation

Purpose: Compute standard binary-classification metrics for the face morphing
detection system and produce a viva-ready evaluation report.

Label convention used throughout this module
─────────────────────────────────────────────
  0 = REAL   (the class we want to *accept*)
  1 = MORPHED (the class we want to *reject*)

Confusion-matrix layout
───────────────────────
                   Predicted REAL  Predicted MORPHED
  Actual MORPHED      FP               TN
  Actual REAL         TP               FN

  → [[TN, FP],
     [FN, TP]]

This layout keeps the security-domain definitions consistent:
  • FP  = a MORPHED face incorrectly *accepted* as REAL   ← "False Acceptance"
  • FN  = a REAL face incorrectly *rejected* as MORPHED   ← "False Rejection"

Metric formulas
───────────────
  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  FAR       = FP / (FP + TN)   — False Acceptance Rate (morphed slips through)
  FRR       = FN / (FN + TP)   — False Rejection Rate  (real user blocked)
  Precision = TP / (TP + FP)   — of all "REAL" predictions, how many are correct
  Recall    = TP / (TP + FN)   — of all actual REAL faces, how many we accepted
  F1 Score  = 2 × Precision × Recall / (Precision + Recall)

Real-world stakes
─────────────────
  FAR matters in border control, banking, or document verification — a high FAR
  means attackers slip through.  FRR matters for user experience — a high FRR
  means genuine users are blocked.  FAR and FRR trade off: tightening the
  classifier to reduce FAR typically raises FRR.  The F1 score balances both.

Synthetic evaluation note
─────────────────────────
  Because no labelled real-world face-morphing dataset is bundled with this
  project, ``evaluate_on_synthetic()`` generates 200 test vectors drawn from
  the same distribution used during K-Means training (100 REAL tight cluster,
  100 MORPHED loose cluster) but with a different random seed.  Replace with a
  real holdout dataset for production use.
"""

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from app.ml.clustering.kmeans_classifier import KMeansClassifier


_FUSED_VECTOR_SIZE = 1083
_N_SYNTHETIC_TEST_REAL = 100
_N_SYNTHETIC_TEST_MORPHED = 100
_TEST_REAL_STD = 0.05
_TEST_MORPHED_STD = 0.15
_TEST_RANDOM_SEED = 99  # different from training seed (42)


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """All binary-classification metrics for one evaluation run."""

    accuracy: float
    far: float
    frr: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]
    total_samples: int
    correct_predictions: int
    tp: int
    tn: int
    fp: int
    fn: int
    evaluation_time_ms: float


class EvaluationReport(BaseModel):
    """Full viva-ready report wrapping metrics with interpretation and context."""

    metrics: EvaluationResult
    model_info: dict
    interpretation: dict
    recommendations: list[str]
    timestamp: str
    data_source: str


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """
    Module 8 — evaluates the K-Means classifier and produces an EvaluationReport.

    Usage
    -----
    Constructed with a fitted KMeansClassifier.  Call ``generate_report()``
    once at startup to cache the evaluation results.

    Public methods
    --------------
    evaluate(y_true, y_pred) → EvaluationResult
        Compute all metrics from pre-computed label arrays.
    evaluate_on_synthetic()  → EvaluationResult
        Generate synthetic test vectors, classify them, then evaluate.
    generate_report()        → EvaluationReport
        Full viva-ready report (calls evaluate_on_synthetic internally).
    """

    def __init__(self, classifier: "KMeansClassifier") -> None:
        self._classifier = classifier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: list[int],
        y_pred: list[int],
    ) -> EvaluationResult:
        """
        Compute all metrics from ground-truth and predicted label arrays.

        Args:
            y_true: Ground-truth labels — 0=REAL, 1=MORPHED.
            y_pred: Predicted labels   — 0=REAL, 1=MORPHED.

        Returns:
            EvaluationResult with accuracy, FAR, FRR, precision, recall,
            F1, confusion matrix, and raw TP/TN/FP/FN counts.

        Raises:
            ValueError: When y_true and y_pred have different lengths or are empty.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length "
                f"(got {len(y_true)} and {len(y_pred)})."
            )
        if not y_true:
            raise ValueError("y_true must not be empty.")

        t_start = time.perf_counter()

        # --- raw counts ---------------------------------------------------
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)  # REAL → REAL
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)  # MORPHED → MORPHED
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)  # MORPHED → REAL (false accept)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)  # REAL → MORPHED (false reject)

        n = tp + tn + fp + fn

        # --- metrics ------------------------------------------------------
        accuracy  = (tp + tn) / n if n > 0 else 0.0
        far       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "Evaluation | n={} acc={:.3f} FAR={:.3f} FRR={:.3f} F1={:.3f} ({:.1f}ms)",
            n, accuracy, far, frr, f1, elapsed_ms,
        )

        return EvaluationResult(
            accuracy=round(accuracy, 4),
            far=round(far, 4),
            frr=round(frr, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            confusion_matrix=[[tn, fp], [fn, tp]],
            total_samples=n,
            correct_predictions=tp + tn,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            evaluation_time_ms=round(elapsed_ms, 3),
        )

    def evaluate_on_synthetic(self) -> EvaluationResult:
        """
        Generate 200 synthetic test vectors, classify each with K-Means,
        and compute evaluation metrics.

        The test vectors are drawn from the *same* Gaussian distributions
        used in training (tight for REAL, loose for MORPHED) but with a
        different random seed (99 vs 42) so there is no data leakage.

        Returns:
            EvaluationResult computed against the synthetic test labels.
        """
        kmeans = self._classifier._kmeans
        if kmeans is None:
            raise RuntimeError("KMeansClassifier is not fitted — cannot evaluate.")

        rng = np.random.default_rng(_TEST_RANDOM_SEED)

        real_cluster_id = self._label_to_cluster_id("REAL")
        morphed_cluster_id = self._label_to_cluster_id("MORPHED")

        real_centroid = kmeans.cluster_centers_[real_cluster_id]
        morphed_centroid = kmeans.cluster_centers_[morphed_cluster_id]

        # Test vectors — same spread as training but independent seed
        real_test = np.clip(
            real_centroid + rng.normal(0, _TEST_REAL_STD, (_N_SYNTHETIC_TEST_REAL, _FUSED_VECTOR_SIZE)),
            0.0, 1.0,
        )
        morphed_test = np.clip(
            morphed_centroid + rng.normal(0, _TEST_MORPHED_STD, (_N_SYNTHETIC_TEST_MORPHED, _FUSED_VECTOR_SIZE)),
            0.0, 1.0,
        )

        y_true: list[int] = [0] * _N_SYNTHETIC_TEST_REAL + [1] * _N_SYNTHETIC_TEST_MORPHED
        y_pred: list[int] = []

        for vec in real_test:
            pred = self._classifier.predict(vec.tolist())
            y_pred.append(0 if pred.prediction == "REAL" else 1)

        for vec in morphed_test:
            pred = self._classifier.predict(vec.tolist())
            y_pred.append(0 if pred.prediction == "REAL" else 1)

        return self.evaluate(y_true, y_pred)

    def generate_report(self) -> EvaluationReport:
        """
        Produce a complete viva-ready evaluation report.

        Runs ``evaluate_on_synthetic()`` internally, then wraps the metrics
        with plain-English interpretation, model context, and recommendations.

        Returns:
            EvaluationReport ready to serve from the /evaluation endpoint.
        """
        metrics = self.evaluate_on_synthetic()
        info = self._classifier.get_model_info()

        model_info = {
            "type": "K-Means Clustering",
            "k": info.n_clusters,
            "features": f"LBP (59) + DCT (1024) = {_FUSED_VECTOR_SIZE} dimensions",
            "training_samples": info.training_samples,
            "training_data": (
                f"Synthetic ({info.training_samples // 2} REAL + "
                f"{info.training_samples // 2} MORPHED samples)"
                if info.trained_on_synthetic
                else "Real labelled face images"
            ),
            "cluster_labels": info.cluster_labels,
            "inertia": info.inertia,
        }

        interpretation = self._build_interpretation(metrics)
        recommendations = self._build_recommendations(metrics)

        return EvaluationReport(
            metrics=metrics,
            model_info=model_info,
            interpretation=interpretation,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data_source="synthetic",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label_to_cluster_id(self, label: str) -> int:
        for cid, lbl in self._classifier._cluster_labels.items():
            if lbl == label:
                return cid
        raise RuntimeError(
            f"Label '{label}' not found in cluster labels: {self._classifier._cluster_labels}"
        )

    @staticmethod
    def _build_interpretation(m: EvaluationResult) -> dict:
        far_1_in_n = f"1 in {int(round(1 / m.far))}" if m.far > 0 else "none"
        frr_1_in_n = f"1 in {int(round(1 / m.frr))}" if m.frr > 0 else "none"

        f1_quality = (
            "Excellent balance" if m.f1_score >= 0.90
            else "Good balance" if m.f1_score >= 0.80
            else "Moderate balance" if m.f1_score >= 0.70
            else "Poor balance — model needs retraining"
        )

        return {
            "accuracy": (
                f"{m.accuracy * 100:.1f}% of test images correctly classified "
                f"({m.correct_predictions}/{m.total_samples})"
            ),
            "far": (
                f"{m.far * 100:.1f}% of morphed faces incorrectly accepted as real "
                f"({far_1_in_n} morphed images slips through). "
                "In a border-control context this is the primary security metric."
            ),
            "frr": (
                f"{m.frr * 100:.1f}% of real faces incorrectly rejected as morphed "
                f"({frr_1_in_n} genuine users blocked). "
                "A high FRR harms user experience."
            ),
            "precision": (
                f"{m.precision * 100:.1f}% of images predicted as REAL are actually real. "
                "High precision = few false alarms."
            ),
            "recall": (
                f"{m.recall * 100:.1f}% of all real faces are correctly accepted. "
                "High recall = few real users blocked."
            ),
            "f1_score": (
                f"{f1_quality} between precision and recall "
                f"(F1 = {m.f1_score:.3f}). "
                "F1 is the harmonic mean — useful when classes are imbalanced."
            ),
            "confusion_matrix": (
                f"[[TN={m.tn}, FP={m.fp}], [FN={m.fn}, TP={m.tp}]] — "
                f"rows = actual label, columns = predicted label. "
                f"FP (top-right) drives FAR; FN (bottom-left) drives FRR."
            ),
        }

    @staticmethod
    def _build_recommendations(m: EvaluationResult) -> list[str]:
        recs: list[str] = []

        if m.far > 0.20:
            recs.append(
                "FAR is high (> 20%). Consider increasing Haar min_neighbors in face "
                "detection to only process high-confidence face crops, or collect more "
                "diverse morphed training examples."
            )
        if m.frr > 0.20:
            recs.append(
                "FRR is high (> 20%). The REAL cluster may be too tight. Consider "
                "increasing the std of synthetic REAL training vectors or adding "
                "data augmentation."
            )
        if m.accuracy < 0.70:
            recs.append(
                "Accuracy is below 70%. Replace synthetic training data with a real "
                "labelled face-morphing dataset (e.g., MorGAN or SMDD benchmarks) for "
                "significant performance gains."
            )
        if m.f1_score >= 0.80:
            recs.append(
                "F1 score is good. For further improvement, tune the DCT block size "
                "(currently 32×32) or experiment with additional feature descriptors "
                "such as HOG or SIFT."
            )
        if not recs:
            recs.append(
                "Metrics look healthy. For production deployment, validate on a real "
                "holdout dataset and monitor FAR/FRR in live traffic."
            )

        return recs
