"""
Unit tests for Module 8 — ModelEvaluator.

Tests verify metric formulas with known inputs, boundary conditions
(perfect / worst classifier), synthetic evaluation, and report structure.
"""

import math
from pathlib import Path

import pytest

from app.ml.clustering.kmeans_classifier import KMeansClassifier
from app.ml.evaluation.evaluator import ModelEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmp_models_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("eval_models")


@pytest.fixture(scope="module")
def classifier(tmp_models_dir: Path) -> KMeansClassifier:
    return KMeansClassifier(models_dir=tmp_models_dir)


@pytest.fixture(scope="module")
def evaluator(classifier: KMeansClassifier) -> ModelEvaluator:
    return ModelEvaluator(classifier)


# Known test scenario for deterministic assertions
#
#   y_true = [0,0,0,1,1,1]  (3 REAL, 3 MORPHED)
#   y_pred = [0,0,1,0,1,1]
#
#   TP = actual REAL (0), pred REAL (0)  → indices 0,1         → TP = 2
#   TN = actual MORPHED (1), pred MORPHED (1) → indices 4,5    → TN = 2
#   FP = actual MORPHED (1), pred REAL (0) → index 3           → FP = 1
#   FN = actual REAL (0), pred MORPHED (1) → index 2           → FN = 1
#
#   accuracy  = 4/6 ≈ 0.6667
#   FAR       = 1/(1+2) ≈ 0.3333
#   FRR       = 1/(1+2) ≈ 0.3333
#   precision = 2/(2+1) ≈ 0.6667
#   recall    = 2/(2+1) ≈ 0.6667
#   f1        = 2*(2/3*2/3)/(2/3+2/3) = 2/3 ≈ 0.6667
#   confusion  = [[TN,FP],[FN,TP]] = [[2,1],[1,2]]

_Y_TRUE = [0, 0, 0, 1, 1, 1]
_Y_PRED = [0, 0, 1, 0, 1, 1]
_TOL = 1e-3


# ---------------------------------------------------------------------------
# Metric formula correctness
# ---------------------------------------------------------------------------


def test_accuracy_calculation_correct(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert abs(result.accuracy - 4 / 6) < _TOL


def test_far_calculation_correct(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert abs(result.far - 1 / 3) < _TOL


def test_frr_calculation_correct(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert abs(result.frr - 1 / 3) < _TOL


def test_precision_calculation_correct(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert abs(result.precision - 2 / 3) < _TOL


def test_recall_calculation_correct(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert abs(result.recall - 2 / 3) < _TOL


def test_f1_calculation_correct(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert abs(result.f1_score - 2 / 3) < _TOL


def test_confusion_matrix_correct(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert result.confusion_matrix == [[2, 1], [1, 2]]


def test_tp_tn_fp_fn_values(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert result.tp == 2
    assert result.tn == 2
    assert result.fp == 1
    assert result.fn == 1


def test_tp_plus_tn_plus_fp_plus_fn_equals_total(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert result.tp + result.tn + result.fp + result.fn == result.total_samples


def test_correct_predictions_equals_tp_plus_tn(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    assert result.correct_predictions == result.tp + result.tn


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def test_perfect_classifier_all_metrics_1(evaluator: ModelEvaluator) -> None:
    y = [0, 0, 0, 1, 1, 1]
    result = evaluator.evaluate(y, y)
    assert result.accuracy == 1.0
    assert result.far == 0.0
    assert result.frr == 0.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0
    assert result.fp == 0
    assert result.fn == 0


def test_worst_classifier_all_wrong(evaluator: ModelEvaluator) -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [1, 1, 1, 0, 0, 0]
    result = evaluator.evaluate(y_true, y_pred)
    assert result.accuracy == 0.0
    assert result.far == 1.0
    assert result.frr == 1.0
    assert result.tp == 0
    assert result.tn == 0


def test_all_real_predictions(evaluator: ModelEvaluator) -> None:
    """When everything is predicted REAL, FAR = 1.0 and FRR = 0."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]
    result = evaluator.evaluate(y_true, y_pred)
    assert result.far == 1.0
    assert result.frr == 0.0


def test_all_morphed_predictions(evaluator: ModelEvaluator) -> None:
    """When everything is predicted MORPHED, FAR = 0 and FRR = 1.0."""
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 1, 1]
    result = evaluator.evaluate(y_true, y_pred)
    assert result.far == 0.0
    assert result.frr == 1.0


def test_metrics_between_0_and_1(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate(_Y_TRUE, _Y_PRED)
    for metric in (result.accuracy, result.far, result.frr,
                   result.precision, result.recall, result.f1_score):
        assert 0.0 <= metric <= 1.0, f"Metric out of range: {metric}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raises_value_error(evaluator: ModelEvaluator) -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluator.evaluate([0, 1, 0], [1, 0])


def test_empty_inputs_raises_value_error(evaluator: ModelEvaluator) -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate([], [])


# ---------------------------------------------------------------------------
# Synthetic evaluation
# ---------------------------------------------------------------------------


def test_evaluate_on_synthetic_returns_valid_metrics(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate_on_synthetic()
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.far <= 1.0
    assert 0.0 <= result.frr <= 1.0
    assert 0.0 <= result.f1_score <= 1.0


def test_synthetic_total_samples_is_200(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate_on_synthetic()
    assert result.total_samples == 200


def test_synthetic_evaluation_counts_sum_correctly(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate_on_synthetic()
    assert result.tp + result.tn + result.fp + result.fn == 200


def test_synthetic_accuracy_is_reasonable(evaluator: ModelEvaluator) -> None:
    """With well-separated synthetic clusters, accuracy should be > 70%."""
    result = evaluator.evaluate_on_synthetic()
    assert result.accuracy > 0.70, f"Accuracy too low: {result.accuracy}"


def test_synthetic_evaluation_time_is_populated(evaluator: ModelEvaluator) -> None:
    result = evaluator.evaluate_on_synthetic()
    assert result.evaluation_time_ms >= 0.0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def test_generate_report_has_required_fields(evaluator: ModelEvaluator) -> None:
    report = evaluator.generate_report()
    assert hasattr(report, "metrics")
    assert hasattr(report, "model_info")
    assert hasattr(report, "interpretation")
    assert hasattr(report, "recommendations")
    assert hasattr(report, "timestamp")
    assert hasattr(report, "data_source")


def test_report_data_source_is_synthetic(evaluator: ModelEvaluator) -> None:
    report = evaluator.generate_report()
    assert report.data_source == "synthetic"


def test_report_model_info_has_type(evaluator: ModelEvaluator) -> None:
    report = evaluator.generate_report()
    assert report.model_info.get("type") == "K-Means Clustering"
    assert report.model_info.get("k") == 2


def test_report_interpretation_covers_all_metrics(evaluator: ModelEvaluator) -> None:
    report = evaluator.generate_report()
    for key in ("accuracy", "far", "frr", "precision", "recall", "f1_score"):
        assert key in report.interpretation, f"Missing interpretation for '{key}'"


def test_report_recommendations_is_non_empty_list(evaluator: ModelEvaluator) -> None:
    report = evaluator.generate_report()
    assert isinstance(report.recommendations, list)
    assert len(report.recommendations) > 0


def test_report_timestamp_is_iso_format(evaluator: ModelEvaluator) -> None:
    from datetime import datetime
    report = evaluator.generate_report()
    # Should not raise
    datetime.fromisoformat(report.timestamp.replace("Z", "+00:00"))
