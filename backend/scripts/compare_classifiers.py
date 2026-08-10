"""
Measurement-only comparison: does a supervised classifier beat K-Means on the
existing 1083-dim LBP+DCT feature vectors?

This script does NOT wire anything into the live /classify endpoint. It loads
the feature vectors already extracted by train_on_real_dataset.py, trains
five classifiers on an identical train/test split, and evaluates all five
with the existing ModelEvaluator so the metrics are directly comparable to
the K-Means numbers already reported (71.18% accuracy, FAR 50.25%, FRR
7.29%, F1 0.7624).

The KMeansClassifier instance used here is constructed with a temporary
models_dir so its save_model() calls cannot overwrite the production model
at backend/app/ml/models/kmeans_model.joblib.
"""

import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

from app.ml.clustering.kmeans_classifier import KMeansClassifier
from app.ml.evaluation.evaluator import ModelEvaluator

FEATURES_PATH = Path("/app/ml_data/checkpoints/features_sample.npz")
CHECKPOINT_DIR = Path("/app/ml_data/checkpoints")
PRODUCTION_MODELS_DIR = Path("/app/app/ml/models")
RANDOM_STATE = 42

LBP_RANGE = range(0, 59)   # indices 0-58
DCT_RANGE = range(59, 1083)  # indices 59-1082

KMEANS_BASELINE_FAR = 0.5025


def label_to_binary(labels: np.ndarray) -> np.ndarray:
    """0 = REAL, 1 = MORPHED — matches ModelEvaluator.evaluate() convention."""
    return np.array([0 if lbl == "REAL" else 1 for lbl in labels], dtype=int)


def print_production_model_state(when: str) -> None:
    meta_path = PRODUCTION_MODELS_DIR / "kmeans_meta.joblib"
    meta = joblib.load(meta_path)
    print(
        f"[{when}] kmeans_meta.joblib | training_samples={meta.get('training_samples')} "
        f"trained_on_synthetic={meta.get('trained_on_synthetic')}",
        flush=True,
    )


def run_kmeans(X_train, y_train_str, X_test) -> np.ndarray:
    """Train KMeansClassifier in an isolated temp dir; never touches production files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        clf = KMeansClassifier(models_dir=Path(tmp_dir))
        clf.train(X_train.tolist(), labels=y_train_str.tolist(), trained_on_synthetic=False)
        y_pred = []
        for vec in X_test:
            pred = clf.predict(vec.tolist())
            y_pred.append(0 if pred.prediction == "REAL" else 1)
    return np.array(y_pred, dtype=int)


def main() -> None:
    print("Loading features from", FEATURES_PATH, flush=True)
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    print(f"Loaded X shape={X.shape} y shape={y.shape}", flush=True)

    print_production_model_state("BEFORE")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(
        f"Split: train={len(X_train)} test={len(X_test)} "
        f"(stratified, test_size=0.2, random_state={RANDOM_STATE})",
        flush=True,
    )

    y_train_bin = label_to_binary(y_train)
    y_test_bin = label_to_binary(y_test)

    evaluator = ModelEvaluator(None)  # evaluate() is a pure function of y_true/y_pred

    results = {}
    rf_model = None

    # --- K-Means (isolated) -------------------------------------------------
    print("\nTraining KMeansClassifier (isolated temp dir)...", flush=True)
    y_pred = run_kmeans(X_train, y_train, X_test)
    results["KMeans"] = evaluator.evaluate(y_test_bin.tolist(), y_pred.tolist())

    # --- LogisticRegression ---------------------------------------------------
    print("Training LogisticRegression...", flush=True)
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train_bin)
    y_pred = logreg.predict(X_test)
    results["LogisticRegression"] = evaluator.evaluate(y_test_bin.tolist(), y_pred.tolist())
    joblib.dump(logreg, CHECKPOINT_DIR / "logreg.joblib")

    # --- SVC ------------------------------------------------------------------
    print("Training SVC (rbf)...", flush=True)
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train_bin)
    y_pred = svm.predict(X_test)
    results["SVM"] = evaluator.evaluate(y_test_bin.tolist(), y_pred.tolist())
    joblib.dump(svm, CHECKPOINT_DIR / "svm.joblib")

    # --- RandomForest -----------------------------------------------------------
    print("Training RandomForestClassifier...", flush=True)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train_bin)
    y_pred = rf_model.predict(X_test)
    results["RandomForest"] = evaluator.evaluate(y_test_bin.tolist(), y_pred.tolist())
    joblib.dump(rf_model, CHECKPOINT_DIR / "randomforest.joblib")

    # --- XGBoost ----------------------------------------------------------------
    print("Training XGBClassifier...", flush=True)
    xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
    xgb.fit(X_train, y_train_bin)
    y_pred = xgb.predict(X_test)
    results["XGBoost"] = evaluator.evaluate(y_test_bin.tolist(), y_pred.tolist())
    joblib.dump(xgb, CHECKPOINT_DIR / "xgboost.joblib")

    # --- Comparison table, sorted by FAR ascending -------------------------------
    rows = sorted(results.items(), key=lambda kv: kv[1].far)
    print("\n" + "=" * 100, flush=True)
    print("CLASSIFIER COMPARISON (sorted by FAR ascending — best first)", flush=True)
    print("=" * 100, flush=True)
    header = f"{'Classifier':<20}{'Accuracy':>10}{'FAR':>10}{'FRR':>10}{'Precision':>11}{'Recall':>10}{'F1':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for name, m in rows:
        print(
            f"{name:<20}{m.accuracy * 100:>9.2f}%{m.far * 100:>9.2f}%{m.frr * 100:>9.2f}%"
            f"{m.precision * 100:>10.2f}%{m.recall * 100:>9.2f}%{m.f1_score:>10.4f}",
            flush=True,
        )
    print("=" * 100, flush=True)

    best_name, best_metrics = rows[0]
    improvement = KMEANS_BASELINE_FAR - best_metrics.far
    print(
        f"\nWinner by FAR: {best_name} — FAR={best_metrics.far * 100:.2f}% "
        f"vs K-Means baseline FAR={KMEANS_BASELINE_FAR * 100:.2f}% "
        f"(improvement: {improvement * 100:.2f} percentage points)",
        flush=True,
    )

    # --- RandomForest feature importance breakdown -------------------------------
    importances = rf_model.feature_importances_
    top20_idx = np.argsort(importances)[::-1][:20]
    lbp_count = sum(1 for i in top20_idx if i in LBP_RANGE)
    dct_count = sum(1 for i in top20_idx if i in DCT_RANGE)

    print("\n" + "=" * 100, flush=True)
    print("RANDOM FOREST — TOP 20 FEATURE IMPORTANCES", flush=True)
    print("=" * 100, flush=True)
    for rank, idx in enumerate(top20_idx, start=1):
        feature_type = "LBP" if idx in LBP_RANGE else "DCT"
        print(f"{rank:>2}. index={idx:<6} type={feature_type:<4} importance={importances[idx]:.6f}", flush=True)
    print(f"\nLBP features in top 20 (range 0-58):   {lbp_count}", flush=True)
    print(f"DCT features in top 20 (range 59-1082): {dct_count}", flush=True)
    print("=" * 100, flush=True)

    # --- Verify production model untouched ---------------------------------------
    print_production_model_state("AFTER")

    # --- Save full results -----------------------------------------------------
    output = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": RANDOM_STATE,
        "kmeans_baseline_far": KMEANS_BASELINE_FAR,
        "results": {name: m.model_dump() for name, m in results.items()},
        "winner_by_far": best_name,
        "far_improvement_vs_kmeans": improvement,
        "random_forest_feature_importance": {
            "top_20_indices": [int(i) for i in top20_idx],
            "lbp_count": lbp_count,
            "dct_count": dct_count,
        },
    }
    results_path = CHECKPOINT_DIR / "classifier_comparison.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
