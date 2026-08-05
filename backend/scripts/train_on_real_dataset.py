"""
Train the K-Means classifier (Module 7) on real Kaggle SSMD face images,
replacing the synthetic bootstrap data used at first startup.

Usage (inside the backend container, dataset mounted read-only at
/app/ml_data/dataset):

    python scripts/train_on_real_dataset.py --sample-per-class 2000
    python scripts/train_on_real_dataset.py --full

Every image is pushed through the full Modules 1-6 pipeline (validate ->
detect -> preprocess -> LBP -> DCT -> fuse) to build a 1083-dim feature
vector, exactly as the /classify endpoint does at inference time. Any image
that fails a pipeline step is logged and skipped -- a handful of bad files
must never abort a multi-thousand-image run.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# `app` lives at /app/app inside the container; make it importable
# regardless of the working directory this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split

from app.ml.clustering.kmeans_classifier import KMeansClassifier
from app.ml.detectors.face_detector import FaceDetector
from app.ml.evaluation.evaluator import ModelEvaluator
from app.ml.exceptions import ImageProcessingError, InvalidImageError
from app.ml.feature_extractors.feature_fusion import FeatureFusion
from app.ml.preprocessors.image_preprocessor import ImagePreprocessor
from app.ml.validators.image_validator import ImageValidator

DEFAULT_DATASET_ROOT = Path("/app/ml_data/dataset")
DEFAULT_CHECKPOINT_DIR = Path("/app/ml_data/checkpoints")
RANDOM_STATE = 42

_validator = ImageValidator()
_detector = FaceDetector()
_preprocessor = ImagePreprocessor()
_feature_fusion = FeatureFusion()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train K-Means on a sample (or all) of the real Kaggle SSMD dataset."
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=2000,
        help="Number of images to sample from EACH class (default: 2000). Ignored with --full.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use every image in both folders instead of sampling.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Root directory containing real/ and morphed/ (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory to write features_sample.npz and the evaluation report (default: {DEFAULT_CHECKPOINT_DIR}).",
    )
    return parser.parse_args()


def select_files(folder: Path, sample_size: int, full: bool, label: str) -> list[Path]:
    """List all PNGs in `folder`, then sample (seed=42) unless --full is set."""
    all_files = sorted(folder.glob("*.png"))
    available = len(all_files)

    if full:
        print(f"[{label}] available={available} | using ALL (--full)")
        return all_files

    n = min(sample_size, available)
    if n < sample_size:
        print(
            f"[{label}] WARNING: requested {sample_size} but only {available} available — using {n}."
        )
    random.seed(RANDOM_STATE)
    sampled = random.sample(all_files, n)
    print(f"[{label}] available={available} | sampled={n}")
    return sampled


def process_image(path: Path, label: str) -> tuple[list[float] | None, str | None]:
    """
    Run one image through Modules 1-6 (validate -> detect -> preprocess ->
    LBP -> DCT -> fuse) and return (fused_vector, failure_reason).

    Exactly one of the two return values is non-None.
    """
    try:
        file_bytes = path.read_bytes()
    except Exception as exc:
        return None, f"decode_error: could not read file ({exc})"

    try:
        validation = _validator.validate(file_bytes, path.name)
    except InvalidImageError as exc:
        return None, f"validation_failed: {exc}"
    if not validation.is_valid:
        return None, f"validation_failed: {'; '.join(validation.errors)}"

    try:
        detection = _detector.detect(file_bytes, strict=False)
    except ImageProcessingError as exc:
        return None, f"decode_error: {exc}"

    if detection.face_count == 0 or detection.largest_face is None:
        return None, "no_face_detected"

    if detection.face_count > 1:
        print(f"    note: {path.name} — {detection.face_count} faces detected, using largest")

    largest_area = detection.largest_face.area
    largest_index = 0
    for i, box in enumerate(detection.faces):
        if box.area == largest_area:
            largest_index = i
            break

    try:
        import base64

        crop_bytes = base64.b64decode(detection.cropped_faces_b64[largest_index])
        normalized_array = _preprocessor.get_normalized_array(crop_bytes)
        fusion = _feature_fusion.extract_full_pipeline(normalized_array)
    except ImageProcessingError as exc:
        return None, f"decode_error: {exc}"
    except Exception as exc:  # noqa: BLE001 - never let one bad image kill the run
        return None, f"other: {exc}"

    return fusion.fused_vector, None


def categorize(reason: str) -> str:
    if reason.startswith("no_face_detected"):
        return "no_face_detected"
    if reason.startswith("validation_failed"):
        return "validation_failed"
    if reason.startswith("decode_error"):
        return "decode_error"
    return "other"


def run_pipeline(
    files: list[tuple[Path, str]],
) -> tuple[list[list[float]], list[str], list[tuple[str, str]]]:
    """Process every (path, label) pair, printing progress every 100 images."""
    total = len(files)
    X: list[list[float]] = []
    y: list[str] = []
    failed: list[tuple[str, str]] = []

    success_count = 0
    fail_count = 0
    t_start = time.perf_counter()

    for i, (path, label) in enumerate(files, start=1):
        vector, reason = process_image(path, label)
        if vector is not None:
            X.append(vector)
            y.append(label)
            success_count += 1
        else:
            failed.append((path.name, reason))
            fail_count += 1

        if i % 100 == 0 or i == total:
            elapsed = time.perf_counter() - t_start
            avg_per_image = elapsed / i
            eta = avg_per_image * (total - i)
            print(
                f"Processed {i}/{total} | success={success_count} fail={fail_count} "
                f"| elapsed={elapsed:.1f}s | ETA={eta:.1f}s"
            )

    return X, y, failed


def print_summary(
    real_sampled: int,
    morphed_sampled: int,
    y: list[str],
    failed: list[tuple[str, str]],
    files: list[tuple[Path, str]],
) -> None:
    real_success = sum(1 for lbl in y if lbl == "REAL")
    morphed_success = sum(1 for lbl in y if lbl == "MORPHED")

    failed_by_file = dict(failed)
    real_failed = sum(
        1 for path, label in files if label == "REAL" and path.name in failed_by_file
    )
    morphed_failed = sum(
        1 for path, label in files if label == "MORPHED" and path.name in failed_by_file
    )

    reason_counts = Counter(categorize(reason) for _, reason in failed)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Class':<10}{'Sampled':>10}{'Success':>10}{'Failed':>10}")
    print(f"{'REAL':<10}{real_sampled:>10}{real_success:>10}{real_failed:>10}")
    print(f"{'MORPHED':<10}{morphed_sampled:>10}{morphed_success:>10}{morphed_failed:>10}")
    print(
        f"{'TOTAL':<10}{real_sampled + morphed_sampled:>10}"
        f"{real_success + morphed_success:>10}{real_failed + morphed_failed:>10}"
    )
    print("\nFailure reason breakdown:")
    if reason_counts:
        for reason, count in reason_counts.most_common():
            print(f"  {reason:<20}: {count}")
    else:
        print("  (no failures)")

    total_success = real_success + morphed_success
    print("\nFinal class balance:")
    if total_success > 0:
        print(f"  REAL:    {real_success} ({real_success / total_success * 100:.1f}%)")
        print(f"  MORPHED: {morphed_success} ({morphed_success / total_success * 100:.1f}%)")
    print("=" * 60 + "\n")


def main() -> None:
    script_start = time.perf_counter()
    args = parse_args()

    real_dir = args.dataset_root / "real"
    morphed_dir = args.dataset_root / "morphed"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    real_files = select_files(real_dir, args.sample_per_class, args.full, "REAL")
    morphed_files = select_files(morphed_dir, args.sample_per_class, args.full, "MORPHED")

    files: list[tuple[Path, str]] = [(p, "REAL") for p in real_files] + [
        (p, "MORPHED") for p in morphed_files
    ]

    print(f"\nProcessing {len(files)} images through Modules 1-6 pipeline...\n")
    X_list, y_list, failed = run_pipeline(files)
    print_summary(len(real_files), len(morphed_files), y_list, failed, files)

    if len(X_list) < 4:
        print("ERROR: too few successfully processed images to train/evaluate. Aborting.")
        sys.exit(1)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=str)

    sample_size_meta = {
        "requested_per_class": args.sample_per_class if not args.full else None,
        "full": args.full,
        "real_sampled": len(real_files),
        "morphed_sampled": len(morphed_files),
        "real_success": int(np.sum(y == "REAL")),
        "morphed_success": int(np.sum(y == "MORPHED")),
    }

    checkpoint_path = args.checkpoint_dir / "features_sample.npz"
    np.savez(
        checkpoint_path,
        X=X,
        y=y,
        failed_files=np.array(failed, dtype=object),
        sample_size=np.array(sample_size_meta, dtype=object),
    )
    print(f"Checkpoint saved -> {checkpoint_path}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(
        f"\nTrain/test split: train={len(X_train)} test={len(X_test)} "
        f"(stratified, test_size=0.2, random_state={RANDOM_STATE})"
    )

    # Train K-Means on the real feature vectors — overwrites the saved model.
    classifier = KMeansClassifier()
    training_result = classifier.train(
        X_train.tolist(), labels=y_train.tolist(), trained_on_synthetic=False
    )
    print(
        f"\nTraining complete | samples={training_result.samples_trained} "
        f"inertia={training_result.inertia:.2f} iterations={training_result.iterations} "
        f"converged={training_result.converged} cluster_labels={training_result.cluster_labels} "
        f"train_accuracy={training_result.accuracy}"
    )

    # Evaluate on the held-out REAL test set.
    evaluator = ModelEvaluator(classifier)
    y_true = [0 if lbl == "REAL" else 1 for lbl in y_test]
    y_pred: list[int] = []
    for vec in X_test:
        pred = classifier.predict(vec.tolist())
        y_pred.append(0 if pred.prediction == "REAL" else 1)

    eval_result = evaluator.evaluate(y_true, y_pred)

    print("\n" + "=" * 60)
    print("REAL DATASET EVALUATION REPORT")
    print("=" * 60)
    print(f"Accuracy:  {eval_result.accuracy * 100:.2f}%")
    print(f"FAR:       {eval_result.far * 100:.2f}%  (morphed accepted as real)")
    print(f"FRR:       {eval_result.frr * 100:.2f}%  (real rejected as morphed)")
    print(f"Precision: {eval_result.precision * 100:.2f}%")
    print(f"Recall:    {eval_result.recall * 100:.2f}%")
    print(f"F1 Score:  {eval_result.f1_score:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {eval_result.confusion_matrix}")
    print(f"Total test samples: {eval_result.total_samples} | Correct: {eval_result.correct_predictions}")
    print("=" * 60 + "\n")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size_meta,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": eval_result.model_dump(),
        "data_source": "real_kaggle_ssmd_set_sample",
    }
    report_path = args.checkpoint_dir / "real_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Evaluation report saved -> {report_path}")

    total_elapsed = time.perf_counter() - script_start
    print(f"\nTotal script runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
