"""Run a robustness experiment that reduces obvious dataset-source artifacts.

This experiment:
- uses train.csv and validation.csv only
- leaves test.csv untouched
- masks obvious source-like tokens such as years, phone-like numbers, and corpus-specific markers
- retrains the same TF-IDF + Logistic Regression baseline
- compares validation performance with the original baseline
- saves feature weights from the artifact-reduced model

Outputs:
    models/baseline_artifact_reduced.joblib
    results/robustness/
        robustness_metrics.csv
        performance_comparison.csv
        top_safe_features_artifact_reduced.csv
        top_phishing_features_artifact_reduced.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

BASELINE_METRICS_PATH = (
    PROJECT_ROOT / "results" / "baseline" / "validation_metrics.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "robustness"

MODEL_PATH = MODELS_DIR / "baseline_artifact_reduced.joblib"

RANDOM_SEED = 42
TOP_FEATURES = 50

SOURCE_MARKERS = {
    "enron",
    "spamassassin",
    "trec",
    "ceas",
    "nazario",
    "nigerian",
}

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
LONG_NUMBER_PATTERN = re.compile(r"\b\d{3,}\b")
WHITESPACE_PATTERN = re.compile(r"\s+")


def load_split(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} split not found at {path}.")

    df = pd.read_csv(path)

    required = {"body", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} split is missing required columns: {sorted(missing)}"
        )

    if df["body"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{name} split contains missing body/label values.")

    return df


def reduce_artifacts(text: str) -> str:
    text = text.lower()

    for marker in SOURCE_MARKERS:
        text = re.sub(
            rf"\b{re.escape(marker)}\b",
            " source_marker ",
            text,
        )

    text = YEAR_PATTERN.sub(" year_token ", text)
    text = LONG_NUMBER_PATTERN.sub(" number_token ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict:
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "model": "Artifact-reduced TF-IDF + Logistic Regression",
        "split": "validation",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main() -> None:
    print("Loading train and validation splits...")
    train_df = load_split(TRAIN_PATH, "Training")
    validation_df = load_split(VALIDATION_PATH, "Validation")

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print("Test split is not being read.")

    print("\nReducing obvious source-related artifacts...")
    train_text = train_df["body"].astype(str).map(reduce_artifacts)
    validation_text = validation_df["body"].astype(str).map(reduce_artifacts)

    y_train = train_df["label"].astype(int)
    y_validation = validation_df["label"].astype(int)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=200_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )

    print("Training artifact-reduced baseline...")
    pipeline.fit(train_text, y_train)

    print("Running validation predictions...")
    predictions = pipeline.predict(validation_text)

    metrics = calculate_metrics(y_validation, predictions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "robustness_metrics.csv",
        index=False,
    )

    comparison_rows = []

    if BASELINE_METRICS_PATH.exists():
        original = pd.read_csv(BASELINE_METRICS_PATH).iloc[0].to_dict()

        comparison_rows.append(
            {
                "model": "Original baseline",
                "accuracy": original.get("accuracy"),
                "precision": original.get("precision"),
                "recall": original.get("recall"),
                "f1": original.get("f1"),
                "false_positive_rate": original.get("false_positive_rate"),
                "false_negative_rate": original.get("false_negative_rate"),
            }
        )

    comparison_rows.append(
        {
            "model": metrics["model"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": metrics["false_positive_rate"],
            "false_negative_rate": metrics["false_negative_rate"],
        }
    )

    pd.DataFrame(comparison_rows).to_csv(
        RESULTS_DIR / "performance_comparison.csv",
        index=False,
    )

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]

    safe_indices = np.argsort(coefficients)[:TOP_FEATURES]
    phishing_indices = np.argsort(coefficients)[-TOP_FEATURES:][::-1]

    pd.DataFrame(
        {
            "feature": feature_names[safe_indices],
            "coefficient": coefficients[safe_indices],
        }
    ).to_csv(
        RESULTS_DIR / "top_safe_features_artifact_reduced.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "feature": feature_names[phishing_indices],
            "coefficient": coefficients[phishing_indices],
        }
    ).to_csv(
        RESULTS_DIR / "top_phishing_features_artifact_reduced.csv",
        index=False,
    )

    joblib.dump(pipeline, MODEL_PATH)

    print("\n=== ROBUSTNESS RESULTS ===")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"F1 score:            {metrics['f1']:.4f}")
    print(
        f"False positive rate: "
        f"{metrics['false_positive_rate']:.4f}"
    )
    print(
        f"False negative rate: "
        f"{metrics['false_negative_rate']:.4f}"
    )

    print("\nConfusion matrix:")
    print(f"True negatives:  {metrics['true_negatives']:,}")
    print(f"False positives: {metrics['false_positives']:,}")
    print(f"False negatives: {metrics['false_negatives']:,}")
    print(f"True positives:  {metrics['true_positives']:,}")

    print("\nTop phishing-associated features after masking:")
    for index in phishing_indices[:10]:
        print(
            f"  {feature_names[index]}: "
            f"{coefficients[index]:.4f}"
        )

    print("\nTop safe-associated features after masking:")
    for index in safe_indices[:10]:
        print(
            f"  {feature_names[index]}: "
            f"{coefficients[index]:.4f}"
        )

    if BASELINE_METRICS_PATH.exists():
        original = pd.read_csv(BASELINE_METRICS_PATH).iloc[0]
        print("\n=== CHANGE FROM ORIGINAL BASELINE ===")
        print(
            f"Accuracy change: "
            f"{metrics['accuracy'] - original['accuracy']:+.4f}"
        )
        print(
            f"F1 change:       "
            f"{metrics['f1'] - original['f1']:+.4f}"
        )
        print(
            f"FPR change:      "
            f"{metrics['false_positive_rate'] - original['false_positive_rate']:+.4f}"
        )
        print(
            f"FNR change:      "
            f"{metrics['false_negative_rate'] - original['false_negative_rate']:+.4f}"
        )

    print(f"\nSaved robustness results to: {RESULTS_DIR}")
    print(f"Saved artifact-reduced model to: {MODEL_PATH}")
    print("Test split was not read or modified.")


if __name__ == "__main__":
    main()
