"""Run stronger robustness checks for dataset artifacts.

This script:
- uses train.csv and validation.csv only
- never reads test.csv
- removes known corpus/source markers entirely
- masks years and long number sequences
- retrains the TF-IDF + Logistic Regression baseline
- trains a separate length-only Logistic Regression model
- compares both against the original baseline

Outputs:
    models/baseline_artifact_removed.joblib
    models/length_only_logreg.joblib
    results/robustness_stronger/
        text_model_metrics.csv
        length_only_metrics.csv
        performance_comparison.csv
        top_safe_features.csv
        top_phishing_features.csv
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
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

BASELINE_METRICS_PATH = (
    PROJECT_ROOT / "results" / "baseline" / "validation_metrics.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "robustness_stronger"

TEXT_MODEL_PATH = MODELS_DIR / "baseline_artifact_removed.joblib"
LENGTH_MODEL_PATH = MODELS_DIR / "length_only_logreg.joblib"

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


def remove_artifacts(text: str) -> str:
    text = text.lower()

    for marker in SOURCE_MARKERS:
        text = re.sub(rf"\b{re.escape(marker)}\b", " ", text)

    text = YEAR_PATTERN.sub(" year_token ", text)
    text = LONG_NUMBER_PATTERN.sub(" number_token ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def metrics_dict(name: str, y_true, predictions) -> dict:
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def build_length_features(series: pd.Series) -> np.ndarray:
    char_length = series.str.len().astype(float).to_numpy()
    word_count = series.str.split().str.len().astype(float).to_numpy()
    return np.column_stack([char_length, word_count])


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"F1 score:            {metrics['f1']:.4f}")
    print(f"False positive rate: {metrics['false_positive_rate']:.4f}")
    print(f"False negative rate: {metrics['false_negative_rate']:.4f}")
    print("Confusion matrix:")
    print(f"True negatives:  {metrics['true_negatives']:,}")
    print(f"False positives: {metrics['false_positives']:,}")
    print(f"False negatives: {metrics['false_negatives']:,}")
    print(f"True positives:  {metrics['true_positives']:,}")


def main() -> None:
    print("Loading train and validation splits...")
    train_df = load_split(TRAIN_PATH, "Training")
    validation_df = load_split(VALIDATION_PATH, "Validation")

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print("Test split is not being read.")

    raw_train_text = train_df["body"].astype(str)
    raw_validation_text = validation_df["body"].astype(str)

    y_train = train_df["label"].astype(int)
    y_validation = validation_df["label"].astype(int)

    print("\nRemoving known source markers and masking obvious metadata-like tokens...")
    train_text = raw_train_text.map(remove_artifacts)
    validation_text = raw_validation_text.map(remove_artifacts)

    text_pipeline = Pipeline(
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

    print("Training artifact-removed text model...")
    text_pipeline.fit(train_text, y_train)

    print("Running text-model validation predictions...")
    text_predictions = text_pipeline.predict(validation_text)
    text_metrics = metrics_dict(
        "Artifact-removed TF-IDF + Logistic Regression",
        y_validation,
        text_predictions,
    )

    print("\nTraining length-only model...")
    X_train_length = build_length_features(raw_train_text)
    X_validation_length = build_length_features(raw_validation_text)

    length_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    length_pipeline.fit(X_train_length, y_train)

    print("Running length-only validation predictions...")
    length_predictions = length_pipeline.predict(X_validation_length)
    length_metrics = metrics_dict(
        "Length-only Logistic Regression",
        y_validation,
        length_predictions,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([text_metrics]).to_csv(
        RESULTS_DIR / "text_model_metrics.csv",
        index=False,
    )

    pd.DataFrame([length_metrics]).to_csv(
        RESULTS_DIR / "length_only_metrics.csv",
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

    for result in [text_metrics, length_metrics]:
        comparison_rows.append(
            {
                "model": result["model"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "false_positive_rate": result["false_positive_rate"],
                "false_negative_rate": result["false_negative_rate"],
            }
        )

    pd.DataFrame(comparison_rows).to_csv(
        RESULTS_DIR / "performance_comparison.csv",
        index=False,
    )

    vectorizer = text_pipeline.named_steps["tfidf"]
    classifier = text_pipeline.named_steps["classifier"]

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
        RESULTS_DIR / "top_safe_features.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "feature": feature_names[phishing_indices],
            "coefficient": coefficients[phishing_indices],
        }
    ).to_csv(
        RESULTS_DIR / "top_phishing_features.csv",
        index=False,
    )

    joblib.dump(text_pipeline, TEXT_MODEL_PATH)
    joblib.dump(length_pipeline, LENGTH_MODEL_PATH)

    print_metrics("ARTIFACT-REMOVED TEXT MODEL", text_metrics)
    print_metrics("LENGTH-ONLY MODEL", length_metrics)

    print("\nTop phishing-associated features:")
    for index in phishing_indices[:10]:
        print(
            f"  {feature_names[index]}: "
            f"{coefficients[index]:.4f}"
        )

    print("\nTop safe-associated features:")
    for index in safe_indices[:10]:
        print(
            f"  {feature_names[index]}: "
            f"{coefficients[index]:.4f}"
        )

    if BASELINE_METRICS_PATH.exists():
        original = pd.read_csv(BASELINE_METRICS_PATH).iloc[0]
        print("\n=== CHANGE FROM ORIGINAL BASELINE ===")
        print(
            "Artifact-removed accuracy change: "
            f"{text_metrics['accuracy'] - original['accuracy']:+.4f}"
        )
        print(
            "Artifact-removed F1 change:       "
            f"{text_metrics['f1'] - original['f1']:+.4f}"
        )
        print(
            "Length-only accuracy difference: "
            f"{length_metrics['accuracy'] - original['accuracy']:+.4f}"
        )

    print(f"\nSaved results to: {RESULTS_DIR}")
    print(f"Saved text model to: {TEXT_MODEL_PATH}")
    print(f"Saved length-only model to: {LENGTH_MODEL_PATH}")
    print("Test split was not read or modified.")


if __name__ == "__main__":
    main()
