"""Train the baseline phishing-email classifier.

Model:
    TF-IDF vectorizer + Logistic Regression

Inputs:
    data/processed/train.csv
    data/processed/validation.csv

Outputs:
    models/baseline_tfidf_logreg.joblib
    results/baseline/validation_metrics.csv
    results/baseline/confusion_matrix.csv
    results/baseline/confusion_matrix.png

Important:
    test.csv is intentionally not read in this step.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"

MODEL_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"

RANDOM_SEED = 42


def load_split(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} split not found at {path}. Run src/create_splits.py first."
        )

    df = pd.read_csv(path)

    required_columns = {"body", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{name} split is missing required columns: {sorted(missing_columns)}"
        )

    if df["body"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{name} split contains missing body/label values.")

    return df


def main() -> None:
    print("Loading train and validation splits...")
    train_df = load_split(TRAIN_PATH, "Training")
    validation_df = load_split(VALIDATION_PATH, "Validation")

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print("Test split is not being read.")

    X_train = train_df["body"].astype(str)
    y_train = train_df["label"].astype(int)

    X_validation = validation_df["body"].astype(str)
    y_validation = validation_df["label"].astype(int)

    print("\nBuilding baseline pipeline...")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
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

    print("Training TF-IDF + Logistic Regression baseline...")
    pipeline.fit(X_train, y_train)

    print("Running validation predictions...")
    validation_predictions = pipeline.predict(X_validation)

    accuracy = accuracy_score(y_validation, validation_predictions)
    precision = precision_score(
        y_validation, validation_predictions, zero_division=0
    )
    recall = recall_score(
        y_validation, validation_predictions, zero_division=0
    )
    f1 = f1_score(
        y_validation, validation_predictions, zero_division=0
    )

    cm = confusion_matrix(y_validation, validation_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "TF-IDF + Logistic Regression",
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
        ]
    )

    metrics_df.to_csv(
        RESULTS_DIR / "validation_metrics.csv",
        index=False,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual Safe", "Actual Phishing-class"],
        columns=["Predicted Safe", "Predicted Phishing-class"],
    )
    confusion_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")

    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm)
    plt.title("Baseline Validation Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["Safe", "Phishing-class"])
    plt.yticks([0, 1], ["Safe", "Phishing-class"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=180,
    )
    plt.close()

    joblib.dump(pipeline, MODEL_PATH)

    print("\n=== BASELINE VALIDATION RESULTS ===")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 score:            {f1:.4f}")
    print(f"False positive rate: {false_positive_rate:.4f}")
    print(f"False negative rate: {false_negative_rate:.4f}")

    print("\nConfusion matrix:")
    print(f"True negatives:  {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives:  {tp:,}")

    print(f"\nSaved trained model to: {MODEL_PATH}")
    print(f"Saved validation results to: {RESULTS_DIR}")
    print("Test split was not read or modified.")


if __name__ == "__main__":
    main()
