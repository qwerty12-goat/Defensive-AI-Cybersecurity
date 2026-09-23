"""Interpret the baseline phishing-email classifier and analyze validation errors.

Inputs:
    data/processed/validation.csv
    models/baseline_tfidf_logreg.joblib

Outputs:
    results/interpretation/
        top_safe_features.csv
        top_phishing_features.csv
        validation_error_summary.csv
        false_positives_sample.csv
        false_negatives_sample.csv
        confidence_distribution.csv

The test split is intentionally not read.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_tfidf_logreg.joblib"
RESULTS_DIR = PROJECT_ROOT / "results" / "interpretation"

TOP_FEATURES = 50
ERROR_SAMPLE_SIZE = 100


def load_validation() -> pd.DataFrame:
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation split not found at {VALIDATION_PATH}."
        )

    df = pd.read_csv(VALIDATION_PATH)

    required = {"body", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Validation data is missing required columns: {sorted(missing)}"
        )

    if df["body"].isna().any() or df["label"].isna().any():
        raise ValueError("Validation split contains missing body/label values.")

    return df


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained baseline model not found at {MODEL_PATH}. "
            "Run src/train_baseline.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trained baseline model...")
    pipeline = joblib.load(MODEL_PATH)

    print("Loading validation split only...")
    df = load_validation()

    X = df["body"].astype(str)
    y = df["label"].astype(int).to_numpy()

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
            "direction": "safe",
        }
    ).to_csv(RESULTS_DIR / "top_safe_features.csv", index=False)

    pd.DataFrame(
        {
            "feature": feature_names[phishing_indices],
            "coefficient": coefficients[phishing_indices],
            "direction": "phishing-class",
        }
    ).to_csv(RESULTS_DIR / "top_phishing_features.csv", index=False)

    print("Generating validation predictions and probabilities...")
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    analysis_df = df[["body", "label"]].copy()
    analysis_df["prediction"] = predictions
    analysis_df["phishing_probability"] = probabilities
    analysis_df["char_length"] = analysis_df["body"].str.len()
    analysis_df["word_count"] = analysis_df["body"].str.split().str.len()

    false_positives = analysis_df[
        (analysis_df["label"] == 0) & (analysis_df["prediction"] == 1)
    ].copy()

    false_negatives = analysis_df[
        (analysis_df["label"] == 1) & (analysis_df["prediction"] == 0)
    ].copy()

    false_positives["error_type"] = "false_positive"
    false_negatives["error_type"] = "false_negative"

    error_summary = pd.DataFrame(
        [
            {
                "error_type": "false_positive",
                "count": len(false_positives),
                "median_char_length": float(false_positives["char_length"].median()),
                "median_word_count": float(false_positives["word_count"].median()),
                "mean_phishing_probability": float(
                    false_positives["phishing_probability"].mean()
                ),
            },
            {
                "error_type": "false_negative",
                "count": len(false_negatives),
                "median_char_length": float(false_negatives["char_length"].median()),
                "median_word_count": float(false_negatives["word_count"].median()),
                "mean_phishing_probability": float(
                    false_negatives["phishing_probability"].mean()
                ),
            },
        ]
    )
    error_summary.to_csv(
        RESULTS_DIR / "validation_error_summary.csv",
        index=False,
    )

    false_positives.sort_values(
        "phishing_probability", ascending=False
    ).head(ERROR_SAMPLE_SIZE).to_csv(
        RESULTS_DIR / "false_positives_sample.csv",
        index=False,
    )

    false_negatives.sort_values(
        "phishing_probability", ascending=True
    ).head(ERROR_SAMPLE_SIZE).to_csv(
        RESULTS_DIR / "false_negatives_sample.csv",
        index=False,
    )

    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    labels = [
        "0.00-0.10",
        "0.10-0.25",
        "0.25-0.50",
        "0.50-0.75",
        "0.75-0.90",
        "0.90-1.00",
    ]

    confidence_df = pd.DataFrame(
        {
            "label": y,
            "prediction": predictions,
            "phishing_probability": probabilities,
        }
    )
    confidence_df["probability_bin"] = pd.cut(
        confidence_df["phishing_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    (
        confidence_df.groupby(
            ["probability_bin", "label", "prediction"],
            observed=False,
        )
        .size()
        .reset_index(name="count")
        .to_csv(RESULTS_DIR / "confidence_distribution.csv", index=False)
    )

    print("\n=== MODEL INTERPRETATION SUMMARY ===")
    print("Top phishing-associated features:")
    for feature, coef in zip(
        feature_names[phishing_indices[:10]],
        coefficients[phishing_indices[:10]],
    ):
        print(f"  {feature}: {coef:.4f}")

    print("\nTop safe-associated features:")
    for feature, coef in zip(
        feature_names[safe_indices[:10]],
        coefficients[safe_indices[:10]],
    ):
        print(f"  {feature}: {coef:.4f}")

    print("\n=== VALIDATION ERROR SUMMARY ===")
    print(f"False positives: {len(false_positives):,}")
    print(f"False negatives: {len(false_negatives):,}")

    if len(false_positives):
        print(
            "False-positive median length: "
            f"{false_positives['char_length'].median():,.0f} characters / "
            f"{false_positives['word_count'].median():,.0f} words"
        )

    if len(false_negatives):
        print(
            "False-negative median length: "
            f"{false_negatives['char_length'].median():,.0f} characters / "
            f"{false_negatives['word_count'].median():,.0f} words"
        )

    print(f"\nSaved interpretation outputs to: {RESULTS_DIR}")
    print("Test split was not read or modified.")


if __name__ == "__main__":
    main()
