"""Formal exploratory data analysis using the training split only.

This script intentionally does not read validation.csv or test.csv.

Input:
    data/processed/train.csv

Outputs:
    results/eda/
        summary_overall.csv
        summary_by_class.csv
        length_threshold_counts.csv
        class_distribution.png
        character_length_distribution.png
        word_count_distribution.png
        character_length_by_class.png
        word_count_by_class.png
        body_length_correlation.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "eda"


def save_histogram(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.hist(series, bins=80)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Email Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training split not found at {TRAIN_PATH}. "
            "Run src/create_splits.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training split only: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH)

    required_columns = {"body", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Training data is missing required columns: {sorted(missing_columns)}"
        )

    df["body"] = df["body"].astype(str)
    df["char_length"] = df["body"].str.len()
    df["word_count"] = df["body"].str.split().str.len()

    print(f"Training rows: {len(df):,}")

    # ---------------------------
    # Overall summary
    # ---------------------------
    overall_summary = pd.DataFrame(
        [
            {
                "rows": len(df),
                "safe_rows": int((df["label"] == 0).sum()),
                "phishing_rows": int((df["label"] == 1).sum()),
                "safe_percent": round((df["label"] == 0).mean() * 100, 2),
                "phishing_percent": round((df["label"] == 1).mean() * 100, 2),
                "median_char_length": float(df["char_length"].median()),
                "p90_char_length": float(df["char_length"].quantile(0.90)),
                "p95_char_length": float(df["char_length"].quantile(0.95)),
                "p99_char_length": float(df["char_length"].quantile(0.99)),
                "max_char_length": int(df["char_length"].max()),
                "median_word_count": float(df["word_count"].median()),
                "p90_word_count": float(df["word_count"].quantile(0.90)),
                "p95_word_count": float(df["word_count"].quantile(0.95)),
                "p99_word_count": float(df["word_count"].quantile(0.99)),
                "max_word_count": int(df["word_count"].max()),
            }
        ]
    )
    overall_summary.to_csv(RESULTS_DIR / "summary_overall.csv", index=False)

    # ---------------------------
    # Per-class summary
    # ---------------------------
    class_rows = []

    for label, group in df.groupby("label"):
        class_rows.append(
            {
                "label": int(label),
                "class_name": "Safe" if label == 0 else "Phishing-class",
                "rows": len(group),
                "percent_of_train": round(len(group) / len(df) * 100, 2),
                "mean_char_length": round(group["char_length"].mean(), 2),
                "median_char_length": float(group["char_length"].median()),
                "p90_char_length": float(group["char_length"].quantile(0.90)),
                "p95_char_length": float(group["char_length"].quantile(0.95)),
                "p99_char_length": float(group["char_length"].quantile(0.99)),
                "max_char_length": int(group["char_length"].max()),
                "mean_word_count": round(group["word_count"].mean(), 2),
                "median_word_count": float(group["word_count"].median()),
                "p90_word_count": float(group["word_count"].quantile(0.90)),
                "p95_word_count": float(group["word_count"].quantile(0.95)),
                "p99_word_count": float(group["word_count"].quantile(0.99)),
                "max_word_count": int(group["word_count"].max()),
            }
        )

    summary_by_class = pd.DataFrame(class_rows)
    summary_by_class.to_csv(RESULTS_DIR / "summary_by_class.csv", index=False)

    # ---------------------------
    # Length threshold counts
    # ---------------------------
    thresholds = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    threshold_rows = []

    for threshold in thresholds:
        total_count = int((df["char_length"] > threshold).sum())
        safe_count = int(
            ((df["label"] == 0) & (df["char_length"] > threshold)).sum()
        )
        phishing_count = int(
            ((df["label"] == 1) & (df["char_length"] > threshold)).sum()
        )

        threshold_rows.append(
            {
                "threshold_characters": threshold,
                "total_count": total_count,
                "total_percent": round(total_count / len(df) * 100, 4),
                "safe_count": safe_count,
                "phishing_count": phishing_count,
            }
        )

    pd.DataFrame(threshold_rows).to_csv(
        RESULTS_DIR / "length_threshold_counts.csv", index=False
    )

    # ---------------------------
    # Simple correlation check
    # ---------------------------
    correlations = df[["label", "char_length", "word_count"]].corr()
    correlations.to_csv(RESULTS_DIR / "body_length_correlation.csv")

    # ---------------------------
    # Plots
    # ---------------------------
    plt.figure(figsize=(6, 4.5))
    df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Training Set Class Distribution")
    plt.xlabel("Label (0 = Safe, 1 = Phishing-class)")
    plt.ylabel("Email Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=180)
    plt.close()

    char_cap = df["char_length"].quantile(0.99)
    word_cap = df["word_count"].quantile(0.99)

    save_histogram(
        df["char_length"].clip(upper=char_cap),
        "Training Email Character-Length Distribution\n(Clipped at 99th Percentile)",
        "Characters",
        RESULTS_DIR / "character_length_distribution.png",
    )

    save_histogram(
        df["word_count"].clip(upper=word_cap),
        "Training Email Word-Count Distribution\n(Clipped at 99th Percentile)",
        "Words",
        RESULTS_DIR / "word_count_distribution.png",
    )

    plt.figure(figsize=(8, 4.5))
    for label, group in df.groupby("label"):
        clipped = group["char_length"].clip(upper=char_cap)
        plt.hist(
            clipped,
            bins=80,
            alpha=0.5,
            label="Safe" if label == 0 else "Phishing-class",
        )

    plt.title("Character-Length Distribution by Class\n(Clipped at 99th Percentile)")
    plt.xlabel("Characters")
    plt.ylabel("Email Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "character_length_by_class.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for label, group in df.groupby("label"):
        clipped = group["word_count"].clip(upper=word_cap)
        plt.hist(
            clipped,
            bins=80,
            alpha=0.5,
            label="Safe" if label == 0 else "Phishing-class",
        )

    plt.title("Word-Count Distribution by Class\n(Clipped at 99th Percentile)")
    plt.xlabel("Words")
    plt.ylabel("Email Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "word_count_by_class.png", dpi=180)
    plt.close()

    # ---------------------------
    # Console research summary
    # ---------------------------
    safe = df[df["label"] == 0]
    phishing = df[df["label"] == 1]

    print("\n=== FORMAL EDA SUMMARY ===")
    print(
        f"Class balance: {(df['label'] == 0).mean() * 100:.2f}% safe / "
        f"{(df['label'] == 1).mean() * 100:.2f}% phishing-class"
    )
    print(
        f"Median character length: safe={safe['char_length'].median():,.0f}, "
        f"phishing-class={phishing['char_length'].median():,.0f}"
    )
    print(
        f"Median word count: safe={safe['word_count'].median():,.0f}, "
        f"phishing-class={phishing['word_count'].median():,.0f}"
    )
    print(
        f"99th percentile character length: "
        f"{df['char_length'].quantile(0.99):,.0f}"
    )
    print(
        "Potential dataset artifact to monitor: email length differs noticeably "
        "between the two labels, so later model interpretation should check "
        "whether classification depends heavily on source/format differences."
    )

    print(f"\nSaved EDA outputs to: {RESULTS_DIR}")
    print("Validation and test sets were not read or modified.")


if __name__ == "__main__":
    main()
