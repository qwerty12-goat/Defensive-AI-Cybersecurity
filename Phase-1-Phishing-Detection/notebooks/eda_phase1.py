"""Exploratory data analysis for the cleaned Phase 1 phishing-email dataset.

This script reads the processed dataset, generates summary statistics and plots,
and saves reusable outputs in the results/ directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "phase1_dataset_v1.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_PATH}. "
            "Run src/prepare_dataset.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    df["char_length"] = df["body"].astype(str).str.len()
    df["word_count"] = df["body"].astype(str).str.split().str.len()

    print("\n=== LABEL BALANCE ===")
    print(df["label"].value_counts().sort_index())
    print((df["label"].value_counts(normalize=True).sort_index() * 100).round(2))

    print("\n=== CHARACTER LENGTH BY CLASS ===")
    print(df.groupby("label")["char_length"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    print("\n=== WORD COUNT BY CLASS ===")
    print(df.groupby("label")["word_count"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    print("\n=== EXTREME LENGTH CHECK ===")
    thresholds = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    for threshold in thresholds:
        count = (df["char_length"] > threshold).sum()
        print(f"Emails longer than {threshold:,} characters: {count:,}")

    longest = df.nlargest(20, "char_length")[
        ["label", "char_length", "word_count", "body"]
    ].copy()
    longest["body_preview"] = (
        longest["body"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.slice(0, 300)
    )
    longest.drop(columns=["body"]).to_csv(
        RESULTS_DIR / "longest_emails_preview.csv", index=False
    )

    shortest = df.nsmallest(20, "char_length")[
        ["label", "char_length", "word_count", "body"]
    ].copy()
    shortest["body_preview"] = (
        shortest["body"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.slice(0, 300)
    )
    shortest.drop(columns=["body"]).to_csv(
        RESULTS_DIR / "shortest_emails_preview.csv", index=False
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "total_rows",
                "safe_rows",
                "phishing_rows",
                "median_char_length",
                "p95_char_length",
                "p99_char_length",
                "max_char_length",
                "median_word_count",
                "p95_word_count",
                "p99_word_count",
                "max_word_count",
            ],
            "value": [
                len(df),
                (df["label"] == 0).sum(),
                (df["label"] == 1).sum(),
                df["char_length"].median(),
                df["char_length"].quantile(0.95),
                df["char_length"].quantile(0.99),
                df["char_length"].max(),
                df["word_count"].median(),
                df["word_count"].quantile(0.95),
                df["word_count"].quantile(0.99),
                df["word_count"].max(),
            ],
        }
    )
    summary.to_csv(RESULTS_DIR / "eda_summary.csv", index=False)

    plt.figure(figsize=(7, 4))
    df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Label (0 = Safe, 1 = Phishing-class)")
    plt.ylabel("Email Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close()

    clipped = df["char_length"].clip(upper=df["char_length"].quantile(0.99))
    plt.figure(figsize=(8, 4))
    plt.hist(clipped, bins=80)
    plt.title("Email Character-Length Distribution (Clipped at 99th Percentile)")
    plt.xlabel("Characters")
    plt.ylabel("Email Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "email_length_distribution.png", dpi=150)
    plt.close()

    for label, group in df.groupby("label"):
        clipped_group = group["char_length"].clip(
            upper=df["char_length"].quantile(0.99)
        )
        plt.figure(figsize=(8, 4))
        plt.hist(clipped_group, bins=80)
        plt.title(f"Email Length Distribution for Label {label}")
        plt.xlabel("Characters")
        plt.ylabel("Email Count")
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR / f"email_length_distribution_label_{label}.png",
            dpi=150,
        )
        plt.close()

    print("\nSaved EDA outputs to:")
    print(RESULTS_DIR)
    print("\nNo dataset files were modified.")


if __name__ == "__main__":
    main()
