"""Clean the raw phishing-email dataset and save a processed copy.

This script never modifies the original raw dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Merged_Dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "phase1_dataset_v1.csv"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. "
            "Run src/download_dataset.py first."
        )

    print(f"Loading raw dataset: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    print(f"Initial rows: {len(df):,}")

    required_columns = {"body", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    # Keep only the columns used in this phase.
    df = df[["body", "label"]].copy()

    # Remove rows with missing labels or missing message bodies.
    before = len(df)
    df = df.dropna(subset=["body", "label"])
    print(f"Removed rows with missing body/label: {before - len(df):,}")

    # Normalize body type and remove empty/whitespace-only messages.
    df["body"] = df["body"].astype(str)
    before = len(df)
    df = df[df["body"].str.strip().ne("")]
    print(f"Removed empty/whitespace-only bodies: {before - len(df):,}")

    # Keep only expected binary labels.
    before = len(df)
    df = df[df["label"].isin([0, 1])]
    print(f"Removed rows with unexpected labels: {before - len(df):,}")

    # Convert labels to integer after validation.
    df["label"] = df["label"].astype(int)

    # Remove duplicate email bodies so the same text cannot leak across splits.
    before = len(df)
    df = df.drop_duplicates(subset=["body"], keep="first")
    print(f"Removed duplicate email bodies: {before - len(df):,}")

    # Reset row numbering for a clean processed dataset.
    df = df.reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== CLEANED DATASET SUMMARY ===")
    print(f"Final rows: {len(df):,}")
    print("Label counts:")
    print(df["label"].value_counts().sort_index())
    print("\nLabel percentages:")
    print((df["label"].value_counts(normalize=True).sort_index() * 100).round(2))

    lengths = df["body"].str.len()
    print("\nEmail length statistics:")
    print(lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")
    print("Raw dataset was not modified.")


if __name__ == "__main__":
    main()
