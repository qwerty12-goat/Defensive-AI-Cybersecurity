"""Inspect the raw phishing-email dataset without modifying it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Merged_Dataset.csv"


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Run src/download_dataset.py first."
        )

    print(f"Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    print("\n=== BASIC INFO ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)

    print("\n=== LABEL CHECK ===")
    if "label" in df.columns:
        print("Label counts:")
        print(df["label"].value_counts(dropna=False).sort_index())
        print("\nLabel percentages:")
        print((df["label"].value_counts(normalize=True, dropna=False) * 100).round(2))
        print(f"Unexpected label values: {sorted(set(df['label'].dropna()) - {0, 1})}")
    else:
        print("WARNING: 'label' column not found.")

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\n=== EMPTY EMAIL BODIES ===")
    if "body" in df.columns:
        body_as_text = df["body"].fillna("").astype(str)
        empty_body_count = body_as_text.str.strip().eq("").sum()
        print(f"Empty or whitespace-only bodies: {empty_body_count:,}")
    else:
        print("WARNING: 'body' column not found.")

    print("\n=== DUPLICATES ===")
    exact_row_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_row_duplicates:,}")

    if "body" in df.columns:
        duplicate_bodies = df["body"].duplicated(keep=False).sum()
        unique_duplicate_bodies = df["body"].duplicated().sum()
        print(f"Rows involved in duplicate-body groups: {duplicate_bodies:,}")
        print(f"Duplicate body rows beyond first occurrence: {unique_duplicate_bodies:,}")

    print("\n=== EMAIL LENGTH STATISTICS ===")
    if "body" in df.columns:
        lengths = df["body"].fillna("").astype(str).str.len()
        print(lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
        print(f"Shortest body length: {lengths.min():,} characters")
        print(f"Longest body length: {lengths.max():,} characters")

    print("\n=== MEMORY USAGE ===")
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Approximate in-memory size: {memory_mb:.2f} MB")

    print("\nInspection complete. No files were modified.")


if __name__ == "__main__":
    main()
