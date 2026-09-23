"""Create reproducible stratified train/validation/test splits.

Input:
    data/processed/phase1_dataset_v1.csv

Outputs:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv

Split:
    70% train
    15% validation
    15% test

The split is stratified by label and uses a fixed random seed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "phase1_dataset_v1.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VALIDATION_PATH = PROCESSED_DIR / "validation.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

RANDOM_SEED = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name.upper()} SPLIT ===")
    print(f"Rows: {len(df):,}")

    counts = df["label"].value_counts().sort_index()
    percentages = (
        df["label"].value_counts(normalize=True).sort_index() * 100
    ).round(2)

    print("Label counts:")
    print(counts)

    print("Label percentages:")
    print(percentages)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {INPUT_PATH}. "
            "Run src/prepare_dataset.py first."
        )

    print(f"Loading processed dataset: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    required_columns = {"body", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if df["label"].isna().any():
        raise ValueError("Missing labels found in processed dataset.")

    unexpected_labels = sorted(set(df["label"].unique()) - {0, 1})
    if unexpected_labels:
        raise ValueError(
            f"Unexpected label values found: {unexpected_labels}"
        )

    duplicate_bodies = df["body"].duplicated().sum()
    if duplicate_bodies:
        raise ValueError(
            f"Found {duplicate_bodies:,} duplicate email bodies. "
            "Re-run dataset cleaning before creating splits."
        )

    print(f"Total rows: {len(df):,}")
    print(
        f"Target split: {TRAIN_SIZE:.0%} train / "
        f"{VALIDATION_SIZE:.0%} validation / "
        f"{TEST_SIZE:.0%} test"
    )
    print(f"Random seed: {RANDOM_SEED}")

    train_df, temp_df = train_test_split(
        df,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    validation_fraction_of_temp = VALIDATION_SIZE / (
        VALIDATION_SIZE + TEST_SIZE
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1 - validation_fraction_of_temp,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    expected_total = len(train_df) + len(validation_df) + len(test_df)
    if expected_total != len(df):
        raise RuntimeError("Split row counts do not add up to original dataset.")

    # Defensive leakage check.
    train_bodies = set(train_df["body"])
    validation_bodies = set(validation_df["body"])
    test_bodies = set(test_df["body"])

    if train_bodies & validation_bodies:
        raise RuntimeError("Data leakage detected between train and validation.")
    if train_bodies & test_bodies:
        raise RuntimeError("Data leakage detected between train and test.")
    if validation_bodies & test_bodies:
        raise RuntimeError("Data leakage detected between validation and test.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    validation_df.to_csv(VALIDATION_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print_split_summary("train", train_df)
    print_split_summary("validation", validation_df)
    print_split_summary("test", test_df)

    print("\n=== LEAKAGE CHECK ===")
    print("No duplicate email bodies appear across the three splits.")

    print("\nSaved:")
    print(TRAIN_PATH)
    print(VALIDATION_PATH)
    print(TEST_PATH)

    print(
        "\nImportant: keep the test split isolated from model selection "
        "and hyperparameter tuning."
    )


if __name__ == "__main__":
    main()
