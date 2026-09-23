"""Validation error analysis for the fine-tuned DistilBERT phishing classifier.

This script:
- reads validation.csv only
- loads the trained DistilBERT model
- does not read test.csv
- generates class probabilities and predictions
- saves false positives and false negatives
- saves a compact summary for later comparison with the baseline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_phishing"
DRIVE_MODEL_DIR = Path(
    "/content/drive/MyDrive/Defensive-AI-Cybersecurity/"
    "transformer_artifacts/distilbert_phishing"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "transformer_error_analysis"

MAX_LENGTH = 256
BATCH_SIZE = 32


def resolve_model_dir() -> Path:
    if LOCAL_MODEL_DIR.exists():
        return LOCAL_MODEL_DIR
    if DRIVE_MODEL_DIR.exists():
        return DRIVE_MODEL_DIR
    raise FileNotFoundError(
        "Could not find the trained DistilBERT model locally or in Google Drive."
    )


def main() -> None:
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation split not found at {VALIDATION_PATH}"
        )

    print("Loading validation split...")
    df = pd.read_csv(VALIDATION_PATH)

    required = {"body", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Validation split is missing required columns: {sorted(missing)}"
        )

    df = df[["body", "label"]].copy()

    model_dir = resolve_model_dir()
    print(f"Loading trained model from: {model_dir}")
    print(f"Validation rows: {len(df):,}")
    print("Test split is not being read.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    dataset = Dataset.from_pandas(
        df.rename(columns={"body": "text"}),
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Tokenizing validation data...")
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )
    dataset = dataset.rename_column("label", "labels")

    args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "tmp"),
        per_device_eval_batch_size=BATCH_SIZE,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Generating validation predictions...")
    output = trainer.predict(dataset)

    logits = output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    predictions = np.argmax(probabilities, axis=1)
    phishing_probability = probabilities[:, 1]

    analysis = df.copy()
    analysis["prediction"] = predictions
    analysis["phishing_probability"] = phishing_probability
    analysis["correct"] = analysis["prediction"] == analysis["label"]
    analysis["char_length"] = analysis["body"].astype(str).str.len()
    analysis["word_count"] = (
        analysis["body"].astype(str).str.split().str.len()
    )

    false_positives = analysis[
        (analysis["label"] == 0) & (analysis["prediction"] == 1)
    ].copy()
    false_negatives = analysis[
        (analysis["label"] == 1) & (analysis["prediction"] == 0)
    ].copy()

    correct = analysis[analysis["correct"]].copy()
    errors = analysis[~analysis["correct"]].copy()

    summary = pd.DataFrame(
        [
            {
                "validation_rows": len(analysis),
                "correct_predictions": len(correct),
                "total_errors": len(errors),
                "false_positives": len(false_positives),
                "false_negatives": len(false_negatives),
                "error_rate": len(errors) / len(analysis),
                "fp_median_chars": false_positives["char_length"].median(),
                "fp_median_words": false_positives["word_count"].median(),
                "fn_median_chars": false_negatives["char_length"].median(),
                "fn_median_words": false_negatives["word_count"].median(),
                "error_median_confidence": (
                    np.maximum(
                        errors["phishing_probability"],
                        1 - errors["phishing_probability"],
                    ).median()
                    if len(errors)
                    else np.nan
                ),
            }
        ]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    analysis.to_csv(
        RESULTS_DIR / "validation_predictions.csv",
        index=False,
    )
    false_positives.to_csv(
        RESULTS_DIR / "false_positives.csv",
        index=False,
    )
    false_negatives.to_csv(
        RESULTS_DIR / "false_negatives.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS_DIR / "summary.csv",
        index=False,
    )

    print("\n=== TRANSFORMER ERROR ANALYSIS ===")
    print(f"Validation rows: {len(analysis):,}")
    print(f"Correct predictions: {len(correct):,}")
    print(f"Total errors: {len(errors):,}")
    print(f"False positives: {len(false_positives):,}")
    print(f"False negatives: {len(false_negatives):,}")

    if len(false_positives):
        print(
            "FP median length: "
            f"{false_positives['char_length'].median():.0f} chars / "
            f"{false_positives['word_count'].median():.0f} words"
        )

    if len(false_negatives):
        print(
            "FN median length: "
            f"{false_negatives['char_length'].median():.0f} chars / "
            f"{false_negatives['word_count'].median():.0f} words"
        )

    print(f"\nSaved outputs to: {RESULTS_DIR}")
    print("Test split was not read or modified.")


if __name__ == "__main__":
    main()
