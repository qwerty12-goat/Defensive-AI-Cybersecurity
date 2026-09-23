"""Colab-ready transformer training for phishing-email classification.

This script is intended for a GPU environment such as Google Colab.

It:
- reads train.csv and validation.csv only
- leaves test.csv untouched
- fine-tunes DistilBERT for binary classification
- computes accuracy, precision, recall, F1, FPR, and FNR
- saves validation metrics and the fine-tuned model
- copies final artifacts to persistent Google Drive storage when available

Expected training runtime: longer than 1 minute, commonly tens of minutes on a Colab GPU.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "transformer"
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_phishing"

DRIVE_ROOT = Path("/content/drive/MyDrive/Defensive-AI-Cybersecurity")
DRIVE_ARTIFACT_DIR = DRIVE_ROOT / "transformer_artifacts"
DRIVE_MODEL_DIR = DRIVE_ARTIFACT_DIR / "distilbert_phishing"
DRIVE_RESULTS_DIR = DRIVE_ARTIFACT_DIR / "results"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
SEED = 42


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

    return df[["body", "label"]].copy()


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def save_to_drive() -> None:
    if not Path("/content/drive/MyDrive").exists():
        print("Google Drive is not mounted. Persistent copy skipped.")
        return

    DRIVE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if DRIVE_MODEL_DIR.exists():
        shutil.rmtree(DRIVE_MODEL_DIR)
    shutil.copytree(MODEL_DIR, DRIVE_MODEL_DIR)

    DRIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        RESULTS_DIR / "validation_metrics.csv",
        DRIVE_RESULTS_DIR / "validation_metrics.csv",
    )

    print(f"Persistent model copy saved to: {DRIVE_MODEL_DIR}")
    print(f"Persistent metrics copy saved to: {DRIVE_RESULTS_DIR}")


def main() -> None:
    print("Loading train and validation splits...")
    train_df = load_split(TRAIN_PATH, "Training")
    validation_df = load_split(VALIDATION_PATH, "Validation")

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print("Test split is not being read.")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(
        train_df.rename(columns={"body": "text"}),
        preserve_index=False,
    )
    validation_dataset = Dataset.from_pandas(
        validation_df.rename(columns={"body": "text"}),
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Tokenizing training data...")
    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    print("Tokenizing validation data...")
    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    train_dataset = train_dataset.rename_column("label", "labels")
    validation_dataset = validation_dataset.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "transformer_checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=250,
        seed=SEED,
        fp16=True,
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting transformer fine-tuning...")
    trainer.train()

    print("Evaluating on validation split...")
    evaluation = trainer.evaluate()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_false_positive_rate",
        "eval_false_negative_rate",
    ]

    clean_metrics = {
        key.removeprefix("eval_"): evaluation[key]
        for key in metric_keys
        if key in evaluation
    }

    pd.DataFrame([clean_metrics]).to_csv(
        RESULTS_DIR / "validation_metrics.csv",
        index=False,
    )

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print("\n=== TRANSFORMER VALIDATION RESULTS ===")
    for key, value in clean_metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"\nSaved model to: {MODEL_DIR}")
    print(f"Saved validation metrics to: {RESULTS_DIR}")
    save_to_drive()
    print("Test split was not read or modified.")


if __name__ == "__main__":
    main()
