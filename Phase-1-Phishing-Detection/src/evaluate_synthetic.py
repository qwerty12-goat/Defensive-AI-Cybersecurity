"""Evaluate frozen synthetic phishing-class messages with trained baseline and DistilBERT models.

This script is for defensive robustness evaluation only. It does not generate, send,
or modify phishing messages. The synthetic dataset must be frozen before evaluation.

Example:
    python src/evaluate_synthetic.py \
        --data data/synthetic/synthetic_ai_phishing_evaluation_v1.csv \
        --baseline-model models/baseline_tfidf_logreg.joblib \
        --transformer-model models/distilbert_phishing \
        --output-dir results/synthetic_evaluation
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REQUIRED_COLUMNS = {
    "sample_id",
    "category",
    "text",
    "label",
    "generation_source",
    "generation_date",
}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - margin, center + margin


def validate_dataset(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique.")
    if df["text"].isna().any():
        raise ValueError("Synthetic evaluation text contains missing values.")
    if not (df["label"] == 1).all():
        raise ValueError("This evaluation expects a positive-only phishing-class dataset.")


def load_baseline(path: Path):
    model = joblib.load(path)
    # Compatibility for older serialized LogisticRegression objects.
    classifier = model.named_steps.get("classifier")
    if classifier is not None and not hasattr(classifier, "multi_class"):
        classifier.multi_class = "auto"
    return model


def predict_baseline(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    model = load_baseline(model_path)
    out = df.copy()
    out["baseline_phishing_probability"] = model.predict_proba(out["text"])[:, 1]
    out["baseline_prediction"] = (
        out["baseline_phishing_probability"] >= 0.5
    ).astype(int)
    return out


def predict_transformer(
    df: pd.DataFrame,
    model_path: Path,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    probabilities: list[float] = []
    texts = df["text"].astype(str).tolist()

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
        probabilities.extend(probs.cpu().numpy().tolist())

    out = df.copy()
    out["distilbert_phishing_probability"] = probabilities
    out["distilbert_prediction"] = (
        out["distilbert_phishing_probability"] >= 0.5
    ).astype(int)
    return out


def grouped_detection(
    df: pd.DataFrame,
    prediction_column: str,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    group_specs = [
        ("overall", "all", df),
        *[("category", str(name), group) for name, group in df.groupby("category")],
        *[
            ("generator", str(name), group)
            for name, group in df.groupby("generation_source")
        ],
    ]

    for group_type, group_name, group in group_specs:
        total = len(group)
        detected = int((group[prediction_column] == 1).sum())
        low, high = wilson_interval(detected, total)
        rows.append(
            {
                "model": model_name,
                "group_type": group_type,
                "group": group_name,
                "detected": detected,
                "missed": total - detected,
                "total": total,
                "detection_rate": detected / total,
                "ci_95_lower": low,
                "ci_95_upper": high,
            }
        )
    return pd.DataFrame(rows)


def exact_mcnemar(b: int, c: int) -> tuple[int, float, float]:
    discordant = b + c
    if discordant == 0:
        return 0, 0.0, 1.0
    smaller = min(b, c)
    lower_tail = sum(
        math.comb(discordant, k) * (0.5**discordant)
        for k in range(smaller + 1)
    )
    exact_p = min(1.0, 2 * lower_tail)
    chi_square = (abs(b - c) - 1) ** 2 / discordant
    return discordant, chi_square, exact_p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--baseline-model", required=True, type=Path)
    parser.add_argument("--transformer-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    validate_dataset(df)

    baseline = predict_baseline(df, args.baseline_model)
    transformer = predict_transformer(
        df, args.transformer_model, args.max_length, args.batch_size
    )

    comparison = baseline.merge(
        transformer[
            ["sample_id", "distilbert_prediction", "distilbert_phishing_probability"]
        ],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(df):
        raise ValueError("Model prediction files did not match every frozen sample.")

    comparison["model_result"] = np.select(
        [
            (comparison["baseline_prediction"] == 1)
            & (comparison["distilbert_prediction"] == 1),
            (comparison["baseline_prediction"] == 0)
            & (comparison["distilbert_prediction"] == 0),
            (comparison["baseline_prediction"] == 1)
            & (comparison["distilbert_prediction"] == 0),
        ],
        ["both_detected", "both_missed", "baseline_only"],
        default="distilbert_only",
    )

    baseline_results = grouped_detection(comparison, "baseline_prediction", "Baseline")
    transformer_results = grouped_detection(
        comparison, "distilbert_prediction", "DistilBERT"
    )
    detection_results = pd.concat(
        [baseline_results, transformer_results], ignore_index=True
    )

    overlap = comparison["model_result"].value_counts()
    b = int(overlap.get("baseline_only", 0))
    c = int(overlap.get("distilbert_only", 0))
    discordant, chi_square, exact_p = exact_mcnemar(b, c)

    comparison.to_csv(args.output_dir / "synthetic_model_comparison.csv", index=False)
    detection_results.to_csv(
        args.output_dir / "synthetic_detection_results.csv", index=False
    )

    mcnemar = {
        "samples": int(len(comparison)),
        "both_detected": int(overlap.get("both_detected", 0)),
        "both_missed": int(overlap.get("both_missed", 0)),
        "baseline_only": b,
        "distilbert_only": c,
        "discordant_pairs": discordant,
        "continuity_corrected_chi_square": chi_square,
        "exact_two_sided_p_value": exact_p,
    }
    with open(args.output_dir / "synthetic_mcnemar_test.json", "w", encoding="utf-8") as f:
        json.dump(mcnemar, f, indent=2)

    print("\nSynthetic evaluation complete.")
    for model_name, pred_col in [
        ("Baseline", "baseline_prediction"),
        ("DistilBERT", "distilbert_prediction"),
    ]:
        detected = int((comparison[pred_col] == 1).sum())
        total = len(comparison)
        print(f"{model_name}: {detected}/{total} detected ({detected / total:.2%})")

    print(
        f"McNemar: baseline-only={b}, DistilBERT-only={c}, "
        f"exact two-sided p={exact_p:.6g}"
    )
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
