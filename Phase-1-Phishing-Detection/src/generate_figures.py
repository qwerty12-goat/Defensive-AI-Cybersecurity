"""Generate publication-quality figures from the committed aggregate results.

Usage:
    python src/generate_figures.py

Inputs:
    results/results_summary.csv

Outputs:
    results/figures/*.png
    results/figures/*.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INPUT = RESULTS / "results_summary.csv"
OUTPUT = RESULTS / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
BASELINE = "TF-IDF + Logistic Regression"
BERT = "DistilBERT"


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def value(section, model, group, metric):
    row = df[
        (df["section"] == section)
        & (df["model"] == model)
        & (df["split_or_group"] == group)
        & (df["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError(
            f"Expected one row for {section=} {model=} {group=} {metric=}; "
            f"found {len(row)}"
        )
    return float(row.iloc[0]["value"])


# Figure 1: historical validation vs final-test performance
metrics = ["accuracy", "precision", "recall", "f1"]
labels = ["Accuracy", "Precision", "Recall", "F1"]
series = [
    ("Baseline validation", BASELINE, "validation"),
    ("Baseline final test", BASELINE, "final_test"),
    ("DistilBERT validation", BERT, "validation"),
    ("DistilBERT final test", BERT, "final_test"),
]

x = np.arange(len(metrics))
width = 0.19
fig, ax = plt.subplots(figsize=(9, 5.2))
for i, (name, model, split) in enumerate(series):
    vals = [100 * value("historical", model, split, metric) for metric in metrics]
    ax.bar(x + (i - 1.5) * width, vals, width, label=name)
ax.set_ylabel("Performance (%)")
ax.set_title("Historical Validation and Final-Test Performance")
ax.set_xticks(x, labels)
ax.set_ylim(96, 100)
ax.legend(frameon=False, ncols=2)
ax.grid(axis="y", alpha=0.25)
save_figure(fig, "figure_1_historical_validation_vs_test")


# Figure 2: historical final-test recall vs synthetic detection
models = [BASELINE, BERT]
model_labels = ["TF-IDF + Logistic Regression", "DistilBERT"]
historical = [100 * value("historical", m, "final_test", "recall") for m in models]
synthetic = [100 * value("synthetic", m, "overall", "detection_rate") for m in models]

x = np.arange(2)
width = 0.34
fig, ax = plt.subplots(figsize=(8, 5.2))
b1 = ax.bar(x - width / 2, historical, width, label="Historical final-test recall")
b2 = ax.bar(x + width / 2, synthetic, width, label="Synthetic detection rate")
ax.bar_label(b1, fmt="%.1f%%", padding=3)
ax.bar_label(b2, fmt="%.1f%%", padding=3)
ax.set_ylabel("Detection / Recall (%)")
ax.set_title("Historical Generalization vs Controlled Synthetic Distribution")
ax.set_xticks(x, model_labels)
ax.set_ylim(0, 110)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
save_figure(fig, "figure_2_historical_vs_synthetic")


# Figure 3: synthetic detection by communication category
categories = [
    ("account_security", "Account / security"),
    ("delivery_service", "Delivery / service"),
    ("general_social_engineering", "General social engineering"),
    ("promotional_offer", "Promotional / offer"),
    ("workplace_business", "Workplace / business"),
]

y = np.arange(len(categories))
width = 0.35
baseline_vals = [
    100 * value("synthetic", BASELINE, key, "detection_rate") for key, _ in categories
]
bert_vals = [
    100 * value("synthetic", BERT, key, "detection_rate") for key, _ in categories
]

fig, ax = plt.subplots(figsize=(9, 5.6))
b1 = ax.barh(y - width / 2, baseline_vals, width, label="TF-IDF + Logistic Regression")
b2 = ax.barh(y + width / 2, bert_vals, width, label="DistilBERT")
ax.bar_label(b1, fmt="%.0f%%", padding=3)
ax.bar_label(b2, fmt="%.0f%%", padding=3)
ax.set_xlabel("Detection rate (%)")
ax.set_title("Synthetic Detection Rate by Communication Style")
ax.set_yticks(y, [label for _, label in categories])
ax.set_xlim(0, 110)
ax.invert_yaxis()
ax.legend(frameon=False)
ax.grid(axis="x", alpha=0.25)
save_figure(fig, "figure_3_synthetic_detection_by_category")


# Figure 4: paired synthetic prediction overlap
overlap_metrics = [
    ("both_detected", "Both detected"),
    ("baseline_only_detected", "Baseline only"),
    ("distilbert_only_detected", "DistilBERT only"),
    ("both_missed", "Both missed"),
]
overlap_models = {
    "both_detected": "Both models",
    "baseline_only_detected": BASELINE,
    "distilbert_only_detected": BERT,
    "both_missed": "Both models",
}
counts = [
    value("synthetic_overlap", overlap_models[m], "overall", m)
    for m, _ in overlap_metrics
]
labels_overlap = [label for _, label in overlap_metrics]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
bars = ax.bar(labels_overlap, counts)
ax.bar_label(bars, fmt="%.0f", padding=3)
ax.set_ylabel("Messages (n = 500)")
ax.set_title("Paired Model Outcomes on Controlled Synthetic Messages")
ax.set_ylim(0, max(counts) * 1.18)
ax.grid(axis="y", alpha=0.25)
save_figure(fig, "figure_4_synthetic_prediction_overlap")


# Figure 5: confidence among synthetic misses
rates = [
    100 * value("synthetic_confidence", BASELINE, "misses", "high_confidence_miss_rate"),
    100 * value("synthetic_confidence", BERT, "misses", "high_confidence_miss_rate"),
]
medians = [
    100 * value("synthetic_confidence", BASELINE, "misses", "median_miss_confidence"),
    100 * value("synthetic_confidence", BERT, "misses", "median_miss_confidence"),
]

x = np.arange(2)
width = 0.34
fig, ax = plt.subplots(figsize=(8, 5.2))
b1 = ax.bar(x - width / 2, rates, width, label="Misses with confidence ≥ 90%")
b2 = ax.bar(x + width / 2, medians, width, label="Median confidence among misses")
ax.bar_label(b1, fmt="%.1f%%", padding=3)
ax.bar_label(b2, fmt="%.1f%%", padding=3)
ax.set_ylabel("Percent (%)")
ax.set_title("Confidence of Incorrect Legitimate Predictions on Synthetic Messages")
ax.set_xticks(x, ["TF-IDF + Logistic Regression", "DistilBERT"])
ax.set_ylim(0, 110)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.text(
    0.5,
    -0.18,
    "Confidence values are raw model outputs and are not calibrated probabilities.",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
)
save_figure(fig, "figure_5_synthetic_miss_confidence")

print(f"Generated figures in: {OUTPUT}")
for path in sorted(OUTPUT.iterdir()):
    print(path.name)
