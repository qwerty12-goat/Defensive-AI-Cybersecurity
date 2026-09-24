# Defensive AI Cybersecurity

A multi-layered defensive AI research project investigating whether machine-learning systems can detect AI-assisted cybersecurity threats at both the human communication layer and the network layer.

## Research Goal

The overall research question is:

**Can a layered machine-learning defense detect AI-assisted cyberattacks at both the human communication layer and network layer while maintaining an acceptably low false-positive rate?**

The project is organized around two complementary defensive layers:

- **Human Layer:** detect phishing-class email content before a user interacts with it.
- **Network Layer:** detect suspicious or automated malicious network behavior using machine-learning methods.

The Human Layer is the current research component and is in final repository polish. The Network Layer is planned as the next major component.

---

## Human Layer: Phishing Detection Under Controlled Synthetic Distribution Shift

### Research Question

**How accurately can machine-learning and Natural Language Processing models distinguish phishing-class from legitimate historical emails while maintaining a low false-positive rate, and how robust are those frozen classifiers to a controlled synthetic phishing-class distribution?**

A secondary question asks how detection performance on controlled synthetic phishing-class messages compares with performance on held-out historical phishing-class emails.

### Models

Two classifiers were developed and compared:

1. **TF-IDF + Logistic Regression**
2. **DistilBERT**

The baseline provides an interpretable traditional NLP comparison, while DistilBERT tests whether a transformer-based language model improves classification performance.

---

## Dataset

The primary historical dataset is based on the **Phishing-Email-Detection-Dataset**, which merges email corpora including Enron, SpamAssassin, TREC, CEAS, Nazario, Nigerian scam collections, and other sources.

After reproducible cleaning and exact-body deduplication:

- **208,161 emails**
- **108,953 legitimate**
- **99,208 phishing-class**
- approximately **52.34% legitimate / 47.66% phishing-class**

The processed data was split with a fixed random seed into:

- **Training:** 145,712
- **Validation:** 31,224
- **Test:** 31,225

No exact duplicate email bodies were allowed across the splits.

The test set remained isolated throughout model development and was consumed exactly once after all model-development decisions were frozen. No post-test tuning, retraining, threshold adjustment, calibration fitting, feature selection, or model selection was permitted.

---

## Traditional Validation Results

| Model | Accuracy | Precision | Recall | F1 | False-Positive Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 98.24% | 97.96% | 98.37% | 98.16% | 1.87% |
| DistilBERT | 99.21% | 99.25% | 99.09% | 99.17% | 0.68% |

On the historical validation distribution, DistilBERT improved both phishing recall and false-positive performance.

The DistilBERT validation confusion matrix was:

- True negatives: 16,232
- False positives: 111
- False negatives: 135
- True positives: 14,746

Compared with the baseline, DistilBERT reduced total validation errors from **548 to 246**.

---

## Final Held-Out Historical Test Results

After model development was complete, both frozen classifiers were evaluated once on the previously untouched **31,225-email** historical test set.

| Model | Accuracy | Precision | Recall | F1 | False-Positive Rate | False-Negative Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 98.25% | 97.86% | 98.47% | 98.17% | 1.96% | 1.53% |
| DistilBERT | 99.23% | 99.23% | 99.16% | 99.19% | 0.70% | 0.84% |

The final confusion-matrix counts were:

| Model | TN | FP | FN | TP | Total Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 16,023 | 320 | 227 | 14,655 | 547 |
| DistilBERT | 16,228 | 115 | 125 | 14,757 | 240 |

The close validation-to-test agreement provides evidence of strong same-distribution generalization for both frozen models. This makes the later decline on the controlled synthetic distribution distinct from ordinary failure to generalize to unseen historical examples.

The test set is now permanently treated as consumed. Future model improvements require a new independent holdout rather than further tuning against these test results.

---

## Robustness Analysis

Strong validation accuracy does not automatically imply robustness.

Several checks were performed to investigate whether performance could be explained by simple dataset artifacts:

- removal of obvious source and year markers;
- a length-only classifier;
- model error overlap;
- high-confidence error analysis;
- token-length and context-window analysis.

Removing obvious source artifacts barely changed baseline performance.

A length-only classifier reached only about **59.9% accuracy**, indicating that message length alone cannot explain the approximately 98% baseline validation accuracy.

Increasing DistilBERT inference length from 256 to 512 tokens changed only **45 of 31,224 predictions**, suggesting that truncation was not the main source of validation errors.

---

## Controlled Synthetic Evaluation

A separate frozen evaluation set of **500 controlled synthetic phishing-class messages** was created before either model was evaluated on it.

The set contains five communication styles:

- account/security;
- workplace/business;
- delivery/service;
- promotional/offer;
- general social engineering.

Two generation sources contributed equally:

- Copilot: 250 messages
- Gemini: 250 messages

The messages were intentionally non-operational and generated under defensive safety constraints. No real credential collection, malware delivery, payment collection, real targets, or deployable malicious links were used.

Because all 500 messages belong to the positive phishing class, the primary synthetic metric is **detection rate / recall**.

### Synthetic Detection Results

| Model | Detected | Detection Rate | 95% Wilson Interval |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 277 / 500 | 55.4% | 51.0%-59.7% |
| DistilBERT | 181 / 500 | 36.2% | 32.1%-40.5% |

This created an important reversal:

- on traditional validation, DistilBERT performed better;
- on the controlled synthetic distribution, the simpler baseline detected more phishing-class messages.

### Detection by Style

| Category | Baseline | DistilBERT |
| --- | ---: | ---: |
| Account/security | 99% | 82% |
| Delivery/service | 55% | 28% |
| General social engineering | 20% | 2% |
| Promotional/offer | 84% | 59% |
| Workplace/business | 19% | 10% |

Routine workplace/business and general social-engineering language was especially difficult for both models.

### Model Overlap

Across the 500-message synthetic set:

- both models detected: **168**
- both models missed: **210**
- baseline only detected: **109**
- DistilBERT only detected: **13**

An exact paired McNemar test produced **p < 0.001** within this evaluation set.

This is evidence of a meaningful paired performance difference on the controlled dataset, but it does **not** establish that logistic regression is universally more robust than DistilBERT.

---

## Confidence and Distribution Shift

The DistilBERT synthetic errors were frequently high-confidence failures.

Among its **319 missed synthetic messages**:

- **290** were predicted legitimate with at least 90% confidence;
- **233** were predicted legitimate with at least 99% confidence;
- median confidence on missed messages was **99.92%**.

This result is important because it shows that low uncertainty did not reliably identify out-of-distribution failure.

Synthetic messages were short overall, with a median length of approximately **193 characters / 26 words**, so simple message length or the 256-token input limit does not explain the synthetic performance drop.

The strongest pattern was communication style. Security/account vocabulary was associated with higher detection, while routine organizational language such as project, schedule, meeting, workspace, team, and document was associated with misses.

These are descriptive associations, not causal word-level explanations.

---

## Defensive Demo

An interactive Streamlit prototype converts the DistilBERT classifier into a usable defensive research demo.

The user can paste email text and receive:

- a legitimate vs. potential-phishing classification;
- phishing probability;
- legitimate probability;
- model information;
- a warning that high model confidence does not guarantee correctness.

Demo code:

`Phase-1-Phishing-Detection/demo/phishing_detector_app.py`

Demo documentation:

`Phase-1-Phishing-Detection/demo/README.md`

The trained model is hosted on Hugging Face:

https://huggingface.co/ozuvadh/defensive-ai-phishing-distilbert

The demo loads the model directly from Hugging Face rather than relying on a researcher-specific local path.

---

## Reproducibility

The repository includes scripts for:

- dataset inspection and cleaning;
- train/validation/test splitting;
- formal exploratory data analysis;
- TF-IDF + Logistic Regression training;
- baseline interpretation;
- artifact and length robustness checks;
- DistilBERT training;
- transformer error analysis;
- controlled synthetic evaluation;
- paired model comparison;
- defensive demo inference.

Important research artifacts and detailed synthetic-evaluation outputs are documented under:

`Phase-1-Phishing-Detection/documentation/`

Core synthetic evaluation can be reproduced with:

`Phase-1-Phishing-Detection/src/evaluate_synthetic.py`

---

## Repository Structure

```text
Defensive-AI-Cybersecurity/
├── README.md
└── Phase-1-Phishing-Detection/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── demo/
    │   ├── phishing_detector_app.py
    │   ├── requirements.txt
    │   └── README.md
    ├── documentation/
    │   ├── transformer_colab.md
    │   ├── synthetic_evaluation_methodology.md
    │   ├── synthetic_evaluation_results.md
    │   ├── final_test_evaluation.md
    │   ├── reproducibility_audit.md
    │   └── human_layer_research_report.md
    ├── models/
    ├── notebooks/
    ├── results/
    ├── src/
    └── requirements-transformer.txt
```

Large raw datasets and trained model weights are intentionally not committed directly to GitHub.

---

## Safety and Ethics

This project is defensive research.

The work uses public historical datasets and controlled synthetic materials. It does not involve targeting real people, collecting credentials, compromising accounts, delivering malware, or deploying phishing campaigns.

Synthetic messages were created for classifier evaluation only and were intentionally constrained to be non-operational.

The defensive demo is a research prototype and should not be treated as a production email-security system.

---

## Limitations

Important limitations include:

- the historical dataset combines several older email corpora and a broad phishing-class definition;
- corpus-specific language may influence model behavior;
- the controlled synthetic set is not representative of every form of real-world or AI-assisted phishing;
- only two generation sources and five synthetic communication categories were evaluated;
- the synthetic evaluation set contains only positive phishing-class examples;
- strong validation performance did not guarantee out-of-distribution robustness;
- model confidence was not reliably calibrated under synthetic distribution shift.

The frozen synthetic evaluation set should not be reused for tuning without creating a new independent holdout for final evaluation.

---

## Current Project Status

### Human Layer

- Historical dataset preparation: complete
- Exploratory analysis: complete
- Baseline model: complete
- DistilBERT model: complete
- Robustness analysis: complete
- Controlled synthetic evaluation: complete
- Defensive demo: complete
- Final held-out historical test evaluation: complete
- Final report and repository polish: in progress

### Network Layer

Planned next research component.

The Network Layer will investigate machine-learning detection of suspicious or automated network behavior using controlled or public network-security datasets.

---

## Core Finding

The Human Layer experiments show two things at the same time:

1. both classifiers generalized strongly from validation to a previously untouched historical test set;
2. that strong same-distribution performance still degraded sharply when the writing distribution changed.

Within the controlled synthetic evaluation, communication style appeared more informative than simple message length or generator source, and DistilBERT frequently made highly confident incorrect predictions.

The project therefore supports evaluating defensive AI systems not only by standard validation accuracy, but also by robustness under realistic distribution shifts.
