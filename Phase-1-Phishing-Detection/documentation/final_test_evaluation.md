# Final Held-Out Test Evaluation

## Purpose

This document records the one-shot final evaluation of the frozen Human Layer phishing-classification models on the previously isolated test split.

The test set was not used during model training, validation-based model selection, robustness analysis, synthetic evaluation, threshold selection, or error-driven model modification.

Once this evaluation was run, the test set was considered **consumed**. No subsequent model tuning, retraining, threshold adjustment, or model selection may be based on these test results.

## Frozen Test Artifact

The original test split created by `src/create_splits.py` was used without regeneration or modification.

- Rows: **31,225**
- Legitimate (label 0): **16,343**
- Phishing-class (label 1): **14,882**
- Missing bodies: **0**
- Missing labels: **0**
- Duplicate bodies within test split: **0**
- SHA-256: `751534eb068c2eabbd0a71d31fe219954df82c1b279ed44f41b8f166a592177a`

The SHA-256 fingerprint was recorded before either frozen model was evaluated on the test set.

## Frozen Model Artifacts

### TF-IDF + Logistic Regression

- Artifact size: **9,615,568 bytes**
- SHA-256: `706c674849627561fc28bc397afcadd2e8f75a972cd2aba8cd3da0ce07f2122d`

### DistilBERT

Frozen DistilBERT files were fingerprinted before test evaluation.

| File | Size | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` | 267,832,560 bytes | `5f1a7c01832ed2a2c93b5836815a853946f0d1b9fb8f6a9fc9d5b7725ded223f` |
| `config.json` | 664 bytes | `c4676b590fc6fd53fae40787cee9a49e6c04a8fb7d67b7312f50ffd61cd53350` |
| `tokenizer.json` | 711,494 bytes | `8b79639ec74b46604e730f505186eaafb1006d2fd00f2c4930d168bb7f894680` |
| `tokenizer_config.json` | 351 bytes | `e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd` |

Combined model-manifest SHA-256:

`32b9e1ee3e9ee7a5476c7acdd27f06d1ec97c297b13f6774643a0f36fe3bf480`

## Evaluation Protocol

Both models were evaluated once on the same 31,225-row test split.

The baseline used its frozen TF-IDF vectorizer and Logistic Regression classifier.

DistilBERT used:

- frozen fine-tuned model weights;
- frozen tokenizer;
- maximum input length: **256 tokens**;
- inference batch size: **32**;
- GPU inference;
- no threshold adjustment.

Classification used each model's existing decision rule. No test-set examples were used to modify preprocessing, features, weights, hyperparameters, thresholds, or model selection.

## Final Test Results

| Metric | TF-IDF + Logistic Regression | DistilBERT |
| --- | ---: | ---: |
| Accuracy | **98.2482%** | **99.2314%** |
| Precision | **97.8631%** | **99.2267%** |
| Recall | **98.4747%** | **99.1601%** |
| F1 | **98.1679%** | **99.1934%** |
| False-positive rate | **1.9580%** | **0.7037%** |
| False-negative rate | **1.5253%** | **0.8399%** |
| True negatives | 16,023 | 16,228 |
| False positives | 320 | 115 |
| False negatives | 227 | 125 |
| True positives | 14,655 | 14,757 |
| Total errors | **547** | **240** |

Observed runtime in the final environment:

- baseline: **48.78 seconds**
- DistilBERT: **285.11 seconds**

Runtime is environment-dependent and is not treated as a model-quality result.

## Validation-to-Test Comparison

| Metric | Baseline Validation | Baseline Test | DistilBERT Validation | DistilBERT Test |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 98.24% | 98.25% | 99.21% | 99.23% |
| Precision | 97.96% | 97.86% | 99.25% | 99.23% |
| Recall | 98.37% | 98.47% | 99.09% | 99.16% |
| F1 | 98.16% | 98.17% | 99.17% | 99.19% |
| False-positive rate | 1.87% | 1.96% | 0.68% | 0.70% |
| False-negative rate | 1.63% | 1.53% | 0.91% | 0.84% |
| Total errors | 548 | 547 | 246 | 240 |

The close validation-to-test agreement for both models provides evidence that their strong performance generalizes to unseen examples drawn from the same historical dataset distribution.

This result also sharpens the interpretation of the controlled synthetic experiment. The synthetic performance decline cannot be explained simply by the models failing on any unseen data: both models performed similarly on validation and the untouched historical test set, while performance dropped substantially on the controlled synthetic distribution.

This is consistent with a distribution-shift robustness problem rather than ordinary same-distribution generalization failure.

## scikit-learn Serialization Compatibility

The frozen baseline artifact was originally serialized under **scikit-learn 1.9.1** and was loaded for final evaluation in an environment using **scikit-learn 1.6.1**.

scikit-learn emitted `InconsistentVersionWarning` messages for the serialized:

- `TfidfTransformer`;
- `TfidfVectorizer`;
- `LogisticRegression`;
- `Pipeline`.

The serialized `LogisticRegression` artifact also lacked the `multi_class` attribute expected by the evaluation environment. The previously verified compatibility adjustment

```python
classifier.multi_class = "auto"
```

was applied only when the attribute was absent. This did not alter the learned coefficients or retrain the model.

Before final test evaluation, this compatibility-loaded artifact had already reproduced the original validation metrics exactly. The final test results should nevertheless be interpreted with the version mismatch documented as a reproducibility limitation.

For stronger future reproducibility, the exact training environment should be preserved or the model should be retrained and serialized under a pinned environment before a new independent evaluation is designed. The consumed test set must not be reused to validate such a retrained artifact.

## Saved Evaluation Artifacts

The final evaluation generated:

- `final_test_manifest.json`
- `final_test_metrics.csv`
- `final_test_predictions.csv`

The working persistent location used during evaluation was:

`/content/drive/MyDrive/Defensive-AI-Cybersecurity/final_test_evaluation/`

The manifest records the frozen artifact fingerprints, evaluation configuration, runtime, and the fact that the test set was consumed for final evaluation.

## Post-Test Rule

**The 31,225-row historical test split is now consumed.**

It must not be used for:

- hyperparameter tuning;
- retraining decisions;
- feature selection;
- threshold selection;
- calibration fitting;
- model selection;
- error-driven model modification.

Any future improved model should be developed without consulting this test set and should receive a newly designed independent final evaluation if an unbiased final performance estimate is required.
