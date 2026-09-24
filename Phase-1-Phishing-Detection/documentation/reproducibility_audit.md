# Reproducibility and Environment Audit

## Purpose

This audit records the software and artifact conditions that matter for reproducing the Human Layer experiments. It distinguishes exact observed versions from broad installation requirements and documents known limitations rather than implying bit-for-bit reproducibility where the original environment was not fully preserved.

## Reproducibility Status

| Component | Status | Notes |
| --- | --- | --- |
| Dataset preparation | Strong | Cleaning and split scripts are committed; random seed 42 is recorded. |
| Historical split integrity | Strong | Train/validation/test sizes are documented and duplicate bodies were checked across splits. |
| Baseline model configuration | Strong | TF-IDF and Logistic Regression hyperparameters are committed in `src/train_baseline.py`. |
| Baseline environment | Limited | Frozen artifact was serialized with scikit-learn 1.9.1, while final evaluation used scikit-learn 1.6.1. |
| Transformer configuration | Strong | Model name, max length, epochs, batch sizes, learning rate, weight decay, seed, and model-selection metric are committed. |
| Transformer environment | Moderate | Training used a Colab NVIDIA Tesla T4 environment, but a complete original package lockfile was not captured before training. |
| Frozen model identity | Strong | Final-test documentation records SHA-256 fingerprints for the baseline artifact and DistilBERT files. |
| Final historical evaluation | Strong | Test artifact was fingerprinted and consumed once after model development was frozen. |
| Synthetic evaluation | Strong | Evaluation code, methodology, aggregate results, and paired statistical comparison are committed. |
| Figure generation | Strong | Figures are regenerated from the committed aggregate results file by `src/generate_figures.py`. |

## Recorded Environments

### Baseline Development

The baseline was developed locally with Python 3.13.5. The frozen serialized model records a scikit-learn 1.9.1 origin through the later compatibility warning observed during evaluation.

The baseline training script also depends on:

- pandas;
- joblib;
- matplotlib;
- scikit-learn.

The repository did not preserve an exact package lockfile from the original baseline-training environment. Therefore, the committed source code and frozen model fingerprint are stronger provenance records than the current broad dependency specifications.

### Transformer Training

DistilBERT training was performed in Google Colab on an NVIDIA Tesla T4 GPU.

The committed training configuration is:

- base model: `distilbert-base-uncased`;
- maximum sequence length: 256;
- epochs: 2;
- train batch size: 16;
- evaluation batch size: 32;
- learning rate: 2e-5;
- weight decay: 0.01;
- validation model-selection metric: F1;
- FP16: enabled;
- random seed: 42.

The repository's `requirements-transformer.txt` uses minimum-version constraints rather than an exact historical lockfile. Those requirements are suitable for reconstructing a compatible environment, but they should not be interpreted as the exact package versions used in the completed experiment.

### Final Historical Test Evaluation

The final baseline evaluation environment used scikit-learn 1.6.1 to load an artifact serialized with scikit-learn 1.9.1. The resulting compatibility warnings and the narrowly scoped `multi_class` compatibility adjustment are documented in `documentation/final_test_evaluation.md`.

The compatibility-loaded baseline reproduced the original validation metrics exactly before the test set was consumed. No learned coefficients were changed and no retraining occurred.

The final DistilBERT evaluation used the frozen model and tokenizer, maximum input length 256, inference batch size 32, and GPU inference.

## Frozen Artifact Fingerprints

### Historical test split

SHA-256:

`751534eb068c2eabbd0a71d31fe219954df82c1b279ed44f41b8f166a592177a`

### Baseline model

SHA-256:

`706c674849627561fc28bc397afcadd2e8f75a972cd2aba8cd3da0ce07f2122d`

### DistilBERT

| Artifact | SHA-256 |
| --- | --- |
| `model.safetensors` | `5f1a7c01832ed2a2c93b5836815a853946f0d1b9fb8f6a9fc9d5b7725ded223f` |
| `config.json` | `c4676b590fc6fd53fae40787cee9a49e6c04a8fb7d67b7312f50ffd61cd53350` |
| `tokenizer.json` | `8b79639ec74b46604e730f505186eaafb1006d2fd00f2c4930d168bb7f894680` |
| `tokenizer_config.json` | `e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd` |

Combined DistilBERT manifest SHA-256:

`32b9e1ee3e9ee7a5476c7acdd27f06d1ec97c297b13f6774643a0f36fe3bf480`

## Determinism and Remaining Sources of Variation

Random seed 42 is used for the historical split, baseline classifier, and transformer training configuration. This improves repeatability but does not guarantee bit-identical GPU training. CUDA kernels, library versions, hardware, and upstream implementation changes can introduce numerical variation.

Downloading `distilbert-base-uncased` by model name also depends on the upstream model repository unless a specific revision is preserved. The completed experiment is therefore best reproduced from the frozen fine-tuned artifacts and recorded hashes when exact model identity matters.

## Dependency Policy Going Forward

The existing dependency files are intentionally left as compatibility-oriented installation requirements rather than being rewritten to claim exact historical versions that were not recorded at training time.

For future experiments:

1. capture `python --version` and `pip freeze` before training;
2. record GPU and CUDA information;
3. pin the upstream pretrained-model revision;
4. save the exact environment alongside model artifacts;
5. hash final datasets and models before evaluation;
6. keep development, validation, and final holdout decisions separated.

A future retraining performed under a newly pinned environment would constitute a new model artifact. It must not be tuned or selected using the already consumed historical test set or frozen synthetic evaluation set.

## Reproduction Entry Points

Key scripts are:

- `src/prepare_dataset.py`
- `src/create_splits.py`
- `src/train_baseline.py`
- `src/run_stronger_robustness_checks.py`
- `src/train_transformer.py`
- `src/analyze_transformer_errors.py`
- `src/evaluate_synthetic.py`
- `src/generate_figures.py`

Supporting records are:

- `documentation/transformer_colab.md`
- `documentation/synthetic_evaluation_methodology.md`
- `documentation/synthetic_evaluation_results.md`
- `documentation/final_test_evaluation.md`
- `results/results_summary.csv`

## Audit Conclusion

The project preserves the experiment design, frozen model identities, final test identity, major hyperparameters, aggregate results, and evaluation protocol well enough for independent inspection and substantial reproduction. The main reproducibility limitation is that exact package lockfiles were not captured before the original baseline and transformer training runs. That limitation is documented explicitly rather than retroactively assigning package versions that cannot be verified.
