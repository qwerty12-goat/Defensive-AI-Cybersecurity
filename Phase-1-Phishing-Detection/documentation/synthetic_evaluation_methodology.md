# Synthetic AI Phishing-Class Evaluation Methodology

## Purpose

This evaluation tests whether phishing-detection models trained on a historical real-world email corpus remain effective on a controlled set of AI-generated phishing-class messages.

The synthetic set is an out-of-distribution robustness evaluation. It is not training data, and its results are not used to retrain or tune the evaluated models.

## Research Context

The Human Layer compares two classifiers:

- TF-IDF + Logistic Regression baseline
- DistilBERT advanced NLP model

Both models were trained before the synthetic evaluation. The traditional validation split remained separate from the synthetic dataset.

The synthetic evaluation addresses the secondary research question:

> Are AI-generated phishing emails more difficult for machine-learning models to detect than traditional phishing emails?

Results should be interpreted only within the scope of this controlled experiment. The synthetic messages are not assumed to represent every real-world AI-assisted phishing attack.

## Dataset Design

The frozen synthetic evaluation dataset contains **500 phishing-class messages**.

Five communication categories were used, with 100 messages per category:

| Category | Samples |
| --- | ---: |
| Account/security notification style | 100 |
| Workplace/business communication style | 100 |
| Delivery/service notification style | 100 |
| Promotional/offer style | 100 |
| General social-engineering style | 100 |
| **Total** | **500** |

Two AI generation systems contributed equally:

| Generation source | Samples |
| --- | ---: |
| Copilot, model displayed in the generation interface as GPT-5.6 Quick Response | 250 |
| Gemini, model displayed in the generation interface as Gemini Flash 3.6 | 250 |

Each generator contributed 50 messages to each of the five categories.

The dataset records:

- `sample_id`
- `category`
- `text`
- `label`
- `generation_source`
- `generation_date`

The assembled dataset uses generation date `2026-09-23`.

## Safety Controls

The messages were generated only for defensive cybersecurity research.

Generation instructions required the samples to remain controlled and non-operational. They did not use real people, real companies, or real domains as targets. The dataset was not designed for sending messages to real recipients.

Potentially actionable elements were replaced with inert placeholders:

- `[LINK_REMOVED]`
- `[ATTACHMENT_REMOVED]`

The generation process did not request credential collection, payments, malware, account compromise, real targeting, or optimization for bypassing security detectors.

## Generation Procedure

The same overall experimental structure was used for both generation sources.

For each source:

1. Generate 50 controlled samples for each category.
2. Preserve category and generation-source metadata.
3. Review the batch for formatting and safety compliance.
4. Reject or correct generation artifacts before model evaluation.
5. Assemble all accepted batches into one dataset.
6. Validate the complete dataset before exposing it to either classifier.

This produced 250 accepted Copilot samples and 250 accepted Gemini samples.

## Quality Assurance and Corrections

Quality assurance occurred before classifier evaluation.

One Gemini delivery/service batch was rejected because it contained a malformed row and material that did not satisfy the experiment's safety constraints. The entire affected batch was regenerated before either classifier evaluated the synthetic dataset.

One Copilot promotional/offer output ended prematurely at sample 046. Samples 046 through 050 were completed before assembly and before model evaluation.

During final assembly, one metadata-only identifier typo, `GEM_SERVICE_026`, was deterministically corrected to `GEM_PROMO_026`. The message text was not changed.

No message was rewritten based on how either classifier predicted it.

## Dataset Freeze

The complete 500-message dataset was assembled and validated before either trained classifier was run against it.

Final validation confirmed:

- 500 rows
- 500 unique sample IDs
- 100 samples in each category
- 250 Copilot samples
- 250 Gemini samples
- one documented metadata correction

After this validation, the dataset was treated as frozen evaluation data.

This ordering is important because modifying messages after observing classifier predictions could introduce post-hoc bias into the evaluation.

## Evaluation Procedure

The TF-IDF + Logistic Regression baseline and the saved DistilBERT model were evaluated independently on the same frozen 500 messages.

The primary synthetic-set metric is **detection rate/recall**, because every message in the synthetic dataset belongs to the positive phishing class.

Precision and false-positive rate cannot be meaningfully estimated from this positive-only synthetic set. False-positive behavior is therefore reported separately using legitimate messages from the traditional held-out validation data.

For DistilBERT, inference used the same 256-token maximum sequence length used in the primary trained model configuration.

## Robustness Analyses

The synthetic evaluation includes:

- overall detection rate
- detection rate by communication category
- detection rate by generation source
- 95% Wilson confidence intervals
- direct prediction overlap between the two models
- message-length analysis
- word/document-frequency analysis for detected and missed messages
- prediction-confidence analysis
- high-confidence miss analysis by category and generator
- exact paired McNemar comparison

These analyses are interpretive. The frozen synthetic dataset is not used to tune either model after the results are observed.

## Statistical Comparison

Because both classifiers evaluated the exact same 500 messages, their paired outcomes are compared with McNemar's test rather than treating the two sets of predictions as independent samples.

The primary paired comparison focuses on messages detected by only one of the two models.

## Interpretation Limits

This experiment measures robustness to one deliberately constructed synthetic distribution. Several limitations should remain explicit:

- The synthetic set contains only phishing-class messages.
- The messages were generated under safety constraints and are intentionally non-operational.
- The five categories are experimental groupings, not a complete taxonomy of phishing.
- Results from these two generation systems should not be generalized to every AI model.
- Differences associated with individual words are correlations within this dataset and do not establish that those words caused model predictions.
- Strong performance on the historical validation distribution does not guarantee robustness to a different synthetic distribution.
- Synthetic-set results should not be used to tune the evaluated models unless a new independent held-out synthetic evaluation set is created.

## Reproducibility

The repository includes `src/evaluate_synthetic.py` for reproducing the core evaluation from:

1. the frozen synthetic dataset,
2. the saved TF-IDF + Logistic Regression model, and
3. the saved DistilBERT model.

The script validates the dataset, runs both classifiers, calculates grouped detection results and Wilson intervals, compares model overlap, performs the paired McNemar test, and saves reproducible output files.

## Provenance Note

The synthetic messages were produced with AI assistance as part of a controlled defensive cybersecurity experiment. AI assistance was also used during development of portions of the research workflow and code. Experimental decisions, dataset freezing, model evaluation, result verification, and interpretation are documented so the work can be reviewed transparently.
