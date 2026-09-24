# Human Layer Dataset Documentation

This folder documents the datasets used for the Human Layer phishing-detection research project.

## Dataset Requirements

The initial dataset should include:

- Legitimate emails
- Traditional phishing emails
- Email text content, not only URLs
- Reliable labels identifying legitimate and phishing messages
- A public source suitable for research use

A separate controlled synthetic phishing-class evaluation set was later created for robustness testing after the historical baseline dataset and models were established.

## Dataset Sources

### Dataset 1: Phishing-Email-Detection-Dataset

- **Name:** Phishing-Email-Detection-Dataset
- **Primary source:** Zenodo record 17314806
- **Related repository:** Manar-ibr/Phishing-Email-Detection-Dataset
- **Date accessed:** September 23, 2026
- **Content:** Labeled email text for safe/legitimate and phishing-class messages
- **Labels:** 0 = safe/legitimate, 1 = phishing-class
- **Underlying corpora:** The dataset aggregates multiple established public email corpora, including Enron, SpamAssassin, TREC, CEAS, Nazario, Nigerian Fraud, and Ling.
- **How it was used:** Training, validation, robustness analysis, and the one-shot final held-out historical evaluation for legitimate vs. phishing-class email detection. The separate controlled synthetic set was not used for training.
- **Important limitation:** The positive class groups phishing, spam, and scam-style messages together. Because of this, results from this dataset should be described as detection of a broader phishing-class category rather than strictly credential-phishing only.
- **License or usage restrictions:** Verify and retain the license/usage terms from the dataset record before publishing redistributed raw data.

## Folder Structure

- `raw/` stores original downloaded data
- `processed/` stores cleaned and transformed datasets used for modeling

## Research Notes

The final test set remained isolated from training, validation-based model selection, robustness analysis, and synthetic evaluation until model development was frozen. It was then consumed exactly once for the final historical evaluation and must not be reused for tuning. Exact duplicate bodies were checked across the train, validation, and test partitions to reduce leakage.

Source-specific formatting should also be checked so the model does not learn dataset origin instead of phishing-related characteristics.

## Raw Dataset Acquisition

The selected dataset files are **not stored in the source GitHub repository**. The repository contains documentation only; the actual dataset is distributed through Zenodo under DOI `10.5281/zenodo.17314806`.

For reproducibility, the original downloaded dataset should be placed unchanged in:

`Phase-1-Phishing-Detection/data/raw/`

Do not edit the raw file in place. Cleaning, deduplication, normalization, and train/validation/test preparation will happen later in `data/processed/`.

### Source verification

The source repository confirms two available dataset variants:

- **Merged Dataset:** combines nine public corpora (Ling, Enron, SpamAssassin, TREC 2005/2006/2007, CEAS, Nazario_5, Nigerian_5)
- **Balanced Dataset:** a down-sampled version with legitimate and phishing-class samples balanced for ML experiments

Both use columns `body` and `label`, with `0 = Safe Email` and `1 = Phishing Email`.

The source repository also states that spam, scam, and phishing emails are grouped together under the phishing label.
